"""The live path end to end: what reaches the page, and what a failure costs."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import asr, config, main, store as store_mod, translate
from .e2e_support import FixedTranscriber, OneSpeaker, StubTranslator, headless_pipeline
from .pipeline import Pipeline


def test_pipeline_emits_line_then_update(tmp: Path) -> None:
    """The core subtitle contract, exercised on a real wav through the real VAD and ASR."""
    wav = config.MODELS_DIR / "sherpa-onnx-whisper-tiny" / "test_wavs" / "1.wav"
    if not wav.is_file():
        print("  (skipped: whisper test wav not present)")
        return

    import soundfile as sf

    audio, sr = sf.read(str(wav), dtype="float32")
    assert sr == config.SAMPLE_RATE

    # This test is about the sherpa-onnx wiring and a known-language wav; the GPU model would
    # substitute a different recogniser and decode this English clip as Mandarin.
    os.environ["POLYMINUTES_NO_GPU"] = "1"

    st = store_mod.Store(tmp / "pipeline.db")
    session = st.start_session("now", str(wav))
    events: list[dict] = []
    cfg = config.Config(languages=["en", "zh"], whisper_model="tiny")

    # A second of silence between repeats, or VAD sees one unbroken 50 s utterance rather than
    # three — which is also how a real meeting separates turns.
    gap = np.zeros(config.SAMPLE_RATE, dtype="float32")
    feed = np.concatenate([audio, gap, audio, gap, audio, gap])

    pipe = Pipeline(cfg, st, session, StubTranslator(), events.append)
    pipe.start()
    try:
        for i in range(0, len(feed), config.BLOCK_SIZE):
            pipe.tap.put(feed[i : i + config.BLOCK_SIZE])
        pipe.tap.put(None)
        pipe.join()

        kinds = [e["type"] for e in events]
        assert kinds.count("line") >= 3, kinds
        assert "update" in kinds, f"no refinement emitted: {kinds}"
        assert pipe.errors == 0, f"{pipe.errors} segment errors"

        lines = {e["line"]["id"] for e in events if e["type"] == "line"}
        updates = {e["line"]["id"] for e in events if e["type"] == "update"}
        # An update must target a line already sent, or the page would have nothing to rewrite.
        assert updates <= lines, (lines, updates)

        stored = st.lines(session)
        assert len(stored) == len(lines), "every emitted line must be persisted"
        refined = [r for r in stored if r["refined"]]
        assert refined and refined[0]["source"] == "corrected source"
        assert refined[0]["translations"]["zh"] == "[zh] corrected"
    finally:
        st.close()


def test_a_rerun_always_answers_in_the_same_shape(tmp: Path) -> None:
    """The page reads the reply straight into state, so both outcomes must carry the same keys.

    They drifted apart once — one exit returned `line`, the other `lines` — which would have
    blanked the transcript the rerun was supposed to be repairing.
    """
    st = store_mod.Store(tmp / "shape.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "s.wav"))
        st.add_line(session_id, 0.0, "S1", "zh", "一行", {"en": "a line"})
        st.set_speaker_name(session_id, "S1", "陳經理")

        original, main.store = main.store, st
        try:
            for status in ("ok", "asr_failed", "translate_failed"):
                body = main._transcript(session_id, status)
                assert set(body) == {"lines", "speakers", "status"}, body
                assert body["status"] == status
                assert body["lines"] and body["speakers"] == {"S1": "陳經理"}, body
        finally:
            main.store = original
    finally:
        st.close()


def test_a_failed_translation_costs_the_translation_not_the_line(tmp: Path) -> None:
    """It used to raise into the handler's catch-all and drop the whole utterance.

    The room would then see nothing where it should have seen the original text untranslated —
    a translation outage reading as a speaker who never spoke.
    """
    class Exploding:
        def translate(self, *a, **k):
            raise RuntimeError("no key")

    st = store_mod.Store(tmp / "translate-fail.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "t.wav"))
        emitted: list[dict] = []
        pipe = headless_pipeline(config.Config(), st, session_id, Exploding(), emitted.append)
        pipe._transcriber = FixedTranscriber("這句話有說出來")
        pipe._diarizer = OneSpeaker()

        pipe._handle(asr.Segment(np.zeros(config.SAMPLE_RATE, dtype="float32"), 0.0))

        rows = st.lines(session_id)
        assert len(rows) == 1, rows
        assert rows[0]["source"] == "這句話有說出來", rows
        assert rows[0]["status"] == "translate_failed", rows
        assert rows[0]["translations"] == {}, rows
        assert pipe.errors == 0, "a translation outage is not a pipeline error"
        assert emitted and emitted[0]["line"]["status"] == "translate_failed", emitted
    finally:
        st.close()


def test_a_source_only_refinement_keeps_the_translations_on_screen(tmp: Path) -> None:
    """A revision that touches only the source must not blank the subtitle under it.

    The store merges (update_line upserts), so the database kept the old translations — but the
    update event carried `previous_translations` verbatim. The subtitle page replaces the line
    with whatever the update holds, so a source-only revision erased the translated text the room
    was reading, and a partial one (en revised, vi not) erased the other language.
    """
    from .e2e_support import headless_pipeline

    class ReviseSourceOnly:
        def __init__(self):
            self.calls = 0

        def translate(self, line, targets, context=None, previous=None, terms=None,
                      prev_targets=None):
            self.calls += 1
            out = {t: f"[{t}] {line.text}" for t in targets}
            if self.calls == 2:   # judge the first line wrong, but revise only its source text
                return translate.Result(out, "修過的第一句", {})
            if self.calls == 3:   # revise the second line's en only; vi was fine
                return translate.Result(out, "修過的第二句", {"en": "revised en"})
            return translate.Result(out)

    st = store_mod.Store(tmp / "refine-emit.db")
    session = st.start_session("now", "x.wav")
    events: list[dict] = []
    cfg = config.Config(languages=["zh", "vi", "en"])
    pipe = headless_pipeline(cfg, st, session, ReviseSourceOnly(), events.append)

    for k, text in enumerate(["第一句", "第二句", "第三句"]):
        pipe._transcriber = type("T", (), {"transcribe": staticmethod(lambda s, l, _t=text: (_t, "zh")),
                                           "set_hotwords": staticmethod(lambda h: None)})()
        pipe._handle(asr.Segment(np.zeros(1600, dtype="float32"), start=float(k)))
    assert pipe.errors == 0

    updates = [e["line"] for e in events if e["type"] == "update"]
    assert len(updates) == 2, [e["type"] for e in events]

    # Source-only revision: the event must still carry the translations the line already had.
    first = updates[0]
    assert first["source"] == "修過的第一句"
    assert first["translations"].get("en") == "[en] 第一句", first["translations"]
    assert first["translations"].get("vi") == "[vi] 第一句", first["translations"]

    # Partial revision: en replaced, vi preserved.
    second = updates[1]
    assert second["translations"].get("en") == "revised en", second["translations"]
    assert second["translations"].get("vi") == "[vi] 第二句", second["translations"]

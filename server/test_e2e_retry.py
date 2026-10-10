"""Utterances that decoded to nothing: held, retried once, and never silently lost."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import asr, config, retry as retry_mod, store as store_mod
from .e2e_support import ByLanguage, OneSpeaker, headless_pipeline


def test_a_failed_decode_is_retried_once_the_speaker_language_is_known(tmp: Path) -> None:
    """Live used to bin these in silence. The post-meeting pass recovers 992 such lines.

    The first utterance decodes to nothing under auto-detect. The second establishes that this
    speaker is speaking Chinese. The first must then be re-decoded under Chinese and appear.
    """
    st = store_mod.Store(tmp / "retry.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "r.wav"))
        emitted: list[dict] = []
        pipe = headless_pipeline(config.Config(languages=["zh", "en"]), st, session_id, None, emitted.append)
        pipe._transcriber = ByLanguage({"": ("", ""), "zh": ("補回來的那一句", "zh")})
        pipe._diarizer = OneSpeaker()

        # Fails under auto-detect and is held, not dropped.
        pipe._handle(asr.Segment(np.zeros(config.SAMPLE_RATE, dtype="float32"), 10.0))
        assert st.lines(session_id) == [], "a held utterance must not be stored yet"
        assert len(pipe._retries.held) == 1, pipe._retries.held

        # A later utterance settles the speaker's language, which triggers the retry.
        pipe._transcriber.table[""] = ("這句話正常", "zh")
        pipe._handle(asr.Segment(np.zeros(config.SAMPLE_RATE, dtype="float32"), 20.0))

        rows = st.lines(session_id)
        assert [r["source"] for r in rows] == ["補回來的那一句", "這句話正常"], rows
        assert pipe._retries.recovered == 1 and pipe._retries.dropped == 0
        assert pipe._retries.held == []

        # It voted once, not twice: the retry must not count the same audio toward the speaker.
        assert pipe._diarizer.votes == ["zh"], pipe._diarizer.votes

        # And it is not offered to the next line as context, nor as the line to refine.
        assert pipe._previous[2].text == "這句話正常"
        assert [l.text for l in pipe._context] == ["這句話正常"]

        # The recovered line is emitted with its own start, so the page can place it correctly.
        late = [e for e in emitted if e["line"]["source"] == "補回來的那一句"]
        assert late and late[0]["line"]["start"] == 10.0, emitted
    finally:
        st.close()


def test_live_drops_the_glossary_prompt_before_holding_a_clip(tmp: Path) -> None:
    """A clip the prompt silenced is decoded again without it, live, not held and lost."""
    class PromptLocked(ByLanguage):
        _hotwords = "工序"

        def transcribe_unbiased(self, samples, language):
            return "因為這些零件的外觀標準要求特別高", "zh"

    st = store_mod.Store(tmp / "live-unbiased.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "u.wav"))
        pipe = headless_pipeline(config.Config(languages=["zh"]), st, session_id, None, lambda e: None)
        pipe._transcriber = PromptLocked({"": ("", ""), "zh": ("", "")})
        pipe._diarizer = OneSpeaker()

        pipe._handle(asr.Segment(np.zeros(16000, dtype="float32"), 1.0))

        assert pipe._retries.held == [], "a clip the unbiased decode recovered was held"
        assert [l["source"] for l in st.lines(session_id)] == ["因為這些零件的外觀標準要求特別高"]
    finally:
        st.close()


def test_a_retry_behind_another_speakers_held_clip_does_not_raise(tmp: Path) -> None:
    """list.remove() matches by identity first, so this only bites when someone else's clip is held
    ahead — then it compared the two Segments, i.e. their sample arrays of different lengths, and
    raised. Found by a real-time replay of the 2026-10-05 meeting, where it cost a live line."""
    from types import SimpleNamespace

    other, mine = SimpleNamespace(code="S2"), SimpleNamespace(code="S1")
    retries = retry_mod.Retries()
    retries.hold(asr.Segment(np.zeros(24000, dtype="float32"), 1.0), other, "")
    retries.hold(asr.Segment(np.zeros(16000, dtype="float32"), 2.0), mine, "")

    tried = []
    retries.retry(mine, "zh", lambda seg, spk, lang: tried.append(seg.start) or True)

    assert tried == [2.0] and retries.recovered == 1
    assert [h[1].code for h in retries.held] == ["S2"], "the other speaker's clip must stay held"


def test_the_retry_buffer_cannot_grow_without_bound(tmp: Path) -> None:
    """Every held utterance keeps its raw audio, so a room that decodes nothing must not fill RAM."""
    st = store_mod.Store(tmp / "retry-cap.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "c.wav"))
        pipe = headless_pipeline(config.Config(languages=["zh"]), st, session_id, None, lambda e: None)
        pipe._transcriber = ByLanguage({"": ("", ""), "zh": ("", "")})
        pipe._diarizer = OneSpeaker()

        for i in range(retry_mod.RETRY_BUFFER + 8):
            pipe._handle(asr.Segment(np.zeros(1600, dtype="float32"), float(i)))

        assert len(pipe._retries.held) == retry_mod.RETRY_BUFFER, len(pipe._retries.held)
        assert pipe._retries.dropped == 8, pipe._retries.dropped
        # The oldest went first, so what is still held is the most recent audio.
        assert pipe._retries.held[0][0].start == 8.0, pipe._retries.held[0][0].start
    finally:
        st.close()


def test_a_retry_that_explodes_is_counted_not_lost(tmp: Path) -> None:
    """The held entry is already off the list by then, so an escape would lose it silently.

    It must also not fail the live utterance that triggered the retry: recovering something older
    is a bonus on top of that line, never a risk to it.
    """
    class Exploding(ByLanguage):
        def transcribe(self, samples, language):
            if language == "zh" and len(samples) == 4242:
                raise RuntimeError("decoder blew up")
            return super().transcribe(samples, language)

    st = store_mod.Store(tmp / "retry-boom.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "b.wav"))
        pipe = headless_pipeline(config.Config(languages=["zh"]), st, session_id, None, lambda e: None)
        pipe._transcriber = Exploding({"": ("", ""), "zh": ("正常的一句", "zh")})
        pipe._diarizer = OneSpeaker()

        pipe._handle(asr.Segment(np.zeros(4242, dtype="float32"), 5.0))  # decodes to nothing, held
        assert len(pipe._retries.held) == 1, pipe._retries.held

        # The next utterance decodes, which settles the speaker's language and fires the retry.
        pipe._transcriber.table[""] = ("正常的一句", "zh")
        pipe._handle(asr.Segment(np.zeros(1600, dtype="float32"), 12.0))

        assert pipe._retries.dropped == 1, pipe._retries.dropped
        assert pipe._retries.recovered == 0
        assert pipe._retries.held == []
        # The live line survived the failed retry.
        assert [r["source"] for r in st.lines(session_id)] == ["正常的一句"]
        assert pipe.errors == 0, "a failed retry is not a failure of the live segment"
    finally:
        st.close()

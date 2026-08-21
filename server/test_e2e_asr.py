"""Which weights get loaded, and what the recogniser does with a language it was not given."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import asr, asr_gpu, config


def test_weight_selection_prefers_quantized_for_live(tmp: Path) -> None:
    """Live capture wants int8; postprocess wants float32 and falls back if it is absent."""
    d = config.MODELS_DIR / "sherpa-onnx-whisper-tiny"
    if not d.is_dir():
        print("  (skipped: whisper model not present)")
        return

    live_enc, _, _ = asr.Transcriber(model_dir=d)._paths()
    assert live_enc.endswith(".int8.onnx"), live_enc

    slow_enc, _, _ = asr.Transcriber(model_dir=d, quantized=False)._paths()
    assert slow_enc.endswith(".onnx") and not slow_enc.endswith(".int8.onnx"), slow_enc

    assert 2 <= asr.default_threads() <= 4


def test_gpu_backend_declines_cleanly_when_disabled(tmp: Path) -> None:
    """The GPU path must be optional: every caller falls back to sherpa-onnx when it says no."""
    original = os.environ.get("POLYMINUTES_NO_GPU")
    try:
        os.environ["POLYMINUTES_NO_GPU"] = "1"
        assert asr_gpu.maybe(["zh", "en"]) is None
    finally:
        os.environ.pop("POLYMINUTES_NO_GPU", None)
        if original is not None:
            os.environ["POLYMINUTES_NO_GPU"] = original


def test_autodetect_reports_the_language(tmp: Path) -> None:
    """Auto-detect must return which language it decoded in.

    Without this a speaker's language can never be established: every utterance would report ''
    and the pipeline would stay on auto-detect for the whole meeting, which is the exact fragility
    the per-speaker language design exists to avoid.
    """
    wav = config.MODELS_DIR / "sherpa-onnx-whisper-tiny" / "test_wavs" / "1.wav"
    if not wav.is_file():
        print("  (skipped: whisper test wav not present)")
        return

    import soundfile as sf

    audio, _ = sf.read(str(wav), dtype="float32")
    tr = asr.Transcriber(model_dir=config.MODELS_DIR / "sherpa-onnx-whisper-tiny")

    text, detected = tr.transcribe(audio, "")
    assert text
    assert detected == "en", f"auto-detect reported {detected!r}"


def test_long_utterance_is_not_truncated(tmp: Path) -> None:
    """Whisper drops everything past 30 s; the decoder must chunk rather than lose speech.

    Exercises `_decode` rather than `transcribe`, because the only test audio available is two
    short clips and anything long enough to need chunking has to repeat them — which is genuinely
    degenerate, and `transcribe` now refuses degenerate output. That refusal is right for a
    transcript and makes the fixture useless for measuring length, so the two are tested apart.
    """
    clips = [config.MODELS_DIR / "sherpa-onnx-whisper-tiny" / "test_wavs" / f"{n}.wav"
             for n in (0, 1)]
    if not all(c.is_file() for c in clips):
        print("  (skipped: whisper test wavs not present)")
        return

    import soundfile as sf

    audio = [sf.read(str(c), dtype="float32")[0] for c in clips]
    tr = asr.Transcriber(model_dir=config.MODELS_DIR / "sherpa-onnx-whisper-tiny")

    short, _ = tr._decode(audio[1], "en")
    # Past the 25 s the decoder allows per pass, so it must split and every part must contribute.
    long_text, _ = tr._decode(np.concatenate([audio[1], audio[0], audio[1]]), "en")

    assert short, "baseline transcription is empty"
    assert len(long_text) > len(short) * 1.5, (len(short), len(long_text))


class _FakeBatched:
    """A batched pipeline that runs out of memory until the batch is small enough.

    Stands in for faster_whisper's BatchedInferencePipeline so the shrink-on-OOM retry can be
    tested without a GPU or a model: real OOM cannot be provoked on the build machine, and the
    logic — wait once, then halve — is what has to be right.
    """

    def __init__(self, ok_at: int) -> None:
        self._ok_at = ok_at  # succeeds once batch_size <= this
        self.batches: list[int] = []

    def transcribe(self, audio, batch_size, **kw):
        """Lazy, like the real one. faster-whisper decodes on iteration, not on the call, so a
        fake that raises eagerly cannot see whether the retry actually wraps the decode — and for
        a while it did not."""
        self.batches.append(batch_size)
        too_big = batch_size > self._ok_at

        def segments():
            if too_big:
                raise RuntimeError("CUDA failed with error out of memory")
            return iter(())

        class _Info:
            language = "zh"

        return _LazySegments(segments), _Info()


class _LazySegments:
    """Iterable that does its work — and raises — only when something iterates it."""

    def __init__(self, produce) -> None:
        self._produce = produce

    def __iter__(self):
        return self._produce()


def _fake_transcriber(fake: _FakeBatched) -> asr_gpu.Transcriber:
    tr = asr_gpu.Transcriber.__new__(asr_gpu.Transcriber)
    tr._languages, tr._hotwords, tr._batched = ["zh"], "", fake
    tr._batch = asr_gpu.BATCH_SIZE
    return tr


def test_auto_detect_retry_drops_a_language_the_room_never_configured(tmp: Path) -> None:
    """A forced decode that collapses is retried with auto-detect — but that retry must obey the
    same language allow-list the forced pass does. Otherwise Whisper guessing Portuguese on noise
    reaches the subtitles as valid text and counts toward what the speaker is taken to speak."""
    tr = asr.Transcriber.__new__(asr.Transcriber)
    tr._languages = ["zh", "en"]

    forced = "產品 " * 8  # eight identical tokens: a collapse, so the retry fires
    calls: list[str] = []

    def fake_decode(samples, language):
        calls.append(language)
        if language:
            return forced.strip(), "zh"          # forced pass collapsed
        return "obrigado pela reunião de hoje", "pt"  # auto-detect: clean, but off-list

    tr._decode = fake_decode
    assert asr.is_degenerate(forced), "fixture is not degenerate; the retry would not fire"

    text, used = tr.transcribe(np.zeros(16000, dtype=np.float32), "zh")
    assert calls == ["zh", ""], calls  # forced, then auto-detect
    assert text == "", f"off-list retry text leaked: {text!r}"
    assert used == "pt"


def test_is_oom_recognises_the_failure_but_not_ordinary_errors(tmp: Path) -> None:
    assert asr_gpu._is_oom(RuntimeError("CUDA failed with error out of memory"))
    assert asr_gpu._is_oom(RuntimeError("CUBLAS_STATUS_ALLOC_FAILED"))
    assert not asr_gpu._is_oom(RuntimeError("clip_timestamps out of range"))
    assert not asr_gpu._is_oom(ValueError("out of memory"))  # wrong type, not a decode OOM


def test_a_contended_card_waits_then_shrinks_the_batch(tmp: Path) -> None:
    """OOM at full batch: wait once at 32, then halve until it fits."""
    fake = _FakeBatched(ok_at=8)
    tr = _fake_transcriber(fake)
    import server.asr_gpu as m
    original = m.OOM_WAIT_SECONDS
    m.OOM_WAIT_SECONDS = 0.0  # no real sleep in the test
    try:
        out = tr.transcribe_many([np.zeros(16000, dtype=np.float32)], "zh")
    finally:
        m.OOM_WAIT_SECONDS = original

    # 32 twice (the initial try and the post-wait retry), then 16, then 8 which succeeds.
    assert fake.batches == [32, 32, 16, 8], fake.batches
    assert out == [("", "zh")]  # empty fake result, judged and returned

    # The card's capacity does not change between batches, so the next decode starts at 8 rather
    # than paying the discovery again. A 55-minute reprocess ran 88s against over three minutes.
    fake.batches.clear()
    tr.transcribe_many([np.zeros(16000, dtype=np.float32)], "zh")
    assert fake.batches == [8], fake.batches


def test_a_card_too_full_for_one_utterance_gives_up(tmp: Path) -> None:
    """Batch one still OOMing is not batch greed — the weights do not fit. Re-raise."""
    fake = _FakeBatched(ok_at=0)  # never succeeds
    tr = _fake_transcriber(fake)
    import server.asr_gpu as m
    original = m.OOM_WAIT_SECONDS
    m.OOM_WAIT_SECONDS = 0.0
    try:
        raised = False
        try:
            tr.transcribe_many([np.zeros(16000, dtype=np.float32)], "zh")
        except RuntimeError as exc:
            raised = True
            assert "out of memory" in str(exc).lower()
    finally:
        m.OOM_WAIT_SECONDS = original

    assert raised, "a card that cannot decode one utterance must fail, not loop"
    # It went all the way down to the floor before giving up.
    assert fake.batches[-1] == asr_gpu.MIN_BATCH, fake.batches


def test_a_non_cuda_error_is_not_retried(tmp: Path) -> None:
    """A bug in our own batching must surface, not be mistaken for memory pressure."""
    class _Broken:
        def __init__(self): self.calls = 0
        def transcribe(self, *a, **k):
            self.calls += 1
            raise RuntimeError("clip_timestamps out of range")

    broken = _Broken()
    tr = _fake_transcriber(broken)
    raised = False
    try:
        tr.transcribe_many([np.zeros(16000, dtype=np.float32)], "zh")
    except RuntimeError:
        raised = True
    assert raised and broken.calls == 1, "a non-CUDA error must not be retried"

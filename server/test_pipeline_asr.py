"""The recogniser's two inputs from outside: the language whitelist and the glossary bias."""

from __future__ import annotations

from . import asr, asr_gpu, config, pipeline, store


def test_language_whitelist_rejects_what_was_never_configured() -> None:
    """A zh/vi/en meeting produced pt, bo, ja, ko and it on a real recording — all from noise."""
    tr = asr.Transcriber(languages=["zh", "vi", "en"])
    assert tr._allowed("zh") and tr._allowed("vi") and tr._allowed("en")
    assert not tr._allowed("pt") and not tr._allowed("bo") and not tr._allowed("ja")
    # Whisper reports bare codes, settings may carry a region; both must match.
    assert asr.Transcriber(languages=["zh-TW", "en"])._allowed("zh")
    # Auto-detect that reported nothing, and an unconfigured meeting, stay permissive.
    assert tr._allowed("") and asr.Transcriber()._allowed("pt")


def test_live_silence_default_rides_over_breaths() -> None:
    """Half-sentences on the TV traced to a 0.5s silence cutting speakers mid-breath. 0.7 still
    split sentences at thinking pauses, so the default is 0.9; locked here so a future edit can't
    quietly drop it back.

    Model-free on purpose — constructing a Vad would load silero, which the bare CI runner lacks.
    The Vad(min_silence=…) passthrough is exercised on the live pipeline against real models.
    """
    assert config.Config().vad_min_silence == 0.9


class _Seg:
    def __init__(self, text: str, no_speech_prob: float):
        self.text, self.no_speech_prob, self.start, self.end = text, no_speech_prob, 0.0, 1.0


def test_confident_silence_segments_are_dropped() -> None:
    """Whisper fills gaps between speakers with confident boilerplate — the dominant Vietnamese
    failure. A high no_speech_prob is the signal faster-whisper's own coupled no_speech_threshold
    misses when the avg_logprob is high, so we filter on it directly."""
    assert asr_gpu._spoken(_Seg("real speech", 0.10))
    assert asr_gpu._spoken(_Seg("borderline", asr_gpu.NO_SPEECH_MAX - 0.01))
    assert not asr_gpu._spoken(_Seg("đăng ký kênh", 0.95))

    # A segment carrying no such score (a fake in another test, an older path) is never dropped by
    # accident: absent the signal, the text is kept and the phrase filter still guards it downstream.
    class _Bare:
        text = "x"
    assert asr_gpu._spoken(_Bare())


def _term(source: str, mode: str = "hint") -> store.Term:
    return store.Term(id=0, source=source, lang="", mode=mode, category="", targets={})


def test_hotwords_leave_protected_words_alone() -> None:
    """`protect` means "do not rewrite this", not "listen for this".

    才夠 is registered only to stop the corrector rewriting it. Biasing the decoder toward an
    ordinary word would manufacture the mistake the entry exists to prevent — and 採購, the word it
    would be manufactured from, is the one people actually say in these meetings.
    """
    words = asr_gpu.hotwords_from([_term("生管"), _term("才夠", "protect"), _term("工程變更")])
    assert "才夠" not in words, words
    assert "生管" in words and "工程變更" in words, words

    # The other three modes are all "make this term happen", so all three are biased.
    every = asr_gpu.hotwords_from([_term("甲", "translate"), _term("乙", "keep"), _term("丙", "hint")])
    assert every.split() == ["甲", "乙", "丙"], every


def test_hotwords_stay_inside_whisper_prompt_window() -> None:
    """Past ~224 tokens Whisper drops the tail itself, and says nothing about it.

    Silent truncation is the failure mode here: recognition quietly degrades for whichever terms
    sorted last, and nothing in the transcript says the glossary stopped applying.
    """
    many = [_term(f"專有名詞{i:03d}") for i in range(200)]
    words = asr_gpu.hotwords_from(many)
    assert len(words) <= asr_gpu.HOTWORD_BUDGET, len(words)
    # Truncation is not silent from our side, and it keeps a prefix rather than returning nothing.
    assert words.startswith("專有名詞000"), words[:40]

    # A glossary that fits is passed through whole, in order.
    small = [_term("生管"), _term("工程變更")]
    assert asr_gpu.hotwords_from(small) == "生管 工程變更"
    assert asr_gpu.hotwords_from([]) == ""


def test_a_term_added_mid_meeting_biases_the_next_utterance() -> None:
    """It used to reach only the corrector, and bias nothing until the next meeting."""
    class Recorder:
        def __init__(self) -> None:
            self.hotwords = "生管"

        def set_hotwords(self, hotwords: str) -> None:
            self.hotwords = hotwords

    pipe = pipeline.Pipeline.__new__(pipeline.Pipeline)
    pipe._transcriber = Recorder()
    pipe._hotwords = "生管"

    # Unchanged glossary must not touch the recogniser at all.
    pipe._rebias([_term("生管")])
    assert pipe._transcriber.hotwords == "生管"

    pipe._rebias([_term("生管"), _term("工程變更")])
    assert pipe._transcriber.hotwords == "生管 工程變更"

    # And a protected term added mid-meeting still does not become a decoder target.
    pipe._rebias([_term("生管"), _term("工程變更"), _term("才夠", "protect")])
    assert pipe._transcriber.hotwords == "生管 工程變更"


def test_the_cpu_recogniser_accepts_hotwords_and_ignores_them() -> None:
    """sherpa-onnx cannot bias Whisper at all; callers must not have to know which one they hold."""
    cpu = asr.Transcriber.__new__(asr.Transcriber)
    cpu.set_hotwords("生管 工程變更")  # must not raise

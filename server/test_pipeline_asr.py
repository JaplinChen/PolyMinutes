"""The recogniser's two inputs from outside: the language whitelist and the glossary bias."""

from __future__ import annotations

import numpy as np

from . import asr, asr_gpu, config, pipeline, postprocess, store


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

    # Only `hint` is. The prompt is opt-in per term, not the default for the glossary.
    every = asr_gpu.hotwords_from([_term("甲", "translate"), _term("丙", "hint")])
    assert every.split() == ["丙"], every


def test_hotwords_leave_customer_names_alone() -> None:
    """`keep` is proper nouns, and biasing a proper noun makes Whisper write it into audio it is
    not in. 比雅久 was in the glossary for the 2026-08-05 meeting and came back four times in
    places the human transcript has no customer name at all. The corrector still recovers a name
    that really was said, from pinyin, on evidence rather than on prior."""
    words = asr_gpu.hotwords_from([_term("生管"), _term("比雅久", "keep"), _term("Vinfast", "keep")])
    assert words == "生管", words


def test_a_longer_prompt_is_not_free() -> None:
    """Eleven process terms added to the prompt cost a whole 20-second line on the 2026-08-05
    meeting — 六合找我們說那個 SP13 的模具送來試做, present with the short prompt and gone with the
    long one. So a term reaches the corrector by default and the prompt only when asked."""
    corrector_only = [_term(s, "translate") for s in ("檢具", "包料", "鍍層剝落", "合資案")]
    assert asr_gpu.hotwords_from(corrector_only) == ""
    # And the ones marked for it still get there.
    assert asr_gpu.hotwords_from(corrector_only + [_term("生管", "hint")]) == "生管"


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


def test_a_high_no_speech_score_drops_only_boilerplate_not_content() -> None:
    """A near-silence score no longer discards real speech — only text that reads as hallucination.

    Two ways a real utterance scores as silence. Hotwords: faster-whisper passes them as a decoder
    prompt prefix and the score is read at the first decoding step, so with a prefix in front of it
    that step answers a different question (2026-08-10: the same clips moved 0.19 -> 0.99 with the
    prompt on, text identical). And natively: on the 2026-08-05 morning meeting, accented room-mic
    Mandarin scored 0.86–0.96 with no prompt at all — 62 of 69 blanked clips were this gate firing
    on real speech. So the gate reads the text at a high score: content survives, and only the
    YouTube boilerplate it exists for (caught by is_hallucination/is_noise/is_degenerate) is dropped.
    """
    # Content text at a near-silence score is kept — with a glossary prompt or without one.
    assert asr_gpu._spoken(_Seg("real speech scored as silence", 0.99))
    assert asr_gpu._spoken(_Seg("real speech scored as silence", 0.99), biased=True)
    # The boilerplate the gate exists for is still dropped: the text gives it away, not the score.
    assert not asr_gpu._spoken(_Seg("đăng ký kênh", 0.95))


def test_an_empty_decode_is_retried_without_the_glossary_prompt() -> None:
    """The prompt is a prior, and a prior can talk the decoder out of a sentence.

    5 of 33 utterances lost on the 2026-08-10 meeting came back only once it was removed, so an
    utterance is not written off until the recogniser has been asked without it.
    """
    class Recogniser:
        def __init__(self) -> None:
            self.plain = 0

        def transcribe(self, samples, language):
            return "", language

        def transcribe_unbiased(self, samples, language):
            self.plain += 1
            return "六合找我們說那個 SP13 的模具送來試做", language

    said = postprocess.Utterance(0.0, np.zeros(16000, dtype=np.float32), speaker="S1")
    said.text, said.lang = "", "zh"
    recogniser = Recogniser()
    postprocess.transcribe_all([said], recogniser, forced={"S1": "zh"})
    assert said.text == "六合找我們說那個 SP13 的模具送來試做"
    assert recogniser.plain == 1

    # A recogniser without the second attempt (the sherpa fallback) must still work untouched.
    class Plain:
        def transcribe(self, samples, language):
            return "", language

    quiet = postprocess.Utterance(0.0, np.zeros(16000, dtype=np.float32), speaker="S1")
    quiet.text, quiet.lang = "", "zh"
    postprocess.transcribe_all([quiet], Plain(), forced={"S1": "zh"})
    assert quiet.text == ""

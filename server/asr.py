"""Voice activity detection and transcription, both from sherpa-onnx.

Two things here are not obvious and are load-bearing:

1. Whisper's language is fixed when the recognizer is built — sherpa-onnx has no per-utterance
   language option. So there is one recognizer per language, created on demand: a meeting that
   only uses two of three configured languages never loads the third.

2. Forcing the wrong language does not degrade gracefully, it collapses into repeated filler
   ("前來,前來,前來,..." for English audio decoded as Chinese). That collapse is detectable, so a
   suspect result is re-decoded with auto-detect rather than shipped as the transcript.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sherpa_onnx
from opencc import OpenCC

from . import config

# s2twp = Simplified -> Traditional with Taiwan phrase conversion ("軟件" -> "軟體").
# Whisper emits Simplified for zh regardless of the speaker, and Simplified subtitles on the
# meeting-room TV are an immediately visible failure.
_to_traditional = OpenCC("s2twp")

# Han, kana and hangul count as one token each; Latin words as one per word. Kana and hangul are
# here because Whisper reaches for them when decoding noise, and that output collapses too.
_CJK_OR_WORD = re.compile(r"[一-鿿ぁ-ヿ가-힣]|[A-Za-zÀ-ỹ]+")
# Bracketed spans, closing bracket optional: Whisper truncates these as often as it closes them
# ("[static" and "[Bell" both showed up in a 15-minute sample).
_ANNOTATION = re.compile(r"[\[(（【][^\])）】]*[\])）】]?")

# Whisper accepts at most 30 seconds and silently discards the rest — sherpa logs a warning and
# returns a transcript of the first 30 s only. VAD is configured to cut before this, but a guard
# here covers every caller instead of trusting one setting.
MAX_DECODE_SECONDS = 25.0


@dataclass
class Segment:
    """One utterance cut out by VAD."""

    samples: np.ndarray
    start: float  # seconds from the beginning of the session

    @property
    def duration(self) -> float:
        return len(self.samples) / config.SAMPLE_RATE


def is_degenerate(text: str) -> bool:
    """True when the decode collapsed into repetition, the signature of a wrong forced language.

    Measured over tokens rather than characters so that Vietnamese and English, which repeat
    characters far more than Chinese does, are judged on the same scale.
    """
    tokens = _CJK_OR_WORD.findall(text)
    if len(set(tokens)) == 1 and len(tokens) >= 5:
        return True  # one token, over and over: YAMAHA YAMAHA YAMAHA YAMAHA YAMAHA
    if len(tokens) < 8:
        return False  # too short to tell repetition from a genuinely terse utterance
    return len(set(tokens)) / len(tokens) < 0.3


# Whisper was trained on YouTube subtitles and reproduces their sign-offs when handed audio it
# cannot read. Measured over seven real interviews: 39 of 265 Vietnamese lines were channel
# promotion, none of it spoken. Matched as phrases, not keywords — a meeting may legitimately say
# "subscribe" or "訂閱", but not "subscribe to our channel".
_HALLUCINATIONS = re.compile(
    r"đăng ký kênh|theo dõi và (hẹn|đăng)|hẹn gặp lại|subscribe cho kênh|la la school"
    r"|thank(s| you) for watching|subscribe to (our|the) channel|please subscribe"
    # Whole line only. Whisper answers silence with a bare sign-off; a participant thanking
    # someone says what for, so a line that is nothing but the thanks is the decoder talking.
    r"|^thank you( very much)?[.!]?$|^see you next time[.!]?$"
    # The bell icon exists only on YouTube.
    r"|小鈴鐺"
    r"|訂閱(我們的)?頻道|點選訂閱|歡迎訂閱|請不吝|點贊|打賞|明鏡|點點欄目"
    # Subtitling credits and sign-offs, all of them from the same training data. Seen on seven
    # real interviews, where they arrive in the silence between speakers.
    r"|ming pao|amara\.org|字幕(組|小組)|字幕由|本期的?(影片|視頻|節目)|謝謝觀看|感謝觀看"
    # Whole line only. This room builds subtitling software and says 字幕 in earnest —
    # 會議紀錄的中文字幕要不要做 is a sentence someone here will say. A line that is nothing but
    # 中文字幕提供 is a credit.
    r"|^中文字幕(提供)?$"
    # Broadcast sign-offs, from a factory morning meeting where three of them arrived in a row
    # just before the real agenda started. 本集 and 本節目 only ever introduce a programme; a
    # meeting that says 本集團 or 年終節目 keeps both, since neither matches these.
    # 本集 followed by an ending, with room for the words between: the line seen in the wild was
    # 本集就這樣結束了, not 本集完. The lookahead keeps 本集團 out — 本集團的專案結束了 is a
    # sentence this room says.
    r"|歡迎收[看睇]|感謝收[看睇]|本集(?!團).{0,4}(完|結束|到此)|本期(完|播放)|本節目|下部節目"
    # 收看 is television vocabulary; a meeting says 看 or 收到. The gap allows 謝謝大家的收看,
    # which is what the recording actually produced. 收睇 and 多謝 are the same sign-off in
    # Cantonese — 多謝您收睇時局新聞,再會! arrived on a re-run over near-silence.
    r"|(請您|敬請|歡迎)關注|(謝謝|感謝|多謝)[^。！!]{0,4}收[看睇]"
    # The editor's credit, whole-line only. A wildcard after 剪輯 would eat 剪輯機台 and
    # 影片剪輯外包, and this filter runs on the live path too, where a false positive drops a
    # subtitle with nothing to show it happened. So: the line is nothing but 剪輯 and a name.
    r"|^剪輯[·:：]?[一-鿿]{2,3}$"
    # Anchored at the end: a sign-off is the last thing in the line, while 我們下次見面再談 is
    # someone arranging a meeting — and 我們下回見面再談 is the same sentence.
    r"|(下次|下期|下集|下回)見(再見|囉)?[。！!]?$|多多支援|請按贊|支持明鏡",
    re.IGNORECASE,
)


def is_hallucination(text: str) -> bool:
    """True for Whisper's YouTube boilerplate, which is never something a participant said.

    The line is normalised first, because it is judged before `_post` runs and Whisper always
    emits Simplified: every phrase below is written in Traditional, so 欢迎收看 and 本期视频 slid
    straight past a list that names them, and were converted to Traditional on the way to the
    transcript. Spaces go too — the decoder writes Mandarin word by word, so the same boilerplate
    also arrives as 請您 關注 and 訂閱 我們的 頻道.
    """
    return any(_HALLUCINATIONS.search(form) for form in _forms(text))


def _forms(text: str) -> tuple[str, ...]:
    """The line as written, converted to Traditional, and both again without spaces."""
    converted = _to_traditional.convert(text)
    return (text, converted, text.replace(" ", ""), converted.replace(" ", ""))


def is_noise(text: str) -> bool:
    """True for Whisper's non-speech annotations: `[MUSIC PLAYING]`, `(static)`, `[BLANK_AUDIO]`.

    Whisper does not stay silent when handed silence — it emits one of these tags. Measured on a
    real interview recording, every one of the first ten minutes of room noise before the meeting
    started produced one, and each became its own phantom speaker with its own phantom language.
    Only a decode that is *nothing but* annotation is dropped, so a real utterance that happens to
    contain a parenthesis survives.
    """
    return bool(text.strip()) and not _ANNOTATION.sub("", text).strip(" \t-–—.,")


class Vad:
    """Streaming VAD. Feed capture blocks, take completed utterances out."""

    def __init__(self, model: Path | None = None, buffer_seconds: int = 60,
                 min_speech: float = 0.25, min_silence: float = 0.5):
        """`min_speech` is a parameter because the two callers want different answers.

        Live capture wants everything: 「好」「對」「收到」 land under a second and belong on the
        TV. The post-meeting pass wants the opposite — a quarter-second of speech cannot carry a
        sentence, and Whisper never declines, so it answers those with the subtitle sign-offs it
        read in training. See POST_MEETING_MIN_SPEECH in postprocess.

        `min_silence` is the pause that ends an utterance, and it is a knob because how long a
        speaker pauses mid-thought varies by person and room. Too short cuts a breath into two
        half-sentences — the first arrives at Whisper as a fragment that collapses into filler —
        so the live pipeline runs it higher than this default. See vad_min_silence in config.
        """
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(model or config.VAD_MODEL)
        cfg.silero_vad.threshold = 0.5
        cfg.silero_vad.min_silence_duration = min_silence
        cfg.silero_vad.min_speech_duration = min_speech
        self.min_silence = min_silence
        cfg.silero_vad.max_speech_duration = 20.0  # force a cut so one monologue can't stall the pipeline
        cfg.sample_rate = config.SAMPLE_RATE
        self._vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=buffer_seconds)
        self._consumed = 0

    def push(self, block: np.ndarray) -> list[Segment]:
        """Feed one capture block; return whatever utterances completed as a result."""
        flat = block.reshape(-1).astype(np.float32)
        self._vad.accept_waveform(flat)
        self._consumed += len(flat)
        return self._drain()

    def flush(self) -> list[Segment]:
        """End of session: emit any utterance still buffered."""
        self._vad.flush()
        return self._drain()

    def _drain(self) -> list[Segment]:
        out = []
        while not self._vad.empty():
            seg = self._vad.front
            out.append(Segment(np.array(seg.samples, dtype=np.float32), seg.start / config.SAMPLE_RATE))
            self._vad.pop()
        return out


def default_threads() -> int:
    """Enough threads to matter, few enough to leave room for diarization and the rest of the box.

    Measured on a 20-core machine: small int8 runs at 0.23 realtime on 2 threads and 0.20 on 4,
    so past a handful the returns are not worth the contention.
    """
    return min(4, max(2, (os.cpu_count() or 4) // 4))


def missing_models(cfg: config.Config) -> list[str]:
    """Which of the files a live pipeline needs are not on disk, named so they can be gone and got.

    Constructing a Transcriber does not check — recognizers are built on first use, and that is
    also how the language-selection logic is tested on machines with no models. The VAD is worse:
    sherpa raises its own error mentioning a path, from a background thread.
    """
    absent = [str(p) for p in (config.VAD_MODEL, config.SPEAKER_MODEL) if not p.is_file()]
    try:
        Transcriber(model_dir=cfg.whisper_dir(), languages=cfg.languages)._paths()
    except FileNotFoundError as exc:
        absent.append(str(exc))
    return absent


class Transcriber:
    """Whisper recognizers, one per language, created on first use."""

    def __init__(self, model_dir: Path | None = None, num_threads: int | None = None,
                 provider: str = "cpu", quantized: bool = True,
                 languages: list[str] | None = None):
        self._dir = model_dir or config.WHISPER_DIRS["small"]
        self._threads = num_threads or default_threads()
        self._provider = provider
        # Quantized weights for live use; postprocess has no latency budget and prefers float32.
        self._quantized = quantized
        # Whisper auto-detect ranks every language it knows, so a noisy segment in a zh/vi/en
        # meeting comes back as Portuguese or Tibetan. Both were measured on a real recording.
        # Anything outside the configured set is a detection failure, not a participant.
        self._languages = list(languages or [])
        self._cache: dict[str, sherpa_onnx.OfflineRecognizer] = {}

    def _paths(self) -> tuple[str, str, str]:
        stem = self._dir.name.replace("sherpa-onnx-whisper-", "")
        suffixes = [".int8.onnx", ".onnx"] if self._quantized else [".onnx", ".int8.onnx"]

        def pick(kind: str) -> Path:
            for suffix in suffixes:
                candidate = self._dir / f"{stem}-{kind}{suffix}"
                if candidate.is_file():
                    return candidate
            raise FileNotFoundError(f"no {kind} weights for {stem} under {self._dir}")

        tok = self._dir / f"{stem}-tokens.txt"
        if not tok.is_file():
            raise FileNotFoundError(f"Whisper tokens file missing: {tok}")
        return str(pick("encoder")), str(pick("decoder")), str(tok)

    def _recognizer(self, language: str) -> sherpa_onnx.OfflineRecognizer:
        if language not in self._cache:
            enc, dec, tok = self._paths()
            self._cache[language] = sherpa_onnx.OfflineRecognizer.from_whisper(
                encoder=enc,
                decoder=dec,
                tokens=tok,
                language=language,  # '' means auto-detect
                num_threads=self._threads,
                provider=self._provider,
            )
        return self._cache[language]

    def _decode_chunk(self, samples: np.ndarray, language: str) -> tuple[str, str]:
        rec = self._recognizer(language)
        stream = rec.create_stream()
        stream.accept_waveform(config.SAMPLE_RATE, samples)
        rec.decode_stream(stream)
        result = stream.result
        # Whisper reports the language it decoded in, which is the only way to learn what an
        # auto-detected utterance actually was. Without it a speaker's language could never be
        # established and every utterance would stay on auto-detect forever.
        detected = (getattr(result, "lang", "") or "").strip().strip("<|>")
        return result.text.strip(), detected or language

    def _decode(self, samples: np.ndarray, language: str) -> tuple[str, str]:
        limit = int(MAX_DECODE_SECONDS * config.SAMPLE_RATE)
        if len(samples) <= limit:
            return self._decode_chunk(samples, language)

        parts = [self._decode_chunk(samples[i : i + limit], language) for i in range(0, len(samples), limit)]
        text = " ".join(t for t, _ in parts if t)
        return text, next((lang for _, lang in parts if lang), language)

    def _allowed(self, detected: str) -> bool:
        """Whisper reports bare codes ('zh'), settings may carry a region ('zh-TW'); match the base."""
        if not self._languages or not detected:
            return True
        base = detected.split("-")[0]
        return any(base == code.split("-")[0] for code in self._languages)

    def set_hotwords(self, hotwords: str) -> None:
        """Accepted and ignored: sherpa-onnx cannot bias Whisper decoding at all.

        Present so callers do not have to ask which recogniser they hold. The glossary still
        reaches this path through the post-decode corrector, which is the half that was measured to
        matter anyway.
        """

    def transcribe(self, samples: np.ndarray, language: str) -> tuple[str, str]:
        """Return (text, language_actually_used).

        A forced language that collapses is retried with auto-detect, and the caller is told which
        language produced the text it got so the speaker's language stats stay honest.
        """
        text, detected = self._decode(samples, language)
        if is_noise(text) or is_hallucination(text) or not self._allowed(detected):
            return "", detected

        if is_degenerate(text):
            # A collapse is a collapse whoever chose the language. This check used to run only
            # when one was forced, so a first-pass auto-detect could return 產品 產品 產品 產品
            # and have the language it invented for that counted as evidence of what the speaker
            # speaks. Retrying is only worth it when there is another language to try.
            if not language:
                return "", detected
            fallback, fallback_lang = self._decode(samples, "")
            # Same allow-list the forced pass is held to above. Without it the retry's auto-detect
            # could return a language the room never configured — Whisper guessing Portuguese or
            # Tibetan on noise — and it would reach the subtitles as valid text and count toward
            # what this speaker is taken to speak. The forced path filters that; the retry must too.
            if (is_noise(fallback) or is_hallucination(fallback) or is_degenerate(fallback)
                    or not self._allowed(fallback_lang)):
                return "", fallback_lang
            return _post(fallback, fallback_lang), fallback_lang

        return _post(text, detected), detected


def _post(text: str, language: str) -> str:
    # Only Chinese needs conversion. Whisper reports zh for both scripts and always emits
    # Simplified, so this is what keeps Simplified characters off the meeting-room TV.
    return _to_traditional.convert(text) if language.startswith("zh") else text

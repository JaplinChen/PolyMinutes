"""CTranslate2 transcriber, interchangeable with the sherpa-onnx one in `asr`.

Measured on this meeting room's box (RTX 5060 Ti, 20 cores): sherpa-onnx running Whisper small on
the CPU reaches 0.57 realtime only by taking every core, which makes the machine unusable for
anything else. The same recording through CTranslate2 on the GPU runs large-v3 at 0.064 — a nine
times faster wall clock on a far better model, with the CPU free.

Only the recogniser changes. VAD and speaker embeddings stay on sherpa-onnx: they are cheap, and
they are what the live path's latency actually depends on.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import asr, config

log = logging.getLogger("polyminutes.asr_gpu")

# Utterances decoded together. Thirty-two fits in 16 GB beside a large-v3 in float16.
BATCH_SIZE = 32
# Smallest batch the shrink-on-OOM retry will fall to before giving up. One utterance at a time is
# the floor: below it there is nothing left to shrink, and the failure is no longer about batch.
MIN_BATCH = 1
# Seconds to wait before the first retry, still at full batch. A contended card is usually the
# other consumer spiking for a moment — waiting it out keeps the wide batch, which shrinking spends.
OOM_WAIT_SECONDS = 1.0


def _is_oom(exc: Exception) -> bool:
    """Whether a decode failure is the GPU running short of memory rather than something else.

    Matched on the message because ctranslate2 surfaces a CUDA allocation failure as a plain
    RuntimeError with no type to catch. The strings are what it and cuBLAS actually print; the
    match is loose because the exact wording is not contractual and a missed OOM would fail the
    whole pass instead of shrinking.
    """
    if not isinstance(exc, RuntimeError):
        return False
    text = str(exc).lower()
    return ("out of memory" in text or "oom" in text
            or "cublas_status_alloc_failed" in text)
# Silence inserted between utterances when they are laid end to end for batching. Every gap is
# real audio through the encoder, so it stays as short as the boundaries tolerate.
BATCH_GAP_SECONDS = 0.2
# Greedy decoding measurably dropped whole utterances, so this is 5 as that note anticipated.
# Alternating four runs over 6.7 minutes of a real morning meeting, 47 utterances:
#
#     beam 1   12.9s   2 utterances decoded to nothing   1528 characters
#     beam 5   13.2s   0                                 1669
#
# One of the two it silently dropped was a hundred characters of customer-visit detail. The cost
# is 2-4% wall clock: at batch 32 the GPU is not compute-bound, so the wider search rides along.
BEAM_SIZE = 5

# A segment the model itself scores as very likely silence, yet returned text for, is Whisper
# hallucinating in a gap between speakers — the dominant Vietnamese failure (its YouTube-subtitle
# training makes it fill unclear audio with channel sign-offs). That boilerplate is often confident,
# a high avg_logprob, so it slips past faster-whisper's own no_speech_threshold, which only suppresses
# when the logprob is ALSO low. Filtering on no_speech_prob alone catches the confident case the
# coupled check misses. Set high so it only ever drops near-certain silence — real speech, even weak
# Vietnamese, scores far below this. Tune against real audio with `scripts.eval_harness`.
NO_SPEECH_MAX = 0.85

# Greedy, single attempt. faster-whisper defaults to a temperature ladder [0, 0.2 ... 1.0]: when a
# decode trips its own compression-ratio or logprob check it re-rolls at a higher temperature and
# keeps re-rolling until something passes. On real speech that rescues the occasional segment; on
# the silence between speakers there is nothing to decode, so every re-roll is the model inventing
# text more freely than the last, and the thing it invents is its training data's subtitle
# boilerplate. Pinning to a scalar disables the ladder — a segment that fails now comes back as it
# was decoded once, and the no-speech gate above throws it away.
TEMPERATURE = 0.0


def _spoken(seg, biased: bool = False) -> bool:
    """False for a segment the decoder is near-certain is silence — a hallucinated line in a gap.

    `biased` says a glossary prompt was in the decode, and then this gate is off, because
    no_speech_prob is not a reading of the audio any more. faster-whisper passes hotwords as a
    decoder prompt prefix, and no_speech_prob is the probability of the <|nospeech|> token at the
    first decoding step — with a prefix in front of it, that step is answering a different
    question. Measured on the 2026-08-10 meeting, same clips, same weights, prompt the only
    difference:

        1132.3s  0.44 -> 0.86    這個焊接氣體從這邊挑出來
        1324.4s  0.68 -> 0.99    貼在閥門上面…標籤貼的位置要確認一下
        1669.7s  0.27 -> 0.91    像剛剛那個不曉得漢管要再確認一下…
        2113.1s  0.35 -> 0.94    我們到底有哪些浪費要趕快處理掉…
        9509.0s  0.19 -> 0.99    大家如果有空的話…

    The text was correct in both columns; only the score moved, and this gate then threw the whole
    utterance away. Across five real meetings that was 3 to 7 minutes of speech per meeting.
    Hallucinations stay out through the text filters in `_judge`, which read what was written
    rather than a score: eight seconds of pure silence still decodes to nothing with the gate off.
    """
    return biased or getattr(seg, "no_speech_prob", 0.0) < NO_SPEECH_MAX


def _add_cuda_dlls() -> None:
    """Put the pip-installed CUDA runtime on PATH before CTranslate2 loads.

    `os.add_dll_directory` is not enough — CTranslate2 resolves cuBLAS and cuDNN through the
    default search order, which on Windows means PATH. Without this the model loads and then fails
    on the first encode with 'Library cublas64_12.dll is not found'.
    """
    nvidia = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
    if not nvidia.is_dir():
        return
    dirs = [str(p) for p in nvidia.glob("*/bin") if p.is_dir()]
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])


def available() -> bool:
    """True when a CUDA device and the CTranslate2 runtime are both present."""
    try:
        _add_cuda_dlls()
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False


class Transcriber:
    """Same surface as `asr.Transcriber`: transcribe(samples, language) -> (text, language)."""

    def __init__(self, model: str | Path | None = None, device: str = "cuda",
                 compute_type: str = "float16", languages: list[str] | None = None,
                 hotwords: str = "", live: bool = False):
        _add_cuda_dlls()
        from faster_whisper import WhisperModel

        self._languages = list(languages or [])
        self._hotwords = hotwords
        name = str(model or config.gpu_model(self._languages, live=live))
        index = config.gpu_index() if device == "cuda" else 0
        self._model = WhisperModel(name, device=device, device_index=index,
                                   compute_type=compute_type)
        self._batched = None
        # The largest batch this card has been seen to hold, remembered across decodes.
        self._batch = BATCH_SIZE
        log.info("ct2 model %s on %s:%d/%s", name, device, index, compute_type)

    def set_hotwords(self, hotwords: str) -> None:
        """Re-bias without reloading the model.

        A term added during a meeting used to bias nothing until the next one, because the string
        was baked in when the recogniser was built. Only the prompt text changes here; the weights
        are untouched, so this costs nothing and can run between utterances.
        """
        self._hotwords = hotwords

    def transcribe_many(self, clips: list[np.ndarray], language: str) -> list[tuple[str, str]]:
        """Decode many utterances in one pass, keeping every boundary.

        Whisper's encoder always processes a thirty-second window, so a five-second utterance
        costs the same as a thirty-second one — and a meeting is thousands of short utterances.
        Measured on this box: 0.186 realtime one at a time against 0.045 batched.

        The clips are laid end to end with a second of silence between them and handed over with
        `clip_timestamps`, which is what keeps the boundaries. Without it the batching pipeline
        applies its own VAD and returns five segments where there were twenty-one — and speaker
        identity, per-speaker language and the subtitle line are all pinned to our boundaries, so
        letting the model re-segment would take the transcript apart.
        """
        if not clips:
            return []

        gap = np.zeros(int(BATCH_GAP_SECONDS * config.SAMPLE_RATE), dtype=np.float32)
        spans, parts, at = [], [], 0.0
        for clip in clips:
            seconds = len(clip) / config.SAMPLE_RATE
            spans.append({"start": at, "end": at + seconds})
            parts += [clip.astype(np.float32), gap]
            at += seconds + BATCH_GAP_SECONDS

        if self._batched is None:
            # Imported here, not at module load: faster-whisper is a GPU-only dependency, and this
            # is the one place that constructs the pipeline. A test that injects a fake `_batched`
            # never reaches it, which is what lets the OOM retry be exercised without a GPU.
            from faster_whisper import BatchedInferencePipeline

            self._batched = BatchedInferencePipeline(model=self._model)
        segments, info = self._decode_batched(np.concatenate(parts), language, spans)

        # Each segment is placed by its midpoint, so a decode that runs slightly over its clip
        # still lands on the utterance it came from.
        texts = ["" for _ in clips]
        biased = bool(self._hotwords)
        for seg in segments:
            # A confident-silence segment is Whisper filling a gap between speakers; dropping it here
            # keeps the hallucinated text out of the utterance it would otherwise be assigned to.
            if not _spoken(seg, biased):
                continue
            middle = (seg.start + seg.end) / 2
            for i, span in enumerate(spans):
                if span["start"] <= middle <= span["end"]:
                    texts[i] = (texts[i] + seg.text).strip()
                    break

        detected = (info.language or language or "").strip()
        return [self._judge(text, detected) for text in texts]

    def _decode_batched(self, audio: np.ndarray, language: str, spans: list[dict]):
        """Run the batched decode, giving the card room when it is short of memory.

        The pressure this handles is a second consumer on the same GPU — a local LLM running the
        summary or correction stage while a recording is being reprocessed. Two moves, cheapest
        first: wait once at full batch, because contention is usually a passing spike; then halve
        the batch and try again, down to one utterance. Shrinking cuts the activation memory the
        batch needs, which is the part we control — it does nothing for the resident weights, so a
        card too full to hold the model at all still fails at batch one, and that failure says the
        real problem is elsewhere (the other consumer, or a context left unusable by the OOM).
        """
        # Starts where the last decode ended up, not at BATCH_SIZE. The card's capacity does not
        # change between two batches of the same meeting, so re-learning it each time costs a
        # failed attempt plus OOM_WAIT_SECONDS on every batch — measured on a 55-minute recording
        # after the glossary grew, that was the difference between 88 seconds and over three
        # minutes, all of it spent discovering the same answer.
        batch = self._batch
        waited = False
        while True:
            try:
                segments, info = self._batched.transcribe(
                    audio, language=language or None, beam_size=BEAM_SIZE,
                    temperature=TEMPERATURE,
                    batch_size=batch, vad_filter=False, clip_timestamps=spans,
                    hotwords=self._hotwords or None, condition_on_previous_text=False,
                )
                # Drained here, inside the guard. faster-whisper returns a generator and does the
                # decoding on iteration, so returning it unread put every OOM this function exists
                # to catch outside the try — the caller hit it in its own `for seg in segments`
                # and the wait-and-shrink below never ran once. Found when a longer glossary
                # pushed a real reprocess over the card: it failed outright with no warning
                # logged, at batch 32, with a step down to 16 sitting right here unused.
                self._batch = batch
                return list(segments), info
            except RuntimeError as exc:
                # Any CUDA-flavoured error triggers a step back, not only a recognised OOM string:
                # once the card is contended the wording is not guaranteed, and stepping back is
                # the right response to all of them.
                if not (_is_oom(exc) or "cuda" in str(exc).lower()):
                    raise
                if not waited:
                    waited = True
                    log.warning("GPU short of memory, waiting %.0fs and retrying at batch %d",
                                OOM_WAIT_SECONDS, batch)
                    time.sleep(OOM_WAIT_SECONDS)
                    continue
                if batch > MIN_BATCH:
                    batch = max(MIN_BATCH, batch // 2)
                    log.warning("GPU still short, decoding at batch %d", batch)
                    continue
                # Nothing left to give up. This is not batch 32 being greedy — at batch one the
                # weights alone do not fit, so another process holds the memory or the context is
                # spent. Re-raised for the caller (the post-meeting pass) to fail visibly.
                log.error("GPU cannot decode even one utterance; another process likely holds the "
                          "card, or its context is unusable — a restart may be needed")
                raise

    def _judge(self, text: str, detected: str) -> tuple[str, str]:
        if (asr.is_noise(text) or asr.is_hallucination(text) or asr.is_degenerate(text)
                or not self._allowed(detected)):
            return "", detected
        return asr._post(text, detected), detected

    def transcribe(self, samples: np.ndarray, language: str) -> tuple[str, str]:
        segments, info = self._model.transcribe(
            samples.astype(np.float32),
            language=language or None,  # None means detect
            beam_size=BEAM_SIZE,
            temperature=TEMPERATURE,
            # Hotwords are the biasing sherpa-onnx cannot do for Whisper at all.
            hotwords=self._hotwords or None,
            condition_on_previous_text=False,  # one VAD utterance at a time carries no history
        )
        text = "".join(s.text for s in segments if _spoken(s, bool(self._hotwords))).strip()
        detected = (info.language or language or "").strip()

        # Same three refusals as the sherpa path, including the collapse check: a first-pass
        # auto-detect that returns 產品 產品 產品 產品 must not have the language it invented for
        # that counted as evidence of what the speaker speaks.
        return self._judge(text, detected)

    def transcribe_unbiased(self, samples: np.ndarray, language: str) -> tuple[str, str]:
        """One more attempt with the glossary prompt taken out, for a clip that came back empty.

        The prompt is a prior, and a prior can talk the decoder out of a sentence: measured on the
        2026-08-10 meeting, 5 of 33 lost utterances decoded to text only once the hotwords were
        gone. Nothing else changes — same weights, same clip — so this is the cheapest thing to
        try before writing an utterance off, and it runs on the handful that failed rather than
        the thousands that did not. With no prompt in the decode the no-speech gate is meaningful
        again and applies as usual.
        """
        saved = self._hotwords
        self._hotwords = ""
        try:
            return self.transcribe(samples, language)
        finally:
            self._hotwords = saved

    def _allowed(self, detected: str) -> bool:
        if not self._languages or not detected:
            return True
        base = detected.split("-")[0]
        return any(base == code.split("-")[0] for code in self._languages)


def maybe(languages: list[str], hotwords: str = "", live: bool = False) -> Transcriber | None:
    """The GPU recogniser when this machine can run it, otherwise None so the caller falls back.

    Auto-enabled rather than configured: it is faster and more accurate on every axis measured, so
    a knob would only ever be turned one way. `POLYMINUTES_NO_GPU=1` exists for the case where
    the card is needed for something else.
    """
    if os.environ.get("POLYMINUTES_NO_GPU"):
        return None
    if not available():
        return None
    try:
        return Transcriber(languages=languages, hotwords=hotwords, live=live)
    except Exception:
        log.exception("GPU transcriber unavailable, falling back to CPU")
        return None


# Characters of glossary allowed into the decoder prompt. Whisper reserves half its 448-token
# context for prompt text, so hotwords past ~224 tokens are silently dropped — and it drops the
# tail, which is whichever terms happen to sort last. Budgeting in characters rather than tokens
# is deliberately pessimistic: one token per character is the worst case (Chinese), so 200 stays
# inside the window even for a glossary that is entirely CJK, and leaves room for the scaffolding
# faster-whisper wraps around it.
HOTWORD_BUDGET = 200


def hotwords_from(terms: list) -> str:
    """faster-whisper takes one string; the glossary is a list of terms.

    Three things are filtered out. `protect` terms, because that mode means "this word is real, do
    not rewrite it" — 才夠 is registered only to shield it from the corrector, and biasing the
    decoder toward an ordinary word would manufacture the very mistake the glossary entry exists to
    prevent. And anything past the budget, because the alternative is Whisper truncating it for us,
    without saying so.

    Only `hint` is biased, which is the mode whose whole meaning is "listen for this". Every other
    mode reaches the recogniser through `correct.Corrector` instead, which rewrites on evidence —
    it needs the pinyin to match what was actually decoded — where the prompt acts on prior alone.
    Two measurements on the 2026-08-05 meeting, both against the human transcript:

      * 比雅久 was in the glossary as a customer name, and Whisper wrote it into four places the
        meeting has no customer name at all: 「收料，比雅久」「比雅久分的資格」「觀念大概比雅久」.
      * Adding eleven process terms to the prompt cost a whole 20-second line — 六合找我們說那個
        SP13 的模具送來試做 — which decoded fine with the shorter prompt and is gone with the
        longer one. Same audio, same weights, same batch size; only the prompt changed.

    The prompt is not free and it is not evidence. What it buys is a term the decoder would not
    otherwise produce at all, which is a deliberate choice about a specific word — so it is opt-in,
    by marking the term `hint`, rather than the default for everything in the glossary.
    """
    usable = [t for t in terms if getattr(t, "mode", "") == "hint"]
    kept, used = [], 0
    for term in usable:
        cost = len(term.source) + 1  # the joining space
        if used + cost > HOTWORD_BUDGET:
            continue
        kept.append(term.source)
        used += cost
    if len(kept) < len(usable):
        # Named, not counted: knowing which terms lost their bias is the difference between
        # diagnosing a recognition complaint and guessing at it.
        dropped = [t.source for t in usable if t.source not in kept]
        log.warning("glossary exceeds the %d-character hotword budget; %d term(s) not biased: %s",
                    HOTWORD_BUDGET, len(dropped), ", ".join(dropped[:20]))
    return " ".join(kept)

"""Wires capture into subtitles: VAD -> speaker -> transcribe -> translate -> emit.

Order matters and is not interchangeable. The speaker must be identified *before* transcription
because Whisper's language is chosen per recognizer, and forcing the wrong one does not degrade —
it collapses into repeated filler. Speaker embeddings need only the waveform, so putting
clustering first costs no extra latency.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import asr, asr_gpu, config, correct, diarize, translate
from .retry import Retries
from .store import Store

log = logging.getLogger("polyminutes.pipeline")

# Blocks the pipeline may fall behind before it starts dropping audio. 600 blocks = 60 s.
TAP_CAPACITY = 600
# Warn once the backlog passes this; a sustained backlog means the realtime factor is above 1
# and subtitles will drift further behind for the rest of the meeting.
BACKLOG_WARN = 100


@dataclass
class Emitted:
    """One subtitle line as the browser sees it."""

    id: int
    start: float
    speaker: str
    lang: str
    source: str
    translations: dict[str, str] = field(default_factory=dict)
    refined: bool = False
    status: str = "ok"

    def event(self, kind: str) -> dict:
        return {
            "type": kind,
            "line": {
                "id": self.id,
                "start": round(self.start, 2),
                "speaker": self.speaker,
                "lang": self.lang,
                "source": self.source,
                "translations": self.translations,
                "refined": self.refined,
                "status": self.status,
            },
        }


class Pipeline:
    """Consumes audio blocks on a worker thread and emits subtitle events."""

    def __init__(self, cfg: config.Config, store: Store, session_id: int,
                 translator: translate.Translator | None,
                 emit: Callable[[dict], None], channels: int = 1):
        self._cfg = cfg
        self._store = store
        self._session = session_id
        self._translator = translator
        self._emit = emit

        # Every model this needs, checked before any of them is opened. They load lazily or raise
        # library errors that say nothing useful, and the pipeline runs on its own thread — so a
        # machine with nothing downloaded used to accept "start recording", capture the whole
        # meeting and produce zero lines, with the only trace a log line nobody was watching.
        missing = asr.missing_models(cfg)
        if missing:
            raise FileNotFoundError("speech models not found: " + ", ".join(missing))

        # Items are (source, block); a bare ndarray is accepted as source "" so the single-channel
        # feed (and every test that pushes plain blocks) is unchanged. (source, None) ends one
        # channel, a bare None ends the only channel.
        self.tap: queue.Queue = queue.Queue(maxsize=TAP_CAPACITY)
        self._channels = max(1, channels)
        # One VAD per channel: interleaving room mic and Teams loopback through a single VAD would
        # splice a local and a remote turn into one utterance. Created on first block from a source.
        self._vad: dict[str, asr.Vad] = {}
        # GPU first: measured on this box it is both faster and markedly more accurate.
        self._hotwords = asr_gpu.hotwords_from(store.glossary())
        self._transcriber = (asr_gpu.maybe(cfg.languages, self._hotwords, live=True)
                             or asr.Transcriber(model_dir=cfg.whisper_dir(),
                                                languages=cfg.languages))
        self._diarizer = diarize.Diarizer(cfg=cfg, known=diarize.load_known(store),
                                          known_languages=store.speaker_languages())
        self._thread: threading.Thread | None = None

        self._context: list[translate.Line] = []
        # (line id, start seconds, line, translations as emitted) of the utterance eligible for one
        # refinement pass. The translations ride along because a revision may touch only the source
        # — the update event must still carry the rest, or the page replaces the line with one that
        # has no subtitle under it.
        self._previous: tuple[int, float, translate.Line, dict[str, str]] | None = None
        self._retries = Retries()
        self.backlog_peak = 0
        self.errors = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="pipeline", daemon=True)
        self._thread.start()

    def join(self, timeout: float = 30) -> None:
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _vad_for(self, source: str) -> asr.Vad:
        vad = self._vad.get(source)
        if vad is None:
            vad = self._vad[source] = asr.Vad(min_silence=self._cfg.vad_min_silence)
        return vad

    def _run(self) -> None:
        try:
            ended = 0
            while (item := self.tap.get()) is not None:
                source, block = item if isinstance(item, tuple) else ("", item)
                if block is None:  # this channel is done
                    ended += 1
                    for segment in self._vad_for(source).flush():
                        self._handle(segment, source)
                    if ended >= self._channels:
                        break
                    continue
                self.backlog_peak = max(self.backlog_peak, self.tap.qsize())
                if self.tap.qsize() == BACKLOG_WARN:
                    log.warning("pipeline backlog %d blocks — realtime factor above 1", self.tap.qsize())
                for segment in self._vad_for(source).push(block):
                    self._handle(segment, source)
            else:
                # A bare None ends the single-channel feed; flush every VAD that saw audio.
                for source, vad in self._vad.items():
                    for segment in vad.flush():
                        self._handle(segment, source)
            self._retries.drain(self._diarizer.language_for, self._recover)
        except Exception:  # a crashed pipeline must not take the recording with it
            log.exception("pipeline stopped")

    def _handle(self, segment: asr.Segment, source: str = "") -> None:
        try:
            speaker = self._diarizer.assign(segment.samples, source)
            # A voice the room already knows arrives named. The centroid is stored either way, so
            # naming an unknown speaker afterwards is enough to recognise them next time.
            self._store.save_voiceprint(self._session, speaker.code, speaker.centroid.tobytes())
            if name := self._diarizer.recognised.pop(speaker.code, ""):
                self._store.set_speaker_name(self._session, speaker.code, name)
            # The glossary is read per utterance so a term added mid-meeting takes effect at once,
            # which is how the glossary page is used in practice. Read before decoding, not after,
            # so the same read biases the recogniser as well as correcting what it returns —
            # a term used to reach only the corrector and bias nothing until the next meeting.
            terms = self._store.glossary()
            self._rebias(terms)

            forced = self._diarizer.language_for(speaker)
            text, used = self._transcriber.transcribe(segment.samples, forced)
            if not text:
                # Held rather than dropped. The post-meeting pass recovered 992 real lines this way
                # across seven interviews — a decode that fails under one language routinely
                # succeeds under the speaker's own, and the live path used to bin them in silence.
                self._retries.hold(segment, speaker, forced)
                return
            self._diarizer.observe_language(speaker, used)
            text = correct.Corrector(terms, self._store.corrections()).fix(text)

            line = translate.Line(text=text, lang=used or forced, speaker=speaker.code)
            targets = [c for c in self._cfg.languages if c != line.lang]

            # A translation that fails must cost the translation, not the utterance. This used to
            # raise into the handler's catch-all, so an API hiccup dropped the whole line — the
            # room saw nothing at all where it should have seen the original text untranslated.
            status = "ok"
            try:
                result = self._translate(line, targets)
            except Exception:
                log.exception("translation failed at %.2fs", segment.start)
                result, status = translate.Result({}), "translate_failed"

            line_id = self._store.add_line(
                self._session, segment.start, speaker.code, line.lang, text, result.translations,
                status=status, end_time=segment.start + segment.duration,
            )
            self._emit(Emitted(line_id, segment.start, speaker.code, line.lang, text,
                               result.translations, status=status).event("line"))

            self._apply_refinement(result)

            self._previous = (line_id, segment.start, line, result.translations)
            self._context = (self._context + [line])[-config.CONTEXT_LINES:]

            # Only now, with this speaker's language possibly just settled, is a retry worth
            # spending GPU on. Guarded rather than delegated, so a meeting that is decoding fine
            # pays one list check here instead of a lookup per segment.
            if self._retries.held:
                self._retries.retry(speaker, self._diarizer.language_for(speaker), self._recover)
        except Exception:
            self.errors += 1
            log.exception("segment at %.2fs failed", segment.start)

    def _recover(self, segment: asr.Segment, speaker: diarize.Speaker, language: str) -> bool:
        """Decode a held utterance again under `language` and emit it. True if anything came back.

        Allowed to raise: `Retries` takes the entry off the held list before calling, and counts
        both the failure and the escape.
        """
        text, used = self._transcriber.transcribe(segment.samples, language)
        if not text:
            return False

        # Deliberately not observe_language: this segment is being decoded a second time, and
        # letting it vote again would count one utterance twice toward what this speaker speaks.
        text = correct.Corrector(self._store.glossary(), self._store.corrections()).fix(text)
        line = translate.Line(text=text, lang=used or language, speaker=speaker.code)
        targets = [c for c in self._cfg.languages if c != line.lang]

        translations, status = {}, "ok"
        if self._translator and targets:
            try:
                # No context and no `previous`: this utterance is arriving out of order, so the
                # surrounding lines are not the ones that surrounded it, and offering them as
                # context would mislead the translator rather than help it.
                translations = self._translator.translate(
                    line, targets, terms=self._store.glossary()).translations
            except Exception:
                log.exception("late translation failed at %.2fs", segment.start)
                status = "translate_failed"

        line_id = self._store.add_line(self._session, segment.start, speaker.code, line.lang, text,
                                       translations, status=status,
                                       end_time=segment.start + segment.duration)
        # Not added to `_context` or `_previous`: those model what was just said, and this was said
        # earlier. Feeding it in would hand the next line the wrong neighbour and spend that line's
        # one refinement pass revising an utterance from further back.
        self._emit(Emitted(line_id, segment.start, speaker.code, line.lang, text, translations,
                           status=status).event("line"))
        log.info("recovered the utterance at %.2fs under %s", segment.start, line.lang)
        return True

    def _rebias(self, terms: list) -> None:
        """Push the glossary into the recogniser when it has changed since the last utterance.

        Compared as a string rather than tracked with a revision counter: the glossary is already
        being read for the corrector, so this is a comparison of two short strings on a path that
        was about to do a model inference. Nothing is stored that could go stale.
        """
        hotwords = asr_gpu.hotwords_from(terms)
        if hotwords != self._hotwords:
            self._hotwords = hotwords
            self._transcriber.set_hotwords(hotwords)
            log.info("glossary changed mid-meeting, re-biasing the recogniser")

    def _translate(self, line: translate.Line, targets: list[str]) -> translate.Result:
        if not self._translator or not targets:
            return translate.Result({})
        prev_line = self._previous[2] if self._previous else None
        # The previous line's correction targets are its own languages, not this line's: in a mixed
        # zh/en/vi meeting the two lines rarely share a language, and reusing `targets` asked the
        # model to re-translate the previous line into the wrong ones.
        prev_targets = ([c for c in self._cfg.languages if c != prev_line.lang]
                        if prev_line else None)
        return self._translator.translate(
            line, targets, context=self._context, previous=prev_line,
            terms=self._store.glossary(), prev_targets=prev_targets
        )

    def _apply_refinement(self, result: translate.Result) -> None:
        """Rewrite the previous line if the model judged it wrong in hindsight.

        Each line gets exactly one chance: `_previous` advances every segment, so a corrected line
        is never revisited. Subtitles that keep shifting are harder to read than subtitles that are
        slightly off, which is why this is one-shot rather than iterative.
        """
        if not self._previous:
            return
        if not (result.previous_source or result.previous_translations):
            return

        prev_id, prev_start, prev_line, prev_translations = self._previous
        source = result.previous_source or prev_line.text
        self._store.update_line(prev_id, source, result.previous_translations)
        # The event mirrors what the store now holds: revised languages replaced, the rest kept.
        # Emitted verbatim, a source-only revision blanked the subtitle the room was reading, and
        # a partial one (en revised, vi not) blanked the other language — the page replaces the
        # whole line with whatever the update carries.
        merged = {**prev_translations, **result.previous_translations}
        self._emit(Emitted(prev_id, prev_start, prev_line.speaker, prev_line.lang, source,
                           merged, refined=True).event("update"))

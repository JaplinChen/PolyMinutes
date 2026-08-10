"""The held-utterance buffer: what the live pipeline does with a decode that came back empty.

A decode that fails under one language routinely succeeds under the speaker's own, and the live
path used to bin those in silence — the post-meeting pass recovered 992 real lines that way across
seven interviews. So an empty decode is held rather than dropped, and retried once the speaker's
language has settled.

Kept apart from the pipeline because it is the one part with a policy of its own: a memory budget,
an eviction order, and a rule that each utterance gets exactly one more attempt whatever happens.
"""

from __future__ import annotations

import logging
from typing import Callable

from . import asr, diarize

log = logging.getLogger("polyminutes.pipeline")

# Utterances held back for a second attempt once their speaker's language is known. Each one keeps
# its raw float32 audio — 20 s of it is about 1.3 MB — so this is a memory budget as much as a
# policy one. Twenty-four is roughly thirty seconds of held speech spread across the room, past
# which a meeting is failing to decode so consistently that retrying is not the answer.
RETRY_BUFFER = 24

# Decodes one held utterance under a language and returns whether anything came back.
Recover = Callable[[asr.Segment, diarize.Speaker, str], bool]


class Retries:
    """Utterances waiting on a second attempt, and the tally of how those attempts went."""

    def __init__(self, capacity: int = RETRY_BUFFER) -> None:
        self.capacity = capacity
        # (segment, speaker, language already tried).
        self.held: list[tuple[asr.Segment, diarize.Speaker, str]] = []
        self.recovered = 0
        self.dropped = 0

    def hold(self, segment: asr.Segment, speaker: diarize.Speaker, tried: str) -> None:
        """Keep a failed utterance for one more attempt, oldest evicted first."""
        if len(self.held) >= self.capacity:
            evicted = self.held.pop(0)
            self.dropped += 1
            log.info("retry buffer full, giving up on the utterance at %.2fs", evicted[0].start)
        self.held.append((segment, speaker, tried))

    def retry(self, speaker: diarize.Speaker, language: str, recover: Recover) -> None:
        """Re-decode this speaker's held utterances now that their language may have settled.

        Only theirs, and only when the language to try differs from the one that already failed —
        re-running the same audio under the same language would produce the same nothing. Each
        utterance gets exactly one retry whatever the outcome, so a room full of noise cannot build
        a backlog of audio the pipeline keeps paying to decode.
        """
        if not language:
            return
        ready = [h for h in self.held if h[1].code == speaker.code and h[2] != language]
        for entry in ready:
            self.held.remove(entry)
            self._attempt(entry[0], speaker, language, recover)

    def drain(self, language_for: Callable[[diarize.Speaker], str], recover: Recover) -> None:
        """Last attempt at whatever is still held when the meeting ends.

        By now every speaker has said all they are going to, so a language that never settled never
        will. Anything still failing is left to the post-meeting pass, which re-derives the whole
        recording anyway — this is about not throwing away what one more try would recover.
        """
        for segment, speaker, tried in list(self.held):
            language = language_for(speaker)
            if language and language != tried:
                self._attempt(segment, speaker, language, recover)
            else:
                self.dropped += 1
        self.held.clear()
        if self.recovered or self.dropped:
            log.info("held utterances: %d recovered, %d dropped", self.recovered, self.dropped)

    def _attempt(self, segment: asr.Segment, speaker: diarize.Speaker, language: str,
                 recover: Recover) -> None:
        # Never raises. The caller has already taken this utterance off the held list, so an
        # exception escaping here would lose it without even counting it — the silent drop this
        # whole path exists to remove. It also must not fail the live segment that triggered the
        # retry: recovering an old utterance is strictly a bonus on top of that one.
        try:
            got = recover(segment, speaker, language)
        except Exception:
            self.dropped += 1
            log.exception("retrying the utterance at %.2fs failed", segment.start)
            return
        if got:
            self.recovered += 1
        else:
            self.dropped += 1

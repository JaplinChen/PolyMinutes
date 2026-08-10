"""Who spoke: per-session voiceprints, and the names this room has learned.

Naming S1 as "Vincent" is knowledge the meeting room throws away every time. Kept here, the next
meeting recognises the voice instead of asking again. Three tables because the facts arrive at
different times: the centroid exists while the meeting runs, the name usually arrives afterwards
from the transcript page, and only then can it be promoted to something the room knows by voice.

A mixin over `Store` rather than an object of its own: every method here is one statement against
the same connection under the same lock, so giving it a connection of its own would mean a second
writer on a database whose whole locking story is "one connection, one lock".
"""

from __future__ import annotations

import sqlite3
import threading

import numpy as np

from . import config

# A voice needs a handful of anchors, not a history: past this many prints per name the least
# informative one is dropped when a new variant arrives.
KNOWN_PRINT_CAP = 8
# A new print this close to one already stored teaches nothing — skip it, so the set stays a spread
# of a person's real variants rather than the same utterance saved over and over.
KNOWN_DUP_THRESHOLD = 0.85


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _sample(row: sqlite3.Row | None) -> tuple[str, float, float | None] | None:
    """A sample row as (recording, start, seconds), with seconds unknown on transcripts written
    before end_time was recorded."""
    if row is None:
        return None
    end = row["end_time"]
    span = float(end) - float(row["start"]) if end is not None else None
    return (row["wav"], row["start"], span if span and span > 0 else None)


class SpeakerStore:
    """Speaker tables. Mixed into `Store`, which opens the connection and owns the lock."""

    _db: sqlite3.Connection
    _lock: threading.Lock

    def set_speaker_name(self, session_id: int, code: str, name: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO speaker_name (session_id, code, name) VALUES (?,?,?) "
                "ON CONFLICT(session_id, code) DO UPDATE SET name=excluded.name",
                (session_id, code, name),
            )
            self._db.commit()

    def clear_speaker_names(self, session_id: int) -> None:
        """Drop every name mapping for a session. Used before a reprocess renames the new codes:
        the old mappings point at codes the re-derive renumbered, so keeping them mislabels."""
        with self._lock:
            self._db.execute("DELETE FROM speaker_name WHERE session_id=?", (session_id,))
            self._db.commit()

    def speaker_names(self, session_id: int) -> dict[str, str]:
        with self._lock:
            return {r["code"]: r["name"] for r in self._db.execute(
                "SELECT code, name FROM speaker_name WHERE session_id=?", (session_id,)
            )}

    def save_voiceprint(self, session_id: int, code: str, centroid: bytes) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO voiceprint (session_id, code, centroid) VALUES (?,?,?) "
                "ON CONFLICT(session_id, code) DO UPDATE SET centroid=excluded.centroid",
                (session_id, code, centroid),
            )
            self._db.commit()

    def voiceprint(self, session_id: int, code: str) -> bytes | None:
        with self._lock:
            row = self._db.execute(
                "SELECT centroid FROM voiceprint WHERE session_id=? AND code=?", (session_id, code)
            ).fetchone()
        return row["centroid"] if row else None

    def remember_speaker(self, name: str, centroid: bytes) -> None:
        """Add one voiceprint variant for a name this room knows.

        A person split across several codes in one meeting is named on each of them, and every
        distinct print must be learned — matching takes the closest, so a variant kept is another
        chance to put the right name on a returning voice, and a variant dropped is that chance
        lost. known_speaker still holds one representative print (newest wins) for the Learned page
        and as a fallback; the variants live in known_voiceprint, deduped and capped so the set is
        a spread of a voice rather than an archive of it.
        """
        with self._lock:
            self._db.execute(
                "INSERT INTO known_speaker (name, centroid) VALUES (?,?) "
                "ON CONFLICT(name) DO UPDATE SET centroid=excluded.centroid, "
                "sessions=known_speaker.sessions + 1",
                (name, centroid),
            )
            vec = np.frombuffer(centroid, dtype=np.float32)
            rows = list(self._db.execute(
                "SELECT rowid, centroid FROM known_voiceprint WHERE name=?", (name,)))
            sims = [(_cosine(vec, np.frombuffer(r["centroid"], dtype=np.float32)), r["rowid"])
                    for r in rows]
            if sims and max(s for s, _ in sims) >= KNOWN_DUP_THRESHOLD:
                self._db.commit()  # already know a print this close — nothing new to learn
                return
            if len(rows) >= KNOWN_PRINT_CAP:
                # Drop the print most like the newcomer: keeps the spread, sheds the redundancy.
                self._db.execute("DELETE FROM known_voiceprint WHERE rowid=?", (max(sims)[1],))
            self._db.execute(
                "INSERT INTO known_voiceprint (name, centroid) VALUES (?,?)", (name, centroid))
            self._db.commit()

    def unlearn_speaker(self, name: str, centroid: bytes) -> None:
        """Drop the stored variant that made this voice answer to `name`.

        Correcting a speaker from B to A carries a second fact besides "this is A": whatever print
        under B pulled this voice in is wrong, and left in place it will misname the same voice
        next meeting. Only the closest variant goes, and only if it is close enough to have caused
        the match — deleting a genuinely-B print because of an unrelated typo fix would erode B.
        The known_speaker fallback centroid is left alone: for a voice learned before variants
        existed it is the only print, and removing it would forget the person outright.
        """
        with self._lock:
            vec = np.frombuffer(centroid, dtype=np.float32)
            rows = list(self._db.execute(
                "SELECT rowid, centroid FROM known_voiceprint WHERE name=?", (name,)))
            sims = [(_cosine(vec, np.frombuffer(r["centroid"], dtype=np.float32)), r["rowid"])
                    for r in rows]
            if sims and max(sims)[0] >= config.KNOWN_SPEAKER_THRESHOLD:
                self._db.execute("DELETE FROM known_voiceprint WHERE rowid=?", (max(sims)[1],))
                self._db.commit()

    def set_speaker_language(self, name: str, language: str) -> None:
        """Force the language a known voice is transcribed in. '' returns them to auto-detect.

        Keyed by name, not by the per-meeting S-code: the code is renumbered every reprocess, so the
        setting has to live with the identity the room recognises, which is the name.
        """
        with self._lock:
            self._db.execute("UPDATE known_speaker SET language=? WHERE name=?", (language, name))
            self._db.commit()

    def speaker_languages(self) -> dict[str, str]:
        """Every known voice's forced language, '' where none is set. Read for the Learned page and,
        filtered to the non-empty ones, to override a recognised speaker's transcription language."""
        with self._lock:
            return {r["name"]: r["language"] for r in self._db.execute(
                "SELECT name, language FROM known_speaker WHERE name != ''")}

    def known_voiceprints(self) -> list[tuple[str, bytes]]:
        """Every stored voiceprint with its current name, for recognising a voice next meeting.

        Unlike known_speakers() — one row per name, for the Learned page — this returns every
        variant, because matching takes the closest and each variant is another way in. Falls back
        to known_speaker's single centroid for any name learned before variants were stored, so an
        upgrade never forgets a voice the room already knew.
        """
        with self._lock:
            return [(r["name"], r["centroid"]) for r in self._db.execute(
                "SELECT name, centroid FROM known_voiceprint WHERE name != '' "
                "UNION "
                "SELECT name, centroid FROM known_speaker WHERE name != '' "
                "AND name NOT IN (SELECT DISTINCT name FROM known_voiceprint)")]

    def known_speakers(self) -> list[tuple[str, bytes]]:
        """Every voice the room can name on sight, with the centroid to recognise it by.

        The names come from speaker_name — the live mapping — not from known_speaker's own name
        column, which remember_speaker only ever inserts into: renaming a speaker updated
        speaker_name in place but left the old name behind in known_speaker, so the same voice
        showed up twice on the Learned page. Joining on the current name drops the orphan; the
        centroid still comes from known_speaker, keyed by whatever name is stored there for it.
        Ordered by how many meetings named the voice, most-confirmed first.
        """
        with self._lock:
            return [(r["name"], r["centroid"]) for r in self._db.execute(
                "SELECT DISTINCT sn.name, ks.centroid "
                "FROM speaker_name sn JOIN known_speaker ks ON ks.name = sn.name "
                "WHERE sn.name != '' "
                "GROUP BY sn.name "
                "ORDER BY COUNT(DISTINCT sn.session_id) DESC")]

    def speaker_sessions(self) -> dict[str, int]:
        """How many distinct meetings each *known* voice was named in, counted from the source.

        Not read off known_speaker.sessions, which remember_speaker increments on every save: naming
        a voice, then fixing a typo in that name, both save — so a within-meeting rename inflated the
        count, and it read as more meetings than the voice was ever in. speaker_name holds one row
        per (session, code), so counting distinct sessions per name is the true figure regardless of
        how many times it was saved.

        Joined to known_speaker so this returns only voices the room still knows: forget_speaker
        removes a voice from known_speaker but leaves its historical transcript names in place (a
        past meeting keeps the name it was given), and without the join this counted those forgotten
        names too. get_known_speakers only ever asks about names it got from known_speakers(), so
        that mismatch was latent — but the two now agree at the source instead of by that accident.
        """
        with self._lock:
            return {r["name"]: r["n"] for r in self._db.execute(
                "SELECT sn.name, COUNT(DISTINCT sn.session_id) AS n FROM speaker_name sn "
                "JOIN known_speaker ks ON ks.name = sn.name "
                "WHERE sn.name != '' GROUP BY sn.name")}

    def speaker_sample(self, name: str) -> tuple[str, float, float | None] | None:
        """Where to hear this voice, and how long that utterance lasts.

        Derived rather than stored — a name is only ever attached on the transcript page, so the
        transcript already knows which recording and which second to play.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT s.wav_path AS wav, l.start AS start, l.end_time AS end_time "
                "FROM speaker_name sn "
                "JOIN line l ON l.session_id=sn.session_id AND l.speaker=sn.code "
                "JOIN session s ON s.id=sn.session_id "
                "WHERE sn.name=? ORDER BY sn.session_id DESC, l.start LIMIT 1",
                (name,),
            ).fetchone()
        return _sample(row)

    def session_speaker_sample(self, session_id: int, code: str) -> tuple[str, float, float | None] | None:
        """Where to hear S3 in this meeting, before anyone has said who S3 is.

        speaker_sample() goes through speaker_name, so it can only find a voice that already has a
        name — which is exactly the voice nobody needs to hear. This one is keyed on the diariser's
        own code, so it works while the field beside it is still empty.

        The longest utterance, not the first: "謝謝" identifies nobody, and picking by length costs
        an ORDER BY. Falls back to text length where the recording has no end time.
        """
        with self._lock:
            row = self._db.execute(
                "SELECT s.wav_path AS wav, l.start AS start, l.end_time AS end_time FROM line l "
                "JOIN session s ON s.id = l.session_id "
                "WHERE l.session_id=? AND l.speaker=? "
                "ORDER BY COALESCE(l.end_time - l.start, 0) DESC, LENGTH(l.source) DESC LIMIT 1",
                (session_id, code),
            ).fetchone()
        return _sample(row)

    def rename_speaker(self, old: str, new: str) -> None:
        """Rename a learned voice everywhere it is used, transcripts included.

        Leaving old transcripts on the wrong name would make the rename look like it half-worked.
        """
        with self._lock:
            if new != old and self._db.execute(
                "SELECT 1 FROM known_speaker WHERE name=?", (new,)
            ).fetchone():
                # This used to DELETE the row already holding `new` to dodge the primary-key clash,
                # which silently destroyed that voice's print and merged two people into one name.
                # Refuse the collision, like edit_correction does — the user removes or renames the
                # existing speaker first.
                raise ValueError(f"a speaker named {new} already exists")
            self._db.execute("UPDATE known_speaker SET name=? WHERE name=?", (new, old))
            self._db.execute("UPDATE known_voiceprint SET name=? WHERE name=?", (new, old))
            self._db.execute("UPDATE speaker_name SET name=? WHERE name=?", (new, old))
            self._db.commit()

    def forget_speaker(self, name: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM known_speaker WHERE name=?", (name,))
            self._db.execute("DELETE FROM known_voiceprint WHERE name=?", (name,))
            self._db.commit()

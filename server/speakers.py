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

# The band a naming clip wants: long enough to identify a voice, short enough that a missed
# speaker turn is unlikely to be inside it. Sampling the *longest* utterance — the previous rule —
# adversely selected the dirtiest line: a line is long exactly when the segmenter missed a turn
# inside it, so the sample opened with somebody else's voice. Under three seconds only as a last
# resort ("謝謝" identifies nobody); otherwise mid-monologue lines — flanked by the same speaker
# on both sides — before lines at a speaker boundary, which is where a missed turn leaves the
# other voice; then the closest to eight seconds; text length breaks ties for transcripts written
# before end_time existed. Measured on a real 2.7h import: the previous rule picked a boundary
# line (S8 before, S3 after) for a speaker who had a same-length line inside his own monologue.
SAMPLE_MIN_SECONDS = 3.0
SAMPLE_IDEAL_SECONDS = 8.0
_SAMPLE_LINES = ("WITH l AS (SELECT line.*, "
                 "LAG(line.speaker) OVER w AS prev_speaker, "
                 "LEAD(line.speaker) OVER w AS next_speaker FROM line "
                 "WINDOW w AS (PARTITION BY line.session_id ORDER BY line.start)) ")
_SAMPLE_ORDER = ("ORDER BY (COALESCE(l.end_time - l.start, 0) < "
                 f"{SAMPLE_MIN_SECONDS}), "
                 "(l.prev_speaker IS NOT l.speaker OR l.next_speaker IS NOT l.speaker), "
                 f"ABS(COALESCE(l.end_time - l.start, 0) - {SAMPLE_IDEAL_SECONDS}), "
                 "LENGTH(l.source) DESC")
# Which edges of the picked line touch a speaker handover. Preferring mid-monologue lines is not
# enough: a fragment voice has nothing *but* boundary lines to offer, and the other voice sits in
# the edge that touches the handover. _sample shaves that edge off.
_SAMPLE_RISK = (", (l.prev_speaker IS NOT NULL AND l.prev_speaker <> l.speaker) AS head_risk"
                ", (l.next_speaker IS NOT NULL AND l.next_speaker <> l.speaker) AS tail_risk ")
# Capped at a quarter of the span per edge so a one-second fragment still plays its middle.
SAMPLE_EDGE_TRIM = 0.6


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _sample(row: sqlite3.Row | None) -> tuple[str, float, float | None] | None:
    """A sample row as (recording, start, seconds), with seconds unknown on transcripts written
    before end_time was recorded."""
    if row is None:
        return None
    start, end = float(row["start"]), row["end_time"]
    span = float(end) - start if end is not None else None
    if span and span > 0 and "head_risk" in row.keys():
        if row["head_risk"]:
            cut = min(SAMPLE_EDGE_TRIM, span * 0.25)
            start += cut
            span -= cut
        if row["tail_risk"]:
            span -= min(SAMPLE_EDGE_TRIM, span * 0.25)
    return (row["wav"], start, span if span and span > 0 else None)


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

    def delete_voiceprint(self, session_id: int, code: str) -> None:
        """Drop a code's stored print — for a code that no longer labels any line after a merge or
        reassign, whose print would otherwise keep describing a voice the meeting no longer has."""
        with self._lock:
            self._db.execute("DELETE FROM voiceprint WHERE session_id=? AND code=?",
                             (session_id, code))
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

    def set_speaker_department(self, name: str, department: str) -> None:
        with self._lock:
            self._db.execute("UPDATE known_speaker SET department=? WHERE name=?", (department, name))
            self._db.commit()

    def speaker_departments(self) -> dict[str, str]:
        """Every known voice's department, '' where none is set — for the Learned page and the
        summary prompt, where a name plus a department is what lets stance be read, not guessed."""
        with self._lock:
            return {r["name"]: r["department"] for r in self._db.execute(
                "SELECT name, department FROM known_speaker WHERE name != ''")}

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

        A name qualifies if a live meeting still references it OR a harvested clip is kept for it:
        the clip is what lets an identified voice outlive the deletion of every meeting it spoke
        in — before this, deleting the last meeting silently dropped the voice from this page while
        the recogniser kept using its prints. The reference requirement (rather than reading
        known_speaker unfiltered) still hides orphans left behind by the era when renames updated
        speaker_name but not known_speaker. Ordered by how many meetings named the voice.
        """
        with self._lock:
            return [(r["name"], r["centroid"]) for r in self._db.execute(
                "SELECT ks.name, ks.centroid FROM known_speaker ks "
                "WHERE ks.name != '' AND ("
                "  EXISTS(SELECT 1 FROM speaker_name sn WHERE sn.name = ks.name)"
                "  OR EXISTS(SELECT 1 FROM known_speaker_clip c WHERE c.name = ks.name)) "
                "ORDER BY (SELECT COUNT(DISTINCT sn.session_id) FROM speaker_name sn "
                "          WHERE sn.name = ks.name) DESC")]

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

    def speaker_clip_sources(self, name: str) -> list[tuple[int, bool]]:
        """Every meeting this voice can be heard from, newest first, as (session_id, stored).

        Two sources: meetings still on disk (cut from the wav on demand) and clips harvested when
        a meeting was deleted (stored, in known_speaker_clip). A meeting present in both plays the
        live cut — the harvest is only there for when the wav is gone.
        """
        with self._lock:
            live = [r["session_id"] for r in self._db.execute(
                "SELECT DISTINCT sn.session_id AS session_id FROM speaker_name sn "
                "JOIN line l ON l.session_id=sn.session_id AND l.speaker=sn.code "
                "WHERE sn.name=?", (name,))]
            stored = [r["session_id"] for r in self._db.execute(
                "SELECT session_id FROM known_speaker_clip WHERE name=?", (name,))]
        merged = {sid: False for sid in live}
        merged.update({sid: True for sid in stored if sid not in merged})
        return sorted(merged.items(), key=lambda kv: kv[0], reverse=True)

    def session_codes_for(self, name: str, session_id: int) -> list[str]:
        """Every code this meeting gave to `name` — the handles a per-sample undo has to work on."""
        with self._lock:
            return [r["code"] for r in self._db.execute(
                "SELECT code FROM speaker_name WHERE session_id=? AND name=?", (session_id, name))]

    def unname_speaker(self, session_id: int, name: str) -> None:
        """Withdraw one meeting's naming of `name`, leaving its other meetings alone."""
        with self._lock:
            self._db.execute("DELETE FROM speaker_name WHERE session_id=? AND name=?",
                             (session_id, name))
            self._db.commit()

    def delete_speaker_clip(self, name: str, session_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM known_speaker_clip WHERE name=? AND session_id=?",
                             (name, session_id))
            self._db.commit()

    def move_speaker_clip(self, name: str, session_id: int, new: str) -> None:
        """Hand a harvested clip to another name. OR REPLACE: if the target already kept a clip
        from the same meeting, one of the two is enough."""
        with self._lock:
            self._db.execute(
                "UPDATE OR REPLACE known_speaker_clip SET name=? WHERE name=? AND session_id=?",
                (new, name, session_id))
            self._db.commit()

    def speaker_sample(self, name: str, session_id: int) -> tuple[str, float, float | None] | None:
        """Where to hear this voice in this meeting, and how long that utterance lasts.

        Derived rather than stored — a name is only ever attached on the transcript page, so the
        transcript already knows which recording and which second to play. Picked by
        `_SAMPLE_ORDER`, not by length: see the note on it.
        """
        with self._lock:
            row = self._db.execute(
                f"{_SAMPLE_LINES}"
                "SELECT s.wav_path AS wav, l.start AS start, l.end_time AS end_time"
                f"{_SAMPLE_RISK}"
                "FROM speaker_name sn "
                "JOIN l ON l.session_id=sn.session_id AND l.speaker=sn.code "
                "JOIN session s ON s.id=sn.session_id "
                f"WHERE sn.name=? AND sn.session_id=? {_SAMPLE_ORDER} LIMIT 1",
                (name, session_id),
            ).fetchone()
        return _sample(row)

    def speaker_sample_info(self, name: str, session_id: int) -> tuple[str, str] | None:
        """When the meeting behind a sample ran and what its picked line says.

        Same `_SAMPLE_ORDER` pick as speaker_sample(), so the text shown is the text of the line
        that actually plays.
        """
        with self._lock:
            row = self._db.execute(
                f"{_SAMPLE_LINES}"
                "SELECT s.started AS started, l.source AS source "
                "FROM speaker_name sn "
                "JOIN l ON l.session_id=sn.session_id AND l.speaker=sn.code "
                "JOIN session s ON s.id=sn.session_id "
                f"WHERE sn.name=? AND sn.session_id=? {_SAMPLE_ORDER} LIMIT 1",
                (name, session_id),
            ).fetchone()
        return (row["started"], row["source"]) if row else None

    def save_speaker_clip(self, name: str, session_id: int, audio: bytes) -> None:
        """Keep this voice's sound beyond the meeting it came from. Capped like voiceprints: the
        oldest clips fall off so a long-lived name does not accumulate audio without bound."""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO known_speaker_clip (name, session_id, audio) VALUES (?,?,?)",
                (name, session_id, audio))
            self._db.execute(
                "DELETE FROM known_speaker_clip WHERE name=? AND session_id NOT IN "
                "(SELECT session_id FROM known_speaker_clip WHERE name=? "
                "ORDER BY session_id DESC LIMIT ?)",
                (name, name, KNOWN_PRINT_CAP))
            self._db.commit()

    def stored_clip(self, name: str, session_id: int) -> bytes | None:
        with self._lock:
            row = self._db.execute(
                "SELECT audio FROM known_speaker_clip WHERE name=? AND session_id=?",
                (name, session_id)).fetchone()
        return row["audio"] if row else None

    def session_speaker_sample(self, session_id: int, code: str) -> tuple[str, float, float | None] | None:
        """Where to hear S3 in this meeting, before anyone has said who S3 is.

        speaker_sample() goes through speaker_name, so it can only find a voice that already has a
        name — which is exactly the voice nobody needs to hear. This one is keyed on the diariser's
        own code, so it works while the field beside it is still empty.

        Picked by `_SAMPLE_ORDER`, not by length: see the note on it.
        """
        with self._lock:
            row = self._db.execute(
                f"{_SAMPLE_LINES}"
                "SELECT s.wav_path AS wav, l.start AS start, l.end_time AS end_time"
                f"{_SAMPLE_RISK}FROM l "
                "JOIN session s ON s.id = l.session_id "
                f"WHERE l.session_id=? AND l.speaker=? {_SAMPLE_ORDER} LIMIT 1",
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
            self._db.execute("UPDATE known_speaker_clip SET name=? WHERE name=?", (new, old))
            self._db.commit()

    def forget_speaker(self, name: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM known_speaker WHERE name=?", (name,))
            self._db.execute("DELETE FROM known_voiceprint WHERE name=?", (name,))
            self._db.execute("DELETE FROM known_speaker_clip WHERE name=?", (name,))
            self._db.commit()

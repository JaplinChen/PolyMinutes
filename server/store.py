"""SQLite persistence: glossary, sessions, transcript lines and corrections.

One connection with `check_same_thread=False` because the capture pipeline writes from a worker
thread while FastAPI reads from the event loop; a lock serialises them. WAL keeps a long-running
write from blocking the subtitle page's reads.

The tables themselves live in `schema`, and the speaker half of the API in `speakers` — both share
this connection and this lock, so `Store` stays the only thing that talks to the database.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from . import config, schema
from .speakers import SpeakerStore

DB_PATH = config.ROOT / "polyminutes.db"


def _migrate_legacy_db() -> None:
    """Carry a pre-rename meettranslate.db over to the new name, WAL sidecars included.

    Renaming only the main file would strand an unreplayed WAL and silently lose the tail of
    the last session, so the three files move together or not at all.
    """
    if DB_PATH.exists():
        return
    legacy = config.ROOT / "meettranslate.db"
    if not legacy.exists():
        return
    for suffix in ("", "-wal", "-shm"):
        old = legacy.with_name(legacy.name + suffix)
        if old.exists():
            old.rename(DB_PATH.with_name(DB_PATH.name + suffix))


_migrate_legacy_db()

# How a glossary term is applied. `keep` exists because code-switched English terms
# ("schedule", "delay") are shared vocabulary in cross-border teams — translating them into
# Vietnamese makes the subtitle harder to read, not easier.
#
# `protect` is the opposite of the others: it declares a word real so the corrector leaves it
# alone, and never rewrites anything into it. Needed because 才夠 and 採購 are homophones and both
# are ordinary speech — registering 才夠 to shield it made it a target instead, and 採購 was
# rewritten to 才夠 211 times across seven interviews.
TERM_MODES = ("translate", "keep", "hint", "protect")


def _portable(wav_path: str) -> str:
    """Store a recording relative to the repo root when it lives under it.

    Read back through `config.recording_path`. A path outside the root has nothing to be relative
    to, so it is stored as given.
    """
    try:
        return Path(wav_path).resolve().relative_to(config.ROOT).as_posix()
    except ValueError:
        return wav_path


@dataclass
class Term:
    id: int
    source: str
    lang: str
    mode: str
    category: str
    targets: dict[str, str]


class Store(SpeakerStore):
    def __init__(self, path: Path | None = None):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path or DB_PATH), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            schema.apply(self._db)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ── glossary ────────────────────────────────────────────────────────

    def add_term(self, source: str, targets: dict[str, str], lang: str = "", mode: str = "translate",
                 category: str = "") -> Term:
        if mode not in TERM_MODES:
            raise ValueError(f"mode must be one of {TERM_MODES}")
        source = source.strip()
        if not source:
            raise ValueError("source must not be empty")

        with self._lock:
            cur = self._db.execute(
                "INSERT INTO glossary (source, lang, mode, category) VALUES (?,?,?,?) "
                "ON CONFLICT(source, lang) DO UPDATE SET mode=excluded.mode, category=excluded.category "
                "RETURNING id",
                (source, lang, mode, category),
            )
            term_id = cur.fetchone()[0]
            self._db.execute("DELETE FROM glossary_target WHERE term_id=?", (term_id,))
            self._db.executemany(
                "INSERT INTO glossary_target (term_id, lang, text) VALUES (?,?,?)",
                [(term_id, k, v) for k, v in targets.items() if v.strip()],
            )
            self._db.commit()
        return Term(term_id, source, lang, mode, category, dict(targets))

    def remove_term(self, source: str, lang: str = "") -> None:
        with self._lock:
            self._db.execute("DELETE FROM glossary WHERE source=? AND lang=?", (source, lang))
            self._db.commit()

    def glossary(self) -> list[Term]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM glossary ORDER BY source").fetchall()
            targets: dict[int, dict[str, str]] = {}
            for t in self._db.execute("SELECT * FROM glossary_target"):
                targets.setdefault(t["term_id"], {})[t["lang"]] = t["text"]
        return [Term(r["id"], r["source"], r["lang"], r["mode"], r["category"], targets.get(r["id"], {})) for r in rows]

    # ── sessions and lines ──────────────────────────────────────────────

    def start_session(self, started: str, wav_path: str) -> int:
        with self._lock:
            cur = self._db.execute("INSERT INTO session (started, wav_path) VALUES (?,?)",
                                   (started, _portable(wav_path)))
            self._db.commit()
            return int(cur.lastrowid)

    def end_session(self, session_id: int, ended: str) -> None:
        with self._lock:
            self._db.execute("UPDATE session SET ended=? WHERE id=?", (ended, session_id))
            self._db.commit()

    def add_line(self, session_id: int, start: float, speaker: str, lang: str, source: str,
                 translations: dict[str, str], status: str = "ok",
                 end_time: float | None = None) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO line (session_id, start, speaker, lang, source, status, end_time) "
                "VALUES (?,?,?,?,?,?,?)",
                (session_id, start, speaker, lang, source, status, end_time),
            )
            line_id = int(cur.lastrowid)
            self._db.executemany(
                "INSERT INTO line_translation (line_id, lang, text) VALUES (?,?,?)",
                [(line_id, k, v) for k, v in translations.items()],
            )
            self._db.commit()
        return line_id

    def update_line(self, line_id: int, source: str | None, translations: dict[str, str]) -> None:
        """Apply a refinement. Marks the line refined so it is never rewritten twice."""
        with self._lock:
            if source is not None:
                self._db.execute("UPDATE line SET source=? WHERE id=?", (source, line_id))
            self._db.executemany(
                "INSERT INTO line_translation (line_id, lang, text) VALUES (?,?,?) "
                "ON CONFLICT(line_id, lang) DO UPDATE SET text=excluded.text",
                [(line_id, k, v) for k, v in translations.items()],
            )
            self._db.execute("UPDATE line SET refined=1 WHERE id=?", (line_id,))
            self._bump_rev_for_line(line_id)
            self._db.commit()

    def set_line_speaker(self, line_id: int, speaker: str) -> None:
        """Reassign one line to a different speaker.

        The room's shared mic collapses everyone into one voice often enough that the automatic
        clustering cannot be trusted; this is the human putting the attribution back by hand. It
        only relabels — the voiceprint is not recomputed, because the centroid the clustering
        derived from a collapsed speaker is exactly what was wrong. Naming the speaker afterwards
        is what attaches a fresh voiceprint.
        """
        with self._lock:
            self._db.execute("UPDATE line SET speaker=? WHERE id=?", (speaker, line_id))
            self._bump_rev_for_line(line_id)
            self._db.commit()

    def merge_speakers(self, session_id: int, into: str, sources: list[str]) -> None:
        """Collapse several diariser codes for one person into one speaker.

        The clustering splits a drifting voice into S17/S18/S20; this is the human saying they are
        the same person. Every line moves to `into`, and the absorbed codes' name and per-session
        voiceprint go with them — the codes then vanish from the meeting, because the speaker list
        is derived from which codes still label a line. What the room has already learned is not
        touched: naming those codes taught it (see remember_speaker), and that lives on the known
        tables, not here. Merge only tidies this transcript.
        """
        sources = [c for c in sources if c and c != into]
        if not sources:
            return
        marks = ",".join("?" * len(sources))
        with self._lock:
            self._db.execute(
                f"UPDATE line SET speaker=? WHERE session_id=? AND speaker IN ({marks})",
                (into, session_id, *sources))
            self._db.execute(
                f"DELETE FROM speaker_name WHERE session_id=? AND code IN ({marks})",
                (session_id, *sources))
            self._db.execute(
                f"DELETE FROM voiceprint WHERE session_id=? AND code IN ({marks})",
                (session_id, *sources))
            self._db.execute(
                "UPDATE session SET lines_rev = lines_rev + 1 WHERE id=?", (session_id,))
            self._db.commit()

    def line(self, line_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM line WHERE id=?", (line_id,)).fetchone()
        return dict(row) if row else None

    def replace_line(self, line_id: int, source: str, lang: str, translations: dict[str, str],
                     status: str, refined: bool = False) -> None:
        """Overwrite one line after re-running it. Leaves `refined` alone by default.

        `refined` records that the translator revised this line in hindsight, which is a different
        claim from "someone re-ran it", and conflating the two would let a rerun suppress the one
        refinement pass the line is still entitled to. A human edit is the exception: it passes
        refined=True, because the post-meeting pass rewriting what someone typed by hand is the
        one thing that flag exists to prevent.
        """
        with self._lock:
            try:
                self._db.execute("UPDATE line SET source=?, lang=?, status=? WHERE id=?",
                                 (source, lang, status, line_id))
                if refined:
                    self._db.execute("UPDATE line SET refined=1 WHERE id=?", (line_id,))
                self._db.execute("DELETE FROM line_translation WHERE line_id=?", (line_id,))
                self._db.executemany(
                    "INSERT INTO line_translation (line_id, lang, text) VALUES (?,?,?)",
                    [(line_id, k, v) for k, v in translations.items()],
                )
                self._bump_rev_for_line(line_id)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise

    def _bump_rev_for_line(self, line_id: int) -> None:
        """Record that this line's session no longer says what it said. Caller holds the lock.

        The revision is what lets anything derived from the transcript — the summary — admit it
        is describing an older version. Bumped inside the same transaction as the edit, so a
        rollback takes the bump with it.
        """
        self._db.execute(
            "UPDATE session SET lines_rev = lines_rev + 1 "
            "WHERE id = (SELECT session_id FROM line WHERE id=?)", (line_id,))

    def replace_lines(self, session_id: int, rows: list[dict]) -> None:
        """Swap a session's whole transcript in one transaction.

        The obvious shape — delete the old lines, then insert each new one — commits after every
        line, so a run that dies in the middle leaves the transcript half-replaced, and a run that
        dies right after the delete leaves it empty. That is data loss, not a slow path: the
        recording is still on disk but the meeting's transcript is gone until someone notices.

        Every insert here shares one implicit transaction and one commit, so the transcript is
        either entirely the old one or entirely the new one. The caller must have finished
        translating before calling: holding a write lock across an LLM round trip would block the
        next meeting's first line on `database is locked`.
        """
        with self._lock:
            try:
                self._db.execute("DELETE FROM line WHERE session_id=?", (session_id,))
                for row in rows:
                    cur = self._db.execute(
                        "INSERT INTO line (session_id, start, speaker, lang, source, status, end_time) "
                        "VALUES (?,?,?,?,?,?,?)",
                        (session_id, row["start"], row["speaker"], row["lang"], row["source"],
                         row.get("status", "ok"), row.get("end_time")),
                    )
                    self._db.executemany(
                        "INSERT INTO line_translation (line_id, lang, text) VALUES (?,?,?)",
                        [(int(cur.lastrowid), k, v) for k, v in row.get("translations", {}).items()],
                    )
                self._db.execute("UPDATE session SET lines_rev = lines_rev + 1 WHERE id=?",
                                 (session_id,))
                self._db.commit()
            except Exception:
                # Without this the half-finished transaction stays open on a connection every other
                # method shares, so the next commit anywhere in the Store would commit this delete.
                # The failure would surface later, somewhere else, as a transcript that vanished.
                self._db.rollback()
                raise

    def session(self, session_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM session WHERE id=?", (session_id,)).fetchone()
        return dict(row) if row else None

    # ── cross-meeting search ────────────────────────────────────────────

    def search_lines(self, keywords: list[str], since: str = "", until: str = "",
                     limit: int = 400) -> list[dict]:
        """Lines from any meeting containing any of these keywords, newest sessions first.

        A plain LIKE scan, deliberately. A year of meetings is about 190,000 lines and 2.5 MB, and
        scanning all of it takes 14 ms — an index would be machinery to keep in step with every
        correction for no measurable gain. FTS5 was the other candidate and is worse here: its
        trigram tokenizer needs three characters, so it returns nothing at all for 交期, 產能 or
        交貨, which is most of what anyone searches for.

        One query rather than one per keyword: each is a full scan taking this store's only lock,
        and the capture thread needs that lock for every utterance of a meeting in progress.
        """
        terms = [k.strip() for k in keywords if k and len(k.strip()) >= 2]
        if not terms:
            return []
        # % and _ are LIKE wildcards, and a keyword may contain either — "50%" would otherwise match
        # any line with "50", "Q_3" any line with "Q" then any character. Escaped with a backslash,
        # declared by ESCAPE, so a keyword matches itself literally. The backslash itself is escaped
        # first, or a keyword ending in one would escape the closing %.
        def _like(term: str) -> str:
            escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            return f"%{escaped}%"
        # Parenthesised before any date AND: SQL binds AND tighter than OR, so an unwrapped
        # `A OR B AND date` means `A OR (B AND date)` and the cutoff would only constrain the last
        # keyword. Wrapping once keeps `(A OR B) AND date` whichever bounds are present.
        where = "(" + " OR ".join(["l.source LIKE ? ESCAPE '\\'"] * len(terms)) + ")"
        params: list = [_like(t) for t in terms]
        if since:
            where, params = f"{where} AND s.started >= ?", params + [since]
        if until:
            where, params = f"{where} AND s.started <= ?", params + [until + "T23:59:59"]
        with self._lock:
            rows = self._db.execute(
                "SELECT l.id, l.session_id, l.start, l.speaker, l.source "
                "FROM line l JOIN session s ON s.id = l.session_id "
                f"WHERE {where} ORDER BY l.session_id DESC, l.start LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def summaries(self) -> dict[int, dict]:
        """Every stored summary, keyed by session. The retrieval index for cross-meeting questions."""
        with self._lock:
            rows = self._db.execute("SELECT * FROM summary").fetchall()
        return {r["session_id"]: dict(r) for r in rows}

    # ── summary ─────────────────────────────────────────────────────────

    def set_summary(self, session_id: int, json_text: str, status: str, lines_rev: int,
                    created: str) -> None:
        """One summary per session, latest wins.

        `lines_rev` is the session's revision at generation time. A later read that finds the
        session's current revision has moved on knows the summary describes a transcript that no
        longer exists — stale is a comparison, not a stored flag, so nothing has to remember to
        set it.
        """
        with self._lock:
            self._db.execute(
                "INSERT INTO summary (session_id, json, status, lines_rev, created) "
                "VALUES (?,?,?,?,?) ON CONFLICT(session_id) DO UPDATE SET "
                "json=excluded.json, status=excluded.status, lines_rev=excluded.lines_rev, "
                "created=excluded.created",
                (session_id, json_text, status, lines_rev, created))
            self._db.commit()

    def summary(self, session_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM summary WHERE session_id=?",
                                   (session_id,)).fetchone()
        return dict(row) if row else None

    def summary_and_rev(self, session_id: int) -> tuple[dict | None, int]:
        """The stored summary and the session's current revision, from one lock hold.

        The regenerate endpoint compares the two to decide whether the transcript has changed since
        the summary was made. Read apart — session() then summary() — an edit landing between them
        moves the revision under the comparison, so it can conclude "unchanged" against a version
        that already differs and refuse a regeneration that should have run. One acquisition makes
        the pair consistent, the same reason lines_with_rev exists.
        """
        with self._lock:
            summ = self._db.execute("SELECT * FROM summary WHERE session_id=?",
                                    (session_id,)).fetchone()
            sess = self._db.execute("SELECT lines_rev FROM session WHERE id=?",
                                    (session_id,)).fetchone()
        rev = int(sess["lines_rev"]) if sess else 0
        return (dict(summ) if summ else None), rev

    def lines_with_rev(self, session_id: int) -> tuple[list[dict], int]:
        """The lines and the session's revision, read under one lock hold.

        The summary pass reads both to decide what to describe and to stamp the result with the
        revision it observed. Read separately, an edit landing between the two — replace_line takes
        the lock, changes a line and bumps lines_rev — let the summary be generated from the old
        text and stamped with the new revision, so it never showed as stale though it was. One
        acquisition makes the pair consistent.
        """
        with self._lock:
            rev_row = self._db.execute(
                "SELECT lines_rev FROM session WHERE id=?", (session_id,)
            ).fetchone()
            rev = int(rev_row["lines_rev"]) if rev_row else 0
            rows = self._db.execute(
                "SELECT * FROM line WHERE session_id=? ORDER BY start", (session_id,)
            ).fetchall()
            tr: dict[int, dict[str, str]] = {}
            for t in self._db.execute(
                "SELECT lt.* FROM line_translation lt JOIN line l ON l.id=lt.line_id WHERE l.session_id=?",
                (session_id,),
            ):
                tr.setdefault(t["line_id"], {})[t["lang"]] = t["text"]
        return [{**dict(r), "translations": tr.get(r["id"], {})} for r in rows], rev

    def lines(self, session_id: int) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM line WHERE session_id=? ORDER BY start", (session_id,)
            ).fetchall()
            tr: dict[int, dict[str, str]] = {}
            for t in self._db.execute(
                "SELECT lt.* FROM line_translation lt JOIN line l ON l.id=lt.line_id WHERE l.session_id=?",
                (session_id,),
            ):
                tr.setdefault(t["line_id"], {})[t["lang"]] = t["text"]
        return [{**dict(r), "translations": tr.get(r["id"], {})} for r in rows]

    def sessions(self) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._db.execute(
                "SELECT s.*, (SELECT COUNT(*) FROM line WHERE session_id=s.id) AS lines "
                "FROM session s ORDER BY s.id DESC"
            )]

    def transcript_text(self, limit: int = 20000) -> str:
        """Every line this room has recorded, as one string.

        The corpus for deciding whether a glossary term is safe is the meeting history itself —
        what these people actually say, rather than a dictionary of what Mandarin permits.
        """
        with self._lock:
            rows = self._db.execute(
                "SELECT source FROM line ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return chr(10).join(r["source"] for r in rows)

    # ── corrections ─────────────────────────────────────────────────────
    #
    # An edit made on the transcript page is ground truth: this is what the recogniser wrote and
    # this is what was actually said. Nothing else in the system is labelled by a human who was in
    # the room, so it outranks every heuristic that guesses from pinyin.

    def add_correction(self, wrong: str, right: str, lang: str = "") -> None:
        wrong, right = wrong.strip(), right.strip()
        if not wrong or not right or wrong == right:
            return
        with self._lock:
            self._db.execute(
                "INSERT INTO correction (wrong, right, lang) VALUES (?,?,?) "
                "ON CONFLICT(wrong) DO UPDATE SET right=excluded.right, count=correction.count + 1",
                (wrong, right, lang),
            )
            self._db.commit()

    def corrections(self) -> dict[str, str]:
        with self._lock:
            return {r["wrong"]: r["right"] for r in
                    self._db.execute("SELECT wrong, right FROM correction ORDER BY LENGTH(wrong) DESC")}

    def edit_correction(self, old_wrong: str, wrong: str, right: str) -> None:
        """Fix a learned pair in place, either side of it.

        Until now the only repair was to delete and re-learn, which means finding the line it came
        from and correcting it again — and a pair learned from a typo is exactly the one you cannot
        reproduce on demand. `wrong` is the key, so changing it is a rename rather than an update.
        """
        wrong, right = wrong.strip(), right.strip()
        if not wrong or not right:
            raise ValueError("both sides of a correction must be filled in")
        if wrong == right:
            raise ValueError("a correction that rewrites text to itself would never stop matching")
        with self._lock:
            if wrong != old_wrong and self._db.execute(
                "SELECT 1 FROM correction WHERE wrong=?", (wrong,)
            ).fetchone():
                # Overwriting would silently discard whichever pair the user was not looking at.
                raise ValueError(f"there is already a correction for {wrong}")
            changed = self._db.execute(
                "UPDATE correction SET wrong=?, right=? WHERE wrong=?", (wrong, right, old_wrong)
            ).rowcount
            if not changed:
                raise KeyError(old_wrong)
            self._db.commit()

    def forget_correction(self, wrong: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM correction WHERE wrong=?", (wrong,))
            self._db.commit()

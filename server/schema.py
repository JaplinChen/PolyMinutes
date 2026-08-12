"""The SQLite schema, and the migrations that bring an older database up to it.

Kept apart from `store` because it is declarative: DDL and a list of columns added after the fact,
with no query logic and nothing to decide at runtime.
"""

from __future__ import annotations

import sqlite3
from pathlib import PurePath

SCHEMA = """
CREATE TABLE IF NOT EXISTS glossary (
    id        INTEGER PRIMARY KEY,
    source    TEXT NOT NULL,
    lang      TEXT NOT NULL DEFAULT '',
    mode      TEXT NOT NULL DEFAULT 'translate',
    category  TEXT NOT NULL DEFAULT '',
    UNIQUE(source, lang)
);
CREATE TABLE IF NOT EXISTS glossary_target (
    term_id   INTEGER NOT NULL REFERENCES glossary(id) ON DELETE CASCADE,
    lang      TEXT NOT NULL,
    text      TEXT NOT NULL,
    PRIMARY KEY (term_id, lang)
);
CREATE TABLE IF NOT EXISTS session (
    id        INTEGER PRIMARY KEY,
    started   TEXT NOT NULL,
    ended     TEXT,
    wav_path  TEXT NOT NULL,
    lines_rev INTEGER NOT NULL DEFAULT 0,
    reference TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS line (
    id         INTEGER PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    start      REAL NOT NULL,
    speaker    TEXT NOT NULL,
    lang       TEXT NOT NULL,
    source     TEXT NOT NULL,
    orig_source TEXT,
    refined    INTEGER NOT NULL DEFAULT 0,
    status     TEXT NOT NULL DEFAULT 'ok',
    end_time   REAL
);
CREATE TABLE IF NOT EXISTS line_translation (
    line_id   INTEGER NOT NULL REFERENCES line(id) ON DELETE CASCADE,
    lang      TEXT NOT NULL,
    text      TEXT NOT NULL,
    PRIMARY KEY (line_id, lang)
);
CREATE TABLE IF NOT EXISTS correction (
    wrong  TEXT PRIMARY KEY,
    right  TEXT NOT NULL,
    lang   TEXT NOT NULL DEFAULT '',
    count  INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS voiceprint (
    session_id INTEGER REFERENCES session(id) ON DELETE CASCADE,
    code       TEXT NOT NULL,
    centroid   BLOB NOT NULL,
    PRIMARY KEY (session_id, code)
);
CREATE TABLE IF NOT EXISTS known_speaker (
    name     TEXT PRIMARY KEY,
    centroid BLOB NOT NULL,
    sessions INTEGER NOT NULL DEFAULT 1,
    language TEXT NOT NULL DEFAULT '',
    department TEXT NOT NULL DEFAULT ''
);
-- A voice sample that outlives its meeting. Clips are normally cut from the session's wav on
-- demand; deleting the meeting deletes that wav, and an identified voice must not lose its sound
-- with it. Harvested at meeting deletion, removed only when the voice itself is forgotten — so no
-- FK to session, deliberately.
CREATE TABLE IF NOT EXISTS known_speaker_clip (
    name       TEXT NOT NULL,
    session_id INTEGER NOT NULL,
    audio      BLOB NOT NULL,
    PRIMARY KEY (name, session_id)
);
-- One person sounds different across the room, the mic and their mood, and a single meeting can
-- split them across several codes. Naming all of them stores every variant here, not just the
-- last: matching takes the closest stored print, so more true variants means more of that voice's
-- future utterances land on their name. known_speaker keeps one representative print per name (the
-- Learned page, and a fallback for voices learned before this table existed).
CREATE TABLE IF NOT EXISTS known_voiceprint (
    name     TEXT NOT NULL,
    centroid BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS known_voiceprint_name ON known_voiceprint(name);
CREATE TABLE IF NOT EXISTS speaker_name (
    session_id INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    code       TEXT NOT NULL,
    name       TEXT NOT NULL,
    PRIMARY KEY (session_id, code)
);
CREATE TABLE IF NOT EXISTS summary (
    session_id INTEGER PRIMARY KEY REFERENCES session(id) ON DELETE CASCADE,
    json       TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'ok',
    lines_rev  INTEGER NOT NULL DEFAULT 0,
    created    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS line_session ON line(session_id, start);
"""

# `CREATE TABLE IF NOT EXISTS` is a no-op against a table that already exists, so a database
# created before a column was added never gets it. New machines and CI pass either way; the meeting
# room's database is the one that breaks, and it breaks inside the capture thread where
# Pipeline._handle swallows it as one more error count. Each entry is (column, DDL), applied only
# when the column is absent.
LINE_COLUMNS = (
    ("status", "ALTER TABLE line ADD COLUMN status TEXT NOT NULL DEFAULT 'ok'"),
    # No NOT NULL: rows written before this column existed have no end to backfill, and guessing
    # one would be worse than admitting it is unknown.
    ("end_time", "ALTER TABLE line ADD COLUMN end_time REAL"),
    # What the line said before a human first corrected it, kept so the transcript can show the
    # edit as a strike-through/highlight diff. NULL means never hand-edited.
    ("orig_source", "ALTER TABLE line ADD COLUMN orig_source TEXT"),
)


# Bumped by every write that changes what a line says, so anything derived from the transcript —
# today the summary — can tell whether it still describes what is stored. A max-id-plus-count check
# would miss the most common change of all: an in-place correction, which alters no id and no count.
SESSION_COLUMNS = (
    ("lines_rev", "ALTER TABLE session ADD COLUMN lines_rev INTEGER NOT NULL DEFAULT 0"),
    # Pre-meeting notes — agenda, attendees, slides — pasted by the user. Fed to the summary prompt
    # so it knows the meeting's own terms and priorities; '' for every session recorded before it.
    ("reference", "ALTER TABLE session ADD COLUMN reference TEXT NOT NULL DEFAULT ''"),
)


# The language a known voice speaks, forced on their utterances so a recognised speaker is never
# mis-detected — the room's Vietnamese speaker stops being decoded as Chinese. '' means auto-detect,
# which is what every voice learned before this column existed keeps.
KNOWN_SPEAKER_COLUMNS = (
    ("language", "ALTER TABLE known_speaker ADD COLUMN language TEXT NOT NULL DEFAULT ''"),
    # Which side of the table a voice sits on. Fed to the summary prompt so stance and follow-ups
    # are read against the speaker's role, not guessed from tone.
    ("department", "ALTER TABLE known_speaker ADD COLUMN department TEXT NOT NULL DEFAULT ''"),
)


def _relativise_recordings(db: sqlite3.Connection) -> None:
    """Rewrite absolute session.wav_path rows as paths relative to the repo root.

    Renaming the project directory left every stored recording pointing at a folder that no longer
    exists, and nothing reads a session without reading its audio. Only paths that carry a
    `recordings` component are touched — anything else was put there deliberately and is left alone.
    """
    updates = []
    for row in db.execute("SELECT id, wav_path FROM session"):
        parts = PurePath(row["wav_path"]).parts
        if not PurePath(row["wav_path"]).is_absolute() or "recordings" not in parts:
            continue
        tail = parts[parts.index("recordings"):]
        updates.append(("/".join(tail), row["id"]))
    if updates:
        db.executemany("UPDATE session SET wav_path=? WHERE id=?", updates)
        db.commit()


def apply(db: sqlite3.Connection) -> None:
    """Create what is missing, then add the columns the schema gained since.

    Deliberately not caught: starting with a stale schema is worse than not starting. The
    alternative is a room that records a whole meeting into a table that rejects every insert.
    """
    db.executescript(SCHEMA)
    db.commit()
    added = []
    for table, columns in (("line", LINE_COLUMNS), ("session", SESSION_COLUMNS),
                           ("known_speaker", KNOWN_SPEAKER_COLUMNS)):
        have = {r["name"] for r in db.execute(f"PRAGMA table_info({table})")}
        added += [ddl for column, ddl in columns if column not in have]
    for ddl in added:
        db.execute(ddl)
    if added:
        db.commit()
    _relativise_recordings(db)

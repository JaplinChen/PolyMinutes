"""Diagnose one session's transcript quality (read-only).

    python -m scripts.diag_session SESSION_ID [SESSION_ID ...] [--db polyminutes.db]
    python -m scripts.diag_session 2 3 4 --fix [--api http://127.0.0.1:8010]

Status counts, hallucination and repeat-loop survivors, how confidence is spread overall and per
speaker, and whether the summary is current and cites the transcript — the numbers to look at
before deciding whether a meeting needs a rerun.

--fix retranslates every translate_failed line through the running server's API (never the DB
directly): those failures are sporadic, and every one retried on 2026-10-08 came back on the first
ask. Everything else stays read-only.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import urllib.request
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import asr, store  # noqa: E402

# Same threshold the transcript page fades lines at: dashboard/src/components/sessions/TranscriptRow.tsx
LOW_CONFIDENCE = -0.9


def percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_ids", type=int, nargs="+")
    ap.add_argument("--db", type=Path, default=store.DB_PATH)
    ap.add_argument("--fix", action="store_true", help="retranslate translate_failed lines via the API")
    ap.add_argument("--api", default="http://127.0.0.1:8010")
    args = ap.parse_args()
    db = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)
    for sid in args.session_ids:
        diagnose(db, sid, args.api if args.fix else None)
        print()


def summary_report(db: sqlite3.Connection, session_id: int) -> None:
    row = db.execute("SELECT s.json, s.lines_rev, ss.lines_rev FROM summary s JOIN session ss "
                     "ON ss.id = s.session_id WHERE s.session_id=?", (session_id,)).fetchone()
    if not row:
        print("summary: none")
        return
    items = cited = 0
    for value in (json.loads(row[0]).get("zh") or {}).values():
        if isinstance(value, list):
            items += len(value)
            cited += sum(isinstance(i, dict) and i.get("line") is not None for i in value)
    stale = " STALE" if row[1] != row[2] else ""
    print(f"summary: {cited}/{items} items cite a line{stale}")


def retranslate(api: str, session_id: int, line_id: int) -> str:
    req = urllib.request.Request(f"{api}/api/sessions/{session_id}/lines/{line_id}/retranslate",
                                 method="POST")
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["status"]


def diagnose(db: sqlite3.Connection, session_id: int, fix_api: str | None) -> None:
    lines = db.execute("SELECT id, speaker, lang, source, status, confidence FROM line "
                       "WHERE session_id=? ORDER BY start", (session_id,)).fetchall()
    if not lines:
        print(f"session {session_id} has no lines")
        return

    print(f"session {session_id}: {len(lines)} lines")
    print("status:", dict(Counter(l[4] for l in lines)))
    summary_report(db, session_id)
    failed = [l for l in lines if l[4] == "translate_failed"]
    for l in failed[:10]:
        print(f"  translate_failed #{l[0]}: {l[3][:50]}")
    if fix_api and failed:
        results = Counter(retranslate(fix_api, session_id, l[0]) for l in failed)
        print(f"  retranslated {len(failed)}: {dict(results)}")

    halluc = [l for l in lines if l[2].startswith("zh") and asr.is_hallucination(l[3])]
    print(f"\nhallucination-flagged zh lines: {len(halluc)}")
    for l in halluc[:10]:
        print(f"  #{l[0]} {l[1]}: {l[3][:60]}")
    loops = [(l, m.group(1)) for l in lines if (m := asr._REPEAT_LOOP.search(l[3]))]
    print(f"repeat-loop lines: {len(loops)}")
    for l, sig in loops[:10]:
        print(f"  #{l[0]} {l[1]} [{sig}]: {l[3][:60]}")

    conf = [l[5] for l in lines if l[5] is not None]
    if conf:
        low = sum(c < LOW_CONFIDENCE for c in conf)
        print(f"\nconfidence ({len(conf)} scored): " + "  ".join(
            f"p{p}={percentile(conf, p):.2f}" for p in (5, 10, 25, 50))
            + f"  below {LOW_CONFIDENCE}: {low / len(conf):.1%}")

    by_speaker: dict[str, list] = defaultdict(list)
    for l in lines:
        by_speaker[l[1]].append(l[5])
    print(f"\n{'speaker':<10} {'lines':>6} {'low%':>6}")
    for spk, cs in sorted(by_speaker.items(), key=lambda kv: -len(kv[1])):
        scored = [c for c in cs if c is not None]
        share = sum(c < LOW_CONFIDENCE for c in scored) / len(scored) if scored else 0.0
        print(f"{spk:<10} {len(cs):>6} {share:>6.1%}")


if __name__ == "__main__":
    main()

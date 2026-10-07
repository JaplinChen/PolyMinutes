"""Diagnose one session's transcript quality (read-only).

    python -m scripts.diag_session SESSION_ID [--db polyminutes.db]

Status counts, hallucination and repeat-loop survivors, and how confidence is spread overall and
per speaker — the numbers to look at before deciding whether a meeting needs a rerun.
"""

from __future__ import annotations

import argparse
import sqlite3
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
    ap.add_argument("session_id", type=int)
    ap.add_argument("--db", type=Path, default=store.DB_PATH)
    args = ap.parse_args()
    db = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)
    lines = db.execute("SELECT id, speaker, lang, source, status, confidence FROM line "
                       "WHERE session_id=? ORDER BY start", (args.session_id,)).fetchall()
    if not lines:
        sys.exit(f"session {args.session_id} has no lines")

    print(f"session {args.session_id}: {len(lines)} lines")
    print("status:", dict(Counter(l[4] for l in lines)))

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

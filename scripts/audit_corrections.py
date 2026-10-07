"""Audit learned correction rules against the transcripts they would rewrite (read-only).

    python -m scripts.audit_corrections [--db polyminutes.db]

A rule is a literal replacement everywhere, so its risk is how often the wrong side already appears
as ordinary text. Rules whose wrong side is still common are likely real words, not mishearings.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import correct, store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=store.DB_PATH)
    args = ap.parse_args()
    db = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)

    def impact(text: str) -> int:
        return db.execute("SELECT COUNT(*) FROM line WHERE instr(source, ?) > 0", (text,)).fetchone()[0]

    rows = []
    for wrong, right, count in db.execute("SELECT wrong, right, count FROM correction"):
        flags = []
        n_wrong = impact(wrong)
        if n_wrong >= correct.CONFIRM_IMPACT:
            flags.append("wrong side still common — real word?")
        if not correct._is_a_term_pair(wrong, right):
            flags.append("not a term pair")
        rows.append((n_wrong, impact(right), wrong, right, count, flags))
    rows.sort(key=lambda r: (-r[0], -r[1]))

    print(f"{'wrong→right':<24} {'learned':>7} {'right#':>6} {'wrong#':>6}  flags")
    for n_wrong, n_right, wrong, right, count, flags in rows:
        print(f"{wrong + '→' + right:<24} {count:>7} {n_right:>6} {n_wrong:>6}  {'; '.join(flags)}")
    flagged = sum(1 for r in rows if r[5])
    print(f"\n{len(rows)} rules, {flagged} flagged (CONFIRM_IMPACT={correct.CONFIRM_IMPACT})")


if __name__ == "__main__":
    main()

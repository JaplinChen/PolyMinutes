"""Audit learned correction rules against the transcripts they would rewrite (read-only).

    python -m scripts.audit_corrections [--db polyminutes.db] [--raw raw.json]

A rule is a literal replacement everywhere, so its risk is how often the wrong side already appears
as ordinary text. Rules whose wrong side is still common are likely real words, not mishearings.

Without --raw the wrong side is counted in the stored transcripts, where every active rule has
already replaced it — so only the right side's reach is informative. With --raw (from
scripts/raw_corpus.py) it is counted in an uncorrected decode, and each flagged rule shows one
sentence it would rewrite: that is how 15 rules were found rewriting NG, Media and 廁所 (2026-10-07).
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import correct, store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=store.DB_PATH)
    ap.add_argument("--raw", type=Path, help="uncorrected decode from scripts/raw_corpus.py")
    args = ap.parse_args()
    db = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)
    raw = ([t for v in json.loads(args.raw.read_text(encoding="utf-8")).values() for t in v]
           if args.raw else None)

    def impact(text: str) -> int:
        return db.execute("SELECT COUNT(*) FROM line WHERE instr(source, ?) > 0", (text,)).fetchone()[0]

    rows = []
    for wrong, right, count in db.execute("SELECT wrong, right, count FROM correction"):
        flags = []
        hits = [t for t in raw if wrong in t] if raw is not None else []
        n_wrong = len(hits) if raw is not None else impact(wrong)
        if n_wrong >= correct.CONFIRM_IMPACT:
            flags.append("wrong side still common — real word?")
            if hits:
                at = hits[0].find(wrong)
                flags.append("e.g. " + hits[0][max(0, at - 10):at + len(wrong) + 10])
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

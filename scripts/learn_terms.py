"""Turn the LLM pass's refused corrections into glossary candidates.

    python -m scripts.learn_terms transcripts/clean/*.txt --ollama qwen3:14b
    python -m scripts.learn_terms --apply          # write the accepted ones to the glossary

The refine pass throws away every correction its guards refuse. Those refusals are the most
informative thing it produces: a model that keeps wanting to write 工程變更 where the recogniser
wrote 一夕變更 is naming a term the glossary does not have. A term that shows up this way in
several places is worth adding — and once it is in the glossary, the same correction is allowed
through on the next run.

Refusals are stored as they are found, so mining and applying are separate steps and you get to
look at the list before it changes anything.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import correct, llm, postmeeting, refine  # noqa: E402
from server.correct import diff_terms  # noqa: E402
from server.store import Store  # noqa: E402

LINE = re.compile(r"^\[(\d+:\d+)\] (S\d+) \((\w+)\) (.*)$")
HAN = re.compile(r"[一-鿿]")


def read(path: Path) -> list[refine.Line]:
    lines = []
    for row in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := LINE.match(row):
            lines.append(refine.Line(m.group(2), m.group(3), m.group(4)))
    return lines


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", nargs="*", type=Path)
    ap.add_argument("--ollama", nargs="?", const="qwen3:14b", default=None)
    ap.add_argument("--model", default="")
    ap.add_argument("--topic", default="SAP ERP 導入訪談")
    ap.add_argument("--min-count", type=int, default=2,
                    help="times a term must be proposed before it counts as evidence")
    ap.add_argument("--max-sound", type=float, default=0.45,
                    help="how far a candidate may be from what was heard, as a fraction of pinyin")
    ap.add_argument("--apply", action="store_true", help="add the surviving candidates to the glossary")
    args = ap.parse_args()

    if not args.transcripts:
        print("nothing to read", file=sys.stderr)
        return 1

    cfg = llm.load_llm()
    if args.ollama:
        chat = refine.ollama_chat(args.model or args.ollama, llm.DEFAULT_ENDPOINTS["ollama"], think=True)
    else:
        if not cfg.api_key and cfg.provider != "ollama":
            print(f"no API key for {cfg.provider}; save one on the LLM settings page, or pass --ollama",
                  file=sys.stderr)
            return 1
        chat = postmeeting.chat_for(cfg, cfg.api_key, max_tokens=4000, model=args.model)

    store = Store()
    terms = store.glossary()
    known = {t.source for t in terms}
    proposed: Counter[tuple[str, str]] = Counter()

    for path in args.transcripts:
        lines = read(path)
        if not lines:
            continue
        rejected: list[refine.Rejected] = []
        refine.Refiner(chat, topic=args.topic).refine(lines, terms, rejected)
        for r in rejected:
            for before, after in diff_terms(r.original, r.candidate):
                if HAN.search(after) and after not in known:
                    proposed[(before, after)] += 1
        print(f"{path.name}: {len(rejected)} refused corrections")

    # Ranked by how far the candidate is from what was heard, because repetition alone turned out
    # not to separate signal from noise. Measured over seven interviews: every candidate within
    # 0.33 was a real term the recogniser mangled, and everything past 0.6 was the model guessing
    # from context — 心理 for 流程, 頭髮 for 流程, 午 for 長期. Distance is the axis that splits them.
    scored = []
    for (before, after), n in proposed.items():
        if n < args.min_count:
            continue
        pa = correct.pinyin_of(before, tones=False)
        pb = correct.pinyin_of(after, tones=False)
        scored.append((correct.edit_distance(pa, pb) / max(len(pa), len(pb), 1), before, after, n))
    scored.sort()

    survivors = [(b, a, n) for d, b, a, n in scored if d <= args.max_sound]
    dropped = len(scored) - len(survivors)
    print(f"{chr(10)}{len(survivors)} candidates seen at least {args.min_count} times, "
          f"nearest first ({dropped} dropped as too far from what was heard):")
    for d, before, after, n in scored:
        if d > args.max_sound:
            continue
        # A candidate the corrector could already reach needs no glossary entry.
        reachable = correct.pinyin_of(before, tones=False) == correct.pinyin_of(after, tones=False)
        print(f"  {d:.2f}  x{n}  {before} -> {after}{'   (already reachable)' if reachable else ''}")

    if args.apply:
        added = [after for _, after, _ in survivors if after not in known]
        for term in added:
            store.add_term(term, {}, mode="hint")
        print(f"\nadded {len(added)} terms to the glossary")
    elif survivors:
        print("\nrun again with --apply to add these, or add the ones you recognise by hand")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Repair an existing transcript with an LLM that can see the conversation around each line.

    python -m scripts.refine_transcript transcripts/clean/DXC-0721-開發.txt --topic "SAP ERP 導入訪談"

Writes `<name>.refined.txt` next to the input and prints every change, so the pass can be judged
rather than trusted. The provider, key and model come from llm.json (the LLM settings page);
ANTHROPIC_API_KEY is used only when the provider is anthropic and no key is saved.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import llm, postmeeting, refine  # noqa: E402
from server.store import Store  # noqa: E402

LINE = re.compile(r"^\[(\d+:\d+)\] (S\d+) \((\w+)\) (.*)$")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts", nargs="+", type=Path)
    ap.add_argument("--topic", default="SAP ERP 導入訪談", help="what the meeting is about")
    ap.add_argument("--model", default="", help="override the model in llm.json")
    ap.add_argument("--ollama", nargs="?", const="qwen3:14b", default=None,
                    help="use a local Ollama model instead of the cloud (default qwen3:14b)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N lines, for a trial run")
    ap.add_argument("--redo", action="store_true",
                    help="refine transcripts that already have a .refined.txt beside them")
    ap.add_argument("--think", action="store_true",
                    help="let a reasoning model deliberate; far slower, and not obviously better")
    args = ap.parse_args()

    cfg = llm.load_llm()
    if args.ollama:
        chat = refine.ollama_chat(args.model or args.ollama, llm.DEFAULT_ENDPOINTS["ollama"],
                                  think=args.think)
        label = f"ollama/{args.model or args.ollama}"
    else:
        key = cfg.api_key or (os.environ.get("ANTHROPIC_API_KEY", "")
                              if cfg.provider == "anthropic" else "")
        if not key and cfg.provider != "ollama":
            print(f"no API key for {cfg.provider}: save one on the LLM settings page, set "
                  f"ANTHROPIC_API_KEY for anthropic, or pass --ollama", file=sys.stderr)
            return 1
        chat = postmeeting.chat_for(cfg, key, max_tokens=4000, model=args.model)
        label = f"{cfg.provider}/{args.model or cfg.model}"

    # A run over seven interviews takes hours, and the reasons it stops are mundane: the machine
    # is needed, a model hangs, someone reboots. Finished transcripts are skipped and the pid is
    # on disk, so stopping it is one command and restarting it costs nothing already done.
    pid_file = Path("transcripts/.refine.pid")
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()), encoding="utf-8")

    try:
        for path in args.transcripts:
            target = path.with_suffix(".refined.txt")
            if target.exists() and target.stat().st_size and not args.redo:
                print(f"{path.name}: already done, skipping")
                continue
            refine_one(path, target, chat, label, args)
    finally:
        pid_file.unlink(missing_ok=True)

    return 0


def refine_one(path: Path, target: Path, chat, label: str, args) -> None:
    stamps: list[tuple[str, str, str]] = []
    lines: list[refine.Line] = []
    for row in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := LINE.match(row):
            stamp, speaker, lang, text = m.groups()
            stamps.append((stamp, speaker, lang))
            lines.append(refine.Line(speaker, lang, text))

    if not lines:
        print(f"{path.name}: no transcript lines", file=sys.stderr)
        return

    if args.limit:
        stamps, lines = stamps[: args.limit], lines[: args.limit]

    chunks = (len(lines) + refine.CHUNK_LINES - 1) // refine.CHUNK_LINES
    print(f"{chr(10)}{path.name}: {len(lines)} lines, {chunks} requests, model {label}")
    coverage = refine.Coverage()
    fixed = refine.Refiner(chat, topic=args.topic).refine(lines, Store().glossary(),
                                                          coverage=coverage)

    changed = 0
    out = []
    for (stamp, speaker, lang), before, after in zip(stamps, (l.text for l in lines), fixed):
        if before != after:
            changed += 1
            print(f"  [{stamp}] {before}{chr(10)}         -> {after}")
        out.append(f"[{stamp}] {speaker} ({lang}) {after}")

    target.write_text(chr(10).join(out) + chr(10), encoding="utf-8")
    print(f"{changed}/{len(lines)} lines changed -> {target}")
    if coverage.skipped:
        # Said out loud rather than left in the log: an unchanged line that was never looked at
        # reads exactly like one that was looked at and found correct.
        print(f"{coverage.skipped} lines ({coverage.fraction:.0%}) were left unchecked — the model "
              f"returned a whole chunk rewritten, which is restructuring rather than correcting")


if __name__ == "__main__":
    raise SystemExit(main())

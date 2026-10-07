"""Decode recorded sessions with no corrections applied, for auditing rules (read-only on the DB).

    python -m scripts.raw_corpus raw.json [--sessions 2 3 4]

The stored transcripts already have every learned rule applied, so a rule's wrong side can never be
seen in them — a good rule and a bad one both read as zero. This re-decodes each session's wav
(VAD + Whisper, no corrector, no translation) and writes {session_id: [zh lines]} for
`audit_corrections --raw`. 2.5 hours of audio decoded in about 4 minutes per session on one GPU.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import asr_gpu, config, postprocess, store  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--sessions", type=int, nargs="*")
    ap.add_argument("--db", type=Path, default=store.DB_PATH)
    args = ap.parse_args()
    db = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    ids = args.sessions or [r[0] for r in db.execute("SELECT id FROM session ORDER BY id")]
    terms = store.Store(args.db).glossary()
    transcriber = asr_gpu.Transcriber(languages=["zh", "vi"], hotwords=asr_gpu.hotwords_from(terms))

    out = {}
    for sid in ids:
        row = db.execute("SELECT wav_path FROM session WHERE id=?", (sid,)).fetchone()
        wav = config.recording_path(row[0]) if row else None
        if not wav or not wav.is_file():
            print(f"session {sid}: no recording, skipped", file=sys.stderr)
            continue
        started = time.monotonic()
        utterances = postprocess.segment(wav)
        postprocess.transcribe_all(utterances, transcriber)
        out[sid] = [u.text for u in utterances if u.text and u.lang.startswith("zh")]
        print(f"session {sid}: {len(out[sid])} lines, {time.monotonic() - started:.0f}s", flush=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()

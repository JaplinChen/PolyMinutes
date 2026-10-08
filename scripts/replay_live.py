"""Replay a recorded meeting through the live pipeline in real time, against a copy of the DB.

    python -m scripts.replay_live 8 --start 5880 --end 6300 [--speed 1] [--compare]

Real VAD, recogniser, diariser and translator, fed 100 ms blocks at the pace a capture would. The
live path had never been run on real audio until this replay found four bugs in an afternoon
(#198-#201) — a crash on two held clips, Mandarin labelled English, a lost closing brace, and six
of eleven held clips that were real speech. Run it after touching asr, pipeline, retry or translate.

The database is copied first and POLYMINUTES_DB pointed at the copy before anything from `server`
is imported, so the replay's lines never reach the room's transcripts.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BLOCK = 1600  # 100 ms at 16 kHz, what the capture hands the pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("session_id", type=int, help="session whose recording to replay")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=300.0)
    ap.add_argument("--speed", type=float, default=1.0, help="2 = twice real time")
    ap.add_argument("--compare", action="store_true", help="count the post-meeting lines in the window")
    ap.add_argument("--dump", type=Path, help="write the replayed lines here as JSON")
    args = ap.parse_args()

    real_db = ROOT / "polyminutes.db"
    copy = Path(tempfile.mkdtemp()) / "replay.db"
    shutil.copy(real_db, copy)
    os.environ["POLYMINUTES_DB"] = str(copy)

    import soundfile as sf  # noqa: E402

    from server import asr, config, llm, pipeline, postmeeting, translate  # noqa: E402
    from server.store import Store  # noqa: E402

    store = Store()
    cfg, llm_cfg = config.load(), llm.load_llm()
    wav = config.recording_path(store.session(args.session_id)["wav_path"])
    audio, _ = sf.read(str(wav), dtype="float32", start=int(args.start * config.SAMPLE_RATE),
                       frames=int((args.end - args.start) * config.SAMPLE_RATE))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    session = store.start_session("2000-01-01T00:00:00", str(wav))
    chat = postmeeting.chat_for(llm_cfg, "", max_tokens=1500, model=llm_cfg.translate_model)

    shown: dict[int, float] = {}  # line id -> seconds into the replay it first reached the page
    began = 0.0

    def emit(event: dict) -> None:
        line = event.get("line") or {}
        if line.get("id") is not None and not line.get("refined"):
            shown.setdefault(line["id"], (time.monotonic() - began) * args.speed)

    pipe = pipeline.Pipeline(cfg, store, session, translate.Translator(chat) if chat else None, emit)
    pipe.start()
    began = time.monotonic()
    for i in range(0, len(audio), BLOCK):
        pipe.tap.put(audio[i:i + BLOCK])
        due = (i + BLOCK) / config.SAMPLE_RATE / args.speed
        time.sleep(max(0.0, due - (time.monotonic() - began)))
    pipe.tap.put(None)
    pipe.join(timeout=600)

    lines = store.lines(session)
    lags = sorted(shown[l["id"]] - l["end_time"] for l in lines if l["id"] in shown and l["end_time"])
    zh = [l["source"] for l in lines if l["lang"].startswith("zh")]
    print(f"{args.end - args.start:.0f}s replayed: {len(lines)} lines, "
          f"{sum(bool(l['translations']) for l in lines)} translated, "
          f"statuses {dict((s, sum(l['status'] == s for l in lines)) for s in {l['status'] for l in lines})}")
    print(f"held clips: {pipe._retries.recovered} recovered, {pipe._retries.dropped} dropped; "
          f"errors={pipe.errors} backlog_peak={pipe.backlog_peak} blocks")
    if lags:
        print(f"lag after the speaker stopped: p50={lags[len(lags) // 2]:.1f}s "
              f"p95={lags[int(len(lags) * 0.95)]:.1f}s max={lags[-1]:.1f}s")
    print("languages:", dict((g, sum(l['lang'] == g for l in lines)) for g in {l["lang"] for l in lines}))
    print("hallucination survivors:", [t[:40] for t in zh if asr.is_hallucination(t)])
    print("loop survivors:", [t[:40] for t in zh if asr._REPEAT_LOOP.search(t)])
    if args.compare:
        db = sqlite3.connect(f"file:{real_db.as_posix()}?mode=ro", uri=True)
        n, secs = db.execute("SELECT COUNT(*), COALESCE(SUM(end_time - start), 0) FROM line "
                             "WHERE session_id=? AND start >= ? AND start < ?",
                             (args.session_id, args.start, args.end)).fetchone()
        covered = sum(l["end_time"] - l["start"] for l in lines if l["end_time"])
        print(f"post-meeting in the same window: {n} lines covering {secs:.0f}s; "
              f"live covered {covered:.0f}s")
    if args.dump:
        args.dump.write_text(json.dumps(
            [{**l, "shown": shown.get(l["id"])} for l in lines], ensure_ascii=False), encoding="utf-8")
    store.close()


if __name__ == "__main__":
    main()

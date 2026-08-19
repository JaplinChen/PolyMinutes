"""Give already-imported sessions the meeting time their source video was named after.

    python -m scripts.backfill_meeting_time PATH-TO-VIDEOS
    python -m scripts.backfill_meeting_time PATH-TO-VIDEOS --apply

Imports used to stamp `session.started` with the moment someone dragged the file in, so the meeting
list read as a list of evenings-spent-importing. New imports take the time out of the recorder's
own filename; these are the ones already in the database.

The wav name kept no trace of the video it came from, so the match is by duration: a session's wav
and its source video are the same recording, to within a second. Anything that matches more than
one video is left alone and printed — a coin flip is worse than a wrong-looking date you can fix.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import config  # noqa: E402
from server.ingest import meeting_time  # noqa: E402
from server.store import DB_PATH, Store  # noqa: E402

TOLERANCE = 2.0


def video_seconds(path: Path) -> float | None:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "default=nw=1:nk=1", str(path)],
                         capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_dir", help="folder holding the source recordings")
    ap.add_argument("--glob", default="*.mp4")
    ap.add_argument("--apply", action="store_true", help="write the new times (default: dry run)")
    args = ap.parse_args()

    videos = []
    for p in sorted(Path(args.video_dir).glob(args.glob)):
        if (secs := video_seconds(p)) is not None:
            videos.append((p, secs))
    if not videos:
        print(f"no readable videos in {args.video_dir}")
        return 1

    store = Store(DB_PATH)
    changed = 0
    for row in store.sessions():
        wav = config.recording_path(row["wav_path"])
        if not wav.exists():
            print(f"session {row['id']}: wav missing ({row['wav_path']})")
            continue
        secs = sf.info(str(wav)).duration
        hits = [p for p, v in videos if abs(v - secs) <= TOLERANCE]
        if len(hits) != 1:
            print(f"session {row['id']} ({row['started']}, {secs:.0f}s): {len(hits)} matches — skipped")
            continue
        started = meeting_time(hits[0].name)
        if started == row["started"] and started == row["ended"]:
            continue
        print(f"session {row['id']}: {row['started']} -> {started}  [{hits[0].name}]")
        changed += 1
        if args.apply:
            store.set_session_started(row["id"], started)
            # An import stamps both ends with the same time; leaving `ended` on the import clock
            # would have the meeting appear to run for three days.
            store.end_session(row["id"], started)
    print(f"{changed} session(s) {'updated' if args.apply else 'would change — rerun with --apply'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

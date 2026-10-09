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
    ap.add_argument("--lid", action="store_true",
                    help="re-detect each speaker's language from the audio (loads the GPU model)")
    args = ap.parse_args()
    db = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)
    for sid in args.session_ids:
        diagnose(db, sid, args.api if args.fix else None)
        if args.lid:
            language_check(db, sid)
        print()


_detector = None


def language_check(db: sqlite3.Connection, session_id: int, per_speaker: int = 4) -> None:
    """Each speaker's stored language against what their own longest clips detect as.

    A Vietnamese speaker sat labelled zh for days on the 2026-10-05 meeting: every line faded as
    low-confidence, every line a stilted Chinese "translation", and the diagnosis kept treating the
    symptom. Four clips of audio settled it in seconds — so the check is here, run on demand.
    """
    global _detector
    import soundfile as sf
    from server import asr_gpu, config

    if _detector is None:
        _detector = asr_gpu.Transcriber(languages=config.load().languages)
    wav = config.recording_path(db.execute("SELECT wav_path FROM session WHERE id=?",
                                           (session_id,)).fetchone()[0])
    speakers = db.execute("SELECT speaker, COUNT(*) FROM line WHERE session_id=? GROUP BY speaker "
                          "HAVING COUNT(*) >= 5", (session_id,)).fetchall()
    flagged = 0
    for speaker, _ in speakers:
        stored = Counter(r[0] for r in db.execute(
            "SELECT lang FROM line WHERE session_id=? AND speaker=?", (session_id, speaker)))
        clips = db.execute("SELECT start, end_time FROM line WHERE session_id=? AND speaker=? "
                           "AND end_time IS NOT NULL ORDER BY end_time - start DESC LIMIT ?",
                           (session_id, speaker, per_speaker)).fetchall()
        heard = Counter()
        for start, end in clips:
            audio, _ = sf.read(str(wav), dtype="float32", start=int(start * config.SAMPLE_RATE),
                               frames=int(min(end - start, 30) * config.SAMPLE_RATE))
            heard[_detector.detect_language(audio.mean(axis=1) if audio.ndim > 1 else audio)] += 1
        heard.pop("", None)
        said, labelled = (heard.most_common(1) or [("", 0)])[0][0], stored.most_common(1)[0][0]
        # Three confident clips agreeing, not a majority of two: one or two English-sounding clips
        # in a Mandarin speaker are code-switching, and flagged S4/S8/S26 for nothing.
        if said and said != labelled and heard[said] >= 3:
            flagged += 1
            print(f"  LANGUAGE? {speaker}: stored {dict(stored)}, audio detects {dict(heard)}")
    print(f"language check: {len(speakers)} speakers, {flagged} flagged")


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
    # Both kinds were reported by the user before this script saw them (2026-10): a line left as
    # 未能辨識, and a 0.77 s line too short for a re-run to recover what was said around it. Short
    # alone is ~25 a meeting of 謝謝大家/請繼續; short *and* unsure is where 採訪/剪輯 credits live.
    short = db.execute("SELECT COUNT(*) FROM line WHERE session_id=? AND end_time - start < 1.0",
                       (session_id,)).fetchone()[0]
    for row in db.execute("SELECT id, start, end_time - start, speaker, status, source FROM line "
                          "WHERE session_id=? AND (status='asr_failed' OR "
                          "(end_time - start < 1.0 AND confidence < ?)) ORDER BY start",
                          (session_id, LOW_CONFIDENCE)):
        kind = "asr_failed" if row[4] == "asr_failed" else f"short+unsure {row[2]:.2f}s"
        print(f"  {kind} #{row[0]} at {int(row[1]) // 60}:{int(row[1]) % 60:02d} {row[3]}: {row[5][:30]!r}")
    print(f"lines under 1 s: {short}")
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

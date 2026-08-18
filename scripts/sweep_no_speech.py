"""Sweep NO_SPEECH_MAX against a real recording.

    python -m scripts.sweep_no_speech recordings/import-20260817-093917.wav

Decodes once, keeping every segment's no_speech_prob and text, then scores each candidate
threshold offline. Two numbers per threshold: how much of what it drops is boilerplate the phrase
filter already recognises (the kill it is there for), and how much is text that filter would keep
(the collateral). The knee is the lowest threshold before collateral starts climbing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server import asr, asr_gpu, config, diarize, postprocess  # noqa: E402

CACHE = Path("scratch-no-speech.json")
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 1.01]


def decode(wav: Path) -> list[dict]:
    utterances = postprocess.segment(wav)
    if turns := diarize.turns(wav):
        utterances = postprocess.split_on_turns(utterances, turns)
    print(f"{len(utterances)} utterances")

    t = asr_gpu.Transcriber(languages=[])
    from faster_whisper import BatchedInferencePipeline

    t._batched = BatchedInferencePipeline(model=t._model)

    out = []
    step = postprocess.BATCH_UTTERANCES
    for start in range(0, len(utterances), step):
        group = utterances[start : start + step]
        gap = np.zeros(int(asr_gpu.BATCH_GAP_SECONDS * config.SAMPLE_RATE), dtype=np.float32)
        spans, parts, at = [], [], 0.0
        for u in group:
            seconds = len(u.samples) / config.SAMPLE_RATE
            spans.append({"start": at, "end": at + seconds})
            parts += [u.samples.astype(np.float32), gap]
            at += seconds + asr_gpu.BATCH_GAP_SECONDS
        segments, _ = t._decode_batched(np.concatenate(parts), "", spans)
        for seg in segments:
            out.append({"text": seg.text.strip(),
                        "no_speech": float(getattr(seg, "no_speech_prob", 0.0)),
                        "seconds": float(seg.end - seg.start)})
        print(f"  {min(start + step, len(utterances))}/{len(utterances)}", flush=True)
    return out


def report(rows: list[dict]) -> None:
    kept = [r for r in rows if r["text"]]
    # Judged by the filters that already exist: what they flag is the thing the threshold is
    # supposed to catch, and what they keep is what it must not.
    for r in kept:
        r["junk"] = bool(asr.is_hallucination(r["text"]) or asr.is_noise(r["text"])
                         or asr.is_degenerate(r["text"]))
    junk = sum(1 for r in kept if r["junk"])
    print(f"\n{len(kept)} decoded segments, {junk} flagged junk by the phrase filter\n")
    print(f"{'thresh':>7} {'kept':>6} {'junk kept':>10} {'clean dropped':>14}")
    for t in THRESHOLDS:
        survives = [r for r in kept if r["no_speech"] < t]
        print(f"{t:>7} {len(survives):>6} {sum(1 for r in survives if r['junk']):>10}"
              f" {sum(1 for r in kept if r['no_speech'] >= t and not r['junk']):>14}")


if __name__ == "__main__":
    wav = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if CACHE.exists() and not wav:
        rows = json.loads(CACHE.read_text(encoding="utf-8"))
    else:
        rows = decode(wav)
        CACHE.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    report(rows)

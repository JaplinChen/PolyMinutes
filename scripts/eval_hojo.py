"""Offline A/B: large-v3 (as PolyMinutes ships it) vs Hojo-ASR-Multi-V1, on the same clips.

Motivation: Hojo (HojoAI/Hojo-ASR-Multi-V1, Apache 2.0) is an Encoder-Adapter-LLM ASR — a
Qwen3-Omni audio encoder feeding a Qwen3-4B decoder. The claim worth testing on *our* audio is
that an LLM decoder disambiguates homophones/dialect from language knowledge (the room-mic zh<->vi
pain) and hallucinates far less on non-speech. Leaderboard WER is not our domain; this measures it
against the recordings we actually fail on.

    # manifest.jsonl — one clip per line (paths relative to the manifest file):
    #   {"audio": "clips/a.wav", "lang": "zh", "ref": "巴適得很"}
    #   {"audio": "clips/street.wav", "lang": "zh", "ref": "", "noise": true}
    python -m scripts.eval_hojo --manifest transcripts/eval/hojo.jsonl
    python -m scripts.eval_hojo --manifest ... --raw     # bypass PolyMinutes guards on large-v3
    python -m scripts.eval_hojo --selfcheck              # pure-function asserts, no GPU/models

Fairness notes, stated because they change how the numbers read:
- Recognition, not LID. Both models decode each clip under the clip's *known* language (large-v3 is
  forced to it; Hojo auto-detects, its only mode), and every hypothesis is scored in that same
  language bucket. Language identification is a separate axis the live pipeline handles per speaker.
- Default large-v3 output goes through the shipping guards (`Transcriber.transcribe` → `_judge`),
  because that is what PolyMinutes really emits — including the no-speech gate that already blanks
  most noise. `--raw` decodes the bare model so the noise/hallucination comparison is model-vs-model
  rather than guards-vs-model. Run both; the gap between them is what the guards buy today.
- Noise emission: on clips marked `noise` (or with empty ref), the metric is characters emitted —
  lower is better. This is the article's "Whisper 100 chars vs Hojo 5" headline, on our clips.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.bench_wav import normalize  # noqa: E402
from scripts.eval_harness import Line, accuracy, hallucination_rate  # noqa: E402
from server import asr_gpu, config  # noqa: E402

HOJO_ID = "HojoAI/Hojo-ASR-Multi-V1"

# A hand-typed reference line: `HH:MM:SS.mmm  <speaker>：<text>` (full- or half-width colon). The
# timecode shares the recording's 00:00 origin, so it slices the wav directly. Untimed lines (no
# leading timecode) can't be placed on the audio and are skipped.
REF_LINE = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d+)\s+(.+?)[：:](.*)$")


def build_manifest(transcript: Path, wav: Path, clips_dir: Path, out: Path,
                   lang: str, max_clip: float, char_rate: float) -> None:
    """Slice a wav into one clip per reference line and write the manifest to score against.

    The reference is a *selective* record, not a gapless verbatim — consecutive lines can be tens of
    seconds apart with untranscribed speech between them. So a clip cannot run to the next line's
    timecode: it would swallow all that untranscribed speech and the hypothesis explodes with text
    the reference never had. Instead each clip is bounded to the line's own estimated duration
    (characters / char_rate, a Mandarin speaking-rate proxy), capped by the next line's start and
    max_clip. This keeps the decoded audio close to the sentence the line actually holds.
    """
    rows = []
    for l in transcript.read_text(encoding="utf-8").splitlines():
        if m := REF_LINE.match(l):
            h, mi, s, ms, _spk, text = m.groups()
            text = text.strip()
            if text:
                rows.append((int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000, text))
    if not rows:
        raise SystemExit(f"no timecoded lines parsed from {transcript}")

    audio = read_16k_mono(wav)
    sr = config.SAMPLE_RATE
    clips_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as fh:
        for i, (start, text) in enumerate(rows):
            nxt = rows[i + 1][0] if i + 1 < len(rows) else start + max_clip
            # +0.8s slack for the tail the rate proxy underestimates; still far short of the gap.
            est = len(text) / char_rate + 0.8
            end = min(nxt, start + min(est, max_clip))
            seg = audio[int(start * sr):int(end * sr)]
            if len(seg) < int(0.3 * sr):
                continue
            name = f"{i:04d}_{int(start)}.wav"
            sf.write(str(clips_dir / name), seg, sr)
            rel = (clips_dir / name).relative_to(out.parent)
            fh.write(json.dumps({"audio": rel.as_posix(), "lang": lang, "ref": text},
                                ensure_ascii=False) + "\n")
            written += 1
    print(f"built {written} clips from {len(rows)} lines -> {out}")


def load_clips(manifest: Path) -> list[dict]:
    base = manifest.parent
    clips = []
    for row in manifest.read_text(encoding="utf-8").splitlines():
        row = row.strip()
        if not row or row.startswith("#"):
            continue
        c = json.loads(row)
        c["path"] = (base / c["audio"]).resolve()
        c["ref"] = c.get("ref", "")
        c["noise"] = bool(c.get("noise")) or not c["ref"].strip()
        clips.append(c)
    return clips


def read_16k_mono(path: Path) -> np.ndarray:
    audio, rate = sf.read(str(path), dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if rate != config.SAMPLE_RATE:
        # ponytail: linear resample, fine for an ASR-quality eval; swap to soxr if it ever matters.
        n = int(len(audio) * config.SAMPLE_RATE / rate)
        audio = np.interp(np.linspace(0, len(audio), n, endpoint=False),
                          np.arange(len(audio)), audio).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def run_large_v3(clips: list[dict], raw: bool) -> list[str]:
    t = asr_gpu.Transcriber(live=False)
    out = []
    for c in clips:
        samples = read_16k_mono(c["path"])
        if raw:
            # Bare decoder, no _judge — mirrors Transcriber.transcribe() minus the guards.
            segments, _ = t._model.transcribe(
                samples, language=c["lang"] or None, beam_size=asr_gpu.BEAM_SIZE,
                temperature=asr_gpu.TEMPERATURE, condition_on_previous_text=False)
            out.append("".join(s.text for s in segments).strip())
        else:
            out.append(t.transcribe(samples, c["lang"])[0])
    del t
    _free_gpu()
    return out


def run_hojo(clips: list[dict], device: str, batch_size: int) -> list[str] | None:
    try:
        from hojo_asr import HOJO_ASR
    except ImportError:
        print("hojo_asr not installed — `pip install -U hojo-asr` to include the Hojo column.\n",
              file=sys.stderr)
        return None
    model = HOJO_ASR.load_model(HOJO_ID, device=device)
    res = model.run_infer([str(c["path"]) for c in clips], batch_size=batch_size)
    if len(res) != len(clips):
        raise RuntimeError(f"Hojo returned {len(res)} results for {len(clips)} clips")
    out = [str(r.get("text", "")).strip() for r in res]
    del model
    _free_gpu()
    return out


def _free_gpu() -> None:
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:
        pass


_CC = None
_CN2AN = None


def fold(text: str) -> str:
    """Canonicalise Traditional/Simplified to one script before scoring.

    Two conventions differ across models and must be neutralised, or the CER measures style not
    recognition — and unequally, which poisons the very large-v3<->Hojo delta the A/B rests on:
      - script: the reference mixes 繁/简 ("质量表现" beside 繁体 lines), large-v3 emits 繁体, an LLM
        decoder tends to 简体 → fold everything to Simplified with opencc.
      - numerals: the reference and large-v3 write "471"/"第5周", the LLM decoder spells them out
        ("四百七十一"/"第五周") → fold Chinese numerals to Arabic with cn2an.
    Each fold is a no-op if its library is absent (the numbers then carry that penalty).
    """
    global _CC, _CN2AN
    if _CC is None:
        try:
            import opencc

            _CC = opencc.OpenCC("t2s")
        except Exception:
            _CC = False
    if _CN2AN is None:
        try:
            import cn2an

            _CN2AN = cn2an
        except Exception:
            _CN2AN = False
    if _CC:
        text = _CC.convert(text)
    if _CN2AN:
        try:
            text = _CN2AN.transform(text, "cn2an")
        except Exception:
            pass  # cn2an raises on some mixed strings; leave the text as-is for those.
    return text


def scorecard(clips: list[dict], texts: list[str]) -> dict:
    texts = [fold(t) for t in texts]
    clips = [{**c, "ref": fold(c["ref"])} for c in clips]
    # Speech clips only in the accuracy buckets — a noise clip's ref is empty and its emitted
    # garbage would inflate the wrong-language bucket. Non-speech is judged solely by noise_chars.
    speech = [(c, txt) for c, txt in zip(clips, texts) if not c["noise"]]
    ref = [Line(i, "S1", c["lang"], c["ref"]) for i, (c, _) in enumerate(speech)]
    hyp = [Line(i, "S1", c["lang"], txt) for i, (c, txt) in enumerate(speech)]
    noise_chars = sum(len(normalize(txt)) for c, txt in zip(clips, texts) if c["noise"])
    noise_clips = sum(1 for c in clips if c["noise"])
    return {"accuracy": accuracy(ref, hyp),
            "hallucination": hallucination_rate(hyp)["rate"],
            "noise_chars": noise_chars, "noise_clips": noise_clips}


def report(cards: dict[str, dict]) -> None:
    names = list(cards)
    langs = sorted({lang for c in cards.values() for lang in c["accuracy"]})
    w = 12
    print("\n--- recognition error (per language, lower is better) ---")
    print(f"  {'lang':6}" + "".join(f"{n:>{w}}" for n in names))
    for lang in langs:
        cells = []
        for n in names:
            a = cards[n]["accuracy"].get(lang)
            cells.append(f"{a['metric'].upper()} {a['rate']:.1%}" if a else "-")
        print(f"  {lang:6}" + "".join(f"{c:>{w}}" for c in cells))

    print("\n--- hallucination (boilerplate share, lower is better) ---")
    print(f"  {'':6}" + "".join(f"{cards[n]['hallucination']:>{w}.1%}" for n in names))

    print("\n--- noise emission (chars on non-speech clips, lower is better) ---")
    nc = cards[names[0]]["noise_clips"]
    print(f"  over {nc} noise clip(s):")
    print(f"  {'':6}" + "".join(f"{cards[n]['noise_chars']:>{w}}" for n in names))


def demo() -> None:
    """Pure-function self-check: manifest shape, bucketing, noise metric. No GPU, no models."""
    clips = [{"lang": "zh", "ref": "巴適得很", "noise": False},
             {"lang": "vi", "ref": "toi dong y", "noise": False},
             {"lang": "zh", "ref": "", "noise": True}]
    # Perfect on speech, and one model dumps garbage on the noise clip while the other stays quiet.
    good = scorecard(clips, ["巴適得很", "toi dong y", ""])
    bad = scorecard(clips, ["巴適得狠", "toi dong y", "嗨嗨嗨嗨嗨"])
    assert good["accuracy"]["zh"]["rate"] == 0.0, good
    assert bad["accuracy"]["zh"]["errors"] == 1, bad          # 很 -> 狠
    assert good["noise_chars"] == 0 and bad["noise_chars"] == 5, (good, bad)
    assert good["noise_clips"] == 1, good
    report({"clean": good, "noisy": bad})
    print("\nselfcheck ok")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, help="JSONL of {audio, lang, ref, noise?}")
    ap.add_argument("--raw", action="store_true", help="bypass PolyMinutes guards on large-v3")
    ap.add_argument("--device", default="cuda:0", help="Hojo device (default cuda:0)")
    ap.add_argument("--batch-size", type=int, default=8, help="Hojo run_infer batch size")
    ap.add_argument("--refresh", action="store_true", help="re-decode even if a hyp cache exists")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--build", type=Path, help="transcript to slice into a manifest (HH:MM:SS.mmm)")
    ap.add_argument("--wav", type=Path, help="recording the --build transcript aligns to")
    ap.add_argument("--clips-dir", type=Path, help="where --build writes per-line clips")
    ap.add_argument("--out", type=Path, help="manifest path --build writes")
    ap.add_argument("--lang", default="zh", help="language bucket for --build clips (default zh)")
    ap.add_argument("--max-clip", type=float, default=30.0, help="clip length cap, seconds")
    ap.add_argument("--char-rate", type=float, default=4.5,
                    help="Mandarin chars/sec used to bound each clip to its line's duration")
    args = ap.parse_args()

    if args.selfcheck:
        demo()
        return 0
    if args.build:
        if not (args.wav and args.clips_dir and args.out):
            ap.error("--build needs --wav, --clips-dir and --out")
        build_manifest(args.build, args.wav, args.clips_dir, args.out, args.lang,
                       args.max_clip, args.char_rate)
        return 0
    if not args.manifest:
        ap.error("--manifest is required (or use --selfcheck, or --build)")

    clips = load_clips(args.manifest)
    missing = [str(c["path"]) for c in clips if not c["path"].exists()]
    if missing:
        print("missing audio:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1
    print(f"{len(clips)} clips, {sum(c['noise'] for c in clips)} marked noise")

    # Decoding is the slow part; scoring is being iterated. Cache each model's raw hypotheses next
    # to the manifest so re-scoring (fold, bucketing) is instant. --refresh forces a re-decode.
    def decode_cached(tag: str, decode) -> list[str] | None:
        cache = args.manifest.with_suffix(f".hyp_{tag}.json")
        if cache.exists() and not args.refresh:
            print(f"using cached {tag} ({cache.name})")
            return json.loads(cache.read_text(encoding="utf-8"))
        print(f"decoding {tag} ...")
        texts = decode()
        if texts is not None:
            cache.write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")
        return texts

    cards = {}
    lv = f"large-v3{'+raw' if args.raw else ''}"
    cards[lv] = scorecard(clips, decode_cached(lv, lambda: run_large_v3(clips, args.raw)))
    hojo = decode_cached("hojo", lambda: run_hojo(clips, args.device, args.batch_size))
    if hojo is not None:
        cards["hojo"] = scorecard(clips, hojo)

    report(cards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

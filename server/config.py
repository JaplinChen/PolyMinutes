"""Settings for PolyMinutes. Loaded from config.json next to the repo root, env vars override."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.json"
RECORDINGS_DIR = ROOT / "recordings"
MODELS_DIR = ROOT / "models"


def recording_path(stored: str) -> Path:
    """Resolve a session's stored wav path against the repo root.

    Recordings are stored relative so renaming the project directory — or restoring a backup
    somewhere else — does not strand every meeting: absolute paths written before that survive
    only as long as the folder keeps its name, and this project has already outlived one name.
    """
    p = Path(stored)
    return p if p.is_absolute() else ROOT / p

# One file both sides read, so a bump moves the number everywhere it is shown. /api/health used to
# answer a literal "0.1.0" that was tied to nothing — and the dashboard overwrites its own
# build-time version with that answer, so the literal decided what the sidebar said. A bump in
# package.json alone would have left the UI reporting the old number.
VERSION = (ROOT / "VERSION").read_text(encoding="utf-8").strip() if (ROOT / "VERSION").is_file() else "0.0.0"

# Whisper wants 16 kHz mono; capturing at that rate avoids a resample step later.
SAMPLE_RATE = 16_000
CHANNELS = 1
# Frames per callback. 1600 @ 16 kHz = 100 ms — small enough that stopping feels instant,
# large enough that the callback isn't called so often it starves.
BLOCK_SIZE = 1600

VAD_MODEL = MODELS_DIR / "silero_vad.onnx"
SPEAKER_MODEL = MODELS_DIR / "speaker_embedding.onnx"
# Finds where one person stops and the next starts, which the VAD cannot: it hears speech against
# silence, so two people talking without a pause between them arrive as one utterance. Optional —
# without it the offline pass falls back to clustering whole VAD utterances.
SPEAKER_SEGMENTATION_MODEL = MODELS_DIR / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx"

# Whisper model directories, smallest first. The realtime tier is picked from `whisper_model`;
# postprocess always uses the largest available.
WHISPER_DIRS = {
    "tiny": MODELS_DIR / "sherpa-onnx-whisper-tiny",
    "base": MODELS_DIR / "sherpa-onnx-whisper-base",
    "small": MODELS_DIR / "sherpa-onnx-whisper-small",
    "medium": MODELS_DIR / "sherpa-onnx-whisper-medium",
    "large-v3": MODELS_DIR / "sherpa-onnx-whisper-large-v3",
}

# Below this cosine similarity to every known centroid, a segment starts a new speaker.
#
# Chosen on how the speech divides, not on how many clusters appear. Counting clusters is what
# produced the previous value of 0.45, which looked tidy — no tail of one-utterance speakers — and
# was tidy because everyone had been merged into one. Measured across two full interviews:
#
#   threshold   warehouse interview (37 min)   sales interview (67 min)
#   0.45        one cluster, 100%              49 / 14 / 3 / 1
#   0.55        one cluster, 100%              23 / 14 / 12 / 9 / 4 / 3
#   0.65        29 / 7                         21 / 14 / 12 / 8 / 4 / 3
#
# 0.65 is the only value that separates anyone in the first recording while staying stable in the
# second. Even there it is fragile: a room microphone behind Teams' noise suppression flattens the
# differences between voices, and on some recordings nothing separates them at all.
SPEAKER_THRESHOLD = 0.65
# Segments shorter than this give unstable embeddings; they inherit the previous speaker.
MIN_EMBED_SECONDS = 1.0
# Utterances of context sent with each translation request. Lives here because the live pipeline
# and the postprocess pass both send it, and they drifted once already — one read a named constant
# while the other sliced a literal 3.
CONTEXT_LINES = 3
# Similarity required before a stored voiceprint puts a name on a speaker.
#
# Unvalidated, and honestly so. The material that should have tested it — seven interviews across
# two firms sharing participants — could not, because clustering had merged each meeting into a
# single speaker, so comparing meetings compared everyone against everyone. Those pairs scored
# 0.69 to 0.94, which says nothing about whether two people are the same person.
#
# Kept equal to SPEAKER_THRESHOLD until a recording exists where clustering works and a known
# person appears twice.
KNOWN_SPEAKER_THRESHOLD = 0.65


@dataclass
class Display:
    """Subtitle presentation. Tuned for a TV at meeting-room viewing distance, not a desk monitor."""

    font_size: int = 40           # px for the source line; translations scale from this
    lines: int = 6                # utterances kept on screen before older ones scroll away
    show_source: str = "top"      # top | bottom | hidden
    show_speaker: bool = True
    colour_speakers: bool = True
    theme: str = "dark"           # dark | light


@dataclass
class Config:
    # Language codes present in the meeting. First entry is the display-order primary.
    # zh = Traditional Chinese (Taiwan) — ASR output is converted, see plan.md.
    languages: list[str] = field(default_factory=lambda: ["zh", "vi", "en"])
    # Substring matched against input device names, case-insensitive. Empty = system default.
    # Set this to the virtual audio device carrying the meeting audio (VB-Cable / BlackHole).
    input_device: str = ""
    whisper_model: str = "small"
    # Consecutive detections disagreeing with a speaker's established language before switching.
    # Higher for zh<->en because Taiwanese Mandarin routinely embeds English words and would
    # otherwise flip the speaker's language mid-meeting. See plan.md decision 5.
    language_switch_after: int = 3
    language_switch_after_zh_en: int = 6
    # Silence (seconds) that ends a live utterance. 0.5 cut speakers mid-breath into half-sentences
    # that reached Whisper as fragments and collapsed into filler; 0.7 rides over the pauses while
    # still feeling responsive. Tune per room — quieter, slower speakers may want it higher.
    vad_min_silence: float = 0.7
    # Speaker code -> language code. Pins a speaker so detection never overrides it.
    pinned_languages: dict[str, str] = field(default_factory=dict)
    display: Display = field(default_factory=Display)

    def save(self) -> None:
        CONFIG_PATH.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")

    def whisper_dir(self) -> Path:
        return WHISPER_DIRS.get(self.whisper_model, WHISPER_DIRS["small"])


def load() -> Config:
    data = {}
    if CONFIG_PATH.exists():
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

    known = {k: v for k, v in data.items() if k in Config.__dataclass_fields__}
    # Nested dataclass: json gives a plain dict, and an older file may lack newer display keys.
    if isinstance(known.get("display"), dict):
        known["display"] = Display(**{k: v for k, v in known["display"].items()
                                      if k in Display.__dataclass_fields__})
    cfg = Config(**known)

    if env := os.environ.get("POLYMINUTES_INPUT_DEVICE"):
        cfg.input_device = env
    if env := os.environ.get("POLYMINUTES_LANGUAGES"):
        cfg.languages = [s.strip() for s in env.split(",") if s.strip()]
    if env := os.environ.get("POLYMINUTES_WHISPER_MODEL"):
        cfg.whisper_model = env

    return cfg


def gpu_model(languages: list[str] | None = None, live: bool = False) -> str:
    """CTranslate2 model for the GPU path.

    The live path runs large-v3-turbo, the post-meeting path large-v3. Turbo keeps large-v3's
    encoder and cuts the decoder from 32 layers to 4: measured on pmc.wav here, 116.7s of wall
    against 44.2s, realtime 0.17 against 0.06, and whole-transcript CER within three points of
    large-v3 even scored against a reference large-v3 itself produced. What it loses is language
    identification — the same fifteen minutes came back 79 lines of Chinese from large-v3 and 60
    Vietnamese / 31 Chinese from turbo. Live can absorb that because the language is forced per
    speaker (`Diarizer.language_for`) rather than read off the decode; the post-meeting pass has
    no such anchor on a first run and an hour of decoding is not worth rushing, so it keeps v3.

    Breeze ASR 25, a large-v2 fine-tune for Taiwanese Mandarin and Mandarin-English code-switching,
    was tried here and dropped. On a real interview it and large-v3 differed on five lines out of
    a hundred and thirty-seven, all of them pre-meeting chatter where neither was clearly right,
    at the same realtime factor. What actually improved the transcript was leaving Whisper small
    behind; the fine-tune added nothing on top of that, and it does not know Vietnamese, which
    every meeting in this room contains.

    `languages` is accepted because the choice is language-dependent in principle — it just has
    one answer today.
    """
    if live:
        return os.environ.get("POLYMINUTES_GPU_MODEL_LIVE", "large-v3-turbo")
    return os.environ.get("POLYMINUTES_GPU_MODEL", "large-v3")


def gpu_index() -> int:
    """Which CUDA device the recogniser runs on. Zero unless told otherwise.

    This box has two cards. Whisper takes one; the other is free for a local LLM — the summary and
    correction stages can run on Ollama, and Ollama landing on the same card as live decoding is
    the one thing that makes the recogniser run short of memory mid-meeting. Ollama's card is set
    where its daemon starts (CUDA_VISIBLE_DEVICES); this sets Whisper's, so the two can be kept
    apart. A machine with one card ignores this and shares it, as it did before.
    """
    try:
        return int(os.environ.get("POLYMINUTES_GPU_INDEX", "0"))
    except ValueError:
        return 0


def available_whisper_models() -> list[str]:
    """Model tiers actually present on disk, smallest first."""
    return [name for name, path in WHISPER_DIRS.items() if path.is_dir()]

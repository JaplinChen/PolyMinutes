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
# Measured at last. The 2026-08-05 morning meeting has a human-reviewed transcript naming who is
# speaking, so its speakers can be labelled and the recogniser scored against the answer. 32
# prints from 5 people, each built the way `_remember_voices` builds one — the mean of five
# utterance embeddings:
#
#     same person, 131 pairs        min 0.758   p5 0.816   median 0.894
#     different people, 365 pairs   max 0.759   p95 0.686  median 0.470
#
# The two distributions meet at 0.759 and overlap nowhere. What decides the threshold is not that
# crossing point but the open-set case, which is the one that actually happens: hold a person out
# of the roster entirely and ask the roster who they are. A stranger has no right answer, so every
# name asserted is wrong — and strangers score as high as 0.759, because 總經理 and 人事Martin經理
# resemble each other more than either resembles the rest of the room.
#
#     threshold   names kept   strangers wrongly named
#          0.65        31/32                     14/32
#          0.70        31/32                     11/32
#          0.75        31/32                      2/32
#          0.80        31/32                      0/32
#
# 0.65 named fourteen strangers out of thirty-two. That is not a corner case: on this recording it
# put 人事Martin經理 on forty seconds of a production report he did not give, at 0.688. When the
# right person *is* in the roster they score 0.854 at worst, so 0.80 costs nothing and closes the
# whole gap. The one print of the 32 that is not recognised is the only print its speaker has —
# held out against itself, there is nothing left of them to match.
#
# What is not measured here is the same person across two *meetings*: every pair above shares one
# microphone and one hour. Cross-meeting scores will sit lower, so if real names start going
# unrecognised this number is the first suspect — and the near-misses land on the suggestions
# endpoint rather than vanishing.
KNOWN_SPEAKER_THRESHOLD = 0.80
# Above this, a recognition is trusted enough to feed back into learning: the matched centroid is
# stored as a new variant, so a voice that drifts (mic, cold, years) keeps refreshing its own
# prints without anyone renaming it. A notch above the assertion bar because a wrong auto-learned
# print compounds — it pulls the next meeting's match further off.
#
# It was 0.75, which stopped being "a notch above" when the assertion bar moved to 0.80 and became
# a notch below: everything asserted would also have been learned, which is the opposite of the
# rule. That inversion had already fired — a code holding eight seconds of the QC manager was
# named 三董 at 0.764 and folded into his roster, where nothing would ever have found it. 0.85
# because a genuine match scores 0.854 at worst on the measured set (see KNOWN_SPEAKER_THRESHOLD),
# so learning stays reserved for matches at the strong end of that range rather than its floor.
AUTO_LEARN_THRESHOLD = 0.85
# When the two best-matching *people* score within this of each other, no name is asserted — the
# match goes to the suggestions endpoint for a human with the audio instead. With up to 8 variants
# per person competing, a hair's-breadth win between two similar voices is noise, not identity.
RECOGNISE_MARGIN = 0.05
# Segments a live speaker accumulates before recognition is retried on the refined centroid. The
# first attempt runs on one utterance's embedding; an atypical opening sentence should not cost
# the whole meeting.
RECOGNISE_RECHECK_SEGMENTS = 5


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
    # Optional second capture device, matched the same way. When set, its audio is a separate
    # channel: room mic on `input_device`, remote participants via Teams loopback here. Speaker
    # clustering never crosses the two, which stops Teams' compressed voiceprints from collapsing
    # into the room's. Empty = single-channel capture, unchanged from before.
    loopback_device: str = ""
    whisper_model: str = "small"
    # Consecutive detections disagreeing with a speaker's established language before switching.
    # Higher for zh<->en because Taiwanese Mandarin routinely embeds English words and would
    # otherwise flip the speaker's language mid-meeting. See plan.md decision 5.
    language_switch_after: int = 3
    language_switch_after_zh_en: int = 6
    # Silence (seconds) that ends a live utterance. 0.5 cut speakers mid-breath into half-sentences
    # that reached Whisper as fragments and collapsed into filler; 0.7 still split sentences at
    # thinking pauses often enough that the post-meeting pass grew a merge stage. 0.9 rides over
    # those too, at the cost of subtitles arriving a beat later. Tune per room.
    vad_min_silence: float = 0.9
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

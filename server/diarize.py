"""Speaker separation and per-speaker language tracking.

Everything arrives on one audio stream — the machine is a silent listener in the meeting — so
speaker identity is the only way to tell participants apart, and it also decides which language
each utterance is transcribed in. A clustering mistake therefore costs twice: wrong name and
wrong language. Hence the hysteresis before ever changing a speaker's language.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sherpa_onnx
import soundfile as sf

from . import config

log = logging.getLogger("polyminutes.diarize")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def _similarities(rows: np.ndarray) -> np.ndarray:
    """Cosine of every row against every other — same rule as `cosine`, including the zero vector
    scoring zero rather than dividing by nothing."""
    norms = np.linalg.norm(rows, axis=1)
    safe = np.where(norms == 0, 1.0, norms)
    unit = rows / safe[:, None]
    unit[norms == 0] = 0.0
    return (unit @ unit.T).astype(np.float32)


@dataclass
class Speaker:
    code: str
    centroid: np.ndarray
    segments: int = 0
    language: str = ""  # established language; '' until the first transcription lands
    counts: dict[str, int] = field(default_factory=dict)
    _pending: tuple[str, int] = ("", 0)


@dataclass(frozen=True)
class Turn:
    """One stretch of one person talking, as the segmentation model heard it."""

    start: float
    end: float
    speaker: int


# Speech shorter than this is not a turn, and a gap shorter than this does not end one. Both are
# sherpa's own defaults for meeting audio; loosening either turns a breath into a speaker change.
MIN_TURN_ON = 0.3
MIN_TURN_OFF = 0.5

# How the recording is cut up for the workers. The cost of segmentation is exactly linear in audio
# length — four quarters take the same total as the whole — so the only way to spend less wall
# clock on it is to run pieces at the same time. Measured on 15 minutes of meeting audio: 47.7s in
# one process, 10.6s across eight.
CHUNK_SECONDS = 180.0
# Every chunk carries this much of its neighbour on each side. The segmentation model decides from
# a window about ten seconds wide, so audio within ten seconds of a cut is judged with one side
# missing. Overlapping by twice that means every instant is seen once with both sides intact, at
# the cost of ~11% redundant work.
CHUNK_OVERLAP_SECONDS = 20.0
# The threshold for grouping turns into speakers once the chunks are back. Not SPEAKER_THRESHOLD:
# that one is sherpa's FastClustering over whole VAD utterances, and this is complete linkage over
# one embedding per turn — a different rule over shorter clips, so the number that suits one does
# not suit the other. Swept against the whole-file result on a 2h19m meeting:
#
#     threshold   speakers   biggest speaker   agreement with whole-file
#          0.30         32               57%                      87.3%
#          0.40         56               57%                      89.2%
#          0.65        227               28%                      60.5%
#
# At 0.65 the chair alone came back as dozens of people. 0.40 puts the biggest speaker at 57%
# against the whole file's 56%, and scores identically on the ten utterances that can be
# attributed by what is being said: 33 of 33 pairs from different people kept apart.
TURN_CLUSTER_THRESHOLD = 0.40
# Workers, whatever the core count. Each one holds both models and its slice of audio — call it
# 300 MB — so a twenty-core box would otherwise start eighteen of them and want five gigabytes for
# a background task. Eight already took 15 minutes of audio from 47.7s to 10.6s; past that the
# curve flattens and only the memory keeps growing.
MAX_WORKERS = 8


def _chunk_spans(duration: float, chunk: float = CHUNK_SECONDS,
                 overlap: float = CHUNK_OVERLAP_SECONDS) -> list[tuple[float, float, float, float]]:
    """`(read_from, read_to, core_from, core_to)` per chunk.

    A chunk is decoded over its padded span but only owns its core. The cores tile the recording
    exactly — no gap, no double coverage — so a turn belongs to exactly one chunk and reconciling
    them afterwards is a question of where a turn sits, not of how similar two turns look.
    """
    if overlap < 10.0:
        raise ValueError("overlap must cover the model's window; below 10s a cut has no context")
    if duration <= chunk:
        return [(0.0, duration, 0.0, duration)]

    spans, at = [], 0.0
    while at < duration:
        core_to = min(at + chunk, duration)
        spans.append((max(at - overlap, 0.0), min(core_to + overlap, duration), at, core_to))
        at = core_to
    return spans


def _segment_chunk(job: tuple[str, float, float, float, float, float]) -> tuple[list[tuple[float, float]], list[list[float]]]:
    """One chunk, in a worker process: boundaries and one embedding per turn.

    Runs in a spawned process, so it takes a path and two times rather than audio — pickling the
    samples would send the recording through the parent for every chunk. Returns absolute times:
    sherpa answers relative to what it was handed, and forgetting the offset does not raise, it
    silently stacks the whole meeting at the front.

    The speaker ids sherpa assigns are per-chunk and meaningless across chunks, so they are
    dropped and the embeddings come back instead, to be clustered once over the whole recording.
    """
    wav, read_from, read_to, core_from, core_to, threshold = job
    import sherpa_onnx as so

    from . import config as cfg_mod

    with sf.SoundFile(wav) as fh:
        rate = fh.samplerate
        fh.seek(int(read_from * rate))
        audio = fh.read(int((read_to - read_from) * rate), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # One thread each: the pool already uses every core, and two layers of parallelism fight.
    cfg = so.OfflineSpeakerDiarizationConfig(
        segmentation=so.OfflineSpeakerSegmentationModelConfig(
            pyannote=so.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(cfg_mod.SPEAKER_SEGMENTATION_MODEL)), num_threads=1),
        embedding=so.SpeakerEmbeddingExtractorConfig(model=str(cfg_mod.SPEAKER_MODEL),
                                                     num_threads=1),
        clustering=so.FastClusteringConfig(threshold=threshold),
        min_duration_on=MIN_TURN_ON, min_duration_off=MIN_TURN_OFF)
    result = so.OfflineSpeakerDiarization(cfg).process(audio).sort_by_start_time()

    extractor = so.SpeakerEmbeddingExtractor(
        so.SpeakerEmbeddingExtractorConfig(model=str(cfg_mod.SPEAKER_MODEL), num_threads=1))
    kept, vectors = [], []
    for s in result:
        start, end = float(s.start) + read_from, float(s.end) + read_from
        # Owned by whichever chunk its middle falls in. Positional, so a turn crossing a cut is
        # taken once rather than by both neighbours or by neither.
        middle = (start + end) / 2
        if not (core_from <= middle < core_to or (core_to >= read_to and middle >= core_from)):
            continue
        clip = audio[int((start - read_from) * rate):int((end - read_from) * rate)]
        if len(clip) == 0:
            continue
        stream = extractor.create_stream()
        stream.accept_waveform(rate, clip)
        stream.input_finished()
        kept.append((start, end))
        vectors.append([float(v) for v in extractor.compute(stream)])
    return kept, vectors


def _heal(found: list[Turn]) -> list[Turn]:
    """Join neighbouring turns that ended up on the same speaker with no real gap between them.

    Two chunks each decode their own side of a cut, so one person talking across it comes back as
    two turns that meet in the middle. After global clustering they carry the same label, and a
    gap shorter than the one that ends a turn is not a turn boundary.
    """
    healed: list[Turn] = []
    for turn in found:
        if healed and healed[-1].speaker == turn.speaker and turn.start - healed[-1].end < MIN_TURN_OFF:
            healed[-1] = Turn(healed[-1].start, max(healed[-1].end, turn.end), turn.speaker)
        else:
            healed.append(turn)
    return healed


def turns(wav: Path, threshold: float | None = None) -> list[Turn] | None:
    """Where the speaker changes across a whole recording, or None if the model is not installed.

    The VAD cannot answer this. It hears speech against silence, so two people talking without a
    pause between them arrive as one utterance and get one embedding — an average of both, which
    then clusters as if it were a third person. Measured on a 2h19m meeting: 72 of 859 utterances
    (8.4%, ten minutes of speech) held more than one voice, one of them four.

    Optional on purpose: the model is a separate 6 MB download, and a machine without it should
    still be able to process a recording.

    The answer is cached beside the recording. It takes eight minutes on a 2h19m meeting and
    depends on nothing but the audio, the two models and the thresholds — so every reprocess after
    the first was paying eight minutes for a byte-identical result, and reprocessing is the normal
    path: it is what someone does after naming the speakers.
    """
    model = config.SPEAKER_SEGMENTATION_MODEL
    if not model.is_file():
        # Deliberately not cached: installing the model later must take effect at once.
        return None

    resolved = config.SPEAKER_THRESHOLD if threshold is None else threshold
    key = _turns_key(wav, resolved)
    cached = _read_turns(wav, key)
    if cached is not None:
        log.info("%d turns for %s from cache", len(cached), wav.name)
        return cached

    info = sf.info(str(wav))
    if info.samplerate != config.SAMPLE_RATE:
        raise ValueError(f"{wav} is {info.samplerate} Hz, expected {config.SAMPLE_RATE}")
    duration = float(info.duration)

    spans = _chunk_spans(duration)
    jobs = [(str(wav), a, b, c, d, resolved) for a, b, c, d in spans]
    workers = min(len(jobs), MAX_WORKERS, max((os.cpu_count() or 4) - 2, 1))
    log.info("segmenting %s in %d chunks across %d workers", wav.name, len(jobs), workers)

    boundaries: list[tuple[float, float]] = []
    vectors: list[np.ndarray] = []
    if workers == 1:
        pieces = [_segment_chunk(job) for job in jobs]
    else:
        try:
            # spawn on Windows, so the worker imports `server.diarize` fresh. It must never reach
            # `server.main`, which builds a Store and reads config at import.
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pieces = list(pool.map(_segment_chunk, jobs))
        except (BrokenProcessPool, OSError):
            # A pool that will not start is a slower import, not a failed one. Spawn re-imports
            # the parent's `__main__`, so an embedder whose entry point lacks the usual guard
            # takes this path rather than losing the recording.
            log.warning("segmentation pool unavailable; falling back to one process", exc_info=True)
            pieces = [_segment_chunk(job) for job in jobs]
    for kept, embedded in pieces:
        boundaries.extend(kept)
        vectors.extend(np.asarray(v, dtype=np.float32) for v in embedded)

    if not boundaries:
        _write_turns(wav, key, [], duration)
        return []

    order = sorted(range(len(boundaries)), key=lambda i: boundaries[i][0])
    # One clustering pass over the whole meeting. Each chunk numbered its own speakers and those
    # numbers mean nothing to each other, so the labels have to be decided here or a person who
    # spoke in two chunks arrives as two people.
    labels = cluster_offline([vectors[i] for i in order], threshold=TURN_CLUSTER_THRESHOLD)
    found = _heal([Turn(boundaries[i][0], boundaries[i][1], label)
                   for i, label in zip(order, labels)])

    # An offset lost in a worker does not raise, it silently moves speech to the wrong minute.
    if found and found[-1].end > duration + 1.0:
        raise ValueError(f"a turn ends past {wav.name}: {found[-1].end:.1f}s of {duration:.1f}s")

    _write_turns(wav, key, found, duration)
    return found


# Bumped by hand when the shape of a cached entry changes, so old files miss instead of
# deserialising into something that no longer means what it meant.
_CACHE_FORMAT = 2


def _stamp(path: Path) -> str:
    try:
        st = path.stat()
        return f"{path}:{st.st_size}:{st.st_mtime_ns}"
    except OSError:
        return f"{path}:missing"


def _turns_key(wav: Path, threshold: float) -> str:
    """Everything the answer depends on, hashed.

    The recording is hashed by content, not by size and mtime. A stale hit here does not fail
    loudly — it attributes one meeting's speakers to another meeting's audio, and re-running does
    not help because the miss is deterministic too. Half a second of sha256 against an eight
    minute computation is not a trade worth making.
    """
    digest = hashlib.sha256()
    with wav.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    parts = [
        f"format={_CACHE_FORMAT}",
        f"audio={digest.hexdigest()}",
        f"segmentation={_stamp(config.SPEAKER_SEGMENTATION_MODEL)}",
        f"embedding={_stamp(config.SPEAKER_MODEL)}",
        f"threshold={threshold!r}",
        f"cluster={TURN_CLUSTER_THRESHOLD!r}",
        f"on={MIN_TURN_ON!r}",
        f"off={MIN_TURN_OFF!r}",
        f"chunk={CHUNK_SECONDS!r}/{CHUNK_OVERLAP_SECONDS!r}",
        f"sherpa={getattr(sherpa_onnx, '__version__', 'unknown')}",
    ]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _cache_path(wav: Path) -> Path:
    return wav.with_suffix(wav.suffix + ".turns.json")


def _read_turns(wav: Path, key: str) -> list[Turn] | None:
    path = _cache_path(wav)
    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A truncated or absent file is a miss, never an error: the answer is recomputable.
        return None
    if blob.get("key") != key:
        return None
    try:
        found = [Turn(float(a), float(b), int(c)) for a, b, c in blob["turns"]]
    except (KeyError, TypeError, ValueError):
        return None
    # Turns past the end of the audio mean the file was written by something that got its offsets
    # wrong. Refusing them here is cheap, and the alternative is a transcript nobody can explain.
    duration = float(blob.get("duration", 0.0))
    if any(t.end > duration + 1.0 for t in found):
        log.warning("ignoring %s: a turn ends past the recording", path.name)
        return None
    return found


def _write_turns(wav: Path, key: str, found: list[Turn], duration: float) -> None:
    path = _cache_path(wav)
    blob = {"key": key, "duration": duration,
            "turns": [[t.start, t.end, t.speaker] for t in found]}
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        # Written through a temp file: a half-written json sitting next to a recording would be a
        # permanently broken session, and this runs at the end of an eight minute computation.
        temp.write_text(json.dumps(blob), encoding="utf-8")
        os.replace(temp, path)
    except OSError:
        log.warning("could not cache turns for %s", wav.name)
        temp.unlink(missing_ok=True)


class Diarizer:
    """Online clustering plus language bookkeeping.

    Online rather than offline because subtitles cannot wait for the meeting to end. The offline
    pass in postprocess sees every segment at once and corrects what this got wrong.
    """

    def __init__(self, model: str | None = None, threshold: float | None = None,
                 cfg: config.Config | None = None, known: list[tuple[str, np.ndarray]] | None = None,
                 known_languages: dict[str, str] | None = None):
        ec = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(model or config.SPEAKER_MODEL), num_threads=1
        )
        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(ec)
        self._threshold = config.SPEAKER_THRESHOLD if threshold is None else threshold
        self._cfg = cfg or config.Config()
        self.speakers: list[Speaker] = []
        self._last_code: str | None = None
        # Voices this room has met before, as (name, centroid). A new speaker whose embedding
        # matches one is named on the spot instead of arriving as another anonymous Sn.
        self._known = list(known or [])
        # name -> forced language, for the voices the room has set one on. A recognised speaker is
        # transcribed in it rather than left to auto-detect, which flips zh to vi on this room's mic.
        self._known_languages = dict(known_languages or {})
        self.recognised: dict[str, str] = {}

    def embed(self, samples: np.ndarray) -> np.ndarray:
        stream = self._extractor.create_stream()
        stream.accept_waveform(config.SAMPLE_RATE, samples)
        stream.input_finished()
        return np.array(self._extractor.compute(stream), dtype=np.float32)

    def assign(self, samples: np.ndarray) -> Speaker:
        """Identify the speaker of one utterance, creating a new one if nothing matches."""
        duration = len(samples) / config.SAMPLE_RATE

        # Short clips give unstable embeddings — a hummed "OK" would otherwise mint a new speaker
        # every time. Inheriting the previous speaker is right far more often than guessing.
        if duration < config.MIN_EMBED_SECONDS and self._last_code:
            return self._by_code(self._last_code)

        emb = self.embed(samples)

        best, best_score = None, -1.0
        for spk in self.speakers:
            score = cosine(emb, spk.centroid)
            if score > best_score:
                best, best_score = spk, score

        if best is None or best_score < self._threshold:
            best = Speaker(code=f"S{len(self.speakers) + 1}", centroid=emb)
            if name := self._recognise(emb):
                self.recognised[best.code] = name
            self.speakers.append(best)
        else:
            # Running mean: later segments refine the centroid without a stored history.
            n = best.segments
            best.centroid = (best.centroid * n + emb) / (n + 1)

        best.segments += 1
        self._last_code = best.code
        return best

    def _recognise(self, emb: np.ndarray) -> str:
        """Name a freshly minted speaker if a known voiceprint is close enough.

        Held to a higher bar than in-meeting clustering. Merging two segments of one meeting wrongly
        costs a split transcript; putting last week's name on this week's stranger is a mistake
        nobody reading the transcript would think to check.
        """
        best, score = "", -1.0
        for name, centroid in self._known:
            if (s := cosine(emb, centroid)) > score:
                best, score = name, s
        return best if score >= config.KNOWN_SPEAKER_THRESHOLD else ""

    def language_for(self, speaker: Speaker) -> str:
        """Language to force on this speaker's next utterance. '' means let Whisper auto-detect."""
        if pinned := self._cfg.pinned_languages.get(speaker.code):
            return pinned
        # A recognised voice the room set a language on wins over the language auto-detect drifted
        # into: identity is surer than a per-utterance guess on a noisy room mic.
        if name := self.recognised.get(speaker.code):
            if lang := self._known_languages.get(name):
                return lang
        return speaker.language

    def observe_language(self, speaker: Speaker, detected: str) -> None:
        """Record which language an utterance actually turned out to be.

        Switching needs several consecutive disagreements, and many more between Chinese and
        English: Taiwanese Mandarin routinely embeds English words, so a single English-heavy
        sentence must not flip the speaker and wreck every following transcription.
        """
        if not detected or speaker.code in self._cfg.pinned_languages:
            return

        speaker.counts[detected] = speaker.counts.get(detected, 0) + 1

        if not speaker.language:
            # Two agreeing detections before a new speaker is locked in. The lock is one-way in
            # practice: `language_for` then forces this language on every later utterance, so the
            # decode can only ever come back agreeing with it and no disagreement arrives to switch
            # it. A single mis-identification on someone's opening sentence would own them for the
            # rest of the meeting — and the live recogniser is turbo, whose language identification
            # is the one thing it is measurably worse at.
            first, _ = speaker._pending
            speaker._pending = ("", 0) if first == detected else (detected, 1)
            if first == detected:
                speaker.language = detected
            return

        if detected == speaker.language:
            speaker._pending = ("", 0)
            return

        lang, count = speaker._pending
        count = count + 1 if lang == detected else 1
        needed = self._switch_threshold(speaker.language, detected)

        if count >= needed:
            speaker.language = detected
            speaker._pending = ("", 0)
        else:
            speaker._pending = (detected, count)

    def _switch_threshold(self, current: str, candidate: str) -> int:
        if {current, candidate} == {"zh", "en"}:
            return self._cfg.language_switch_after_zh_en
        return self._cfg.language_switch_after

    def _by_code(self, code: str) -> Speaker:
        return next(s for s in self.speakers if s.code == code)


def load_known(store) -> list[tuple[str, np.ndarray]]:
    """Known voiceprints as arrays. Stored as raw float32 bytes — the embedder's own layout.

    Every variant, not one per name: _recognise takes the closest match, so a person's several
    stored prints each get a shot at naming a returning voice.
    """
    return [(name, np.frombuffer(blob, dtype=np.float32))
            for name, blob in store.known_voiceprints()]


def cluster_offline(embeddings: list[np.ndarray], threshold: float | None = None) -> list[int]:
    """Agglomerative clustering over every segment of a finished meeting, on complete linkage.

    Seeing all segments at once fixes the online pass's mistakes: two clusters that online kept
    apart because the speaker's first few seconds were atypical get merged here.

    Two clusters join only when their *worst* cross-pair still clears the threshold. Averaging
    them into a centroid instead — which this did — produces a mean voice that resembles everyone
    a little, so the biggest cluster keeps absorbing and one speaker ends up holding the meeting.
    On a 2h19m morning meeting, 859 segments:

        linkage    thr    clusters   biggest   different speakers kept apart
        centroid   0.65        35     93.2%     0 of 33
        centroid   0.80       176     31.1%    33 of 33
        complete   0.65       134     14.1%    33 of 33

    Ten utterances there were attributable by content — the chair, a sales report, a production
    report — and at 0.65 every one of the 33 pairs from different people had been merged. That is
    the transcript naming almost every line after whoever S1 turned out to be.

    The threshold stays where two earlier interviews put it. Complete linkage splits one speaker
    across more codes than centroid merging did, which is the trade taken deliberately: naming the
    same person into several boxes is tedious but possible, while a cluster holding four people
    cannot be separated by anything the person reading the transcript can do.
    """
    if not embeddings:
        return []

    thr = config.SPEAKER_THRESHOLD if threshold is None else threshold
    n = len(embeddings)
    members: list[list[int]] = [[i] for i in range(n)]

    # Every pair's similarity, computed once and then repaired in place. Rescanning all pairs in
    # Python each round is O(n^3), and a two-hour meeting segments into the thousands: that spent
    # over an hour here with the GPU sitting idle, never reaching transcription at all.
    #
    # A merged cluster is masked rather than deleted — deleting means copying the whole matrix
    # every round, which is the same cubic cost again with a smaller constant.
    sims = _similarities(np.asarray(embeddings, dtype=np.float32))
    np.fill_diagonal(sims, -np.inf)
    alive = np.ones(n, dtype=bool)

    # Each row's best partner, so choosing the pair to merge is a scan of n values, not n^2.
    best_at = sims.argmax(axis=1) if n > 1 else np.zeros(n, dtype=int)
    best = sims[np.arange(n), best_at] if n > 1 else np.full(n, -np.inf)

    for _ in range(n - 1):
        i = int(np.argmax(np.where(alive, best, -np.inf)))
        j = int(best_at[i])
        if best[i] < thr:
            break
        if i > j:
            i, j = j, i

        # Complete linkage: the merged cluster sits as far from every other as its worse half did.
        row = np.minimum(sims[i], sims[j])
        members[i] += members[j]
        members[j] = []
        alive[j] = False

        row[j] = -np.inf
        row[i] = -np.inf
        sims[i, :] = row
        sims[:, i] = row
        sims[j, :] = -np.inf
        sims[:, j] = -np.inf
        best[j] = -np.inf
        best_at[i] = int(np.argmax(row))
        best[i] = row[best_at[i]]

        # Similarities only ever fall here, so a row's cached best can only be stale downwards —
        # and only if it pointed at one of the two clusters that just became one.
        stale = alive & ((best_at == i) | (best_at == j))
        stale[i] = False
        for r in np.flatnonzero(stale):
            best_at[r] = int(np.argmax(sims[r]))
            best[r] = sims[r, best_at[r]]

    labels = [0] * n
    for label, group in enumerate([m for m in members if m]):
        for idx in group:
            labels[idx] = label
    return labels

"""Audio capture. PortAudio (sounddevice) so Windows and macOS share one code path.

The meeting audio must reach us through a virtual audio device (VB-Cable on Windows,
BlackHole on macOS) with Teams output pointed at it. Muting the system playback device
instead would make the capture silent — see plan.md.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
import soxr

from . import config


class DeviceNotFound(Exception):
    pass


# How long stop() waits to hand a consumer its end-of-stream sentinel before giving up. A live
# consumer frees a slot well within this; a dead one would otherwise block stop() forever.
SENTINEL_TIMEOUT = 5.0


def _signal_end(q: "queue.Queue", sentinel=None) -> None:
    try:
        q.put(sentinel, timeout=SENTINEL_TIMEOUT)
    except queue.Full:
        pass


def list_input_devices() -> list[dict]:
    """Every device that can be recorded from, in PortAudio index order."""
    return [
        {"index": i, "name": d["name"], "channels": d["max_input_channels"], "hostapi": sd.query_hostapis(d["hostapi"])["name"]}
        for i, d in enumerate(sd.query_devices())
        if d["max_input_channels"] > 0
    ]


# Windows exposes the same device under MME, DirectSound, WASAPI and WDM-KS; macOS uses Core Audio.
# Lower latency first — but which of them will actually open varies by driver and even by moment,
# so this is a preference order to try, not a choice to commit to. See candidate_devices.
_HOSTAPI_RANK = {"Core Audio": 0, "Windows WASAPI": 1, "Windows WDM-KS": 2, "Windows DirectSound": 3, "MME": 4}

# How long a freshly started stream has to deliver its first block before it is judged dead.
# Blocks are ~100 ms, so this is generous; it is only paid when a candidate is broken.
FIRST_BLOCK_TIMEOUT = 1.0


def candidate_devices(name_fragment: str) -> list[int | None]:
    """Every device matching the name, best host API first. `[None]` means the system default.

    Returns a list rather than one index because a device that PortAudio lists is not necessarily
    a device it can open: on the same machine the same Stereo Mix opened under MME one minute and
    only under WDM-KS the next. Trying in order and taking the first that starts is the only
    reliable approach.

    Raises DeviceNotFound rather than falling back silently — a silent fallback to the wrong
    device is the failure that presents as "it recorded nothing".
    """
    if not name_fragment:
        return [None]

    needle = name_fragment.casefold()
    matches = [d for d in list_input_devices() if needle in d["name"].casefold()]

    if not matches:
        available = ", ".join(d["name"] for d in list_input_devices()) or "(none)"
        raise DeviceNotFound(f"No input device matching {name_fragment!r}. Available: {available}")

    matches.sort(key=lambda d: _HOSTAPI_RANK.get(d["hostapi"], 9))
    return [d["index"] for d in matches]


def resolve_device(name_fragment: str) -> int | None:
    """First-choice device index. None = system default."""
    return candidate_devices(name_fragment)[0]


def device_format(device: int | None) -> tuple[int, int]:
    """Native (sample rate, channels) to open the stream with.

    Capturing at 16 kHz directly does not work in general: WASAPI shared mode only accepts the
    device's own rate, and virtual cables commonly run at 44.1 or 48 kHz. So capture native and
    convert afterwards rather than asking the driver for something it will refuse.
    """
    info = sd.query_devices(sd.default.device[0] if device is None else device)
    rate = int(info["default_samplerate"]) or config.SAMPLE_RATE
    # Two channels is plenty to downmix from; asking for more of a multichannel device is wasted work.
    channels = min(int(info["max_input_channels"]), 2) or 1
    return rate, channels


def to_mono_16k(block: np.ndarray, rate: int) -> np.ndarray:
    """Downmix then resample. Whisper and the speaker model both want 16 kHz mono."""
    mono = block.mean(axis=1) if block.ndim > 1 and block.shape[1] > 1 else block.reshape(-1)
    if rate == config.SAMPLE_RATE:
        return mono.astype(np.float32)
    return soxr.resample(mono.astype(np.float32), rate, config.SAMPLE_RATE)


@dataclass
class RecorderStatus:
    recording: bool
    path: str | None
    seconds: float
    peak: float  # 0.0-1.0 of the most recent block; 0.0 for a long stretch means no audio is arriving
    dropped_blocks: int


class Recorder:
    """Captures to a wav file on a writer thread.

    File IO is kept off the PortAudio callback so a slow disk cannot cause an overrun, and the
    recording is deliberately independent of everything downstream: without Graph API this wav is
    the only source for the post-meeting transcript, so it must survive a pipeline crash.
    """

    def __init__(self, device: "int | None | list[int | None]" = None,
                 tap: "queue.Queue | None" = None, source: str = ""):
        # Accepts a list of candidates so start() can fall through to the next host API.
        self._candidates: list[int | None] = device if isinstance(device, list) else [device]
        self.device: int | None = self._candidates[0]
        # Channel label carried to the pipeline so it can keep this device's speakers apart from the
        # other channel's. "" for single-channel capture. Two recorders share one tap; each tags its
        # own blocks, so the single consumer can tell them apart.
        self._source = source
        self._queue: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=256)
        # Optional second consumer (the transcription pipeline). It is deliberately separate from
        # the writer queue: if the pipeline stalls, blocks are dropped from the tap only and the
        # recording — the sole source for the post-meeting transcript — stays complete.
        self._tap = tap
        self._stream: sd.InputStream | None = None
        self._writer: threading.Thread | None = None
        self._path: Path | None = None
        self._frames = 0
        self._peak = 0.0
        self._dropped = 0
        self.tap_dropped = 0
        self._first_block = threading.Event()
        self.rate, self.channels = device_format(self.device)

    def _callback(self, indata, _frames, _time, status) -> None:
        self._first_block.set()
        if status:
            self._dropped += 1
        # Convert on the audio thread: it is a few hundred microseconds per block, and doing it
        # here means the wav and the pipeline both get 16 kHz mono without converting twice.
        block = to_mono_16k(indata, self.rate)
        self._peak = float(np.abs(block).max()) if len(block) else 0.0
        try:
            self._queue.put_nowait(block)
        except queue.Full:
            # Writer is wedged. Losing a block beats blocking the audio thread and cascading.
            self._dropped += 1
        if self._tap is not None:
            try:
                self._tap.put_nowait((self._source, block))
            except queue.Full:
                self.tap_dropped += 1

    def _write_loop(self, path: Path) -> None:
        with sf.SoundFile(path, mode="w", samplerate=config.SAMPLE_RATE, channels=config.CHANNELS, subtype="PCM_16") as f:
            while (block := self._queue.get()) is not None:
                f.write(block)
                self._frames += len(block)

    def start(self, path: Path) -> None:
        if self._stream is not None:
            raise RuntimeError("already recording")

        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._frames = 0
        self._peak = 0.0
        self._dropped = 0

        self._writer = threading.Thread(target=self._write_loop, args=(path,), daemon=True)
        self._writer.start()

        try:
            self._stream = self._open()
        except Exception:
            # Do not leave the writer thread waiting on a queue nothing will ever fill.
            self._queue.put(None)
            self._writer.join(timeout=5)
            self._writer = None
            raise

    def _open(self) -> sd.InputStream:
        """Open the first candidate that both starts *and* delivers audio.

        Starting successfully is not enough. On Windows a WDM-KS endpoint opens without error and
        then never fires a callback, which presents as an empty transcript with no error anywhere —
        the worst possible failure for an unattended meeting recorder. So each candidate has to
        prove itself by producing a block before it is accepted.
        """
        errors = []
        for device in self._candidates:
            self.rate, self.channels = device_format(device)
            stream = None
            try:
                # blocksize scales with the native rate so each callback still covers ~100 ms.
                stream = sd.InputStream(
                    samplerate=self.rate,
                    channels=self.channels,
                    blocksize=int(config.BLOCK_SIZE * self.rate / config.SAMPLE_RATE),
                    device=device,
                    dtype="float32",
                    callback=self._callback,
                )
                self._first_block.clear()
                stream.start()
                if not self._first_block.wait(FIRST_BLOCK_TIMEOUT):
                    raise RuntimeError(f"opened but delivered no audio within {FIRST_BLOCK_TIMEOUT}s")
                self.device = device
                return stream
            except Exception as exc:  # PortAudio raises several unrelated types
                errors.append(f"device {device}: {exc}")
                if stream is not None:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass

        raise RuntimeError("could not open any matching input device — " + "; ".join(errors))

    def stop(self) -> Path | None:
        if self._stream is None:
            return None

        self._stream.stop()
        self._stream.close()
        self._stream = None

        # End-of-stream to both consumers, but never on a blocking put. The pipeline is designed to
        # crash and let recording continue (Pipeline._run swallows and returns) — which is exactly
        # when its tap sits full at capacity with nothing draining it. A bare put(None) there would
        # hang stop() forever, and stop() runs before release_gpu(), so the card would stay claimed
        # and no later recording could start. Bounded: a live-but-backlogged consumer still gets the
        # sentinel within the wait; a dead one no longer wedges the stop.
        # ponytail: SENTINEL_TIMEOUT ceiling — a consumer that cannot free one slot in that window
        # loses the sentinel and its daemon thread lingers until process exit, which is harmless.
        _signal_end(self._queue)
        if self._writer:
            self._writer.join(timeout=10)
            self._writer = None
        if self._tap is not None:
            _signal_end(self._tap, (self._source, None))

        return self._path

    @property
    def native_format(self) -> str:
        return f"{self.rate} Hz / {self.channels} ch"

    def status(self) -> RecorderStatus:
        return RecorderStatus(
            recording=self._stream is not None,
            path=str(self._path) if self._path else None,
            seconds=self._frames / config.SAMPLE_RATE,
            peak=self._peak,
            dropped_blocks=self._dropped,
        )


def new_session_path() -> Path:
    return config.RECORDINGS_DIR / f"session_{time.strftime('%Y%m%d_%H%M%S')}.wav"

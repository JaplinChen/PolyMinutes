"""WASAPI loopback: record what a speaker plays, so no virtual cable is needed on Windows."""

from __future__ import annotations

import sys
import threading

RATE, CHANNELS = 48000, 2


def list_output_devices() -> list[str]:
    """Speakers whose playback can be looped back. Empty off Windows, where loopback is unsupported."""
    if sys.platform != "win32":
        return []
    import soundcard as sc
    return [s.name for s in sc.all_speakers()]


class LoopbackStream:
    """Speaker loopback with the start/stop/close surface Recorder expects of a stream."""

    def __init__(self, device: str, blocksize: int, callback):
        self._speaker = device.partition(":")[2]
        self._blocksize, self._callback = blocksize, callback
        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def _run(self) -> None:
        try:
            import soundcard as sc  # Windows-only path; CI and macOS never import it
            speaker = sc.get_speaker(self._speaker) if self._speaker else sc.default_speaker()
            mic = sc.get_microphone(speaker.name, include_loopback=True)
            with mic.recorder(samplerate=RATE, channels=CHANNELS, blocksize=self._blocksize) as rec:
                while self._running.is_set():
                    self._callback(rec.record(numframes=self._blocksize), None, None, None)
        except Exception as exc:
            self._error = exc

    def start(self) -> None:
        self._running.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2)

    def close(self) -> None:
        pass

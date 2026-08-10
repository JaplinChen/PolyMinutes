"""One card, one claimant. The gate that keeps the post-meeting pass off the live pipeline's GPU.

Two Whisper models do not fit the way the arithmetic suggests they might. A large-v3 in float16
plus a batched pipeline at BATCH_SIZE=32 is sized to have the card to itself; loading a second one
beside the live recogniser does not merely halve throughput, it pushes the live realtime factor
past 1, and once that happens `Pipeline.tap` fills its 600-block backlog in a minute and starts
discarding audio. Automating the post-meeting pass without this gate would trade a transcript
nobody re-ran for subtitles the room watched go missing.

The live meeting always wins. A background pass is worth minutes of GPU time; a meeting happening
right now is not repeatable. Cancellation is cooperative — `postprocess` checks between batches —
and safe to act on, because `Store.replace_lines` only swaps the transcript once, at the end. An
abandoned pass leaves the previous transcript exactly as it was.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

log = logging.getLogger("polyminutes.jobs")

# Seconds a starting meeting waits for a cancelled pass to notice and let go. Cancellation is
# checked between decode batches, so the wait is one batch, not one meeting.
YIELD_TIMEOUT = 30.0

_gpu = threading.BoundedSemaphore(1)
# One offline pass at a time, whoever holds the card. Until the GPU gate was narrowed to the
# decode alone, this was a side effect of that gate: a pass held the card start to finish, so a
# second one could not begin. Narrowing it left the CPU stages free to overlap, and two of them
# together is two full recordings in memory and two pools of segmentation workers. Separate from
# the card because a pass now spends most of its life not holding one.
_pass = threading.BoundedSemaphore(1)


@dataclass
class Job:
    """What the dashboard needs to say about a session's post-meeting pass."""

    state: str = "refining"  # refining | refined | failed | cancelled
    # Which part of the pass is running. The state stays "refining" throughout so every existing
    # consumer — the /refine endpoint, claim_gpu's stand-down scan, the tests — keeps working;
    # the stage is extra detail for a page that wants to say "summarizing" instead of "refining".
    stage: str = "rewrite"  # rewrite | refine | summarize
    error: str = ""
    cancel: threading.Event = field(default_factory=threading.Event)
    thread: threading.Thread | None = None


_jobs: dict[int, Job] = {}
_lock = threading.Lock()


def state(session_id: int) -> dict | None:
    with _lock:
        job = _jobs.get(session_id)
        return {"state": job.state, "stage": job.stage, "error": job.error} if job else None


def states() -> dict[int, dict]:
    with _lock:
        return {sid: {"state": j.state, "stage": j.stage, "error": j.error}
                for sid, j in _jobs.items()}


def schedule(session_id: int, run: Callable[[threading.Event], None],
             followup: Callable[[threading.Event, Callable[[str], None]], None] | None = None,
             needs_gpu: bool = True) -> bool:
    """Run `run` on a worker once the GPU is free. False when this session already has a pass.

    `run` is handed the cancel event and is expected to check it; a pass that ignores it simply
    finishes, which is correct but makes the next meeting wait.

    `followup` runs after the gate is released. It exists because the pass grew stages that never
    touch the card — the LLM correction and the summary. Held inside the gate, a minutes-long
    Ollama call would keep `claim_gpu` waiting past its timeout and the room would be told it
    cannot start recording, on account of work that was not using the GPU at all. The followup
    gets the cancel event and a stage setter; its failure fails the job the same way `run`'s does,
    but whatever `run` already landed stays landed.
    """
    with _lock:
        existing = _jobs.get(session_id)
        # Thread liveness, not just the recorded state: a worker that has finished its run but not
        # yet been marked still holds the gate, and replacing its entry here would orphan it —
        # the permit never comes back and every later pass waits on a job nobody is tracking.
        if existing and (existing.state == "refining"
                         or (existing.thread and existing.thread.is_alive())):
            return False
        job = Job()
        _jobs[session_id] = job

    def set_stage(stage: str) -> None:
        with _lock:
            job.stage = stage

    def _run_gpu_stage() -> bool:
        """The card-bound stage. Returns False if the job ended inside it (cancelled or failed).

        Held under the GPU gate only when `needs_gpu`. A summarize-only regeneration passes a no-op
        run and needs_gpu=False: entering the gate for it would make a pure-LLM job wait behind a
        meeting's recording, which is the exact block the followup split exists to avoid — and this
        endpoint is one someone clicks while another meeting may be underway.
        """
        if job.cancel.is_set():
            _finish(job, "cancelled")
            return False
        try:
            run(job.cancel)
        except Cancelled:
            _finish(job, "cancelled")
            return False
        except Exception as exc:
            # A failed pass is a visible state, not a log line nobody reads. The transcript it
            # was rewriting is still whole, so this is recoverable by re-running.
            log.exception("post-meeting pass failed for session %d", session_id)
            _finish(job, "failed", f"{type(exc).__name__}: {exc}")
            return False
        return True

    def worker() -> None:
        with _pass:
            if needs_gpu:
                with _gpu:
                    if not _run_gpu_stage():
                        return
            elif not _run_gpu_stage():
                return
        # The card is free (or was never taken). A meeting can start while the followup is still
        # talking to a language model, which is the entire point of the split.
        if followup:
            try:
                followup(job.cancel, set_stage)
            except Cancelled:
                _finish(job, "cancelled")
                return
            except Exception as exc:
                log.exception("post-meeting followup failed for session %d", session_id)
                _finish(job, "failed", f"{type(exc).__name__}: {exc}")
                return
        _finish(job, "refined")

    job.thread = threading.Thread(target=worker, name=f"reprocess-{session_id}", daemon=True)
    job.thread.start()
    return True


def _finish(job: Job, state_: str, error: str = "") -> None:
    with _lock:
        job.state, job.error = state_, error


class Cancelled(Exception):
    """Raised inside a pass when a meeting needs the card back. Nothing was written."""


# Defined above `schedule` in spirit but placed here for readability; `schedule` catches it to mark
# the job cancelled rather than failed. One name for the concept, so a pass that yields politely is
# never filed as a crash.


def claim_gpu(timeout: float = YIELD_TIMEOUT) -> bool:
    """Take the card for a live meeting, asking any background pass to stand down first."""
    with _lock:
        running = [j for j in _jobs.values() if j.state == "refining"]
    for job in running:
        job.cancel.set()
    return _gpu.acquire(timeout=timeout)


def release_gpu() -> None:
    try:
        _gpu.release()
    except ValueError:
        # Releasing a gate this process never took would mask the bug rather than fix it, so it is
        # logged and swallowed: a stop must never fail on account of bookkeeping.
        log.warning("release_gpu called without a matching claim")


@contextmanager
def borrow_gpu(timeout: float = 900.0):
    """Wait your turn for the card. For work that is not a live meeting and must not preempt one.

    An import runs the same post-meeting pass, but nothing about it is time-critical, so it queues
    behind whatever holds the card instead of asking it to stand down.
    """
    if not _gpu.acquire(timeout=timeout):
        raise TimeoutError("the GPU is busy")
    try:
        yield
    finally:
        release_gpu()


@contextmanager
def one_pass(timeout: float = 900.0):
    """Hold the offline-pass slot for work that does not go through `schedule`.

    An import calls `rewrite_session` straight from the request. Scheduled passes take this slot
    in their worker; an import has to take it here, or an import and a reprocess overlap and the
    machine holds two recordings and two sets of segmentation workers at once.
    """
    if not _pass.acquire(timeout=timeout):
        raise TimeoutError("another recording is still being processed")
    try:
        yield
    finally:
        _pass.release()


def cancel_all(wait: float = 0.0) -> None:
    """Ask every running pass to stop. Used on shutdown, where nothing should outlive the process."""
    with _lock:
        running = [j for j in _jobs.values() if j.state == "refining"]
    for job in running:
        job.cancel.set()
    if wait:
        for job in running:
            if job.thread:
                job.thread.join(timeout=wait)


def reset() -> None:
    """Drop all job state. Tests only."""
    with _lock:
        _jobs.clear()

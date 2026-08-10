"""Live capture: start, stop, status, and the subtitle websocket."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from . import audio, jobs, main
from .pipeline import Pipeline

log = logging.getLogger("polyminutes")

router = APIRouter()


@router.post("/api/recording/start")
def start_recording() -> dict:
    if main.state["recorder"]:
        raise HTTPException(409, "already recording")

    cfg = main.state["cfg"]
    try:
        candidates = audio.candidate_devices(cfg.input_device)
    except audio.DeviceNotFound as exc:
        raise HTTPException(400, str(exc)) from exc

    # The meeting wins the card. Any post-meeting pass is asked to stand down and gets one batch to
    # notice; a meeting starting now is not something anyone can repeat later.
    if not jobs.claim_gpu():
        raise HTTPException(503, "a post-meeting pass is still finishing — try again in a moment")
    main.state["gpu"] = True

    path = audio.new_session_path()
    session_id = None
    # One guard over everything after the claim. Any escape without releasing strands the permit,
    # and a stranded permit is silent: every later pass simply waits, forever, on nothing.
    try:
        session_id = main.store.start_session(time.strftime("%Y-%m-%dT%H:%M:%S"), str(path))
        try:
            # Constructing the pipeline is what loads the recogniser, the VAD and the speaker model
            # off disk. Missing weights raised out of here as an unhandled error, so the page showed
            # "HTTP 500" — while the exception itself named the exact file. That message is the
            # whole answer on a machine where the models were never downloaded.
            pipe = Pipeline(cfg, main.store, session_id, main._make_translator(), main.hub.publish)
        except FileNotFoundError as exc:
            raise HTTPException(503, f"speech model not ready: {exc}") from exc
        rec = audio.Recorder(candidates, tap=pipe.tap)
        try:
            rec.start(path)
        except RuntimeError as exc:
            raise HTTPException(400, str(exc)) from exc
        pipe.start()
    except BaseException:
        if session_id is not None:
            main.store.end_session(session_id, time.strftime("%Y-%m-%dT%H:%M:%S"))
        jobs.release_gpu()
        main.state["gpu"] = False
        raise

    log.info("capturing from device %s at %s", rec.device, rec.native_format)
    main.state.update(recorder=rec, pipeline=pipe, session=session_id)
    return recording_status()


@router.post("/api/recording/stop")
def stop_recording() -> dict:
    if not main.state["recorder"]:
        raise HTTPException(409, "not recording")
    return main._stop_capture()


@router.get("/api/recording/status")
def recording_status() -> dict:
    rec, pipe = main.state["recorder"], main.state["pipeline"]
    if not rec:
        return {"recording": False, "path": None, "seconds": 0.0, "peak": 0.0,
                "droppedBlocks": 0, "sessionId": None, "backlog": 0, "errors": 0}
    s = rec.status()
    return {
        "recording": s.recording,
        "path": s.path,
        "seconds": round(s.seconds, 2),
        "peak": round(s.peak, 4),
        "droppedBlocks": s.dropped_blocks,
        "sessionId": main.state["session"],
        "backlog": pipe.tap.qsize() if pipe else 0,
        "errors": pipe.errors if pipe else 0,
    }


@router.websocket("/ws/live")
async def live(ws: WebSocket) -> None:
    await ws.accept()
    queue_ = main.hub.subscribe()
    try:
        cfg = main.state["cfg"]
        await ws.send_json({"type": "config", "languages": cfg.languages, "display": asdict(cfg.display)})
        while True:
            await ws.send_json(await queue_.get())
    except WebSocketDisconnect:
        pass
    finally:
        main.hub.unsubscribe(queue_)

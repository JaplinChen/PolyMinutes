"""Health, audio devices and the capture config."""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException

from . import audio, config, main

router = APIRouter()


@router.get("/api/health")
def health() -> dict:
    return {"status": "ok", "version": config.VERSION}


@router.get("/api/devices")
def devices() -> dict:
    cfg = main.state["cfg"]
    try:
        selected = audio.resolve_device(cfg.input_device)
        error = None
    except audio.DeviceNotFound as exc:
        selected, error = None, str(exc)
    return {"devices": audio.list_input_devices(), "configured": cfg.input_device,
            "selected": selected, "error": error}


@router.get("/api/config")
def get_config() -> dict:
    cfg = main.state["cfg"]
    return {
        "languages": cfg.languages,
        "inputDevice": cfg.input_device,
        "loopbackDevice": cfg.loopback_device,
        "whisperModel": cfg.whisper_model,
        "availableModels": config.available_whisper_models(),
        "pinnedLanguages": cfg.pinned_languages,
        "translatorReady": bool(main._api_key_present()),
        "display": asdict(cfg.display),
    }


@router.put("/api/config")
def put_config(body: dict) -> dict:
    """Validate everything first, then apply.

    A partially applied update is worse than a rejected one: the caller sees a 400 and assumes
    nothing changed, while the running config has silently drifted.
    """
    cfg = main.state["cfg"]
    pending: dict = {}

    if "languages" in body:
        langs = [str(s) for s in body["languages"]]
        if not 2 <= len(langs) <= 3:
            raise HTTPException(400, "languages must contain 2 or 3 entries")
        # A duplicate would mean translating a line into its own source language.
        if len(set(langs)) != len(langs):
            raise HTTPException(400, "languages must be distinct")
        pending["languages"] = langs

    if "display" in body:
        patch = {k: v for k, v in dict(body["display"]).items() if k in config.Display.__dataclass_fields__}
        try:
            candidate = config.Display(**{**asdict(cfg.display), **patch})
            candidate.font_size = int(candidate.font_size)
            candidate.lines = int(candidate.lines)
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"invalid display settings: {exc}") from exc
        if candidate.show_source not in ("top", "bottom", "hidden"):
            raise HTTPException(400, "show_source must be top, bottom or hidden")
        if not 1 <= candidate.lines <= 20:
            raise HTTPException(400, "lines must be between 1 and 20")
        if not 12 <= candidate.font_size <= 200:
            raise HTTPException(400, "font_size must be between 12 and 200")
        if candidate.theme not in ("dark", "light"):
            raise HTTPException(400, "theme must be dark or light")
        pending["display"] = candidate

    if "inputDevice" in body:
        pending["input_device"] = str(body["inputDevice"])
    if "loopbackDevice" in body:
        pending["loopback_device"] = str(body["loopbackDevice"])
    if "whisperModel" in body:
        pending["whisper_model"] = str(body["whisperModel"])
    if "pinnedLanguages" in body:
        pending["pinned_languages"] = {str(k): str(v) for k, v in dict(body["pinnedLanguages"]).items()}

    for key, value in pending.items():
        setattr(cfg, key, value)
    cfg.save()
    # The subtitle page applies display changes live rather than needing a reload on the TV.
    main.hub.publish({"type": "config", "languages": cfg.languages, "display": asdict(cfg.display)})
    return get_config()

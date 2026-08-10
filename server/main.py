"""FastAPI entry point. Serves the built dashboard and the capture API on localhost.

The endpoints live in the `routes_*` modules; what stays here is the app, the process-wide
singletons they read through, and the handful of helpers that touch more than one of them.
Routers reach those through this module at call time rather than importing the objects, so
swapping `main.store` or `main.postprocess` under a running app takes effect everywhere.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import config, jobs, llm, postmeeting, postprocess, translate
from .hub import Hub
from .store import Store

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("polyminutes")

DIST = config.ROOT / "dashboard" / "dist"
CLIP_SECONDS = 4

state: dict = {"recorder": None, "loopback": None, "pipeline": None, "session": None,
               "gpu": False, "cfg": config.load(), "llm": llm.load_llm()}
store = Store()
keys = llm.KeyStore()
hub = Hub()


def _api_key_present() -> bool:
    """Whether translation can run. Separate from _api_key so a status check never burns a rotation
    slot — next_key() advances the cursor and increments the key's request count."""
    cfg: llm.LlmConfig = state["llm"]
    pooled = any(k["provider"] == cfg.provider for k in keys.list())
    return pooled or bool(cfg.api_key) or bool(os.environ.get("ANTHROPIC_API_KEY"))


def _api_key() -> str:
    """Key precedence: rotation pool, then the LLM settings page, then the environment."""
    cfg: llm.LlmConfig = state["llm"]
    return keys.next_key(cfg.provider) or cfg.api_key or os.environ.get("ANTHROPIC_API_KEY", "")


def _make_translator() -> translate.Translator | None:
    """No LLM configured is a supported mode: transcription still runs, translations stay empty.

    The chat callable comes from the same dispatcher the post-meeting pass uses, so live translation
    honours the provider chosen on the settings page (Ollama needs no key; the cloud providers do)."""
    key = _api_key()

    def on_reject(exc: Exception) -> None:
        # The provider rejected this key. Bench it so the next rotation skips it — a rate limit for
        # its cooldown, a bad key until it is removed — instead of dealing every request the same
        # dud. Only pooled keys are tracked; a key from settings or the environment has nothing to
        # mark, and mark_failure simply finds no match.
        if kind := llm.rejection(exc):
            keys.mark_failure(key, limited=(kind == "limited"))
            log.warning("provider rejected key %s (%s), benching it", llm.mask(key), kind)

    # ponytail: Ollama's chat timeout is 900s (refine.ollama_chat default); on the live path a slow
    # local model lags subtitles rather than dropping them. Add a shorter live timeout if that bites.
    chat = postmeeting.chat_for(state["llm"], key, max_tokens=1500,
                                model=state["llm"].translate_model)
    if chat is None:
        log.warning("no API key configured — transcribing without translation")
        return None
    return translate.Translator(chat, on_reject=on_reject)


def _refine(session_id: int, wav: Path) -> None:
    """Queue the post-meeting pass for a session that just ended.

    This used to be something a person had to remember. An imported recording was refined on the
    way in (see `import_recording`) while a meeting the room actually captured was not, so the same
    audio produced a better transcript when uploaded through the dashboard than when recorded in
    the room it was built for. Whether a transcript got the large model, offline clustering and the
    per-speaker language pass came down to whether anyone clicked a button.
    """
    def run(cancel: threading.Event) -> None:
        postprocess.rewrite_session(store, session_id, wav, state["cfg"], _make_translator(),
                                    should_stop=cancel.is_set, gpu=jobs.borrow_gpu)

    # The LLM stages ride as a followup so the GPU gate is already released while they run —
    # a meeting can start recording while this session is still being corrected and summarized.
    llm_stages = postmeeting.followup(store, state["cfg"].languages, state["llm"], _api_key(),
                                      session_id)
    # needs_gpu=False for the same reason as /reprocess: `run` takes the card around its decode,
    # and holding it here too would be a second acquire of a one-slot semaphore.
    if not jobs.schedule(session_id, run, followup=llm_stages, needs_gpu=False):
        log.info("session %d is already being refined", session_id)


def _stop_capture(refine: bool = True) -> dict:
    rec, loop, pipe = state["recorder"], state["loopback"], state["pipeline"]
    session_id, holds_gpu = state["session"], state["gpu"]
    path = rec.stop() if rec else None
    # The loopback channel, if any, is stopped too — its (source, None) sentinel is what lets the
    # pipeline's per-channel end count complete, so pipe.join() below would otherwise hang.
    if loop:
        loop.stop()
    if pipe:
        pipe.join()
    if session_id is not None:
        store.end_session(session_id, time.strftime("%Y-%m-%dT%H:%M:%S"))
    state.update(recorder=None, loopback=None, pipeline=None, session=None, gpu=False)
    # Released before scheduling, or the pass would wait on a gate this thread still holds.
    if holds_gpu:
        jobs.release_gpu()
    # The session row, not the recorder's return value: a stop that fails to hand back a path would
    # otherwise skip the refine silently, which is the exact failure this whole change removes.
    if refine and session_id is not None:
        session = store.session(session_id)
        wav = config.recording_path(session["wav_path"]) if session else None
        if wav and wav.exists():
            _refine(session_id, wav)
        else:
            log.warning("session %d has no recording on disk, not refining", session_id)
    return {"recording": False, "path": str(path) if path else None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    hub.bind(asyncio.get_running_loop())
    yield
    # No refine on the way out: the worker is a daemon thread, so scheduling one here would either
    # be killed halfway or hold the process open past the point the user asked it to stop.
    _stop_capture(refine=False)
    jobs.cancel_all(wait=2.0)
    store.close()


app = FastAPI(title="PolyMinutes", lifespan=lifespan)

# The Vite dev server runs on its own port; the packaged app is same-origin so this is dev-only.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:2886", "http://127.0.0.1:2886"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Imported here, not at the top: each router reads `main.store` and friends, so they have to exist
# before the modules are loaded.
from . import (  # noqa: E402
    routes_ask, routes_capture, routes_core, routes_glossary, routes_llm, routes_sessions,
    routes_speakers,
)

for _r in (routes_core, routes_glossary, routes_llm, routes_sessions, routes_speakers,
           routes_capture, routes_ask):
    app.include_router(_r.router)

# The e2e suite checks this shape directly rather than through an endpoint, because the two rerun
# outcomes it guards are reached by way of a GPU decode.
_transcript = routes_sessions._transcript


# ── unknown API paths ───────────────────────────────────────────────────

# Detail carried by the 404 for an /api path this build has no route for, so a caller can tell it
# from "that endpoint says no" — the same status, opposite problems. Read by the dashboard to say
# "restart the backend" instead of reporting the absence of data that is actually there.
# Changing this string changes a contract; both sides assert it.
NO_SUCH_ENDPOINT = "no such endpoint in this build"


# Registered after every real router, and unconditionally: this is an API guarantee, and hanging it
# off the static-dashboard block made it depend on whether the frontend had been built — true when
# serving the bundled app, false on a bare API deployment and in CI.
@app.api_route("/api/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
def unknown_api(path: str) -> None:
    raise HTTPException(404, NO_SUCH_ENDPOINT)


# ── static dashboard ────────────────────────────────────────────────────

if DIST.is_dir():
    app.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
    def spa(path: str) -> FileResponse:
        """Client-side routing: unknown paths return index.html, real files are served as-is."""
        # /api paths never reach here — unknown_api above claims them, for every method, so a
        # stale endpoint answers a named 404 rather than the HTML shell with status 200 (which
        # surfaced as a JSON parse error) or a bare 405 from falling through to a GET-only route.
        candidate = DIST / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(DIST / "index.html")

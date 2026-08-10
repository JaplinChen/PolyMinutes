"""Cross-meeting questions. Its own module because the answer is not about one session.

Every other sessions endpoint is scoped to a meeting the caller names. This one searches all of
them, so it does not belong under `/sessions/{id}` and does not fit the shape of `routes_sessions`.
"""

from __future__ import annotations

import logging
import threading
import time

from fastapi import APIRouter, HTTPException

from . import ask as ask_mod, main

log = logging.getLogger("polyminutes.ask")

router = APIRouter()

# One question at a time, and a floor between them. Unlike /summarize there is no session id to
# deduplicate on and no job to poll — the caller is waiting for the answer — so the protection is
# a plain gate. Both exist because this endpoint has no authentication in front of it and each
# question costs two model calls: real money on the cloud path, minutes of the machine on Ollama.
_gate = threading.Semaphore(1)
_last_answered = 0.0


def _index_rows() -> list[dict]:
    """The tier-one index: what each meeting was about, small enough to send in one prompt.

    Built from the summaries rather than from a second index of its own. They are already
    maintained, already regenerated when a transcript changes, and already the short form of a
    meeting — a separate retrieval index would be the same information kept in step twice.
    """
    import json as _json

    summaries = main.store.summaries()
    rows = []
    for session in main.store.sessions():
        row = summaries.get(session["id"])
        entry = {"id": session["id"], "started": session["started"], "title": "", "decisions": []}
        if row and row["status"] != "failed":
            try:
                per_language = _json.loads(row["json"])
            except ValueError:
                per_language = {}
            # Any language will do for picking a meeting; the first is as good as another, and
            # sending all of them would triple the index for no extra discrimination.
            first = next(iter(per_language.values()), {})
            entry["title"] = first.get("title", "")
            entry["decisions"] = first.get("decisions", [])
        rows.append(entry)
    # Newest first, then capped: a question about "last week" is answered by recent meetings, and
    # an index that grows without bound is the thing that breaks first as the room fills up.
    return rows[: ask_mod.INDEX_SESSIONS]


@router.post("/api/ask")
def ask_question(body: dict) -> dict:
    question = str(body.get("question", "")).strip()
    if not question:
        raise HTTPException(400, "no question")
    if len(question) > ask_mod.MAX_QUESTION_CHARS:
        raise HTTPException(400, f"question is longer than {ask_mod.MAX_QUESTION_CHARS} characters")

    llm_cfg = main.state["llm"]
    provider = llm_cfg.provider
    budget = ask_mod.budget_for(provider)
    # Through main, not a direct import: the e2e suite swaps main.postmeeting for a stub, the same
    # way it swaps main.store, and a captured module reference would reach past it to the real one.
    chat = main.postmeeting.chat_for(llm_cfg, main._api_key(), max_tokens=2000)
    if chat is None:
        raise HTTPException(503, "no language model configured — set one on the LLM settings page")

    if not _gate.acquire(blocking=False):
        raise HTTPException(429, "another question is being answered — try again in a moment")
    try:
        global _last_answered
        reached_model = False
        wait = ask_mod.ASK_COOLDOWN_SECONDS - (time.time() - _last_answered)
        if wait > 0:
            # Not counted against the cooldown: a request bounced before it spends a model call
            # has cost nothing, so bouncing it must not push the window further out.
            raise HTTPException(429, f"try again in {wait:.0f}s")

        def search(keywords, since, until):
            hits: dict[int, int] = {}
            for row in main.store.search_lines(keywords, since or "", until or ""):
                hits[row["session_id"]] = hits.get(row["session_id"], 0) + 1
            return hits

        def load_lines(session_ids):
            out = []
            for sid in session_ids:
                out += [ask_mod.AskLine(r["id"], sid, r["start"], r["speaker"], r["source"])
                        for r in main.store.lines(sid)]
            return out

        names: dict[str, str] = {}
        for session in main.store.sessions():
            names.update(main.store.speaker_names(session["id"]))

        # From here the model is called; before this nothing has cost anything.
        reached_model = True
        try:
            result = ask_mod.ask(question, chat, search, _index_rows(), load_lines, names, provider)
        except ValueError as exc:
            # The model answered with something unusable twice. That is a failure worth naming,
            # not a 500 that says nothing about which half of the pipeline gave up.
            log.exception("could not answer %r", question[:80])
            raise HTTPException(502, f"the model did not answer usefully: {exc}") from exc
        return {**result, "budget": {"provider": provider, "chars": budget.input_chars,
                                     "sessions": budget.max_sessions}}
    finally:
        # Start the cooldown for anything that reached the model, success or failure. A question the
        # model could not answer still spent two calls, and its failure is often a rate limit that
        # retrying at once makes worse — so it must not be a way around the floor. The early 429s
        # above return before the acquire's try body sets this in motion, so they do not extend it.
        if reached_model:
            _last_answered = time.time()
        _gate.release()

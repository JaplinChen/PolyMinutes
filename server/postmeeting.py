"""The stages of the post-meeting pass that talk to a language model instead of the GPU.

Kept out of `jobs` because they need the store and the LLM configuration, and out of `main`
because they are policy, not routing. The split that matters is the one `jobs.schedule` enforces:
everything here runs as a followup, after the GPU gate is released, because none of it touches
the card — and holding the gate through a minutes-long Ollama call meant the next meeting was
told it could not start recording.

Two stages, landing independently:

refine — the correction pass that until now only existed as a CLI script over exported
transcripts. The guards, the coverage accounting and the prompt were all written and validated
there; this wires the same `Refiner` over the stored lines, so every transcript gets the
context-aware fixes instead of only the ones someone exported and re-imported by hand.

summarize — one structured summary per configured language, generated from the refined lines so
it describes the transcript the reader will actually see.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from typing import Callable

from . import jobs, llm, refine, segment, summarize
from .store import Store

log = logging.getLogger("polyminutes.postmeeting")


def chat_for(llm_cfg: llm.LlmConfig, api_key: str, max_tokens: int,
             num_ctx: int = 8192, num_predict: int | None = None,
             temperature: float = 0.0, repeat_penalty: float | None = None,
             model: str = "") -> Callable[[str], str] | None:
    """The chat callable the configured provider implies. None means no LLM stage can run.

    Follows the provider chosen on the LLM settings page — the settings page offers nine, and every
    one of them has to reach its own endpoint, not Anthropic's. This is the single dispatcher both the
    live translator (`main._make_translator`) and the post-meeting stages route through, so a verified
    OpenAI/Gemini/Groq key translates in a meeting instead of silently going nowhere.

    `num_ctx`/`num_predict`/`temperature`/`repeat_penalty` only reach Ollama: the summary stage
    sends a large transcript excerpt AND wants a long detailed reply, so it raises the window, caps
    the reply, and lifts the temperature with a repetition penalty — greedy decoding on a long CJK
    summary loops on one sentence. The cloud providers size their own context and take `max_tokens`.
    """
    provider = llm_cfg.provider
    endpoint = llm_cfg.endpoint or llm.DEFAULT_ENDPOINTS.get(provider, "")
    # A per-function override (translation vs summary) falls back to the one configured model when empty.
    model = model or llm_cfg.model
    if provider == "ollama":
        return refine.ollama_chat(model, endpoint or llm.DEFAULT_ENDPOINTS["ollama"],
                                  num_ctx=num_ctx, num_predict=num_predict,
                                  temperature=temperature, repeat_penalty=repeat_penalty)
    if not api_key:
        # Privacy mode: the chosen cloud provider has no key, but a local Ollama model is configured,
        # so the summary and refine run on this machine instead of not at all. Configuring the Ollama
        # model is the opt-in — a client interview that must not reach a cloud can leave the key blank
        # and still get a summary, and one with no local model set falls through to no_llm as before.
        return _ollama_fallback(llm_cfg, num_ctx, num_predict, temperature, repeat_penalty)
    if provider == "anthropic":
        return refine.anthropic_chat(api_key, model, max_tokens=max_tokens)
    if provider == "gemini":
        return refine.gemini_chat(api_key, model, endpoint, max_tokens=max_tokens)
    if provider == "azure":
        return refine.azure_chat(api_key, endpoint, max_tokens=max_tokens)
    # openai, groq, mistral, openrouter, nvidia_nim — all speak the OpenAI REST shape.
    return refine.openai_chat(api_key, model, endpoint, max_tokens=max_tokens)


def _ollama_fallback(llm_cfg: llm.LlmConfig, num_ctx: int, num_predict: int | None,
                     temperature: float, repeat_penalty: float | None) -> Callable[[str], str] | None:
    """The local Ollama chat to use when the chosen provider has no key, or None if none is set up.

    Reads the Ollama the user already configured on the settings page (its own model and endpoint),
    never the cloud provider's model name — `claude-opus-5` is not something Ollama can pull. No
    liveness probe: if the daemon is down the stage fails and says so, the same as any other provider
    that cannot be reached, and a probe here would just move that failure a few milliseconds earlier.
    """
    ollama = llm_cfg.providers.get("ollama", {})
    model = ollama.get("model", "")
    if not model:
        return None
    log.info("no key for %s; falling back to local Ollama model %s (privacy mode)",
             llm_cfg.provider, model)
    return refine.ollama_chat(model, ollama.get("endpoint") or llm.DEFAULT_ENDPOINTS["ollama"],
                              num_ctx=num_ctx, num_predict=num_predict,
                              temperature=temperature, repeat_penalty=repeat_penalty)


def _cancellable(chat: Callable[[str], str], cancel: threading.Event) -> Callable[[str], str]:
    """Checked before every model call, so cancellation waits for one chunk, not one transcript."""
    def wrapped(prompt: str) -> str:
        if cancel.is_set():
            raise jobs.Cancelled()
        return chat(prompt)
    return wrapped


def _segment_stage(store: Store, session_id: int, chat: Callable[[str], str] | None) -> None:
    """Join the fragments the VAD cut mid-sentence, then have the model restore punctuation.

    Runs before refine, deliberately: the punctuation write must not mark lines refined (it uses
    set_line_source, which doesn't), and refine reads whole sentences better than fragments. The
    merge is pure arithmetic and runs even with no LLM configured; only punctuation needs `chat`.
    """
    rows = store.lines(session_id)
    joined = 0
    for group in segment.merge_groups(rows):
        keep, absorbed = rows[group[0]], [rows[i] for i in group[1:]]
        text, end_time, translations = segment.join(keep, absorbed)
        store.merge_lines(keep["id"], [r["id"] for r in absorbed], text, end_time, translations)
        joined += len(absorbed)
    if chat is None:
        log.info("segment stage: %d fragments joined, no LLM for punctuation", joined)
        return
    if joined:
        rows = store.lines(session_id)
    punctuated = 0
    jobs.set_progress(session_id, 0, len(rows))
    for start in range(0, len(rows), refine.CHUNK_LINES):
        chunk = rows[start : start + refine.CHUNK_LINES]
        texts = [r["source"] for r in chunk]
        out = None
        for attempt in (1, 2):
            try:
                out = segment.parse_response(chat(segment.build_prompt(texts)), texts)
                break
            except jobs.Cancelled:
                raise
            except Exception:
                log.exception("punctuation failed at line %d (attempt %d)", start, attempt)
        jobs.set_progress(session_id, done=min(start + refine.CHUNK_LINES, len(rows)),
                          total=len(rows))
        if out is None:
            jobs.set_progress(session_id, skipped_add=len(chunk))
            continue
        for row, text in zip(chunk, out):
            # A human-corrected line's punctuation is the human's choice.
            if text != row["source"] and not row["refined"]:
                store.set_line_source(row["id"], text)
                punctuated += 1
    log.info("segment stage: %d fragments joined, %d lines punctuated", joined, punctuated)


def _refine_stage(store: Store, session_id: int, chat: Callable[[str], str]) -> None:
    rows = store.lines(session_id)
    if not rows:
        return
    lines = [refine.Line(r["speaker"], r["lang"], r["source"]) for r in rows]
    coverage = refine.Coverage()
    corrected = refine.Refiner(chat).refine(
        lines, terms=store.glossary(), coverage=coverage,
        on_progress=lambda done, total: jobs.set_progress(session_id, done=done, total=total))
    jobs.set_progress(session_id, skipped_add=coverage.skipped)
    changed = 0
    for row, text in zip(rows, corrected):
        # A refined line is never rewritten twice (update_line's own promise): a human correction is
        # ground truth this pass must not clobber, and an earlier pass's own output stays put. Refined
        # lines still go to the model above as read-only context, they just are not written back.
        if text != row["source"] and not row["refined"]:
            # update_line, not replace_line: this is the one writer entitled to mark `refined`,
            # and it must not touch status — a line the decoder failed on stays visibly failed.
            store.update_line(row["id"], text, {})
            changed += 1
    log.info("refine stage: %d/%d lines corrected, %.0f%% of the transcript checked",
             changed, len(rows), (1 - coverage.fraction) * 100)


def _summarize_stage(store: Store, session_id: int, languages: list[str],
                     llm_cfg: llm.LlmConfig, api_key: str, cancel: threading.Event) -> None:
    # Read after the refine stage so the summary describes the transcript the reader will see, and
    # capture the revision the SAME read observed — one lock hold, so an edit cannot slip between
    # the content and the number and leave a stale summary stamped current.
    rows, rev = store.lines_with_rev(session_id)
    if not rows:
        return

    lines = [summarize.SummaryLine(r["speaker"], r["lang"], r["source"]) for r in rows]
    target = summarize.target_chars(sum(len(l.text) for l in lines))
    # Room for the transcript excerpt (INPUT_BUDGET chars) plus a long detailed reply; the default
    # 8192 leaves no output budget and Ollama answers with a single terse line. num_predict caps
    # the reply so it stops instead of generating until the window fills.
    mt = summarize.max_tokens_for(target)
    chat = chat_for(llm_cfg, api_key, max_tokens=mt, num_ctx=16384, num_predict=mt,
                    temperature=0.5, repeat_penalty=1.2, model=llm_cfg.summary_model)
    if chat is None:
        # No LLM configured. Record it rather than returning silently: the card otherwise shows
        # "no summary yet" forever, indistinguishable from never-clicked, hiding that the fix is a
        # settings change, not a retry.
        store.set_summary(session_id, "{}", "no_llm", rev,
                          time.strftime("%Y-%m-%dT%H:%M:%S"))
        return

    session = store.session(session_id)
    reference = session.get("reference", "") if session else ""
    depts = store.speaker_departments()
    speakers = {code: f"{name} ({depts[name]})" if depts.get(name) else name
                for code, name in store.speaker_names(session_id).items() if name}
    result, status = summarize.summarize(lines, languages, _cancellable(chat, cancel),
                                         speakers=speakers, should_stop=cancel.is_set,
                                         reference=reference)
    # Landed even when partial: two of three languages beats none, the card says so, and
    # regenerating later is one click. Nothing at all came back → failed is still worth storing,
    # because "tried and failed" and "never ran" are different answers to "where is my summary".
    store.set_summary(session_id, json.dumps(result, ensure_ascii=False), status, rev,
                      time.strftime("%Y-%m-%dT%H:%M:%S"))


def followup(store: Store, languages: list[str], llm_cfg: llm.LlmConfig, api_key: str,
             session_id: int) -> Callable[[threading.Event, Callable[[str], None]], None]:
    """The post-GPU stages, as one callable for `jobs.schedule`.

    Stages land independently: a summary that fails does not undo a refine that succeeded, which
    is why each writes to the store as it finishes rather than at the end.
    """
    def run(cancel: threading.Event, set_stage: Callable[[str], None]) -> None:
        # Refine corrects the transcript's own words, a Traditional-Chinese comprehension job closer to
        # the summary than to translation, so it rides the summary model rather than the translator's.
        chat = chat_for(llm_cfg, api_key, max_tokens=4000, model=llm_cfg.summary_model)

        set_stage("segment")
        if cancel.is_set():
            raise jobs.Cancelled()
        _segment_stage(store, session_id, _cancellable(chat, cancel) if chat else None)

        # Refine needs a model; with none configured it is simply skipped. Summarize still runs,
        # because _summarize_stage records a "no_llm" state that the session card reads as "configure
        # an LLM" — returning here instead left the card on the generic "no summary yet", the exact
        # ambiguity that state exists to remove, until someone clicked generate by hand.
        if chat is not None:
            set_stage("refine")
            if cancel.is_set():
                raise jobs.Cancelled()
            _refine_stage(store, session_id, _cancellable(chat, cancel))
        else:
            log.warning("no LLM configured — transcript stays unrefined")

        set_stage("summarize")
        if cancel.is_set():
            raise jobs.Cancelled()
        _summarize_stage(store, session_id, languages, llm_cfg, api_key, cancel)

        if cancel.is_set():
            raise jobs.Cancelled()

    return run

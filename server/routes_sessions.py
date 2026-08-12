"""Recorded sessions: transcripts, corrections, re-derivation and import."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from pathlib import Path

import soundfile as sf
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from . import asr, asr_gpu, config, correct, ingest, jobs, main, routes_speakers, translate

log = logging.getLogger("polyminutes")

router = APIRouter()

# Longest single utterance a rerun will decode. VAD cuts at 20 s, so anything past this is a row
# whose end_time is wrong rather than a real utterance, and decoding it would tie up the card.
RERUN_MAX_SECONDS = 60.0
RERUN_PAD_SECONDS = 0.5


@router.get("/api/sessions")
def get_sessions() -> list[dict]:
    """Sessions, each carrying where its post-meeting pass got to.

    Carried on the list rather than fetched per session so the page can show which meetings are
    still being refined before the user picks one, instead of after.
    """
    running = jobs.states()
    # Whether the recording is still on disk. Everything that reads audio — playing a line, hearing
    # a speaker, re-deriving the transcript — fails without it, and the page could only find out by
    # trying: 943 play buttons that each produce the same error is not a way to learn that.
    return [{**s, "hasRecording": config.recording_path(s["wav_path"]).is_file(),
             "refine": running.get(s["id"], {"state": "idle", "error": ""})}
            for s in main.store.sessions()]


@router.get("/api/sessions/{session_id}/lines")
def get_lines(session_id: int) -> dict:
    return {"lines": main.store.lines(session_id), "speakers": main.store.speaker_names(session_id)}


@router.put("/api/sessions/{session_id}/reference")
def put_reference(session_id: int, body: dict) -> dict:
    """Store pre-meeting notes for this session. Folded into the summary prompt when it regenerates."""
    if not main.store.session(session_id):
        raise HTTPException(404, "no such session")
    reference = str(body.get("reference", ""))
    main.store.set_reference(session_id, reference)
    return {"reference": reference}


@router.put("/api/sessions/{session_id}/lines/{line_id}")
def put_line(session_id: int, line_id: int, body: dict) -> dict:
    """Correct one transcript line, and learn the pair.

    The edit is the only ground truth this system ever sees — someone who was in the room saying
    what was actually said. Storing the before/after means the same mistake is fixed automatically
    everywhere it appears next time, live as well as after the fact.
    """
    source = str(body.get("source", "")).strip()
    if not source:
        raise HTTPException(400, "source required")

    before = next((l for l in main.store.lines(session_id) if l["id"] == line_id), None)
    if before is None:
        raise HTTPException(404, "no such line in this session")

    # The old translations were made from the wrong words, so keeping them would leave the meeting
    # saying one thing in the source and another in every other language.
    translations, status = _translate(source, before["lang"], before["speaker"], line_id)
    main.store.replace_line(line_id, source, before["lang"],
                            translations or before["translations"], status, refined=True)
    for wrong, right in correct.diff_terms(before["source"], source):
        main.store.add_correction(wrong, right, before["lang"])
    return _transcript(session_id, status)


@router.put("/api/sessions/{session_id}/lines/{line_id}/speaker")
def put_line_speaker(session_id: int, line_id: int, body: dict) -> dict:
    """Reassign one line to a different speaker.

    The clustering flattens a shared-mic room into one voice more often than not, and language is
    picked per speaker — so a wrong attribution also decodes the line in the wrong language. This is
    the human splitting the collapsed speaker back apart. The code can be one the meeting already
    has or a fresh S-code the caller minted; either is just a label until someone names it.
    """
    speaker = str(body.get("speaker", "")).strip()
    if not speaker:
        raise HTTPException(400, "speaker required")

    before = next((l for l in main.store.lines(session_id) if l["id"] == line_id), None)
    if before is None:
        raise HTTPException(404, "no such line in this session")

    main.store.set_line_speaker(line_id, speaker)
    return {"lines": main.store.lines(session_id), "speakers": main.store.speaker_names(session_id)}


@router.get("/api/corrections")
def get_corrections() -> list[dict]:
    return [{"wrong": w, "right": r} for w, r in main.store.corrections().items()]


@router.put("/api/corrections/{wrong}")
def put_correction(wrong: str, body: dict) -> list[dict]:
    try:
        main.store.edit_correction(wrong, str(body.get("wrong", wrong)), str(body.get("right", "")))
    except KeyError as exc:
        raise HTTPException(404, f"no correction for {wrong}") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return get_corrections()


@router.delete("/api/corrections/{wrong}")
def delete_correction(wrong: str) -> list[dict]:
    main.store.forget_correction(wrong)
    return get_corrections()


@router.delete("/api/sessions/{session_id}")
def delete_session(session_id: int) -> dict:
    """Remove a meeting: the database rows and the recording on disk.

    Refused while the meeting is recording or refining — a background pass would be writing lines
    into a session that no longer exists. The wav goes too: without its session nothing can ever
    reach it again, and a recording nobody can play is just disk the room quietly loses.
    """
    session = main.store.session(session_id)
    if not session:
        raise HTTPException(404, "no such session")
    if main.state["session"] == session_id:
        raise HTTPException(409, "session is still recording")
    if (jobs.state(session_id) or {}).get("state") == "refining":
        raise HTTPException(409, "session is being refined — wait for it to finish")

    # Before the rows and the wav go: an identified voice keeps a playable sample even after every
    # meeting it spoke in is deleted. Only forgetting the voice removes these.
    routes_speakers.harvest_speaker_clips(session_id)
    main.store.delete_session(session_id)
    wav = config.recording_path(session["wav_path"])
    try:
        wav.unlink(missing_ok=True)
    except OSError:
        # The rows are gone either way; a wav Windows still holds open is lost disk, not a failure.
        log.warning("could not remove recording %s", wav)
    return {"deleted": session_id}


@router.post("/api/sessions/{session_id}/reprocess")
def reprocess(session_id: int) -> dict:
    """Queue a re-derivation from the recording, with the largest model and offline clustering.

    Queued rather than run inline, and through the same gate as the automatic pass: two of these at
    once would put two Whisper models on one card, and the second would be rewriting a transcript
    the first is already rewriting.
    """
    session = main.store.session(session_id)
    if not session:
        raise HTTPException(404, "no such session")
    if main.state["session"] == session_id:
        raise HTTPException(409, "session is still recording")

    wav = config.recording_path(session["wav_path"])
    if not wav.is_file():
        raise HTTPException(404, f"recording not found: {wav}")

    # needs_gpu=False because the pass takes the card itself, around the decode alone. Scheduling
    # it under the gate instead would hold the card through the VAD, the speaker segmentation and
    # every translation round trip — eight minutes of it on a long recording — and taking the gate
    # in both places at once is a deadlock, not a double-guard.
    if not jobs.schedule(
            session_id,
            lambda cancel: main.postprocess.rewrite_session(
                main.store, session_id, wav, main.state["cfg"], main._make_translator(),
                should_stop=cancel.is_set, gpu=jobs.borrow_gpu),
            followup=main.postmeeting.followup(main.store, main.state["cfg"].languages,
                                               main.state["llm"], main._api_key(), session_id),
            needs_gpu=False):
        raise HTTPException(409, "already refining this session")
    return {"session": session_id, **(jobs.state(session_id) or {})}


# Minutes between summary regenerations of an unchanged transcript. The endpoint has no
# authentication, and each call spends real money or minutes of local compute; a stale summary is
# exempt because regenerating one is the entire point of tracking staleness.
SUMMARY_COOLDOWN_MINUTES = 5


def _summary_body(session_id: int) -> dict:
    """One shape for both summary endpoints, stale computed at read time."""
    row = main.store.summary(session_id)
    session = main.store.session(session_id)
    job = jobs.state(session_id) or {}
    generating = job.get("state") == "refining"
    if not row:
        return {"session": session_id, "state": "generating" if generating else "none",
                "stale": False, "summary": None}
    stale = bool(session) and int(session["lines_rev"]) != int(row["lines_rev"])
    return {"session": session_id,
            "state": "generating" if generating else row["status"],
            "stale": stale,
            "created": row["created"],
            "summary": json.loads(row["json"])}


@router.get("/api/sessions/{session_id}/summary")
def get_summary(session_id: int) -> dict:
    if not main.store.session(session_id):
        raise HTTPException(404, "no such session")
    return _summary_body(session_id)


@router.post("/api/sessions/{session_id}/summarize")
def summarize_session(session_id: int) -> dict:
    """Regenerate the summary alone — no ASR, no GPU, just the LLM stages over stored lines."""
    if not main.store.session(session_id):
        raise HTTPException(404, "no such session")
    if main.state["session"] == session_id:
        raise HTTPException(409, "session is still recording")

    # One read for the summary and the revision: apart, an edit between them can move the revision
    # under this comparison and make an unchanged verdict against a transcript that already differs.
    existing, rev = main.store.summary_and_rev(session_id)
    if existing and rev == int(existing["lines_rev"]):
        age = time.time() - time.mktime(time.strptime(existing["created"], "%Y-%m-%dT%H:%M:%S"))
        if existing["status"] == "ok" and age < SUMMARY_COOLDOWN_MINUTES * 60:
            raise HTTPException(
                429, f"this summary is {int(age)}s old and the transcript has not changed — "
                     f"edit a line first, or wait {SUMMARY_COOLDOWN_MINUTES} minutes")

    def stages(cancel, set_stage):
        set_stage("summarize")
        main.postmeeting._summarize_stage(main.store, session_id, main.state["cfg"].languages,
                                          main.state["llm"], main._api_key(), cancel)

    # Through the job registry for its dedup — two generations must not race over one session — but
    # needs_gpu=False: this is the LLM alone over stored lines, and entering the GPU gate would make
    # it wait behind a meeting recording on another session, for work that never touches the card.
    if not jobs.schedule(session_id, lambda cancel: None, followup=stages, needs_gpu=False):
        raise HTTPException(409, "already refining this session")
    return _summary_body(session_id)


def _translate(text: str, lang: str, speaker: str, line_id: int) -> tuple[dict[str, str], str]:
    """Translate one line into every other configured language. Never raises."""
    translator = main._make_translator()
    targets = [c for c in main.state["cfg"].languages if c != lang]
    if not translator or not targets:
        return {}, "ok"
    try:
        return translator.translate(
            translate.Line(text=text, lang=lang, speaker=speaker),
            targets, terms=main.store.glossary()).translations, "ok"
    except Exception:
        log.exception("translation failed for line %d", line_id)
        return {}, "translate_failed"


def _transcript(session_id: int, status: str) -> dict:
    """The shape every transcript-mutating endpoint returns.

    One helper rather than a literal per exit: the two rerun outcomes drifted apart once already,
    one returning `line` where the other returned `lines`, which the page reads straight into
    state — so a failed rerun blanked the transcript it was supposed to be fixing.
    """
    return {"lines": main.store.lines(session_id), "speakers": main.store.speaker_names(session_id),
            "status": status}


@router.post("/api/sessions/{session_id}/lines/{line_id}/retranslate")
def retranslate_line(session_id: int, line_id: int) -> dict:
    """Translate one line again from the text already on screen. No audio, no GPU."""
    line = main.store.line(line_id)
    if not line or line["session_id"] != session_id:
        raise HTTPException(404, "no such line")

    translations, status = _translate(line["source"], line["lang"], line["speaker"], line_id)
    main.store.replace_line(line_id, line["source"], line["lang"], translations, status)
    return _transcript(session_id, status)


def _mostly_glossary(text: str, terms: list) -> bool:
    """True when stripping every glossary term leaves less than a third of the text."""
    stripped = text
    for term in terms:
        stripped = stripped.replace(term.source, "")
    kept = len(stripped.replace(" ", ""))
    total = len(text.replace(" ", ""))
    return total > 0 and kept / total < 1 / 3


@router.post("/api/sessions/{session_id}/lines/{line_id}/rerun")
def rerun_line(session_id: int, line_id: int) -> dict:
    """Decode and translate one line again from the recording.

    The audio comes from the session row, never from the request: the caller names a line, not a
    path or an offset, so there is nothing here to point at another file. The work is bounded by
    the line's own span and takes the same GPU gate as a full pass, because this endpoint has no
    authentication in front of it and a loop over it would otherwise starve a live meeting.
    """
    if main.state["session"] == session_id:
        raise HTTPException(409, "session is still recording")
    session = main.store.session(session_id)
    line = main.store.line(line_id)
    if not session or not line or line["session_id"] != session_id:
        raise HTTPException(404, "no such line")

    wav = config.recording_path(session["wav_path"])
    if not wav.is_file():
        raise HTTPException(404, f"recording not found: {wav}")

    # Half a second of padding on both sides: the live VAD cuts on silence, which clips the
    # opening and closing consonants, and a one-second slice with both ends shaved is the
    # main reason a re-run comes back empty.
    start = max(float(line["start"]) - RERUN_PAD_SECONDS, 0.0)
    end = line["end_time"] if line["end_time"] is not None else start + RERUN_MAX_SECONDS
    seconds = min(max(float(end) + RERUN_PAD_SECONDS - start, 0.0), RERUN_MAX_SECONDS)
    if seconds <= 0:
        raise HTTPException(400, "line has no duration to re-run")

    with jobs.borrow_gpu():
        try:
            # Only this line's span is read, not the whole meeting: a ninety-minute wav does not
            # belong in memory to re-decode four seconds of it.
            samples, rate = sf.read(str(wav), dtype="float32", start=int(start * config.SAMPLE_RATE),
                                    frames=int(seconds * config.SAMPLE_RATE), always_2d=False)
        except Exception as exc:
            raise HTTPException(400, f"could not read the recording: {exc}") from exc
        if rate != config.SAMPLE_RATE:
            raise HTTPException(400, f"{wav.name} is {rate} Hz, expected {config.SAMPLE_RATE}")
        if getattr(samples, "ndim", 1) > 1:
            samples = samples.mean(axis=1)

        transcriber = (asr_gpu.maybe(main.state["cfg"].languages,
                                     asr_gpu.hotwords_from(main.store.glossary()))
                       or asr.Transcriber(model_dir=main.postprocess.best_model(), quantized=False,
                                          languages=main.state["cfg"].languages))
        text, used = transcriber.transcribe(samples, line["lang"] or "")
        if not text and line["lang"]:
            # A forced language on a mumbled clip can decode to nothing; one auto-detect
            # retry before declaring the line unrecognisable.
            text, used = transcriber.transcribe(samples, "")
        # Near-silent audio makes the decoder hallucinate the glossary hotwords back at
        # us — from either attempt — so a result that is mostly glossary terms is
        # discarded rather than saved as if someone had said it.
        if text and _mostly_glossary(text, main.store.glossary()):
            text, used = "", ""

    if not text:
        # An empty source, not the old text: what was there before a failed re-run is either a
        # bad guess or a hallucination, and showing it under a 未能辨識 badge reads as if
        # someone said it.
        main.store.replace_line(line_id, "", line["lang"], {}, "asr_failed")
        return _transcript(session_id, "asr_failed")

    text = correct.Corrector(main.store.glossary(), main.store.corrections()).fix(text)
    translations, status = _translate(text, used or line["lang"], line["speaker"], line_id)
    main.store.replace_line(line_id, text, used or line["lang"], translations, status)
    return _transcript(session_id, status)


@router.get("/api/sessions/{session_id}/refine")
def refine_state(session_id: int) -> dict:
    """Where the post-meeting pass got to. `idle` means there has not been one this run.

    The dashboard reads refine state off the session list instead — carried there so it can show
    which meetings are still being refined before you pick one. This is the per-session probe the
    e2e suite waits on, which is why it has no browser caller and is not dead.
    """
    return {"session": session_id, **(jobs.state(session_id) or {"state": "idle", "error": ""})}


def _import_slot() -> tuple[str, Path]:
    """A unique import-<tag> name, the wav reserved on disk at once.

    Reserved by creating it, not by checking: a URL import only writes its wav minutes after
    picking the name, and two imports inside the same second would otherwise share one — the first
    session would end up pointing at the second one's audio.
    """
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag, n = stamp, 1
    while True:
        wav = config.RECORDINGS_DIR / f"import-{tag}.wav"
        try:
            wav.touch(exist_ok=False)
            return tag, wav
        except FileExistsError:
            tag, n = f"{stamp}-{n}", n + 1


@router.post("/api/sessions/import")
async def import_recording(request: Request, filename: str = "upload") -> dict:
    """Learn from a meeting that was recorded somewhere else.

    The upload becomes an ordinary session, so everything the room learns from a live capture — a
    voice once someone names it, a correction once someone fixes a line — is learned from a file
    the same way. Nothing downstream is told it came from an upload.
    """
    # ponytail: raw body rather than multipart, so no python-multipart dependency. Streamed to
    # disk because a meeting recording does not belong in memory.
    stem = re.sub(r"[^\w.-]", "_", Path(filename).name).strip("._") or "upload"
    tag, wav = _import_slot()
    source = config.RECORDINGS_DIR / f"import-{tag}-{stem}"

    written = 0
    with source.open("wb") as out:
        async for chunk in request.stream():
            written += out.write(chunk)
    if not written:
        source.unlink(missing_ok=True)
        wav.unlink(missing_ok=True)
        raise HTTPException(400, "empty upload")

    try:
        # ffmpeg is a blocking subprocess; off the event loop or the whole server — every live
        # meeting's socket included — freezes for the length of the transcode.
        await asyncio.to_thread(ingest.extract_audio, source, wav)
    except ValueError as exc:
        # ffmpeg creates the output before it discovers it cannot read the input.
        wav.unlink(missing_ok=True)
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
    finally:
        # The original video is not evidence — every stage after this reads the wav.
        source.unlink(missing_ok=True)

    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    session_id = main.store.start_session(now, str(wav))
    main.store.end_session(session_id, now)
    # Queued like a meeting that just ended, not decoded inline: a long recording used to hold this
    # request open for the whole pass — past most proxies' timeout — with nothing on screen saying
    # why. Through `_refine` the response returns as soon as the session exists, the dashboard's
    # refine chip tracks the pass, and an import gets the same LLM correction and summary stages a
    # recorded meeting does instead of stopping at the rewrite.
    main._refine(session_id, wav)
    return {"id": session_id, **(jobs.state(session_id) or {})}


@router.post("/api/sessions/import-url")
def import_url(body: dict) -> dict:
    """Import a meeting from a link — YouTube, a shared recording, anywhere yt-dlp can reach.

    The download can take minutes, so the whole chain — fetch, extract, rewrite — runs as the
    session's refine job: the response returns as soon as the session exists and the refine chip
    tracks it, exactly like a file import. A failed download is a failed job on the chip, with
    yt-dlp's own last line as the reason.
    """
    url = str(body.get("url", "")).strip()
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "a http(s) URL is required")
    # Checked here rather than discovered by the job: a missing tool is the operator's problem to
    # fix now, not a failure to find on a chip minutes later.
    if not ingest.have_downloader():
        raise HTTPException(503, "yt-dlp is not installed — pip install yt-dlp")

    tag, wav = _import_slot()
    source = config.RECORDINGS_DIR / f"import-{tag}.download"
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    session_id = main.store.start_session(now, str(wav))
    main.store.end_session(session_id, now)

    def run(cancel):
        try:
            ingest.download_audio(url, source)
            ingest.extract_audio(source, wav)
        finally:
            # The download is not evidence — every stage after this reads the wav.
            source.unlink(missing_ok=True)
        main.postprocess.rewrite_session(main.store, session_id, wav, main.state["cfg"],
                                         main._make_translator(), should_stop=cancel.is_set,
                                         gpu=jobs.borrow_gpu)

    # Same wiring as `_refine`, plus the fetch in front; needs_gpu=False for the same reason —
    # `rewrite_session` takes the card itself, around the decode alone.
    jobs.schedule(session_id, run,
                  followup=main.postmeeting.followup(main.store, main.state["cfg"].languages,
                                                     main.state["llm"], main._api_key(), session_id),
                  needs_gpu=False)
    return {"id": session_id, **(jobs.state(session_id) or {})}


@router.get("/api/sessions/{session_id}/markdown")
def session_markdown(session_id: int) -> PlainTextResponse:
    if not main.store.session(session_id):
        raise HTTPException(404, "no such session")
    return PlainTextResponse(main.postprocess.to_markdown(main.store, session_id),
                             media_type="text/markdown")


@router.get("/api/sessions/{session_id}/docx")
def session_docx(session_id: int) -> Response:
    """The same export as markdown, as a Word document — what an enterprise meeting hands over."""
    if not main.store.session(session_id):
        raise HTTPException(404, "no such session")
    return Response(main.postprocess.to_docx(main.store, session_id),
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document")

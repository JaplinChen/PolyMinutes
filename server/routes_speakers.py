"""Speaker names, per session and across the ones already recorded."""

from __future__ import annotations

import io

import soundfile as sf
from fastapi import APIRouter, HTTPException, Response

from . import config, main

router = APIRouter()

# A line whose end_time is wrong (or missing, on older rows) must not turn into a request for the
# rest of the recording. Long enough for any real utterance; short enough to stay a clip.
MAX_LINE_SECONDS = 60.0


@router.put("/api/sessions/{session_id}/speakers")
def put_speaker_names(session_id: int, body: dict) -> dict:
    previous = main.store.speaker_names(session_id)
    for code, name in body.items():
        code, name = str(code), str(name).strip()
        main.store.set_speaker_name(session_id, code, name)
        centroid = main.store.voiceprint(session_id, code)
        if centroid is None:
            continue
        # Correcting a wrong name is the correction, not just a new label: the print that pulled
        # this voice onto the old name is what will misname it again next meeting.
        if (old := previous.get(code, "")) and old != name:
            main.store.unlearn_speaker(old, centroid)
        # Naming a speaker is the only labelled data this system ever gets. Attaching it to the
        # voiceprint is what stops the next meeting asking the same question.
        if name:
            main.store.remember_speaker(name, centroid)
    return main.store.speaker_names(session_id)


@router.post("/api/sessions/{session_id}/speakers/merge")
def merge_speakers(session_id: int, body: dict) -> dict:
    """Fold several codes for one person into one, and return the redrawn transcript.

    Same response shape as reassigning a single line, because the speaker page and the transcript
    both read it: the absorbed codes leave the meeting and their lines now carry the kept one.
    """
    into = str(body.get("into", "")).strip()
    sources = [str(c).strip() for c in body.get("from", []) if str(c).strip()]
    if not into or not sources:
        raise HTTPException(400, "into and from required")
    main.store.merge_speakers(session_id, into, sources)
    return {"lines": main.store.lines(session_id), "speakers": main.store.speaker_names(session_id)}


@router.get("/api/speakers/known")
def get_known_speakers() -> list[dict]:
    counts = main.store.speaker_sessions()
    langs = main.store.speaker_languages()
    return [{"name": name, "sessions": counts.get(name, 0), "language": langs.get(name, "")}
            for name, _ in main.store.known_speakers()]


def _clip(sample: tuple[str, float, float | None] | None, seconds: float | None = None) -> Response:
    """A slice of a recording as a WAV, from wherever the sample points at.

    `seconds` defaults to CLIP_SECONDS, which is the right length for "is this the same voice",
    but never runs past the utterance the sample points at: four seconds from the start of a
    one-second answer is three seconds of whoever spoke next, and a sample holding two voices
    cannot tell you whose voice it is. A transcript line passes its own duration instead — the
    question there is whether the text matches what was said, and a sentence cut off at four
    seconds cannot answer it.
    """
    if sample is None:
        raise HTTPException(404, "no recording for this voice")
    wav_path, start, span = sample
    if seconds is None:
        seconds = min(main.CLIP_SECONDS, span) if span else main.CLIP_SECONDS
    wav = config.recording_path(wav_path)
    if not wav.is_file():
        raise HTTPException(404, f"recording not found: {wav}")

    with sf.SoundFile(wav) as f:
        f.seek(min(int(start * f.samplerate), max(len(f) - 1, 0)))
        block = f.read(int(seconds * f.samplerate), dtype="int16")
        rate = f.samplerate
    buf = io.BytesIO()
    sf.write(buf, block, rate, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


@router.get("/api/speakers/known/{name}/clip")
def get_speaker_clip(name: str) -> Response:
    """A few seconds of the voice behind the name, so a wrong match is audible rather than guessed."""
    return _clip(main.store.speaker_sample(name))


@router.get("/api/sessions/{session_id}/speakers/{code}/clip")
def get_session_speaker_clip(session_id: int, code: str) -> Response:
    """The same, for a speaker this meeting has not named yet.

    The one above needs a name to find a voice, which leaves the naming screen — thirty-five boxes
    labelled S1..S35 — as the one place in the app asking a question it gave you nothing to answer.
    """
    return _clip(main.store.session_speaker_sample(session_id, code))


@router.get("/api/sessions/{session_id}/lines/{line_id}/clip")
def get_line_clip(session_id: int, line_id: int) -> Response:
    """What was actually said on this line.

    Correcting a transcript means deciding whether the text matches the audio, and until now the
    audio was the one thing the page did not have. Plays the line's own span, so a long sentence is
    not cut short — capped, because a bad end_time should not stream the rest of the meeting.
    """
    line = main.store.line(line_id)
    if line is None or line["session_id"] != session_id:
        raise HTTPException(404, f"no line {line_id} in session {session_id}")
    session = main.store.session(session_id)
    if session is None:
        raise HTTPException(404, f"no session {session_id}")

    end = line["end_time"]
    span = max(float(end) - float(line["start"]), 0.0) if end is not None else 0.0
    return _clip((session["wav_path"], line["start"], span or None), min(span, MAX_LINE_SECONDS) or None)


@router.put("/api/speakers/known/{name}")
def rename_known_speaker(name: str, body: dict) -> list[dict]:
    new = str(body.get("name", "")).strip()
    if not new:
        raise HTTPException(400, "name required")
    try:
        main.store.rename_speaker(name, new)
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return get_known_speakers()


@router.put("/api/speakers/known/{name}/language")
def set_known_speaker_language(name: str, body: dict) -> list[dict]:
    language = str(body.get("language", "")).strip()
    # '' is auto-detect; anything else must be a language this room actually runs, or a typo would
    # force a recognised speaker into a recogniser that was never built for the meeting.
    allowed = {""} | set(config.load().languages)
    if language not in allowed:
        raise HTTPException(400, f"language must be one of {sorted(allowed)}")
    main.store.set_speaker_language(name, language)
    return get_known_speakers()


@router.delete("/api/speakers/known/{name}")
def delete_known_speaker(name: str) -> list[dict]:
    main.store.forget_speaker(name)
    return get_known_speakers()

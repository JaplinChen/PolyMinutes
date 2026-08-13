"""Speaker names, per session and across the ones already recorded."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Response

from . import config, main, speakers

router = APIRouter()

# A line whose end_time is wrong (or missing, on older rows) must not turn into a request for the
# rest of the recording. Long enough for any real utterance; short enough to stay a clip.
MAX_LINE_SECONDS = 60.0


_extractor = None


def _embed(samples: np.ndarray, rate: int) -> np.ndarray:
    global _extractor
    if _extractor is None:
        import sherpa_onnx
        _extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(config.SPEAKER_MODEL), num_threads=1))
    stream = _extractor.create_stream()
    stream.accept_waveform(rate, samples)
    stream.input_finished()
    return np.array(_extractor.compute(stream), dtype=np.float32)


def _derive_voiceprint(session_id: int, code: str) -> bytes | None:
    """Cut a centroid for a code that has none, from the lines it labels.

    A code minted by reassigning transcript lines never went through the diariser, so naming it
    used to teach nothing. Its lines still point at the recording; that audio is the voiceprint.
    """
    session = main.store.session(session_id)
    if session is None or not session["wav_path"]:
        return None
    wav = config.recording_path(session["wav_path"])
    if not wav.is_file():
        return None
    spans = [(float(l["start"]), float(l["end_time"]) - float(l["start"]))
             for l in main.store.lines(session_id)
             if l["speaker"] == code and l["end_time"] is not None
             and config.MIN_EMBED_SECONDS <= float(l["end_time"]) - float(l["start"]) <= MAX_LINE_SECONDS]
    if not spans:
        return None
    embeddings = []
    with sf.SoundFile(wav) as f:
        for start, span in sorted(spans, key=lambda s: -s[1])[:3]:
            f.seek(min(int(start * f.samplerate), max(len(f) - 1, 0)))
            samples = f.read(int(span * f.samplerate), dtype="float32")
            embeddings.append(_embed(samples, f.samplerate))
    centroid = np.mean(embeddings, axis=0).astype(np.float32).tobytes()
    main.store.save_voiceprint(session_id, code, centroid)
    return centroid


def refresh_voiceprint(session_id: int, code: str) -> None:
    """Re-derive a code's print after its lines changed, and correct what the stale one taught.

    Splitting a collapsed speaker or merging fragments changes which audio a code stands for, but
    its stored voiceprint kept describing the old mix — and if the code was named, that polluted
    print was already learned and would misname people next meeting. The stale learned variant is
    unlearned and the fresh one learned in its place. When derivation cannot run (no wav, no line
    long enough) the old print is left alone: a slightly stale print beats none.
    """
    old = main.store.voiceprint(session_id, code)
    name = main.store.speaker_names(session_id).get(code, "").strip()
    if not any(l["speaker"] == code for l in main.store.lines(session_id)):
        main.store.delete_voiceprint(session_id, code)
        if name and old:
            main.store.unlearn_speaker(name, old)
        return
    fresh = _derive_voiceprint(session_id, code)
    if fresh is None:
        return
    if name:
        if old:
            main.store.unlearn_speaker(name, old)
        main.store.remember_speaker(name, fresh)


@router.put("/api/sessions/{session_id}/speakers")
def put_speaker_names(session_id: int, body: dict) -> dict:
    previous = main.store.speaker_names(session_id)
    for code, name in body.items():
        code, name = str(code), str(name).strip()
        main.store.set_speaker_name(session_id, code, name)
        centroid = main.store.voiceprint(session_id, code) or _derive_voiceprint(session_id, code)
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
    # The fragments that forced this merge are usually the same fragments the VAD cut mid-sentence
    # — a stutter's pieces handed to a phantom code. Now that they share a code, the segment join
    # can finally reach them. Arithmetic only (chat=None): the LLM punctuation pass stays in the
    # refine followup, because a click must not wait on a model.
    main.postmeeting._segment_stage(main.store, session_id, None)
    # The kept code now stands for more audio and the absorbed ones for none; their prints follow.
    refresh_voiceprint(session_id, into)
    for code in sources:
        refresh_voiceprint(session_id, code)
    return {"lines": main.store.lines(session_id), "speakers": main.store.speaker_names(session_id)}


# A resemblance worth mentioning, well under the bar for saying it outright. Between the two, the
# match is a hint for a human holding the audio; below the floor it is noise that would teach
# people to ignore the hints.
SUGGEST_FLOOR = 0.45


@router.get("/api/sessions/{session_id}/speakers/suggestions")
def speaker_suggestions(session_id: int) -> dict:
    """Who each unnamed code sounds most like, for the naming screen.

    The post-meeting pass names a code only when the best match clears KNOWN_SPEAKER_THRESHOLD —
    right for writing a name into a transcript nobody would think to check, but the near-misses it
    throws away are exactly what the naming screen needs: a person with the audio in their ears can
    verify a 0.5 hint that the pass rightly refused to assert. Codes without a stored voiceprint —
    fragments too short to embed — get no suggestion; they are merge fodder, not naming candidates.
    """
    if not main.store.session(session_id):
        raise HTTPException(404, "no such session")
    known = main.store.known_voiceprints()
    if not known:
        return {}
    names = main.store.speaker_names(session_id)
    out: dict[str, dict] = {}
    for code in {l["speaker"] for l in main.store.lines(session_id)}:
        if names.get(code, "").strip():
            continue
        stored = main.store.voiceprint(session_id, code)
        if stored is None:
            continue
        emb = np.frombuffer(stored, dtype=np.float32)
        per_name: dict[str, float] = {}
        for name, print_ in known:
            candidate = np.frombuffer(print_, dtype=np.float32)
            if candidate.shape != emb.shape:
                continue
            score = speakers._cosine(emb, candidate)
            if score > per_name.get(name, 0.0):
                per_name[name] = score
        ranked = sorted(per_name.items(), key=lambda kv: -kv[1])
        if not ranked or ranked[0][1] < SUGGEST_FLOOR:
            continue
        # Same rule the recogniser applies (diarize): the best NAME must beat the runner-up name
        # by RECOGNISE_MARGIN. A polluted variant sits close to everyone, so without the margin it
        # tops every unnamed code's list and the naming screen fills with the same wrong name —
        # measured on a real meeting: five codes all hinting one person at margins of 0.01-0.04.
        if len(ranked) > 1 and ranked[0][1] - ranked[1][1] < config.RECOGNISE_MARGIN:
            continue
        out[code] = {"name": ranked[0][0], "similarity": round(ranked[0][1], 2)}
    return out


def _clip_entry(name: str, sid: int, stored: bool) -> dict:
    """One sample as the Learned page shows it: which meeting, when it ran, what the line says.

    A bare session id identifies nothing a human recognises — the date and the picked line's text
    are what let someone decide which sample to play. A harvested clip's meeting is deleted, so
    both are null there; the session id still keys the audio URL either way.
    """
    if stored:
        return {"session": sid, "started": None, "text": None}
    info = main.store.speaker_sample_info(name, sid)
    return {"session": sid, "started": info[0] if info else None, "text": info[1] if info else None}


@router.get("/api/speakers/known")
def get_known_speakers() -> list[dict]:
    counts = main.store.speaker_sessions()
    langs = main.store.speaker_languages()
    depts = main.store.speaker_departments()
    return [{"name": name, "sessions": counts.get(name, 0), "language": langs.get(name, ""),
             "department": depts.get(name, ""),
             # The meetings this voice can still be heard from (newest first) — live wavs plus
             # clips harvested from deleted meetings, which is what the Learned page renders
             # players for. Keyed by session id rather than position: a positional index shifts
             # when a sample is deleted, and the browser then replays the old audio for the
             # same URL.
             "clip_sessions": [_clip_entry(name, sid, stored)
                               for sid, stored in main.store.speaker_clip_sources(name)]}
            for name, _ in main.store.known_speakers()]


def clip_bytes(sample: tuple[str, float, float | None], seconds: float | None = None) -> bytes | None:
    """The same slice as _clip, as raw WAV bytes — None when the recording is missing.

    Split out so session deletion can harvest a voice's sound before the wav it lives in is
    removed, without fabricating an HTTP response to unwrap.
    """
    wav_path, start, span = sample
    if seconds is None:
        seconds = min(main.CLIP_SECONDS, span) if span else main.CLIP_SECONDS
        if span and span > seconds:
            # The middle of the utterance, not its head: when the segmenter missed a turn, the
            # other voice sits at the edges — the head is the previous speaker finishing. A line
            # clip passes explicit seconds and is untouched; hearing the whole line is its point.
            start += (span - seconds) / 2
    wav = config.recording_path(wav_path)
    if not wav.is_file():
        return None
    with sf.SoundFile(wav) as f:
        f.seek(min(int(start * f.samplerate), max(len(f) - 1, 0)))
        block = f.read(int(seconds * f.samplerate), dtype="int16")
        rate = f.samplerate
    buf = io.BytesIO()
    sf.write(buf, block, rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def harvest_speaker_clips(session_id: int) -> None:
    """Save each named, still-known voice's sample from this meeting before its wav disappears.

    An identified voice must keep a playable sound even after every meeting it spoke in is deleted;
    only forgetting the voice itself removes these.
    """
    known = {name for name, _ in main.store.known_speakers()}
    for name in set(main.store.speaker_names(session_id).values()):
        if name not in known:
            continue
        sample = main.store.speaker_sample(name, session_id)
        if sample and (audio := clip_bytes(sample)):
            main.store.save_speaker_clip(name, session_id, audio)


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
    audio = clip_bytes(sample, seconds)
    if audio is None:
        raise HTTPException(404, f"recording not found: {config.recording_path(sample[0])}")
    return Response(audio, media_type="audio/wav")


def _clip_source(name: str, session: int) -> bool:
    """Whether the sample for this meeting is a harvested clip — 404 when there is none.

    Samples are addressed by session id, not position: an index shifts when a sample is deleted,
    and the browser, replaying the same URL, would serve the deleted voice's audio.
    """
    for session_id, stored in main.store.speaker_clip_sources(name):
        if session_id == session:
            return stored
    raise HTTPException(404, "no such sample")


@router.get("/api/speakers/known/{name}/clip")
def get_speaker_clip(name: str, session: int | None = None) -> Response:
    """A few seconds of the voice behind the name, so a wrong match is audible rather than guessed.

    `session` picks which meeting to hear it from — meetings still on disk are cut live,
    deleted ones play the clip harvested when they were removed. Omitted, it plays the newest:
    callers like the naming screen's compare button just ask what this person sounds like, and
    making the parameter mandatory silently muted them (their fetch 422'd into a catch).
    """
    if session is None:
        sources = main.store.speaker_clip_sources(name)
        if not sources:
            raise HTTPException(404, "no recording for this voice")
        session = sources[0][0]
    if _clip_source(name, session):
        audio = main.store.stored_clip(name, session)
        if audio is None:
            raise HTTPException(404, "no recording for this voice")
        return Response(audio, media_type="audio/wav")
    return _clip(main.store.speaker_sample(name, session))


@router.delete("/api/speakers/known/{name}/clip")
def delete_known_speaker_clip(name: str, session: int) -> list[dict]:
    """Drop one bad sample: undo the meeting that taught it, not just hide the audio.

    A sample sounds wrong because that meeting named somebody else's code as this person, and
    remember_speaker already learned the polluted print. So deleting the sample withdraws the
    naming and unlearns what it taught, then removes the harvested clip. For a meeting already
    deleted (stored clip), codes and voiceprints are gone with the session, so the loops are
    naturally empty and only the clip row goes.

    Withdrawing the last sample forgets the voice outright: with the evidence gone the name
    disappears from the Learned page, but its prints would keep recognising people with no page
    left to unteach them from — an orphan voiceprint nobody can see or remove.
    """
    _clip_source(name, session)
    for code in main.store.session_codes_for(name, session):
        if old := main.store.voiceprint(session, code):
            main.store.unlearn_speaker(name, old)
    main.store.unname_speaker(session, name)
    main.store.delete_speaker_clip(name, session)
    if not main.store.speaker_clip_sources(name):
        main.store.forget_speaker(name)
    return get_known_speakers()


@router.put("/api/speakers/known/{name}/clip")
def reassign_known_speaker_clip(name: str, body: dict, session: int) -> list[dict]:
    """The same undo, but the sample belongs to somebody the room knows: hand it over.

    The meeting's codes are renamed to the right person, each voiceprint is unlearned from the
    wrong name and learned by the right one, and the harvested clip follows. Same session-deleted
    caveat as deletion: with no codes left, only the clip row moves.
    """
    new = str(body.get("name", "")).strip()
    if not new or new == name:
        raise HTTPException(400, "target name required")
    _clip_source(name, session)
    for code in main.store.session_codes_for(name, session):
        main.store.set_speaker_name(session, code, new)
        if centroid := main.store.voiceprint(session, code):
            main.store.unlearn_speaker(name, centroid)
            main.store.remember_speaker(new, centroid)
    main.store.move_speaker_clip(name, session, new)
    return get_known_speakers()


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


@router.put("/api/speakers/known/{name}/department")
def set_known_speaker_department(name: str, body: dict) -> list[dict]:
    main.store.set_speaker_department(name, str(body.get("department", "")).strip())
    return get_known_speakers()


@router.delete("/api/speakers/known/{name}")
def delete_known_speaker(name: str) -> list[dict]:
    main.store.forget_speaker(name)
    return get_known_speakers()

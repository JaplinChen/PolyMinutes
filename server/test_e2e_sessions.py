"""Sessions over HTTP: recording control, the subtitle socket, import, and what 404s."""

from __future__ import annotations

import asyncio
import io
import json
import shutil
import subprocess
import threading
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from . import config, jobs, main
from .e2e_support import seed_session, wait_for


def test_recording_lifecycle(client: TestClient) -> None:
    assert client.post("/api/recording/stop").status_code == 409
    status = client.get("/api/recording/status").json()
    assert status["recording"] is False and status["sessionId"] is None


def test_websocket_receives_config_and_events(client: TestClient) -> None:
    with client.websocket_connect("/ws/live") as ws:
        first = ws.receive_json()
        assert first["type"] == "config"
        assert "display" in first and "languages" in first

        # Publishing crosses the thread boundary the pipeline uses.
        threading.Thread(target=lambda: main.hub.publish({"type": "line", "line": {"id": 1}})).start()
        assert ws.receive_json()["line"]["id"] == 1


def test_known_voice_can_be_heard_and_renamed(client: TestClient) -> None:
    """A learned voice is only inspectable if you can play it back and fix the name on it."""
    import soundfile as sf

    wav = config.RECORDINGS_DIR / "voice.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), np.zeros(config.SAMPLE_RATE * 10, dtype="float32"), config.SAMPLE_RATE)

    session = main.store.start_session("now", str(wav))
    main.store.add_line(session, 5.0, "S1", "en", "hello", {})
    main.store.save_voiceprint(session, "S1", b"\x00" * 8)
    assert client.put(f"/api/sessions/{session}/speakers", json={"S1": "Ana"}).status_code == 200

    known = client.get("/api/speakers/known").json()
    assert [s["name"] for s in known] == ["Ana"] and known[0]["sessions"] >= 1

    clip = client.get("/api/speakers/known/Ana/clip")
    assert clip.status_code == 200 and clip.headers["content-type"] == "audio/wav"
    heard, rate = sf.read(io.BytesIO(clip.content))
    assert len(heard) == main.CLIP_SECONDS * rate, len(heard)

    renamed = client.put("/api/speakers/known/Ana", json={"name": "Ana Lee"}).json()
    assert [s["name"] for s in renamed] == ["Ana Lee"]
    # The transcript must follow the rename, or it keeps showing a name that no longer exists.
    assert client.get(f"/api/sessions/{session}/lines").json()["speakers"]["S1"] == "Ana Lee"
    assert client.get("/api/speakers/known/Ana/clip").status_code == 404

    assert client.delete("/api/speakers/known/Ana%20Lee").json() == []


def test_an_identified_voice_survives_its_meetings_deletion(client: TestClient) -> None:
    """Deleting a meeting removes its wav — but an identified voice must keep a playable sample.

    The clip is harvested into the database at deletion time and only forgetting the voice itself
    removes it; otherwise "who does this name sound like" becomes unanswerable the day the last
    meeting they spoke in is cleaned up.
    """
    import soundfile as sf

    wav = config.RECORDINGS_DIR / "kept-voice.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), np.zeros(config.SAMPLE_RATE * 10, dtype="float32"), config.SAMPLE_RATE)

    session = main.store.start_session("now", str(wav))
    main.store.add_line(session, 1.0, "S1", "zh", "這段話夠長，聽得出是誰", {}, end_time=6.0)
    main.store.save_voiceprint(session, "S1", b"\x00" * 8)
    assert client.put(f"/api/sessions/{session}/speakers", json={"S1": "林保留"}).status_code == 200

    assert client.get("/api/speakers/known/%E6%9E%97%E4%BF%9D%E7%95%99/clip?idx=0").status_code == 200
    assert client.delete(f"/api/sessions/{session}").status_code == 200
    assert not wav.exists()

    # The meeting and its wav are gone; the voice still plays, from the harvested clip.
    clip = client.get("/api/speakers/known/%E6%9E%97%E4%BF%9D%E7%95%99/clip?idx=0")
    assert clip.status_code == 200 and clip.headers["content-type"] == "audio/wav"
    sf.read(io.BytesIO(clip.content))  # valid WAV, not an empty blob

    kept = [s for s in client.get("/api/speakers/known").json() if s["name"] == "林保留"]
    assert kept and kept[0]["clips"] == 1

    # Manual forget is the one thing that removes it.
    client.delete("/api/speakers/known/%E6%9E%97%E4%BF%9D%E7%95%99")
    assert client.get("/api/speakers/known/%E6%9E%97%E4%BF%9D%E7%95%99/clip?idx=0").status_code == 404


def test_a_line_can_be_reassigned_to_another_speaker(client: TestClient) -> None:
    """The shared-mic collapse puts every line on one speaker; the human splits them back apart.

    Reassigning relabels the line and leaves the voiceprint alone — the centroid the clustering
    built from a collapsed speaker is the thing that was wrong.
    """
    wav = config.RECORDINGS_DIR / "split.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    sf.write(str(wav), np.zeros(config.SAMPLE_RATE * 10, dtype="float32"), config.SAMPLE_RATE)

    session = main.store.start_session("now", str(wav))
    a = main.store.add_line(session, 1.0, "S1", "en", "mine", {})
    b = main.store.add_line(session, 5.0, "S1", "en", "actually yours", {})

    moved = client.put(f"/api/sessions/{session}/lines/{b}/speaker", json={"speaker": "S2"})
    assert moved.status_code == 200, moved.text
    by_id = {l["id"]: l for l in moved.json()["lines"]}
    assert by_id[a]["speaker"] == "S1" and by_id[b]["speaker"] == "S2", by_id

    assert client.put(f"/api/sessions/{session}/lines/{b}/speaker", json={"speaker": ""}).status_code == 400
    assert client.put(f"/api/sessions/{session}/lines/99999/speaker", json={"speaker": "S2"}).status_code == 404


def test_several_codes_for_one_person_merge_into_one(client: TestClient) -> None:
    """The diariser splits one drifting voice into several codes; merging folds them back to one.

    Every absorbed line moves to the kept code, and the absorbed codes leave the meeting — the
    speaker list is derived from which codes still label a line.
    """
    session = main.store.start_session("now", "")
    a = main.store.add_line(session, 1.0, "S1", "en", "one", {})
    b = main.store.add_line(session, 5.0, "S17", "en", "two", {})
    c = main.store.add_line(session, 9.0, "S20", "en", "three", {})
    main.store.save_voiceprint(session, "S17", b"\x00" * 8)
    client.put(f"/api/sessions/{session}/speakers", json={"S1": "Vince", "S17": "Vince"})

    merged = client.post(f"/api/sessions/{session}/speakers/merge",
                         json={"into": "S1", "from": ["S17", "S20"]})
    assert merged.status_code == 200, merged.text
    by_id = {l["id"]: l["speaker"] for l in merged.json()["lines"]}
    assert by_id == {a: "S1", b: "S1", c: "S1"}, by_id
    # The absorbed codes are gone from the meeting: no line carries them, no name lingers.
    assert set(l["speaker"] for l in merged.json()["lines"]) == {"S1"}
    assert "S17" not in merged.json()["speakers"] and "S20" not in merged.json()["speakers"]
    assert main.store.voiceprint(session, "S17") is None

    assert client.post(f"/api/sessions/{session}/speakers/merge",
                       json={"into": "S1", "from": []}).status_code == 400


def test_a_session_exports_as_a_word_document(client: TestClient) -> None:
    """The enterprise deliverable is a .docx, and it has to carry the transcript, not just open."""
    session = main.store.start_session("now", "")
    main.store.add_line(session, 1.0, "S1", "en", "quarterly numbers", {"zh": "季度數字"})
    main.store.add_line(session, 5.0, "S2", "zh", "料號要確認", {})

    r = client.get(f"/api/sessions/{session}/docx")
    assert r.status_code == 200, r.text
    assert "wordprocessingml" in r.headers["content-type"], r.headers["content-type"]
    assert r.content[:2] == b"PK", "a .docx is a zip and must start with the zip magic"

    from docx import Document
    text = "\n".join(p.text for p in Document(io.BytesIO(r.content)).paragraphs)
    assert "quarterly numbers" in text and "料號要確認" in text, text
    assert "季度數字" in text, "translations must survive into the export"

    assert client.get("/api/sessions/99999/docx").status_code == 404


def test_importing_a_recording_makes_it_a_session(client: TestClient) -> None:
    """An uploaded file has to land as an ordinary session, or nothing can be learned from it."""
    if shutil.which("ffmpeg") is None:
        print("  (skipped: ffmpeg not installed)")
        return

    import soundfile as sf

    # Not named import-*: that prefix belongs to what the endpoint writes, and this test counts those.
    src = config.RECORDINGS_DIR / "fixture.m4a"
    src.parent.mkdir(parents=True, exist_ok=True)
    tone = np.sin(np.arange(config.SAMPLE_RATE * 3) * 0.05).astype("float32")
    raw = config.RECORDINGS_DIR / "fixture.wav"
    sf.write(str(raw), tone, config.SAMPLE_RATE)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), str(src)], check=True)

    before = len(client.get("/api/sessions").json())
    r = client.post("/api/sessions/import?filename=meeting.m4a", content=src.read_bytes())
    assert r.status_code == 200, r.text
    session_id = r.json()["id"]
    # The response comes back before the pass: the decode is queued, not run inside the request.
    assert r.json()["state"] == "refining", r.text
    assert wait_for(lambda: session_id in main.postprocess.calls), "import never queued its pass"
    assert wait_for(lambda: (jobs.state(session_id) or {}).get("state") == "refined")

    listed = client.get("/api/sessions").json()
    assert len(listed) == before + 1
    imported = next(s for s in listed if s["id"] == session_id)
    # Ended, or /reprocess and the clip endpoint would treat it as still recording.
    assert imported["ended"] and Path(imported["wav_path"]).is_file()
    # The upload itself is not kept: everything downstream reads the extracted wav.
    assert not list(config.RECORDINGS_DIR.glob("import-*-meeting.m4a"))

    assert client.post("/api/sessions/import?filename=empty.mp4", content=b"").status_code == 400
    assert client.post("/api/sessions/import?filename=x.mp4", content=b"not a video").status_code == 400
    # A rejected upload must leave nothing behind, or the next import inherits a stale wav.
    assert len(list(config.RECORDINGS_DIR.glob("import-*.wav"))) == 1

    # A second import in the same second must not overwrite the first one's audio.
    again = client.post("/api/sessions/import?filename=meeting.m4a", content=src.read_bytes())
    assert again.status_code == 200, again.text
    imports = [s for s in client.get("/api/sessions").json() if s["id"] in (session_id, again.json()["id"])]
    assert len({s["wav_path"] for s in imports}) == 2, imports


def test_importing_from_a_url_makes_it_a_session(client: TestClient) -> None:
    """A link is imported the way a file is: the fetch rides inside the refine job, so the
    response returns at once and a bad link is a failed job on the chip, not a hung request."""
    from . import ingest

    # Not a link, and no such file on the server: named as a path problem, not swallowed as a job.
    r = client.post("/api/sessions/import-url", json={"url": "\\\\nas\\nowhere\\gone.mp4"})
    assert r.status_code == 400 and "no file" in r.json()["detail"], r.text
    assert client.post("/api/sessions/import-url", json={}).status_code == 400

    real_have = ingest.have_downloader
    ingest.have_downloader = lambda: False
    try:
        r = client.post("/api/sessions/import-url", json={"url": "https://example.com/v"})
        assert r.status_code == 503, r.text
    finally:
        ingest.have_downloader = real_have

    if shutil.which("ffmpeg") is None or not ingest.have_downloader():
        print("  (skipped success path: ffmpeg or yt-dlp not installed)")
        return

    import soundfile as sf

    def fake_download(url: str, dest: Path) -> None:
        tone = np.sin(np.arange(config.SAMPLE_RATE * 2) * 0.05).astype("float32")
        raw = config.RECORDINGS_DIR / "urlfixture.wav"
        sf.write(str(raw), tone, config.SAMPLE_RATE)
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-f", "mp4", str(dest)],
                       check=True)
        raw.unlink()

    real_download = ingest.download_audio
    real_subs = ingest.download_subtitles
    ingest.download_audio = fake_download
    # Subtitle probing hits the network; these checks are about the audio path.
    ingest.download_subtitles = lambda *a: None
    try:
        r = client.post("/api/sessions/import-url", json={"url": "https://example.com/talk"})
        assert r.status_code == 200 and r.json()["state"] == "refining", r.text
        session_id = r.json()["id"]
        assert wait_for(lambda: (jobs.state(session_id) or {}).get("state") == "refined")
    finally:
        ingest.download_audio = real_download

    imported = next(s for s in client.get("/api/sessions").json() if s["id"] == session_id)
    wav = Path(config.recording_path(imported["wav_path"]))
    assert imported["ended"] and wav.is_file() and wav.stat().st_size > 0
    assert session_id in main.postprocess.calls
    # The fetched media is not kept — only the extracted wav is.
    assert not list(config.RECORDINGS_DIR.glob("import-*.download"))
    wav.unlink(missing_ok=True)

    # A download that fails must surface on the job, with the tool's reason.
    ingest.download_audio = lambda url, dest: (_ for _ in ()).throw(ValueError("HTTP 404"))
    try:
        r = client.post("/api/sessions/import-url", json={"url": "https://example.com/gone"})
        failed = r.json()["id"]
        assert wait_for(lambda: (jobs.state(failed) or {}).get("state") == "failed")
        assert "404" in (jobs.state(failed) or {}).get("error", "")
    finally:
        ingest.download_audio = real_download
    Path(config.recording_path(
        next(s for s in client.get("/api/sessions").json() if s["id"] == failed)["wav_path"]
    )).unlink(missing_ok=True)

    # A non-link is a path the server reads directly — and the original must survive the import.
    shared = config.RECORDINGS_DIR / "share-fixture.m4a"
    fake_download("", shared)
    r = client.post("/api/sessions/import-url", json={"url": str(shared)})
    assert r.status_code == 200, r.text
    from_path = r.json()["id"]
    assert wait_for(lambda: (jobs.state(from_path) or {}).get("state") == "refined")
    kept = next(s for s in client.get("/api/sessions").json() if s["id"] == from_path)
    assert Path(config.recording_path(kept["wav_path"])).stat().st_size > 0
    assert shared.is_file(), "the source on the share must not be deleted"
    shared.unlink()
    Path(config.recording_path(kept["wav_path"])).unlink(missing_ok=True)

    # A link whose uploader wrote subtitles skips the decode: the cues become the transcript.
    def fake_subs(url, languages, dest_dir, stem):
        vtt = dest_dir / f"{stem}.en.vtt"
        vtt.write_text("WEBVTT\n\n00:00.000 --> 00:02.000\nhello there\n", encoding="utf-8")
        return vtt, "en"

    ingest.download_audio = fake_download
    ingest.download_subtitles = fake_subs
    try:
        r = client.post("/api/sessions/import-url", json={"url": "https://example.com/subbed"})
        subbed = r.json()["id"]
        assert wait_for(lambda: (jobs.state(subbed) or {}).get("state") == "refined")
    finally:
        ingest.download_audio = real_download
        ingest.download_subtitles = real_subs
    assert subbed in main.postprocess.subtitle_calls, "subtitles must feed the subtitle path"
    assert subbed not in main.postprocess.calls, "with subtitles there must be no decode"
    row = next(s for s in client.get("/api/sessions").json() if s["id"] == subbed)
    wav = Path(config.recording_path(row["wav_path"]))
    assert wav.stat().st_size > 0, "the audio is still fetched for playback and reprocess"
    assert not list(config.RECORDINGS_DIR.glob("import-*.vtt")), "the vtt is not kept"
    wav.unlink(missing_ok=True)


def test_subtitle_cues_become_ordinary_translated_lines(tmp: Path) -> None:
    """The real subtitle path — parse_vtt through subtitle_session — off the stubs: cues land as
    S1 lines with translations from the shared row loop, and markup/karaoke tags are stripped."""
    from . import ingest, postprocess as postprocess_mod
    from .e2e_support import StubTranslator

    vtt = tmp / "talk.en.vtt"
    vtt.write_text(
        "WEBVTT\nKind: captions\n\n"
        "1\n00:01.000 --> 00:03,500\n<c.yellow>Hello</c> <b>everyone</b>\n\n"
        "NOTE a comment block with no timestamp\n\n"
        "2\n01:00:00.000 --> 01:00:02.000\nsecond line\nwrapped onto two\n\n"
        "00:05.000 --> 00:06.000\n   \n",  # tag-only/blank cue must be dropped
        encoding="utf-8")
    cues = ingest.parse_vtt(vtt)
    assert cues == [(1.0, 3.5, "Hello everyone"), (3600.0, 3602.0, "second line wrapped onto two")]

    session_id = main.store.start_session("2026-01-01T09:00:00", str(tmp / "talk.wav"))
    main.store.end_session(session_id, "2026-01-01T10:00:00")
    main.state["cfg"].languages = ["en", "zh"]
    rows = postprocess_mod.subtitle_session(main.store, session_id, cues, "en",
                                            main.state["cfg"], StubTranslator())
    stored = main.store.lines(session_id)
    assert len(stored) == 2 and all(l["speaker"] == "S1" for l in stored)
    assert stored[0]["lang"] == "en" and stored[0]["source"] == "Hello everyone"
    assert stored[0]["translations"]["zh"] == "[zh] Hello everyone"
    assert rows[1]["start"] == 3600.0

    try:
        postprocess_mod.subtitle_session(main.store, session_id, [], "en", main.state["cfg"])
        raise AssertionError("an empty cue list must not wipe the transcript")
    except ValueError:
        pass
    assert len(main.store.lines(session_id)) == 2


def test_recordings_survive_a_renamed_project_directory(tmp: Path) -> None:
    """Stored paths must not pin a recording to the folder name the project had that day.

    Renaming the project once stranded every session: wav_path held an absolute path into the old
    directory, and nothing reads a session without reading its audio. Not reachable through the
    API — `isolate` puts recordings outside the repo root, which is exactly the case that stays
    absolute — so the three pieces are checked where they live.
    """
    import sqlite3

    from . import schema, store as store_mod

    under_root = config.ROOT / "recordings" / "meeting.wav"
    assert store_mod._portable(str(under_root)) == "recordings/meeting.wav"
    # Nothing to be relative to: kept as given rather than mangled into a wrong path.
    outside = tmp.resolve() / "elsewhere.wav"
    assert store_mod._portable(str(outside)) == str(outside)

    assert config.recording_path("recordings/meeting.wav") == under_root
    assert config.recording_path(str(outside)) == outside

    db = sqlite3.connect(":memory:")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE session (id INTEGER PRIMARY KEY, wav_path TEXT NOT NULL)")
    # Absolute, and built from a real path so this reads the same on the mac side of the project.
    stale = tmp.resolve() / "OldProjectName" / "recordings" / "old.wav"
    rows = [(1, str(stale)),
            (2, "recordings/already-relative.wav"),
            (3, str(outside))]
    db.executemany("INSERT INTO session (id, wav_path) VALUES (?,?)", rows)
    schema._relativise_recordings(db)
    after = {r["id"]: r["wav_path"] for r in db.execute("SELECT id, wav_path FROM session")}
    db.close()
    assert after[1] == "recordings/old.wav", after
    # Untouched: one is already relative, and the other was deliberately stored outside the project.
    assert after[2] == "recordings/already-relative.wav" and after[3] == str(outside), after


def test_import_decodes_off_the_event_loop(client: TestClient) -> None:
    """ffmpeg and the ASR decode are blocking; on the event loop they freeze every other request —
    a live meeting's subtitle socket included — for the length of the import. They must run in a
    worker thread. Proven by whether extract_audio sees a running loop in its own thread."""
    if shutil.which("ffmpeg") is None:
        print("  (skipped: ffmpeg not installed)")
        return

    import soundfile as sf

    from . import ingest

    src = config.RECORDINGS_DIR / "loopprobe.m4a"
    src.parent.mkdir(parents=True, exist_ok=True)
    tone = np.sin(np.arange(config.SAMPLE_RATE * 2) * 0.05).astype("float32")
    raw = config.RECORDINGS_DIR / "loopprobe.wav"
    sf.write(str(raw), tone, config.SAMPLE_RATE)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), str(src)], check=True)

    real = ingest.extract_audio
    saw: dict[str, bool] = {}

    def spy(s: Path, d: Path) -> None:
        try:
            asyncio.get_running_loop()
            saw["on_loop"] = True
        except RuntimeError:
            saw["on_loop"] = False
        return real(s, d)

    ingest.extract_audio = spy
    try:
        r = client.post("/api/sessions/import?filename=probe.m4a", content=src.read_bytes())
    finally:
        ingest.extract_audio = real

    assert r.status_code == 200, r.text
    assert saw.get("on_loop") is False, "import decode ran on the event loop thread"

    # Let the queued pass finish before deleting the wav from under it.
    wait_for(lambda: (jobs.state(r.json()["id"]) or {}).get("state") == "refined")
    # Shared RECORDINGS_DIR: leave no import-*.wav behind, or the next import test miscounts.
    imported = next(s for s in client.get("/api/sessions").json() if s["id"] == r.json()["id"])
    Path(imported["wav_path"]).unlink(missing_ok=True)
    raw.unlink(missing_ok=True)
    src.unlink(missing_ok=True)


def test_summary_lifecycle_none_then_stored_then_stale(client: TestClient) -> None:
    """The three answers to "where is my summary": never ran, here it is, and it is out of date."""
    jobs.reset()
    session_id = seed_session("summary-life.wav")

    body = client.get(f"/api/sessions/{session_id}/summary").json()
    assert body["state"] == "none" and body["summary"] is None, body
    assert client.get("/api/sessions/999999/summary").status_code == 404

    rev = main.store.session(session_id)["lines_rev"]
    main.store.set_summary(session_id, json.dumps(
        {"zh": {"title": "週會", "summary": "談了排程", "decisions": ["交期延後"],
                "actions": [{"text": "追蹤供應商", "speaker": "S1"}]}}, ensure_ascii=False),
        "ok", rev, "2026-08-05T10:00:00")

    body = client.get(f"/api/sessions/{session_id}/summary").json()
    assert body["state"] == "ok" and not body["stale"], body
    assert body["summary"]["zh"]["title"] == "週會", body

    # Editing a line moves the revision; the same summary now admits it is stale.
    line_id = main.store.lines(session_id)[0]["id"]
    main.store.update_line(line_id, "改過的一行", {})
    body = client.get(f"/api/sessions/{session_id}/summary").json()
    assert body["stale"] is True, body


def test_regenerating_a_fresh_summary_is_refused(client: TestClient) -> None:
    """No auth in front of this endpoint, and every call spends money or minutes of compute."""
    import time as _time

    jobs.reset()
    session_id = seed_session("summary-cool.wav")
    rev = main.store.session(session_id)["lines_rev"]
    now = _time.strftime("%Y-%m-%dT%H:%M:%S")
    main.store.set_summary(session_id, "{}", "ok", rev, now)

    assert client.post(f"/api/sessions/{session_id}/summarize").status_code == 429

    # A stale summary is exempt: regenerating it is the point of tracking staleness.
    line_id = main.store.lines(session_id)[0]["id"]
    main.store.update_line(line_id, "剛改的一行", {})
    r = client.post(f"/api/sessions/{session_id}/summarize")
    assert r.status_code == 200, r.text
    jobs.cancel_all(wait=2.0)

    # And a failed summary is also exempt — retrying a failure is not abuse.
    main.store.set_summary(session_id, "{}", "failed",
                           main.store.session(session_id)["lines_rev"], now)
    jobs.reset()
    r = client.post(f"/api/sessions/{session_id}/summarize")
    assert r.status_code == 200, r.text
    jobs.cancel_all(wait=2.0)


def test_markdown_export_carries_the_summary(client: TestClient) -> None:
    """The export's reader was not in the room; the summary rides along, staleness admitted."""
    jobs.reset()
    session_id = seed_session("summary-md.wav")
    rev = main.store.session(session_id)["lines_rev"]
    main.store.set_summary(session_id, json.dumps(
        {"zh": {"title": "產線週會", "summary": "確認委外排程。", "decisions": ["交期延到週五"],
                "actions": [{"text": "回覆採購單", "speaker": "S1"}]},
         "en": {"title": "Line weekly", "summary": "Confirmed outsourcing schedule.",
                "decisions": [], "actions": []}}, ensure_ascii=False),
        "ok", rev, "2026-08-05T10:00:00")
    main.store.set_speaker_name(session_id, "S1", "陳經理")

    md = client.get(f"/api/sessions/{session_id}/markdown").text
    assert "## 會議摘要" in md and "產線週會" in md and "Line weekly" in md, md[:400]
    assert "陳經理：回覆採購單" in md, md
    assert "⚠" not in md

    # Post-summary edits surface as a warning, not as silence.
    line_id = main.store.lines(session_id)[0]["id"]
    main.store.update_line(line_id, "後來改的", {})
    md = client.get(f"/api/sessions/{session_id}/markdown").text
    assert "摘要生成後逐字稿曾被修改" in md

    # A failed summary never reaches the export.
    main.store.set_summary(session_id, "{}", "failed", 0, "2026-08-05T10:00:00")
    md = client.get(f"/api/sessions/{session_id}/markdown").text
    assert "## 會議摘要" not in md


def test_unknown_api_path_is_json_404(client: TestClient) -> None:
    r = client.get("/api/definitely-not-a-route")
    assert r.status_code == 404
    assert r.headers["content-type"].startswith("application/json")

    # Whatever the method: a 405 here would report "wrong method" for a route that does not exist,
    # which is exactly how a server running older code presents itself.
    for send in (client.post, client.put, client.delete):
        r = send("/api/definitely-not-a-route")
        assert r.status_code == 404, (send.__name__, r.status_code)
        assert r.headers["content-type"].startswith("application/json")


def test_rerunning_a_line_refuses_what_it_should(client: TestClient) -> None:
    """The rerun endpoint has no authentication in front of it, so its guards are the only ones."""
    jobs.reset()
    session_id = seed_session("rerun.wav")
    line_id = main.store.lines(session_id)[0]["id"]

    assert client.post(f"/api/sessions/{session_id}/lines/999999/rerun").status_code == 404
    assert client.post(f"/api/sessions/999999/lines/{line_id}/rerun").status_code == 404

    # A line id belonging to another session must not be reachable through this session's path.
    other = seed_session("rerun-other.wav")
    other_line = main.store.lines(other)[0]["id"]
    assert client.post(f"/api/sessions/{session_id}/lines/{other_line}/rerun").status_code == 404

    # Not while that session is recording: the wav is still being written.
    main.state["session"] = session_id
    try:
        assert client.post(f"/api/sessions/{session_id}/lines/{line_id}/rerun").status_code == 409
    finally:
        main.state["session"] = None

    # A line with no duration is refused rather than decoded as a 60-second span.
    main.store.replace_line(line_id, "一行", "zh", {}, "ok")
    assert client.post(f"/api/sessions/{session_id}/lines/{line_id}/rerun").status_code == 400


def test_unnamed_speaker_can_be_heard_before_naming(client: TestClient) -> None:
    """The naming screen shows S1..S35 and asks who they are; it has to let you hear them.

    /api/speakers/known/{name}/clip resolves a voice through the name attached to it, so it can only
    play back a speaker who has already been identified — no use to the screen doing the identifying.
    """
    import soundfile as sf

    wav = config.RECORDINGS_DIR / "unnamed.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), np.zeros(config.SAMPLE_RATE * 30, dtype="float32"), config.SAMPLE_RATE)

    session = main.store.start_session("now", str(wav))
    # "謝謝" first and longer speech later: picking the earliest line would play the useless one.
    main.store.add_line(session, 1.0, "S1", "zh", "謝謝", {}, end_time=1.4)
    main.store.add_line(session, 12.0, "S1", "zh", "這段話長得多，聽得出是誰", {}, end_time=20.0)
    main.store.add_line(session, 3.0, "S2", "zh", "另一個人", {}, end_time=5.0)

    # Nobody has been named — the endpoint keyed on names finds nothing to play.
    assert client.get("/api/speakers/known/S1/clip").status_code == 404

    clip = client.get(f"/api/sessions/{session}/speakers/S1/clip")
    assert clip.status_code == 200 and clip.headers["content-type"] == "audio/wav"
    heard, rate = sf.read(io.BytesIO(clip.content))
    assert len(heard) == main.CLIP_SECONDS * rate, len(heard)

    # Which line it picked cannot be heard — the fixture is silence — so assert it directly:
    # the 8-second utterance at 12.0s, not the 0.4-second "謝謝" that comes first.
    assert main.store.session_speaker_sample(session, "S1") == (str(wav), 12.0, 8.0)

    assert client.get(f"/api/sessions/{session}/speakers/S2/clip").status_code == 200
    assert client.get(f"/api/sessions/{session}/speakers/S9/clip").status_code == 404
    assert client.get(f"/api/sessions/999999/speakers/S1/clip").status_code == 404


def test_a_sample_never_runs_past_the_utterance_it_came_from(client: TestClient) -> None:
    """A speaker whose longest utterance is short must not be sampled over the next person.

    Four seconds from the start of a two-second answer is two seconds of whoever spoke next, and a
    sample holding two voices cannot answer the only question the naming screen asks. Measured on a
    real meeting before this: 5 of 14 sampled speakers had someone else starting inside the clip.
    """
    import soundfile as sf

    wav = config.RECORDINGS_DIR / "short-turn.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), np.zeros(config.SAMPLE_RATE * 30, dtype="float32"), config.SAMPLE_RATE)

    session = main.store.start_session("now", str(wav))
    main.store.add_line(session, 2.0, "S1", "zh", "好 沒問題", {}, end_time=4.0)
    main.store.add_line(session, 4.1, "S2", "zh", "接下來換我報告這一段", {}, end_time=12.0)

    heard, rate = sf.read(io.BytesIO(client.get(f"/api/sessions/{session}/speakers/S1/clip").content))
    assert len(heard) == 2.0 * rate, f"{len(heard)/rate}s played of a 2s utterance"

    # The long one is still capped at CLIP_SECONDS: the question is whose voice, not what was said.
    heard, rate = sf.read(io.BytesIO(client.get(f"/api/sessions/{session}/speakers/S2/clip").content))
    assert len(heard) == main.CLIP_SECONDS * rate

    # A transcript written before end_time existed has nothing to bound the clip with, and still
    # has to play something.
    main.store.add_line(session, 20.0, "S3", "zh", "沒有結束時間的舊資料", {})
    heard, rate = sf.read(io.BytesIO(client.get(f"/api/sessions/{session}/speakers/S3/clip").content))
    assert len(heard) == main.CLIP_SECONDS * rate


def test_a_sample_avoids_the_longest_line_and_plays_the_middle(client: TestClient) -> None:
    """The longest line is the dirtiest sample: a line is long exactly when the segmenter missed a
    speaker turn inside it. Prefer a mid-length utterance, and cut the clip from its middle — the
    edges are where the other voice sits."""
    import soundfile as sf

    wav = config.RECORDINGS_DIR / "clean-sample.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    # Nonzero audio only inside 18s–22s: the middle of the 10s–30s utterance. A clip cut from the
    # head would read silence; only a centred cut reads ones.
    audio = np.zeros(config.SAMPLE_RATE * 40, dtype="float32")
    audio[config.SAMPLE_RATE * 18:config.SAMPLE_RATE * 22] = 1.0
    sf.write(str(wav), audio, config.SAMPLE_RATE)

    session = main.store.start_session("now", str(wav))
    main.store.add_line(session, 0.0, "S1", "zh", "六十秒的超長句最可能藏著漏切的換手", {}, end_time=60.0)
    main.store.add_line(session, 10.0, "S1", "zh", "二十秒的中等句比較乾淨", {}, end_time=30.0)
    main.store.add_line(session, 31.0, "S1", "zh", "謝謝", {}, end_time=31.5)

    # The 20s line wins over both the 60s monster and the sub-3s "謝謝".
    assert main.store.session_speaker_sample(session, "S1") == (str(wav), 10.0, 20.0)

    heard, rate = sf.read(io.BytesIO(client.get(f"/api/sessions/{session}/speakers/S1/clip").content))
    assert len(heard) == main.CLIP_SECONDS * rate
    assert float(np.abs(heard).mean()) > 0.9, "the clip must come from the utterance's middle"

    # A line's own clip still plays from its start, whole: 10s–30s, capped at 60s elsewhere.
    line_id = next(l["id"] for l in main.store.lines(session) if l["start"] == 10.0)
    heard, rate = sf.read(io.BytesIO(client.get(f"/api/sessions/{session}/lines/{line_id}/clip").content))
    assert len(heard) == 20.0 * rate, "a line clip is the whole line, not a centred slice"

    main.store.delete_session(session)


def test_a_sample_prefers_mid_monologue_over_a_speaker_boundary(client: TestClient) -> None:
    """A line flanked by other speakers sits exactly where a missed turn leaves their voice in it.

    Found on a real import: the sample rule picked a 7.8s line wedged between two other people
    while the same speaker had a 7.3s line inside his own monologue. Duration proximity must not
    outrank being nowhere near a speaker change.
    """
    import soundfile as sf

    wav = config.RECORDINGS_DIR / "boundary-sample.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), np.zeros(config.SAMPLE_RATE * 60, dtype="float32"), config.SAMPLE_RATE)

    session = main.store.start_session("now", str(wav))
    # An 8s line wedged between two other speakers — ideal length, dirtiest position.
    main.store.add_line(session, 0.0, "S2", "zh", "前一位", {}, end_time=9.5)
    main.store.add_line(session, 10.0, "S1", "zh", "夾在別人中間的八秒句", {}, end_time=18.0)
    main.store.add_line(session, 18.5, "S3", "zh", "後一位", {}, end_time=20.0)
    # A 5s line inside S1's own monologue — farther from eight, nowhere near a boundary.
    main.store.add_line(session, 30.0, "S1", "zh", "獨白第一句", {}, end_time=36.0)
    main.store.add_line(session, 36.5, "S1", "zh", "獨白中段的五秒句", {}, end_time=41.5)
    main.store.add_line(session, 42.0, "S1", "zh", "獨白最後一句", {}, end_time=50.0)

    assert main.store.session_speaker_sample(session, "S1") == (str(wav), 36.5, 5.0)

    # The named path applies the same rule.
    main.store.set_speaker_name(session, "S1", "邊界測試員")
    assert main.store.speaker_sample("邊界測試員", session) == (str(wav), 36.5, 5.0)

    main.store.delete_session(session)
    wav.unlink(missing_ok=True)


def test_a_learned_correction_can_be_fixed_in_place(client: TestClient) -> None:
    """Deleting and re-learning means reproducing the line it came from, which a typo rarely is."""
    session = seed_session("editable.wav")
    # This suite shares one store in one global order, so a check that seeds rows has to remove
    # them again — a later one asserts the exact contents of this table.
    before = len(client.get("/api/corrections").json())
    main.store.add_correction("缺消疫", "切削夜")
    main.store.add_correction("CNT", "吸菸")

    # The right-hand side was itself mistyped: fix it without touching what it matches.
    fixed = client.put("/api/corrections/缺消疫", json={"right": "切削液"}).json()
    assert {c["wrong"]: c["right"] for c in fixed}["缺消疫"] == "切削液"

    # The left-hand side is the key, so changing it is a rename: the old text stops matching.
    renamed = client.put("/api/corrections/缺消疫", json={"wrong": "缺خ疫", "right": "切削液"}).json()
    pairs = {c["wrong"]: c["right"] for c in renamed}
    assert "缺消疫" not in pairs and pairs["缺خ疫"] == "切削液"

    # Renaming onto an existing pair would drop whichever one the user was not looking at.
    assert client.put("/api/corrections/缺خ疫", json={"wrong": "CNT", "right": "切削液"}).status_code == 400
    assert client.put("/api/corrections/缺خ疫", json={"right": ""}).status_code == 400
    assert client.put("/api/corrections/缺خ疫", json={"wrong": "同", "right": "同"}).status_code == 400
    assert client.put("/api/corrections/nothing-here", json={"right": "x"}).status_code == 404

    # The rejected edits above left nothing behind: still the two seeded here and whatever existed.
    assert len(client.get("/api/corrections").json()) == before + 2
    assert client.get(f"/api/sessions/{session}/lines").status_code == 200

    client.delete("/api/corrections/缺خ疫")
    client.delete("/api/corrections/CNT")
    assert len(client.get("/api/corrections").json()) == before


def test_a_transcript_line_can_be_played_back(client: TestClient) -> None:
    """Correcting a line means judging text against audio; the page had only the text."""
    import soundfile as sf

    wav = config.RECORDINGS_DIR / "playable.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), np.zeros(config.SAMPLE_RATE * 120, dtype="float32"), config.SAMPLE_RATE)

    session = main.store.start_session("now", str(wav))
    short = main.store.add_line(session, 2.0, "S1", "zh", "短句", {}, end_time=5.0)
    # Longer than the 4s a voice sample uses: a sentence cut off there cannot be checked.
    long = main.store.add_line(session, 10.0, "S1", "zh", "很長的一句話", {}, end_time=25.0)
    unbounded = main.store.add_line(session, 40.0, "S1", "zh", "沒有結束時間", {})

    heard, rate = sf.read(io.BytesIO(client.get(f"/api/sessions/{session}/lines/{short}/clip").content))
    assert len(heard) == 3 * rate, len(heard)

    heard, rate = sf.read(io.BytesIO(client.get(f"/api/sessions/{session}/lines/{long}/clip").content))
    assert len(heard) == 15 * rate, len(heard)

    # No end_time: falls back to the sample length rather than reading to the end of the meeting.
    heard, rate = sf.read(io.BytesIO(client.get(f"/api/sessions/{session}/lines/{unbounded}/clip").content))
    assert len(heard) == main.CLIP_SECONDS * rate, len(heard)

    # A line id from another session must not be playable through this one's recording.
    other = seed_session("elsewhere.wav")
    assert client.get(f"/api/sessions/{other}/lines/{short}/clip").status_code == 404
    assert client.get(f"/api/sessions/{session}/lines/999999/clip").status_code == 404


def test_a_session_exports_as_markdown(client: TestClient) -> None:
    """The point of correcting 943 lines is taking the result somewhere else."""
    session = seed_session("exportable.wav")
    main.store.add_line(session, 65.0, "S2", "zh", "切削液要換了", {"en": "the coolant needs changing"})
    main.store.set_speaker_name(session, "S2", "王經理")

    md = client.get(f"/api/sessions/{session}/markdown")
    assert md.status_code == 200
    assert md.headers["content-type"].startswith("text/markdown")
    body = md.text

    # A named speaker appears by name; one nobody named is still listed, marked as such.
    assert "- **王經理**" in body
    assert "（未命名）" in body

    # Timestamp, speaker and the line itself, with the translation stacked under it.
    assert "**[1:05] 王經理**" in body
    assert "> 切削液要換了" in body
    assert "> _en_ the coolant needs changing" in body

    assert client.get("/api/sessions/999999/markdown").status_code == 404


def test_starting_without_models_says_which_file_is_missing(client: TestClient) -> None:
    """The one thing a first run gets wrong, and the app knows exactly what it is.

    Loading the recogniser happens while the request is still open, so a missing weights file left
    the route as an unhandled FileNotFoundError — a bare 500 with no body, which the dashboard can
    only render as "HTTP 500". The exception names the file; the answer is to say so.
    """
    import os

    from . import audio

    cfg = main.state["cfg"]
    saved = (cfg.whisper_model, config.WHISPER_DIRS["small"], audio.candidate_devices,
             os.environ.get("POLYMINUTES_NO_GPU"))
    # Deterministic on any machine: point the size this config resolves to at a path that is not
    # there (whisper_dir falls back to "small" for anything it does not know), keep the GPU
    # recogniser out of it since that one would succeed, and stub device resolution because it runs
    # first and fails on any runner without a virtual audio device.
    cfg.whisper_model = "small"
    config.WHISPER_DIRS["small"] = config.MODELS_DIR / "definitely-not-downloaded"
    audio.candidate_devices = lambda fragment: [None]
    os.environ["POLYMINUTES_NO_GPU"] = "1"
    try:
        started = client.post("/api/recording/start")
    finally:
        cfg.whisper_model, config.WHISPER_DIRS["small"], audio.candidate_devices, no_gpu = saved
        if no_gpu is None:
            os.environ.pop("POLYMINUTES_NO_GPU", None)
        else:
            os.environ["POLYMINUTES_NO_GPU"] = no_gpu

    assert started.status_code != 500, started.text
    assert started.status_code == 503, f"{started.status_code}: {started.text}"
    detail = started.json()["detail"]
    assert "model" in detail.lower(), detail
    # It has to name the thing to go and get, not just say something is missing.
    assert "whisper" in detail.lower() or "weights" in detail.lower(), detail

    # And the failed start must not leave the card claimed or a session half-open.
    assert client.get("/api/recording/status").json()["recording"] is False

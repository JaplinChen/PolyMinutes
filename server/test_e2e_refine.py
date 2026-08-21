"""The post-meeting pass: when it is queued, who gets the card, and what a failure leaves behind."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from . import jobs, main, store as store_mod
from .e2e_support import seed_session, wait_for


def test_stopping_a_recording_refines_it_without_being_asked(client: TestClient) -> None:
    """The whole point: a transcript's quality must not depend on anyone clicking a button.

    An imported recording was always refined on the way in. A meeting the room captured was not,
    so the same audio came out better when uploaded through the dashboard than when recorded in
    the room this was built for.

    Driven through `_stop_capture` rather than the HTTP endpoints because starting a real capture
    needs a sound card, and a test that quietly skips on the build machine is not a test.
    """
    jobs.reset()
    stub = main.postprocess
    stub.calls.clear()
    session_id = seed_session("stopped.wav")

    main.state.update(session=session_id, recorder=None, pipeline=None, gpu=False)
    main._stop_capture()

    assert wait_for(lambda: session_id in stub.calls), f"never refined: {stub.calls}"
    assert wait_for(
        lambda: client.get(f"/api/sessions/{session_id}/refine").json()["state"] == "refined"
    ), client.get(f"/api/sessions/{session_id}/refine").json()

    listed = next(s for s in client.get("/api/sessions").json() if s["id"] == session_id)
    assert listed["refine"]["state"] == "refined", listed
    assert stub.calls.count(session_id) == 1, stub.calls


def test_shutting_down_does_not_queue_a_pass_that_can_never_finish(client: TestClient) -> None:
    """The worker is a daemon thread: one queued on the way out is killed or holds the exit open."""
    jobs.reset()
    stub = main.postprocess
    stub.calls.clear()
    session_id = seed_session("shutdown.wav")

    main.state.update(session=session_id, recorder=None, pipeline=None, gpu=False)
    main._stop_capture(refine=False)
    time.sleep(0.05)
    assert stub.calls == [], stub.calls


def test_a_meeting_takes_the_card_back_from_a_running_pass(client: TestClient) -> None:
    """One GPU. A pass in flight must yield to a meeting starting now, and yield without damage.

    Two Whisper models on one 16 GB card pushes the live realtime factor past 1, and once that
    happens the capture backlog fills and the room's subtitles start dropping. The meeting always
    wins, and winning must cost the transcript nothing.
    """
    jobs.reset()
    stub = main.postprocess
    stub.calls.clear()
    stub.block.clear()  # the pass spins until it is asked to stop
    session_id = seed_session("held.wav")

    try:
        assert jobs.schedule(session_id, lambda cancel: main.postprocess.rewrite_session(
            main.store, session_id, None, None, None, should_stop=cancel.is_set))
        assert wait_for(lambda: session_id in stub.calls), "pass never started"
        assert jobs.state(session_id)["state"] == "refining"

        # This is what /api/recording/start does before it builds a Pipeline.
        assert jobs.claim_gpu(timeout=5.0), "the meeting never got the card"
        try:
            assert wait_for(lambda: jobs.state(session_id)["state"] == "cancelled"), \
                jobs.state(session_id)
            # Yielding cost nothing: the transcript it was part-way through rewriting is intact.
            assert [l["source"] for l in main.store.lines(session_id)] == ["精修前就在的一行"]
        finally:
            jobs.release_gpu()
    finally:
        stub.block.set()


def test_two_passes_over_one_session_do_not_overlap(client: TestClient) -> None:
    """Both would call replace_lines on the same session, and both would want the card."""
    jobs.reset()
    stub = main.postprocess
    stub.calls.clear()
    stub.block.clear()
    session_id = seed_session("twice.wav")

    try:
        assert jobs.schedule(session_id, lambda cancel: main.postprocess.rewrite_session(
            main.store, session_id, None, None, None, should_stop=cancel.is_set))
        assert wait_for(lambda: session_id in stub.calls)

        assert client.post(f"/api/sessions/{session_id}/reprocess").status_code == 409
        assert not jobs.schedule(session_id, lambda cancel: None)
    finally:
        stub.block.set()
        jobs.cancel_all(wait=1.0)


def test_llm_stages_do_not_hold_the_gpu_gate(client: TestClient) -> None:
    """The blocker this whole split exists for: a meeting must be able to start while a followup
    stage is still talking to a language model.

    Held inside the gate, a minutes-long Ollama call keeps `claim_gpu` waiting past its timeout
    and the room is told it cannot start recording — on account of work that was not using the
    GPU at all.
    """
    import threading

    jobs.reset()
    session_id = seed_session("llm-stage.wav")

    in_followup = threading.Event()
    release = threading.Event()

    def followup(cancel, set_stage):
        set_stage("summarize")
        in_followup.set()
        release.wait(10)

    try:
        assert jobs.schedule(session_id, lambda cancel: None, followup=followup)
        assert wait_for(in_followup.is_set), "followup never started"

        # The pass is mid-followup and must not be holding the card.
        assert jobs.state(session_id) == {"state": "refining", "stage": "summarize", "error": "",
                                          "done": 0, "total": 0, "skipped": 0}
        assert jobs.claim_gpu(timeout=1.0), "the LLM stage is holding the GPU gate"
        jobs.release_gpu()
    finally:
        release.set()
        jobs.cancel_all(wait=1.0)

    assert wait_for(lambda: jobs.state(session_id)["state"] in ("refined", "cancelled"))


def test_a_summarize_only_job_does_not_wait_for_the_gpu(client: TestClient) -> None:
    """Regenerating a summary must not queue behind a meeting recording on another session.

    A summarize-only job is the LLM alone over stored lines — it never touches the card. Scheduled
    with needs_gpu=True it entered the GPU gate anyway and, with the card held by a live recording,
    sat there for the length of that meeting. needs_gpu=False runs it straight to the followup.
    """
    import threading

    jobs.reset()
    session_id = seed_session("summarize-only.wav")

    ran = threading.Event()

    def followup(cancel, set_stage):
        set_stage("summarize")
        ran.set()

    # Hold the card as if a meeting on another session were recording.
    assert jobs.claim_gpu(timeout=1.0)
    try:
        # needs_gpu=True would block here; needs_gpu=False must reach the followup regardless.
        assert jobs.schedule(session_id, lambda cancel: None, followup=followup, needs_gpu=False)
        assert wait_for(ran.is_set, 2.0), "the summarize-only job waited for the GPU gate"
    finally:
        jobs.release_gpu()
        jobs.cancel_all(wait=1.0)

    assert wait_for(lambda: jobs.state(session_id)["state"] in ("refined", "cancelled"))


def test_a_gpu_job_still_waits_for_the_card(client: TestClient) -> None:
    """The default path is unchanged: a job that needs the card queues behind whoever holds it."""
    import threading

    jobs.reset()
    session_id = seed_session("gpu-job.wav")

    started = threading.Event()

    assert jobs.claim_gpu(timeout=1.0)
    try:
        assert jobs.schedule(session_id, lambda cancel: started.set())  # needs_gpu defaults True
        # It must NOT run while the card is held.
        assert not wait_for(started.is_set, 0.8), "a GPU job ran without the card"
    finally:
        jobs.release_gpu()
    # Once the card is free it proceeds.
    assert wait_for(started.is_set, 2.0), "the GPU job never ran after release"
    jobs.cancel_all(wait=1.0)


def test_a_failed_followup_keeps_what_the_rewrite_landed(client: TestClient) -> None:
    """Stages land independently: a summary that fails must not undo a refine that succeeded."""
    jobs.reset()
    session_id = seed_session("followup-fail.wav")

    def followup(cancel, set_stage):
        raise RuntimeError("summary model unreachable")

    assert jobs.schedule(session_id, lambda cancel: None, followup=followup)
    assert wait_for(lambda: jobs.state(session_id)["state"] == "failed")
    assert "summary model unreachable" in jobs.state(session_id)["error"]
    # The transcript the rewrite stage owns is untouched by the followup's failure.
    assert [l["source"] for l in main.store.lines(session_id)] == ["精修前就在的一行"]


def test_legacy_session_table_gains_lines_rev(tmp: Path) -> None:
    """Same trap as `status`/`end_time`: an old database's session table never gains the column."""
    import sqlite3

    path = tmp / "legacy-rev.db"
    old = sqlite3.connect(str(path))
    old.executescript(
        """
        CREATE TABLE session (id INTEGER PRIMARY KEY, started TEXT NOT NULL, ended TEXT,
                              wav_path TEXT NOT NULL);
        INSERT INTO session (started, wav_path) VALUES ('2026-01-01T09:00:00', 'old.wav');
        """
    )
    old.commit()
    old.close()

    st = store_mod.Store(path)
    try:
        row = st.session(1)
        assert row is not None and row["lines_rev"] == 0, row
        # And the migrated column actually moves when a line changes.
        st.add_line(1, 0.0, "S1", "zh", "一行", {})
        line_id = st.lines(1)[0]["id"]
        st.update_line(line_id, "改過的一行", {})
        assert st.session(1)["lines_rev"] == 1, st.session(1)
    finally:
        st.close()


def test_every_edit_path_moves_the_revision(tmp: Path) -> None:
    """update_line, replace_line and replace_lines each change what the transcript says.

    The revision is what lets the summary admit it describes an older transcript; an edit path
    that forgets to bump it makes stale look fresh, which is the exact lie this exists to stop.
    """
    st = store_mod.Store(tmp / "rev.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "r.wav")
        st.add_line(sid, 0.0, "S1", "zh", "原句", {})
        line_id = st.lines(sid)[0]["id"]
        assert st.session(sid)["lines_rev"] == 0

        st.update_line(line_id, "人工修正", {})
        assert st.session(sid)["lines_rev"] == 1

        st.replace_line(line_id, "重跑結果", "zh", {}, "ok")
        assert st.session(sid)["lines_rev"] == 2

        st.replace_lines(sid, [{"start": 0.0, "speaker": "S1", "lang": "zh",
                                "source": "精修結果", "translations": {}}])
        assert st.session(sid)["lines_rev"] == 3
    finally:
        st.close()


def test_a_hand_edit_keeps_its_pre_edit_text_and_a_rerun_drops_it(tmp: Path) -> None:
    """orig_source is the diff the transcript renders: set once by the first human edit,
    kept across further edits and retranslation, dropped when a re-run replaces the words."""
    st = store_mod.Store(tmp / "trace.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "r.wav")
        st.add_line(sid, 0.0, "S1", "zh", "原始辨識", {})
        line_id = st.lines(sid)[0]["id"]
        assert st.lines(sid)[0]["orig_source"] is None

        st.replace_line(line_id, "人工修正一", "zh", {}, "ok", refined=True)
        assert st.lines(sid)[0]["orig_source"] == "原始辨識"

        # A second edit keeps the first original, not the intermediate text.
        st.replace_line(line_id, "人工修正二", "zh", {}, "ok", refined=True)
        assert st.lines(sid)[0]["orig_source"] == "原始辨識"

        # Retranslate re-writes the line with the same words; the trace stands.
        st.replace_line(line_id, "人工修正二", "zh", {}, "ok")
        assert st.lines(sid)[0]["orig_source"] == "原始辨識"

        # A re-run that decoded different words replaces the human's edit; its trace would lie.
        st.replace_line(line_id, "重跑結果", "zh", {}, "ok")
        assert st.lines(sid)[0]["orig_source"] is None
    finally:
        st.close()


def test_a_reprocess_carries_hand_edits_onto_the_new_transcript(tmp: Path) -> None:
    """An edit is the only ground truth this system gets, and replace_lines used to drop it.

    Matched by time rather than by text, because the whole point of a reprocess is that the words
    come out different. A new line far longer than the edited one is a merge of several utterances,
    and writing one correction over all of them would delete its neighbours — so that one is left
    behind rather than misplaced.
    """
    st = store_mod.Store(tmp / "carry.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "r.wav")
        st.add_line(sid, 10.0, "S1", "zh", "前下足", {}, end_time=12.0)
        st.add_line(sid, 30.0, "S1", "zh", "誰也沒改過這句", {}, end_time=33.0)
        typed, untouched = (l["id"] for l in st.lines(sid))
        st.replace_line(typed, "前下組，支架有 7 pieces", "zh", {"en": "bracket, 7 pieces"},
                        "ok", refined=True)

        st.replace_lines(sid, [
            {"start": 10.2, "end_time": 12.1, "speaker": "S2", "lang": "zh",
             "source": "前下組", "translations": {"en": "front lower"}},
            {"start": 30.0, "end_time": 33.0, "speaker": "S2", "lang": "zh",
             "source": "重跑後的新句子", "translations": {}},
        ])
        after = st.lines(sid)
        assert after[0]["source"] == "前下組，支架有 7 pieces", after[0]
        assert after[0]["orig_source"] == "前下足", after[0]
        assert after[0]["refined"] == 1
        assert after[0]["translations"] == {"en": "bracket, 7 pieces"}, after[0]
        # The line nobody touched is whatever the re-decode said.
        assert after[1]["source"] == "重跑後的新句子", after[1]

        # A new line that swallowed the edited one plus its neighbours keeps its own words.
        st.replace_lines(sid, [{"start": 5.0, "end_time": 45.0, "speaker": "S2", "lang": "zh",
                                "source": "一整段合併起來的話", "translations": {}}])
        merged = st.lines(sid)
        assert len(merged) == 1 and merged[0]["source"] == "一整段合併起來的話", merged
    finally:
        st.close()


def test_an_empty_retranscription_does_not_wipe_the_existing_transcript(tmp: Path) -> None:
    """A re-transcription that decodes nothing must leave the old transcript standing.

    replace_lines deletes before it inserts, so an empty result is a bare delete. A pass whose
    utterances all came back empty (a near-silent segment, a decode that returned "" without
    raising) would otherwise erase a transcript that was already there — the same data loss the
    no-utterances early return already guards against. Here the utterance survives segmentation but
    transcribes to nothing; the stored line must still be there afterward.
    """
    import numpy as np

    from . import config
    from . import postprocess as pp

    st = store_mod.Store(tmp / "wipe.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "w.wav")
        st.add_line(sid, 0.0, "S1", "zh", "會議記錄不能被空結果抹掉", {})

        saved = {n: getattr(pp, n) for n in ("segment", "assign_speakers",
                                             "_remember_voices", "transcribe_all")}
        pp.segment = lambda wav: [pp.Utterance(start=0.0, samples=np.zeros(16000, dtype=np.float32))]
        pp.assign_speakers = lambda *a, **k: None
        pp._remember_voices = lambda *a, **k: None
        pp.transcribe_all = lambda *a, **k: None  # leaves every utterance's text ""
        maybe, diarizer = pp.asr_gpu.maybe, pp.diarize.Diarizer
        turns = pp.diarize.turns
        pp.asr_gpu.maybe = lambda *a, **k: object()
        pp.diarize.Diarizer = lambda *a, **k: object()
        pp.diarize.turns = lambda wav: []
        try:
            raised = False
            try:
                pp.rewrite_session(st, sid, Path("w.wav"), config.Config())
            except ValueError:
                raised = True
        finally:
            for n, fn in saved.items():
                setattr(pp, n, fn)
            pp.asr_gpu.maybe, pp.diarize.Diarizer, pp.diarize.turns = maybe, diarizer, turns

        assert raised, "an empty re-transcription should have refused rather than wiped"
        assert [l["source"] for l in st.lines(sid)] == ["會議記錄不能被空結果抹掉"]
    finally:
        st.close()


def test_the_auto_pass_records_no_llm_so_the_card_can_say_so(tmp: Path) -> None:
    """With no model configured the after-meeting pass cannot refine or summarize, but it must still
    record a no_llm summary state. The session card reads that as "configure an LLM"; a bare unset
    state is indistinguishable from a summary nobody has generated yet, which is the exact ambiguity
    the state exists to remove. The pass used to return early and leave it unset."""
    import threading

    from . import llm, postmeeting

    st = store_mod.Store(tmp / "nollm.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "n.wav")
        st.add_line(sid, 0.0, "S1", "zh", "會議內容", {})

        # LlmConfig defaults to provider=anthropic; an empty key makes chat_for return None.
        run = postmeeting.followup(st, ["zh"], llm.LlmConfig(), "", sid)
        run(threading.Event(), lambda _stage: None)

        summary = st.summary(sid)
        assert summary is not None and summary["status"] == "no_llm", summary
    finally:
        st.close()


def test_lines_with_rev_pairs_content_and_revision_from_one_read(tmp: Path) -> None:
    """The content and the revision come back consistent, so the summary cannot be built from one
    version of the transcript and stamped with another's revision.

    Reading store.lines() then store.session() separately let an edit land between them —
    replace_line takes the lock, changes a line and bumps the revision — so the summary was
    generated from old text yet stamped current, and never showed as stale. This pair is read under
    a single lock hold; the check is that the rev it returns matches the session's own count and the
    content is the content at that revision.
    """
    st = store_mod.Store(tmp / "atomic.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "a.wav")
        st.add_line(sid, 0.0, "S1", "zh", "第一版", {})
        line_id = st.lines(sid)[0]["id"]

        rows, rev = st.lines_with_rev(sid)
        assert rev == st.session(sid)["lines_rev"] == 0
        assert [r["source"] for r in rows] == ["第一版"]

        # After an edit, a fresh read reflects both the new text and the new revision together —
        # never the old text with the new number, which was the stale-summary bug.
        st.replace_line(line_id, "第二版", "zh", {}, "ok")
        rows2, rev2 = st.lines_with_rev(sid)
        assert rev2 == 1
        assert [r["source"] for r in rows2] == ["第二版"]
    finally:
        st.close()


def test_summary_and_rev_pairs_the_stored_summary_with_the_current_revision(tmp: Path) -> None:
    """The regenerate endpoint's freshness check reads both from one lock hold.

    Read apart — session() then summary() — an edit between them moves the revision under the
    comparison, so a summary made at rev N could be judged current against a transcript already at
    N+1 and a legitimate regeneration refused. The pair here is consistent by construction.
    """
    st = store_mod.Store(tmp / "sumrev.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "sr.wav")
        st.add_line(sid, 0.0, "S1", "zh", "一行", {})
        line_id = st.lines(sid)[0]["id"]

        # No summary yet: None, and the live revision.
        summ, rev = st.summary_and_rev(sid)
        assert summ is None and rev == 0

        st.set_summary(sid, '{"zh": {"title": "T"}}', "ok", lines_rev=0, created="2026-01-01T10:00:00")
        summ, rev = st.summary_and_rev(sid)
        assert summ["lines_rev"] == 0 and rev == 0   # matches → fresh

        # An edit moves the session revision; the stored summary's stamp stays behind.
        st.update_line(line_id, "改過", {})
        summ, rev = st.summary_and_rev(sid)
        assert summ["lines_rev"] == 0 and rev == 1   # differs → the endpoint would allow regeneration
    finally:
        st.close()


def test_lines_with_rev_is_atomic_under_concurrent_edits(tmp: Path) -> None:
    """Text and revision agree, hammered with edits from another thread.

    Each write sets the line's text to the revision it produces, so text and revision are locked
    together: after the Nth edit the line reads "v{N}" and lines_rev is N. A reader that took the
    old two-call form — store.lines() then store.session() — could observe "v5" with rev 6, because
    a write landed between the two calls. Under lines_with_rev the two come from one lock hold, so
    every snapshot must satisfy source == "v{rev}". One torn read fails the test.
    """
    import threading

    st = store_mod.Store(tmp / "race.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "race.wav")
        st.add_line(sid, 0.0, "S1", "zh", "v0", {})
        line_id = st.lines(sid)[0]["id"]

        stop = threading.Event()
        errors: list[str] = []

        def writer():
            while not stop.is_set():
                # rev is about to become one higher than it is now; write that number as the text.
                nxt = int(st.session(sid)["lines_rev"]) + 1
                st.replace_line(line_id, f"v{nxt}", "zh", {}, "ok")

        t = threading.Thread(target=writer, daemon=True)
        t.start()
        try:
            for _ in range(3000):
                rows, rev = st.lines_with_rev(sid)
                if rows and rows[0]["source"] != f"v{rev}":
                    errors.append(f"torn read: source={rows[0]['source']!r} rev={rev}")
                    break
        finally:
            stop.set()
            t.join(timeout=5)
        assert not errors, errors[0]
    finally:
        st.close()

def test_summary_roundtrip_and_cascade_delete(tmp: Path) -> None:
    st = store_mod.Store(tmp / "summary.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "s.wav")
        assert st.summary(sid) is None

        st.set_summary(sid, '{"zh": {"title": "週會"}}', "ok", lines_rev=0,
                       created="2026-01-01T10:00:00")
        row = st.summary(sid)
        assert row["json"] == '{"zh": {"title": "週會"}}' and row["status"] == "ok", row

        # Latest wins: regeneration overwrites in place.
        st.set_summary(sid, '{"zh": {"title": "週會 v2"}}', "partial", lines_rev=3,
                       created="2026-01-01T11:00:00")
        row = st.summary(sid)
        assert row["lines_rev"] == 3 and row["status"] == "partial", row

        # The summary dies with its session.
        with st._lock:
            st._db.execute("DELETE FROM session WHERE id=?", (sid,))
            st._db.commit()
        assert st.summary(sid) is None
    finally:
        st.close()


def test_a_database_made_before_the_columns_existed_gains_them(tmp: Path) -> None:
    """The meeting room's database predates `status` and `end_time`.

    `CREATE TABLE IF NOT EXISTS` does nothing to a table that already exists, so without a
    migration this only breaks where it matters: on the one machine holding real recordings, inside
    the capture thread, as a swallowed error count.
    """
    import sqlite3

    path = tmp / "legacy.db"
    old = sqlite3.connect(str(path))
    old.executescript(
        """
        CREATE TABLE session (id INTEGER PRIMARY KEY, started TEXT NOT NULL, ended TEXT,
                              wav_path TEXT NOT NULL);
        CREATE TABLE line (id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL, start REAL NOT NULL,
                           speaker TEXT NOT NULL, lang TEXT NOT NULL, source TEXT NOT NULL,
                           refined INTEGER NOT NULL DEFAULT 0);
        INSERT INTO session (started, wav_path) VALUES ('2026-01-01T09:00:00', 'old.wav');
        INSERT INTO line (session_id, start, speaker, lang, source) VALUES (1, 1.5, 'S1', 'zh', '舊的一行');
        """
    )
    old.commit()
    old.close()

    st = store_mod.Store(path)
    try:
        columns = {r[1] for r in st._db.execute("PRAGMA table_info(line)")}
        assert "status" in columns, columns
        assert "end_time" in columns, columns

        kept = st.lines(1)
        assert len(kept) == 1, kept
        assert kept[0]["source"] == "舊的一行", kept
        assert kept[0]["status"] == "ok", kept
        assert kept[0]["end_time"] is None, kept

        # And the migrated table still accepts writes, which is the part that was breaking.
        st.replace_lines(1, [{"start": 0.0, "speaker": "S1", "lang": "zh", "source": "新的一行",
                              "translations": {"en": "a new line"}, "status": "ok"}])
        assert [l["source"] for l in st.lines(1)] == ["新的一行"]
    finally:
        st.close()


def test_a_rewrite_drops_the_prints_of_codes_it_renumbered_away(tmp: Path) -> None:
    """A reprocess renumbers speakers from scratch, so a code that comes back means somebody else.

    Session 3 was holding sixteen prints for codes its transcript no longer had — S34 to S43 among
    them, left by the pass that merged those labels into one speaker. Nothing read them, which is
    exactly the danger: the next reprocess to produce an S34 would have named that voice from a
    print belonging to a different person, and a wrong print is self-consistent forever.
    """
    st = store_mod.Store(tmp / "prints.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "a.wav"))
        st.add_line(session_id, 0.0, "S1", "zh", "第一位", {})
        st.add_line(session_id, 5.0, "S2", "zh", "第二位", {})
        st.save_voiceprint(session_id, "S1", b"" * 8)
        st.save_voiceprint(session_id, "S2", b"" * 8)

        # The rewrite keeps S1 and drops S2 entirely.
        st.replace_lines(session_id, [{"start": 0.0, "speaker": "S1", "lang": "zh",
                                       "source": "只剩一位", "translations": {}}])

        assert st.voiceprint(session_id, "S1") == b"" * 8, "a surviving code kept its print"
        assert st.voiceprint(session_id, "S2") is None, "S2's print outlived the code it described"

        # Another session's prints are none of this session's business.
        other = st.start_session("2026-01-02T09:00:00", str(tmp / "b.wav"))
        st.add_line(other, 0.0, "S2", "zh", "別場會議", {})
        st.save_voiceprint(other, "S2", b"" * 8)
        st.replace_lines(session_id, [{"start": 0.0, "speaker": "S1", "lang": "zh",
                                       "source": "再寫一次", "translations": {}}])
        assert st.voiceprint(other, "S2") == b"" * 8, "pruned across sessions"
    finally:
        st.close()


def test_a_failed_rewrite_leaves_the_old_transcript_alone(tmp: Path) -> None:
    """Replacing a transcript is all-or-nothing.

    The failure this guards is not a slow path, it is data loss: the delete lands, the inserts do
    not, and the meeting's transcript is gone while its recording sits on disk looking fine.
    """
    st = store_mod.Store(tmp / "atomic.db")
    try:
        session_id = st.start_session("2026-01-01T09:00:00", str(tmp / "a.wav"))
        st.add_line(session_id, 0.0, "S1", "zh", "原本就在的一行", {"en": "already here"})
        before = st.lines(session_id)
        assert len(before) == 1, before

        # The fifth row is missing "source". The delete and four inserts have already run.
        rows = [{"start": float(i), "speaker": "S1", "lang": "zh", "source": f"新 {i}",
                 "translations": {}} for i in range(4)]
        rows.append({"start": 4.0, "speaker": "S1", "lang": "zh", "translations": {}})

        failed = False
        try:
            st.replace_lines(session_id, rows)
        except KeyError:
            failed = True
        assert failed, "replace_lines should have raised on the malformed row"

        after = st.lines(session_id)
        assert len(after) == 1, f"transcript was clobbered by a failed rewrite: {after}"
        assert after[0]["source"] == "原本就在的一行", after

        # The rollback must not have poisoned the connection for everyone else.
        st.add_line(session_id, 9.0, "S2", "en", "still writable", {})
        assert len(st.lines(session_id)) == 2
    finally:
        st.close()


def test_the_cpu_stages_of_a_pass_do_not_hold_the_card(client: TestClient) -> None:
    """Pressing record must not wait behind work that never touches the GPU.

    A pass spends most of its wall clock on the CPU: the VAD, the speaker segmentation (eight
    minutes on a 2h19m recording) and a translation round trip per line. Holding the card across
    all of it meant `claim_gpu` gave up after its thirty seconds and recording would not start.
    The gate now wraps the decode alone, so the card is free everywhere else.
    """
    import threading

    jobs.reset()
    session_id = seed_session("cpu-stages.wav")

    in_cpu_stage, release = threading.Event(), threading.Event()

    def run(cancel):
        # Stands in for everything before the decode.
        in_cpu_stage.set()
        release.wait(5.0)

    try:
        assert jobs.schedule(session_id, run, needs_gpu=False)
        assert wait_for(in_cpu_stage.is_set), "the pass never started"
        assert jobs.claim_gpu(timeout=1.0), "a CPU stage is holding the GPU gate"
        jobs.release_gpu()
    finally:
        release.set()
        jobs.cancel_all(wait=1.0)


def test_a_pass_can_take_the_card_while_its_caller_already_scheduled_it(client: TestClient) -> None:
    """The regression guard for the gate having two owners.

    `rewrite_session` takes the card around its decode. Three callers reach it — the import, the
    reprocess endpoint, and the automatic pass after a meeting stops — and if any of them also
    held the gate, the second acquire would block on a semaphore of one. The scheduled path waits
    without a timeout, so that shape hangs the worker forever and every later `claim_gpu` then
    fails: recording never starts again.

    This test would hang before the callers stopped taking the gate on the pass's behalf.
    """
    import threading

    jobs.reset()
    session_id = seed_session("gate-owner.wav")

    finished = threading.Event()

    def run(cancel):
        # What rewrite_session now does for its decode.
        with jobs.borrow_gpu(timeout=2.0):
            pass
        finished.set()

    try:
        assert jobs.schedule(session_id, run, needs_gpu=False)
        assert wait_for(finished.is_set, 4.0), "the pass could not take the card its caller left free"
    finally:
        jobs.cancel_all(wait=1.0)


def test_two_offline_passes_do_not_run_at_once(client: TestClient) -> None:
    """Narrowing the GPU gate removed the serialization it used to provide as a side effect.

    A pass used to hold the card start to finish, so a second could not begin. With the gate down
    to the decode, two of them would otherwise overlap — two recordings in memory and two sets of
    segmentation workers on the same box.
    """
    import threading

    jobs.reset()
    first, second = seed_session("pass-one.wav"), seed_session("pass-two.wav")
    running, release = threading.Event(), threading.Event()
    both = threading.Event()

    def hold(cancel):
        running.set()
        release.wait(5.0)

    def other(cancel):
        both.set()

    try:
        assert jobs.schedule(first, hold, needs_gpu=False)
        assert wait_for(running.is_set), "the first pass never started"
        assert jobs.schedule(second, other, needs_gpu=False)
        assert not wait_for(both.is_set, 0.8), "a second offline pass ran alongside the first"
        release.set()
        assert wait_for(both.is_set, 3.0), "the second pass never ran after the first let go"
    finally:
        release.set()
        jobs.cancel_all(wait=1.0)


def test_the_auto_refine_pass_leaves_a_human_corrected_line_alone(tmp: Path) -> None:
    """A hand-typed correction is the only ground truth this system gets. `update_line` marks the
    line `refined` precisely so it is never rewritten twice — but the auto refine pass read every
    line and overwrote whatever the model proposed, silently clobbering the correction a user made
    while (or just before) the pass ran. The refined flag must actually gate the write-back."""
    from . import postmeeting

    st = store_mod.Store(tmp / "refined-guard.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "g.wav")
        human = st.add_line(sid, 0.0, "S1", "zh", "機器聽錯的原文", {})
        auto = st.add_line(sid, 1.0, "S1", "zh", "另一行", {})
        st.update_line(human, "人工修正過的正確原文", {})  # a person fixed this line → refined=1

        class RewriteEverything:
            def __init__(self, chat, topic="會議"):
                pass

            def refine(self, lines, terms=None, rejected=None, coverage=None, on_progress=None):
                return [f"機器改寫{i}" for i, _ in enumerate(lines)]

        saved = postmeeting.refine.Refiner
        postmeeting.refine.Refiner = RewriteEverything
        try:
            postmeeting._refine_stage(st, sid, lambda _p: "")
        finally:
            postmeeting.refine.Refiner = saved

        rows = {r["id"]: r["source"] for r in st.lines(sid)}
        assert rows[human] == "人工修正過的正確原文", "the human correction was clobbered"
        assert rows[auto] == "機器改寫1", "the unrefined line should still get refined"
    finally:
        st.close()


def test_the_model_decides_where_the_sentences_end(tmp: Path) -> None:
    """Boundaries belong to whatever reads the words, not to a character counter.

    The arithmetic stops merging when a limit runs out, which lands mid-sentence: 19 rows across
    three real meetings ended exactly at the 120-character cap. Here the model is handed the run
    the arithmetic says *could* join and answers where the sentences actually end — the cut points
    are still the recording's own fragment boundaries, so no timestamp is invented.
    """
    from . import postmeeting

    st = store_mod.Store(tmp / "grouping.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "g.wav")
        for i, text in enumerate(["我們今天要討論的是", "交期的問題", "另外一件事情是品質"]):
            st.add_line(sid, i * 3.5, "S1", "zh", text, {}, end_time=i * 3.5 + 3.0)

        seen = []

        def chat(prompt: str) -> str:
            seen.append(prompt)
            if "待補標點" in prompt:  # the punctuation pass that follows
                return "NONE"
            return "1-2: 我們今天要討論的是，交期的問題。\n3-3: 另外一件事情是品質。"

        postmeeting._segment_stage(st, sid, chat)
        assert any("待補標點" not in p for p in seen), "the grouping question was never asked"
        rows = st.lines(sid)
        assert [r["source"] for r in rows] == ["我們今天要討論的是，交期的問題。",
                                               "另外一件事情是品質。"], rows
        # The merged row keeps the second fragment's real end, not an interpolated one.
        assert rows[0]["start"] == 0.0 and rows[0]["end_time"] == 6.5
        assert rows[0]["refined"] == 0, "grouping must not block the refine pass"
    finally:
        st.close()


def test_a_grouping_that_rewrites_the_words_is_refused(tmp: Path) -> None:
    """The guard is the same one punctuation has, applied over the whole run: strip the punctuation
    and the text must be exactly what went in, or the arithmetic decides as it always did."""
    from . import postmeeting

    st = store_mod.Store(tmp / "grouping-guard.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "g2.wav")
        for i, text in enumerate(["我們今天要討論的是", "交期的問題"]):
            st.add_line(sid, i * 3.5, "S1", "zh", text, {}, end_time=i * 3.5 + 3.0)

        calls = []

        def chat(prompt: str) -> str:
            calls.append(prompt)
            if "待補標點" in prompt:
                return "NONE"
            # Drops "今天" while regrouping — plausible-looking, and wrong.
            return "1-2: 我們要討論的是，交期的問題。"

        postmeeting._segment_stage(st, sid, chat)
        rows = st.lines(sid)
        assert len(calls) >= 1
        # Refused, so the arithmetic merged them instead and the words are intact.
        assert len(rows) == 1 and "今天" in rows[0]["source"], rows
    finally:
        st.close()


def test_a_merged_line_stops_growing_in_time_as_well_as_characters(tmp: Path) -> None:
    """Slow speech reaches ninety seconds while still under the character limit.

    The VAD never returns an utterance past max_speech_duration (20s), so every longer line on the
    page was built by this merge — and the only limit it had was MAX_MERGED_CHARS, which sparse
    speech never reaches. Measured across three real meetings: 273 of 1456 lines ran past 20s, the
    longest 90.1 seconds for 118 characters — one row nobody can follow, and one clip nobody can
    scrub. The cut costs nothing invented: the pieces keep the VAD's own boundaries and own text.
    """
    from . import postmeeting, segment

    st = store_mod.Store(tmp / "long-merge.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "long.wav")
        # Six 8-second pieces, four characters each: 51 seconds, nowhere near 120 characters.
        for i in range(6):
            st.add_line(sid, i * 8.5, "S1", "zh", "慢慢講的", {}, end_time=i * 8.5 + 8.0)
        postmeeting._segment_stage(st, sid, None)

        rows = st.lines(sid)
        spans = [r["end_time"] - r["start"] for r in rows]
        assert max(spans) <= segment.MAX_MERGED_SECONDS, spans
        # Still merged, just not without end: six pieces became fewer rows, not six.
        assert 1 < len(rows) < 6, [(r["start"], r["end_time"]) for r in rows]
        # Every second of speech survives the split — the pieces still tile the original span.
        assert rows[0]["start"] == 0.0 and rows[-1]["end_time"] == 5 * 8.5 + 8.0
    finally:
        st.close()


def test_vad_cut_fragments_are_merged_and_punctuated(tmp: Path) -> None:
    """The segment stage joins fragments the VAD cut mid-sentence, then restores punctuation —
    without marking anything refined, so the correction pass that follows still gets its turn."""
    from . import postmeeting

    st = store_mod.Store(tmp / "segment-stage.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "seg.wav")
        st.add_line(sid, 0.0, "S1", "zh", "我們今天要討論的是", {"en": "what we discuss is"},
                    end_time=2.0)
        st.add_line(sid, 2.3, "S1", "zh", "交期的問題", {"en": "the delivery problem"},
                    end_time=4.0)
        st.add_line(sid, 9.0, "S2", "zh", "好", {}, end_time=9.5)

        def chat(prompt: str) -> str:
            # The grouping question comes first and carries the fragments separately; the
            # punctuation pass that follows sees whatever grouping produced.
            if "待補標點" not in prompt:
                assert "1: 我們今天要討論的是" in prompt and "2: 交期的問題" in prompt, prompt
                return "1-2: 我們今天要討論的是，交期的問題。"
            assert "我們今天要討論的是，交期的問題。" in prompt, prompt
            return "NONE"

        postmeeting._segment_stage(st, sid, chat)

        rows = st.lines(sid)
        assert [r["source"] for r in rows] == ["我們今天要討論的是，交期的問題。", "好"], rows
        assert rows[0]["end_time"] == 4.0, rows[0]
        assert rows[0]["translations"] == {"en": "what we discuss is the delivery problem"}, rows[0]
        assert rows[0]["refined"] == 0, "punctuation must not block the refine pass"
        assert st.session(sid)["lines_rev"] >= 1, st.session(sid)

        # With no LLM the merge still happens — it is pure arithmetic over timestamps.
        sid2 = st.start_session("2026-01-01T10:00:00", "seg2.wav")
        st.add_line(sid2, 0.0, "S1", "zh", "沒有模型", {}, end_time=1.0)
        st.add_line(sid2, 1.2, "S1", "zh", "也要合併", {}, end_time=2.0)
        postmeeting._segment_stage(st, sid2, None)
        assert [r["source"] for r in st.lines(sid2)] == ["沒有模型也要合併"]
    finally:
        st.close()


def test_a_merge_does_not_close_over_somebody_elses_interruption(tmp: Path) -> None:
    """A gap the other speaker was talking in is a speaker change, not a breath.

    The splitter cuts where the segmenter heard the change, but a piece too short to decode leaves
    no line, and the hole it leaves reads here exactly like a VAD cut. Merging across it puts two
    people back on one line under one name — 6 of 14 cuts on a real 2.7h meeting, including the one
    reported: 636.5s–643.2s, 林瓚文 and 吳仲琪 as a single 林瓚文 line.
    """
    import json

    from . import config, postmeeting

    st = store_mod.Store(tmp / "interruption.db")
    wav = config.RECORDINGS_DIR / "interrupted.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    try:
        # No audio needed — only the cached turns beside it, which is what the stage reads.
        wav.write_bytes(b"")
        turns = [[636.51, 639.20, 11], [639.20, 639.74, 18], [639.74, 643.18, 11]]
        for path, key in ((wav.with_suffix(".wav.turns.json"), None),):
            path.write_text(json.dumps({"key": key, "duration": 700.0, "turns": turns}),
                            encoding="utf-8")

        sid = st.start_session("2026-01-01T09:00:00", str(wav))
        st.add_line(sid, 636.51, "S12", "zh", "他們有些自己請出來", {}, end_time=639.20)
        st.add_line(sid, 639.74, "S12", "zh", "我們還是一樣先出去對", {}, end_time=643.18)

        # A stale cache key means no barriers, and the merge behaves as it always did.
        postmeeting._segment_stage(st, sid, None)
        assert len(st.lines(sid)) == 1, "without turns the gap is still just a gap"

        # The real cache, written under the key this recording hashes to.
        from . import diarize
        key = diarize._turns_key(wav, config.SPEAKER_THRESHOLD)
        diarize._cache_path(wav).write_text(
            json.dumps({"key": key, "duration": 700.0, "turns": turns}), encoding="utf-8")

        sid2 = st.start_session("2026-01-01T10:00:00", str(wav))
        st.add_line(sid2, 636.51, "S12", "zh", "他們有些自己請出來", {}, end_time=639.20)
        st.add_line(sid2, 639.74, "S12", "zh", "我們還是一樣先出去對", {}, end_time=643.18)
        postmeeting._segment_stage(st, sid2, None)

        rows = st.lines(sid2)
        assert len(rows) == 2, [(r["start"], r["source"]) for r in rows]
    finally:
        st.close()
        diarize._cache_path(wav).unlink(missing_ok=True)
        wav.with_suffix(".wav.turns.json").unlink(missing_ok=True)
        wav.unlink(missing_ok=True)


def test_a_punctuation_batch_that_fails_once_is_retried(tmp: Path) -> None:
    """LLM 批次偶發失敗（超時、壞 JSON）一次就丟掉整批太浪費——重試一次通常就過。"""
    from . import postmeeting

    st = store_mod.Store(tmp / "segment-retry.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "retry.wav")
        st.add_line(sid, 0.0, "S1", "zh", "第一句沒有標點", {}, end_time=5.0)

        calls = []

        def flaky(prompt: str) -> str:
            calls.append(1)
            if len(calls) == 1:
                raise RuntimeError("boom")
            return "1: 第一句，沒有標點。"

        postmeeting._segment_stage(st, sid, flaky)
        assert [r["source"] for r in st.lines(sid)] == ["第一句，沒有標點。"]
        assert len(calls) == 2, calls
    finally:
        st.close()


def test_a_twice_failed_punctuation_batch_is_counted_as_skipped(tmp: Path) -> None:
    """重試後仍失敗的批次保留原文並計入 skipped，讓儀表板說得出「幾行沒處理到」。"""
    from . import postmeeting

    jobs.reset()
    st = store_mod.Store(tmp / "segment-skip.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "skip.wav")
        st.add_line(sid, 0.0, "S1", "zh", "壞掉的一批", {}, end_time=5.0)

        def broken(prompt: str) -> str:
            raise RuntimeError("boom")

        def followup(cancel, set_stage):
            set_stage("segment")
            postmeeting._segment_stage(st, sid, broken)

        assert jobs.schedule(sid, lambda cancel: None, followup=followup, needs_gpu=False)
        assert wait_for(lambda: (jobs.state(sid) or {}).get("state") == "refined")
        s = jobs.state(sid)
        assert (s["skipped"], s["done"], s["total"]) == (1, 1, 1), s
        assert [r["source"] for r in st.lines(sid)] == ["壞掉的一批"]
    finally:
        st.close()
        jobs.cancel_all(wait=1.0)


def test_markdown_export_lists_speakers_in_numeric_order(tmp: Path) -> None:
    """The exported transcript's speaker list is sorted by code. Lexicographic sort put S10 right
    after S1 and ahead of S2 once a meeting had ten or more speakers (the app allows up to S35),
    so the downloaded file showed speakers in visibly wrong order."""
    from . import postprocess

    st = store_mod.Store(tmp / "md.db")
    try:
        sid = st.start_session("2026-01-01T09:00:00", "m.wav")
        for i in (1, 2, 10):
            st.add_line(sid, float(i), f"S{i}", "zh", f"第 {i} 句", {})

        md = postprocess.to_markdown(st, sid)
        section = md.split("## 發言者")[1].split("## 逐字稿")[0]
        listed = [ln for ln in section.splitlines() if ln.startswith("- ")]
        assert listed == ["- **S1**（未命名）", "- **S2**（未命名）", "- **S10**（未命名）"], listed
    finally:
        st.close()

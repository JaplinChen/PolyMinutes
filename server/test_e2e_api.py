"""Config, glossary, LLM settings and the two things an edit teaches the room."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from . import config, llm, llm_probe, main, translate


def test_health_and_config_roundtrip(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    # From the VERSION file, not a literal: the dashboard overwrites its own build-time version with
    # this answer, so a literal here decided what the sidebar said no matter what was released.
    assert health["version"] == config.VERSION
    assert health["version"] == (config.ROOT / "VERSION").read_text(encoding="utf-8").strip()

    cfg = client.get("/api/config").json()
    assert cfg["languages"] == ["zh", "vi", "en"]
    assert cfg["display"]["show_source"] == "top"

    r = client.put("/api/config", json={"languages": ["zh", "vi"], "display": {"font_size": 64, "theme": "light"}})
    assert r.status_code == 200, r.text
    assert r.json()["languages"] == ["zh", "vi"]
    assert r.json()["display"]["font_size"] == 64
    # An unspecified field must survive a partial patch.
    assert r.json()["display"]["lines"] == 6


def test_config_rejects_bad_input(client: TestClient) -> None:
    assert client.put("/api/config", json={"languages": ["zh"]}).status_code == 400
    assert client.put("/api/config", json={"languages": ["zh", "zh"]}).status_code == 400
    assert client.put("/api/config", json={"display": {"lines": 99}}).status_code == 400
    assert client.put("/api/config", json={"display": {"show_source": "sideways"}}).status_code == 400


def test_glossary_crud(client: TestClient) -> None:
    r = client.post("/api/glossary", json={"source": "產能", "targets": {"vi": "công suất", "en": "capacity"}})
    assert r.status_code == 200, r.text
    assert r.json()[0]["targets"]["vi"] == "công suất"

    # `keep` is the code-switching case: the term must never be translated.
    client.post("/api/glossary", json={"source": "schedule", "mode": "keep"})
    modes = {t["source"]: t["mode"] for t in client.get("/api/glossary").json()}
    assert modes == {"產能": "translate", "schedule": "keep"}

    assert client.post("/api/glossary", json={"source": "x", "mode": "nonsense"}).status_code == 400
    assert client.post("/api/glossary", json={"source": "   "}).status_code == 400

    left = client.request("DELETE", "/api/glossary", params={"source": "schedule"}).json()
    assert [t["source"] for t in left] == ["產能"]


def test_glossary_reports_what_a_term_would_overwrite(client: TestClient) -> None:
    """Adding a term is not obviously destructive, which is the problem.

    料號 and 料耗 are both liaohao and 料耗 is a term of the trade; adding 料號 rewrote it
    forty-two times across seven interviews and nothing said so. The answer comes from the
    meetings already recorded — what these people say, not what Mandarin permits.
    """
    session = main.store.start_session("now", "x.wav")
    for text in ("這個料耗的部分", "料耗變動原則", "刀具的料耗很多"):
        main.store.add_line(session, 1.0, "S1", "zh", text, {})

    hits = client.get("/api/glossary/collisions", params={"source": "料號"}).json()
    assert hits["collisions"] == [{"text": "料耗", "count": 3}]

    # A word already in the glossary is not collateral: registering it is how two real
    # homophones are made to coexist.
    client.post("/api/glossary", json={"source": "料耗", "mode": "protect"})
    assert client.get("/api/glossary/collisions", params={"source": "料號"}).json()["collisions"] == []

    assert client.get("/api/glossary/collisions", params={"source": "交貨"}).json()["collisions"] == []
    assert client.get("/api/glossary/collisions", params={"source": " "}).status_code == 400


def test_llm_config_never_returns_the_key(client: TestClient) -> None:
    client.put("/api/translate/config", json={
        "llmProvider": "anthropic", "llmModel": "claude-opus-5", "llmApiKey": "sk-secret-value-1234",
        "llmEndpoint": "https://api.anthropic.com", "llmTemperature": 0,
        "llmFallbackModels": [], "llmProviderConfigs": {},
    })
    body = client.get("/api/translate/config").json()
    assert body["llmApiKey"] == ""
    assert body["apiKeySet"] is True
    assert "sk-secret-value-1234" not in json.dumps(body)

    # A blank key on a later save means "keep the stored one", not "clear it".
    client.put("/api/translate/config", json={"llmModel": "claude-sonnet-5", "llmApiKey": ""})
    body = client.get("/api/translate/config").json()
    assert body["apiKeySet"] is True
    assert body["llmModel"] == "claude-sonnet-5"


def test_keyproxy_masks_and_rotates(client: TestClient) -> None:
    client.post("/api/keyproxy/keys", json={"provider": "anthropic", "apiKey": "sk-aaaabbbbccccdddd", "account": "a"})
    client.post("/api/keyproxy/keys", json={"provider": "anthropic", "apiKey": "sk-eeeeffffgggghhhh", "account": "b"})
    listed = client.get("/api/keyproxy/keys").json()
    assert len(listed) == 2
    assert all("sk-aaaa" not in json.dumps(k) or k["masked"].startswith("sk-a") for k in listed)
    assert listed[0]["masked"] == "sk-a…dddd"

    got = {main.keys.next_key("anthropic") for _ in range(2)}
    assert got == {"sk-aaaabbbbccccdddd", "sk-eeeeffffgggghhhh"}, "rotation did not visit both keys"

    assert client.post("/api/keyproxy/keys", json={"provider": "anthropic", "apiKey": ""}).status_code == 400
    assert client.delete("/api/keyproxy/keys/anthropic/9").status_code == 404
    assert client.delete("/api/keyproxy/keys/anthropic/0").json() == client.get("/api/keyproxy/keys").json()


def test_rejection_classifies_the_provider_error(tmp: Path) -> None:
    class Err(Exception):
        def __init__(self, code):
            self.status_code = code

    assert llm.rejection(Err(429)) == "limited"
    assert llm.rejection(Err(401)) == "failed"
    assert llm.rejection(Err(403)) == "failed"
    assert llm.rejection(Err(500)) is None       # a server fault is not the key's problem
    assert llm.rejection(Exception("no status")) is None


def test_a_rejected_key_is_skipped_and_a_rate_limited_one_recovers(tmp: Path) -> None:
    """mark_failure benches a key so the rotation stops handing it out: a bad key for good, a
    rate-limited one only until its cooldown passes."""
    ks = llm.KeyStore(tmp / "rotate-keys.json")
    ks.add("anthropic", "sk-key-alpha-0001")
    ks.add("anthropic", "sk-key-bravo-0002")

    # A hard rejection (wrong or disabled key) drops out of the rotation entirely.
    ks.mark_failure("sk-key-alpha-0001", limited=False)
    assert {ks.next_key("anthropic") for _ in range(4)} == {"sk-key-bravo-0002"}

    # A rate limit benches the last usable key too: nothing ready, so next_key reports none.
    ks.mark_failure("sk-key-bravo-0002", limited=True)
    assert ks.next_key("anthropic") is None, "a cooling-down key must not be handed out"

    # Once its window elapses it heals back to ready and returns to the rotation.
    for k in ks._keys:
        if k.key == "sk-key-bravo-0002":
            k.limited_until = 0.0
    assert ks.next_key("anthropic") == "sk-key-bravo-0002"


def test_a_rejected_translation_benches_its_key(tmp: Path) -> None:
    """The wiring end to end: a provider 429 during translation marks the key limited, and the next
    rotation skips it — the failure the pool exists to route around no longer repeats silently."""
    ks = llm.KeyStore(tmp / "reject-keys.json")
    ks.add("anthropic", "sk-only-key-9999")

    class RateLimited(Exception):
        status_code = 429

    def on_reject(exc):
        if kind := llm.rejection(exc):
            ks.mark_failure("sk-only-key-9999", limited=(kind == "limited"))

    def chat(_prompt):
        raise RateLimited()

    tr = translate.Translator(chat, on_reject=on_reject)

    raised = False
    try:
        tr.translate(translate.Line("hello", "en"), ["zh"])
    except RateLimited:
        raised = True
    assert raised, "the provider error must propagate, not be swallowed"
    assert ks.next_key("anthropic") is None, "the benched key must be skipped by the next rotation"


def test_naming_a_speaker_teaches_the_room_their_voice(client: TestClient) -> None:
    """The one piece of labelled data this system ever gets, kept instead of discarded."""
    session = main.store.start_session("now", "x.wav")
    main.store.save_voiceprint(session, "S1", np.array([1.0], dtype="float32").tobytes())

    client.put(f"/api/sessions/{session}/speakers", json={"S1": "Vincent"})
    assert [s["name"] for s in client.get("/api/speakers/known").json()] == ["Vincent"]

    # A speaker with no stored voiceprint is still nameable, it just teaches nothing.
    client.put(f"/api/sessions/{session}/speakers", json={"S9": "Nobody"})
    assert [s["name"] for s in client.get("/api/speakers/known").json()] == ["Vincent"]

    assert client.delete("/api/speakers/known/Vincent").json() == []


def test_naming_a_reassigned_code_learns_the_voice_from_its_lines(client: TestClient) -> None:
    """A code minted by reassigning lines has no diariser voiceprint; naming it must derive one
    from the audio its lines point at instead of silently teaching nothing."""
    import soundfile as sf

    from . import routes_speakers

    wav = config.RECORDINGS_DIR / "derive.wav"
    wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(wav), np.zeros(config.SAMPLE_RATE * 10, dtype="float32"), config.SAMPLE_RATE)
    session = main.store.start_session("now", str(wav))
    main.store.add_line(session, 1.0, "S31", "zh", "夠長的一句話", {}, end_time=4.0)

    original = routes_speakers._embed
    routes_speakers._embed = lambda samples, rate: np.array([1.0, 0.0], dtype="float32")
    NAME = "審查改派學聲"
    try:
        client.put(f"/api/sessions/{session}/speakers", json={"S31": NAME})
    finally:
        routes_speakers._embed = original
    assert main.store.voiceprint(session, "S31") is not None
    assert any(n == NAME for n, _ in main.store.known_voiceprints())
    client.delete(f"/api/speakers/known/{NAME}")


def test_naming_one_person_across_codes_learns_every_voice(client: TestClient) -> None:
    """One person split across several codes must teach every variant, not just the last.

    The diariser flattens a room unevenly: a voice that drifts gets minted as S1, S17, S20. Naming
    all of them the same person used to keep only the print saved last (name was the primary key),
    so three of four voice variants were thrown away. Now each distinct print is stored, and a
    near-duplicate adds nothing.
    """
    session = main.store.start_session("2026-02-02T09:00:00", "multi.wav")
    NAME = "審查臨時多聲"
    # Two orthogonal prints — genuinely different variants of the one person.
    main.store.save_voiceprint(session, "S1", np.array([1.0, 0.0, 0.0, 0.0], "float32").tobytes())
    main.store.save_voiceprint(session, "S2", np.array([0.0, 1.0, 0.0, 0.0], "float32").tobytes())
    client.put(f"/api/sessions/{session}/speakers", json={"S1": NAME, "S2": NAME})
    assert sum(n == NAME for n, _ in main.store.known_voiceprints()) == 2

    # A near-duplicate of the first print teaches nothing, so the count holds.
    main.store.save_voiceprint(session, "S3", np.array([1.0, 0.0, 0.0, 0.01], "float32").tobytes())
    client.put(f"/api/sessions/{session}/speakers", json={"S3": NAME})
    assert sum(n == NAME for n, _ in main.store.known_voiceprints()) == 2

    # Forgetting the voice drops every variant, not just the representative one.
    client.delete(f"/api/speakers/known/{NAME}")
    assert sum(n == NAME for n, _ in main.store.known_voiceprints()) == 0


def test_clearing_session_names_drops_the_stale_mapping(client: TestClient) -> None:
    """A reprocess renumbers codes, so the old (code -> name) rows must be cleared before the new
    ones are written, or a name lands on whoever the re-clustering happened to call S1."""
    session = main.store.start_session("2026-03-03T09:00:00", "clear.wav")
    main.store.set_speaker_name(session, "S1", "審查臨時清除")
    assert main.store.speaker_names(session).get("S1") == "審查臨時清除"
    main.store.clear_speaker_names(session)
    assert main.store.speaker_names(session) == {}


def test_naming_one_person_across_codes_learns_every_voice(client: TestClient) -> None:
    """One person split across several codes must teach every variant, not just the last.

    The diariser flattens a room unevenly: a voice that drifts gets minted as S1, S17, S20. Naming
    all of them the same person used to keep only the print saved last (name was the primary key),
    so three of four voice variants were thrown away. Now each distinct print is stored, and a
    near-duplicate adds nothing.
    """
    session = main.store.start_session("2026-02-02T09:00:00", "multi.wav")
    NAME = "審查臨時多聲"
    # Two orthogonal prints — genuinely different variants of the one person.
    main.store.save_voiceprint(session, "S1", np.array([1.0, 0.0, 0.0, 0.0], "float32").tobytes())
    main.store.save_voiceprint(session, "S2", np.array([0.0, 1.0, 0.0, 0.0], "float32").tobytes())
    client.put(f"/api/sessions/{session}/speakers", json={"S1": NAME, "S2": NAME})
    assert sum(n == NAME for n, _ in main.store.known_voiceprints()) == 2

    # A near-duplicate of the first print teaches nothing, so the count holds.
    main.store.save_voiceprint(session, "S3", np.array([1.0, 0.0, 0.0, 0.01], "float32").tobytes())
    client.put(f"/api/sessions/{session}/speakers", json={"S3": NAME})
    assert sum(n == NAME for n, _ in main.store.known_voiceprints()) == 2

    # Forgetting the voice drops every variant, not just the representative one.
    client.delete(f"/api/speakers/known/{NAME}")
    assert sum(n == NAME for n, _ in main.store.known_voiceprints()) == 0


def test_a_known_voice_can_be_forced_to_a_language(client: TestClient) -> None:
    """The room's Vietnamese speaker is set to vi once, by name, and stays vi — auto-detect flips
    a Chinese-heavy room's voices between zh and vi, so a recognised voice should not be left to it.
    """
    session = main.store.start_session("2026-04-04T09:00:00", "lang.wav")
    NAME = "審查臨時越語"
    main.store.save_voiceprint(session, "S1", np.array([1.0, 0.0, 0.0, 0.0], "float32").tobytes())
    client.put(f"/api/sessions/{session}/speakers", json={"S1": NAME})

    # Defaults to auto-detect until set.
    row = next(s for s in client.get("/api/speakers/known").json() if s["name"] == NAME)
    assert row["language"] == ""

    updated = client.put(f"/api/speakers/known/{NAME}/language", json={"language": "vi"})
    assert updated.status_code == 200
    assert next(s for s in updated.json() if s["name"] == NAME)["language"] == "vi"
    assert main.store.speaker_languages()[NAME] == "vi"

    # A language the room does not run is refused rather than silently forcing a missing recogniser.
    assert client.put(f"/api/speakers/known/{NAME}/language", json={"language": "fr"}).status_code == 400

    # '' returns the voice to auto-detect.
    client.put(f"/api/speakers/known/{NAME}/language", json={"language": ""})
    assert main.store.speaker_languages()[NAME] == ""

    # Leave the shared store as it was found — a later test asserts the exact known-voice list.
    client.delete(f"/api/speakers/known/{NAME}")


def test_speaker_session_count_is_distinct_meetings_not_saves(client: TestClient) -> None:
    """The "N meetings" figure counts meetings the voice was named in, not times it was saved.

    Naming a speaker then fixing a typo in the name both save, and remember_speaker used to +1 on
    each — so a within-meeting rename read as an extra meeting. Counted from speaker_name, the
    figure is the number of distinct sessions regardless of how many saves produced it.
    """
    a = main.store.start_session("2026-01-01T09:00:00", "a.wav")
    main.store.save_voiceprint(a, "S1", np.array([1.0], dtype="float32").tobytes())

    # Names unique to this check, so a shared-store predecessor cannot seed them.
    OLD, NEW = "審查臨時甲", "審查臨時乙"
    # Name, then correct the name — same meeting, two saves.
    client.put(f"/api/sessions/{a}/speakers", json={"S1": OLD})
    client.put(f"/api/sessions/{a}/speakers", json={"S1": NEW})
    known = {s["name"]: s["sessions"] for s in client.get("/api/speakers/known").json()}
    assert OLD not in known, known  # the old name is gone — no orphan on the Learned page
    assert known.get(NEW) == 1, known  # one meeting, not two saves

    # The same person named in a second meeting is two distinct meetings.
    b = main.store.start_session("2026-01-02T09:00:00", "b.wav")
    main.store.save_voiceprint(b, "S1", np.array([1.0], dtype="float32").tobytes())
    client.put(f"/api/sessions/{b}/speakers", json={"S1": NEW})
    known = {s["name"]: s["sessions"] for s in client.get("/api/speakers/known").json()}
    assert known.get(NEW) == 2, known

    # Shared store: leave the table as it was found.
    from urllib.parse import quote
    client.delete(f"/api/speakers/known/{quote(NEW)}")


def test_correcting_a_misnamed_speaker_unlearns_the_polluting_print(client: TestClient) -> None:
    """Changing a code's name from B to A removes from B the variant that caused the mismatch.

    Without this, the wrong print stays under B and misnames the same voice next meeting. Only the
    close-enough variant goes: B's own genuinely-different print survives the correction.
    """
    from urllib.parse import quote
    A, B = "審查誤認甲", "審查誤認乙"
    v_wrong = np.array([1.0, 0.0], "float32").tobytes()  # the voice that is actually A's
    v_b = np.array([0.0, 1.0], "float32").tobytes()      # B's real voice, orthogonal
    main.store.remember_speaker(B, v_b)

    sid = main.store.start_session("2026-04-01T09:00:00", "u.wav")
    main.store.save_voiceprint(sid, "S1", v_wrong)
    client.put(f"/api/sessions/{sid}/speakers", json={"S1": B})   # misnamed — B learns A's voice
    client.put(f"/api/sessions/{sid}/speakers", json={"S1": A})   # corrected

    prints = main.store.known_voiceprints()
    assert (A, v_wrong) in prints, prints            # A learned the voice
    assert (B, v_wrong) not in prints, prints        # the polluting print left B
    assert (B, v_b) in prints, prints                # B's real print survived

    client.delete(f"/api/speakers/known/{quote(A)}")
    client.delete(f"/api/speakers/known/{quote(B)}")


def test_forgetting_a_speaker_leaves_no_count_behind(client: TestClient) -> None:
    """forget_speaker drops the voice from known_speaker but keeps its historical transcript names.

    speaker_sessions is joined to known_speaker so it counts only voices the room still knows —
    without that join a forgotten voice kept a phantom count, invisible only because get_known_speakers
    happens to ask about it through known_speakers(). The two now agree at the source.
    """
    from urllib.parse import quote
    name = "審查忘記測試"
    sid = main.store.start_session("2026-03-01T09:00:00", "f.wav")
    main.store.save_voiceprint(sid, "S1", np.array([1.0], dtype="float32").tobytes())
    client.put(f"/api/sessions/{sid}/speakers", json={"S1": name})
    assert {s["name"]: s["sessions"] for s in client.get("/api/speakers/known").json()}.get(name) == 1

    client.delete(f"/api/speakers/known/{quote(name)}")
    # Gone from the list, and gone from the count that backs it — no phantom left in either source.
    assert name not in {s["name"] for s in client.get("/api/speakers/known").json()}
    assert name not in main.store.speaker_sessions()
    # The past meeting keeps the name it was given — forget is not a transcript edit.
    assert main.store.speaker_names(sid).get("S1") == name


def test_editing_a_line_teaches_the_correction(client: TestClient) -> None:
    """An edit on the transcript page is the only ground truth this system gets: someone who was
    in the room saying what was actually said. Kept, the same mistake is fixed everywhere next
    time — live as well as after the fact."""
    session = main.store.start_session("now", "x.wav")
    line = main.store.add_line(session, 1.0, "S1", "zh", "那個申管會上系統", {})

    r = client.put(f"/api/sessions/{session}/lines/{line}", json={"source": "那個生管會上系統"})
    assert r.status_code == 200, r.text
    assert r.json()["lines"][0]["source"] == "那個生管會上系統"
    assert {c["wrong"]: c["right"] for c in client.get("/api/corrections").json()} == {"申管": "生管"}

    # What was learned is applied to text the recogniser has not seen yet.
    from . import correct as correct_mod
    fixed = correct_mod.Corrector([], main.store.corrections()).fix("剛剛申管講的")
    assert fixed == "剛剛生管講的"

    assert client.put(f"/api/sessions/{session}/lines/{line}", json={"source": " "}).status_code == 400
    assert client.put(f"/api/sessions/{session}/lines/9999", json={"source": "x"}).status_code == 404
    assert client.delete("/api/corrections/申管").json() == []


def test_an_edited_line_is_retranslated_and_kept_from_the_refiner(client: TestClient) -> None:
    """The translations came from the wrong words, so an edit has to redo them.

    And having redone them by hand, the line is marked refined: the post-meeting pass rewriting
    what someone typed in the room is the one thing that flag exists to prevent.
    """
    session = main.store.start_session("now", "x.wav")
    line = main.store.add_line(session, 1.0, "S1", "zh", "申管會", {"en": "stale", "vi": "cũ"})

    made = []

    class Echo:
        def translate(self, ln, targets, context=None, previous=None, terms=None, prev_targets=None):
            made.append(ln.text)
            return translate.Result({t: f"[{t}] {ln.text}" for t in targets})

    real = main._make_translator
    main._make_translator = lambda: Echo()
    try:
        r = client.put(f"/api/sessions/{session}/lines/{line}", json={"source": "生管會"})
        assert r.status_code == 200, r.text
        assert made == ["生管會"]
        edited = next(l for l in r.json()["lines"] if l["id"] == line)
        assert edited["translations"]["en"] == "[en] 生管會"
        assert main.store.line(line)["refined"] == 1

        # Same text, translated again: for a line whose words are right and whose rendering is not.
        r = client.post(f"/api/sessions/{session}/lines/{line}/retranslate")
        assert r.status_code == 200, r.text
        assert made == ["生管會", "生管會"]
        assert client.post(f"/api/sessions/{session}/lines/9999/retranslate").status_code == 404
    finally:
        main._make_translator = real


def test_every_script_flag_is_wired_to_something(tmp: Path) -> None:
    """A flag that nothing reads is a feature that silently stopped existing.

    scripts/learn_terms.py kept its --max-sound option for a while after a scripted edit removed
    the ranking that used it, so the tool went on accepting the flag and ignoring it. Nothing in
    the output said so.
    """
    import re

    scripts = sorted(Path("scripts").glob("*.py"))
    assert scripts, "no scripts found; is the working directory wrong?"
    for path in scripts:
        source = path.read_text(encoding="utf-8")
        for flag in re.findall(r'add_argument\("--([a-z-]+)"', source):
            assert f"args.{flag.replace('-', '_')}" in source, f"{path.name}: --{flag} is unused"


def test_the_llm_probe_endpoints_answer_in_the_shape_the_page_reads(client: TestClient) -> None:
    """Both buttons on the provider form. The provider itself is stubbed; the wiring is the point.

    These two routes did not exist for a while, and because an unknown /api/* path returns a JSON
    404 rather than the HTML shell, the page reported it as a provider that refused the key —
    a backend gap wearing the costume of a wrong API key.
    """
    seen: list[tuple] = []
    real_check, real_list = llm_probe.check, llm_probe.list_models
    try:
        llm_probe.check = lambda *a: (seen.append(a), (True, "ok"))[1]
        llm_probe.list_models = lambda *a: (seen.append(a), ["m-2", "m-1"])[1]

        body = {"provider": "openai", "endpoint": "https://api.openai.com/v1/chat/completions",
                "model": "gpt-4o", "apiKey": "sk-typed"}
        r = client.post("/api/translate/llm/test", json=body)
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "message": "ok"}
        assert seen[-1] == ("openai", "https://api.openai.com/v1/chat/completions", "gpt-4o", "sk-typed")

        assert client.post("/api/translate/llm/models", json=body).json() == {"models": ["m-2", "m-1"]}

        # No endpoint: the provider's default is used rather than 400ing on a field the page
        # deliberately hides for providers whose endpoint is fixed.
        client.post("/api/translate/llm/test", json={"provider": "groq", "model": "x", "apiKey": "k"})
        assert seen[-1][1] == "https://api.groq.com/openai/v1"

        # A provider the page could not know the endpoint of, and no endpoint typed.
        assert client.post("/api/translate/llm/test", json={"provider": "azure"}).status_code == 400
        assert client.post("/api/translate/llm/test", json={"model": "x"}).status_code == 400

        # A provider that refuses is a 502 on the models route: the page shows the reason, and a
        # 200 with an empty list would read as "this provider has no models".
        def refuse(*_a):
            raise llm_probe.ProbeError("401 Unauthorized")

        llm_probe.list_models = refuse
        assert client.post("/api/translate/llm/models", json=body).status_code == 502
    finally:
        llm_probe.check, llm_probe.list_models = real_check, real_list


def test_the_llm_probe_falls_back_to_the_key_already_stored(client: TestClient) -> None:
    """The page cannot send a key it was never given: `llmApiKey` comes back empty by design.

    Without this, Verify worked once — right after typing the key — and failed on every later
    visit, which reads as a key that expired rather than a page that never had one.
    """
    client.put("/api/translate/config", json={"llmProvider": "anthropic", "llmApiKey": "sk-stored"})
    seen: list[tuple] = []
    real_check = llm_probe.check
    try:
        llm_probe.check = lambda *a: (seen.append(a), (True, "ok"))[1]
        client.post("/api/translate/llm/test", json={"provider": "anthropic", "model": "claude-opus-5"})
        assert seen[-1][3] == "sk-stored", seen
    finally:
        llm_probe.check = real_check


def test_an_endpoint_that_is_not_http_is_refused_by_the_route(client: TestClient) -> None:
    """This process fetches whatever the endpoint box holds, so the scheme is checked at the edge.

    400 rather than 502: nothing upstream failed — there is no upstream for file://.
    """
    for endpoint in ("file:///C:/Windows/win.ini", "ftp://example.com/x"):
        body = {"provider": "ollama", "endpoint": endpoint, "model": "x"}
        assert client.post("/api/translate/llm/models", json=body).status_code == 400, endpoint
        assert client.post("/api/translate/llm/test", json=body).status_code == 400, endpoint


def test_an_unknown_api_path_says_it_is_the_build_not_the_data(client: TestClient) -> None:
    """A stale server answers 404 for every endpoint it has not been restarted into.

    That is the same status a live endpoint uses to say "no such line", so the detail has to
    distinguish them — otherwise the dashboard reports missing data that is sitting right there.
    """
    gone = client.get("/api/sessions/1/lines/1/clip-that-does-not-exist")
    assert gone.status_code == 404
    assert gone.json()["detail"] == main.NO_SUCH_ENDPOINT

    # A real endpoint's 404 must not look like it.
    real = client.get("/api/sessions/999999/lines/1/clip")
    assert real.status_code == 404 and real.json()["detail"] != main.NO_SUCH_ENDPOINT


def test_renaming_a_speaker_onto_another_is_refused_not_a_silent_merge(tmp: Path) -> None:
    """Renaming one learned voice to a name another voice already owns used to DELETE the other's
    voiceprint and merge two people into one name — no error, unrecoverable. It must be refused."""
    from . import store as store_mod

    st = store_mod.Store(tmp / "rename.db")
    try:
        alice, bob = np.array([1.0, 0.0], "float32").tobytes(), np.array([0.0, 1.0], "float32").tobytes()
        sid = st.start_session("2026-01-01T09:00:00", "r.wav")
        st.set_speaker_name(sid, "S1", "Alice")
        st.remember_speaker("Alice", alice)
        st.set_speaker_name(sid, "S2", "Bob")
        st.remember_speaker("Bob", bob)

        raised = False
        try:
            st.rename_speaker("Bob", "Alice")
        except ValueError:
            raised = True
        assert raised, "renaming onto an existing speaker must be refused"

        # Both voices survive with their own prints — nothing merged, nothing deleted.
        prints = dict(st.known_speakers())
        assert prints == {"Alice": alice, "Bob": bob}, prints
    finally:
        st.close()

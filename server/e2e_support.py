"""Fixtures and stubs shared by the e2e checks.

Real audio and a real Claude key are not available here, so the translator and the post-meeting
pass are stubbed and the pipeline is driven directly. Everything in this module exists so the
checks in `test_e2e_*` can state what they are testing without restating how to get there.
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path

import numpy as np

from . import config, jobs, llm, main, postmeeting as postmeeting_mod
from . import postprocess as postprocess_mod
from . import retry as retry_mod, store as store_mod, translate
from .pipeline import Pipeline


def isolate(tmp: Path) -> None:
    """Point every persistent path at a temp dir so a test run cannot touch real data."""
    config.CONFIG_PATH = tmp / "config.json"
    config.RECORDINGS_DIR = tmp / "recordings"
    llm.LLM_PATH = tmp / "llm.json"
    llm.KEYS_PATH = tmp / "llm_keys.json"
    main.store = store_mod.Store(tmp / "test.db")
    main.keys = llm.KeyStore(tmp / "llm_keys.json")
    main.state["cfg"] = config.Config()
    main.state["llm"] = llm.LlmConfig()
    # Stopping a recording now queues the post-meeting pass, and the real one loads a Whisper
    # model. Every lifecycle test would pull one in, so the pass records that it was asked instead.
    main.postprocess = StubPostprocess()
    # Same for the LLM stages, and more urgently: a developer machine with ANTHROPIC_API_KEY in
    # the environment would otherwise make real API calls from inside the test suite — the first
    # run of these checks did exactly that, and failed on the provider's 401 instead of its own
    # assertion. The stages' real logic is covered by test_summarize and the jobs tests.
    main.postmeeting = StubPostmeeting()


class StubPostmeeting:
    """Stands in for the LLM stages: records what was asked, writes nothing, calls nothing.

    `chat_for` is here too because the ask endpoint reaches the model through it — a real key in
    the environment would otherwise send the question to Anthropic from inside the test. Tests that
    exercise /api/ask set `replies` to the JSON they want the model to return, in order.
    """

    def __init__(self) -> None:
        self.followups: list[int] = []
        self.summarize_calls: list[int] = []
        self.segment_calls: list[int] = []
        self.replies: list[str] | None = None
        self.prompts: list[str] = []

    def followup(self, store, languages, llm_cfg, api_key, session_id):
        def run(cancel, set_stage):
            self.followups.append(session_id)
        return run

    def _segment_stage(self, store, session_id, chat):
        # Recorded, then the real thing: with chat=None it is pure store arithmetic — no LLM to
        # keep out of the tests — and the merge endpoint's join behaviour deserves real coverage.
        self.segment_calls.append(session_id)
        postmeeting_mod._segment_stage(store, session_id, chat)

    def _summarize_stage(self, store, session_id, languages, llm_cfg, api_key, cancel):
        self.summarize_calls.append(session_id)

    def chat_for(self, llm_cfg, api_key, max_tokens, model="", **_):
        # None means "no model configured", the 503 path. A list means "answer with these".
        if self.replies is None:
            return None
        queue = list(self.replies)

        def chat(prompt: str) -> str:
            self.prompts.append(prompt)
            return queue.pop(0) if queue else "{}"
        return chat


class StubPostprocess:
    """Stands in for the postprocess module: records calls, honours cancellation, writes nothing."""

    def __init__(self) -> None:
        self.calls: list[int] = []
        self.subtitle_calls: list[int] = []
        # Set means "return at once". A test that needs a pass to still be running clears it.
        self.block = threading.Event()
        self.block.set()

    def subtitle_session(self, store, session_id, cues, lang, cfg, translator=None,
                         should_stop=None):
        self.subtitle_calls.append(session_id)
        return cues

    def rewrite_session(self, store, session_id, wav, cfg, translator=None, should_stop=None,
                        gpu=contextlib.nullcontext):
        self.calls.append(session_id)
        # Entered and left the way the real one does, so a caller that also holds the gate shows
        # up here as a hang rather than passing quietly.
        with gpu():
            pass
        while not self.block.is_set():
            if should_stop and should_stop():
                raise jobs.Cancelled()
            time.sleep(0.005)
        return []

    def to_markdown(self, store, session_id):
        return postprocess_mod.to_markdown(store, session_id)

    def to_docx(self, store, session_id):
        return postprocess_mod.to_docx(store, session_id)


class StubTranslator:
    """Echoes a deterministic translation, and revises the previous line on the third call."""

    def __init__(self) -> None:
        self.calls = 0

    def translate(self, line, targets, context=None, previous=None, terms=None, prev_targets=None):
        self.calls += 1
        out = {t: f"[{t}] {line.text}" for t in targets}
        if self.calls == 3 and previous is not None:
            return translate.Result(out, "corrected source", {t: f"[{t}] corrected" for t in targets})
        return translate.Result(out)


def headless_pipeline(cfg, store, session_id, translator, emit) -> Pipeline:
    """A Pipeline with everything `_handle` needs and nothing it does not.

    `Pipeline.__init__` builds a VAD, which loads silero_vad.onnx — 1.5 GB of models are not in
    version control, so on a bare runner that raises. These checks drive `_handle` directly and
    never feed the VAD, and skipping them where the models are absent would mean the retry logic
    is only ever verified on the one machine that has them, which is not verification.

    Fields are set explicitly rather than copied from `__init__`: if one is added there and missed
    here, these fail with AttributeError rather than quietly testing the wrong thing.
    """
    pipe = Pipeline.__new__(Pipeline)
    pipe._cfg, pipe._store, pipe._session = cfg, store, session_id
    pipe._translator, pipe._emit = translator, emit
    pipe._diarizer = OneSpeaker()
    pipe._transcriber = ByLanguage({})
    pipe._hotwords = ""
    pipe._context, pipe._previous = [], None
    pipe._retries = retry_mod.Retries()
    pipe.errors = pipe.backlog_peak = 0
    return pipe


class ByLanguage:
    """Transcriber returning a canned result per forced language."""

    def __init__(self, table: dict[str, tuple[str, str]]) -> None:
        self.table = table

    def set_hotwords(self, hotwords: str) -> None:
        pass

    def transcribe(self, samples, language):
        return self.table.get(language, ("", language))


class FixedTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text

    def set_hotwords(self, hotwords: str) -> None:
        pass

    def transcribe(self, samples, language):
        return self.text, "zh"


class OneSpeaker:
    """One speaker whose language is unknown until an utterance actually decodes."""

    class _S:
        code = "S1"
        centroid = np.zeros(4, dtype="float32")

    def __init__(self) -> None:
        self.recognised: dict = {}
        self.language = ""
        self.votes: list[str] = []
        self._speaker = self._S()

    def assign(self, samples, source=""):
        return self._speaker

    def language_for(self, speaker):
        return self.language

    def observe_language(self, speaker, used):
        self.votes.append(used)
        if used:
            self.language = used


def wait_for(predicate, seconds: float = 5.0) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def seed_session(name: str) -> int:
    """A finished session with one line and a real file on disk, without needing a microphone."""
    config.RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    wav = config.RECORDINGS_DIR / name
    wav.write_bytes(b"")
    session_id = main.store.start_session("2026-01-01T09:00:00", str(wav))
    main.store.end_session(session_id, "2026-01-01T10:00:00")
    main.store.add_line(session_id, 0.0, "S1", "zh", "精修前就在的一行", {})
    return session_id

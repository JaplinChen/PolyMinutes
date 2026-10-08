"""Self-checks for multi-provider chat wiring. Run: python -m server.test_provider_chat

Model-free and offline: urllib.request.urlopen is faked, so these assert the request SHAPE each
provider builds and that the dispatcher routes to it — the bug was that every provider but Anthropic
(and Ollama, post-meeting only) fell through to the Anthropic client, so a verified OpenAI/Gemini/
Groq key silently translated nothing.
"""

from __future__ import annotations

import io
import json
import urllib.request

from . import llm, postmeeting, refine, translate


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def _capture(reply: dict):
    """Patch urlopen to record the request and return `reply` as JSON. Returns (calls, restore)."""
    calls: list[urllib.request.Request] = []
    real = urllib.request.urlopen

    def fake(req, timeout=None):
        calls.append(req)
        return _FakeResponse(json.dumps(reply).encode())

    urllib.request.urlopen = fake
    return calls, (lambda: setattr(urllib.request, "urlopen", real))


def test_openai_chat_posts_chat_completions_with_bearer():
    reply = {"choices": [{"message": {"content": "hola"}}]}
    calls, restore = _capture(reply)
    try:
        chat = refine.openai_chat("sk-key", "gpt-4o", "https://api.openai.com/v1", max_tokens=50)
        out = chat("hi")
    finally:
        restore()
    assert out == "hola"
    req = calls[0]
    assert req.full_url == "https://api.openai.com/v1/chat/completions"
    assert req.headers["Authorization"] == "Bearer sk-key"
    assert json.loads(req.data)["model"] == "gpt-4o"


def test_gemini_chat_posts_generatecontent_with_key_in_query():
    reply = {"candidates": [{"content": {"parts": [{"text": "bonjour"}]}}]}
    calls, restore = _capture(reply)
    try:
        chat = refine.gemini_chat("gk", "gemini-2.0-flash",
                                  "https://generativelanguage.googleapis.com", max_tokens=50)
        out = chat("hi")
    finally:
        restore()
    assert out == "bonjour"
    assert calls[0].full_url.endswith("/models/gemini-2.0-flash:generateContent?key=gk")


def test_chat_for_routes_openai_provider_to_openai_shape():
    """The regression guard: provider=openai must NOT reach the Anthropic client."""
    cfg = llm.LlmConfig(provider="openai", model="gpt-4o",
                        endpoint="https://api.openai.com/v1")
    reply = {"choices": [{"message": {"content": "ok"}}]}
    calls, restore = _capture(reply)
    try:
        chat = postmeeting.chat_for(cfg, "sk-key", max_tokens=50)
        assert chat is not None
        chat("hi")
    finally:
        restore()
    assert calls[0].full_url == "https://api.openai.com/v1/chat/completions"


def test_chat_for_model_override_picks_a_per_function_model():
    """Per-function split: translation and summary can point at different models; empty falls back
    to the one configured model, so a room that sets neither behaves exactly as before."""
    cfg = llm.LlmConfig(provider="openai", model="base-model", endpoint="https://api.openai.com/v1")
    reply = {"choices": [{"message": {"content": "ok"}}]}
    calls, restore = _capture(reply)
    try:
        postmeeting.chat_for(cfg, "sk", max_tokens=50, model="aya-expanse:8b")("hi")
        postmeeting.chat_for(cfg, "sk", max_tokens=50)("hi")  # empty override
    finally:
        restore()
    assert json.loads(calls[0].data)["model"] == "aya-expanse:8b", "override used"
    assert json.loads(calls[1].data)["model"] == "base-model", "empty override falls back to base"


def test_chat_for_returns_none_without_key_for_cloud_provider():
    cfg = llm.LlmConfig(provider="groq", model="llama", endpoint="")
    assert postmeeting.chat_for(cfg, "", max_tokens=50) is None


def test_chat_for_falls_back_to_local_ollama_without_key():
    """Privacy mode: cloud provider chosen, no key, but a local Ollama model is configured — the
    stage runs on this machine instead of returning no_llm."""
    cfg = llm.LlmConfig(provider="anthropic", model="claude-opus-5", endpoint="", api_key="",
                        providers={"ollama": {"model": "qwen3:14b", "endpoint": ""}})
    reply = {"response": "本地摘要"}
    calls, restore = _capture(reply)
    try:
        chat = postmeeting.chat_for(cfg, "", max_tokens=50)
        assert chat is not None, "a configured local model should carry the stage"
        out = chat("summarize")
    finally:
        restore()
    assert out == "本地摘要"
    # Routed to the local daemon with the Ollama model, never the cloud model name.
    assert calls[0].full_url == "http://localhost:11434/api/generate"
    assert json.loads(calls[0].data)["model"] == "qwen3:14b"


def test_chat_for_no_fallback_when_no_local_model_configured():
    """No Ollama model set up means no fallback — still no_llm, not a broken call to a blank model."""
    cfg = llm.LlmConfig(provider="anthropic", model="claude-opus-5", endpoint="", api_key="")
    assert postmeeting.chat_for(cfg, "", max_tokens=50) is None


def test_translator_uses_injected_chat_callable():
    seen: list[str] = []

    def chat(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps({"translations": {"en": "hello"}})

    tr = translate.Translator(chat)
    result = tr.translate(translate.Line("你好", "zh"), ["en"])
    assert result.translations == {"en": "hello"}
    assert len(seen) == 1  # the prompt reached the injected callable, not an SDK


def test_previous_refinement_targets_the_previous_lines_language():
    """In a mixed-language meeting the previous line's language differs from the current one, so its
    hindsight correction must be solicited and kept in ITS languages, not the current line's targets.
    cfg=[zh,en], previous line zh (needs an en correction), current line en (targets=[zh])."""
    seen: list[str] = []

    def chat(prompt: str) -> str:
        seen.append(prompt)
        return json.dumps({"translations": {"zh": "你好"},
                           "previous": {"source": "前一句修正", "translations": {"en": "corrected"}}})

    tr = translate.Translator(chat)
    prev = translate.Line("前一句中文", "zh")
    result = tr.translate(translate.Line("hello", "en"), ["zh"],
                          previous=prev, prev_targets=["en"])

    assert result.previous_translations == {"en": "corrected"}, result.previous_translations
    # The previous block must ask for the language the previous line actually needs (en), not zh.
    prev_block = seen[0].split('"previous"')[1]
    assert '"en"' in prev_block and '"zh"' not in prev_block, prev_block


def test_translator_reports_rejection_then_reraises():
    rejected: list[Exception] = []

    def chat(prompt: str) -> str:
        raise RuntimeError("boom")

    tr = translate.Translator(chat, on_reject=rejected.append)
    try:
        tr.translate(translate.Line("你好", "zh"), ["en"])
        raise AssertionError("expected the error to re-raise")
    except RuntimeError:
        pass
    assert len(rejected) == 1


def test_translate_retries_a_malformed_json_reply() -> None:
    """A local model emits unparseable JSON sporadically; a re-ask parses it, so no line is lost.

    Measured: 179 of 862 lines on one meeting came back unparseable, and re-asking parsed them all.
    """
    replies = iter(["not json at all", "{bad json", '{"translations": {"en": "hello"}}'])
    calls: list[str] = []

    def chat(prompt: str) -> str:
        calls.append(prompt)
        return next(replies)

    res = translate.Translator(chat).translate(translate.Line("你好", "zh"), ["en"])
    assert res.translations == {"en": "hello"}, res.translations
    assert len(calls) == 3, "retried until it parsed"


def test_a_reply_missing_its_last_brace_still_parses() -> None:
    """The real aya reply logged on 2026-10-08: four objects opened, three closed."""
    raw = ('{"translations": {"vi": "Vì vậy, kế hoạch ngày 9 tháng 10, chúng tôi mời nhà máy Úc cần chú ý.", '
           '"en": "So, the plan for October 9th, we invite the Australian factory to be aware."}, '
           '"previous": {"source": "然後關於BMW訂單 我們上週也已經報告過我們計劃是在10月9號要出1200T", '
           '"translations": {"vi": "Sau đó, về đơn đặt hàng BMW", "en": "Regarding the BMW order"}}')
    res = translate.parse_response(raw, ["vi", "en"])
    assert res.translations["en"].startswith("So, the plan"), res.translations
    assert res.previous_translations["en"] == "Regarding the BMW order", res
    # Unbalanced the other way is not a brace that went missing; still an error.
    raised = False
    try:
        translate.parse_response('{"translations": {"en": "x"}}}', ["en"])
    except ValueError:
        raised = True
    assert raised


def test_translate_surfaces_a_reply_that_never_parses() -> None:
    """Malformed on every attempt is raised, not swallowed as a silently empty translation."""
    raised = False
    try:
        translate.Translator(lambda p: '{"translations": {"en": "say "hi""}}').translate(
            translate.Line("你好", "zh"), ["en"])
    except ValueError as exc:
        raised = True
        # The reply travels with the error, so the log shows what the model wrote.
        assert 'say "hi"' in str(exc), exc
    assert raised


def test_translate_does_not_retry_a_provider_rejection() -> None:
    """A rejected key benches at once — retrying the parse loop would hammer a dud key three times."""
    calls: list[str] = []
    rejected: list[Exception] = []

    def chat(prompt: str) -> str:
        calls.append(prompt)
        raise RuntimeError("rate limited")

    raised = False
    try:
        translate.Translator(chat, on_reject=rejected.append).translate(
            translate.Line("你好", "zh"), ["en"])
    except RuntimeError:
        raised = True
    assert raised and len(calls) == 1 and len(rejected) == 1, (raised, len(calls), len(rejected))


def main() -> None:
    checks = sorted((n, f) for n, f in globals().items() if n.startswith("test_"))
    for name, fn in checks:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(checks)} passed")


if __name__ == "__main__":
    main()

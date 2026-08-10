"""Reaching an LLM provider to answer two questions the settings page asks.

"Does this work?" and "what models can I pick?" — for a provider, an endpoint and a key the user
has typed but not saved yet. Nothing here is on the translation path; it exists so the page can
tell someone their key is wrong before a meeting does.

Four request shapes cover the nine providers the page offers: Anthropic's own, the OpenAI-shaped
majority, Gemini, and Ollama. They differ in where the key goes and where the model list lives,
which is the whole reason this is not one function.

The key never appears in anything this module returns. Gemini in particular carries it in the
query string, so error text is scrubbed rather than trusted — a failed test that echoed the URL
would put the key on screen and in the log.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request

log = logging.getLogger("polyminutes.llm")

# `?key=…` / `&api_key=…` in anything a provider echoed back at us.
_KEY_IN_URL = re.compile(r"([?&](?:key|api_key)=)[^&\s\"']+")

# Providers that speak the OpenAI REST shape: Bearer auth, /models, /chat/completions.
OPENAI_SHAPED = ("openai", "groq", "mistral", "openrouter", "nvidia_nim")
ANTHROPIC_VERSION = "2023-06-01"
# Listing models is a lookup and answers immediately.
LIST_TIMEOUT = 20.0
# Answering one is not. A local model has to be pulled into VRAM first, and on a cold Ollama that
# took longer than the 20 s this used to allow — so the test button reported a timeout for a
# provider that was working, which is the exact wrong answer for a button that exists to tell you
# whether it works.
CHAT_TIMEOUT = 120.0
# Enough to prove the round trip. A test button that spent real tokens would be a tax on checking.
TEST_PROMPT = "hi"


class ProbeError(Exception):
    """A provider said no, or could not be reached. The message is safe to show the user."""


def base_url(endpoint: str) -> str:
    """The provider root, given whatever the settings page had in its endpoint box.

    The page stores a chat URL for some providers and a bare root for others — Ollama's
    `/api/chat`, OpenAI's `/chat/completions`, Gemini's `/v1beta` — because that is what each
    provider's own documentation shows. Normalising here keeps that inconsistency out of every
    call site below.
    """
    url = endpoint.strip().rstrip("/")
    for suffix in ("/chat/completions", "/api/chat", "/v1/messages", "/api/generate"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def _require_http(url: str) -> str:
    """Only http(s). The endpoint is user-supplied and this process fetches it."""
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ProbeError("endpoint must be an http or https URL")
    return url


def _scrub(text: str, api_key: str) -> str:
    """Never let a key back out, whatever the provider echoed."""
    out = text
    if api_key:
        out = out.replace(api_key, "…").replace(urllib.parse.quote(api_key), "…")
    # Gemini and friends put it in the query string, so a URL in the error text is not safe either.
    return _KEY_IN_URL.sub(r"\1…", out)


def _request(url: str, api_key: str, headers: dict | None = None, body: dict | None = None,
             timeout: float = LIST_TIMEOUT) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(_require_http(url), data=data, headers=headers or {},
                                 method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = (exc.read() or b"").decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise ProbeError(_scrub(f"{exc.code} {exc.reason}: {detail}".strip(), api_key)) from exc
    except urllib.error.URLError as exc:
        raise ProbeError(_scrub(f"could not reach the endpoint: {exc.reason}", api_key)) from exc
    # A read that times out raises straight out of the socket rather than as a URLError, so without
    # this a slow provider reached the models route as a 500 instead of an answer.
    except TimeoutError as exc:
        raise ProbeError(f"no answer within {timeout:.0f}s — a local model may still be loading") from exc
    except OSError as exc:
        raise ProbeError(_scrub(f"could not reach the endpoint: {exc}", api_key)) from exc
    except json.JSONDecodeError as exc:
        raise ProbeError(f"the endpoint did not return JSON: {exc}") from exc


def models_call(provider: str, endpoint: str, api_key: str) -> tuple[str, dict]:
    """Where the model list lives for this provider, and what to send with the request."""
    base = base_url(endpoint)
    if provider == "anthropic":
        return f"{base}/v1/models", {"x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION}
    if provider == "gemini":
        return f"{base}/models?key={urllib.parse.quote(api_key)}", {}
    if provider == "ollama":
        return f"{base}/api/tags", {}
    if provider == "azure":
        # The page's Azure endpoint points at one deployment, and a deployment is already a model.
        # Listing them is a management-plane call against a different host entirely.
        raise ProbeError("Azure has no model list at this endpoint — enter the deployment name")
    return f"{base}/models", {"Authorization": f"Bearer {api_key}"}


def parse_models(provider: str, payload: dict) -> list[str]:
    """Model ids out of whichever envelope the provider used."""
    if provider == "gemini":
        # "models/gemini-2.0-flash" — the prefix is part of the resource path, not the name.
        return sorted(str(m.get("name", "")).removeprefix("models/") for m in payload.get("models", []))
    if provider == "ollama":
        return sorted(str(m.get("name", "")) for m in payload.get("models", []))
    return sorted(str(m.get("id", "")) for m in payload.get("data", []))


def chat_call(provider: str, endpoint: str, model: str, api_key: str) -> tuple[str, dict, dict]:
    """The smallest request that proves this endpoint, key and model actually answer."""
    base = base_url(endpoint)
    json_type = {"Content-Type": "application/json"}
    if provider == "anthropic":
        return (f"{base}/v1/messages",
                {**json_type, "x-api-key": api_key, "anthropic-version": ANTHROPIC_VERSION},
                {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": TEST_PROMPT}]})
    if provider == "gemini":
        return (f"{base}/models/{urllib.parse.quote(model)}:generateContent?key={urllib.parse.quote(api_key)}",
                json_type,
                {"contents": [{"parts": [{"text": TEST_PROMPT}]}],
                 "generationConfig": {"maxOutputTokens": 1}})
    if provider == "ollama":
        return (f"{base}/api/chat", json_type,
                {"model": model, "stream": False, "messages": [{"role": "user", "content": TEST_PROMPT}],
                 "options": {"num_predict": 1}})
    if provider == "azure":
        # Deployment-scoped: the model is in the URL the user pasted, and the key is its own header.
        return (endpoint.strip(), {**json_type, "api-key": api_key},
                {"max_tokens": 1, "messages": [{"role": "user", "content": TEST_PROMPT}]})
    return (f"{base}/chat/completions", {**json_type, "Authorization": f"Bearer {api_key}"},
            {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": TEST_PROMPT}]})


def list_models(provider: str, endpoint: str, api_key: str) -> list[str]:
    url, headers = models_call(provider, endpoint, api_key)
    return [m for m in parse_models(provider, _request(url, api_key, headers)) if m]


def check(provider: str, endpoint: str, model: str, api_key: str) -> tuple[bool, str]:
    """Whether this configuration answers. Never raises — the page renders the message either way."""
    if not model and provider != "azure":
        return False, "choose a model first"
    try:
        url, headers, body = chat_call(provider, endpoint, model, api_key)
        _request(url, api_key, headers, body, timeout=CHAT_TIMEOUT)
    except ProbeError as exc:
        return False, str(exc)
    except Exception as exc:  # a shape nobody anticipated must not 500 a settings page
        log.exception("llm probe failed for provider %s", provider)
        return False, _scrub(f"{type(exc).__name__}: {exc}", api_key)
    return True, "ok"

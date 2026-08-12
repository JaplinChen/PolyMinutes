"""LLM pass that repairs recognition errors using the surrounding conversation.

Whisper decodes one utterance at a time and cannot know it is sitting in an SAP ERP interview. A
reader who does knows that `一夕變更` is `工程變更` and that `生技` should be `生管`, because the
sentence around it is about change control and production planning. That is the only signal left
once the acoustics have been squeezed dry, and it is not available to the recogniser.

The danger is the same thing that makes it work: a model asked to fix a transcript will happily
rewrite it into something more fluent than what was said. What keeps it to substitutions it can
justify lives in `guards`; this module is the prompt, the chunking and the reply.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Callable

from opencc import OpenCC

from . import jobs
from .guards import accept
from .store import Term

log = logging.getLogger("polyminutes.refine")

# Lines per request. Large enough that a term corrected early informs the rest, small enough that
# the model still has every line in view when it answers.
CHUNK_LINES = 25
# Lines of already-refined text sent as read-only lead-in, so a chunk boundary does not sit in the
# middle of a sentence with no history.
CONTEXT_LINES = 4

NUMBERED = re.compile(r"^\s*(\d+)\s*[:：.]\s*(.*)$")

# Character-level Simplified -> Traditional, deliberately not the s2twp used on ASR output. The
# model writes the odd Simplified character (保税 for 保稅) and that has to go, but s2twp also
# rewrites vocabulary — 對象 to 物件, 軟件 to 軟體 — which would edit words the speaker chose.
_to_traditional = OpenCC("s2t")


@dataclass
class Line:
    speaker: str
    lang: str
    text: str


def build_prompt(lines: list[Line], context: list[Line], terms: list[Term], topic: str) -> str:
    """Ask for substitutions, not improvements."""
    parts = [
        f"以下是一場「{topic}」的逐字稿片段，由語音辨識產生，含有辨識錯誤。",
        "",
        "你的工作是根據上下文修正**辨識錯誤**，不是改寫。規則：",
        "- 只改明顯錯字：同音字、專有名詞、術語。判斷依據是上下文語意",
        "- 不補字、不刪字、不潤飾語句、不改語氣、不合併或拆分句子",
        "- 口語的重複與贅字是說話者原本就有的，保留",
        "- 不確定的地方保持原樣。寧可留錯，不可改成沒說過的內容",
        "- 中文一律繁體。英文與越南語維持原文",
        "- 不要增刪空白或標點。加空白不算修正",
        "",
        "輸出格式：**只輸出需要修改的行**，每行 `編號: 修正後的完整句子`。",
        "沒有問題的行不要輸出。全部都沒問題就輸出 NONE。不要加任何說明。",
    ]

    if terms:
        parts += ["", "會議專有名詞（出現近似音時應修正為這些寫法）：",
                  "、".join(t.source for t in terms)]

    if context:
        parts += ["", "前文（僅供理解，不要輸出）："]
        parts += [f"{l.speaker}（{l.lang}）：{l.text}" for l in context]

    parts += ["", "待修正："]
    parts += [f"{i}: {l.text}" for i, l in enumerate(lines, 1)]
    return "\n".join(parts)


@dataclass
class Coverage:
    """How much of the transcript the pass actually looked at.

    A chunk the guards throw out whole leaves its lines exactly as the recogniser wrote them, which
    is indistinguishable from a chunk that needed no corrections. Counted, because "checked and
    found clean" and "never checked" are not the same claim to make about a transcript.
    """
    lines: int = 0
    skipped: int = 0

    @property
    def fraction(self) -> float:
        return 0.0 if not self.lines else self.skipped / self.lines


@dataclass
class Rejected:
    """A correction the guards refused, kept because it is the most useful thing they produce.

    A model that repeatedly wants to write 工程變更 where the recogniser wrote 一夕變更 is telling
    you the term exists and that the glossary does not know it. That is the one signal in this
    system that names its own blind spots.
    """
    original: str
    candidate: str


def parse_response(raw: str, lines: list[Line], terms: list[Term] | None = None,
                   rejected: list[Rejected] | None = None,
                   coverage: Coverage | None = None) -> list[str]:
    """Map the numbered reply back onto the input, keeping originals wherever it disagrees.

    Only changed lines come back. Asking for the whole chunk made the model copy out two dozen
    correct sentences to deliver two corrections — measured at 570 output tokens per 25 lines
    against 42 tokens a second, which was most of the runtime.

    An index outside the chunk means the model lost track of the numbering; that line is dropped
    rather than applied to whatever happens to sit at that position.
    """
    got: dict[int, str] = {}
    for row in raw.splitlines():
        if m := NUMBERED.match(row):
            index = int(m.group(1))
            if 1 <= index <= len(lines):
                got[index] = m.group(2)

    # "Most of the chunk" needs a chunk to be about: on the two or three lines left over at the
    # end of a transcript, one correction is already a majority.
    if len(lines) >= 4 and len(got) > len(lines) // 2:
        log.warning("refine rewrote %d of %d lines, keeping originals", len(got), len(lines))
        if coverage is not None:
            coverage.skipped += len(lines)
        return [l.text for l in lines]

    out = []
    for i, line in enumerate(lines, 1):
        candidate = got.get(i, "")
        if not accept(line.text, candidate, terms):
            if rejected is not None and candidate.strip() and candidate.strip() != line.text:
                rejected.append(Rejected(line.text, candidate.strip()))
            out.append(line.text)
            continue
        candidate = candidate.strip()
        out.append(_to_traditional.convert(candidate) if line.lang.startswith("zh") else candidate)
    return out


THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def anthropic_chat(api_key: str, model: str, max_tokens: int = 4000):
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    def chat(prompt: str) -> str:
        message = client.messages.create(model=model, max_tokens=max_tokens,
                                         messages=[{"role": "user", "content": prompt}])
        return "".join(b.text for b in message.content if b.type == "text")

    return chat


class ProviderError(Exception):
    """An HTTP-provider error carrying `status_code`, so `llm.rejection` classifies a rate limit or a
    bad key the same way whether it came from the Anthropic SDK or one of the raw-HTTP providers."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def _post_json(url: str, headers: dict, body: dict, timeout: float) -> dict:
    import json
    import urllib.error
    import urllib.request

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = (exc.read() or b"").decode("utf-8", "replace")[:300]
        except Exception:
            pass
        raise ProviderError(f"{exc.code} {exc.reason}: {detail}".strip(), exc.code) from exc


def openai_chat(api_key: str, model: str, endpoint: str, max_tokens: int = 4000,
                timeout: float = 120):
    """The OpenAI REST shape — Bearer auth, POST /chat/completions — shared by openai, groq, mistral,
    openrouter and nvidia_nim. Same shape the settings-page Verify button proves, so a key that
    tested good here works there."""
    from .llm_probe import base_url
    url = f"{base_url(endpoint)}/chat/completions"

    def chat(prompt: str) -> str:
        payload = _post_json(url, {"Authorization": f"Bearer {api_key}"},
                             {"model": model, "max_tokens": max_tokens,
                              "messages": [{"role": "user", "content": prompt}]}, timeout)
        return THINK.sub("", payload["choices"][0]["message"]["content"])

    return chat


def gemini_chat(api_key: str, model: str, endpoint: str, max_tokens: int = 4000,
                timeout: float = 120):
    """Gemini's generateContent — key in the query string, reply under candidates[].content.parts."""
    import urllib.parse
    from .llm_probe import base_url
    base = base_url(endpoint)

    def chat(prompt: str) -> str:
        url = (f"{base}/models/{urllib.parse.quote(model)}:generateContent"
               f"?key={urllib.parse.quote(api_key)}")
        payload = _post_json(url, {},
                             {"contents": [{"parts": [{"text": prompt}]}],
                              "generationConfig": {"maxOutputTokens": max_tokens}}, timeout)
        parts = payload["candidates"][0]["content"].get("parts", [])
        return THINK.sub("", "".join(p.get("text", "") for p in parts))

    return chat


def azure_chat(api_key: str, endpoint: str, max_tokens: int = 4000, timeout: float = 120):
    """Azure OpenAI: the deployment URL is the endpoint as pasted, and the key is its own header."""
    def chat(prompt: str) -> str:
        payload = _post_json(endpoint.strip(), {"api-key": api_key},
                             {"max_tokens": max_tokens,
                              "messages": [{"role": "user", "content": prompt}]}, timeout)
        return THINK.sub("", payload["choices"][0]["message"]["content"])

    return chat


def ollama_chat(model: str, endpoint: str = "http://127.0.0.1:11434", timeout: float = 900,
                think: bool = False, num_ctx: int = 8192, num_predict: int | None = None,
                temperature: float = 0.0, repeat_penalty: float | None = None):
    """Local models over Ollama's HTTP API — no key, and the transcript never leaves the machine.

    That second point is the reason to prefer it here: these are client interview recordings.

    `num_ctx` is a parameter rather than a constant because Ollama does not complain when a prompt
    exceeds it — it drops the oldest tokens, which are the instructions, and answers whatever is
    left. Callers that send more than a chunk of transcript have to raise it or cut their input to
    fit; both need to know the number.

    `num_predict` caps the reply. Left unset Ollama generates until an end token or the whole
    window fills — on a long-form prompt that means minutes of rambling that overruns the timeout,
    so the summary caller bounds it to what it actually asked for.

    `temperature`/`repeat_penalty` default to greedy-and-unpenalised, which is right for refine —
    the same transcript should correct identically on a re-run. But greedy decoding on a long CJK
    summary collapses into repeating one sentence until the token budget runs out, so the summary
    caller lifts the temperature a little and penalises repetition to break the loop.
    """
    import json
    import urllib.request

    from .llm_probe import base_url

    # The settings page stores Ollama's endpoint as a chat URL (…/api/chat) because that is what
    # Ollama's own docs show, and the test button normalises it the same way. Without this, a saved
    # …/api/chat became …/api/chat/api/generate here and every refine and summary call 404'd.
    root = base_url(endpoint)

    options = {"temperature": temperature, "num_ctx": num_ctx}
    if num_predict is not None:
        options["num_predict"] = num_predict
    if repeat_penalty is not None:
        options["repeat_penalty"] = repeat_penalty

    def chat(prompt: str) -> str:
        body = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            # Reasoning costs minutes per chunk and buys a different failure mode rather than a
            # better one — see --think in scripts/refine_transcript.py. Models with no thinking
            # mode ignore the field.
            "think": think,
            "options": options,
        }).encode()
        req = urllib.request.Request(f"{root}/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            text = json.load(r).get("response", "")
        # Qwen and other reasoning models narrate before answering; the answer is what follows.
        return THINK.sub("", text)

    return chat


class Refiner:
    """Correction driven by whatever `chat` is given — cloud or local, the guards are the same."""

    def __init__(self, chat, topic: str = "會議"):
        self._chat = chat
        self._topic = topic

    def refine(self, lines: list[Line], terms: list[Term] | None = None,
               rejected: list[Rejected] | None = None,
               coverage: Coverage | None = None,
               on_progress: Callable[[int, int], None] | None = None) -> list[str]:
        """Correct a whole transcript, chunk by chunk. Returns one string per input line."""
        out: list[str] = []
        if coverage is not None:
            coverage.lines += len(lines)
        for start in range(0, len(lines), CHUNK_LINES):
            chunk = lines[start : start + CHUNK_LINES]
            context = [Line(l.speaker, l.lang, t)
                       for l, t in zip(lines[max(0, start - CONTEXT_LINES) : start],
                                       out[max(0, start - CONTEXT_LINES) : start])]
            prompt = build_prompt(chunk, context, terms or [], self._topic)
            for attempt in (1, 2):
                try:
                    out += parse_response(self._chat(prompt), chunk, terms, rejected, coverage)
                    break
                except jobs.Cancelled:
                    raise
                except Exception:
                    if attempt == 1:
                        log.exception("refine failed at line %d, retrying once", start)
                        continue
                    log.exception("refine failed twice at line %d, keeping originals", start)
                    if coverage is not None:
                        coverage.skipped += len(chunk)
                    out += [l.text for l in chunk]
            if on_progress is not None:
                on_progress(min(start + CHUNK_LINES, len(lines)), len(lines))
        return out

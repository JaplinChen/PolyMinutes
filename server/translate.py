"""Translation via the Claude API — the only outbound call this app makes.

One request does two jobs: translate the new utterance into every other configured language, and
say whether the *previous* utterance now reads wrong given what was just said. Folding the
refinement into the next line's request is what keeps context-aware polishing from doubling the
API bill, and it is why the subtitle page has to support rewriting a line it already drew.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .store import Term

LANGUAGE_NAMES = {
    "zh": "Traditional Chinese as written in Taiwan (臺灣繁體中文)",
    "vi": "Vietnamese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
    "id": "Indonesian",
}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


@dataclass
class Line:
    text: str
    lang: str
    speaker: str = ""


@dataclass
class Result:
    translations: dict[str, str]
    # Set only when the model judged the previous line wrong in hindsight.
    previous_source: str | None = None
    previous_translations: dict[str, str] = field(default_factory=dict)


def relevant_terms(text: str, terms: list[Term]) -> list[Term]:
    """Glossary entries whose source string actually occurs in this utterance.

    Sending the whole glossary every time would be both slow and counterproductive — a long list
    of irrelevant terms measurably degrades adherence to the ones that matter.
    """
    lowered = text.casefold()
    return [t for t in terms if t.source.casefold() in lowered]


def _term_lines(terms: list[Term], targets: list[str]) -> list[str]:
    out = []
    for t in terms:
        if t.mode == "keep":
            out.append(f'- "{t.source}": keep exactly as-is in every language, do not translate it')
        elif t.mode == "translate":
            pairs = ", ".join(f"{language_name(k)} = \"{v}\"" for k, v in t.targets.items() if k in targets)
            if pairs:
                out.append(f'- "{t.source}": must be rendered as {pairs}')
    return out


def build_prompt(line: Line, targets: list[str], context: list[Line], previous: Line | None,
                 terms: list[Term], prev_targets: list[str] | None = None) -> str:
    """The user-turn content. Kept a pure function so the prompt is testable without the network."""
    parts = [
        "You translate live meeting speech. The text comes from speech recognition, so it may "
        "contain recognition errors, missing punctuation, and words from other languages mixed in.",
        "",
        f"Source language: {language_name(line.lang) if line.lang else 'unknown, infer it'}",
        f"Translate into: {', '.join(language_name(t) for t in targets)}",
    ]

    if glossary := _term_lines(relevant_terms(line.text, terms), targets):
        parts += ["", "Glossary (binding):", *glossary]

    if context:
        parts += ["", "Earlier lines, for context only — do not translate these:"]
        parts += [f"  [{c.speaker or '?'}] {c.text}" for c in context]

    parts += ["", f"Translate this line, spoken by {line.speaker or 'an unknown speaker'}:", line.text]

    if previous:
        parts += [
            "",
            "The line before it was translated without knowing what came next:",
            f"  original: {previous.text}",
            "If the new line shows that translation was wrong — a misheard word, a pronoun with no "
            "referent, a sentence split in the wrong place — provide a corrected version. If it "
            "reads correctly, omit the `previous` field entirely rather than restating it.",
        ]

    schema = {t: "translation" for t in targets}
    # The previous line may be in a different language than this one, so its correction targets are
    # its own (cfg.languages minus its language), not this line's. Sharing one schema asked the model
    # to re-translate the previous line into the wrong languages in any mixed-language meeting.
    prev_schema = {t: "translation" for t in (prev_targets if prev_targets is not None else targets)}
    parts += [
        "",
        "Reply with JSON only, no code fence and no commentary:",
        json.dumps({"translations": schema, "previous": {"source": "...", "translations": prev_schema}}, ensure_ascii=False),
        "Keep translations natural and spoken, not literal. Preserve the speaker's register.",
    ]
    if "zh" in targets:
        parts.append("Chinese output must be Traditional Chinese with Taiwanese vocabulary, never Simplified.")

    return "\n".join(parts)


_JSON = re.compile(r"\{.*\}", re.DOTALL)

# How many times to ask before giving up on a line. The malformed JSON is sporadic, so a re-ask
# almost always parses; three total attempts pushes the residue near zero without hammering a
# genuinely stuck line (or a live subtitle) for long.
PARSE_ATTEMPTS = 3


def parse_response(raw: str, targets: list[str], prev_targets: list[str] | None = None) -> Result:
    """Tolerant parse: models wrap JSON in fences or prose often enough to matter."""
    match = _JSON.search(raw)
    if not match:
        raise ValueError(f"no JSON object in response: {raw[:200]!r}")

    data = json.loads(match.group(0))
    translations = {k: str(v) for k, v in (data.get("translations") or {}).items() if k in targets}

    prev = data.get("previous") or {}
    prev_source = prev.get("source")
    keep = prev_targets if prev_targets is not None else targets
    prev_translations = {k: str(v) for k, v in (prev.get("translations") or {}).items() if k in keep}

    # A `previous` block with nothing usable in it is the model echoing the schema; drop it so the
    # subtitle page is not told to rewrite a line with identical content.
    if not prev_source and not prev_translations:
        return Result(translations)

    return Result(translations, prev_source, prev_translations)


class Translator:
    """Provider-agnostic: driven by a `chat(prompt) -> str` callable, so live translation follows the
    provider chosen on the settings page — the same dispatcher (`postmeeting.chat_for`) the
    post-meeting pass uses, instead of a hard-wired Anthropic client that ignored the choice."""

    def __init__(self, chat: Callable[[str], str],
                 on_reject: Callable[[Exception], None] | None = None):
        self._chat = chat
        # Called with the provider's error when a request is rejected, so the key pool can bench a
        # rate-limited or invalid key. The exception is always re-raised; this only reports it.
        self._on_reject = on_reject

    def translate(self, line: Line, targets: list[str], context: list[Line] | None = None,
                  previous: Line | None = None, terms: list[Term] | None = None,
                  prev_targets: list[str] | None = None) -> Result:
        prompt = build_prompt(line, targets, context or [], previous, terms or [], prev_targets)
        # A local model emits malformed JSON sporadically, not by content: on a 2h19m meeting 179 of
        # 862 lines came back unparseable, and re-asking the same prompt parsed every one. So a parse
        # failure is retried rather than dropped — a provider rejection (bad key, rate limit) is not,
        # it re-raises at once so the key pool can bench the key instead of hammering it.
        last: ValueError | None = None
        for _ in range(PARSE_ATTEMPTS):
            try:
                raw = self._chat(prompt)
            except Exception as exc:
                if self._on_reject:
                    self._on_reject(exc)
                raise
            try:
                return parse_response(raw, targets, prev_targets)
            except ValueError as exc:  # JSONDecodeError is a ValueError subclass
                last = exc
        # The reply itself, not just the decoder's column: 5 lines of a 2026-10-08 reprocess failed
        # all three asks with "Expecting ',' delimiter" and none reproduced afterwards, so what the
        # model actually wrote is the only evidence a tolerant parse could be built from.
        raise ValueError(f"{last} — last reply: {raw[:800]!r}") from last

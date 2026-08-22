"""Post-meeting summary: one LLM call per language, structure-checked, retried once.

The transcript is the input everyone already has; what a meeting produces is the part nobody wrote
down — what was decided and who left owning what. A model asked for that in free prose invents
structure on some days and skips it on others, so the reply is held to four labelled sections
(TITLE/SUMMARY/DECISIONS/ACTIONS) and a bad reply is sent back once with the parser's complaint
attached. JSON was tried first, but a long detailed summary leaves newlines and 「」 unescaped
inside the string and no longer parses; labelled sections carry multi-line prose verbatim. Three
languages in one call truncated the reply inside the token budget, hence one language per call.

Pure orchestration: the chat callable comes from the caller (refine.anthropic_chat or ollama_chat,
so the privacy choice made there carries over) and nothing here touches the store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from . import config
from .translate import language_name

# Fits Ollama's num_ctx=8192 tokens for CJK-heavy text with prompt overhead.
INPUT_BUDGET = 12_000

RULES_PATH = config.ROOT / "summary_rules.md"

DEFAULT_RULES = "\n".join([
    "- 標題一句話",
    "- 摘要客觀轉述會議內容，不加評論",
    "- 決議＝已拍板的事",
    "- 行動項目＝誰要去做什麼",
    "- 沒有就留空陣列，不要硬湊",
])


@dataclass
class SummaryLine:
    speaker: str
    lang: str
    text: str


def target_chars(total_chars: int) -> int:
    """Summary length scales with the transcript, within reason either way.

    The detailed-record rules want a rich narrative that keeps quotes, the interaction chain and a
    per-participant rundown, so the ceiling is generous — a long meeting earns a long summary.
    """
    return max(400, min(8000, total_chars // 6))


def load_rules() -> str:
    """User-editable format rules; the built-in default when the file does not exist."""
    if RULES_PATH.is_file():
        return RULES_PATH.read_text(encoding="utf-8")
    return DEFAULT_RULES


def sample(lines: list[SummaryLine], budget_chars: int = INPUT_BUDGET) -> tuple[list[SummaryLine], bool]:
    """Cut an over-budget transcript by evenly-spaced sampling, per speaker.

    Truncating from the front would summarize the first hour of a two-hour meeting. Sampling
    per speaker keeps each participant's share of the floor, so a quiet decision-maker is not
    sampled out by a talkative colleague.
    """
    total = sum(len(l.text) for l in lines)
    if total <= budget_chars:
        return lines, False

    ratio = budget_chars / total
    by_speaker: dict[str, list[int]] = {}
    for i, line in enumerate(lines):
        by_speaker.setdefault(line.speaker, []).append(i)

    kept: list[int] = []
    for indices in by_speaker.values():
        n = max(1, round(len(indices) * ratio))
        step = len(indices) / n
        kept += [indices[int(k * step)] for k in range(n)]
    kept.sort()

    # The per-speaker floor of one line each keeps a quiet voice in, but with many speakers those
    # floors add up: a meeting with more speakers than the budget has room for came back over
    # budget — 100 speakers of one long line each returned all of them, defeating the cap the whole
    # function exists to enforce. Cap the result with a second even pass, chronological so the
    # thinning is spread across the meeting rather than taken off one end.
    kept_chars = sum(len(lines[i].text) for i in kept)
    if kept_chars > budget_chars and len(kept) > 1:
        keep_n = max(1, round(len(kept) * budget_chars / kept_chars))
        step = len(kept) / keep_n
        kept = [kept[int(k * step)] for k in range(keep_n)]

    return [lines[i] for i in kept], True


# Pre-meeting notes past this many characters are truncated before the prompt. They are context,
# not the subject; a whole slide deck pasted in would crowd out the transcript that is.
REFERENCE_BUDGET = 2000


def build_prompt(lines: list[SummaryLine], lang: str, rules: str,
                 speakers: dict[str, str] | None = None, sampled: bool = False,
                 total_chars: int | None = None, reference: str = "") -> str:
    """One language per call — a multi-language reply truncates inside the token budget."""
    # Length target is set by the whole meeting, not by however much survived sampling.
    total = total_chars if total_chars is not None else sum(len(l.text) for l in lines)
    target = target_chars(total)

    parts = [
        f"Summarize this meeting transcript. Write everything in {language_name(lang)}.",
        "",
        "Rules:",
        rules,
        "",
        f"The summary text should be about {target} characters.",
    ]
    # Reference before the transcript: the model reads the meeting's own agenda, attendees and terms
    # first, so a mis-heard product name or an unstated priority is corrected against it. Marked as
    # background, not content, so nothing here is summarised as if it had been said.
    if reference := reference.strip():
        parts += ["", "Background notes provided before the meeting (context only, not spoken):",
                  reference[:REFERENCE_BUDGET]]
    # Who each code is, with their department where known: stance ("the QA lead objected") can be
    # read from role instead of guessed from tone. Only named codes are listed — an S7 nobody named
    # tells the model nothing it cannot see in the transcript.
    if speakers:
        parts += ["", "Speakers:"] + [f"{code} = {label}" for code, label in speakers.items()]
    if sampled:
        parts += ["", "The transcript below is an evenly-sampled excerpt of a longer meeting."]

    parts += ["", "Transcript:"]
    parts += [f"{l.speaker}({l.lang}): {l.text}" for l in lines]

    # A long detailed summary does not survive being wrapped in a JSON string — local models leave
    # the newlines and 「」 quotes inside it unescaped and the JSON no longer parses. Labelled
    # sections carry multi-line prose verbatim, and this side reassembles the object.
    parts += [
        "",
        "Reply in exactly these four labelled sections, nothing before or after:",
        "TITLE: <one sentence>",
        "SUMMARY:",
        "<the summary; may span many lines>",
        "DECISIONS:",
        "- <one decision per line, or leave this section empty>",
        "ACTIONS:",
        "- <what to do> || <speaker code, or blank>",
        "RISKS:",
        "- <one risk or thing still to confirm per line, or leave this section empty>",
        "OPEN_QUESTIONS:",
        "- <one unaligned issue, unsupported assumption or inference per line, or leave empty>",
        "",
        "Use the speaker codes exactly as they appear in the transcript; leave blank when unclear.",
        "Every section except TITLE and SUMMARY may be empty — do not invent items to fill them.",
        "Put anything you are inferring rather than that was said outright in OPEN_QUESTIONS,",
        "never in SUMMARY as if it were fact.",
    ]
    return "\n".join(parts)


_SECTION = re.compile(
    r"^\s*(TITLE|SUMMARY|DECISIONS|ACTIONS|RISKS|OPEN_QUESTIONS)\s*:\s*(.*)$", re.IGNORECASE)


def _bullets(block: str) -> list[str]:
    out = []
    for line in block.splitlines():
        line = line.strip()
        if line.startswith(("-", "•", "*")):
            line = line[1:].strip()
        if line:
            out.append(line)
    return out


def parse_response(raw: str, valid_speakers: frozenset[str] = frozenset()) -> dict:
    """Split the labelled sections; a missing title or summary raises so the retry loop can say so."""
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in raw.splitlines():
        m = _SECTION.match(line)
        if m:
            current = m.group(1).upper()
            sections[current] = [m.group(2)] if m.group(2).strip() else []
        elif current is not None:
            sections[current].append(line)

    title = "\n".join(sections.get("TITLE", [])).strip()
    summary = "\n".join(sections.get("SUMMARY", [])).strip()
    if not title:
        raise ValueError(f"missing TITLE section: {raw[:200]!r}")
    if not summary:
        raise ValueError(f"missing SUMMARY section: {raw[:200]!r}")

    decisions = _bullets("\n".join(sections.get("DECISIONS", [])))

    clean_actions = []
    for item in _bullets("\n".join(sections.get("ACTIONS", []))):
        text, _, speaker = item.partition("||")
        text, speaker = text.strip(), speaker.strip()
        if not text:
            continue
        # A speaker code the transcript never contained is the model inventing an owner — the same
        # failure the citation path drops outright. Here the action text is still worth keeping, so
        # the false attribution is cleared to "" (the page shows "unassigned") rather than the whole
        # item. When no set is supplied — a test calling parse_response directly — nothing is dropped.
        if valid_speakers and speaker and speaker not in valid_speakers:
            speaker = ""
        clean_actions.append({"text": text, "speaker": speaker})

    risks = _bullets("\n".join(sections.get("RISKS", [])))
    open_questions = _bullets("\n".join(sections.get("OPEN_QUESTIONS", [])))

    return {"title": title, "summary": summary,
            "decisions": decisions, "actions": clean_actions,
            "risks": risks, "open_questions": open_questions}


def retry_prompt(original: str, bad_reply: str, error: str) -> str:
    return "\n".join([
        original,
        "",
        "Your previous reply was rejected:",
        f"  error: {error}",
        f"  reply: {bad_reply[:500]}",
        "Answer again using ONLY the four labelled sections above. No other text.",
    ])


def max_tokens_for(target: int) -> int:
    # Double the character target covers CJK tokenization plus the labelled-section overhead
    # (title, decisions, actions, risks, open_questions).
    return target * 2 + 800


def summarize(lines: list[SummaryLine], languages: list[str], chat: Callable[[str], str],
              speakers: dict[str, str] | None = None,
              should_stop: Callable[[], bool] | None = None,
              reference: str = "") -> tuple[dict, str]:
    """One summary per language; a language whose reply fails schema twice is simply missing."""
    rules = load_rules()
    total = sum(len(l.text) for l in lines)
    sampled_lines, sampled = sample(lines)
    # The codes the transcript actually uses. The model is told to use these exactly; this is what
    # holds it to that when it does not.
    valid = frozenset(l.speaker for l in lines if l.speaker)

    out: dict[str, dict] = {}
    for lang in languages:
        if should_stop and should_stop():
            break
        prompt = build_prompt(sampled_lines, lang, rules, speakers, sampled, total_chars=total,
                              reference=reference)
        raw = chat(prompt)
        try:
            out[lang] = parse_response(raw, valid)
        except ValueError as first:
            try:
                out[lang] = parse_response(chat(retry_prompt(prompt, raw, str(first))), valid)
            except ValueError:
                pass  # recorded as missing; status below says partial/failed

    if len(out) == len(languages):
        return out, "ok"
    return out, "partial" if out else "failed"

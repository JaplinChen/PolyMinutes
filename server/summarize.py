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
    id: int = 0  # transcript line id, so a summary item can cite the line it came from


def item_text(x) -> str:
    """A summary item's text, whether it is a new {text, line} dict or an old bare string.

    Summaries stored before citations were added hold plain strings; readers (export, the ask
    index) must keep working against both shapes without a migration.
    """
    return x.get("text", "") if isinstance(x, dict) else str(x)


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

    parts += ["", "Transcript (each line is tagged [line_id]):"]
    parts += [f"[{l.id}] {l.speaker}({l.lang}): {l.text}" for l in lines]

    # A long detailed summary does not survive being wrapped in a JSON string — local models leave
    # the newlines and 「」 quotes inside it unescaped and the JSON no longer parses. Labelled
    # sections carry multi-line prose verbatim, and this side reassembles the object.
    parts += [
        "",
        "Reply in exactly these six labelled sections, nothing before or after:",
        "TITLE: <one sentence>",
        "SUMMARY:",
        "<the summary; may span many lines>",
        "DECISIONS:",
        "- <one decision per line, or leave this section empty> || <line_id it came from>",
        "ACTIONS:",
        "- <what to do> || <speaker code, or blank> || <line_id it came from>",
        "RISKS:",
        "- <one risk or thing still to confirm> || <line_id it came from>",
        "OPEN_QUESTIONS:",
        "- <one unaligned issue, unsupported assumption or inference> || <line_id it came from>",
        "",
        "Use the speaker codes exactly as they appear in the transcript; leave blank when unclear.",
        "End every DECISIONS/ACTIONS/RISKS/OPEN_QUESTIONS item with the [line_id] of the transcript",
        "line it is drawn from, so a reader can jump to it. Leave the line_id blank only when no single",
        "line applies.",
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


def _cite(raw_id: str, valid_lines: frozenset[int]) -> int | None:
    """The line_id a model appended, verified against the transcript's real ids.

    A number that names no line is the model inventing a citation — the same failure the ask path
    drops. Here the item's text is still worth keeping, so an unverifiable id becomes None (the page
    shows the item without a jump) rather than dropping the whole item. With no set supplied — a test
    calling parse_response directly — a well-formed number is trusted as-is.
    """
    raw_id = raw_id.strip().lstrip("[#").rstrip("]")
    if not raw_id.lstrip("-").isdigit():
        return None
    line_id = int(raw_id)
    if valid_lines and line_id not in valid_lines:
        return None
    return line_id


# The transcript-style [id] models emit inline instead of the asked-for "|| id" — sometimes wrapped
# in （）, sometimes a range [a]-[b], sometimes several ids in one bracket ([31, 98]) or a stray
# trailing "|" — at the very end of an item. qwen/gemma reach for this bracket form (it matches how
# the transcript itself is tagged) far more reliably than the "||" convention, so the parser accepts
# all of these and takes the first id (each id points at a supporting line; any one is a valid jump).
# The first bracket is required — a bare trailing number is content ("第 31 項"), not a citation, and
# only a bracketed id is stripped. After it, any run of more ids/brackets/ranges/pipes/commas is eaten
# too, so `| [31], [98]` and `[125]、[130]` leave clean text. The captured group is the first id.
_TRAIL_CITE = re.compile(
    r"[\s(（|｜&＆]*\[(\d+)(?:\s*[,，]\s*\d+)*\]"
    r"(?:[\s,，、|｜~\-–&＆]*\[?\d+(?:\s*[,，]\s*\d+)*\]?)*"
    r"[)）\s。，,.、|｜&＆]*$")


def _first_valid_id(citation: str, valid_lines: frozenset[int]) -> int | None:
    """First id in a trailing citation that verifies. The model lists several supporting lines and
    the first is often wrong while a later one is real — measured on session 3, `[17445, 17446]` had
    17446 valid but 17445 not — so every id is tried, not just the first. With no valid set (a test
    calling parse_response directly) the first id is trusted as before."""
    ids = [int(x) for x in re.findall(r"\d+", citation)]
    if not ids:
        return None
    if not valid_lines:
        return ids[0]
    return next((i for i in ids if i in valid_lines), None)


def _pull_citation(item: str, valid_lines: frozenset[int]) -> tuple[str, int | None]:
    """Split an item into (text, verified line_id), accepting `... || 12` or a trailing `[12]`."""
    head, sep, tail = item.rpartition("||")
    if sep:
        line = _cite(tail, valid_lines)
        if line is not None:
            return head.strip(), line
    m = _TRAIL_CITE.search(item)
    if m:
        line = _first_valid_id(m.group(0), valid_lines)
        if line is not None:
            return item[:m.start()].rstrip(), line
    return item.strip(), None


def _split_citation(item: str, valid_lines: frozenset[int]) -> tuple[str, str, int | None]:
    """Pull (clean text, speaker, verified line id) out of one bullet.

    Small models don't hold to `<text> || <speaker> || <id>`; measured on the 2026-08-05 meeting they
    also emit `|| 12`, a trailing `[12]`, several ids (`[31, 98]` or `[31], [98]`), a speaker slipped
    into the citation slot (`|| 總經理 [31]`), and the doubled `text [12] || 總經理 || [12]`. Splitting
    on every `||`, the text is the first segment (with its own trailing `[id]` stripped so an inline
    citation does not survive in the shown text), the line is the first verified id in any segment,
    and a non-id segment is the speaker. Callers that have no speaker field simply ignore it.
    """
    parts = item.split("||")
    text, line = _pull_citation(parts[0], valid_lines)
    speaker = ""
    for seg in parts[1:]:
        seg = seg.strip()
        if not seg:
            continue
        _, seg_line = _pull_citation(seg, valid_lines)
        cited = seg_line if seg_line is not None else _cite(seg, valid_lines)
        if cited is not None:
            if line is None:
                line = cited
        elif not speaker:
            speaker = seg
    # A citation the model joined with "&&" instead of "[id]" (qwen/gemma both do it) leaves the
    # "&&" behind once the id is stripped; clear a trailing run of it and of stray pipes. A "&&" left
    # mid-text is the model using it as a conjunction ("A && B") — `_deamp` turns that into a comma.
    return _deamp(text).rstrip(" 　&＆|｜").rstrip(), speaker, line


# "&&" / "＆＆" is never real content — the model reaches for it as "and". A single "&" is left alone
# (R&D, A&B). Collapsed to a comma with its surrounding spaces so "A && B" reads "A，B".
_AMP2 = re.compile(r"\s*[&＆]{2,}\s*")


def _deamp(text: str) -> str:
    return _AMP2.sub("，", text).strip()


def _cited_bullets(block: str, valid_lines: frozenset[int]) -> list[dict]:
    """Bullets of `<text>` with an optional trailing citation; these sections carry no speaker."""
    out = []
    for item in _bullets(block):
        text, _speaker, line = _split_citation(item, valid_lines)
        if not text:
            continue
        out.append({"text": text, "line": line})
    return out


def parse_response(raw: str, valid_speakers: frozenset[str] = frozenset(),
                   valid_lines: frozenset[int] = frozenset()) -> dict:
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

    title = _deamp("\n".join(sections.get("TITLE", [])))
    summary = _deamp("\n".join(sections.get("SUMMARY", [])))
    if not title:
        raise ValueError(f"missing TITLE section: {raw[:200]!r}")
    if not summary:
        raise ValueError(f"missing SUMMARY section: {raw[:200]!r}")

    decisions = _cited_bullets("\n".join(sections.get("DECISIONS", [])), valid_lines)

    clean_actions = []
    for item in _bullets("\n".join(sections.get("ACTIONS", []))):
        # ACTIONS keep the speaker `_split_citation` finds; the other three sections drop it.
        text, speaker, line = _split_citation(item, valid_lines)
        if not text:
            continue
        # A speaker code the transcript never contained is the model inventing an owner — the same
        # failure the citation path drops outright. Here the action text is still worth keeping, so
        # the false attribution is cleared to "" (the page shows "unassigned") rather than the whole
        # item. When no set is supplied — a test calling parse_response directly — nothing is dropped.
        if valid_speakers and speaker and speaker not in valid_speakers:
            speaker = ""
        clean_actions.append({"text": text, "speaker": speaker, "line": line})

    risks = _cited_bullets("\n".join(sections.get("RISKS", [])), valid_lines)
    open_questions = _cited_bullets("\n".join(sections.get("OPEN_QUESTIONS", [])), valid_lines)

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
    # Only the ids the model was shown can be cited; a long meeting is sampled, so an id outside the
    # excerpt is as invented as one that never existed.
    valid_lines = frozenset(l.id for l in sampled_lines if l.id)

    out: dict[str, dict] = {}
    for lang in languages:
        if should_stop and should_stop():
            break
        prompt = build_prompt(sampled_lines, lang, rules, speakers, sampled, total_chars=total,
                              reference=reference)
        raw = chat(prompt)
        try:
            out[lang] = parse_response(raw, valid, valid_lines)
        except ValueError as first:
            try:
                out[lang] = parse_response(chat(retry_prompt(prompt, raw, str(first))), valid, valid_lines)
            except ValueError:
                pass  # recorded as missing; status below says partial/failed

    if len(out) == len(languages):
        return out, "ok"
    return out, "partial" if out else "failed"

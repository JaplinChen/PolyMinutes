"""Cross-meeting Q&A: "who said we were moving the delivery date?" — answered with citations.

No vector database, deliberately. A vector index buys you a fast approximate scan, and this
codebase has nothing slow to fix: a year of meetings at the observed rate is 200 sessions,
190,000 lines, 2.5 MB, and a plain `LIKE '%交貨%'` across all of it comes back in 14 ms. SQLite's
FTS5 with the trigram tokenizer is worse than useless here — it needs three characters, so the
two-character terms these meetings turn on (交期, 產能, 交貨) match nothing at all. And embeddings
would mean torch, two gigabytes of it, on a fixed machine-room box that already has to fit a GPU
decoder. What does the semantic work instead is the per-session summary written for the export:
titles and decisions, maintained anyway, are a good enough first-layer index that a second one
would only be a second thing to keep in sync.

So: keyword hits from the store, plus that summary index, union, load the winners' transcripts,
answer. Pure functions and one orchestrator — the store, the HTTP layer and the model client are
all passed in, as they are in `summarize`.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Callable

from . import summarize

MAX_QUESTION_CHARS = 500
INDEX_SESSIONS = 60
ASK_COOLDOWN_SECONDS = 3


@dataclass
class Budget:
    input_chars: int
    max_sessions: int


def budget_for(provider: str) -> Budget:
    # Ollama defaults to num_ctx=8192 and does not complain when the prompt overruns it — it
    # silently drops the oldest tokens, which are the instructions. So the local path is held to
    # summarize.INPUT_BUDGET worth of characters and two meetings, which fits; a cloud context can
    # hold six meetings and far more text.
    if provider == "ollama":
        return Budget(12_000, 2)
    return Budget(120_000, 6)


@dataclass
class AskLine:
    id: int
    session_id: int
    start: float
    speaker: str
    text: str


@dataclass
class Citation:
    session_id: int
    line_id: int
    start: float
    speaker: str
    text: str


def keyword_prompt(question: str) -> str:
    return "\n".join([
        "A user is searching a collection of meeting transcripts.",
        f"Question: {question}", "",
        "List the words and phrases that would appear verbatim in the transcript lines that",
        "answer it — the terms people say out loud, not the question's own wording. Include",
        "synonyms and alternate phrasings. If the question names a time period, give the range.",
        "",
        "Reply with JSON only, no code fence and no commentary:",
        json.dumps({"keywords": ["..."], "since": "YYYY-MM-DD", "until": "YYYY-MM-DD"}),
        'Use null for since/until when the question does not restrict the dates.',
    ])


_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _extract(raw: str) -> dict | None:
    match = _JSON.search(raw or "")
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _date(value) -> str | None:
    """A YYYY-MM-DD date, or None. Anything else — the model ignoring the format, or answering "上週"
    — is dropped rather than passed on: since/until are string-compared against an ISO timestamp, so
    a non-date does not raise, it silently mis-filters (a value sorting above every real date, used
    as `until`, wipes the result). None means no restriction, which is the safe reading of garbage.
    """
    if not isinstance(value, str):
        return None
    value = value.strip()
    # Zero-padded on purpose: started is compared as a string, and "2026-8-1" sorts differently from
    # the "2026-08-01" the timestamps use, so strptime's tolerance for "2026-8-1" is not enough here.
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return None
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
    return value


def parse_keywords(raw: str) -> tuple[list[str], str | None, str | None]:
    """Tolerant on purpose: a failed expansion means "no keywords", never a failed question."""
    data = _extract(raw)
    if data is None:
        return [], None, None

    words: list[str] = []
    for item in data.get("keywords") or []:
        if not isinstance(item, str):
            continue
        word = item.strip()
        # Single characters match half the transcript; a blank matches all of it.
        if len(word) < 2 or word in words:
            continue
        words.append(word)
    return words, _date(data.get("since")), _date(data.get("until"))


def pick_sessions(hits: dict[int, int], index: list[dict], limit: int) -> list[int]:
    """Keyword hits first, most-hit and most-recent ahead; the summary index fills what is left."""
    order = {int(row["id"]): pos for pos, row in enumerate(index)}
    ranked = sorted(hits, key=lambda sid: (-hits[sid], order.get(sid, len(order))))

    picked: list[int] = []
    for sid in ranked:
        if len(picked) >= limit:
            return picked
        picked.append(sid)
    for row in index:
        if len(picked) >= limit:
            break
        sid = int(row["id"])
        if sid not in picked:
            picked.append(sid)
    return picked


def index_prompt(question: str, index: list[dict]) -> str:
    parts = ["Below is one line per meeting: its id, its date, its title and what it decided.",
             "Pick the meetings whose transcript could answer the question. Pick none rather",
             "than guessing.", "", f"Question: {question}", "", "Meetings:"]
    for row in index:
        decisions = "; ".join(summarize.item_text(d) for d in (row.get("decisions") or []))
        parts.append(f"[{row['id']}] {row.get('started', '')} {row.get('title', '')}"
                     + (f" — {decisions}" if decisions else ""))
    parts += [
        "",
        "Reply with JSON only, no code fence and no commentary:",
        json.dumps({"sessions": [1, 2]}),
    ]
    return "\n".join(parts)


def parse_sessions(raw: str, known_ids) -> list[int]:
    data = _extract(raw)
    if data is None:
        return []
    known = {int(i) for i in known_ids}
    out: list[int] = []
    for item in data.get("sessions") or []:
        try:
            sid = int(item)
        except (ValueError, TypeError):
            continue
        if sid in known and sid not in out:
            out.append(sid)
    return out


def answer_prompt(question: str, sessions: list[dict], lines: list[AskLine],
                  names: dict[str, str] | None = None, truncated=()) -> str:
    names = names or {}
    truncated = set(truncated)

    parts = ["Answer the question using only the meeting transcripts below.",
             "Every claim must be backed by the line it came from.",
             "", f"Question: {question}", "", "Meetings:"]
    for s in sessions:
        note = "  (only part of this meeting was read — an evenly-sampled excerpt)" \
            if int(s["id"]) in truncated else ""
        parts.append(f"[{s['id']}] {s.get('started', '')}{note}")
    if truncated:
        parts += ["", "The meetings marked above were read only in part. If that could change the",
                  "answer — especially if the answer is that nobody said something — say so."]

    parts += ["", "Transcript lines, each tagged [session#line_id]:"]
    for l in lines:
        speaker = names.get(l.speaker, l.speaker)
        parts.append(f"[{l.session_id}#{l.id}] {speaker}: {l.text}")

    parts += [
        "",
        "Reply with JSON only, no code fence and no commentary:",
        json.dumps({"answer": "...", "citations": [{"session_id": 1, "line_id": 2}]}),
        "Cite with the exact session and line_id from the tags. Do not invent ids.",
        "If the transcripts do not answer the question, say so and cite nothing.",
    ]
    return "\n".join(parts)


def fit(lines: list[AskLine], budget: Budget) -> tuple[list[AskLine], set[int]]:
    """Sample each over-budget meeting down to its share, and report which ones were cut."""
    by_session: dict[int, list[AskLine]] = {}
    for line in lines:
        by_session.setdefault(line.session_id, []).append(line)
    if not by_session:
        return [], set()

    per_session = budget.input_chars // max(1, len(by_session))
    kept: list[AskLine] = []
    truncated: set[int] = set()
    for sid, group in by_session.items():
        # Reuse summarize.sample rather than restate the maths: same even spacing, same
        # per-speaker share, so a quiet decision-maker is not sampled out by a talkative one.
        proxies = [summarize.SummaryLine(l.speaker, "", l.text) for l in group]
        back = {id(p): l for p, l in zip(proxies, group)}
        picked, sampled = summarize.sample(proxies, budget_chars=per_session)
        if sampled:
            truncated.add(sid)
        kept += [back[id(p)] for p in picked]

    order = {id(l): i for i, l in enumerate(lines)}
    return sorted(kept, key=lambda l: order[id(l)]), truncated


def parse_answer(raw: str, lines: list[AskLine]) -> tuple[str, list[Citation], int]:
    """Citations are verified against the store rows — the model supplies ids and nothing else."""
    data = _extract(raw)
    if data is None:
        raise ValueError(f"no JSON object in response: {(raw or '')[:200]!r}")
    answer = data.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError(f"answer must be a non-empty string, got {answer!r}")

    rows = {(l.session_id, l.id): l for l in lines}
    out: list[Citation] = []
    dropped = 0
    for item in data.get("citations") or []:
        if not isinstance(item, dict):
            dropped += 1
            continue
        try:
            key = (int(item["session_id"]), int(item["line_id"]))
        except (KeyError, ValueError, TypeError):
            dropped += 1
            continue
        row = rows.get(key)
        if row is None:
            dropped += 1
            continue
        # speaker/start/text come from the row, never from the reply: a plausible timestamp is
        # exactly the kind of thing a model produces when it is making the citation up.
        out.append(Citation(row.session_id, row.id, row.start, row.speaker, row.text))
    return answer.strip(), out, dropped


RETRY_SUFFIX = "\n\nYour previous reply was not valid JSON. Reply with valid JSON only."


def ask(question: str, chat: Callable[[str], str],
        search: Callable[[list[str], str | None, str | None], dict[int, int]],
        index_rows: list[dict], load_lines: Callable[[list[int]], list[AskLine]],
        names: dict[str, str] | None = None, provider: str = "anthropic") -> dict:
    q = (question or "").strip()
    if not q:
        raise ValueError("question is empty")
    if len(q) > MAX_QUESTION_CHARS:
        raise ValueError(f"question is longer than {MAX_QUESTION_CHARS} characters")

    budget = budget_for(provider)
    keywords, since, until = parse_keywords(chat(keyword_prompt(q)))
    hits = dict(search(keywords, since, until)) if keywords else {}

    # One meeting's worth of keyword hits is not enough to trust the recall; ask the summary
    # index too and take the union, so a passing remark and a paraphrase both have a route in.
    if len(hits) < 2 and index_rows:
        known = [int(r["id"]) for r in index_rows]
        for sid in parse_sessions(chat(index_prompt(q, index_rows)), known):
            hits.setdefault(sid, 0)

    session_ids = pick_sessions(hits, index_rows, budget.max_sessions)
    lines, truncated = fit(load_lines(session_ids), budget)

    known_rows = {int(r["id"]): r for r in index_rows}
    sessions = [{"id": sid, "started": (known_rows.get(sid) or {}).get("started", "")}
                for sid in session_ids]

    prompt = answer_prompt(q, sessions, lines, names, truncated)
    raw = chat(prompt)
    try:
        answer, citations, dropped = parse_answer(raw, lines)
    except ValueError:
        answer, citations, dropped = parse_answer(chat(prompt + RETRY_SUFFIX), lines)

    # Resolve the speaker code to a name here, where the names map is: the citation crosses to the
    # page as JSON with no second lookup, and the page has no names map of its own to do it with.
    # An unnamed voice keeps its code, which is what the transcript shows for it too.
    resolved = names or {}
    for c in citations:
        c.speaker = resolved.get(c.speaker, c.speaker)

    return {
        "answer": answer,
        "citations": [asdict(c) for c in citations],
        "sessions": session_ids,
        "truncated": sorted(truncated),
        "dropped_citations": dropped,
        "verified": not (dropped > 0 and not citations),
    }

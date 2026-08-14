"""Post-meeting re-segmentation: joins utterances the VAD cut mid-sentence, then restores punctuation.

The live pipeline ends an utterance on silence, and a speaker who pauses to breathe mid-sentence is
split into fragments — "我們今天要討論的是" / "交期的問題" — that read as two half-thoughts. The
recording cannot be re-cut after the fact, but the transcript can: fragments that share a speaker
and sit close together in time are one sentence, and that is pure arithmetic over timestamps.

Punctuation is the one part that needs a model, and its guard is absolute where refine's is
calibrated: strip the punctuation back out and the text must be exactly what went in. A reply that
changed any word is discarded whole.
"""

from __future__ import annotations

import re
from typing import Sequence

from .refine import NUMBERED

# A pause longer than this between two fragments is a real stop, not a breath. Below it, the same
# speaker continuing is one sentence the VAD cut.
MAX_GAP = 1.0
# A merged line past this stops being a subtitle and starts being a paragraph; the summary reads
# either fine, but the transcript page shows lines.
MAX_MERGED_CHARS = 120
# The same limit in the dimension the character count cannot see. The VAD never returns an
# utterance longer than max_speech_duration (20s), so every longer line on the page was built here
# — and slow, sparse speech reaches 90 seconds while still under 120 characters, which is one
# transcript row nobody can follow and one clip nobody can scrub. Measured across three real
# meetings: 273 of 1456 lines ran past 20s, 61 past 30s, the longest 90.1s for 118 characters.
# Cutting here costs nothing invented — the pieces keep the VAD's own boundaries and their own text.
MAX_MERGED_SECONDS = 30.0
# A fragment already ending in one of these finished its sentence; joining across it would glue
# two complete sentences together.
TERMINAL = "。．.！!？?…"
# Somebody else holding the floor for this long inside a gap makes it a speaker change rather than
# a VAD cut. Matched to the shortest piece the splitter will carve out, so the two stages agree on
# what counts as a turn.
MIN_INTERRUPTION = 0.5

WORD = re.compile(r"\w+", re.UNICODE)
# `3-5: text`, or `3: text` for a fragment that is a sentence on its own.
RANGED = re.compile(r"^(\d+)\s*(?:[-–~]\s*(\d+))?\s*[:：]\s*(.+)$")


Barriers = Sequence[tuple[float, float, str]]


def _interrupted(prev: dict, nxt: dict, barriers: Barriers) -> bool:
    """True when somebody else held the floor in the gap between these two lines.

    A gap is not evidence of silence. The splitter cuts an utterance wherever the segmenter heard a
    speaker change, and a piece too short to decode leaves no line behind — so an interruption the
    recogniser had no words for reaches this stage looking exactly like a breath, and welds two
    lines back over the top of it. On a real 2.7h meeting that undid 6 of the 14 cuts, including a
    line two people were visibly speaking in.
    """
    return any(who != prev["speaker"]
               and min(end, nxt["start"]) - max(start, prev["end_time"]) >= MIN_INTERRUPTION
               for start, end, who in barriers)


def _continuable(prev: dict, nxt: dict, barriers: Barriers = ()) -> bool:
    """Whether these two pieces *could* be one sentence — everything decidable without reading.

    Facts, not judgement: one voice, one language, nothing edited by hand, a pause short enough to
    be a breath, and nobody else talking in between. Whether the sentence actually continues is a
    question about the words, which is `_joinable`'s guess or the model's answer.
    """
    if prev["speaker"] != nxt["speaker"] or prev["lang"] != nxt["lang"]:
        return False
    # A failed decode or a human-corrected line is not raw VAD output; leave both where they are.
    if prev["status"] != "ok" or nxt["status"] != "ok" or prev["refined"] or nxt["refined"]:
        return False
    # Lines written before end_time existed have no gap to measure — and absorbing one as the
    # tail would leave the merged line with no end at all, which the clip picker reads.
    if prev["end_time"] is None or nxt["end_time"] is None:
        return False
    if nxt["start"] - prev["end_time"] > MAX_GAP:
        return False
    return not _interrupted(prev, nxt, barriers)


def _joinable(prev: dict, nxt: dict, length: int, barriers: Barriers = (),
              began: float | None = None) -> bool:
    """Whether `nxt` continues `prev`. `began` is where the run being built started, which is what
    the duration limit measures against — `prev` alone only knows the last piece."""
    if not _continuable(prev, nxt, barriers):
        return False
    if nxt["end_time"] - (prev["start"] if began is None else began) > MAX_MERGED_SECONDS:
        return False
    text = prev["source"].rstrip()
    if not text or text[-1] in TERMINAL:
        return False
    return length + len(nxt["source"]) <= MAX_MERGED_CHARS


def merge_groups(rows: list[dict], barriers: Barriers = ()) -> list[list[int]]:
    """Runs of consecutive lines that are one sentence the VAD cut. Only runs of 2+ come back.

    `barriers` are (start, end, code) spans the segmenter heard a speaker in; a gap holding one of
    somebody else is a speaker change and not a cut to undo. Empty for a recording processed
    without the segmentation model, which leaves the merge exactly as it was.
    """
    groups: list[list[int]] = []
    i = 0
    while i < len(rows):
        group = [i]
        length = len(rows[i]["source"])
        while (j := group[-1] + 1) < len(rows) and _joinable(rows[group[-1]], rows[j], length,
                                                             barriers, rows[i]["start"]):
            group.append(j)
            length += len(rows[j]["source"])
        if len(group) > 1:
            groups.append(group)
        i = group[-1] + 1
    return groups


# Where the model is allowed to let a sentence run to. The limits above are the target shape of a
# transcript row; these are the backstop, because the target is exactly where the arithmetic used
# to stop mid-sentence and letting the sentence finish is the point. A wrong answer then costs one
# long row rather than a meeting glued into one.
HARD_MAX_SECONDS = MAX_MERGED_SECONDS * 2
HARD_MAX_CHARS = MAX_MERGED_CHARS * 2


def candidate_runs(rows: list[dict], barriers: Barriers = ()) -> list[list[int]]:
    """Runs of pieces the model should read together. Only runs of 2+ come back.

    The arithmetic decides what *could* join — one voice, no interruption, no long pause — and the
    model decides where the sentences inside actually end. Sizing the run by the hard maximum rather
    than the target is deliberate: a run cut at the target would hand the model a fragment ending
    mid-sentence and no way to say so.
    """
    runs: list[list[int]] = []
    i = 0
    while i < len(rows):
        run = [i]
        length = len(rows[i]["source"])
        while (j := run[-1] + 1) < len(rows) and _continuable(rows[run[-1]], rows[j], barriers) \
                and rows[j]["end_time"] - rows[i]["start"] <= HARD_MAX_SECONDS \
                and length + len(rows[j]["source"]) <= HARD_MAX_CHARS:
            run.append(j)
            length += len(rows[j]["source"])
        if len(run) > 1:
            runs.append(run)
        i = run[-1] + 1
    return runs


def build_group_prompt(texts: list[str]) -> str:
    """Ask for the sentences these fragments form, punctuated.

    One question, not two: deciding where a sentence ends and punctuating it are the same reading,
    and splitting them meant the boundary was picked by a character counter before anything read
    the words. The fragments are the recording's own boundaries, so the model chooses among real
    cut points and never invents one.
    """
    n = len(texts)
    parts = [
        f"以下是同一位發言者連續的 {n} 個逐字稿片段，編號 1 到 {n}，由語音辨識依停頓切開，"
        "可能把一句話切成好幾段。",
        "請判斷相鄰的哪些片段屬於同一句話，把它們合併並補上標點。規則：",
        f"- 編號只有 1 到 {n}，輸出的編號不得超過 {n}，也不得重複使用",
        "- 這是「片段的編號」，不是你切出來的句子的編號",
        f"- 每個片段恰好用一次，依序涵蓋 1 到 {n}，不可跳號、不可調換順序",
        "- 只能插入標點符號，不得增加、刪除或修改任何字",
        "- 中文用全形標點（，。？），英文與越南語用半形",
        "",
        "輸出格式：每組一行 `起編號-迄編號: 合併並加上標點後的文字`。",
        f"合併 1 和 2 就寫 `1-2: ...`；片段 3 自成一組就寫 `3-3: ...`。最後一行的迄編號必須是 {n}。",
        "不要加任何說明。",
        "",
        "片段：",
    ]
    parts += [f"{i}: {t}" for i, t in enumerate(texts, 1)]
    return "\n".join(parts)


def parse_groups(raw: str, texts: list[str]) -> list[tuple[list[int], str]] | None:
    """The reply as (piece indices, punctuated text) per sentence, or None if it cannot be trusted.

    Trusted means: the ranges tile the fragments in order, exactly once each, and the words come
    back unchanged — the same guard `parse_response` applies, over the whole run rather than one
    line, so a model that rewrites or drops a fragment while regrouping is refused outright.
    """
    out: list[tuple[list[int], str]] = []
    expected = 1
    for row in raw.splitlines():
        if not (m := RANGED.match(row.strip())):
            continue
        first, last, text = int(m.group(1)), int(m.group(2) or m.group(1)), m.group(3).strip()
        if first != expected or last < first or last > len(texts):
            return None
        out.append((list(range(first - 1, last)), text))
        expected = last + 1
    if expected != len(texts) + 1 or not out:
        return None
    if any(_words(text) != _words("".join(texts[i] for i in idx)) for idx, text in out):
        return None
    return out


def join(keep: dict, absorbed: list[dict]) -> tuple[str, float | None, dict[str, str]]:
    """The merged line's text, end time and translations."""
    sep = "" if keep["lang"].startswith("zh") else " "
    rows = [keep, *absorbed]
    text = sep.join(t for r in rows if (t := r["source"].strip()))
    langs: dict[str, list[str]] = {}
    for r in rows:
        for lang, t in r.get("translations", {}).items():
            langs.setdefault(lang, []).append(t)
    return text, absorbed[-1]["end_time"], {lang: " ".join(ts) for lang, ts in langs.items()}


def build_prompt(texts: list[str]) -> str:
    parts = [
        "以下逐字稿由語音辨識產生，缺少標點符號。為每一行補上標點，讓句子容易閱讀。規則：",
        "- 只能插入標點符號，不得增加、刪除或修改任何字",
        "- 不合併行、不拆行，每行獨立處理",
        "- 中文用全形標點（，。？），英文與越南語用半形",
        "",
        "輸出格式：**只輸出有加標點的行**，每行 `編號: 加上標點後的整行`。",
        "不需要加的行不要輸出。全部都不需要就輸出 NONE。不要加任何說明。",
        "",
        "待補標點：",
    ]
    parts += [f"{i}: {t}" for i, t in enumerate(texts, 1)]
    return "\n".join(parts)


def _words(text: str) -> str:
    return "".join(WORD.findall(text))


def parse_response(raw: str, texts: list[str]) -> list[str]:
    """The reply mapped back onto the input; any line whose words changed keeps its original.

    # ponytail: word-level compare ignores whitespace, so re-spaced English slips through; a
    # per-token diff would catch it if it ever shows up in practice.
    """
    got: dict[int, str] = {}
    for row in raw.splitlines():
        if m := NUMBERED.match(row):
            index = int(m.group(1))
            if 1 <= index <= len(texts):
                got[index] = m.group(2)
    out = []
    for i, text in enumerate(texts, 1):
        candidate = got.get(i, "").strip()
        out.append(candidate if candidate and _words(candidate) == _words(text) else text)
    return out

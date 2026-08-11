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

from .refine import NUMBERED

# A pause longer than this between two fragments is a real stop, not a breath. Below it, the same
# speaker continuing is one sentence the VAD cut.
MAX_GAP = 1.0
# A merged line past this stops being a subtitle and starts being a paragraph; the summary reads
# either fine, but the transcript page shows lines.
MAX_MERGED_CHARS = 120
# A fragment already ending in one of these finished its sentence; joining across it would glue
# two complete sentences together.
TERMINAL = "。．.！!？?…"

WORD = re.compile(r"\w+", re.UNICODE)


def _joinable(prev: dict, nxt: dict, length: int) -> bool:
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
    text = prev["source"].rstrip()
    if not text or text[-1] in TERMINAL:
        return False
    return length + len(nxt["source"]) <= MAX_MERGED_CHARS


def merge_groups(rows: list[dict]) -> list[list[int]]:
    """Runs of consecutive lines that are one sentence the VAD cut. Only runs of 2+ come back."""
    groups: list[list[int]] = []
    i = 0
    while i < len(rows):
        group = [i]
        length = len(rows[i]["source"])
        while (j := group[-1] + 1) < len(rows) and _joinable(rows[group[-1]], rows[j], length):
            group.append(j)
            length += len(rows[j]["source"])
        if len(group) > 1:
            groups.append(group)
        i = group[-1] + 1
    return groups


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

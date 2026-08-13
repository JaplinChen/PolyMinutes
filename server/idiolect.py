"""Who someone sounds like in *words*, as evidence independent of the voiceprint.

A room microphone flattens voices — that is the whole reason the recogniser needs a margin. But
how a person talks is a second, uncorrelated signal: a wrongly-learned print pulls a voice onto the
wrong name and no amount of comparing embeddings can ever notice, whereas the words can.

Style beats topic. Function words and sentence rhythm ("齁", "反正", how long a sentence runs) carry
the speaker; content words carry the meeting's subject, which every participant shares. Measured
here by leave-one-meeting-out over the real database (3 meetings, 88 named (session, code) units,
21 names): style features 27% top-1 against 30% for the full set and 5% for chance — the content
half adds almost nothing, and what it does add is really "who attended which meeting".

Below the thresholds it is noise, not weak evidence. Two-way duels by the code's line count:
5-9 lines 51%, 10-24 52%, 25-49 75%, 50+ 88%. Beating *everyone* (the harder question the conflict
scan asks): 25-49 50%, 50+ 80%. Hence MIN_LINES / CONFLICT_LINES below — 25 lines is a coin flip
and must never be allowed to overrule a voiceprint.
"""

from __future__ import annotations

import math
import re
from collections import Counter

MIN_LINES = 25       # 低於此句數兩人對決僅 51%（擲硬幣）；25-49 句 75%、50+ 句 88%
CONFLICT_LINES = 50  # 矛盾偵測要更保守：50+ 句時「勝過所有其他人」達 80%

FILLERS = ["那個", "然後", "就是", "對不對", "我覺得", "其實", "反正", "基本上", "所以說",
           "你知道", "這樣子", "的話", "一個", "問題", "OK", "齁", "啦", "喔", "欸", "嘛",
           "呢", "吧", "嗯", "我們", "你們", "他們", "現在", "如果", "因為", "但是", "可是",
           "而且", "比如", "譬如", "包括", "應該", "可能", "一定", "真的", "當然", "其他"]

_EN = re.compile(r"[A-Za-z]{2,}")
_NUM = re.compile(r"\d")


def features(text: str) -> Counter:
    """One line as the feature counts the ranker scores: fillers, length bucket, script marks."""
    f: Counter = Counter()
    for word in FILLERS:
        if (n := text.count(word)):
            f[f"F:{word}"] += n
    length = len(re.sub(r"\s", "", text))
    f["LEN:" + ("s" if length < 12 else "m" if length < 30 else "l" if length < 60 else "x")] += 1
    if _EN.search(text):
        f["EN:1"] += 1
    if _NUM.search(text):
        f["NUM:1"] += 1
    return f


def profiles(rows: list[tuple[str, int, str]]) -> dict[str, dict[int, Counter]]:
    """name -> session_id -> features, kept split by meeting so a caller can subtract one.

    Never collapse the sessions here: scoring a code against a profile that contains that same
    meeting is the model grading its own training data, and it agrees with itself every time.
    """
    out: dict[str, dict[int, Counter]] = {}
    for name, session_id, text in rows:
        out.setdefault(name, {}).setdefault(session_id, Counter()).update(features(text or ""))
    return out


def rank(profiles_: dict[str, dict[int, Counter]], lines: list[str],
         exclude_session: int | None = None, only: set[str] | None = None) -> list[tuple[str, float]]:
    """Names ordered by how well their wording explains `lines` (add-1 multinomial naive Bayes)."""
    test: Counter = Counter()
    for line in lines:
        test.update(features(line))
    if not test:
        return []
    built: dict[str, Counter] = {}
    for name, by_session in profiles_.items():
        if only is not None and name not in only:
            continue
        prof: Counter = Counter()
        for session_id, counts in by_session.items():
            if session_id != exclude_session:
                prof.update(counts)
        if prof:
            built[name] = prof
    if not built:
        return []
    vocab = set(test)
    for prof in built.values():
        vocab |= set(prof)
    scored = []
    for name, prof in built.items():
        total = sum(prof.values()) + len(vocab)
        scored.append((name, sum(cnt * math.log((prof.get(key, 0) + 1) / total)
                                 for key, cnt in test.items())))
    return sorted(scored, key=lambda kv: -kv[1])

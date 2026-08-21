"""Proper-noun correction over decoded text.

Whisper picks the wrong character far more often than it mishears the syllable, and the wrong
character usually differs only in tone. Comparing toneless pinyin catches those: `公單` and `工單`
are one edit apart with tones and identical without.

It does not catch everything, and is not meant to: `生管` decoded as `生氣` is a different syllable,
not a different character for the same one. That is an acoustic error and belongs to the model,
not to this pass.

Chinese terms are compared as pinyin, Latin ones as lowercase letters. Both use the same edit
distance and both refuse to fire above their threshold: a false insertion puts a word on the
meeting-room TV that nobody said, which is worse than leaving the original mistake alone.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass

from opencc import OpenCC

from .store import Term

# How far apart two spellings may be, as a fraction of the term's own length, before they stop
# counting as the same word. At 0.25 a two-syllable term tolerates one edit, which covers the
# common case — same sound, wrong character — without letting `料號` reach `了好`.
MAX_DISTANCE = 0.25
# Below this many characters a term is too small for a distance to mean anything: at three
# characters of pinyin, every second syllable is within the threshold of every other. Such terms
# are still corrected, but only on an exact toneless match — `直距` for `治具`, both `zhiju`.
MIN_KEY = 6
# Shorter than this and even an exact sound match is too likely to be a different word entirely.
MIN_TERM_KEY = 4

HAN = re.compile(r"[一-鿿]")
LATIN_TOKEN = re.compile(r"[A-Za-zÀ-ỹ][A-Za-zÀ-ỹ'’-]*")


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def pinyin_of(text: str, tones: bool = True) -> str:
    """Pinyin with no separators. Neutral tone is written as 1, matching upstream.

    Correction drops the tones. Whisper picks the wrong character far more often than it mishears
    the syllable itself, and those wrong characters usually differ only in tone — `工單` for `公單`
    is one edit with tones and none without.
    """
    from pypinyin import Style, lazy_pinyin

    if not tones:
        return "".join(lazy_pinyin(text))
    return "".join(p.replace("5", "1") for p in
                   lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))


@dataclass
class _Rule:
    term: str
    key: str      # what we compare against: pinyin for Chinese, lowercase for Latin
    chinese: bool
    toned: str = ""  # the term's pinyin with tones, for choosing between homophone terms

    @property
    def limit(self) -> int:
        """Chinese must match exactly; Latin may differ by a quarter of its length.

        Scanned over seven real transcripts, allowing Chinese a single edit rewrote 知道 to 製造
        156 times, 生產 to 生管 146 times and 材料 to 呆料 71 times — 1578 corruptions from a
        33-term glossary. Mandarin syllables are packed too densely for edit distance to mean
        anything at this length. Latin words are sparse enough for it: incent -> Vincent.
        """
        if self.chinese:
            return 0
        return int(len(self.key) * MAX_DISTANCE) if len(self.key) >= MIN_KEY else 0


def _rules(terms: list[Term]) -> list[_Rule]:
    out: list[_Rule] = []
    for t in terms:
        # A protected word is vocabulary, not a destination. Both halves matter: it is skipped as
        # a window (below, via `_known`) and never used as a replacement.
        if t.mode == "protect":
            continue
        chinese = bool(HAN.search(t.source))
        key = pinyin_of(t.source, tones=False) if chinese else t.source.lower()
        if len(key) >= MIN_TERM_KEY:
            out.append(_Rule(t.source, key, chinese, pinyin_of(t.source) if chinese else ""))
    # Longest first: a term that contains another must win, or the shorter one eats its prefix.
    return sorted(out, key=lambda r: -len(r.term))


class Corrector:
    """Rewrites near-misses of glossary terms to their canonical spelling.

    Two sources, applied in that order of authority. `aliases` are pairs a human fixed on the
    transcript page — what the recogniser wrote, and what was actually said. Nothing else here is
    labelled by someone who was in the room, so they are applied literally and first. The glossary
    rules that follow are inference from pinyin, and only get what the aliases did not already fix.
    """

    def __init__(self, terms: list[Term], aliases: dict[str, str] | None = None):
        self._rules = _rules(terms)
        # Two terms can be homophones of each other — 工序 and 供需 are both gongxu, and both are
        # ordinary vocabulary in a manufacturing interview. Text that already spells one of them
        # is left alone: the glossary saying a word exists is also the glossary saying it is not
        # a mistake. Protected words are here and nowhere else — known, never written.
        self._known = {t.source for t in terms}
        # Longest first: a correction that contains another must win. A registered word is never
        # the wrong side of one — the rule the comment above states applies to what a human typed
        # as much as to what pinyin infers, and only the inference half was honouring it. A room
        # accumulated 工序 -> 工需 that way: 工序 is in its glossary, is fed to the decoder as a
        # hotword, and this pass then rewrote the 19 places the decoder had got right.
        self._aliases = sorted(((wrong, right) for wrong, right in (aliases or {}).items()
                                if wrong not in self._known), key=lambda kv: -len(kv[0]))

    def fix(self, text: str) -> str:
        if not text:
            return text
        text = self._apply_aliases(text)
        for rule in self._rules:
            text = self._fix_chinese(text, rule) if rule.chinese else self._fix_latin(text, rule)
        return text

    def _apply_aliases(self, text: str) -> str:
        """One left-to-right pass over the original text, longest alias first.

        A per-alias `str.replace` loop let one alias's output be consumed by the next — teaching
        生館→生管 and (from an unrelated line) 生管→升官 turned 生館 into 升官. Scanning the original
        and emitting each replacement into a buffer that is never re-scanned keeps every alias
        applied to what the recogniser actually wrote, not to another alias's correction.
        """
        if not self._aliases:
            return text
        out: list[str] = []
        i, n = 0, len(text)
        while i < n:
            for wrong, right in self._aliases:  # longest first, so a term beats its own prefix
                if wrong and text.startswith(wrong, i):
                    out.append(right)
                    i += len(wrong)
                    break
            else:
                out.append(text[i])
                i += 1
        return "".join(out)

    def _fix_chinese(self, text: str, rule: _Rule) -> str:
        """Slide a window the width of the term over every run of Han characters.

        Only Han windows are considered, so a term never swallows the English half of a
        code-switched sentence, which is most of them in this meeting room.
        """
        width = len(rule.term)
        limit = rule.limit
        i = 0
        while i + width <= len(text):
            window = text[i : i + width]
            if len(HAN.findall(window)) != width or window in self._known:
                i += 1
                continue
            if edit_distance(pinyin_of(window, tones=False), rule.key) <= limit                     and self._best_for(window, width) is rule:
                text = text[:i] + rule.term + text[i + width :]
                i += len(rule.term)
            else:
                i += 1
        return text

    def _best_for(self, window: str, width: int) -> _Rule | None:
        """Which term wins when several are homophones of each other.

        Dropping tones is what makes the match work at all — Whisper picks the wrong character far
        more often than it mishears the syllable, and the wrong character usually differs only in
        tone. But two terms can then collide: 生管 and 升官 are both shengguan, and 生館 was
        rewritten to whichever rule happened to be checked first. Tones settle it — 生館 is
        sheng1guan3, which is 生管 exactly and 升官 not at all.
        """
        toneless = pinyin_of(window, tones=False)
        toned = pinyin_of(window)
        rivals = [r for r in self._rules
                  if r.chinese and len(r.term) == width
                  and edit_distance(r.key, toneless) <= r.limit]
        if not rivals:
            return None
        return min(rivals, key=lambda r: (edit_distance(r.toned, toned), r.term))

    def _fix_latin(self, text: str, rule: _Rule) -> str:
        limit = rule.limit

        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            if token == rule.term:
                return token
            return rule.term if edit_distance(token.lower(), rule.key) <= limit else token

        return LATIN_TOKEN.sub(replace, text)


# A candidate has to look like a term rather than a phrase or a stray character.
MIN_LEN, MAX_LEN = 2, 8
# Widening runs into whatever sits next to the edit, and in speech that is usually a particle.
# Trimmed off both ends so the candidate is the term rather than the sentence around it.
PARTICLES = set("的那個這些他她我你們是在了就也都有會要跟和把被對從")


def _widenable(char: str) -> bool:
    """A character worth absorbing into a candidate.

    Particles are excluded, and that is the whole trick: widening 會 -> 櫃 rightwards inside
    排會的需求 gives 會的 -> 櫃的, which as a literal substitution turns 開會的時間 into
    開櫃的時間. Skipping the particle sends the widening left instead and yields 排會 -> 排櫃.
    """
    return bool(HAN.fullmatch(char)) and char not in PARTICLES


# Anything a rule may not be about. A hand edit fixes three things at once — the word that was
# misheard, the punctuation the decoder never wrote, and whichever characters the typist's own
# keyboard produced — and only the first generalises. The other two arrive here looking exactly
# like a term pair and are then applied, literally, to every later transcript.
PUNCTUATION = set(" 	，。、！？：；「」（）,.!?:;-—·")
_to_simplified = OpenCC("t2s")
# The same converter asr.py runs every decode through, so "already Traditional" means the same
# thing on both sides of the pipeline.
_to_traditional = OpenCC("s2tw")


def _is_a_term_pair(before: str, after: str) -> bool:
    """Whether this difference is about a word, and not about typing.

    Audited against the corrections a real room accumulated over three weeks: 239 rules, five of
    them written by this function and none of them about vocabulary —

        內銷 -> 内销        the typist pasted Simplified; the rule then converts correct output
        臺相 -> 台相        back to Simplified, the failure asr.py's OpenCC exists to stop
        盼表現期首 -> 。首   five characters replaced by a full stop
        報 -> ，報          a comma, applied to every 報 in every meeting after it
        下我們 -> 下。

    Each fires as a literal substitution on text nobody was looking at when the edit was made,
    which is the shape of the poisoning that cost a transcript once already. A rule earns its
    place by naming a word the decoder got wrong; punctuation and character variants are edits to
    one line and stay there.
    """
    if any(c in PUNCTUATION for c in after) or any(c in PUNCTUATION for c in before):
        return False
    # Same word, different script. Normalised on both sides, because the pair may be written
    # either direction and neither direction is a vocabulary correction.
    if _to_simplified.convert(before) == _to_simplified.convert(after):
        return False
    # The replacement has to be written the way the transcript is written. 對於沒 -> 这里没 is a
    # real change of words, so the check above lets it through, and it would then write Simplified
    # into every later transcript through the back door — the typist pasted from somewhere else,
    # which is a fact about their clipboard, not about the vocabulary.
    return _to_traditional.convert(after) == after


def _trim(before: str, after: str) -> tuple[str, str]:
    """Drop matching particles from both ends, keeping the two strings aligned."""
    while len(after) > MIN_LEN and before[:1] == after[:1] and after[0] in PARTICLES:
        before, after = before[1:], after[1:]
    while len(after) > MIN_LEN and before[-1:] == after[-1:] and after[-1] in PARTICLES:
        before, after = before[:-1], after[:-1]
    return before, after


def diff_terms(original: str, candidate: str) -> list[tuple[str, str]]:
    """The pieces that differ between two versions of a line, widened to look like terms.

    The interesting correction is usually one character — 申管 for 生管, ELP for ERP — and one
    character is not a glossary entry. Widening into the Han characters on either side turns the
    edit back into the word it sits in, which is what a reader can recognise and what the
    corrector needs to match against.
    """
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, original, candidate).get_opcodes():
        if tag != "replace" or (j2 - j1) > MAX_LEN or (i2 - i1) > MAX_LEN:
            continue
        # Both strings share the text around the edit, so one offset widens both. Widened only
        # as far as MIN_LEN needs, and to the right first: a one-character edit inside 確認料耗
        # widened greedily becomes 認料耗, which then matches nothing else in the transcript,
        # while the shortest widening is 料耗 — the word that was actually wrong.
        left = right = 0
        while (j2 - j1) + left + right < MIN_LEN:
            if j2 + right < len(candidate) and i2 + right < len(original)                     and _widenable(candidate[j2 + right]):
                right += 1
            elif j1 - left - 1 >= 0 and i1 - left - 1 >= 0                     and _widenable(candidate[j1 - left - 1]):
                left += 1
            else:
                break
        before, after = _trim(original[i1 - left : i2 + right], candidate[j1 - left : j2 + right])
        # The wrong side needs the same floor as the right side, and must not be pure particle:
        # a human fixing one 的 that should have been 裂痕 taught 的→裂痕 as a literal alias, which
        # then rewrote every 的 in every later transcript. A one-character or all-particle wrong
        # side is a sentence fragment, not a term worth generalising.
        if MIN_LEN <= len(after) <= MAX_LEN and before != after \
                and len(before) >= MIN_LEN and not all(c in PARTICLES for c in before) \
                and _is_a_term_pair(before, after):
            out.append((before, after))
    return out


def collisions(term: str, text: str, known: set[str]) -> dict[str, int]:
    """Every other spelling in `text` this term would overwrite.

    Adding a term is not obviously destructive, which is the problem: `料號` and `料耗` are both
    liaohao, 料耗 is a term of the trade, and adding 料號 rewrote it forty-two times without
    saying so. Anything already in the glossary is excluded — a registered word is not collateral.
    """
    key = pinyin_of(term, tones=False)
    width = len(term)
    found: dict[str, int] = {}
    for i in range(len(text) - width + 1):
        window = text[i : i + width]
        if window == term or window in known or len(HAN.findall(window)) != width:
            continue
        if pinyin_of(window, tones=False) == key:
            found[window] = found.get(window, 0) + 1
    return found

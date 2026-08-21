"""The corrector: glossary terms, learned corrections, and what must never be rewritten.

The recurring failure these pin down is a homophone rewritten into the wrong word — 採購 into
才夠 211 times across seven interviews — so most of them assert what stays put.
"""

from __future__ import annotations

from . import asr, correct, store


def test_diff_terms_learns_the_word_that_was_wrong() -> None:
    """An edit teaches a substitution, and that substitution is applied literally everywhere
    afterwards — so what gets learned has to be the word, not a fragment of the sentence."""
    # Widened to the shortest thing that is still a word: greedy widening would learn 認料耗 from
    # 確認料耗, which then matches nothing else in the transcript.
    assert correct.diff_terms("確認料耗的數量", "確認料號的數量") == [("料耗", "料號")]
    assert correct.diff_terms("那個申管會上系統", "那個生管會上系統") == [("申管", "生管")]

    # Widening steps over particles rather than absorbing them: 會的 -> 櫃的 would rewrite
    # 開會的時間 to 開櫃的時間, so the widening goes left instead and learns the actual term.
    assert correct.diff_terms("做一個排會的需求", "做一個排櫃的需求") == [("排會", "排櫃")]
    assert correct.diff_terms("一樣的內容", "一樣的內容") == []


def test_diff_terms_never_learns_a_particle_as_the_wrong_side() -> None:
    """的→裂痕, learned from one edit, rewrote every 的 in every later transcript. A wrong side
    shorter than a term, or made only of particles, is a sentence fragment — never a rule."""
    assert correct.diff_terms("有的問題", "有裂痕問題") == []
    assert correct.diff_terms("就是了那個", "就是修模那個") == []


def test_human_corrections_outrank_the_glossary() -> None:
    """An edit is the only thing here labelled by someone who was in the room."""
    c = correct.Corrector([], {"申管": "生管", "ELP系統": "ERP系統"})
    assert c.fix("那個申管會上ELP系統") == "那個生管會上ERP系統"
    assert c.fix("沒有問題的句子") == "沒有問題的句子"
    assert correct.Corrector([], {}).fix("原文不動") == "原文不動"


def test_one_correction_does_not_cascade_into_another() -> None:
    """Two independently-learned pairs must not chain: applying them sequentially with in-place
    replace let one alias's output become the next alias's input, so a line the user only ever
    taught 生館→生管 for came out rewritten by an unrelated 生管→升官 pair."""
    c = correct.Corrector([], {"生館": "生管", "生管": "升官"})
    # 生館 is what the recogniser wrote and 生管 is what was said; the 生管→升官 pair was learned
    # from a different utterance and must not fire on this one's corrected output.
    assert c.fix("生館的事") == "生管的事"
    # A line that genuinely contains 生管 still gets its own correction, once.
    assert c.fix("升官的事") == "升官的事"
    assert c.fix("談生管") == "談升官"


def test_corrector_fixes_near_misses_only() -> None:
    def term(source: str) -> store.Term:
        return store.Term(id=0, source=source, lang="", mode="hint", category="", targets={})

    c = correct.Corrector([term("工單"), term("威剛科技"), term("Vincent"), term("治具")])
    # Wrong character, same sound — what the decode-time replacer misses once the tone is wrong.
    assert c.fix("公單的管理") == "工單的管理"
    assert c.fix("微剛科技的部分") == "威剛科技的部分"
    assert c.fix("直距的管理") == "治具的管理"
    assert c.fix("線上還有問incent") == "線上還有問Vincent"
    # Different words that merely rhyme must survive untouched.
    assert c.fix("我們公司的工作單位") == "我們公司的工作單位"
    assert c.fix("這個 schedule 要 delay 一週") == "這個 schedule 要 delay 一週"
    assert correct.Corrector([]).fix("原文不動") == "原文不動"


def test_tones_decide_between_homophone_terms() -> None:
    """Dropping tones is what makes the match work; keeping them is what makes it unambiguous.

    生管 and 升官 are both shengguan with tones removed, so a misrecognition landed on whichever
    rule happened to be checked first. Tones settle it: 生館 is sheng1guan3, which is 生管 exactly
    and 升官 not at all.
    """
    def term(source: str) -> store.Term:
        return store.Term(id=0, source=source, lang="", mode="hint", category="", targets={})

    c = correct.Corrector([term("生管"), term("升官")])
    assert c.fix("那個生館會上系統") == "那個生管會上系統"
    assert c.fix("盛管那邊的排程") == "生管那邊的排程"
    assert c.fix("他終於昇官了") == "他終於升官了"


def test_protected_words_are_vocabulary_not_destinations() -> None:
    """Declaring a word real must not make it a target.

    才夠 and 採購 are both caigou and both ordinary speech. Registering 才夠 to shield it from the
    corrector turned it into a destination instead, and 採購 — 217 occurrences — was rewritten to
    才夠 211 times before the corpus replay caught it.
    """
    def term(source: str, mode: str = "hint") -> store.Term:
        return store.Term(id=0, source=source, lang="", mode=mode, category="", targets={})

    c = correct.Corrector([term("才夠", "protect"), term("生管")])
    assert c.fix("這次的採購單要重做") == "這次的採購單要重做"
    assert c.fix("真的不是才夠算") == "真的不是才夠算"
    # Ordinary terms still correct as before.
    assert c.fix("那個生館會上系統") == "那個生管會上系統"


def test_collisions_report_what_a_term_would_overwrite() -> None:
    """The check that was missing when 料號 was added and destroyed 料耗 42 times."""
    corpus = "這個料耗的部分料耗要看 才夠算 交貨時間"
    assert correct.collisions("料號", corpus, set()) == {"料耗": 2}
    # A word already in the glossary is not collateral.
    assert correct.collisions("料號", corpus, {"料耗"}) == {}
    assert correct.collisions("交貨", corpus, set()) == {}


def test_a_term_is_never_rewritten_into_another() -> None:
    """The glossary saying a word exists is also the glossary saying it is not a mistake.

    工序 and 供需 are both gongxu and both ordinary vocabulary in a manufacturing interview.
    Registering the one that was being overwritten is what lets them coexist — measured on real
    transcripts, that is a better protection than a tone rule, which would have saved three real
    words and cost seven genuine fixes.
    """
    def term(source: str) -> store.Term:
        return store.Term(id=0, source=source, lang="", mode="hint", category="", targets={})

    assert correct.Corrector([term("工序")]).fix("考慮你供需的狀況") == "考慮你工序的狀況"
    both = correct.Corrector([term("工序"), term("供需")])
    assert both.fix("考慮你供需的狀況") == "考慮你供需的狀況"
    assert both.fix("工序的部分") == "工序的部分"


def test_corrector_never_rewrites_a_near_rhyme() -> None:
    """A single edit of Mandarin pinyin is a different word, not a misspelling of the same one.

    Allowing one, over seven real transcripts and a thirty-three term glossary, rewrote 知道 to
    製造 156 times and 生產 to 生管 146 times — 1578 corruptions. Chinese must match exactly.
    """
    def term(source: str) -> store.Term:
        return store.Term(id=0, source=source, lang="", mode="hint", category="", targets={})

    c = correct.Corrector([term("製造"), term("生管"), term("呆料"), term("委外")])
    assert c.fix("我不知道這件事") == "我不知道這件事"
    assert c.fix("生產線的狀況") == "生產線的狀況"
    assert c.fix("這批材料還在") == "這批材料還在"
    assert c.fix("未來五年的規劃") == "未來五年的規劃"
    # What it must still catch: the same sound, a different character.
    assert c.fix("生館的排程") == "生管的排程"


def test_chinese_output_is_converted_to_traditional() -> None:
    """Whisper emits Simplified for zh regardless of the speaker; Simplified on the meeting-room
    TV is an immediately visible failure, so conversion is not optional."""
    assert asr._post("这个软件的质量", "zh") == "這個軟件的質量"
    converted = asr._post("我们下周确认软件进度", "zh")
    assert "下週" in converted, converted


def test_conversion_leaves_factory_vocabulary_alone() -> None:
    """s2twp rewrote 參數 to 引數 and 項目 to 專案 on a real factory meeting. The Taiwan phrase
    tables are a software glossary; this room welds. See the note in asr.py."""
    converted = asr._post("这些参数和项目都要确认", "zh")
    assert "參數" in converted and "項目" in converted, converted
    assert "引數" not in converted and "專案" not in converted, converted
    # And not the bare s2t either: it writes 纔 for 才, including for 才夠, which the glossary
    # protects by that spelling and the corrector then cannot match.
    assert asr._post("那才是你的成就，才够", "zh") == "那才是你的成就，才夠"
    # Other languages must pass through untouched, diacritics included.
    assert asr._post("Chúng ta cần xác nhận", "vi") == "Chúng ta cần xác nhận"
    assert asr._post("schedule and delay", "en") == "schedule and delay"

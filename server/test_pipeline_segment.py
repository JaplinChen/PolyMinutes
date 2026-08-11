"""The post-meeting merge of VAD-cut fragments, and the punctuation pass's absolute guard."""

from __future__ import annotations

from . import segment


def _row(source: str, start: float = 0.0, end: float | None = None, speaker: str = "S1",
         lang: str = "zh", status: str = "ok", refined: int = 0,
         translations: dict[str, str] | None = None) -> dict:
    return {"speaker": speaker, "lang": lang, "source": source, "start": start,
            "end_time": end, "status": status, "refined": refined,
            "translations": translations or {}}


def test_fragments_of_one_sentence_merge_and_real_breaks_do_not() -> None:
    """Same speaker, a breath apart, sentence unfinished — that is one sentence the VAD cut.
    Anything else — a real pause, another voice, a finished sentence — stays where it is."""
    rows = [_row("我們今天要討論的是", 0.0, 2.0),
            _row("交期的問題", 2.3, 4.0),
            _row("還有產能", 4.5, 5.0)]
    assert segment.merge_groups(rows) == [[0, 1, 2]]

    # A pause longer than MAX_GAP is a real stop.
    assert segment.merge_groups([_row("先到這裡", 0.0, 1.0), _row("下一題", 3.0, 4.0)]) == []
    # Another speaker answering is not a continuation.
    assert segment.merge_groups([_row("交期呢", 0.0, 1.0),
                                 _row("下週三", 1.2, 2.0, speaker="S2")]) == []
    # A sentence that already ended must not swallow the next one.
    assert segment.merge_groups([_row("就這樣決定。", 0.0, 1.0), _row("下一題", 1.2, 2.0)]) == []
    # A human-corrected line is ground truth, not raw VAD output.
    assert segment.merge_groups([_row("人工改過的", 0.0, 1.0, refined=1),
                                 _row("後半句", 1.2, 2.0)]) == []
    # A line from before end_time existed has no gap to measure — on either side: absorbing a
    # tail with no end would leave the merged line endless, and the clip picker reads the end.
    assert segment.merge_groups([_row("舊資料", 0.0, None), _row("後半句", 1.2, 2.0)]) == []
    assert segment.merge_groups([_row("前半句", 0.0, 1.0), _row("舊資料", 1.2, None)]) == []


def test_join_respects_the_language_and_carries_the_translations() -> None:
    zh_text, end, tr = segment.join(
        _row("我們今天要討論的是", 0.0, 2.0, translations={"en": "what we discuss today is"}),
        [_row("交期的問題", 2.3, 4.0, translations={"en": "the delivery problem"})])
    assert zh_text == "我們今天要討論的是交期的問題"
    assert end == 4.0
    assert tr == {"en": "what we discuss today is the delivery problem"}

    en_text, _, _ = segment.join(_row("we need to", 0.0, 1.0, lang="en"),
                                 [_row("check the schedule", 1.2, 2.0, lang="en")])
    assert en_text == "we need to check the schedule"


def test_punctuation_reply_may_only_insert_punctuation() -> None:
    """Strip the punctuation back out and the text must be exactly what went in — a reply that
    changed any word is a rewrite wearing punctuation as a disguise, and is dropped whole."""
    texts = ["我們今天要討論的是交期的問題", "下週三之前要給答案"]

    added = segment.parse_response("1: 我們今天要討論的是，交期的問題。", texts)
    assert added == ["我們今天要討論的是，交期的問題。", texts[1]]

    # One changed character voids the whole line.
    assert segment.parse_response("1: 我們今天要討論的是，交貨的問題。", texts) == texts
    # NONE, chatter, or an index outside the chunk all leave the originals standing.
    assert segment.parse_response("NONE", texts) == texts
    assert segment.parse_response("9: 我們今天要討論的是。", texts) == texts


def test_punctuation_prompt_numbers_every_line() -> None:
    prompt = segment.build_prompt(["第一句", "第二句"])
    assert "1: 第一句" in prompt and "2: 第二句" in prompt
    assert "不得增加、刪除或修改任何字" in prompt

"""The LLM correction pass, and the guards that decide what it is allowed to change.

A model asked to fix a transcript will improve it, and an improved sentence is one nobody said.
Every check here is about the difference between a substitution and a rewrite.
"""

from __future__ import annotations

from . import guards, refine, store


def test_refine_keeps_the_original_when_the_model_rewrites() -> None:
    """The LLM pass may substitute, never restructure.

    Asked to fix a transcript a model will happily improve it, and an improved sentence is one
    nobody said. Anything that changes too much of a line, or that changes the number of lines,
    is discarded in favour of the recognised text.
    """
    lines = [refine.Line("S1", "zh", "我們的料耗其實變動很大"),
             refine.Line("S1", "zh", "生技這邊先開始")]
    fixed = ["我們的料號其實變動很大", "生管這邊先開始"]
    rewrite = "我們的料號變動幅度相當大，這點需要注意"
    terms = [store.Term(id=0, source="生管", lang="", mode="hint", category="", targets={})]

    # Only changed lines come back, and an unmentioned line keeps its original text.
    assert refine.parse_response("1: " + fixed[0], lines, terms) == [fixed[0], lines[1].text]

    # A fluent rewrite of the same meaning must be refused.
    assert refine.parse_response("1: " + rewrite, lines, terms)[0] == lines[0].text

    # An index outside the chunk is the model losing count; it must not land on another line.
    assert refine.parse_response("7: " + fixed[0], lines, terms) == [l.text for l in lines]
    assert refine.parse_response("nothing numbered here", lines, terms) == [l.text for l in lines]

    # Rewriting most of a chunk is restructuring, not correcting; keep all of it.
    many = [refine.Line("S1", "zh", f"第{i}句話沒有問題") for i in range(6)]
    reply_all = chr(10).join(f"{i}: 第{i}句話有問題" for i in range(1, 6))
    assert refine.parse_response(reply_all, many, terms) == [l.text for l in many]


def test_refine_rejects_corrections_that_do_not_sound_alike() -> None:
    """A recognition error is something the recogniser heard. A correction that sounds nothing
    like the text it replaces was invented from context, not recovered from audio.

    All five cases came out of a local model correcting a real interview transcript.
    """
    term = lambda source: store.Term(id=0, source=source, lang="", mode="hint",
                                     category="", targets={})
    terms = [term("工程變更"), term("生管")]

    # Heard: 稍 as 早, 料號 as 料耗, 生管 as 生技.
    assert guards.accept("有聽到聲音嗎早等我一下", "有聽到聲音嗎稍等我一下", terms)
    assert guards.accept("我們的料耗其實變動很大", "我們的料號其實變動很大", terms)
    assert guards.accept("生技這邊先開始", "生管這邊先開始", terms)

    # Guessed: nothing that sounds like 選項 was spoken.
    assert not guards.accept("用延伸的吧他還沒投出來", "用選項的吧他還沒跳出來", terms)

    # Two nonsense characters inside a long sentence are a rounding error to a ratio and still
    # nonsense, so the sound test has an absolute ceiling as well. Both of these were proposed by
    # a local model on a real transcript.
    long_before = "因為你所有的夢表那些什麼包含你的一些標準工時那些全部都要工單的管理"
    assert not guards.accept(long_before, long_before.replace("夢表", "模具"), terms)
    # The same span may still be corrected when the glossary names the destination.
    assert guards.accept(long_before, long_before.replace("夢表", "報表"),
                         terms + [term("報表")])

    # Re-spacing is not a correction.
    assert not guards.accept("呃right nowswitch", "呃 right now switch", terms)

    # A glossary term may travel further, because the recogniser never knew it existed.
    assert guards.accept("一夕變更的流程", "工程變更的流程", terms)
    assert not guards.accept("一夕變更的流程", "工程變更的流程", [])

    # Latitude, not immunity. The term is compared against the text it replaced: measured across
    # the whole line instead, 土壤 became 交貨 and 祂 became 生管 on a real transcript, because
    # two unrelated syllables inside a long sentence look like a rounding error.
    delivery = [term("交貨"), term("生管"), term("收料")]
    long_line = "然後我們這邊的狀況是說土壤的時間會影響到後面所有的排程跟人力安排這件事"
    assert not guards.accept(long_line, long_line.replace("土壤", "交貨"), delivery)
    assert not guards.accept("祂那邊的排程", "生管那邊的排程", delivery)
    assert not guards.accept("浴室量的部分", "收料的部分", delivery)
    # What the latitude is for: the recogniser mangled the term, but it still sounds like it.
    assert guards.accept("那個申管會上系統", "那個生管會上系統", delivery)
    assert guards.accept("收糧的部分", "收料的部分", delivery)


def test_refine_converts_what_the_model_writes_in_simplified() -> None:
    """The recogniser's output is already Traditional; a Simplified character in a correction can
    only have come from the model. Converted character by character, not with the phrase table
    used on ASR output — that one rewrites 對象 to 物件, which is the speaker's word, not an error.
    """
    lines = [refine.Line("S1", "zh", "申報的保税料件"),
             refine.Line("S1", "zh", "這個對象要處理"),
             refine.Line("S1", "en", "the tax iten")]
    assert refine.parse_response("1: 申報的保税料號", lines)[0] == "申報的保稅料號"
    assert refine.parse_response("2: 這個對象要處裡", lines)[1] == "這個對象要處裡"
    assert refine.parse_response("3: the tax item", lines)[2] == "the tax item"


def test_refine_prompt_states_the_domain_and_the_terms() -> None:
    said, earlier, term = "一夕變更的流程", "前面說過的話", "工程變更"
    prompt = refine.build_prompt(
        [refine.Line("S1", "zh", said)],
        [refine.Line("S1", "zh", earlier)],
        [store.Term(id=0, source=term, lang="", mode="hint", category="", targets={})],
        "SAP ERP interview",
    )
    assert "SAP ERP interview" in prompt
    assert term in prompt and earlier in prompt
    assert f"1: {said}" in prompt


def test_refused_corrections_are_kept_as_evidence() -> None:
    """What the guards throw away names the system's own blind spots.

    A model that wants to write 工程變更 where the recogniser wrote 一夕變更 knows a term the
    glossary does not. The correction is still refused — it sounds nothing like what was heard —
    but the refusal is what scripts/learn_terms.py mines.
    """
    lines = [refine.Line("S1", "zh", "一夕變更的流程")]
    rejected: list[refine.Rejected] = []
    assert refine.parse_response("1: 工程變更的流程", lines, [], rejected) == [lines[0].text]
    assert [(r.original, r.candidate) for r in rejected] == [("一夕變更的流程", "工程變更的流程")]

    # Accepted corrections are not evidence of anything missing.
    rejected.clear()
    term = [store.Term(id=0, source="工程變更", lang="", mode="hint", category="", targets={})]
    assert refine.parse_response("1: 工程變更的流程", lines, term, rejected)[0] == "工程變更的流程"
    assert rejected == []


def test_discarded_chunks_are_counted_not_hidden() -> None:
    """A chunk thrown out whole leaves its lines exactly as recognised, which reads the same as a
    chunk that needed nothing. Over seven interviews eleven chunks were discarded — 275 lines that
    looked checked and were not."""
    lines = [refine.Line("S1", "zh", f"第{i}句話沒有問題") for i in range(6)]
    everything = chr(10).join(f"{i}: 第{i}句話有問題" for i in range(1, 6))

    coverage = refine.Coverage()
    coverage.lines = len(lines)
    assert refine.parse_response(everything, lines, None, None, coverage) == [l.text for l in lines]
    assert coverage.skipped == len(lines) and coverage.fraction == 1.0

    # A chunk that was actually read counts as covered, corrections or not.
    coverage = refine.Coverage()
    coverage.lines = len(lines)
    refine.parse_response("1: 第0句話有問題", lines, None, None, coverage)
    assert coverage.skipped == 0


def test_a_failed_chunk_is_retried_once_before_keeping_originals() -> None:
    """偶發的模型失敗重試一次就過；連兩次才放棄，且取消例外必須直接往上拋。"""
    lines = [refine.Line("S1", "zh", "料耗的問題")]

    calls = []

    def flaky(prompt: str) -> str:
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("boom")
        return "1: 料號的問題"

    coverage = refine.Coverage()
    assert refine.Refiner(flaky).refine(lines, coverage=coverage) == ["料號的問題"]
    assert len(calls) == 2 and coverage.skipped == 0

    def broken(prompt: str) -> str:
        raise RuntimeError("boom")

    coverage = refine.Coverage()
    progress = []
    out = refine.Refiner(broken).refine(
        lines, coverage=coverage, on_progress=lambda d, t: progress.append((d, t)))
    assert out == ["料耗的問題"]
    assert coverage.skipped == 1 and progress == [(1, 1)]

    def cancelled(prompt: str) -> str:
        raise refine.jobs.Cancelled()

    try:
        refine.Refiner(cancelled).refine(lines)
        raise AssertionError("Cancelled must propagate, not be retried")
    except refine.jobs.Cancelled:
        pass


def test_short_final_chunk_can_still_be_corrected() -> None:
    """One correction is a majority of a one-line chunk, and the transcript's last few lines
    always land in one. The restructuring guard needs a chunk to be about."""
    lines = [refine.Line("S1", "zh", "料耗的問題")]
    assert refine.parse_response("1: 料號的問題", lines) == ["料號的問題"]

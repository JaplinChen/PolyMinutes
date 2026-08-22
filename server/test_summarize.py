"""Self-checks for the summary pass. Run: python -m server.test_summarize

Model-free: the prompt, the sampler, the schema check and the retry loop are plain functions;
the LLM is a fake chat callable.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Callable

from . import summarize as S
from .summarize import SummaryLine


def _lines(counts: dict[str, int], chars: int = 100) -> list[SummaryLine]:
    out, i = [], 0
    speakers = sorted(counts)
    remaining = dict(counts)
    while any(remaining.values()):
        for sp in speakers:
            if remaining[sp]:
                remaining[sp] -= 1
                out.append(SummaryLine(sp, "zh", f"{i:04d}" + "x" * (chars - 4)))
                i += 1
    return out


def test_target_chars_clamps_both_ends_and_scales():
    assert S.target_chars(0) == 400
    assert S.target_chars(2400) == 400
    assert S.target_chars(12_000) == 2000
    assert S.target_chars(1_000_000) == 8000


def test_load_rules_builtin_default_when_file_absent():
    saved = S.RULES_PATH
    try:
        S.RULES_PATH = Path(tempfile.gettempdir()) / "no_such_summary_rules.md"
        assert S.load_rules() == S.DEFAULT_RULES
        custom = Path(tempfile.mkdtemp()) / "summary_rules.md"
        custom.write_text("- custom rule", encoding="utf-8")
        S.RULES_PATH = custom
        assert S.load_rules() == "- custom rule"
    finally:
        S.RULES_PATH = saved


def test_build_prompt_asks_for_one_language_with_rules_and_excerpt_note():
    lines = [SummaryLine("S1", "zh", "hello")]
    prompt = S.build_prompt(lines, "vi", "- my rule", sampled=True)
    assert "Vietnamese" in prompt
    assert "Chinese" not in prompt and "English" not in prompt
    assert "- my rule" in prompt
    assert "excerpt" in prompt
    assert "[0] S1(zh): hello" in prompt
    assert "excerpt" not in S.build_prompt(lines, "vi", "- my rule", sampled=False)


def test_build_prompt_injects_reference_as_background_before_the_transcript():
    lines = [SummaryLine("S1", "zh", "hello")]
    prompt = S.build_prompt(lines, "en", "-", reference="Agenda: Q3 采购计划\nAttendees: 廖仁成")
    assert "廖仁成" in prompt and "Q3 采购计划" in prompt
    assert "context only, not spoken" in prompt
    # Reference sits ahead of the transcript so the model reads the meeting's own terms first.
    assert prompt.index("廖仁成") < prompt.index("[0] S1(zh): hello")
    # No reference given → no background block at all.
    assert "Background notes" not in S.build_prompt(lines, "en", "-")


def test_build_prompt_truncates_an_overlong_reference():
    lines = [SummaryLine("S1", "zh", "hi")]
    prompt = S.build_prompt(lines, "en", "-", reference="x" * 5000)
    assert "x" * S.REFERENCE_BUDGET in prompt
    assert "x" * (S.REFERENCE_BUDGET + 1) not in prompt


def test_build_prompt_targets_original_total_not_sampled():
    lines = [SummaryLine("S1", "zh", "x" * 100)]
    prompt = S.build_prompt(lines, "en", "-", sampled=True, total_chars=24_000)
    assert str(S.target_chars(24_000)) in prompt


def test_sample_identity_under_budget():
    lines = _lines({"S1": 5}, chars=10)
    out, sampled = S.sample(lines, budget_chars=1000)
    assert out == lines and sampled is False


def test_sample_proportional_and_ordered_over_budget():
    lines = _lines({"S1": 60, "S2": 30, "S3": 10}, chars=100)
    out, sampled = S.sample(lines, budget_chars=3000)  # keep ~30%
    assert sampled is True
    counts = {sp: sum(1 for l in out if l.speaker == sp) for sp in ("S1", "S2", "S3")}
    assert 15 <= counts["S1"] <= 21
    assert 7 <= counts["S2"] <= 11
    assert 2 <= counts["S3"] <= 4
    positions = [lines.index(l) for l in out]
    assert positions == sorted(positions)


def test_sample_caps_the_result_when_per_speaker_floors_overshoot():
    """More speakers than the budget has room for must still come back within the cap.

    Each speaker keeps at least one line so a quiet voice is not sampled out, but with many speakers
    those floors add up: 100 speakers of one long line each used to return all 100, blowing the
    budget the function exists to enforce. A second even pass caps the result. Text is made unique so
    order can be checked without list.index collapsing identical lines.
    """
    lines = [S.SummaryLine(f"S{i}", "zh", f"{i:03d}-" + "x" * 50) for i in range(100)]
    out, sampled = S.sample(lines, budget_chars=200)
    assert sampled is True
    kept_chars = sum(len(l.text) for l in out)
    # Within a small tolerance of the cap — the even step lands on line boundaries, not exact chars.
    assert kept_chars <= 200 * 1.5, kept_chars
    # Still chronological after the second pass.
    idx = [int(l.text[:3]) for l in out]
    assert idx == sorted(idx), idx


def test_sample_keeps_a_quiet_speaker_against_a_talkative_one():
    """The per-speaker floor is what the cap must not undo: one long line from S1, fifty from S2."""
    lines = [S.SummaryLine("S1", "zh", "a" * 100)] + [S.SummaryLine("S2", "zh", "b") for _ in range(50)]
    out, _ = S.sample(lines, budget_chars=80)
    assert "S1" in {l.speaker for l in out}


def _valid(title="T", summary="S", decisions=None, actions=None, risks=None, open_questions=None) -> str:
    parts = [f"TITLE: {title}", "SUMMARY:", summary, "DECISIONS:"]
    parts += [f"- {d}" for d in (decisions or [])]
    parts += ["ACTIONS:"]
    parts += [f"- {text} || {speaker}" for text, speaker in (actions or [])]
    parts += ["RISKS:"]
    parts += [f"- {r}" for r in (risks or [])]
    parts += ["OPEN_QUESTIONS:"]
    parts += [f"- {q}" for q in (open_questions or [])]
    return "\n".join(parts)


def test_parse_response_accepts_valid_with_empty_sections():
    got = S.parse_response("noise before\n" + _valid())
    assert got == {"title": "T", "summary": "S", "decisions": [], "actions": [],
                   "risks": [], "open_questions": []}


def test_parse_response_parses_risks_and_open_questions():
    got = S.parse_response(_valid(risks=["交期可能延誤"], open_questions=["預算尚未確認（推測）"]))
    assert got["risks"] == [{"text": "交期可能延誤", "line": None}]
    assert got["open_questions"] == [{"text": "預算尚未確認（推測）", "line": None}]


def test_parse_response_extracts_and_verifies_line_citations():
    raw = "\n".join([
        "TITLE: T", "SUMMARY:", "S",
        "DECISIONS:", "- 交期延後 || 12", "- 無出處決議",
        "ACTIONS:", "- 追蹤供應商 || S1 || 12", "- 沒有行號 || S1",
        "RISKS:", "- 可能延誤 || 99",  # 99 is not a real line — dropped to None
        "OPEN_QUESTIONS:",
    ])
    got = S.parse_response(raw, valid_speakers=frozenset({"S1"}), valid_lines=frozenset({12}))
    assert got["decisions"] == [{"text": "交期延後", "line": 12}, {"text": "無出處決議", "line": None}]
    assert got["actions"] == [{"text": "追蹤供應商", "speaker": "S1", "line": 12},
                              {"text": "沒有行號", "speaker": "S1", "line": None}]
    assert got["risks"] == [{"text": "可能延誤", "line": None}]  # invented id cleared, text kept


def test_parse_response_keeps_multiline_summary():
    raw = _valid(summary="【背景】第一段\n【討論】第二段「原話」")
    assert S.parse_response(raw)["summary"] == "【背景】第一段\n【討論】第二段「原話」"


def test_parse_response_rejects_missing_title():
    try:
        S.parse_response("SUMMARY:\nS\nDECISIONS:\nACTIONS:")
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "TITLE" in str(e)


def test_parse_response_rejects_missing_summary():
    try:
        S.parse_response("TITLE: T\nDECISIONS:\nACTIONS:")
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "SUMMARY" in str(e)


def test_parse_response_parses_decisions_as_bullets():
    got = S.parse_response(_valid(decisions=["first", "second"]))
    assert got["decisions"] == [{"text": "first", "line": None}, {"text": "second", "line": None}]


def test_parse_response_clears_a_speaker_the_transcript_never_had():
    # A code outside the valid set is the model inventing an owner; the action text stays, the
    # false attribution is cleared to "" so the page shows it unassigned.
    raw = _valid(actions=[("send the report", "S9"), ("book the room", "S1")])
    got = S.parse_response(raw, valid_speakers=frozenset({"S1", "S2"}))
    assert got["actions"] == [{"text": "send the report", "speaker": "", "line": None},
                              {"text": "book the room", "speaker": "S1", "line": None}]


def test_parse_response_keeps_all_speakers_when_no_set_is_given():
    # Called without a set (a test, or a caller that does not have one), nothing is second-guessed.
    raw = _valid(actions=[("x", "S9")])
    assert S.parse_response(raw)["actions"] == [{"text": "x", "speaker": "S9", "line": None}]


def test_summarize_clears_invented_owners_end_to_end():
    # The set is derived from the lines summarize was given, not passed in — a fabricated code in
    # the reply is cleared without the caller doing anything.
    lines = [S.SummaryLine("S1", "zh", "先講進度"), S.SummaryLine("S2", "zh", "我負責報告")]
    reply = _valid(title="會議", summary="兩人討論進度",
                   actions=[("交報告", "S7"), ("訂會議室", "S2")])
    out, status = S.summarize(lines, ["zh"], lambda _p: reply)
    assert status == "ok"
    assert out["zh"]["actions"] == [{"text": "交報告", "speaker": "", "line": None},
                                    {"text": "訂會議室", "speaker": "S2", "line": None}]


def test_retry_prompt_carries_error_and_truncates_bad_reply():
    prompt = S.retry_prompt("ORIGINAL", "y" * 900, "the error")
    assert "ORIGINAL" in prompt
    assert "the error" in prompt
    assert "y" * 500 in prompt
    assert "y" * 501 not in prompt


def test_max_tokens_for():
    assert S.max_tokens_for(1000) == 2800


def _chat_by_lang(replies: dict[str, list[str]]) -> Callable[[str], str]:
    def chat(prompt: str) -> str:
        for lang, name in (("zh", "Chinese"), ("en", "English"), ("vi", "Vietnamese")):
            if name in prompt.splitlines()[0]:
                return replies[lang].pop(0)
        raise AssertionError("unknown language in prompt")
    return chat


def test_summarize_happy_path():
    lines = [SummaryLine("S1", "zh", "hello")]
    out, status = S.summarize(lines, ["zh", "en"],
                              _chat_by_lang({"zh": [_valid("zt")], "en": [_valid("et")]}))
    assert status == "ok"
    assert out["zh"]["title"] == "zt" and out["en"]["title"] == "et"


def test_summarize_retries_once_then_partial():
    calls = {"zh": ["garbage", "still garbage"], "en": [_valid("et")]}
    out, status = S.summarize([SummaryLine("S1", "zh", "hi")], ["zh", "en"], _chat_by_lang(calls))
    assert status == "partial"
    assert "zh" not in out and out["en"]["title"] == "et"
    assert not calls["zh"]  # both attempts consumed: one retry, no more


def test_summarize_retry_succeeds():
    calls = {"zh": ["garbage", _valid("zt")]}
    out, status = S.summarize([SummaryLine("S1", "zh", "hi")], ["zh"], _chat_by_lang(calls))
    assert status == "ok" and out["zh"]["title"] == "zt"


def test_summarize_failed_when_all_garbage():
    calls = {"zh": ["g", "g"], "en": ["g", "g"]}
    out, status = S.summarize([SummaryLine("S1", "zh", "hi")], ["zh", "en"], _chat_by_lang(calls))
    assert status == "failed" and out == {}


def test_summarize_stops_early_when_asked():
    stopped = iter([False, True, True])
    out, status = S.summarize([SummaryLine("S1", "zh", "hi")], ["zh", "en", "vi"],
                              _chat_by_lang({"zh": [_valid("zt")]}),
                              should_stop=lambda: next(stopped))
    assert status == "partial"
    assert list(out) == ["zh"]


def main() -> None:
    checks = sorted((n, f) for n, f in globals().items() if n.startswith("test_"))
    for name, fn in checks:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(checks)} passed")


if __name__ == "__main__":
    main()

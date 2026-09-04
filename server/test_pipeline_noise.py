"""What the recogniser produces that nobody said: collapsed decodes, room noise, boilerplate.

Every string here came out of a real recording. A filter that is too eager deletes speech, so the
checks come in pairs: the hallucination is dropped, and the real sentence that looks like it stays.
"""

from __future__ import annotations

from . import asr


def test_subtitling_credits_are_not_speech() -> None:
    """More of the same training data surfacing in the silence between speakers.

    These arrived once batching changed which hallucination Whisper reached for, which is a good
    reminder that the list is a list and not a rule.
    """
    for text in ("MING PAO CANADA MANGA", "MING PAO CANADA 字幕組",
                 "中文字幕由 Amara.org 社群提供", "本期影片就分享到這裡,謝謝收看",
                 "多謝您的收看,我們下期見!", "希望大家多多支援",
                 "多謝您收睇時局新聞,再會!", "歡迎收睇"):
        assert asr.is_hallucination(text), text

    # A sign-off is the end of a line; arranging a meeting is not.
    assert not asr.is_hallucination("我們下次見面再談這個")
    assert not asr.is_hallucination("本期的採購單要重新確認")
    assert not asr.is_hallucination("這個字幕要放在電視上")


def test_a_collapse_is_dropped_whoever_chose_the_language() -> None:
    """The check used to run only when a language was forced.

    So a first-pass auto-detect could return 產品 產品 產品 產品 產品, keep it, and have the
    language it invented for that counted as evidence of what the speaker speaks — which is how
    433 Chinese lines ended up labelled English across seven interviews.
    """
    assert asr.is_degenerate("產品 產品 產品 產品 產品")
    assert asr.is_degenerate("前來,前來,前來,前來,前來,前來,前來,前來")
    # Short repetition is how people talk.
    assert not asr.is_degenerate("大家好大家好")
    assert not asr.is_degenerate("介紹 介紹 介紹")


def test_degenerate_detects_collapsed_decode() -> None:
    # The real output of forcing zh on English audio.
    assert asr.is_degenerate("前來,前來,前來,前來,前來,前來,前來,前來,前來,前來,前來")
    assert asr.is_degenerate("the the the the the the the the the the the")


def test_noise_annotations_are_dropped() -> None:
    """Every one of these came out of ten minutes of room noise before a real meeting started."""
    for text in ("[MUSIC PLAYING]", "(static)", "[BLANK_AUDIO]", "(upbeat music)", "(indistinct)", "[static"):
        assert asr.is_noise(text), text


def test_youtube_boilerplate_is_dropped() -> None:
    """Whisper answers unreadable audio with subtitle sign-offs. On seven real interviews these
    were 15% of every Vietnamese line, and none of it was spoken."""
    for text in ("Cảm ơn các bạn đã theo dõi và đăng ký kênh của mình.",
                 "Hãy subscribe cho kênh La La School",
                 "Cảm ơn các bạn đã theo dõi và hẹn gặp lại.",
                 "您可以訂閱我們的頻道,並且請點選訂閱",
                 "明鏡及點點欄目",
                 # A fansub credit that slipped through in a silent gap on the 2026-08-22 recording,
                 # once the no-speech gate stopped dropping it by score alone. Simplified as emitted.
                 "整理&字幕志愿者 杨栋梁",
                 "I'll see you in a minute. Thanks for watching."):
        assert asr.is_hallucination(text), text


def test_broadcast_signoffs_are_dropped_even_when_spaced() -> None:
    """Verbatim from a factory morning meeting: three of these landed in a row at 12:21 with the
    real agenda starting at 12:36. The decoder writes Mandarin word by word, so the boilerplate
    arrives spaced — and every phrase here was tried only against the unspaced form before."""
    for text in ("本集完",
                 "本節目 繼續 更多 內容 歡迎收看",
                 "請您 關注",
                 "訂閱 我們的 頻道"):
        assert asr.is_hallucination(text), text


def test_boilerplate_is_caught_before_it_is_converted_to_traditional() -> None:
    """The line is judged before `_post` runs, and Whisper always emits Simplified.

    Every phrase in the list is written in Traditional, so a Simplified sign-off matched nothing
    and was then converted on its way into the transcript: 本期影片就分享到這裡 sat in a real
    meeting transcript while `本期(影片|節目)` had been in the pattern all along.
    """
    for text in ("本期视频就分享到这里", "欢迎收看", "请您关注", "订阅我们的频道", "谢谢观看"):
        assert asr.is_hallucination(text), text

    for text in ("本集团今年的目标是降低不良率", "这个议题请大家多关注", "我们要订阅这个服务吗"):
        assert not asr.is_hallucination(text), text


def test_hallucination_filter_spares_real_speech() -> None:
    """Matched as phrases: a meeting may say 訂閱 or subscribe without meaning a channel."""
    for text in ("Bây giờ mình hiện tại đang làm thủ công bằng Excel.",
                 "我們要訂閱這個服務嗎",
                 "我們的料號其實變動很大",
                 "這個 schedule 要 delay 一週",
                 # Measured on that same recording: 謝謝 appears in 39 lines, plenty of them real
                 # speech, so politeness is never on its own grounds for dropping a line.
                 "好 謝謝 再請社管去檢討一下空壓機的能力夠不夠",
                 "數量比較多 速度又快 之後我再做個整理 再告訴大家 謝謝",
                 # 本集團 and 節目 belong to a real meeting; only the broadcast phrasing goes.
                 "本集團今年的目標是降低不良率",
                 "尾牙的節目安排請各部門回報",
                 "這個議題請大家多關注"):
        assert not asr.is_hallucination(text), text


def test_noise_keeps_speech_containing_brackets() -> None:
    assert not asr.is_noise("這個 (ERP) 系統要換掉")
    assert not asr.is_noise("- All right")
    assert not asr.is_noise("")


def test_degenerate_accepts_normal_speech() -> None:
    assert not asr.is_degenerate("這個 schedule 要 delay 一週，我們下週再確認一次時程")
    assert not asr.is_degenerate("After early nightfall the yellow lamps would light up the squalid quarter")
    assert not asr.is_degenerate("Chúng ta cần xác nhận lại lịch trình vào tuần sau nhé")


def test_degenerate_ignores_short_text() -> None:
    """A terse reply must never be mistaken for a collapse."""
    assert not asr.is_degenerate("好的")
    assert not asr.is_degenerate("OK OK")


def test_the_sign_offs_a_real_meeting_produced() -> None:
    """Six spellings that walked through the list, taken verbatim from one factory meeting.

    Each escaped for its own reason: the credit was a name the list could not know, 本期的影片
    had a 的 in the middle, 字幕小組 an extra character, 下回 was simply absent, and
    本集就這樣結束了 put four words between 本集 and its ending.
    """
    for text in ("剪輯 李宗盛", "本期的影片到這裡 再見", "中文字幕 沛隊字幕小組",
                 "我們下回見 再見", "謝謝您的收看", "下部節目了",
                 "本集就這樣結束了", "本期完", "本期播放",
                 "謝謝大家的收看", "中文字幕提供",
                 # From the 2026-08-05 re-run, over near-silence in the middle of the QC report.
                 "多謝您的觀看 下次再見", "多謝您的觀看"):
        assert asr.is_hallucination(text), text
    # Simplified too: the filter runs before `_post` converts.
    for text in ("谢谢您的收看", "本集就这样结束了"):
        assert asr.is_hallucination(text), text


def test_the_new_sign_off_patterns_spare_a_factory_meeting() -> None:
    """These run on the live path as well, where a false positive drops a subtitle and leaves
    nothing to show it happened. Every widened pattern gets the sentence it must not eat."""
    for text in ("剪輯機台的參數要改",          # 剪輯 as a verb about machinery
                 "影片剪輯外包給誰",            # 剪輯 mid-sentence
                 "剪輯 這個要重做一次不然來不及",  # 剪輯 then real speech, not a name
                 "我們下回見面再談",            # 下回見 continuing into a sentence
                 "我們下次再見面談這個",         # 下次再見 continuing into a sentence
                 "感謝大家的努力",              # real, from the same meeting as 多謝您的觀看
                 "感謝大家的幫忙",              # ditto, the line right after it
                 "本集團今年的目標是降低不良率",   # 本集團, not 本集
                 "本集團的專案結束了",           # 本集團 with an ending after it
                 "尾牙的節目安排請各部門回報",     # 節目 without 本/下部
                 "下部品的檢驗要加嚴",           # 下部 without 節目
                 "這個階段到此告一段落",         # 到此 without 本集
                 "會議到此結束 謝謝大家",        # 結束 without 本集
                 "這個議題請大家多關注",         # 關注 without 請您/敬請/歡迎
                 # This room builds subtitling software and says 字幕 in earnest, so the credit
                 # patterns that overlap real vocabulary match whole lines only.
                 "會議紀錄的中文字幕要不要做",
                 "中文字幕的部分請廠商報價",
                 "字幕提供給客戶的版本",
                 "謝謝大家 我們看一下這份報告"):
        assert not asr.is_hallucination(text), text


def test_english_signoffs_and_single_token_collapse() -> None:
    """Straight off a real transcript page: seven of eleven visible lines were these."""
    for text in ("Thank you for watching!", "Thank you for watching. See you next time.",
                 "Thank you very much.", "點選小鈴鐺,並按下小鈴鐺,才能收到最新訊息"):
        assert asr.is_hallucination(text), text
    # Thanks with a subject is someone talking.
    assert not asr.is_hallucination("Thank you very much for the report")
    assert not asr.is_hallucination("謝謝你幫我確認這個")

    assert asr.is_degenerate("YAMAHA YAMAHA YAMAHA YAMAHA YAMAHA")
    assert not asr.is_degenerate("YAMAHA YAMAHA")

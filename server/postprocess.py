"""Post-meeting pass over the recording.

The live pipeline trades accuracy for latency: a small model, online clustering that cannot see
what comes later, and one-shot refinement. None of those constraints apply once the meeting ends,
so this re-runs the whole thing from the wav with the largest model available and clusters over
every segment at once, then rewrites the stored transcript.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from . import asr, asr_gpu, config, correct, diarize, jobs, translate
from .store import Store

log = logging.getLogger("polyminutes.postprocess")

# Utterances a speaker must have before their own language statistics outweigh the meeting's.
MIN_LANGUAGE_EVIDENCE = 4
# Utterances handed to the recogniser at once. Long enough to amortise the encoder, short enough
# that an interrupted run has reported most of what it did.
BATCH_UTTERANCES = 64
# The shortest piece a speaker change is allowed to carve out of an utterance. Below this it is
# someone agreeing mid-sentence, and cutting there costs a transcript line to gain nothing.
MIN_PIECE_SECONDS = 0.5
# The shortest utterance the post-meeting pass will hand to the recogniser. The live path keeps the
# VAD's own 0.25 because a one-word confirmation belongs on the TV; here the opposite is true.
# Whisper never declines — handed a fragment with no sentence in it, it writes the YouTube subtitle
# sign-offs it read most, and a real meeting came back with "剪輯 李宗盛" four times. Measured on a
# 2h19m factory meeting, VAD then turn-splitting, against seventeen known sign-off lines:
#
#     min speech   utterances   speech kept   sign-offs surviving   long real lines kept
#           0.25          958      105.8 min             15 of 17              6 of 6
#           0.50          894      103.5 min             11 of 17              6 of 6
#           0.80          824       99.7 min              8 of 17              6 of 6
#           1.00          743       94.7 min              7 of 17              5 of 6
#
# 0.80 is the last value that costs no long line. Above it the curve flattens and real speech
# starts going: at 1.00 one of the six reference reports is gone.
POST_MEETING_MIN_SPEECH = 0.80
# Utterances averaged into a speaker's stored voiceprint. Their longest few: enough for a stable
# centroid, and a fixed cost per speaker rather than one embedding per utterance in the meeting.
VOICEPRINT_SAMPLES = 5


@dataclass
class Utterance:
    start: float
    samples: np.ndarray
    speaker: str = ""
    lang: str = ""
    text: str = ""


def best_model() -> Path:
    """Largest Whisper tier present on disk. Accuracy matters here, speed does not."""
    available = config.available_whisper_models()
    if not available:
        raise FileNotFoundError(f"no Whisper model found under {config.MODELS_DIR}")
    order = list(config.WHISPER_DIRS)
    return config.WHISPER_DIRS[max(available, key=order.index)]


def segment(wav: Path) -> list[Utterance]:
    audio, rate = sf.read(str(wav), dtype="float32")
    if rate != config.SAMPLE_RATE:
        raise ValueError(f"{wav} is {rate} Hz, expected {config.SAMPLE_RATE}")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    vad = asr.Vad(min_speech=POST_MEETING_MIN_SPEECH)
    out: list[Utterance] = []
    for i in range(0, len(audio), config.BLOCK_SIZE):
        out += [Utterance(s.start, s.samples) for s in vad.push(audio[i : i + config.BLOCK_SIZE])]
    out += [Utterance(s.start, s.samples) for s in vad.flush()]
    return out


def split_on_turns(utterances: list[Utterance], turns: list[diarize.Turn]) -> list[Utterance]:
    """Cut every utterance where the speaker changes inside it, and label each piece.

    A VAD utterance is speech between silences, which is not the same thing as one person talking:
    on a 2h19m meeting 72 of 859 held more than one voice, one of them four people over twelve
    seconds. Those became a single averaged embedding and a single transcript line reading as
    nonsense, because it was four people's sentences run together.

    Pieces below MIN_PIECE_SECONDS are folded into the neighbour they touch rather than kept: a
    half-second of someone agreeing in the middle of a sentence is not a turn worth a line.
    """
    if not turns:
        return utterances

    out: list[Utterance] = []
    for u in utterances:
        seconds = len(u.samples) / config.SAMPLE_RATE
        cuts = _cut_points(u.start, u.start + seconds, turns)
        if len(cuts) < 2:
            u.speaker = _speaker_code(_dominant(u.start, u.start + seconds, turns))
            out.append(u)
            continue
        for begin, end, who in cuts:
            head = int(round((begin - u.start) * config.SAMPLE_RATE))
            tail = int(round((end - u.start) * config.SAMPLE_RATE))
            out.append(Utterance(begin, u.samples[head:tail], speaker=_speaker_code(who)))

    _inherit_missing(out)
    return out


def _inherit_missing(utterances: list[Utterance]) -> None:
    """Give the segmenter's blind spots to whoever was talking around them.

    A half-second the VAD called speech and the segmentation model called nobody would otherwise
    reach the transcript with a blank where the speaker goes. Five of 1040 pieces on a real
    meeting, all under a second. Same rule the clustering path applies to a clip too short to
    embed: it belongs to the person already speaking.
    """
    previous = ""
    for u in utterances:
        if u.speaker:
            previous = u.speaker
        else:
            u.speaker = previous
    # Anything before the first labelled piece has nothing behind it to inherit from.
    following = ""
    for u in reversed(utterances):
        if u.speaker:
            following = u.speaker
        else:
            u.speaker = following


def _cut_points(start: float, end: float, turns: list[diarize.Turn]) -> list[tuple[float, float, int]]:
    """The utterance split into (start, end, speaker), or one span when nobody else speaks in it."""
    # Sorted here rather than assumed: out-of-order turns would walk `at` backwards and hand two
    # pieces the same audio, which is a duplicated transcript line rather than a crash.
    inside = sorted((t for t in turns if min(t.end, end) - max(t.start, start) >= MIN_PIECE_SECONDS),
                    key=lambda t: t.start)
    if len({t.speaker for t in inside}) < 2:
        return [(start, end, _dominant(start, end, turns))]

    pieces: list[tuple[float, float, int]] = []
    at = start
    for turn in inside:
        edge = min(turn.end, end)
        if edge - at >= MIN_PIECE_SECONDS:
            pieces.append((at, edge, turn.speaker))
            at = edge
    if not pieces:
        return [(start, end, _dominant(start, end, turns))]
    # Whatever is left belongs to whoever held the last piece: trailing audio after the final turn
    # is the same person tailing off, not a new one.
    last_start, _, last_who = pieces[-1]
    pieces[-1] = (last_start, end, last_who)
    return pieces


def _dominant(start: float, end: float, turns: list[diarize.Turn]) -> int:
    """Whoever holds most of this span. -1 when the segmenter heard nobody in it."""
    best, best_overlap = -1, 0.0
    for turn in turns:
        overlap = min(turn.end, end) - max(turn.start, start)
        if overlap > best_overlap:
            best, best_overlap = turn.speaker, overlap
    return best


def _speaker_code(speaker: int) -> str:
    """Segmenter ids are arbitrary integers; the transcript shows S1, S2, ... in that order."""
    return f"S{speaker + 1}" if speaker >= 0 else ""


def _remember_voices(store: Store, session_id: int, utterances: list[Utterance],
                     diarizer: diarize.Diarizer) -> dict[str, str]:
    """Store one voiceprint per speaker, and recognise the ones the room already knows.

    Only the live pipeline stored voiceprints, which meant an imported recording never taught
    anything: naming S3 on the transcript page looked up a voiceprint that had never been written,
    found nothing, and quietly skipped promoting it. The whole point of the naming screen is that
    the next meeting recognises the voice.

    Returns the codes a known voice was recognised on, so a reprocess can put the learned names
    back rather than dropping the room to anonymous Sn again.
    """
    groups: dict[str, list[Utterance]] = {}
    for u in utterances:
        if u.speaker and len(u.samples) / config.SAMPLE_RATE >= config.MIN_EMBED_SECONDS:
            groups.setdefault(u.speaker, []).append(u)

    recognised: dict[str, str] = {}
    for code, said in groups.items():
        # Their longest few, not everything they said: a centroid is an average, and averaging the
        # clearest samples costs a fixed handful of embeddings per speaker instead of one per
        # utterance in the meeting.
        best = sorted(said, key=lambda u: -len(u.samples))[:VOICEPRINT_SAMPLES]
        centroid = np.mean([diarizer.embed(u.samples) for u in best], axis=0).astype(np.float32)
        store.save_voiceprint(session_id, code, centroid.tobytes())
        if name := diarizer._recognise(centroid):
            recognised[code] = name
    return recognised


def assign_speakers(utterances: list[Utterance], diarizer: diarize.Diarizer) -> None:
    """Cluster over the whole meeting at once.

    The fallback for a machine without the segmentation model. Clusters whole VAD utterances, so
    an utterance holding two people is one embedding averaging both.

    This is what the online pass could not do: a speaker whose first few seconds were atypical
    gets merged with their later segments instead of living on as a phantom second participant.
    """
    # Same rule the live path applies in Diarizer.assign: a clip too short to embed reliably
    # inherits the previous speaker. Clustering it instead mints a phantom participant per blip —
    # on a real recording the ten idle minutes before the meeting produced fourteen of them.
    long = [u for u in utterances if len(u.samples) / config.SAMPLE_RATE >= config.MIN_EMBED_SECONDS]
    if not long:
        for u in utterances:
            u.speaker = "S1"
        return

    labels = diarize.cluster_offline([diarizer.embed(u.samples) for u in long])
    for utterance, label in zip(long, labels):
        utterance.speaker = f"S{label + 1}"

    previous = f"S{labels[0] + 1}"
    for u in utterances:
        if u.speaker:
            previous = u.speaker
        else:
            u.speaker = previous


def dominant_languages(utterances: list[Utterance]) -> dict[str, str]:
    """Majority language per speaker, computed after clustering rather than as the meeting ran.

    A speaker needs to have said enough for a majority to mean anything. Raising the clustering
    threshold to separate real participants also produces a long tail of speakers holding two or
    three utterances, and letting those establish their own language put 433 Chinese lines under
    an English label across seven interviews — 產品 產品 產品 decoded as English because one
    stray detection was all the evidence there was.

    Below the minimum a speaker inherits the meeting's language, which is a far better guess than
    a coin flip on two samples.
    """
    counts: dict[str, dict[str, int]] = {}
    overall: dict[str, int] = {}
    for u in utterances:
        # Text-less utterances are dropped noise; their detected language is Whisper guessing at
        # static and must not vote.
        if u.lang and u.text:
            counts.setdefault(u.speaker, {})[u.lang] = counts.setdefault(u.speaker, {}).get(u.lang, 0) + 1
            overall[u.lang] = overall.get(u.lang, 0) + 1

    if not overall:
        return {}
    meeting = max(overall, key=overall.get)
    return {code: (max(langs, key=langs.get) if sum(langs.values()) >= MIN_LANGUAGE_EVIDENCE
                   else meeting)
            for code, langs in counts.items() if langs}


def transcribe_all(utterances: list[Utterance], transcriber: asr.Transcriber,
                   progress: Callable[[Utterance, int, int], None] | None = None,
                   forced: dict[str, str] | None = None) -> None:
    """Two passes: detect each speaker's language, then re-transcribe anyone who was decoded
    under a language that disagrees with their majority.

    `forced` maps a speaker code to the language the room has set for that recognised voice. It wins
    over the detected majority: the majority is a guess from auto-detect, which flips a Chinese
    speaker to Vietnamese often enough that a voice the room has already identified should not be
    left to it. An empty or absent entry falls back to the majority, unchanged.

    `progress` is called after each first-pass decode. A ninety-minute recording spends most of an
    hour in that first loop, and a caller with somewhere to put partial results should not have to
    wait for the whole thing to survive an interruption.
    """
    # The first pass is the expensive one and every utterance in it wants the same thing —
    # auto-detect — so it goes through the recogniser in batches when the recogniser has a batch
    # mode. Boundaries are preserved either way; only the number of round trips changes.
    batch = getattr(transcriber, "transcribe_many", None)
    if batch:
        done = 0
        for start in range(0, len(utterances), BATCH_UTTERANCES):
            group = utterances[start : start + BATCH_UTTERANCES]
            for u, (text, lang) in zip(group, batch([g.samples for g in group], "")):
                u.text, u.lang = text, lang
                done += 1
                if progress:
                    progress(u, done, len(utterances))
    else:
        for i, u in enumerate(utterances, 1):
            u.text, u.lang = transcriber.transcribe(u.samples, "")
            if progress:
                progress(u, i, len(utterances))

    dominant = dominant_languages(utterances)
    forced = forced or {}
    for u in utterances:
        want = forced.get(u.speaker) or dominant.get(u.speaker, "")
        # An utterance the first pass refused is retried too, not only one decoded under the wrong
        # language. Auto-detect collapses on perfectly ordinary speech often enough that dropping
        # those outright cost 992 real lines across seven interviews — 掃描機這件事情有 and
        # 就會直接進到系統變成需求 among them. The speaker's own language usually recovers them.
        if want and (not u.text or want != u.lang):
            text, used = transcriber.transcribe(u.samples, want)
            # Still empty means the speaker's own language decoded this as noise as well, which is
            # what static sounds like to Whisper. Drop it rather than keep a phantom line.
            u.text, u.lang = (text, used) if text else ("", u.lang)


def translated_rows(entries, store: Store, cfg: config.Config,
                    translator: translate.Translator | None,
                    stop: Callable[[], bool]) -> list[dict]:
    """Corrected, translated line rows from (start, end_time, speaker, lang, text) entries.

    One loop for both ways a transcript comes to exist — decoded from audio or read off a subtitle
    track — so the corrector, the rolling context, the glossary and the failed-translation status
    behave identically whichever produced the text.
    """
    terms = store.glossary()
    corrector = correct.Corrector(terms, store.corrections())
    context: list[translate.Line] = []
    rows: list[dict] = []
    for start, end_time, speaker, lang, text in entries:
        if stop():
            raise jobs.Cancelled()
        if not text:
            continue
        text = corrector.fix(text)
        line = translate.Line(text=text, lang=lang, speaker=speaker)
        targets = [c for c in cfg.languages if c != lang]
        translations: dict[str, str] = {}
        status = "ok"
        if translator and targets:
            try:
                translations = translator.translate(
                    line, targets, context=context[-config.CONTEXT_LINES:], terms=terms
                ).translations
            except Exception:
                # Recorded rather than swallowed: a line with no translation and a line whose
                # translation failed look identical in the transcript, and only one of them is
                # worth re-running.
                log.exception("translation failed at %.2fs", start)
                status = "translate_failed"
        rows.append({
            "start": start,
            "end_time": end_time,
            "speaker": speaker,
            "lang": lang,
            "source": text,
            "translations": translations,
            "status": status,
        })
        context.append(line)
    return rows


def subtitle_session(store: Store, session_id: int, cues: list[tuple[float, float, str]],
                     lang: str, cfg: config.Config,
                     translator: translate.Translator | None = None,
                     should_stop: Callable[[], bool] | None = None) -> list[dict]:
    """Store a transcript taken from a subtitle track: no decode, no card, one unknown speaker.

    A subtitle track carries no voices, so every line lands on S1 — reprocess runs the full
    pipeline from the audio if the speakers matter. Everything else is the shared row loop, so the
    lines read like any other transcript's.
    """
    stop = should_stop or (lambda: False)
    rows = translated_rows(((s, e, "S1", lang, t) for s, e, t in cues), store, cfg,
                           translator, stop)
    if not rows:
        raise ValueError("the subtitle track had no usable cues")
    store.replace_lines(session_id, rows)
    return rows


def rewrite_session(store: Store, session_id: int, wav: Path, cfg: config.Config,
                    translator: translate.Translator | None = None,
                    should_stop: Callable[[], bool] | None = None,
                    gpu: Callable[[], AbstractContextManager] = contextlib.nullcontext,
                    ) -> list[Utterance]:
    """Re-derive the transcript and replace the stored lines for this session.

    `should_stop` lets a meeting starting in the room take the GPU back. It is polled between
    decode batches and between translations, and acting on it costs nothing: the stored transcript
    is untouched until `replace_lines` at the very end, so an abandoned pass leaves the session
    exactly as it found it.

    `gpu` guards the one stage that uses the card. Only decoding does: the VAD, the speaker
    segmentation and the per-line translation round trips all run on the CPU or the network, and
    they are most of the wall clock — on a 2h19m recording the segmentation alone is eight
    minutes. Holding the card across them means someone pressing record waits for all of it, and
    `claim_gpu` gives up after thirty seconds.

    The caller passes the guard rather than this function taking one, because the two callers need
    different behaviour and neither can be inferred from here: an import queues for the card
    (`borrow_gpu`), while the pass scheduled after a meeting must yield to the next one. Taking a
    gate here as well as in the caller would deadlock — `jobs.py`'s gate is a semaphore of one and
    the scheduled path waits on it without a timeout.
    """
    stop = should_stop or (lambda: False)
    # GPU first. The CPU fallback keeps float32 weights and every core: this runs after the
    # meeting, so accuracy is the only concern — but it also makes the machine unusable while it
    # runs, which is the other reason the GPU path exists.
    transcriber = (asr_gpu.maybe(cfg.languages, asr_gpu.hotwords_from(store.glossary()))
                   or asr.Transcriber(model_dir=best_model(), quantized=False,
                                      num_threads=os.cpu_count() or 4, languages=cfg.languages))
    # Known voices loaded so a re-derive can put the room's learned names back on the freshly
    # clustered codes — a reprocess renumbers speakers from scratch, and without this every name the
    # user typed would land on a different person or none at all.
    diarizer = diarize.Diarizer(cfg=cfg, known=diarize.load_known(store))

    utterances = segment(wav)
    log.info("%d utterances from %s", len(utterances), wav.name)
    if not utterances:
        return []

    # Speaker turns where the model for them is installed, clustered whole utterances where it is
    # not. The first also decides the transcript's line boundaries, because a line that runs across
    # a speaker change belongs to neither of them.
    speaker_turns = diarize.turns(wav)
    if speaker_turns:
        before = len(utterances)
        utterances = split_on_turns(utterances, speaker_turns)
        log.info("%d turns split %d utterances into %d", len(speaker_turns), before, len(utterances))
    else:
        assign_speakers(utterances, diarizer)
    recognised = _remember_voices(store, session_id, utterances, diarizer)

    def watch(_u: Utterance, _done: int, _total: int) -> None:
        if stop():
            raise jobs.Cancelled()

    # A recognised voice with a language set is transcribed in it, not in whatever auto-detect
    # guessed — the room's Vietnamese speaker stays Vietnamese, its Chinese speakers stay Chinese.
    languages = store.speaker_languages()
    forced = {code: lang for code, name in (recognised or {}).items() if (lang := languages.get(name))}

    with gpu():
        transcribe_all(utterances, transcriber, progress=watch, forced=forced)

    # The stored transcript is not touched until every line is translated. Replacing it line by
    # line as they came in meant a failure halfway through left the session holding half a
    # transcript, and a failure right after the delete left it holding none.
    entries = ((u.start, u.start + len(u.samples) / config.SAMPLE_RATE, u.speaker, u.lang, u.text)
               for u in utterances)
    rows = translated_rows(entries, store, cfg, translator, stop)

    # replace_lines deletes the old transcript before inserting the new. An empty result would make
    # that a bare delete — and a re-transcription that decoded nothing (every utterance came back
    # empty without raising) must not wipe a transcript that already exists, the same way the
    # no-utterances case above returns without touching it. A session that never had lines is still
    # allowed to be created empty, as a silent import is.
    if not rows and store.lines(session_id):
        raise ValueError("re-transcription produced no lines; keeping the existing transcript")
    store.replace_lines(session_id, rows)
    # Only now that the new transcript is committed: the old names were keyed to the old codes, which
    # this pass renumbered, so they are cleared and replaced by whatever the learned voices matched.
    # Done here, not beside _remember_voices, so a cancelled pass leaves the names as it found them.
    store.clear_speaker_names(session_id)
    for code, name in recognised.items():
        store.set_speaker_name(session_id, code, name)
    return utterances


def _summary_markdown(store: Store, session_id: int, names: dict[str, str]) -> list[str]:
    """The summary block of the export, every generated language in full.

    The export's reader is whoever was not in the room, so unlike the page there is no interface
    language to pick by — all of them go in. A summary generated before the transcript's latest
    edit is still included, but says so: an exported file that silently carried outdated decisions
    would be trusted precisely because it is a file.
    """
    row = store.summary(session_id)
    if not row or row["status"] == "failed":
        return []
    try:
        per_language = json.loads(row["json"])
    except ValueError:
        return []

    session = store.session(session_id)
    stale = bool(session) and int(session["lines_rev"]) != int(row["lines_rev"])

    out = ["## 會議摘要", ""]
    if stale:
        out += ["> ⚠ 摘要生成後逐字稿曾被修改，內容可能與下方逐字稿不一致。", ""]
    for lang, s in per_language.items():
        out += [f"### {s.get('title') or lang}（{lang}）", "", s.get("summary", ""), ""]
        if s.get("decisions"):
            out += ["**決議**", ""] + [f"- {d}" for d in s["decisions"]] + [""]
        if s.get("actions"):
            out += ["**行動項目**", ""]
            out += [f"- {names.get(a.get('speaker', ''), a.get('speaker')) or '（未指定）'}："
                    f"{a.get('text', '')}" for a in s["actions"]]
            out += [""]
    return out


def to_markdown(store: Store, session_id: int) -> str:
    """Speaker-attributed transcript with every language stacked under each turn."""
    lines = store.lines(session_id)
    names = store.speaker_names(session_id)
    if not lines:
        return "# 會議紀錄\n\n（無內容）\n"

    out = ["# 會議紀錄", ""]
    out += _summary_markdown(store, session_id, names)
    # Numeric, not lexicographic: codes are S1..S35, and a plain string sort put S10 between S1 and
    # S2. The tuple key keeps any non-S<n> code (there should be none) after the numbered ones
    # without an int-vs-str comparison error.
    speakers = sorted({l["speaker"] for l in lines},
                      key=lambda s: (0, int(s[1:])) if s[1:].isdigit() else (1, s))
    out += ["## 發言者", ""]
    out += [f"- **{names.get(code, code)}**" + ("" if code in names else "（未命名）") for code in speakers]
    out += ["", "## 逐字稿", ""]

    for line in lines:
        stamp = f"{int(line['start']) // 60}:{int(line['start']) % 60:02d}"
        who = names.get(line["speaker"], line["speaker"])
        out.append(f"**[{stamp}] {who}**")
        out.append(f"> {line['source']}")
        for lang, text in line["translations"].items():
            out.append(f"> _{lang}_ {text}")
        out.append("")

    return "\n".join(out) + "\n"


def to_docx(store: Store, session_id: int) -> bytes:
    """The same export as `to_markdown`, as a Word document.

    Enterprise meetings hand over a .docx, not a code fence. It carries the same three blocks in the
    same order — summary, speakers, transcript — and the same honesty: a summary older than the
    latest transcript edit is included but flagged, because a file is trusted precisely for being a
    file. python-docx is imported here, not at module load, so the recognition path never pays for a
    dependency only the export uses.
    """
    from docx import Document  # noqa: PLC0415 — export-only, kept off the hot import path

    lines = store.lines(session_id)
    names = store.speaker_names(session_id)
    doc = Document()
    doc.add_heading("會議紀錄", level=0)
    if not lines:
        doc.add_paragraph("（無內容）")
        return _docx_bytes(doc)

    row = store.summary(session_id)
    per_language = None
    if row and row["status"] != "failed":
        try:
            per_language = json.loads(row["json"])
        except ValueError:
            per_language = None
    if per_language:
        session = store.session(session_id)
        stale = bool(session) and int(session["lines_rev"]) != int(row["lines_rev"])
        doc.add_heading("會議摘要", level=1)
        if stale:
            doc.add_paragraph("⚠ 摘要生成後逐字稿曾被修改，內容可能與下方逐字稿不一致。")
        for lang, s in per_language.items():
            doc.add_heading(f"{s.get('title') or lang}（{lang}）", level=2)
            if s.get("summary"):
                doc.add_paragraph(s["summary"])
            if s.get("decisions"):
                doc.add_paragraph("決議").bold = True
                for d in s["decisions"]:
                    doc.add_paragraph(str(d), style="List Bullet")
            if s.get("actions"):
                doc.add_paragraph("行動項目").bold = True
                for a in s["actions"]:
                    who = names.get(a.get("speaker", ""), a.get("speaker")) or "（未指定）"
                    doc.add_paragraph(f"{who}：{a.get('text', '')}", style="List Bullet")

    speakers = sorted({l["speaker"] for l in lines},
                      key=lambda s: (0, int(s[1:])) if s[1:].isdigit() else (1, s))
    doc.add_heading("發言者", level=1)
    for code in speakers:
        doc.add_paragraph(names.get(code, code) + ("" if code in names else "（未命名）"),
                          style="List Bullet")

    doc.add_heading("逐字稿", level=1)
    for line in lines:
        stamp = f"{int(line['start']) // 60}:{int(line['start']) % 60:02d}"
        who = names.get(line["speaker"], line["speaker"])
        head = doc.add_paragraph()
        head.add_run(f"[{stamp}] {who}").bold = True
        doc.add_paragraph(line["source"])
        for lang, text in line["translations"].items():
            doc.add_paragraph(f"{lang}｜{text}")

    return _docx_bytes(doc)


def _docx_bytes(doc) -> bytes:
    import io
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()

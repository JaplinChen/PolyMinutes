"""Who is speaking: when a speaker's language may change, and how voices are grouped.

The language rules are hysteresis -- one disagreement is noise, several in a row is a speaker who
was misidentified -- and the clustering is judged by the speech it groups, not by cluster count.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from . import config, diarize, postprocess


def test_a_speaker_needs_evidence_before_setting_their_own_language() -> None:
    """Separating real participants also produces a tail of speakers holding two or three
    utterances, and a majority over two samples is a coin flip. Letting those establish their own
    language put 433 Chinese lines under an English label across seven interviews.
    """
    import numpy as np

    def said(speaker: str, lang: str) -> postprocess.Utterance:
        return postprocess.Utterance(0.0, np.zeros(1, dtype="float32"), speaker, lang, "x")

    meeting = [said("S1", "zh")] * 20 + [said("S2", "en")] * 6 + [said("S3", "en")] * 2
    dominant = postprocess.dominant_languages(meeting)
    assert dominant["S1"] == "zh"
    # Enough of its own to disagree with the room.
    assert dominant["S2"] == "en"
    # Not enough; inherits the meeting rather than guessing.
    assert dominant["S3"] == "zh"
    assert postprocess.dominant_languages([]) == {}


def test_clustering_is_judged_by_speech_not_cluster_count() -> None:
    """Two speakers who each hold a real share of the meeting must not be merged.

    The previous threshold was picked by counting clusters, which rewarded merging everyone into
    one: on a 37-minute interview it produced a single speaker holding 100% of the speech, and on
    a 67-minute one it produced 49 minutes against 14 where 0.65 finds 21 / 14 / 12 / 8.
    """
    import numpy as np

    # Two voices that a room microphone would leave closer together than a studio would.
    a = np.array([1.0, 0.35, 0.0], dtype=np.float32)
    b = np.array([0.35, 1.0, 0.0], dtype=np.float32)
    assert diarize.cosine(a, b) < config.SPEAKER_THRESHOLD, "the fixture must be separable"

    labels = diarize.cluster_offline([a, a * 0.9, b, b * 1.1])
    assert len(set(labels)) == 2, labels
    assert labels[0] == labels[1] and labels[2] == labels[3]


def test_known_voice_is_named_on_sight() -> None:
    """A voice the room has met before arrives named instead of as another anonymous Sn.

    Held to a stricter bar than in-meeting clustering: a wrong merge shows up as a split
    transcript, a wrong name is attributed to a real person and nobody thinks to check it.
    """
    import numpy as np

    vincent = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    stranger = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    d = diarize.Diarizer.__new__(diarize.Diarizer)
    d._known = [("Vincent", vincent)]

    assert d._recognise(vincent) == "Vincent"
    # Close, but not close enough to put someone's name on it.
    nearly = np.array([1.0, 1.3, 0.0], dtype=np.float32)
    assert diarize.cosine(nearly, vincent) < config.KNOWN_SPEAKER_THRESHOLD
    assert d._recognise(nearly) == ""
    assert d._recognise(stranger) == ""
    # An empty roster never guesses.
    d._known = []
    assert d._recognise(vincent) == ""


def test_cosine() -> None:
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    assert abs(diarize.cosine(a, a) - 1.0) < 1e-6
    assert abs(diarize.cosine(a, np.array([0.0, 1.0, 0.0], dtype=np.float32))) < 1e-6
    assert diarize.cosine(a, np.zeros(3, dtype=np.float32)) == 0.0


def _speaker(lang: str = "zh") -> diarize.Speaker:
    return diarize.Speaker(code="S1", centroid=np.zeros(4, dtype=np.float32), language=lang)


def _diarizer(cfg: config.Config) -> diarize.Diarizer:
    """A Diarizer without the ONNX extractor — language bookkeeping needs no model."""
    d = diarize.Diarizer.__new__(diarize.Diarizer)
    d._cfg = cfg
    d.speakers = []
    d._last_code = None
    d.recognised = {}
    d._known_languages = {}
    return d


def test_first_language_is_adopted_immediately() -> None:
    d, spk = _diarizer(config.Config()), diarize.Speaker(code="S1", centroid=np.zeros(4, dtype=np.float32))
    d.observe_language(spk, "vi")
    assert spk.language == "vi"


def test_single_disagreement_does_not_switch() -> None:
    d, spk = _diarizer(config.Config()), _speaker("vi")
    d.observe_language(spk, "en")
    assert spk.language == "vi"


def test_switch_after_enough_consecutive_disagreements() -> None:
    cfg = config.Config(language_switch_after=3)
    d, spk = _diarizer(cfg), _speaker("vi")
    for _ in range(3):
        d.observe_language(spk, "en")
    assert spk.language == "en"


def test_agreement_resets_the_pending_switch() -> None:
    """Alternating detections must not accumulate into a switch."""
    cfg = config.Config(language_switch_after=3)
    d, spk = _diarizer(cfg), _speaker("vi")
    for _ in range(2):
        d.observe_language(spk, "en")
        d.observe_language(spk, "vi")
    assert spk.language == "vi"


def test_zh_en_needs_a_higher_bar() -> None:
    """Taiwanese Mandarin embeds English constantly; the zh<->en pair must resist flipping."""
    cfg = config.Config(language_switch_after=3, language_switch_after_zh_en=6)
    d, spk = _diarizer(cfg), _speaker("zh")
    for _ in range(5):
        d.observe_language(spk, "en")
    assert spk.language == "zh", "flipped too early on code-switched speech"
    d.observe_language(spk, "en")
    assert spk.language == "en"


def test_pinned_language_never_changes() -> None:
    cfg = config.Config(pinned_languages={"S1": "zh"}, language_switch_after=1)
    d, spk = _diarizer(cfg), _speaker("zh")
    for _ in range(10):
        d.observe_language(spk, "en")
    assert spk.language == "zh"
    assert d.language_for(spk) == "zh"


def test_a_recognised_voice_is_forced_to_its_set_language() -> None:
    """A voice the room recognised and set to vi is transcribed in vi, even after auto-detect had
    established zh for it — identity beats a per-utterance guess on a noisy room mic."""
    d = _diarizer(config.Config())
    spk = _speaker("zh")  # auto-detect established zh
    d.recognised = {"S1": "阿笑"}
    d._known_languages = {"阿笑": "vi"}
    assert d.language_for(spk) == "vi"
    # A voice with no set language falls back to what auto-detect established.
    d._known_languages = {"阿笑": ""}
    assert d.language_for(spk) == "zh"
    # A voice that was not recognised is untouched.
    d.recognised = {}
    d._known_languages = {"阿笑": "vi"}
    assert d.language_for(spk) == "zh"


def test_transcribe_all_forces_a_recognised_speakers_language() -> None:
    """The forced map (a recognised voice's set language) overrides the detected majority: a
    Chinese speaker auto-detect flipped to vi is put back to zh, and re-decoded under it."""
    class FakeTranscriber:
        def transcribe_many(self, clips, language):
            return [("xin chào", "vi") for _ in clips]  # auto-detect flips this speaker to vi

        def transcribe(self, samples, language):
            return (f"forced-{language}", language)

    utts = [postprocess.Utterance(0.0, np.zeros(16000, dtype=np.float32), "S1")
            for _ in range(3)]
    postprocess.transcribe_all(utts, FakeTranscriber(), forced={"S1": "zh"})
    assert all(u.lang == "zh" and u.text == "forced-zh" for u in utts), [(u.lang, u.text) for u in utts]

    # Without a forced entry, the detected majority (vi) stands.
    utts2 = [postprocess.Utterance(0.0, np.zeros(16000, dtype=np.float32), "S1")]
    postprocess.transcribe_all(utts2, FakeTranscriber())
    assert utts2[0].lang == "vi" and utts2[0].text == "xin chào"


def test_offline_clustering_groups_similar_embeddings() -> None:
    rng = np.random.default_rng(0)
    a = rng.normal(size=64).astype(np.float32)
    b = rng.normal(size=64).astype(np.float32)
    # Three noisy takes of speaker A, two of speaker B.
    embeddings = [a + rng.normal(scale=0.05, size=64).astype(np.float32) for _ in range(3)]
    embeddings += [b + rng.normal(scale=0.05, size=64).astype(np.float32) for _ in range(2)]

    labels = diarize.cluster_offline(embeddings, threshold=0.5)
    assert len(set(labels)) == 2, labels
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4]
    assert labels[0] != labels[3]


def test_no_cluster_holds_two_segments_that_are_not_alike() -> None:
    """A cluster is only as tight as its worst pair — which is what stops it holding four people.

    Real embeddings arrive as a continuum, not as tidy blobs: on a 2h19m meeting the pairwise
    cosines ran smoothly from 0.22 to 0.97 with no gap to cut at. Averaging two clusters into a
    centroid produces a mean voice that resembles everyone a little, so under that rule the biggest
    cluster kept absorbing until it held 93% of the meeting and every one of 33 pairs from
    different people had been merged. The fixture below is that continuum in miniature; under
    centroid merging it puts segments 0.05 apart in one cluster.
    """
    angles = np.sort(np.random.default_rng(0).uniform(0, np.pi, 70))
    points = [np.array([np.cos(t), np.sin(t)], dtype=np.float32) for t in angles]

    labels = np.array(diarize.cluster_offline(points, threshold=0.65))
    for group in set(labels.tolist()):
        members = np.flatnonzero(labels == group)
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                score = diarize.cosine(points[members[a]], points[members[b]])
                assert score >= 0.65, f"cluster {group} holds a pair at {score:.3f}"
    assert len(set(labels.tolist())) > 1, "the fixture must not collapse into one cluster"


def test_offline_clustering_edge_cases() -> None:
    assert diarize.cluster_offline([]) == []
    single = [np.array([1.0, 0.0], dtype=np.float32)]
    assert diarize.cluster_offline(single) == [0]
    # Zero vectors score zero against everything rather than dividing by nothing, so they never
    # merge — the same rule `cosine` applies.
    zeros = [np.zeros(8, dtype=np.float32), np.zeros(8, dtype=np.float32)]
    assert len(set(diarize.cluster_offline(zeros))) == 2


def test_offline_clustering_scales_to_a_long_meeting() -> None:
    """A two-hour recording segments into the thousands, and clustering must not be the wall.

    Comparing every pair in Python each round is O(n^3): at this size it ran for over an hour with
    the GPU idle, so an import never reached transcription at all. The bound is deliberately loose
    — it is here to catch a return to cubic, not to police milliseconds.
    """
    import time

    rng = np.random.default_rng(7)
    bases = rng.normal(size=(5, 192)).astype(np.float32)
    embeddings = [(bases[k % 5] + rng.normal(scale=0.3, size=192)).astype(np.float32)
                  for k in range(1500)]

    start = time.perf_counter()
    labels = diarize.cluster_offline(embeddings)
    elapsed = time.perf_counter() - start

    assert len(set(labels)) == 5, len(set(labels))
    assert elapsed < 20, f"clustering 1500 segments took {elapsed:.1f}s"


def _utterance(start: float, seconds: float) -> postprocess.Utterance:
    """Silence of the right length: these checks are about boundaries, not about audio."""
    return postprocess.Utterance(start, np.zeros(int(seconds * config.SAMPLE_RATE), dtype=np.float32))


def test_an_utterance_is_cut_where_the_speaker_changes() -> None:
    """A VAD utterance is speech between silences, which is not one person talking.

    Measured on a 2h19m meeting: 72 of 859 utterances held more than one voice, one of them four
    people over twelve seconds — decoded as a single line reading as nonsense and embedded as a
    single voice belonging to nobody.
    """
    turns = [diarize.Turn(0.0, 4.0, 0), diarize.Turn(4.0, 10.0, 1)]
    pieces = postprocess.split_on_turns([_utterance(0.0, 10.0)], turns)

    assert [p.speaker for p in pieces] == ["S1", "S2"], [p.speaker for p in pieces]
    assert pieces[0].start == 0.0 and pieces[1].start == 4.0
    # The audio follows the cut, or the second piece would be transcribed as the first one's words.
    assert len(pieces[0].samples) == 4 * config.SAMPLE_RATE
    assert len(pieces[1].samples) == 6 * config.SAMPLE_RATE


def test_an_utterance_with_one_voice_is_left_whole() -> None:
    turns = [diarize.Turn(0.0, 9.0, 3), diarize.Turn(20.0, 30.0, 1)]
    pieces = postprocess.split_on_turns([_utterance(0.5, 8.0)], turns)

    assert len(pieces) == 1
    assert pieces[0].speaker == "S4"
    assert len(pieces[0].samples) == 8 * config.SAMPLE_RATE


def test_a_brief_interjection_does_not_cost_a_line() -> None:
    """Half a second of someone agreeing mid-sentence is not a turn worth splitting a line for."""
    turns = [diarize.Turn(0.0, 5.0, 0), diarize.Turn(2.0, 2.2, 1), diarize.Turn(5.0, 8.0, 0)]
    pieces = postprocess.split_on_turns([_utterance(0.0, 8.0)], turns)

    assert len(pieces) == 1, [(p.start, p.speaker) for p in pieces]
    assert pieces[0].speaker == "S1"


def test_splitting_covers_the_utterance_it_was_given() -> None:
    """Every second of audio has to land in exactly one piece: a gap is speech dropped from the
    transcript, an overlap is speech transcribed twice."""
    turns = [diarize.Turn(0.0, 3.0, 0), diarize.Turn(3.0, 7.0, 1), diarize.Turn(7.0, 12.0, 2)]
    pieces = postprocess.split_on_turns([_utterance(0.0, 12.0)], turns)

    assert len(pieces) == 3
    total = sum(len(p.samples) for p in pieces)
    assert total == 12 * config.SAMPLE_RATE, total
    for earlier, later in zip(pieces, pieces[1:]):
        assert earlier.start + len(earlier.samples) / config.SAMPLE_RATE == later.start


def test_no_turns_leaves_every_utterance_alone() -> None:
    """The machine without the segmentation model still has to process a recording."""
    given = [_utterance(0.0, 3.0), _utterance(4.0, 3.0)]
    assert postprocess.split_on_turns(given, []) is given


def test_a_gap_the_segmenter_heard_nobody_in_still_gets_a_speaker() -> None:
    """The VAD calls it speech, the segmentation model calls it nobody, and the transcript would
    show a line with a blank where the speaker goes. Five of 1040 pieces on a real meeting."""
    turns = [diarize.Turn(0.0, 3.0, 0), diarize.Turn(30.0, 40.0, 1)]
    # The middle utterance sits in the segmenter's blind spot.
    given = [_utterance(0.0, 3.0), _utterance(10.0, 0.6), _utterance(30.0, 5.0)]
    pieces = postprocess.split_on_turns(given, turns)

    assert [p.speaker for p in pieces] == ["S1", "S1", "S2"], [p.speaker for p in pieces]


def test_an_opening_blind_spot_borrows_from_what_follows() -> None:
    """Nothing precedes the first utterance, so inheriting backwards is the only option left."""
    turns = [diarize.Turn(20.0, 30.0, 4)]
    pieces = postprocess.split_on_turns([_utterance(0.0, 0.6), _utterance(20.0, 5.0)], turns)

    assert [p.speaker for p in pieces] == ["S5", "S5"], [p.speaker for p in pieces]


def test_out_of_order_turns_do_not_duplicate_audio() -> None:
    """Pieces are cut by walking forward; a turn list out of order would hand two of them the same
    seconds, which reaches the transcript as the same sentence twice."""
    turns = [diarize.Turn(6.0, 10.0, 1), diarize.Turn(0.0, 6.0, 0)]
    pieces = postprocess.split_on_turns([_utterance(0.0, 10.0)], turns)

    assert [p.speaker for p in pieces] == ["S1", "S2"]
    assert sum(len(p.samples) for p in pieces) == 10 * config.SAMPLE_RATE
    for earlier, later in zip(pieces, pieces[1:]):
        assert earlier.start + len(earlier.samples) / config.SAMPLE_RATE == later.start


def test_an_imported_recording_leaves_a_voiceprint_behind() -> None:
    """Naming a speaker is how the room learns a voice, and it reads what this wrote.

    Only the live pipeline stored voiceprints, so a session that arrived as a file never wrote one:
    naming S3 afterwards looked one up, found nothing, and skipped promoting it without saying so.
    """
    class FakeStore:
        def __init__(self) -> None:
            self.saved: dict[str, bytes] = {}

        def save_voiceprint(self, session_id: int, code: str, centroid: bytes) -> None:
            self.saved[code] = centroid

    class FakeEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, samples: np.ndarray) -> np.ndarray:
            self.calls += 1
            # Length stands in for identity, so the averaged centroid is checkable.
            return np.full(4, len(samples) / config.SAMPLE_RATE, dtype=np.float32)

        def _recognise(self, emb: np.ndarray) -> str:
            return ""  # no learned voices in this fixture

    said = [postprocess.Utterance(0.0, np.zeros(config.SAMPLE_RATE * 4, dtype=np.float32), "S1"),
            postprocess.Utterance(5.0, np.zeros(config.SAMPLE_RATE * 2, dtype=np.float32), "S1"),
            postprocess.Utterance(9.0, np.zeros(config.SAMPLE_RATE * 6, dtype=np.float32), "S2"),
            # Too short to embed reliably: it must not drag a speaker's centroid.
            postprocess.Utterance(20.0, np.zeros(int(config.SAMPLE_RATE * 0.4), dtype=np.float32), "S2")]

    store, embedder = FakeStore(), FakeEmbedder()
    postprocess._remember_voices(store, 7, said, embedder)

    assert sorted(store.saved) == ["S1", "S2"]
    assert np.frombuffer(store.saved["S1"], dtype=np.float32).tolist() == [3.0] * 4
    assert np.frombuffer(store.saved["S2"], dtype=np.float32).tolist() == [6.0] * 4
    assert embedder.calls == 3, "the sub-second clip must not have been embedded"


def test_reprocess_puts_learned_names_back_on_the_new_codes() -> None:
    """A re-derive renumbers speakers from scratch; the room's learned voices must name them again.

    Without this a reprocess dropped every name the user had typed — the old speaker_name rows were
    keyed to codes the re-clustering renumbered. _remember_voices now reports which fresh code each
    known voice landed on, so rewrite_session can write the names back.
    """
    class FakeStore:
        def __init__(self) -> None:
            self.saved: dict[str, bytes] = {}

        def save_voiceprint(self, session_id: int, code: str, centroid: bytes) -> None:
            self.saved[code] = centroid

    class FakeDiarizer:
        def embed(self, samples: np.ndarray) -> np.ndarray:
            # Length stands in for identity, so the centroid is a known value per speaker.
            return np.full(4, len(samples) / config.SAMPLE_RATE, dtype=np.float32)

        def _recognise(self, emb: np.ndarray) -> str:
            # The four-second voice is the one the room knows; the six-second one is a stranger.
            return "廖仁成" if abs(float(emb[0]) - 4.0) < 0.01 else ""

    said = [postprocess.Utterance(0.0, np.zeros(config.SAMPLE_RATE * 4, dtype=np.float32), "S1"),
            postprocess.Utterance(9.0, np.zeros(config.SAMPLE_RATE * 6, dtype=np.float32), "S2")]
    store = FakeStore()
    recognised = postprocess._remember_voices(store, 1, said, FakeDiarizer())

    assert recognised == {"S1": "廖仁成"}, recognised
    assert sorted(store.saved) == ["S1", "S2"], "both voices are still learned, named or not"


def test_a_voiceprint_averages_only_a_speakers_longest_few() -> None:
    """A two-hour meeting has hundreds of utterances per speaker; a centroid needs a handful."""
    class Counter:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, samples: np.ndarray) -> np.ndarray:
            self.calls += 1
            return np.ones(4, dtype=np.float32)

        def _recognise(self, emb: np.ndarray) -> str:
            return ""

    class Sink:
        def save_voiceprint(self, session_id: int, code: str, centroid: bytes) -> None:
            pass

    said = [postprocess.Utterance(float(i), np.zeros(config.SAMPLE_RATE * 2, dtype=np.float32), "S1")
            for i in range(40)]
    counter = Counter()
    postprocess._remember_voices(Sink(), 1, said, counter)

    assert counter.calls == postprocess.VOICEPRINT_SAMPLES, counter.calls


def _wav(path, seconds: float = 2.0, value: float = 0.0):
    import soundfile as sf
    sf.write(str(path), np.full(int(seconds * config.SAMPLE_RATE), value, dtype=np.float32),
             config.SAMPLE_RATE)
    return path


def test_turns_are_cached_against_the_audio_not_its_timestamp(tmp_path=None) -> None:
    """A stale hit here does not fail loudly — it puts one meeting's speakers on another
    meeting's audio, and re-running does not help because the miss is deterministic too.

    So the key is the recording's content. Two files of identical size and mtime holding
    different audio must not share an answer.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        a, b = _wav(root / "a.wav", value=0.0), _wav(root / "b.wav", value=0.5)
        # Same size, and made to share a timestamp: only the samples differ.
        os.utime(b, ns=(a.stat().st_mtime_ns, a.stat().st_mtime_ns))
        assert a.stat().st_size == b.stat().st_size

        key_a = diarize._turns_key(a, 0.65)
        assert diarize._turns_key(a, 0.65) == key_a, "the key must be stable for one file"
        assert diarize._turns_key(b, 0.65) != key_a, "different audio must not share a key"
        assert diarize._turns_key(a, 0.70) != key_a, "the threshold is part of the answer"


def test_a_cached_answer_is_only_used_when_it_still_applies() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        wav = _wav(Path(tmp) / "m.wav", seconds=10.0)
        key = diarize._turns_key(wav, 0.65)
        found = [diarize.Turn(0.0, 4.0, 0), diarize.Turn(4.0, 9.0, 1)]
        diarize._write_turns(wav, key, found, 10.0)

        assert diarize._read_turns(wav, key) == found
        assert diarize._read_turns(wav, "some-other-key") is None

        # A turn past the end of the audio means whoever wrote it got its offsets wrong.
        diarize._write_turns(wav, key, [diarize.Turn(0.0, 99.0, 0)], 10.0)
        assert diarize._read_turns(wav, key) is None

        # Truncation is a miss, not an exception: the answer is always recomputable.
        diarize._cache_path(wav).write_text('{"key": "x", "turns": [[0', encoding="utf-8")
        assert diarize._read_turns(wav, key) is None
        diarize._cache_path(wav).unlink()
        assert diarize._read_turns(wav, key) is None


def test_chunks_tile_the_recording_exactly() -> None:
    """Cores must cover every second once. A gap is speech nobody segments; an overlap is one
    turn claimed by two chunks and decoded twice into the transcript."""
    for duration in (100.0, 600.0, 8343.0):
        spans = diarize._chunk_spans(duration)
        cores = [(c, d) for _, _, c, d in spans]
        assert cores[0][0] == 0.0 and cores[-1][1] == duration, (duration, cores[0], cores[-1])
        for (_, end), (start, _) in zip(cores, cores[1:]):
            assert start == end, f"cores must meet, got {end} then {start}"
        # The padded read never leaves the recording.
        for read_from, read_to, core_from, core_to in spans:
            assert 0.0 <= read_from <= core_from < core_to <= read_to <= duration


def test_an_overlap_smaller_than_the_models_window_is_refused() -> None:
    """The segmentation model decides from about ten seconds. Cutting with less padding than that
    means the audio around every cut is judged with one side missing — silently worse."""
    diarize._chunk_spans(600.0, overlap=10.0)
    for bad in (0.0, 5.0, 9.9):
        try:
            diarize._chunk_spans(600.0, overlap=bad)
        except ValueError:
            continue
        raise AssertionError(f"overlap {bad} was accepted")


def test_a_speaker_talking_across_a_cut_comes_back_as_one_turn() -> None:
    """Both chunks decode their own side of the cut, so the same person arrives as two turns that
    meet in the middle. After global clustering they carry one label and there is no real gap."""
    split = [diarize.Turn(10.0, 180.0, 3), diarize.Turn(180.0, 240.0, 3)]
    assert diarize._heal(split) == [diarize.Turn(10.0, 240.0, 3)]

    # A real pause is still a boundary, and so is a different speaker.
    kept = [diarize.Turn(0.0, 10.0, 1), diarize.Turn(11.0, 20.0, 1), diarize.Turn(20.0, 30.0, 2)]
    assert diarize._heal(kept) == kept


def test_healing_does_not_shorten_a_turn_it_absorbs() -> None:
    """An overlapping pair must end at the later end, or the tail of the sentence is dropped."""
    overlapping = [diarize.Turn(0.0, 20.0, 5), diarize.Turn(19.8, 19.9, 5)]
    assert diarize._heal(overlapping) == [diarize.Turn(0.0, 20.0, 5)]

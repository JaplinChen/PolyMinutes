import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { useSearchParams } from 'react-router-dom';
import { Download, FileText, Link as LinkIcon, RefreshCw, Search, Trash2, Upload, Volume2 } from 'lucide-react';
import { PageHeader } from '../components/PageHeader';
import { PageSkeleton } from '../components/PageSkeleton';
import { TranscriptRow } from '../components/sessions/TranscriptRow';
import { useToast } from '../components/Toast';
import { useDocumentTitle } from '../hooks/useDocumentTitle';
import { appApi, type MeetingSummary, type RefineJob, type RefineStage, type RefineState, type SessionSummary, type SpeakerSuggestion, type TranscriptLine } from '../services/app.api';
import { API_BASE_URL, NO_SUCH_ENDPOINT } from '../services/http';
import { editingLocked } from '../services/sessionSummary';
import './Sessions.css';
import './Sessions.refine.css';
import './Sessions.summary.css';

// How often to re-check a session that is still being refined. The pass takes minutes, so this is
// about noticing it finished rather than tracking progress, and it stops the moment it has.
const REFINE_POLL_MS = 5000;

// 'unsure' means the voiceprints tied and wording couldn't break it, so both candidates are offered
// side by side rather than picking one arbitrarily — silence here read as "the hint disappeared".
function suggestCandidates(s: SpeakerSuggestion): { name: string; similarity: number }[] {
  const head = { name: s.name, similarity: s.similarity };
  return s.basis === 'unsure' && s.alternative ? [head, s.alternative] : [head];
}

// Only ping the OS for a pass that ran long enough that the user has likely walked away — a quick
// meeting refine that finishes while they glanced at another tab does not deserve a notification.
// Reprocessing 9.5 hours of interviews takes an hour and a half; that is the case this is for.
const LONG_FLOW_MS = 3 * 60 * 1000;

// One meeting is an import control, up to 35 speaker fields and ~950 transcript rows. Stacked they
// are one column metres long, where naming a speaker means scrolling past the transcript to find
// the field and scrolling back to see whether it took. Each is its own view; the session picker
// stays outside them because both of the other two are about whichever session it points at.
const TABS = ['import', 'speakers', 'transcript', 'summary'] as const;
type Tab = (typeof TABS)[number];

export function Sessions() {
  const { t, i18n } = useTranslation();
  useDocumentTitle(t('sessions.title'));
  const toast = useToast();

  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selected, setSelected] = useState<number | null>(null);
  const [lines, setLines] = useState<TranscriptLine[]>([]);
  const [names, setNames] = useState<Record<string, string>>({});
  // Codes ticked on the speaker page to fold into one person — the diariser splits a drifting voice.
  const [mergeSel, setMergeSel] = useState<Set<string>>(new Set());
  // Who each unnamed code sounds most like — the hint the naming screen never had.
  const [suggestions, setSuggestions] = useState<Record<string, SpeakerSuggestion>>({});
  const [loading, setLoading] = useState(true);
  const [editing, setEditing] = useState<{ id: number; text: string } | null>(null);
  const [importing, setImporting] = useState(false);
  const [importUrl, setImportUrl] = useState('');
  const [rerunning, setRerunning] = useState<number | null>(null);
  const [tab, setTab] = useState<Tab>('transcript');
  const [playing, setPlaying] = useState<number | null>(null);
  const [query, setQuery] = useState('');
  const [busy, setBusy] = useState(false);
  const [summary, setSummary] = useState<MeetingSummary | null>(null);
  // The per-session probe: stage progress while the pass runs, skipped count once it has finished.
  // Kept apart from the sessions list, which carries state but not the counters.
  const [job, setJob] = useState<RefineJob | null>(null);
  const [sumLang, setSumLang] = useState<string | null>(null);
  const [summarizing, setSummarizing] = useState(false);
  const [reference, setReference] = useState('');
  const [savingRef, setSavingRef] = useState(false);
  const tablistRef = useRef<HTMLDivElement>(null);
  const player = useRef<HTMLAudioElement | null>(null);
  // Read inside the callback so it does not have to depend on `playing` — a callback that changes
  // identity on every play invalidates all 943 memoised rows, which is what this is avoiding.
  const playingRef = useRef<number | null>(null);

  const fail = (err: unknown) => toast.error(err instanceof Error ? err.message : String(err));

  const current = sessions.find(s => s.id === selected);
  const refine: RefineState = current?.refine.state ?? 'idle';
  // The server puts the reason on every /sessions response and the page was dropping it: a failed
  // pass said "精修失敗" and nothing else, while the thing that broke was named in the payload.
  const refineError = current?.refine.error ?? '';
  // Playing a line, hearing a speaker and re-deriving the transcript all read the recording. When
  // it is gone they all fail the same way, so the page says so once instead of per click.
  const hasRecording = current?.hasRecording ?? true;
  // The pass calls replace_lines, which drops every line and writes new ones with new ids. An edit
  // saved during that window is silently discarded while the screen shows it saved, so editing is
  // closed rather than left to look like it worked. Scoped to the mutating stages: a summarize-only
  // regeneration reports "refining" too but never touches a line, so it must not lock the transcript.
  const locked = current ? editingLocked(current.refine) : false;

  const loadLines = useCallback((id: number) => {
    appApi
      .sessionLines(id)
      .then(r => {
        setLines(r.lines);
        setNames(r.speakers);
        setMergeSel(new Set());
      })
      .catch(fail);
  }, []);

  const [params, setParams] = useSearchParams();
  const wantedLine = useRef<number | null>(null);
  const sessionsLoaded = useRef(false);

  useEffect(() => {
    appApi
      .sessions()
      .then(list => {
        setSessions(list);
        sessionsLoaded.current = true;
        // Default selection only when a citation is not steering it; the effect below handles that.
        if (list.length && !params.get('session')) setSelected(list[0].id);
      })
      .catch(fail)
      .finally(() => setLoading(false));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ?session= / ?line= arrive when a Q&A citation is followed here — open that meeting and jump to
  // that line. Reactive to the params (a citation from /ask remounts nothing on some navigations),
  // but self-clearing: once consumed the query is stripped, so a later manual pick is never pulled
  // back to a stale citation, and a refresh does not re-fire it.
  useEffect(() => {
    const wantSession = Number(params.get('session')) || null;
    if (!wantSession) return;
    const consume = () => {
      if (sessions.some(s => s.id === wantSession)) setSelected(wantSession);
      setTab('transcript');
      wantedLine.current = Number(params.get('line')) || null;
      setParams({}, { replace: true });
    };
    if (sessionsLoaded.current) consume();
    // If the list has not arrived yet, the sessions effect will; re-running on `sessions` catches it.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params, sessions]);

  const loadSummary = useCallback((id: number) => {
    appApi.sessionSummary(id).then(setSummary).catch(() => setSummary(null));
  }, []);

  // Silent on failure: an older backend without the progress fields still answers, and one without
  // the route at all should degrade to the bare state chip rather than a toast per poll.
  const loadJob = useCallback((id: number) => {
    appApi.refineJob(id).then(setJob).catch(() => {});
  }, []);

  useEffect(() => {
    setJob(null);
    if (selected !== null) loadJob(selected);
  }, [selected, loadJob]);

  useEffect(() => {
    if (selected !== null) loadLines(selected);
  }, [selected, loadLines]);

  // Refetched whenever the transcript changes: a merge or a reprocess redraws the codes, and a
  // suggestion for a code that no longer exists is worse than none.
  useEffect(() => {
    if (selected === null) { setSuggestions({}); return; }
    appApi.speakerSuggestions(selected).then(setSuggestions).catch(() => setSuggestions({}));
  }, [selected, lines]);

  // Once the lines for a cited session are on screen, scroll the cited one into view and flash it,
  // so following a citation lands on the exact utterance rather than the top of a 943-line list.
  // The effect runs after React has committed the rows to the DOM, so the row exists now; a short
  // timeout gives layout a beat to settle before scrolling, without depending on requestAnimationFrame
  // (which does not fire when the tab is backgrounded, and would strand the highlight there).
  useEffect(() => {
    const lineId = wantedLine.current;
    if (lineId === null || tab !== 'transcript' || !lines.some(l => l.id === lineId)) return;
    wantedLine.current = null;
    const timer = window.setTimeout(() => {
      const row = document.querySelector<HTMLElement>(`[data-line-id="${lineId}"]`);
      if (!row) return;
      row.scrollIntoView({ block: 'center' });
      row.classList.add('sess-line-cited');
      window.setTimeout(() => row.classList.remove('sess-line-cited'), 2400);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [lines, tab]);

  useEffect(() => {
    setSummary(null);
    setSumLang(null);
    if (selected !== null) loadSummary(selected);
  }, [selected, loadSummary]);

  // Poll while anything is running that this page must notice finishing: the refine pass (state on
  // the session) or a summarize-only regeneration (state on the summary). Watching refine alone
  // meant a summary-only job — which never turns refine to "refining" — ran to completion with the
  // page never re-fetching, so the finished summary simply never appeared until a manual reload.
  const summaryGenerating = summary?.state === 'generating';
  const jobRunning = refine === 'refining' || summaryGenerating;
  const wasRunning = useRef(false);
  const runStartedAt = useRef<number | null>(null);
  useEffect(() => {
    if (!jobRunning) {
      if (wasRunning.current && selected !== null) {
        wasRunning.current = false;
        toast.success(t('sessions.refineDone'));
        // One OS notification for a long pass the user walked away from — the single out-of-app
        // channel, deliberately not Teams/Slack/email. Gated on both a long run and a hidden tab so
        // a quick refine never pings, and silent if they never granted permission.
        const ranLong = runStartedAt.current !== null && Date.now() - runStartedAt.current > LONG_FLOW_MS;
        runStartedAt.current = null;
        if (ranLong && document.hidden && 'Notification' in window && Notification.permission === 'granted') {
          new Notification(t('sessions.notifyTitle'), { body: t('sessions.refineDone') });
        }
        loadLines(selected);
        // Both a refine pass and a summarize job end by writing the summary; either way the card is
        // now out of date, and the summary itself must be re-fetched, not only the session list.
        loadSummary(selected);
        // The finished job carries the skipped count the warning banner reads.
        loadJob(selected);
      }
      return;
    }
    if (!wasRunning.current) {
      // The pass just started. Stamp the start for the duration gate, and ask for notification
      // permission now — by the time an hour-long reprocess finishes the answer is long settled.
      runStartedAt.current = Date.now();
      if ('Notification' in window && Notification.permission === 'default') {
        Notification.requestPermission().catch(() => {});
      }
    }
    wasRunning.current = true;
    const timer = window.setInterval(() => {
      appApi.sessions().then(setSessions).catch(() => {});
      // The summary job's completion shows on the summary, not the session, so poll it too — this
      // is what lets a summarize-only run be noticed at all.
      if (selected !== null) {
        loadSummary(selected);
        // Same timer, not a second poll: stage and done/total ride along with each tick.
        loadJob(selected);
      }
    }, REFINE_POLL_MS);
    return () => window.clearInterval(timer);
  }, [jobRunning, selected, loadLines, loadSummary, loadJob, toast, t]);

  // Regenerates the summary alone — no ASR, no GPU. The job registry the refine poll watches
  // dedups it, so refreshing the sessions list is what starts the poll that notices it finish.
  const generateSummary = async () => {
    if (selected === null || summarizing) return;
    setSummarizing(true);
    try {
      setSummary(await appApi.summarizeSession(selected));
      setSessions(await appApi.sessions());
    } catch (err) {
      fail(err);
    } finally {
      setSummarizing(false);
    }
  };

  // Keep the reference box in step with the picked session. `current?.reference` is a plain string,
  // so this fires only when the session or its stored notes actually change, not on every render.
  useEffect(() => {
    setReference(current?.reference ?? '');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, current?.reference]);

  // Pre-meeting notes are folded into the summary prompt when it regenerates — not applied to the
  // current summary, which the user regenerates if they want them included.
  const saveReference = async () => {
    if (selected === null || savingRef) return;
    setSavingRef(true);
    try {
      await appApi.setReference(selected, reference);
      setSessions(prev => prev.map(s => (s.id === selected ? { ...s, reference } : s)));
      toast.success(t('sessions.referenceSaved'));
    } catch (err) {
      fail(err);
    } finally {
      setSavingRef(false);
    }
  };

  // Speakers are identified by voice, not by name — the app never sees the participant list.
  // Naming them once here is what turns S1/S2 into a readable transcript.
  const saveName = async (code: string, name: string) => {
    if (selected === null) return;
    try {
      setNames(await appApi.setSpeakerNames(selected, { [code]: name }));
    } catch (err) {
      fail(err);
    }
  };

  // Correcting a line is the only ground truth the system gets: someone who was in the room
  // saying what was actually said. The backend learns the pair and applies it from then on.
  //
  // A textarea rather than contentEditable: this transcript is mostly Chinese, and an IME
  // composing inside a contentEditable fires input and blur events mid-character.
  const saveLine = useCallback(async (lineId: number, source: string, previous: string) => {
    setEditing(null);
    if (selected === null || source.trim() === previous || !source.trim()) return;
    try {
      const r = await appApi.setLineSource(selected, lineId, source.trim());
      setLines(r.lines);
    } catch (err) {
      fail(err);
    }
    // `fail` is recreated each render by design (it closes over the toast); depending on it would
    // defeat the memoisation these callbacks exist to enable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  // Move one line to another speaker. The shared-mic clustering collapses a room into one voice
  // more often than not (see the README's known limits), and language is chosen per speaker — so a
  // wrong attribution also decoded the line in the wrong language. This is the human putting the
  // split back by hand. `names` follows because a fresh S-code has no name yet and must show as one.
  const reassignLine = useCallback(async (lineId: number, speaker: string) => {
    if (selected === null) return;
    try {
      const r = await appApi.setLineSpeaker(selected, lineId, speaker);
      setLines(r.lines);
      setNames(r.speakers);
    } catch (err) {
      fail(err);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  // A recording made elsewhere teaches the same things a live capture does, once it is a session:
  // names attach to voices, corrections attach to lines.
  const importRecording = async (file: File) => {
    setImporting(true);
    try {
      const added = await appApi.importRecording(file);
      setSessions(await appApi.sessions());
      setSelected(added.id);
    } catch (err) {
      fail(err);
    } finally {
      setImporting(false);
    }
  };

  // Same landing as a file: the session exists at once and the refine chip tracks the download
  // and transcription, so this only has to hand over the URL and select what came back.
  const importFromUrl = async () => {
    const url = importUrl.trim();
    if (!url || importing) return;
    setImporting(true);
    try {
      const added = await appApi.importUrl(url);
      setImportUrl('');
      setSessions(await appApi.sessions());
      setSelected(added.id);
    } catch (err) {
      fail(err);
    } finally {
      setImporting(false);
    }
  };

  // Re-deriving the transcript throws away every hand correction on it, and there is no undo, so
  // it asks first. window.confirm rather than a dialog component: this is the only place in the app
  // that needs one, and a bespoke modal would be more code than the thing it guards.
  const reprocessSession = async () => {
    if (selected === null || !hasRecording || locked) return;
    if (!window.confirm(t('sessions.reprocessConfirm'))) return;
    setBusy(true);
    try {
      await appApi.reprocess(selected);
      // Refetch rather than patch state: the refine chip and the poll both read from this list.
      setSessions(await appApi.sessions());
      toast.success(t('sessions.reprocessQueued'));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  // Same confirm-first shape as reprocess: no undo, and the recording goes with the rows.
  const deleteSession = async () => {
    if (selected === null || locked) return;
    if (!window.confirm(t('sessions.deleteConfirm'))) return;
    setBusy(true);
    try {
      await appApi.deleteSession(selected);
      const list = await appApi.sessions();
      setSessions(list);
      setSelected(list.length ? list[0].id : null);
      toast.success(t('sessions.deleteDone'));
    } catch (err) {
      fail(err);
    } finally {
      setBusy(false);
    }
  };

  // Correcting a line is a judgement about whether the text matches what was said, so the audio has
  // to be reachable from the line. One player for the whole transcript rather than one per row:
  // clicking a second line replaces what is playing, which is also the behaviour you want.
  // No hasRecording check here: the button carries `playable`, so it is disabled when there is
  // nothing to play. Re-checking would also make this callback depend on a value that changes with
  // the session, which is exactly what the memoised rows must not see.
  const playLine = useCallback((lineId: number) => {
    if (selected === null) return;
    const setNow = (v: number | null) => { playingRef.current = v; setPlaying(v); };
    const audio = (player.current ??= new Audio());
    if (playingRef.current === lineId) {
      audio.pause();
      setNow(null);
      return;
    }
    const url = `${API_BASE_URL}/sessions/${selected}/lines/${lineId}/clip`;
    audio.src = url;
    audio.onended = () => setNow(null);
    // <audio> reports that it failed, never why, and the two causes want different answers: this
    // line genuinely has no audio, or the request never reached the endpoint — a backend still
    // running the build from before the route existed answers 404 for every line in the meeting.
    // One request, only on failure, to say which happened instead of guessing.
    audio.onerror = () => {
      setNow(null);
      fetch(url)
        .then(async r => {
          const detail = await r.json().then(b => b?.detail).catch(() => '');
          // The server marks its own "this build has no such endpoint" 404 so it is not mistaken
          // for a line that has no audio — the two are the same status and opposite problems.
          if (detail === NO_SUCH_ENDPOINT) return toast.error(t('sessions.playStaleServer'));
          if (r.status === 404) return toast.error(t('sessions.playFailed'));
          toast.error(t('sessions.playUnavailable', { status: r.status }));
        })
        .catch(() => toast.error(t('sessions.playUnreachable')));
    };
    void audio.play().catch(() => {});
    setNow(lineId);
  }, [selected, t, toast]);

  // Switching session or tab leaves a clip playing over a transcript that is no longer on screen.
  useEffect(() => {
    player.current?.pause();
    setPlaying(null);
  }, [selected, tab]);

  // A query carried into another meeting shows it as empty, which reads as "this session has no
  // lines" rather than "nothing here matches what you typed about the last one".
  useEffect(() => setQuery(''), [selected]);

  // Re-running is per line rather than per transcript: a failure is usually one utterance the
  // decoder gave up on, and re-running the whole meeting to recover it is not a proportionate ask.
  const rerunLine = useCallback(async (lineId: number) => {
    if (selected === null) return;
    setRerunning(lineId);
    try {
      const r = await appApi.rerunLine(selected, lineId);
      setLines(r.lines);
      setNames(r.speakers);
      if (r.status !== 'ok') toast.error(t(`sessions.${r.status === 'asr_failed' ? 'lineFailedAsr' : 'lineFailedTranslate'}`));
    } catch (err) {
      fail(err);
    } finally {
      setRerunning(null);
    }
    // `fail` closes over the toast and is recreated each render; depending on it would defeat the
    // memoisation these stable callbacks exist to enable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, t, toast]);

  // Same gate as a rerun, but no audio and no GPU: this is for a line whose words are right and
  // whose translation is not — a hand-corrected line, or one the model rendered badly.
  const retranslateLine = useCallback(async (lineId: number) => {
    if (selected === null) return;
    setRerunning(lineId);
    try {
      const r = await appApi.retranslateLine(selected, lineId);
      setLines(r.lines);
      setNames(r.speakers);
      if (r.status !== 'ok') toast.error(t('sessions.lineFailedTranslate'));
    } catch (err) {
      fail(err);
    } finally {
      setRerunning(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, t, toast]);

  // 943 lines is not a list you scroll to check a word in. Matches the text as spoken, every
  // translation of it, and the speaker — searching for a name is how you find what someone said
  // once the voices have been named.
  const shown = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return lines;
    return lines.filter(l =>
      l.source.toLowerCase().includes(q) ||
      (names[l.speaker] || l.speaker).toLowerCase().includes(q) ||
      Object.values(l.translations).some(v => v.toLowerCase().includes(q)));
  }, [lines, names, query]);

  const codes = [...new Set(lines.map(l => l.speaker))];
  // A code with only a few lines, none of them inside its own run of consecutive lines, is a
  // fragment cluster: stutter pieces and misjudged embeddings wedged into other people's speech.
  // Its sample has no clean line to play, so the naming row warns instead of misleading.
  const fragCodes = useMemo(() => {
    const sorted = [...lines].sort((a, b) => a.start - b.start);
    const stats = new Map<string, { count: number; midRun: boolean }>();
    sorted.forEach((l, i) => {
      const s = stats.get(l.speaker) ?? { count: 0, midRun: false };
      s.count += 1;
      if (sorted[i - 1]?.speaker === l.speaker && sorted[i + 1]?.speaker === l.speaker) s.midRun = true;
      stats.set(l.speaker, s);
    });
    return new Set([...stats].filter(([, s]) => s.count <= 3 && !s.midRun).map(([c]) => c));
  }, [lines]);
  // Stable identities so a play/edit click doesn't rebuild 943 memoised rows. Options are every
  // speaker the meeting has; the new code is the next free S-number, for splitting the collapse.
  const speakerOptions = useMemo(
    () => [...new Set(lines.map(l => l.speaker))].sort().map(c => ({ code: c, label: names[c] || c })),
    [lines, names]);
  const newSpeakerCode = useMemo(() => {
    const nums = lines.map(l => Number(/^S(\d+)$/.exec(l.speaker)?.[1])).filter(n => !Number.isNaN(n));
    return `S${(nums.length ? Math.max(...nums) : 0) + 1}`;
  }, [lines]);
  // Where the ticked codes fold to: a named one if any is named (that is the real identity), else
  // the first as it appears in the transcript. Shown on the button so the choice is not a surprise.
  const mergeTarget = useMemo(() => {
    const sel = codes.filter(c => mergeSel.has(c));
    return sel.find(c => (names[c] ?? '').trim()) ?? sel[0];
  }, [codes, mergeSel, names]);
  const mergeSelected = async () => {
    if (selected === null || !mergeTarget || mergeSel.size < 2) return;
    const from = codes.filter(c => mergeSel.has(c) && c !== mergeTarget);
    try {
      const r = await appApi.mergeSpeakers(selected, mergeTarget, from);
      setLines(r.lines);
      setNames(r.speakers);
      setMergeSel(new Set());
    } catch (err) {
      fail(err);
    }
  };
  const langs = useMemo(() => [...new Set(lines.flatMap(l => Object.keys(l.translations)))], [lines]);
  const failed = lines.filter(l => l.status !== 'ok');
  // 'generating' is visible two ways — the summary endpoint says so, or the session's refine job
  // is still running (the summary is its last stage). Either alone can be the first one seen.
  const sumGenerating = summary?.state === 'generating' || refine === 'refining';
  // "Summarizing" when that is what is running: a summarize-only job (summary.state generating) is
  // always the summary stage, and a refine pass reports its stage. Only a refine pass not yet at
  // its summary stage says "refining".
  const sumStageLabel = summaryGenerating || current?.refine.stage === 'summarize'
    ? t('sessions.summarizing')
    : t('sessions.refining');
  const sumLangs = summary?.summary ? Object.keys(summary.summary) : [];
  // Prefix-match against the UI language (zh-HK → zh), falling back to whatever exists.
  const uiBase = i18n.language.split('-')[0].toLowerCase();
  const activeSumLang =
    sumLang && sumLangs.includes(sumLang)
      ? sumLang
      : sumLangs.find(l => l.split('-')[0].toLowerCase() === uiBase) ?? sumLangs[0];
  const sumContent = activeSumLang ? summary?.summary?.[activeSumLang] : undefined;
  // Stage from the probe when it has answered, else the session list's copy — the list refreshes on
  // the same tick, so a backend without the probe still names the stage where it can.
  const jobStage: RefineStage | undefined = job?.stage ?? current?.refine.stage;
  const jobDone = job?.done ?? 0;
  const jobTotal = job?.total ?? 0;
  const jobSkipped = job?.skipped ?? 0;
  const stageLabel: Record<RefineStage, string> = {
    rewrite: t('sessions.stageRewrite'),
    segment: t('sessions.stageSegment'),
    refine: t('sessions.stageRefine'),
    summarize: t('sessions.stageSummarize'),
  };
  // segment/refine walk the transcript in order, so `done` is a watermark: rows past it have not
  // been touched yet. Ids rather than indexes because the visible list may be search-filtered.
  const pendingIds =
    refine === 'refining' && jobTotal > 0 && (jobStage === 'segment' || jobStage === 'refine')
      ? new Set(lines.slice(jobDone).map(l => l.id))
      : null;
  const refineLabel: Partial<Record<RefineState, string>> = {
    refining: t('sessions.refining'),
    refined: t('sessions.refined'),
    failed: t('sessions.refineFailed'),
    cancelled: t('sessions.refineCancelled'),
  };

  if (loading) {
    return <PageSkeleton rows={4} />;
  }

  // With nothing imported there is no session for the other two to be about, so the choice is not
  // offered rather than offered and empty.
  const hasSessions = sessions.length > 0;
  const active: Tab = hasSessions ? tab : 'import';

  // Left/Right move between tabs, which is what a tablist is expected to do once its buttons claim
  // role="tab" — without it the role announces an interaction the keyboard cannot perform.
  const onTabKeys = (event: React.KeyboardEvent) => {
    const step = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
    if (!step) return;
    event.preventDefault();
    const enabled = TABS.filter(id => id === 'import' || hasSessions);
    const next = enabled[(enabled.indexOf(active) + step + enabled.length) % enabled.length];
    setTab(next);
    tablistRef.current?.querySelector<HTMLButtonElement>(`#sess-tab-${next}`)?.focus();
  };

  return (
    <div className="etable-page sess-page">
      <PageHeader title={t('sessions.title')} subtitle={t('sessions.subtitle')} />

      {hasSessions && (
        <section className="etable-panel">
          {/* The panel is a stretch column, so unwrapped the button would fill its width. */}
          <div className="sess-picker-row">
          <select className="sess-select" aria-label={t('sessions.pick')} value={selected ?? ''} onChange={e => setSelected(Number(e.target.value))}>
            {sessions.map(s => (
              <option key={s.id} value={s.id}>
                {s.started} — {t('sessions.lineCount', { count: s.lines })}
                {s.refine.state === 'refining' ? ` · ${t('sessions.refining')}` : ''}
              </option>
            ))}
          </select>
          <button
            type="button"
            className="sess-delete"
            disabled={busy || locked || selected === null}
            title={locked ? t('sessions.refiningHint') : t('sessions.deleteHint')}
            onClick={deleteSession}
          >
            <Trash2 size={13} />
            {t('sessions.delete')}
          </button>
          </div>
        </section>
      )}

      <div className="sess-tabs" role="tablist" aria-label={t('sessions.title')} ref={tablistRef} onKeyDown={onTabKeys}>
        {TABS.map(id => (
          <button
            key={id}
            id={`sess-tab-${id}`}
            role="tab"
            type="button"
            aria-selected={active === id}
            aria-controls={`sess-panel-${id}`}
            // Only the active tab is in the tab order; arrows move within the list. Five stops for
            // three tabs is what makes a tablist tedious to tab past.
            tabIndex={active === id ? 0 : -1}
            disabled={id !== 'import' && !hasSessions}
            className={`sess-tab ${active === id ? 'active' : ''}`}
            onClick={() => setTab(id)}
          >
            {t(`sessions.${id}`)}
            {id === 'transcript' && hasSessions && <span className="etable-count">{lines.length}</span>}
          </button>
        ))}
      </div>

      {active === 'import' && (
        <section className="etable-panel" role="tabpanel" id="sess-panel-import" aria-labelledby="sess-tab-import">
          <p className="sess-hint">{t('sessions.importHint')}</p>
          <label className="sess-import">
            <Upload size={16} />
            <span>{importing ? t('sessions.importing') : t('sessions.importPick')}</span>
            <input
              type="file"
              accept="video/*,audio/*"
              disabled={importing}
              onChange={e => {
                const file = e.target.files?.[0];
                e.target.value = '';
                if (file) importRecording(file);
              }}
            />
          </label>
          <div className="sess-import-url">
            <input
              type="text"
              value={importUrl}
              placeholder={t('sessions.importUrlPlaceholder')}
              disabled={importing}
              onChange={e => setImportUrl(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') importFromUrl(); }}
            />
            <button type="button" disabled={importing || !importUrl.trim()} onClick={importFromUrl}>
              <LinkIcon size={16} />
              {t('sessions.importUrlGo')}
            </button>
          </div>
          {!hasSessions && (
            <div className="sess-empty">
              <FileText size={32} strokeWidth={1} />
              <span>{t('sessions.empty')}</span>
            </div>
          )}
        </section>
      )}

      {active === 'speakers' && (
        <section className="etable-panel" role="tabpanel" id="sess-panel-speakers" aria-labelledby="sess-tab-speakers">
          <p className="sess-hint">{t('sessions.speakersHint')}</p>
          <p className="sess-hint">{t('sessions.mergeHint')}</p>
          {!hasRecording && <p className="sess-no-audio">{t('sessions.noRecording')}</p>}
          {mergeSel.size >= 2 && mergeTarget && (
            <div className="sess-merge-bar">
              <span>{t('sessions.mergeSelected', { count: mergeSel.size })}</span>
              <button type="button" className="sess-merge-go" onClick={mergeSelected}>
                {t('sessions.mergeInto', { target: names[mergeTarget] || mergeTarget })}
              </button>
            </div>
          )}
          <div className="sess-names">
            {codes.map(code => (
              <div key={code} className="sess-name">
                <input
                  type="checkbox"
                  className="sess-merge-check"
                  checked={mergeSel.has(code)}
                  aria-label={t('sessions.mergePick', { code })}
                  onChange={() => setMergeSel(prev => {
                    const next = new Set(prev);
                    if (next.has(code)) next.delete(code); else next.add(code);
                    return next;
                  })}
                />
                <label className="sess-name-field">
                  <span>{code}</span>
                  {fragCodes.has(code) && (
                    <span className="sess-frag" title={t('sessions.fragmentHint')}>
                      {t('sessions.fragmentBadge')}
                    </span>
                  )}
                  <input
                    value={names[code] ?? ''}
                    placeholder={t('sessions.namePlaceholder')}
                    onChange={e => setNames(prev => ({ ...prev, [code]: e.target.value }))}
                    onBlur={e => saveName(code, e.target.value)}
                  />
                  {!(names[code] ?? '').trim() && suggestions[code] && (
                    <span className="sess-suggest">
                      {suggestions[code].basis === 'unsure' && (
                        <span className="sess-suggest-basis">{t('sessions.suggestUnsure')}</span>
                      )}
                      {suggestCandidates(suggestions[code]).map(cand => (
                        <span key={cand.name} className="sess-suggest-pick">
                          <button
                            type="button"
                            className="sess-suggest-apply"
                            title={t(
                              suggestions[code].basis === 'unsure'
                                ? 'sessions.suggestApplyHintUnsure'
                                : suggestions[code].basis === 'wording'
                                  ? 'sessions.suggestApplyHintWording'
                                  : 'sessions.suggestApplyHint',
                            )}
                            onClick={() => saveName(code, cand.name)}
                          >
                            {t('sessions.suggestLabel', {
                              name: cand.name,
                              pct: Math.round(cand.similarity * 100),
                            })}
                            {suggestions[code].basis === 'wording' && (
                              <span className="sess-suggest-basis">{t('sessions.suggestByWording')}</span>
                            )}
                          </button>
                          <button
                            type="button"
                            className="sess-suggest-play"
                            title={t('sessions.suggestPlayHint', { name: cand.name })}
                            onClick={() => {
                              new Audio(`${API_BASE_URL}/speakers/known/${encodeURIComponent(cand.name)}/clip`).play().catch(() => {});
                            }}
                          >
                            <Volume2 size={14} />
                          </button>
                        </span>
                      ))}
                    </span>
                  )}
                </label>
                {/* preload="none" because a meeting can have 35 of these and none of them is
                    wanted until someone clicks. Omitted entirely when the recording is gone —
                    35 players that can only fail are worse than none. */}
                {hasRecording && (
                  <audio
                    className="sess-clip"
                    controls
                    preload="none"
                    aria-label={t('sessions.clipLabel', { code })}
                    src={`${API_BASE_URL}/sessions/${selected}/speakers/${encodeURIComponent(code)}/clip`}
                  />
                )}
              </div>
            ))}
          </div>
        </section>
      )}

      {active === 'transcript' && (
        <section className="etable-panel" role="tabpanel" id="sess-panel-transcript" aria-labelledby="sess-tab-transcript">
          <h3 className="etable-panel-title">
            {t('sessions.transcript')}
            <span className="etable-count">{query.trim() ? `${shown.length} / ${lines.length}` : lines.length}</span>
            {refine !== 'idle' && (
              <span className={`sess-refine sess-refine-${refine}`} title={refineError || undefined}>
                {refineLabel[refine]}
              </span>
            )}
            {/* A plain link, not a fetch-and-blob: the browser already knows how to save a
                response, and `download` names the file after the meeting rather than "markdown". */}
            <a
              className="sess-export"
              href={`${API_BASE_URL}/sessions/${selected}/markdown`}
              download={`${t('sessions.title')}-${(current?.started ?? '').slice(0, 10)}.md`}
              title={t('sessions.exportHint')}
            >
              <Download size={13} />
              {t('sessions.export')}
            </a>
            <a
              className="sess-export"
              href={`${API_BASE_URL}/sessions/${selected}/docx`}
              download={`${t('sessions.title')}-${(current?.started ?? '').slice(0, 10)}.docx`}
              title={t('sessions.exportDocxHint')}
            >
              <FileText size={13} />
              {t('sessions.exportDocx')}
            </a>
            <button
              type="button"
              className="sess-reprocess"
              disabled={busy || locked || !hasRecording}
              title={hasRecording ? t('sessions.reprocessHint') : t('sessions.noRecording')}
              onClick={reprocessSession}
            >
              <RefreshCw size={13} />
              {locked ? t('sessions.reprocessing') : t('sessions.reprocess')}
            </button>
          </h3>
          <div className="etable-search">
            <Search className="etable-search-icon" size={15} />
            <input
              type="search"
              value={query}
              placeholder={t('sessions.searchPlaceholder')}
              aria-label={t('sessions.searchPlaceholder')}
              onChange={e => setQuery(e.target.value)}
            />
          </div>
          {locked && <p className="sess-hint">{t('sessions.refiningHint')}</p>}
          {refine === 'refining' && (
            <div className="sess-refine-progress" role="status">
              <span>{jobStage ? stageLabel[jobStage] : t('sessions.refining')}</span>
              {jobTotal > 0 && (
                <span className="sess-refine-count">
                  {t('sessions.refineProgress', { done: jobDone, total: jobTotal })}
                </span>
              )}
            </div>
          )}
          {refine === 'refined' && jobSkipped > 0 && (
            <div className="sess-refine-skipped" role="status">
              <span>{t('sessions.refineSkipped', { count: jobSkipped })}</span>
              {/* The same run as the header's reprocess button — this is a shortcut to it placed
                  where the problem is stated, not a second kind of rerun. */}
              <button
                type="button"
                className="sess-reprocess"
                disabled={busy || locked || !hasRecording}
                title={hasRecording ? t('sessions.reprocessHint') : t('sessions.noRecording')}
                onClick={reprocessSession}
              >
                <RefreshCw size={13} />
                {t('sessions.reprocess')}
              </button>
            </div>
          )}
          {refine === 'failed' && refineError && (
            <p className="sess-refine-error">{t('sessions.refineFailedReason', { reason: refineError })}</p>
          )}
          {!hasRecording && <p className="sess-no-audio">{t('sessions.noRecording')}</p>}
          {failed.length > 0 && (
            // Aggregated as well as marked inline: a two-hour meeting failing 5% is forty-odd
            // marks scattered through the transcript, and nobody finds those by scrolling.
            <p className="sess-failed-summary">{t('sessions.failedCount', { count: failed.length })}</p>
          )}
          <div className="sess-lines">
            {shown.map(line => (
              <TranscriptRow
                key={line.id}
                line={line}
                speakerOptions={speakerOptions}
                newSpeakerCode={newSpeakerCode}
                langs={langs}
                locked={locked}
                pending={pendingIds?.has(line.id) ?? false}
                draftText={editing?.id === line.id ? editing.text : null}
                isRerunning={rerunning === line.id}
                rerunBlocked={rerunning !== null}
                isPlaying={playing === line.id}
                playable={hasRecording}
                onDraft={setEditing}
                onSave={saveLine}
                onRerun={rerunLine}
                onRetranslate={retranslateLine}
                onPlay={playLine}
                onReassign={reassignLine}
              />
            ))}
            {query.trim() && shown.length === 0 && (
              <p className="sess-hint">{t('sessions.searchEmpty', { query: query.trim() })}</p>
            )}
          </div>
        </section>
      )}

      {active === 'summary' && (
        <section className="etable-panel" role="tabpanel" id="sess-panel-summary" aria-labelledby="sess-tab-summary">
          <h3 className="etable-panel-title">
            {t('sessions.summary')}
            {summary?.state === 'partial' && <span className="sess-summary-badge">{t('sessions.summaryPartial')}</span>}
            {summary?.stale && <span className="sess-summary-badge">{t('sessions.summaryStaleShort')}</span>}
            {sumGenerating && <span className="sess-summary-badge">{sumStageLabel}</span>}
          </h3>

          <div className="sess-reference">
            <label className="sess-reference-label" htmlFor="sess-reference-input">
              {t('sessions.reference')}
            </label>
            <p className="sess-hint">{t('sessions.referenceHint')}</p>
            <textarea
              id="sess-reference-input"
              className="sess-reference-input"
              rows={4}
              value={reference}
              placeholder={t('sessions.referencePlaceholder')}
              onChange={e => setReference(e.target.value)}
            />
            <button
              type="button"
              className="sess-summary-regen"
              disabled={savingRef || reference === (current?.reference ?? '')}
              onClick={saveReference}
            >
              {savingRef ? t('sessions.referenceSaving') : t('sessions.referenceSave')}
            </button>
          </div>

          {sumContent ? (
            <div className="sess-summary-body">
              {sumLangs.length > 1 && (
                <div className="sess-summary-langs">
                  {sumLangs.map(l => (
                    <button
                      key={l}
                      type="button"
                      className={`sess-summary-lang ${l === activeSumLang ? 'active' : ''}`}
                      onClick={() => setSumLang(l)}
                    >
                      {t(`lang.${l}`, l)}
                    </button>
                  ))}
                </div>
              )}
              <p className="sess-summary-text">{sumContent.summary}</p>
              {sumContent.decisions.length > 0 && (
                <>
                  <h4 className="sess-summary-heading">{t('sessions.summaryDecisions')}</h4>
                  <ul className="sess-summary-list">
                    {sumContent.decisions.map((d, i) => <li key={i}>{d}</li>)}
                  </ul>
                </>
              )}
              {sumContent.actions.length > 0 && (
                <>
                  <h4 className="sess-summary-heading">{t('sessions.summaryActions')}</h4>
                  <ul className="sess-summary-list">
                    {sumContent.actions.map((a, i) => (
                      <li key={i}>
                        <span className="sess-summary-actor">
                          {a.speaker ? names[a.speaker] || a.speaker : t('sessions.summaryUnassigned')}
                        </span>
                        {a.text}
                      </li>
                    ))}
                  </ul>
                </>
              )}
              {summary?.stale && <p className="sess-summary-stale">{t('sessions.summaryStale')}</p>}
              {summary?.state === 'failed' && <p className="sess-summary-error">{t('sessions.summaryFailed')}</p>}
              <button
                type="button"
                className="sess-summary-regen"
                disabled={summarizing || sumGenerating}
                onClick={generateSummary}
              >
                {summarizing
                  ? t('sessions.summaryRegenerating')
                  : summary?.state === 'failed'
                    ? t('sessions.summaryRetry')
                    : t('sessions.summaryRegenerate')}
              </button>
            </div>
          ) : sumGenerating ? (
            <p className="sess-hint">{sumStageLabel}</p>
          ) : summary?.state === 'no_llm' ? (
            <div className="sess-summary-cta">
              <p className="sess-summary-error">{t('sessions.summaryNoLlm')}</p>
              <button type="button" className="sess-summary-generate" disabled={summarizing} onClick={generateSummary}>
                {summarizing ? t('sessions.summaryGenerating') : t('sessions.summaryRetry')}
              </button>
            </div>
          ) : summary?.state === 'failed' ? (
            <div className="sess-summary-cta">
              <p className="sess-summary-error">{t('sessions.summaryFailed')}</p>
              <button type="button" className="sess-summary-generate" disabled={summarizing} onClick={generateSummary}>
                {summarizing ? t('sessions.summaryGenerating') : t('sessions.summaryRetry')}
              </button>
            </div>
          ) : (
            <div className="sess-summary-cta">
              <p className="sess-hint">{t('sessions.summaryEmpty')}</p>
              <button type="button" className="sess-summary-generate" disabled={summarizing} onClick={generateSummary}>
                {summarizing ? t('sessions.summaryGenerating') : t('sessions.summaryGenerate')}
              </button>
            </div>
          )}
        </section>
      )}
    </div>
  );
}

export default Sessions;

import { request } from './http.ts';
import { withRefine, type RawSessionSummary } from './sessionSummary.ts';

export interface DisplaySettings {
  font_size: number;
  lines: number;
  show_source: 'top' | 'bottom' | 'hidden';
  show_speaker: boolean;
  colour_speakers: boolean;
  theme: 'dark' | 'light';
}

/** A voice the room can name on sight, learned from someone naming a speaker once. */
export interface KnownSpeaker {
  name: string;
  /** How many meetings this voice has been confirmed in. */
  sessions: number;
  /** Forced transcription language for this voice; '' means auto-detect. */
  language: string;
  /** Department this person belongs to; '' when unset. Fed to the summary for stance. */
  department: string;
  /** Playable samples: meetings still on disk plus clips kept from deleted ones. */
  clips: number;
}

/** What the recogniser wrote against what was actually said, learned from an edit. */
export interface LearnedCorrection {
  wrong: string;
  right: string;
}

export interface AppConfig {
  languages: string[];
  inputDevice: string;
  whisperModel: string;
  availableModels: string[];
  pinnedLanguages: Record<string, string>;
  translatorReady: boolean;
  display: DisplaySettings;
}

export interface AudioDevice {
  index: number;
  name: string;
  channels: number;
  hostapi: string;
}

export interface GlossaryTerm {
  id: number;
  source: string;
  lang: string;
  /** translate = force a rendering, keep = never translate it, hint = bias ASR only. */
  mode: 'translate' | 'keep' | 'hint';
  category: string;
  targets: Record<string, string>;
}

export interface RecordingStatus {
  recording: boolean;
  path: string | null;
  seconds: number;
  /** Recent input level. Zero while recording means no audio is reaching the capture device. */
  peak: number;
  droppedBlocks: number;
  sessionId: number | null;
  backlog: number;
  errors: number;
}

/** Where a session's post-meeting pass got to. `idle` means there has not been one this run. */
export type RefineState = 'idle' | 'refining' | 'refined' | 'failed' | 'cancelled';

/** Which part of the post-meeting pass is running; absent when there has not been one. */
export type RefineStage = 'rewrite' | 'segment' | 'refine' | 'summarize';

export interface SessionSummary {
  id: number;
  started: string;
  ended: string | null;
  wav_path: string;
  lines: number;
  refine: { state: RefineState; stage?: RefineStage; error: string };
  /** Whether the recording this session was made from is still on disk. */
  hasRecording: boolean;
  /** Pre-meeting notes — agenda, attendees, slides — folded into the summary prompt. */
  reference: string;
}

/** Where the meeting summary got to. `partial` means some requested languages failed. */
export type SummaryState = 'none' | 'generating' | 'ok' | 'partial' | 'failed' | 'no_llm';

export interface MeetingSummaryLang {
  title: string;
  summary: string;
  decisions: string[];
  /** `speaker` is the diarisation code, not a display name — resolve via the names map. */
  actions: { text: string; speaker: string }[];
}

export interface MeetingSummary {
  session: number;
  state: SummaryState;
  /** The transcript has been edited since this summary was generated. */
  stale: boolean;
  created?: string;
  summary: Record<string, MeetingSummaryLang> | null;
}

/** One utterance the answer rests on, verified against the stored transcript before being returned. */
export interface AskCitation {
  session_id: number;
  line_id: number;
  start: number;
  speaker: string;
  text: string;
}

export interface AskResult {
  answer: string;
  citations: AskCitation[];
  sessions: number[];
  /** Sessions read only in part because the transcript did not fit the model's context. */
  truncated: number[];
  dropped_citations: number;
  /** False when the model returned citations but none matched a real line — treat the answer warily. */
  verified: boolean;
  budget: { provider: string; chars: number; sessions: number };
}

/** How a line ended up in the transcript. Distinct from `refined`, which is an LLM revision. */
export type LineStatus = 'ok' | 'asr_failed' | 'translate_failed';

export interface TranscriptLine {
  id: number;
  start: number;
  speaker: string;
  lang: string;
  source: string;
  /** Pre-edit text kept from the first manual correction, or null when never hand-edited. */
  orig_source: string | null;
  refined: number;
  status: LineStatus;
  end_time: number | null;
  translations: Record<string, string>;
}

export const appApi = {
  getConfig: () => request<AppConfig>('/config'),
  putConfig: (body: Partial<AppConfig>) => request<AppConfig>('/config', { method: 'PUT', body: JSON.stringify(body) }),

  devices: () =>
    request<{ devices: AudioDevice[]; configured: string; selected: number | null; error: string | null }>('/devices'),

  glossary: () => request<GlossaryTerm[]>('/glossary'),
  addTerm: (body: Partial<GlossaryTerm>) =>
    request<GlossaryTerm[]>('/glossary', { method: 'POST', body: JSON.stringify(body) }),
  removeTerm: (source: string, lang = '') =>
    request<GlossaryTerm[]>(`/glossary?source=${encodeURIComponent(source)}&lang=${encodeURIComponent(lang)}`, {
      method: 'DELETE',
    }),

  startRecording: () => request<RecordingStatus>('/recording/start', { method: 'POST' }),
  stopRecording: () => request<RecordingStatus>('/recording/stop', { method: 'POST' }),
  recordingStatus: () => request<RecordingStatus>('/recording/status'),

  sessions: async () => (await request<RawSessionSummary[]>('/sessions')).map(withRefine),
  // Sent as the raw body rather than a form: the server needs the bytes, not a field name.
  importRecording: (file: File) =>
    request<{ id: number; lines: number }>(`/sessions/import?filename=${encodeURIComponent(file.name)}`, {
      method: 'POST',
      body: file,
      headers: { 'Content-Type': 'application/octet-stream' },
    }),
  sessionLines: (id: number) =>
    request<{ lines: TranscriptLine[]; speakers: Record<string, string> }>(`/sessions/${id}/lines`),
  setSpeakerNames: (id: number, names: Record<string, string>) =>
    request<Record<string, string>>(`/sessions/${id}/speakers`, { method: 'PUT', body: JSON.stringify(names) }),
  // Editing a line also teaches the correction: the backend stores what was written against what
  // was said, and applies it to every future transcript.
  // Asked before a term is added, not after: adding 料號 rewrote the real term 料耗 42 times
  // across seven interviews, and nothing said so.
  termCollisions: (source: string) =>
    request<{ source: string; collisions: { text: string; count: number }[] }>(
      `/glossary/collisions?source=${encodeURIComponent(source)}`),
  knownSpeakers: () => request<KnownSpeaker[]>('/speakers/known'),
  renameSpeaker: (name: string, next: string) =>
    request<KnownSpeaker[]>(`/speakers/known/${encodeURIComponent(name)}`, {
      method: 'PUT',
      body: JSON.stringify({ name: next }),
    }),
  forgetSpeaker: (name: string) =>
    request<KnownSpeaker[]>(`/speakers/known/${encodeURIComponent(name)}`, { method: 'DELETE' }),
  setSpeakerLanguage: (name: string, language: string) =>
    request<KnownSpeaker[]>(`/speakers/known/${encodeURIComponent(name)}/language`, {
      method: 'PUT',
      body: JSON.stringify({ language }),
    }),
  setSpeakerDepartment: (name: string, department: string) =>
    request<KnownSpeaker[]>(`/speakers/known/${encodeURIComponent(name)}/department`, {
      method: 'PUT',
      body: JSON.stringify({ department }),
    }),
  corrections: () => request<LearnedCorrection[]>('/corrections'),
  // `wrong` is the key, so sending a different one renames the pair rather than adding another.
  editCorrection: (wrong: string, next: LearnedCorrection) =>
    request<LearnedCorrection[]>(`/corrections/${encodeURIComponent(wrong)}`, {
      method: 'PUT',
      body: JSON.stringify(next),
    }),
  forgetCorrection: (wrong: string) =>
    request<LearnedCorrection[]>(`/corrections/${encodeURIComponent(wrong)}`, { method: 'DELETE' }),
  // Pre-meeting notes for this session, folded into the summary prompt when it regenerates.
  setReference: (id: number, reference: string) =>
    request<{ reference: string }>(`/sessions/${id}/reference`, {
      method: 'PUT',
      body: JSON.stringify({ reference }),
    }),
  setLineSource: (id: number, lineId: number, source: string) =>
    request<{ lines: TranscriptLine[]; speakers: Record<string, string> }>(
      `/sessions/${id}/lines/${lineId}`, { method: 'PUT', body: JSON.stringify({ source }) }),
  // Reassign one line to another speaker: the human splitting the shared-mic collapse back apart.
  // The code may be one the meeting already has or a fresh S-code the caller minted.
  setLineSpeaker: (id: number, lineId: number, speaker: string) =>
    request<{ lines: TranscriptLine[]; speakers: Record<string, string> }>(
      `/sessions/${id}/lines/${lineId}/speaker`, { method: 'PUT', body: JSON.stringify({ speaker }) }),
  // Fold several diariser codes for one person into `into`: the clustering splits a drifting voice
  // into S17/S18/S20, and this is the human saying they are one. Same response shape as reassigning.
  mergeSpeakers: (id: number, into: string, from: string[]) =>
    request<{ lines: TranscriptLine[]; speakers: Record<string, string> }>(
      `/sessions/${id}/speakers/merge`, { method: 'POST', body: JSON.stringify({ into, from }) }),
  // Cross-meeting question. The answer carries citations the server verified against stored lines,
  // so a click can jump to the exact utterance rather than a place the model claimed one was.
  ask: (question: string) =>
    request<AskResult>('/ask', { method: 'POST', body: JSON.stringify({ question }) }),
  // The line is named, never a path or an offset: the server reads the span from its own record.
  rerunLine: (id: number, lineId: number) =>
    request<{ lines: TranscriptLine[]; speakers: Record<string, string>; status: LineStatus }>(
      `/sessions/${id}/lines/${lineId}/rerun`, { method: 'POST' }),
  // Translation only: the source stays exactly as it is, so a hand-corrected line keeps its wording.
  retranslateLine: (id: number, lineId: number) =>
    request<{ lines: TranscriptLine[]; speakers: Record<string, string>; status: LineStatus }>(
      `/sessions/${id}/lines/${lineId}/retranslate`, { method: 'POST' }),
  // Re-derives the whole transcript from the recording with the largest model. Every line is
  // replaced, so anything corrected by hand since the meeting is overwritten.
  sessionSummary: (id: number) => request<MeetingSummary>(`/sessions/${id}/summary`),
  // 409 while a job runs or the meeting is recording; 429 while fresh-and-unchanged.
  summarizeSession: (id: number) => request<MeetingSummary>(`/sessions/${id}/summarize`, { method: 'POST' }),
  // 409 while the meeting is recording or a refine pass is running.
  deleteSession: (id: number) =>
    request<{ deleted: number }>(`/sessions/${id}`, { method: 'DELETE' }),
  reprocess: (id: number) =>
    request<{ session: number; state: RefineState; error: string }>(`/sessions/${id}/reprocess`, {
      method: 'POST',
    }),
};

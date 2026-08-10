import { test } from 'node:test';
import assert from 'node:assert/strict';
import { withRefine, editingLocked } from './sessionSummary.ts';

// A server older than the refine field omits it, and `SessionSummary` says it is always there.
// Every consumer reads `s.refine.state` on that promise — the session dropdown does it inside a
// map over all sessions, so one legacy row threw and the ErrorBoundary blanked the whole
// transcript. Normalising here is what makes the type true; these guard that it stays true.

const raw = { id: 1, started: '2026-01-01 10:00', ended: null, wav_path: 'a.wav', lines: 12 };

test('a session with no refine field gets an idle one', () => {
  const s = withRefine(raw);
  assert.deepEqual(s.refine, { state: 'idle', error: '' });
});

test('reading .refine.state on a legacy session does not throw', () => {
  assert.doesNotThrow(() => withRefine(raw).refine.state);
});

test('an existing refine field is left alone', () => {
  const refine = { state: 'failed' as const, error: 'model timed out' };
  assert.deepEqual(withRefine({ ...raw, refine }).refine, refine);
});

test('the rest of the session is carried through untouched', () => {
  // The fields this fills in are stripped; everything else must arrive exactly as it came.
  const { refine: _refine, hasRecording: _hasRecording, reference: _reference, ...rest } = withRefine(raw);
  assert.deepEqual(rest, raw);
});

test('a server that does not report a reference defaults it to empty', () => {
  assert.equal(withRefine(raw).reference, '');
});

test('an explicit reference is kept', () => {
  assert.equal(withRefine({ ...raw, reference: 'agenda' }).reference, 'agenda');
});

test('a server that does not report hasRecording is assumed to have the audio', () => {
  // Guessing the other way would hide working playback behind a "recording missing" notice.
  assert.equal(withRefine(raw).hasRecording, true);
});

test('an explicit false is kept', () => {
  assert.equal(withRefine({ ...raw, hasRecording: false }).hasRecording, false);
});

test('editing is locked while a full pass rewrites lines', () => {
  assert.equal(editingLocked({ state: 'refining', stage: 'rewrite', error: '' }), true);
  assert.equal(editingLocked({ state: 'refining', stage: 'refine', error: '' }), true);
});

test('editing stays open while only the summary regenerates', () => {
  // A summarize-only job reports state "refining" too, but never touches a transcript line.
  assert.equal(editingLocked({ state: 'refining', stage: 'summarize', error: '' }), false);
});

test('editing is open when the pass is not running', () => {
  assert.equal(editingLocked({ state: 'idle', error: '' }), false);
  assert.equal(editingLocked({ state: 'refined', error: '' }), false);
});

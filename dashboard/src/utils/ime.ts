import type { KeyboardEvent } from 'react';

// While an IME composition is active, keys like Escape belong to the IME (cancel candidate), not the app;
// Safari reports in-composition keys as keyCode 229.
export const isComposingKey = (e: KeyboardEvent) =>
  e.nativeEvent.isComposing || e.keyCode === 229;

// Enter that confirms a zh/vi IME candidate is not a submit: Chrome/Edge flag it isComposing,
// Safari sends it after compositionend as keyCode 229.
export const isSubmitEnter = (e: KeyboardEvent) =>
  e.key === 'Enter' && !e.nativeEvent.isComposing && e.keyCode !== 229;

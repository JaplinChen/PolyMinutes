import type { KeyboardEvent } from 'react';

// Enter that confirms a zh/vi IME candidate is not a submit: Chrome/Edge flag it isComposing,
// Safari sends it after compositionend as keyCode 229.
export const isSubmitEnter = (e: KeyboardEvent) =>
  e.key === 'Enter' && !e.nativeEvent.isComposing && e.keyCode !== 229;

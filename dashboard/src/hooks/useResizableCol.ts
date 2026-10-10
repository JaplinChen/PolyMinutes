import { useCallback, useRef } from 'react';
import { readStorage, writeStorage } from '../utils/storage.ts';

/**
 * Drag-resizes individual grid columns of a panel whose row/header templates read `var(--col-<key>)`.
 * Widths are set as CSS custom properties on the panel element (not inline grid styles, so the
 * stylesheet stays the single source of the template) and persisted in localStorage as a JSON map
 * of column key -> width. Each resizable column header carries `data-col="<key>"` and a handle that
 * calls `startResize('<key>')`.
 */
export function restoreColumnWidths(el: HTMLElement, saved: string | null): void {
  if (!saved) return;
  try {
    const widths = JSON.parse(saved) as Record<string, string>;
    for (const [key, w] of Object.entries(widths)) el.style.setProperty(`--col-${key}`, w);
  } catch {
    // Ignore a corrupt entry — fall back to the stylesheet defaults.
  }
}

export function useResizableCol(storageKey: string) {
  const ref = useRef<HTMLElement | null>(null);

  // A callback ref rather than an effect: the table mounts only after a loading skeleton, so an
  // effect keyed on storageKey runs while the ref is still null and the saved widths are never
  // applied — the table reverted to defaults on every reload. A callback ref fires when the node
  // actually attaches, whenever that happens to be.
  const setRef = useCallback(
    (el: HTMLElement | null) => {
      ref.current = el;
      if (el) restoreColumnWidths(el, readStorage(storageKey));
    },
    [storageKey],
  );

  const startResize = (colKey: string) => (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    const panel = ref.current;
    if (!panel) return;
    const cell = (e.currentTarget as HTMLElement).closest('[data-col]') as HTMLElement | null;
    const startX = e.clientX;
    const startW = cell ? cell.getBoundingClientRect().width : 120;

    const move = (ev: MouseEvent) => {
      const w = Math.max(40, startW + (ev.clientX - startX));
      panel.style.setProperty(`--col-${colKey}`, `${Math.round(w)}px`);
    };
    const up = () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseup', up);
      const widths: Record<string, string> = {};
      for (const name of panel.style) {
        if (name.startsWith('--col-')) widths[name.slice(6)] = panel.style.getPropertyValue(name);
      }
      writeStorage(storageKey, JSON.stringify(widths));
    };
    window.addEventListener('mousemove', move);
    window.addEventListener('mouseup', up);
  };

  return { ref: setRef, startResize };
}

import { useEffect, useRef, useState } from 'react';

// Open/close state for a popup menu that dismisses on outside-click or Escape.
export function useDismissableMenu<T extends HTMLElement>(onEscape?: () => void) {
  const [open, setOpen] = useState(false);
  const ref = useRef<T>(null);
  const onEscapeRef = useRef(onEscape);
  useEffect(() => { onEscapeRef.current = onEscape; });

  useEffect(() => {
    if (!open) return;

    const closeOnOutsideClick = (event: MouseEvent) => {
      if (!ref.current?.contains(event.target as Node)) setOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setOpen(false);
      onEscapeRef.current?.();
    };

    document.addEventListener('mousedown', closeOnOutsideClick);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.removeEventListener('mousedown', closeOnOutsideClick);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [open]);

  return { open, setOpen, ref };
}

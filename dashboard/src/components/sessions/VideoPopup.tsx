import { useEffect, useRef, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { ChevronDown, ChevronUp, GripHorizontal, PictureInPicture2 } from 'lucide-react';
import { useToast } from '../Toast';
import './VideoPopup.css';

// Where the user left the window: position, size and whether it was collapsed. One key, because
// these are one decision — "this is where I watch meetings" — and restoring half of it is worse
// than restoring none.
const STORE = 'polyminutes.videoPopup';
const MIN_W = 240;
const MIN_H = 160;
const MARGIN = 16;

type Box = { x: number; y: number; w: number; h: number; open: boolean };

function stored(): Partial<Box> {
  try {
    return JSON.parse(localStorage.getItem(STORE) || '{}');
  } catch {
    return {};
  }
}

/** Inside the viewport, whatever was saved. A window dragged to a second screen that is no longer
 *  attached would otherwise come back at coordinates nothing can reach. */
function clamp(box: Box): Box {
  const w = Math.min(Math.max(box.w, MIN_W), window.innerWidth - MARGIN * 2);
  const h = Math.min(Math.max(box.h, MIN_H), window.innerHeight - MARGIN * 2);
  return {
    ...box,
    w,
    h,
    x: Math.min(Math.max(box.x, MARGIN), window.innerWidth - w - MARGIN),
    y: Math.min(Math.max(box.y, MARGIN), window.innerHeight - h - MARGIN),
  };
}

interface Props {
  src: string;
  videoRef: React.MutableRefObject<HTMLVideoElement | null>;
  onPause: () => void;
  onTimeUpdate: () => void;
}

/** The meeting's video as a floating window: draggable by its bar, resizable by its corner.
 *
 * A panel in the flow could only ever be a compromise — big enough to read a shared screen means
 * the transcript starts below the fold, small enough to leave the transcript alone means nobody
 * can tell who is speaking. Floating, the two stop competing for the same column.
 */
export function VideoPopup({ src, videoRef, onPause, onTimeUpdate }: Props) {
  const { t } = useTranslation();
  const toast = useToast();
  const panel = useRef<HTMLDivElement | null>(null);
  const [open, setOpen] = useState(stored().open ?? true);

  const save = (extra: Partial<Box> = {}) => {
    const el = panel.current;
    if (!el) return;
    // Collapsed, the element is as tall as its title bar, and writing that would reopen the window
    // at 30 pixels. The height to keep is the one it had when there was something to see.
    const height = open ? el.offsetHeight : stored().h ?? 300;
    const box: Box = {
      x: el.offsetLeft, y: el.offsetTop, w: el.offsetWidth, h: height, open, ...extra,
    };
    localStorage.setItem(STORE, JSON.stringify(box));
  };

  // Position and size live on the element, not in state: the user drags the element itself, and a
  // re-render that reasserted a state value would fight the pointer. Only the collapsed flag is
  // React's, because that one changes what is rendered.
  useEffect(() => {
    const el = panel.current;
    if (!el) return;
    const saved = stored();
    const box = clamp({
      w: saved.w ?? 480,
      h: saved.h ?? 300,
      // First run: bottom-right, out of the way of a transcript that reads left-aligned.
      x: saved.x ?? window.innerWidth - (saved.w ?? 480) - MARGIN,
      y: saved.y ?? window.innerHeight - (saved.h ?? 300) - MARGIN,
      open: true,
    });
    el.style.left = `${box.x}px`;
    el.style.top = `${box.y}px`;
    el.style.width = `${box.w}px`;
    if (open) el.style.height = `${box.h}px`;
  }, [open]);

  const startDrag = (e: React.PointerEvent) => {
    const el = panel.current;
    // Only the primary button, and never the collapse button inside the bar.
    if (!el || e.button !== 0 || (e.target as HTMLElement).closest('button')) return;
    const dx = e.clientX - el.offsetLeft;
    const dy = e.clientY - el.offsetTop;
    const move = (ev: PointerEvent) => {
      el.style.left = `${Math.min(Math.max(ev.clientX - dx, 0), window.innerWidth - el.offsetWidth)}px`;
      el.style.top = `${Math.min(Math.max(ev.clientY - dy, 0), window.innerHeight - el.offsetHeight)}px`;
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
      save();
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  const toggle = () => {
    // Saved before the height changes: collapsing must not overwrite the size it will reopen at.
    save({ open: !open });
    setOpen(o => !o);
  };

  return (
    <div ref={panel} className="vpop" data-open={open} onPointerUp={() => save()}>
      <div className="vpop-bar" onPointerDown={startDrag} title={t('sessions.videoDragHint')}>
        <GripHorizontal size={14} />
        <span className="vpop-title">{t('sessions.videoLabel')}</span>
        {/* The browser's own picture-in-picture: the one window that can leave the page entirely,
            onto a second screen, and keep playing while the user reads something else. Hidden
            where the browser does not offer it rather than offered and failing. */}
        {'requestPictureInPicture' in HTMLVideoElement.prototype && (
          <button
            type="button"
            className="vpop-toggle"
            title={t('sessions.pipHint')}
            aria-label={t('sessions.pipHint')}
            onClick={() => {
              const el = videoRef.current;
              if (!el) return;
              // Already out: the same button puts it back, which is what a toggle has to do.
              if (document.pictureInPictureElement === el) void document.exitPictureInPicture();
              else void el.requestPictureInPicture().catch(() => toast.error(t('sessions.pipFailed')));
            }}
          >
            <PictureInPicture2 size={14} />
          </button>
        )}
        <button
          type="button"
          className="vpop-toggle"
          aria-expanded={open}
          title={t(open ? 'sessions.hideVideo' : 'sessions.showVideo')}
          aria-label={t(open ? 'sessions.hideVideo' : 'sessions.showVideo')}
          onClick={toggle}
        >
          {open ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
        </button>
      </div>
      <video
        ref={videoRef}
        className="vpop-video"
        controls
        preload="metadata"
        aria-label={t('sessions.videoLabel')}
        src={src}
        onPause={onPause}
        onTimeUpdate={onTimeUpdate}
      />
    </div>
  );
}

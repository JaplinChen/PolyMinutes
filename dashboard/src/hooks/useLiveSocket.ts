import { useEffect, useRef, useState } from 'react';
import { API_BASE_URL } from '../services/api';
import { mergeLine } from '../utils/mergeLine';
import type { DisplaySettings } from '../services/app.api';

export type { DisplaySettings };

export interface LiveLine {
  id: number;
  start: number;
  speaker: string;
  lang: string;
  source: string;
  translations: Record<string, string>;
  refined: boolean;
}

const DEFAULT_DISPLAY: DisplaySettings = {
  font_size: 40,
  lines: 6,
  show_source: 'top',
  show_speaker: true,
  colour_speakers: true,
  theme: 'dark',
};

// Live lines kept in memory. The display shows at most display.lines (capped at 20 in settings);
// this dwarfs that, leaving room for out-of-order retries and in-place revisions while bounding the
// buffer over a multi-hour meeting.
const MAX_LIVE_LINES = 200;

// API_BASE_URL is either '/api' (same origin) or 'http://host:port/api' (Vite dev server).
function socketUrl(): string {
  const base = API_BASE_URL.replace(/\/api$/, '');
  if (/^https?:/.test(base)) return `${base.replace(/^http/, 'ws')}/ws/live`;
  return `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/live`;
}

/**
 * Live subtitle feed.
 *
 * The server sends `line` for a new utterance and `update` for one it has revised after seeing
 * what came next, so lines are keyed by id and replaced in place — appending an `update` would
 * show the same sentence twice.
 *
 * New lines are inserted by start time rather than appended, because they do not always arrive in
 * order: an utterance the recogniser gave up on is held and retried once its speaker's language is
 * known, by which point later lines are already on screen.
 */
export function useLiveSocket() {
  const [lines, setLines] = useState<LiveLine[]>([]);
  const [display, setDisplay] = useState<DisplaySettings>(DEFAULT_DISPLAY);
  const [languages, setLanguages] = useState<string[]>([]);
  const [connected, setConnected] = useState(false);
  const retry = useRef<number | undefined>(undefined);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;

    const connect = () => {
      socket = new WebSocket(socketUrl());

      socket.onopen = () => setConnected(true);

      socket.onmessage = event => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'config') {
          setLanguages(msg.languages ?? []);
          if (msg.display) setDisplay({ ...DEFAULT_DISPLAY, ...msg.display });
          return;
        }
        if (msg.type === 'line' || msg.type === 'update') {
          setLines(prev => mergeLine(prev, msg.line, MAX_LIVE_LINES));
        }
      };

      socket.onclose = () => {
        setConnected(false);
        // The TV is unattended, so reconnect on its own rather than waiting for someone to reload.
        if (!closed) retry.current = window.setTimeout(connect, 2000);
      };
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(retry.current);
      socket?.close();
    };
  }, []);

  return { lines, display, languages, connected };
}

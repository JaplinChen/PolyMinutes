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

// The server pings every 15s when idle; silence past two pings means a half-open socket.
const STALE_MS = 40_000;

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
  const sessionId = useRef<number | null>(null);
  const lastMessageAt = useRef(Date.now());

  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;

    // Line start times restart at 0 each meeting, so the old meeting's lines would bury the new ones.
    const enterSession = (id: unknown) => {
      if (typeof id !== 'number' || id === sessionId.current) return;
      sessionId.current = id;
      setLines([]);
    };

    const connect = () => {
      lastMessageAt.current = Date.now();
      socket = new WebSocket(socketUrl());

      socket.onopen = () => setConnected(true);

      socket.onmessage = event => {
        lastMessageAt.current = Date.now();
        let msg;
        try {
          msg = JSON.parse(event.data);
        } catch {
          return;
        }
        if (msg.type === 'config') {
          enterSession(msg.sessionId);
          setLanguages(msg.languages ?? []);
          if (msg.display) setDisplay({ ...DEFAULT_DISPLAY, ...msg.display });
          return;
        }
        if (msg.type === 'session') {
          enterSession(msg.sessionId);
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
    // A dropped Wi-Fi link never fires onclose, and close() on it can wait out the closing handshake,
    // so detach the dead socket and reconnect directly.
    const watchdog = window.setInterval(() => {
      if (socket?.readyState === WebSocket.OPEN && Date.now() - lastMessageAt.current > STALE_MS) {
        socket.onclose = null;
        socket.onmessage = null;
        socket.close();
        setConnected(false);
        connect();
      }
    }, 5000);
    return () => {
      closed = true;
      window.clearTimeout(retry.current);
      window.clearInterval(watchdog);
      if (socket) {
        socket.onclose = null;
        socket.onmessage = null;
        socket.close();
      }
    };
  }, []);

  return { lines, display, languages, connected };
}

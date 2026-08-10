# PolyMinutes Dashboard

React + Vite frontend. The built output is served by the Python service, so in normal use there is
nothing to run here — `start.bat` / `start.command` builds it.

For frontend work, start the Python service on port 8010 first, then:

```bash
npm install
npm run dev
```

The dev server listens on 2886 and proxies `/api` and `/ws` to the service.

| Command | |
|---|---|
| `npm run dev` | dev server with proxy |
| `npm run build` | type-check and build into `dist/` |
| `npm run lint` | ESLint |

## Layout

- `src/pages/Live.tsx` — the subtitle page shown on the meeting-room TV. Sized for viewing
  distance, so it deliberately shares nothing with the dashboard's type scale.
- `src/pages/Capture.tsx` — device selection, start/stop, and the input level meter that reveals
  a silent capture path.
- `src/pages/Sessions.tsx` — past transcripts and speaker naming.
- `src/pages/Glossary.tsx` — term handling.
- `src/pages/Display.tsx` — language set and subtitle layout.
- `src/hooks/useLiveSocket.ts` — subtitle stream. Lines are keyed by id because the server revises
  a line after seeing what followed; appending an update would show the sentence twice.
- `src/services/app.api.ts` — typed API client.

Several pages and components originate from OpenWA-Lab. Login, API keys and role-based access were
removed — this is a single-user local app, so `useRole` returns a constant.

i18n covers Traditional Chinese, Vietnamese and English, with key parity across all three.

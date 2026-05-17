# Frontend Notes

This frontend is a mobile-first PWA shell and can now be run with Vite:

```text
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Vite will print a local URL such as:

```text
VITE v8.x.x ready in ...
Local:   http://localhost:5173/
Network: http://<your-ip>:5173/
```

You can still open it directly from `frontend/index.html` for simple static checks, but the recommended dev flow is the Vite server above.

It is intentionally backend-agnostic for now:

- simulation mode is enabled by default
- live camera uses local mock imagery until the Pi backend exists
- Ask AI is wired as a UI shell only
- Settings are read-only in MVP

Planned backend contract:

```text
GET  /api/status
GET  /api/detections
GET  /api/events
POST /api/ask
GET  /stream.mjpg
```

Kanna assets:

```text
frontend/assets/kanna/
```

The app currently shows a fallback companion card until those PNG files are dropped in.

# Backend Notes

Initial backend contract is implemented with FastAPI-style routes:

```text
GET  /api/status
GET  /api/detections
GET  /api/events
POST /api/dev/emit
POST /api/ask
GET  /stream.mjpg
```

Static mounts:

```text
/           -> frontend/
/other      -> other/
```

Suggested local run once dependencies are installed:

```text
uvicorn api_server.app:app --host 0.0.0.0 --port 8091
```

The frontend auto-detects this backend. If `/api/status` responds, the app switches from local mock mode into backend-driven mode and subscribes to SSE events.

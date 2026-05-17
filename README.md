# Wildlife Detector for Raspberry Pi 5 + Hailo-8L

Real-time wildlife detection and deterrent system built around a Raspberry Pi 5, Hailo-8L, CSI camera, PIR-triggered inference flow, and a companion web app stack.

## Repo layout

```text
api_server/   FastAPI backend
detector/     C++ detector core and Pi deployment files
docs/         Roadmap and hardware documentation
frontend/     Mobile-first PWA
samples/      Demo media used by the app and docs
```

## Architecture

```text
hailo_detector (C++)
  -> camera capture
  -> Hailo inference
  -> PIR/audio/ultrasonic control
  -> MJPEG preview on :8090
  -> JSONL event log
  -> Unix domain socket live events

api_server (FastAPI)
  -> loads recent events
  -> serves status + detections API
  -> streams SSE events
  -> serves frontend shell

frontend (Vite PWA)
  -> dashboard
  -> live view
  -> detections history
  -> companion animation layer
```

## What each folder does

- `detector/`
  Detector-side C++ code, event pipeline, systemd service file, and a small event simulation helper.
- `api_server/`
  Backend API for status, detections, SSE events, and frontend hosting.
- `frontend/`
  PWA shell with live stream UI, detections feed, Ask AI shell, and Kanna companion animation.
- `docs/`
  Project roadmap plus hardware photos.
- `samples/media/`
  Demo images used by the backend and frontend mock mode.

## Key features

- 9-class wildlife detection: `bear`, `coyote`, `deer`, `fox`, `possum`, `raccoon`, `skunk`, `squirrel`, `turkey`
- PIR-gated inference to avoid constant accelerator usage
- MJPEG browser preview
- Detection lifecycle events: `sighting_start`, `sighting_update`, `sighting_end`
- JSONL persistence plus Unix socket live transport
- FastAPI API layer with SSE
- Mobile-friendly frontend with Kanna companion overlay

## Run the backend

```bash
uvicorn api_server.app:app --host 0.0.0.0 --port 8091
```

## Run the frontend

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

## Notes

- Production-facing companion assets live under `frontend/assets/kanna/`.
- Large raw art working folders are intentionally excluded to keep the repo easy to browse.
- The longer implementation plan lives in `docs/FRONTEND_BACKEND_ROADMAP.md`.

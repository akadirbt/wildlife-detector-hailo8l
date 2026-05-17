# Wildlife Detector for Raspberry Pi 5 + Hailo-8L

Real-time wildlife detection and deterrent system built around a Raspberry Pi 5, Hailo-8L, CSI camera, PIR-triggered inference flow, and a companion web app stack.

## What is in this repo

- `hailo_detector.cpp`, `gpio_trigger.hpp`
  Main detector loop, PIR gating, MJPEG preview, deterrent audio, ultrasonic output, and hardware integration.
- `detection_event.hpp`, `sighting_tracker.*`, `event_sink.*`
  Event backbone for start/update/end sighting lifecycles plus JSONL and Unix domain socket output.
- `api_server/`
  FastAPI backend for status, detections, SSE events, and frontend hosting.
- `frontend/`
  Mobile-first PWA dashboard with live stream card, detections feed, Ask AI shell, and Kanna companion UI.
- `tools/`
  Small local simulation helper for Sprint 0 event testing.
- `other/`
  Demo snapshots and hardware photos used by the backend/frontend and project documentation.

## Current architecture

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

## Key features

- 9-class wildlife detection: `bear`, `coyote`, `deer`, `fox`, `possum`, `raccoon`, `skunk`, `squirrel`, `turkey`
- PIR-gated inference to avoid constant accelerator usage
- MJPEG browser preview
- Detection lifecycle events: `sighting_start`, `sighting_update`, `sighting_end`
- JSONL persistence plus Unix socket live transport
- FastAPI API layer with SSE
- Mobile-friendly frontend with Kanna companion overlay

## Repo layout notes

- The repo keeps production-facing frontend assets under `frontend/assets/kanna/`.
- Large raw art working folders are intentionally excluded from version control to keep GitHub cleaner.
- Build outputs, caches, detections, and local runtime logs are ignored.

## Build and run

### Detector

```bash
cmake -B build -S .
cmake --build build --target hailo_detector -j4
./build/hailo_detector
```

Preview:

```text
http://127.0.0.1:8090
```

### Backend

```bash
uvicorn api_server.app:app --host 0.0.0.0 --port 8091
```

### Frontend

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

## Project status

The detector core, event pipeline, backend shell, and frontend MVP are all present in this repository. The longer implementation plan is documented in `FRONTEND_BACKEND_ROADMAP.md`.

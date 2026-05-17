# Backend Documentation

## Role of the backend

The backend is the bridge between the detector and the user interface.

Its job is to take detector-oriented outputs and turn them into app-facing capabilities.

Without the backend, the detector can still run, but the project would remain difficult to consume from a browser or mobile-style interface.

## Main files

- `api_server/app.py`
  Main FastAPI application and route definitions.
- `api_server/event_store.py`
  In-memory event storage, replay, summary, and detection aggregation helpers.
- `api_server/status.py`
  Status payload construction helpers.
- `api_server/demo_data.py`
  Seed/demo event payloads for mock and development mode.
- `api_server/config.yaml`
  Local configuration defaults for backend behavior.
- `api_server/requirements.txt`
  Python dependencies for the backend shell.

## What the backend does

### 1. Load recent event history

The backend seeds itself from JSONL data or demo data.

This allows the UI to show useful information immediately after startup rather than waiting for a brand-new event.

### 2. Maintain an in-memory event store

The backend keeps recent events in memory to support:

- fast reads
- recent detection queries
- SSE streaming
- lightweight summaries

### 3. Expose REST-style endpoints

The backend currently exposes:

- `GET /api/status`
- `GET /api/detections`
- `GET /api/events`
- `POST /api/dev/emit`
- `POST /api/ask`
- `GET /stream.mjpg`

## Endpoint intent

### `GET /api/status`

Returns a product-facing snapshot of system condition, such as:

- detector status
- stream status
- recent event information
- light telemetry

### `GET /api/detections`

Returns recent detections in a frontend-friendly form.

Instead of exposing raw detector internals directly, it shapes data around what the app wants to render.

### `GET /api/events`

Streams events over SSE.

This is the main path used for live frontend updates.

### `POST /api/dev/emit`

Development helper for emitting fake or demo events.

This is especially useful when frontend work continues before the full live detector integration is always available.

### `POST /api/ask`

Current placeholder path for the assistant-style UI layer.

It already gives the repo a product direction beyond pure detection, even though the answer path is still lightweight.

### `GET /stream.mjpg`

Provides or redirects the stream entry point used by the UI.

## Why the backend matters

The detector emits technically useful signals, but the backend makes those signals product-ready.

It does the "translation" layer between:

- hardware-oriented events
- user-facing browser behavior

That includes:

- replay
- shaping payloads
- SSE
- simple summaries
- static serving

## Event store design

The event store is intentionally lightweight.

It keeps enough logic to support the app, while staying simple and easy to reason about.

This is useful for an MVP because it avoids overbuilding before the final event pipeline hardens.

## Demo mode and developer ergonomics

A strong part of the backend design is that it supports development without requiring the physical detector to be live for every single UI iteration.

This matters because:

- frontend iteration is much faster
- demos are easier to stage
- app flows can be tested indoors or away from the full deployment rig

## Static serving role

The backend also serves:

- the frontend shell
- sample media

This helps the repo behave more like one integrated product rather than three disconnected codebases.

## Config and runtime assumptions

The backend assumes project-level paths such as:

- `frontend/`
- `samples/media/`
- detector-produced detections root

These assumptions are codified in `api_server/config.yaml`.

## What this part of the repo shows

The backend demonstrates the service-design side of the project:

- event brokering
- API shaping
- real-time streaming
- app hosting
- mock-friendly development workflow

It is a key part of turning the detector from a hardware prototype into an actual application system.

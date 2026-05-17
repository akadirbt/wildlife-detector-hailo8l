# System Overview

## Purpose

This project is an end-to-end wildlife monitoring and deterrence system designed for Raspberry Pi 5 + Hailo-8L hardware.

The goal is not only to detect animals, but to turn those detections into a usable field system:

- motion-aware on-device inference
- immediate deterrent response
- live browser visibility
- structured event export
- mobile-friendly frontend experience

In practice, this means the repository combines embedded logic, computer vision, event modeling, backend APIs, and frontend UX in one coherent stack.

## High-level stack

```text
Camera + PIR + GPIO + Audio + Ultrasonic
                |
                v
detector/ (C++ on Raspberry Pi)
  - capture frames
  - run Hailo inference
  - gate activity using PIR
  - trigger deterrent outputs
  - publish event lifecycle
                |
                v
api_server/ (FastAPI)
  - load and replay recent detections
  - expose REST API
  - stream SSE events
  - serve frontend shell
                |
                v
frontend/ (PWA)
  - dashboard
  - live stream
  - detections feed
  - Kanna companion layer
```

## Why the architecture is split

The system is intentionally divided into layers because the detector loop and the application layer have very different requirements.

The detector loop needs to stay fast, deterministic, and close to hardware. It should not block on network I/O, frontend rendering, or slower app logic.

The backend and frontend layers need a different style:

- they aggregate data
- expose user-facing views
- recover from restarts
- provide a richer product experience

This separation keeps the Pi-side hot path lean while still making the overall project feel like a full product.

## Main runtime flow

### 1. Motion gating

The PIR sensor acts as a power and compute gate.

Instead of running heavy inference continuously, the system first checks for motion. Once motion is confirmed, the detector enters an active inference window.

### 2. Frame processing

The detector captures frames from the CSI camera and sends them through the Hailo pipeline.

Each detection includes:

- class label
- confidence
- bounding box
- frame dimensions

### 3. Response logic

Depending on the detected class, the detector may:

- draw overlays for preview
- play class-specific deterrent audio
- drive ultrasonic output
- save or reference a snapshot

### 4. Event lifecycle

Raw detections are converted into sighting lifecycle events:

- `sighting_start`
- `sighting_update`
- `sighting_end`

This is important because a product UI does not want a flood of frame-by-frame detections. It wants meaningful "an animal appeared / is still here / has left" state.

### 5. Event export

Events are pushed through two channels:

- JSONL for durable record / replay
- Unix domain socket for low-latency live app updates

### 6. App-layer consumption

The backend reads recent events and exposes:

- status
- detection history
- SSE event stream

The frontend consumes those endpoints and presents the data as a mobile-first experience.

## Main subsystems

## Detector

The detector layer is where most of the hardware-aware behavior lives.

It is responsible for:

- camera ingestion
- Hailo inference
- PIR gating
- deterrent triggering
- live MJPEG preview
- event creation and export

See:

- [Detector Documentation](DETECTOR.md)

## Backend

The backend acts as a broker between detector output and UI consumption.

It is responsible for:

- loading recent event history
- serving detection/status APIs
- SSE broadcasting
- lightweight demo mode behavior
- hosting the frontend shell

See:

- [Backend Documentation](BACKEND.md)

## Frontend

The frontend is the product-facing surface.

It is responsible for:

- rendering the live feed
- showing recent detections
- surfacing status
- exposing mock/demo behavior
- animating the Kanna companion

See:

- [Frontend Documentation](FRONTEND.md)

## Hardware assumptions

The project currently assumes a Raspberry Pi 5-based field build with:

- Hailo-8L accelerator
- CSI camera
- PIR sensor
- audio output path
- ultrasonic deterrent path
- LEDs
- temperature sensing path

The codebase is therefore not a generic desktop detector project. It is shaped around a real hardware installation.

## Repository as a product record

The repo is meant to show not just "the final code", but the actual engineering breadth of the project:

- embedded/hardware integration
- CV pipeline design
- event-system thinking
- backend service design
- PWA and companion UX
- productization decisions

That is why `docs/`, `samples/`, and the demo media are kept alongside the implementation.

## Current maturity

The repository already contains:

- detector-side C++ core
- event lifecycle pipeline
- backend shell
- frontend MVP
- documentation and demo assets

The roadmap file documents where the system is intended to go next:

- [Roadmap](FRONTEND_BACKEND_ROADMAP.md)

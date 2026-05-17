# Detector Documentation

## What the detector is responsible for

The detector is the hardware-adjacent core of the project.

Its job is to turn real-world motion in front of a camera into:

- animal detections
- deterrent actions
- live visual output
- structured events the rest of the app can consume

This layer is designed to run on a Raspberry Pi 5 with a Hailo-8L and attached peripherals.

## Main files

- `detector/hailo_detector.cpp`
  The main application entry point and the largest integration file in the repo.
- `detector/gpio_trigger.hpp`
  GPIO, PIR, LED, and deterrent-related helpers.
- `detector/detection_event.hpp`
  Event model and JSON serialization helpers.
- `detector/sighting_tracker.hpp`
- `detector/sighting_tracker.cpp`
  Convert frame detections into start/update/end sighting lifecycle events.
- `detector/event_sink.hpp`
- `detector/event_sink.cpp`
  Push events to JSONL and Unix domain socket outputs.
- `detector/hailo-detector.service`
  Example systemd unit for Pi deployment.
- `detector/tools/simulate_sprint0.cpp`
  Small local simulation helper for the event pipeline.

## Detector responsibilities in detail

### 1. Camera ingestion

The detector captures frames from the CSI camera and prepares them for inference and preview rendering.

This is the visual input source for the entire system.

### 2. Hailo inference

The detector sends frames through the Hailo inference path and receives class detections and bounding boxes.

The supported wildlife classes are:

- `bear`
- `coyote`
- `deer`
- `fox`
- `possum`
- `raccoon`
- `skunk`
- `squirrel`
- `turkey`

### 3. PIR-gated activity

The PIR sensor is used to avoid running active inference unnecessarily.

This matters for a field device because it reduces wasted compute, avoids running the accelerator constantly, and better matches real-world activity.

### 4. Deterrent control

The detector can react to detections using:

- class-specific audio playback
- ultrasonic deterrent output

This is one of the reasons the project is more than just a detection demo: it includes an actual response layer.

### 5. Preview streaming

The detector serves an MJPEG preview stream that can be opened in a browser.

This gives the operator a live visual view without needing a heavyweight video pipeline in the MVP stage.

### 6. Event export

The detector publishes structured events describing wildlife sightings.

Those events are much more useful downstream than raw per-frame detections because they express lifecycle state.

## Why the event model matters

If the frontend or backend consumed every raw detection independently, the system would feel noisy and difficult to interpret.

Instead, the detector builds a higher-level model:

- when a sighting starts
- when it updates
- when it ends

This keeps the UI meaningful and makes historical reasoning much easier.

## Sighting lifecycle

The detector-side sighting tracker groups detections into a stable lifecycle:

### `sighting_start`

- a new animal has appeared
- a sighting ID is created
- an initial snapshot path can be attached

### `sighting_update`

- the same animal is still being tracked
- confidence and/or location changed
- downstream consumers can update overlays or companion state

### `sighting_end`

- the sighting has timed out or left frame
- the lifecycle is complete
- downstream consumers can clear active tracking state

## Event transport strategy

The detector writes events through two channels:

### JSONL

Used as:

- durable ground truth
- restart-time replay source
- historical debugging artifact

### Unix domain socket

Used as:

- low-latency live transport
- best-effort app-facing signal

This gives the system both reliability and responsiveness.

## Why the detector is kept separate from the backend/frontend

The detector path is timing-sensitive.

It should not block on:

- HTTP work
- frontend rendering
- AI assistant requests
- general product-layer logic

That is why the codebase keeps the detector side as its own subsystem, even though the repo includes product-facing layers too.

## Operational outputs

The detector produces multiple useful outputs at once:

- live MJPEG stream
- audio/ultrasonic deterrence
- event records
- data for backend APIs
- data for frontend visualizations

## Running the detector

Typical build and run flow:

```bash
cmake -B build -S detector
cmake --build build --target hailo_detector -j4
./build/hailo_detector
```

Live preview:

```text
http://127.0.0.1:8090
```

## Deployment

The repo includes:

- `detector/hailo-detector.service`

This is intended as the systemd entrypoint for long-running deployment on the Pi.

## What this part of the repo shows

The detector layer represents a lot of the low-level engineering effort in the project:

- real device assumptions
- sensor gating
- inference integration
- GPIO/peripheral coordination
- event-model design
- preview serving

It is the core that everything else in the stack is built on top of.

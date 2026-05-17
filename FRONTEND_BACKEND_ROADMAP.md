# Wildlife Detector App Roadmap

Bu dokuman, mevcut Raspberry Pi 5 + Hailo detector sistemini bozmadan uzerine backend, PWA frontend, Kanna companion animasyonu ve AI asistan katmani eklemek icin kilitlenen plandir.

Ana karar: calisan C++ detector core sade kalacak. Frontend, backend, AI, auth, animasyon ve mobil app mantigi detector binary'sinin icine girmeyecek.

## Final Architecture

```text
hailo_detector (C++)
  - camera -> Hailo -> tracker -> sighting events
  - GPIO / PIR / audio / ultrasonic deterrent
  - MJPEG stream on port 8090
  - JSONL append as durable ground truth
  - UDS push as live event transport

api_server (FastAPI)
  - UDS listener
  - in-memory ring buffer
  - JSONL replay / startup seed
  - REST API
  - SSE live events
  - Claude Haiku calls
  - serves PWA

PWA frontend
  - mobile Safari app shell
  - dashboard / live / detections / ask AI / settings
  - MJPEG img stream
  - SSE EventSource
  - Kanna companion PNG/CSS overlay
  - offline shell + cached last known state
```

## Core Principles

- C++ detector loop must never block on API, frontend, network, or AI work.
- JSONL is persistence/replay, not live transport.
- UDS is live transport, best-effort, and allowed to drop under pressure.
- C++ always writes both channels:
  - JSONL: durable ground truth
  - UDS: low-latency live signal
- FastAPI is the broker between detector and app.
- PWA never sees API keys.
- Claude/Haiku receives compressed summaries, never raw full logs.
- Kanna reacts to `sighting_start/update/end`, not raw per-frame detections.

## Engineering Discipline

We use an RTL-style design mindset for this project:

```text
fix root causes, not symptoms
keep timing-sensitive paths isolated from slow I/O
define clean contracts before wiring modules together
prefer simple deterministic code over clever abstractions
add dependencies only when they remove real risk or complexity
make backpressure explicit
make failure modes observable
```

Detector hot path rule:

```text
inference loop may create/move event objects
inference loop must not serialize JSON
inference loop must not wait on sockets, frontend, backend, or AI
event_sink writer thread owns slow I/O
```

When a bug appears:

```text
first inspect the contract and state machine
then fix the owner module
avoid local patches that hide broken lifecycle or backpressure behavior
```

## MVP Boundary

The MVP is intentionally narrow. Anything outside this boundary should not delay Sprint 0.

MVP includes:

```text
detector runs under systemd with auto-restart
detector emits sighting_start/update/end events
events are written to JSONL
events are pushed live over UDS
FastAPI exposes status/events/detections
PWA shows live stream, status, detections, and read-only settings
Kanna uses simple PNG + CSS animation
```

MVP excludes:

```text
voice mode
HLS/WebRTC
mobile push notifications
Live2D / Spine
frontend write controls for detector thresholds/settings
privacy mode
multi-user account system
```

Decision:

```text
Settings screen exists in MVP, but is read-only.
Write APIs for detector settings move to hardening/future work.
```

## Event Model

There are three sighting event types:

```text
sighting_start
  New animal sighting begins.
  Companion enters ALERT / POINTING.
  Provisional snapshot is available immediately.

sighting_update
  Existing sighting moved or confidence changed.
  Companion tracks updated bbox.
  Emitted at a throttled interval, for example every 500 ms.

sighting_end
  Sighting was lost for timeout period, for example 2 seconds.
  Companion returns to IDLE.
  Final best-frame snapshot is committed.
```

Example JSON:

```json
{
  "schema_version": 1,
  "detector_id": "garden-pi-01",
  "source": "hailo_detector",
  "event_type": "sighting_start",
  "sighting_id": "20260427-170312-deer-001",
  "timestamp": "2026-04-27T17:03:12-04:00",
  "label": "deer",
  "confidence": 0.83,
  "priority": "routine",
  "frame_w": 1536,
  "frame_h": 864,
  "box": {
    "x1": 408,
    "y1": 137,
    "x2": 592,
    "y2": 360
  },
  "snapshot": "detections/2026-04-27/20260427-170312-deer-001.jpg"
}
```

Required event fields:

```text
schema_version
detector_id
source
event_type
sighting_id
timestamp
label
confidence
priority
frame_w
frame_h
box
snapshot
```

Optional future fields:

```text
track_age_ms
update_index
pir_state
temperature_f
fps
hailo_latency_ms
```

Default values:

```text
detector_id = garden-pi-01
source = hailo_detector
```

## Priority Classes

Priority must be config-driven, not hardcoded in frontend.

Initial default:

```yaml
priority_classes:
  routine:
    - deer
    - squirrel
    - turkey
    - possum
  priority:
    - raccoon
    - fox
    - skunk
    - coyote
  critical:
    - bear
```

Frontend behavior:

```text
routine  -> cute pointing reaction
priority -> urgent pointing reaction
critical -> loud alert / notification path
```

## Snapshot Policy

Goal: give the live UI a frame immediately, but keep the best frame for history.

```text
sighting_start:
  write provisional snapshot immediately

during sighting:
  keep best-confidence frame candidate in memory

sighting_end:
  write final best snapshot to temp path
  close file
  atomic rename to final path
```

Storage layout:

```text
detections/
  2026-04-27/
    events.jsonl
    20260427-170312-deer-001.jpg
    20260427-171044-raccoon-002.jpg
```

Later hardening:

```text
daily JSONL rotation
30 day retention
disk quota
per-class quota if needed
```

## Sprint 0: Event Backbone + Resilience

Duration target: 2-3 days.

This sprint prepares the detector to become a reliable app data source without moving app logic into C++.

Sprint 0 scope is closed:

```text
1. detection_event.hpp
2. sighting_tracker.hpp/.cpp
3. event_sink.hpp/.cpp
4. minimal hailo_detector.cpp wire-in
5. hailo-detector.service
6. README / roadmap MVP boundary
```

Explicitly not Sprint 0:

```text
FastAPI broker
PWA screens
Kanna UI
Haiku assistant
privacy mode
disk retention automation
frontend setting writes
```

### 1. `detection_event.hpp`

Responsibility:

```text
versioned event schema
fixed-size / simple structs
single serialize function
minimal allocation in hot path
```

Serialization decisions:

```text
no JSON library in Sprint 0
manual std::ostringstream serializer
json_escape helper is required
timestamp format is ISO 8601 local time with timezone offset
sighting_id is a string field generated by sighting_tracker
to_json is called by event_sink writer thread, not by inference loop
```

Expected pieces:

```text
enum class DetectionEventType
struct DetectionBox
struct DetectionEvent
std::string to_json(const DetectionEvent&)
std::string current_iso8601_local_timestamp()
```

Required fields:

```text
schema_version
detector_id
source
event_type
sighting_id
timestamp
label
confidence
priority
frame_w
frame_h
box
snapshot
```

Notes:

- JSON generation happens only when writing event out.
- Keep serializer local and predictable.
- Avoid bringing large JSON dependencies into the detector unless the code gets unsafe.

### 2. `sighting_tracker.hpp` / `sighting_tracker.cpp`

Responsibility:

```text
collapse raw detections into sighting lifecycle events
match detections by class + IoU
emit start/update/end
track best frame confidence
throttle update events
```

Initial constants:

```text
TRACK_MATCH_IOU_THRESH = 0.30
TRACK_UPDATE_INTERVAL_MS = 500
TRACK_LOST_TIMEOUT_MS = 2000
```

Per frame behavior:

```text
for each detection:
  if matching active track:
    update track
    maybe emit sighting_update
  else:
    create track
    emit sighting_start

for each active track not seen recently:
  if lost timeout elapsed:
    emit sighting_end
```

### 3. `event_sink.hpp` / `event_sink.cpp`

Responsibility:

```text
background writer thread
JSONL durable append
UDS live push
drop-on-full UDS behavior
telemetry counters
```

Detector loop behavior:

```text
tracker returns events
detector calls event_sink.enqueue(event)
detector immediately returns to inference work
```

Writer thread behavior:

```text
append event to JSONL
try push event over UDS
if UDS unavailable, continue
if UDS queue full, increment drop counter
```

Telemetry:

```text
event_queue_depth
uds_connected
uds_drop_count
jsonl_write_error_count
events_written_count
```

Important invariant:

```text
UDS may drop.
JSONL should not drop during normal operation.
Detector loop should not block.
```

### 4. `hailo_detector.cpp` Integration

Minimal integration points:

```text
create EventSink after RuntimeConfig is loaded
create SightingTracker before inference loop
after run_inference_once(), pass detections + rendered/source frame to tracker
enqueue tracker events into EventSink
keep current MJPEG, GPIO, PIR, deterrent behavior unchanged
```

Build integration note:

```text
The detector target must compile/link these new sources:
  sighting_tracker.cpp
  event_sink.cpp

If the Pi CMakeLists.txt is not in this workspace, add those sources on the Pi
before building hailo_detector.
```

Do not move these into C++:

```text
AI assistant
PWA UI
auth
Kanna animation
HTTP JSON API beyond existing stream
```

### 5. `hailo-detector.service`

The system currently runs manually from terminal. Sprint 0 should add a systemd unit.

Draft file:

```text
hailo-detector.service
```

Draft contents:

```ini
[Unit]
Description=Wildlife Hailo Detector
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/home/ecet400/hailo_detector
ExecStart=/home/ecet400/hailo_detector/build/hailo_detector
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Final paths must match the Pi deployment directory.

Install on Pi:

```bash
sudo cp hailo-detector.service /etc/systemd/system/hailo-detector.service
sudo systemctl daemon-reload
sudo systemctl enable hailo-detector
sudo systemctl start hailo-detector
```

### 6. Sprint 0 Acceptance Tests

Local simulation build:

```bash
g++ -std=c++17 -pthread -I. \
  tools/simulate_sprint0.cpp \
  sighting_tracker.cpp \
  event_sink.cpp \
  -o simulate_sprint0
```

Local simulation run:

```bash
./simulate_sprint0
```

Expected:

```text
stdout shows sighting_start, sighting_update, sighting_end
sim_detections/YYYY-MM-DD/events.jsonl is created
each JSONL line is valid JSON
events share one sighting_id across the simulated deer lifecycle
```

UDS listener test:

```bash
rm -f /tmp/det.sock
nc -lU /tmp/det.sock
```

In another terminal:

```bash
sudo systemctl start hailo-detector
```

Expected:

```text
walk in front of camera
nc receives sighting_start
nc receives throttled sighting_update
leave frame
nc receives sighting_end
```

Restart test:

```bash
sudo systemctl kill -s KILL hailo-detector
```

Expected:

```text
systemd restarts detector in about 5 seconds
MJPEG stream returns
JSONL remains parseable
no corrupted partial final snapshot
```

JSONL test:

```bash
tail -n 5 detections/$(date +%F)/events.jsonl
```

Expected:

```text
each line is valid JSON
events include start/update/end lifecycle
sighting_id stays consistent across one animal sighting
```

## Sprint 1: FastAPI Broker

Duration target: 2 days.

Files/directories:

```text
api_server/
  app.py
  config.yaml
  event_store.py
  uds_listener.py
  status.py
```

Responsibilities:

```text
listen on /tmp/det.sock
load last 100 JSONL events at startup
maintain ring buffer of last 1000 events
broadcast live events over SSE
serve detection history REST API
serve status REST API
later serve frontend static build
```

Initial endpoints:

```text
GET /api/status
GET /api/detections?limit=50&cursor=...
GET /api/events
GET /api/events?since=<timestamp>
GET /api/frames/{date}/{filename}
POST /api/dev/emit    # debug mode only
```

SSE behavior:

```text
client connects
optional since timestamp replays recent events
then receives live UDS events
auto reconnect supported
```

Status fields:

```text
detector_alive
stream_alive
last_event_ts
last_event_age_seconds
last_detection_label
fps
hailo_latency_p50_ms
hailo_latency_p95_ms
disk_usage_percent
event_queue_depth
uds_drop_count
jsonl_write_error_count
camera_reconnect_count
```

Dev/demo mode:

```text
debug mode can emit fake sighting events into ring buffer + SSE
disabled unless backend config explicitly enables debug
used to develop PWA and Kanna without walking in front of the camera
```

Security/deploy:

```text
Tailscale will be installed.
Prefer binding FastAPI to tailscale0 IP, not 0.0.0.0.
Claude API key stays only on backend.
```

## Sprint 2: PWA MVP

Duration target: 3-4 days.

Files/directories:

```text
frontend/
  package.json
  src/
  public/manifest.webmanifest
  public/service-worker.js
```

Screens:

```text
Dashboard
Live Camera
Detections
Settings read-only in MVP
```

Connection state banner:

```text
show detector online/offline
show backend online/offline
show stream connected/disconnected
show last event age
```

Settings behavior:

```text
show current detector/backend settings
do not allow editing in MVP
show read-only lock affordance for each field
write controls move to future work
```

Live stream:

```text
MVP uses MJPEG img from C++ stream.
Add iOS Safari reconnect handling from day one.
```

Required iOS reconnect behavior:

```javascript
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    img.src = streamUrl + "?t=" + Date.now();
  }
});
```

Acceptance:

```text
Add to Home Screen works on iOS Safari
app opens without browser chrome
background -> foreground restores stream in under 2 seconds
SSE reconnect works after backend restart
offline shell shows last known state instead of blank screen
```

Offline behavior:

```text
service worker caches app shell
IndexedDB stores last 24h detection summary
when Pi unreachable, show last known status and detection time
```

BBox mapping helper:

```text
input: event frame_w/frame_h + box + rendered stream element rect
output: screen x/y target for Kanna pointing
must handle contain/letterbox offsets
```

## Sprint 3: Kanna Companion

Duration target: 2-3 days.

Initial asset strategy:

```text
Start with transparent PNG poses + CSS animations.
No animation engine for MVP.
No Live2D / Spine / canvas animation runtime at first.
Live2D can come later only if the simple version feels limiting.
```

Asset folder:

```text
frontend/public/kanna/
  idle.png
  point_left.png
  point_right.png
  jump.png
  walk_1.png
  walk_2.png
  alert.png
  sit.png
  fall.png
```

Asset rules:

```text
transparent background
consistent character scale
about 200x300 px source size
export optimized PNG/WebP variants later if needed
```

First asset source:

```text
AI-generated chibi poses are acceptable for MVP.
Replace with polished custom art later without changing code.
```

Implementation approach:

```text
React changes Kanna state and position.
CSS handles motion, breathing, jump, walk, and transitions.
No separate animation library.
```

Example component shape:

```jsx
<img
  src={`/kanna/${state}.png`}
  className={`kanna kanna-${state}`}
  style={{ left: `${x}px`, bottom: `${y}px` }}
  alt=""
/>
```

Example CSS:

```css
.kanna {
  position: absolute;
  width: 120px;
  transition: left 0.8s ease-out, bottom 0.8s ease-out;
  pointer-events: none;
}

.kanna-idle {
  animation: kanna-breathe 2s infinite;
}

.kanna-jump {
  animation: kanna-jump 0.6s ease-out;
}

@keyframes kanna-breathe {
  0%, 100% { transform: scale(1); }
  50% { transform: scale(1.03); }
}

@keyframes kanna-jump {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-30px); }
}
```

FSM:

```text
IDLE
  sighting_start -> ALERT

ALERT
  routine -> POINTING
  priority -> POINTING_URGENT
  critical -> ALERT_LOUD

POINTING / POINTING_URGENT / ALERT_LOUD
  sighting_update -> update target bbox
  sighting_end -> IDLE

IDLE
  random timer -> WALK

WALK
  sighting_start -> ALERT
  walk timeout -> IDLE
```

Detection reaction:

```text
on sighting_start:
  map bbox to screen coordinates
  choose side based on animal position
  move Kanna near bbox
  state = jump
  after jump animation, state = point_left or point_right

on sighting_update:
  recompute bbox target
  smoothly move Kanna with CSS transition

on sighting_end:
  state = idle
  move Kanna to default safe corner

on backend offline:
  optional sad pose in MVP if asset exists
```

Rules:

```text
event-driven for detection state
random timers only for cute idle/walk behavior
illegal transitions ignored/logged
last sighting class and bbox stored in FSM context
settings can enable/disable companion
```

Acceptance:

```text
on real deer detection, Kanna points to correct animal bbox
on sighting_update, pointing target moves smoothly
on sighting_end, Kanna returns to idle
priority animals use different reaction
iOS Safari stays smooth with CSS-only animation
```

MVP companion limit:

```text
required: new sighting jump + point
optional: backend offline sad pose
defer: no-activity sleep state, complex critical alert choreography
```

## Sprint 4: AI Assistant

Duration target: 2 days.

Backend endpoint:

```text
POST /api/ask
```

Prompt file:

```text
api_server/prompts/wildlife_assistant_system.md
```

Assistant safety policy:

```text
give practical, non-harmful, safety-first advice
do not suggest harming animals
for dangerous animals, recommend distance, securing food/trash, and contacting local wildlife services when appropriate
make clear that advice is general guidance, not emergency instruction
```

Context rule:

```text
Never send raw JSONL to Haiku.
Always send compressed summary.
```

Budget:

```text
system prompt: about 200 tokens
summary context: max 300 tokens
user question: about 50 tokens
answer: max 400 tokens
total target: under 950 tokens
```

Required helper:

```text
summarize_window(start, end) -> str
```

Acceptance:

```text
last_24h_summary output <= 300 tokens
if summary exceeds budget, backend raises/logs an error
"bugun ne oldu?" returns useful answer within about 3 seconds
API key is never exposed to frontend
```

Example summary:

```text
Last 24h: 8 sightings total. 5 deer, 2 raccoon, 1 fox.
Peak activity was 02:00-04:00. Last sighting was raccoon at 04:12.
No critical bear sightings.
```

## Sprint 5: Hardening

Continuous work after MVP.

Items:

```text
Tailscale-only bind
HTTPS through Tailscale serve or reverse proxy
daily JSONL rotation
30 day retention
disk quota
privacy mode if multi-user/family access becomes needed
frontend write controls for detector settings
status telemetry expansion
MJPEG health watchdog
optional HLS stream path
Telegram notification for critical classes
```

MJPEG vs HLS/WebRTC:

```text
MJPEG:
  low latency, current implementation, best for companion sync

HLS:
  better iOS battery/stability, but 2-6s latency

WebRTC:
  low latency, mobile friendly, but more complex
```

MVP decision:

```text
Keep MJPEG.
Add reconnect handling immediately.
Evaluate HLS later for passive viewing mode.
```

## Open Questions

1. Final Pi deployment path for `hailo-detector.service`.
2. Tailscale IP/interface binding details after installation.
3. Notification target for critical animals: Telegram, email, or later mobile push.
4. Whether `detector_id = garden-pi-01` should be changed before first deployment.

## Next Implementation Step

Start Sprint 0 in this order:

```text
1. detection_event.hpp
2. sighting_tracker.hpp/.cpp
3. event_sink.hpp/.cpp
4. hailo_detector.cpp integration
5. hailo-detector.service
6. acceptance tests on Pi
```

The first code change should be small and testable: create `detection_event.hpp`, define the event struct, and serialize one sample event into one JSON line.

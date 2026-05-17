# Frontend Documentation

## What the frontend is

The frontend is the product-facing layer of the project.

Its purpose is to make the detector understandable and usable from a phone or browser rather than only through logs, terminal output, or raw images.

This is why the frontend matters so much in the repo: it shows the jump from "embedded demo" to "user-facing product".

## Main files

- `frontend/index.html`
  Main shell markup.
- `frontend/app.js`
  UI state, rendering, mock mode, event handling, Kanna behavior, and integration logic.
- `frontend/styles.css`
  Visual styling for the shell and app screens.
- `frontend/service-worker.js`
  PWA/offline shell support.
- `frontend/manifest.webmanifest`
  Installable app metadata.
- `frontend/assets/kanna/`
  Production-facing companion animation and pose assets.
- `frontend/public/media/`
  Mock/demo detection images used in the frontend shell.

## Main frontend responsibilities

### 1. Present system status

The frontend surfaces whether the detector, stream, and backend appear available.

This helps the app feel operational rather than just decorative.

### 2. Show the live view

The frontend includes a live stream card and detection overlay behavior.

This is the operator’s visual window into what the system is currently seeing.

### 3. Show recent detections

The frontend turns backend event data into a readable history feed with:

- animal label
- confidence
- summary
- snapshot
- relative timing

### 4. Provide a product shell

The app is structured like a mobile-first PWA rather than a one-off static page.

That means it is intended to feel like a lightweight installed product, not only a development dashboard.

## Kanna companion

One of the most distinctive parts of the repo is the Kanna companion layer.

This is a UI/character layer that reacts to detection state.

It exists because the project is not trying to be a sterile industrial dashboard only. It is also trying to create a memorable product experience.

## Kanna behavior

The frontend includes:

- idle state
- jump state
- point left / point right states
- alert state
- walking behavior
- fall / recover style animations

This companion behavior reacts to detection data, not just to random timers.

That makes it part of the event-driven system design rather than just a visual gimmick.

## Mock mode

The frontend can operate in a backend-independent mode using mock/sample data.

This is important because:

- UI development can continue before full detector integration
- demos are easier to prepare
- behavior can be tested from sample media

The repo intentionally keeps this mode because it improves iteration speed.

## Event integration

When the backend is live, the frontend uses:

- `GET /api/status`
- `GET /api/detections`
- `GET /api/events`

The live event stream is particularly important because it lets the frontend respond immediately when sightings start or update.

## PWA intent

The frontend is built to behave like a mobile-first installable shell.

That means the project is already thinking beyond a developer page and toward:

- field access
- dashboard-at-a-glance behavior
- phone usability
- app-like presentation

## Why the frontend matters in this repo

Many hardware/CV projects stop at "the model works".

This frontend shows the additional engineering and product effort required to turn detector output into something a human actually wants to check and use.

That includes:

- app structure
- interaction design
- status signaling
- live data integration
- character-driven UX

## What this part of the repo shows

The frontend documents a lot of your product effort:

- not just detection
- not just backend plumbing
- but actual experience design

It is one of the clearest signals in the repository that the project aims to be a complete application experience, not only a low-level inference demo.

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .event_store import EventStore
from .status import build_status


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "api_server" / "config.yaml"


class AskPayload(BaseModel):
    question: str


class DevEmitPayload(BaseModel):
    label: str = "deer"


def load_config() -> dict[str, Any]:
    defaults = {
        "debug": True,
        "detector_id": "garden-pi-01",
        "event_history_limit": 1000,
        "seed_event_count": 100,
        "detections_root": "detections",
        "frontend_root": "frontend",
        "sample_media_root": "samples/media",
    }
    if not CONFIG_PATH.exists():
        return defaults
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    return {**defaults, **loaded}


config = load_config()
frontend_root = ROOT / config["frontend_root"]
sample_media_root = ROOT / config["sample_media_root"]
detections_root = ROOT / config["detections_root"]
store = EventStore(detections_root=detections_root, max_events=config["event_history_limit"])
store.seed_from_disk_or_demo(seed_event_count=config["seed_event_count"])

app = FastAPI(title="Wildwatch Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/status")
async def api_status(request: Request) -> dict[str, Any]:
    return build_status(store, detections_root, stream_url=str(request.base_url) + "stream.mjpg")


@app.get("/api/detections")
async def api_detections(limit: int = 50) -> dict[str, Any]:
    return {"items": store.detections(limit=limit)}


@app.get("/api/events")
async def api_events(since: str | None = None) -> StreamingResponse:
    async def event_stream() -> Any:
        replay = store.recent_events(since=since)
        for event in replay:
            yield _sse(event)

        queue = store.subscribe()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=20)
                    yield _sse(event)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
        finally:
            store.unsubscribe(queue)

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/api/dev/emit")
async def api_dev_emit(payload: DevEmitPayload) -> dict[str, Any]:
    if not config.get("debug", False):
        raise HTTPException(status_code=403, detail="Debug emit disabled.")

    event = _make_demo_event(payload.label)
    store.publish(event)
    return {"ok": True, "event": event}


@app.post("/api/ask")
async def api_ask(payload: AskPayload) -> dict[str, Any]:
    summary = store.summary()
    answer = (
        f"{summary} "
        f"For now this is the local backend summary layer, so the answer to "
        f"'{payload.question}' is generated from recent detections rather than Claude."
    )
    return {"answer": answer}


@app.get("/stream.mjpg")
async def stream_preview() -> RedirectResponse:
    latest = store.latest_detection()
    target = latest["image"] if latest else "/media/20260417_202846_829_frame4190_deer.jpg"
    return RedirectResponse(url=target)


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse((frontend_root / "index.html").read_text(encoding="utf-8"))


if sample_media_root.exists():
    app.mount("/media", StaticFiles(directory=sample_media_root), name="media")
if frontend_root.exists():
    app.mount("/", StaticFiles(directory=frontend_root, html=True), name="frontend")


def _make_demo_event(label: str) -> dict[str, Any]:
    label = label.lower().strip() or "deer"
    priority = "routine"
    image = "/media/20260417_202846_829_frame4190_deer.jpg"
    box = {"x1": 420, "y1": 140, "x2": 618, "y2": 366}
    confidence = 0.84
    summary = "Browsing along the tree line."

    if label == "raccoon":
        priority = "priority"
        image = "/media/20260417_200655_359_frame3384_raccoon.jpg"
        box = {"x1": 1000, "y1": 270, "x2": 1290, "y2": 522}
        confidence = 0.79
        summary = "Short pass near the bin line."
    elif label == "bear":
        priority = "critical"
        image = "/media/20260417_203117_022_frame2677_bear.jpg"
        box = {"x1": 630, "y1": 180, "x2": 1126, "y2": 660}
        confidence = 0.92
        summary = "Critical class near the back gate."

    stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    sighting_id = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{label}-{uuid4().hex[:3]}"
    return {
        "schema_version": 1,
        "detector_id": config["detector_id"],
        "source": "backend_demo",
        "event_type": "sighting_start",
        "sighting_id": sighting_id,
        "timestamp": stamp,
        "label": label,
        "confidence": confidence,
        "priority": priority,
        "frame_w": 1536,
        "frame_h": 864,
        "box": box,
        "snapshot": image,
        "summary": summary,
    }


def _sse(event: dict[str, Any]) -> str:
    import json

    payload = json.dumps(event)
    return f"event: {event.get('event_type', 'message')}\ndata: {payload}\n\n"

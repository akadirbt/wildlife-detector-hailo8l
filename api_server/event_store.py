from __future__ import annotations

import asyncio
import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from .demo_data import seed_events


class EventStore:
    def __init__(self, detections_root: Path, max_events: int = 1000) -> None:
        self.detections_root = detections_root
        self.events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self.subscribers: set[asyncio.Queue] = set()

    def seed_from_disk_or_demo(self, seed_event_count: int = 100) -> None:
        loaded = self._load_from_jsonl(limit=seed_event_count)
        if loaded:
            for event in loaded:
                self.events.append(event)
            return

        for event in seed_events():
            self.events.append(event)

    def _load_from_jsonl(self, limit: int) -> list[dict[str, Any]]:
        if not self.detections_root.exists():
            return []

        candidates = sorted(
            self.detections_root.glob("*/events.jsonl"),
            key=lambda path: path.parent.name,
            reverse=True,
        )
        loaded: list[dict[str, Any]] = []
        for path in candidates:
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue
                try:
                    loaded.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(loaded) >= limit:
                    break
            if len(loaded) >= limit:
                break
        loaded.reverse()
        return loaded

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        for queue in list(self.subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                continue

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self.subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self.subscribers.discard(queue)

    def recent_events(self, since: str | None = None) -> list[dict[str, Any]]:
        if not since:
            return list(self.events)

        try:
            since_dt = datetime.fromisoformat(since)
        except ValueError:
            return list(self.events)

        output: list[dict[str, Any]] = []
        for event in self.events:
            timestamp = event.get("timestamp")
            if not timestamp:
                continue
            try:
                event_dt = datetime.fromisoformat(timestamp)
            except ValueError:
                continue
            if event_dt >= since_dt:
                output.append(event)
        return output

    def detections(self, limit: int = 50) -> list[dict[str, Any]]:
        latest_by_sighting: dict[str, dict[str, Any]] = {}
        for event in self.events:
            sighting_id = event.get("sighting_id")
            if not sighting_id:
                continue
            latest_by_sighting[sighting_id] = event

        records = sorted(
            latest_by_sighting.values(),
            key=lambda item: item.get("timestamp", ""),
            reverse=True,
        )
        return [self._detection_payload(item) for item in records[:limit]]

    def latest_detection(self) -> dict[str, Any] | None:
        detections = self.detections(limit=1)
        return detections[0] if detections else None

    def summary(self) -> str:
        detections = self.detections(limit=200)
        if not detections:
            return "No sightings have been recorded yet."

        counts: dict[str, int] = {}
        for item in detections:
            counts[item["label"]] = counts.get(item["label"], 0) + 1

        ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
        parts = [f"{count} {label}" for label, count in ordered[:4]]
        latest = detections[0]
        return (
            f"Recent sightings: {', '.join(parts)}. "
            f"Latest was {latest['label']} at {latest['time']}."
        )

    def _detection_payload(self, event: dict[str, Any]) -> dict[str, Any]:
        timestamp = event.get("timestamp", "")
        return {
            "sighting_id": event.get("sighting_id"),
            "label": event.get("label"),
            "confidence": event.get("confidence"),
            "priority": event.get("priority"),
            "time": self._pretty_time(timestamp),
            "ago": self._ago(timestamp),
            "image": event.get("snapshot"),
            "summary": event.get("summary") or f"{event.get('label', 'animal')} sighting recorded.",
            "box": event.get("box", {}),
            "frame_w": event.get("frame_w", 0),
            "frame_h": event.get("frame_h", 0),
            "timestamp": timestamp,
            "event_type": event.get("event_type"),
        }

    @staticmethod
    def _pretty_time(timestamp: str) -> str:
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime("%I:%M %p").lstrip("0")
        except ValueError:
            return timestamp

    @staticmethod
    def _ago(timestamp: str) -> str:
        try:
            dt = datetime.fromisoformat(timestamp)
            delta = datetime.now(dt.tzinfo) - dt
        except ValueError:
            return "unknown"

        minutes = max(int(delta.total_seconds() // 60), 0)
        if minutes < 1:
            return "just now"
        if minutes < 60:
            return f"{minutes}m ago"
        hours = minutes // 60
        return f"{hours}h ago"

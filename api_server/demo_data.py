from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timedelta, timezone


def _ts(minutes_ago: int) -> str:
    now = datetime.now().astimezone() - timedelta(minutes=minutes_ago)
    return now.isoformat(timespec="seconds")


DEMO_DETECTIONS = [
    {
        "schema_version": 1,
        "detector_id": "garden-pi-01",
        "source": "hailo_detector",
        "event_type": "sighting_start",
        "sighting_id": "20260427-214000-raccoon-001",
        "timestamp": _ts(2),
        "label": "raccoon",
        "confidence": 0.78,
        "priority": "priority",
        "frame_w": 1536,
        "frame_h": 864,
        "box": {"x1": 1020, "y1": 262, "x2": 1310, "y2": 518},
        "snapshot": "/media/20260417_200655_359_frame3384_raccoon.jpg",
        "summary": "North fence line, short pass near the trash bins.",
    },
    {
        "schema_version": 1,
        "detector_id": "garden-pi-01",
        "source": "hailo_detector",
        "event_type": "sighting_start",
        "sighting_id": "20260427-213600-deer-002",
        "timestamp": _ts(6),
        "label": "deer",
        "confidence": 0.83,
        "priority": "routine",
        "frame_w": 1536,
        "frame_h": 864,
        "box": {"x1": 408, "y1": 137, "x2": 592, "y2": 360},
        "snapshot": "/media/20260417_202846_829_frame4190_deer.jpg",
        "summary": "Stayed near the tree line for about 12 seconds.",
    },
    {
        "schema_version": 1,
        "detector_id": "garden-pi-01",
        "source": "hailo_detector",
        "event_type": "sighting_start",
        "sighting_id": "20260427-205800-bear-003",
        "timestamp": _ts(44),
        "label": "bear",
        "confidence": 0.91,
        "priority": "critical",
        "frame_w": 1536,
        "frame_h": 864,
        "box": {"x1": 640, "y1": 180, "x2": 1140, "y2": 660},
        "snapshot": "/media/20260417_203117_022_frame2677_bear.jpg",
        "summary": "Large silhouette near the back gate. Critical class.",
    },
]


def seed_events() -> list[dict]:
    return [deepcopy(event) for event in DEMO_DETECTIONS]

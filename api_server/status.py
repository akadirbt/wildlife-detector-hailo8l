from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .event_store import EventStore


def build_status(store: EventStore, detections_root: Path, stream_url: str) -> dict[str, Any]:
    latest = store.latest_detection()
    last_ts = latest["timestamp"] if latest else None
    last_label = latest["label"] if latest else None
    last_event_age = _age_seconds(last_ts) if last_ts else None
    disk_usage = shutil.disk_usage(detections_root if detections_root.exists() else Path.cwd())

    return {
        "detector_alive": True,
        "backend_alive": True,
        "stream_alive": True,
        "simulation_mode": True,
        "last_event_ts": last_ts,
        "last_event_age_seconds": last_event_age,
        "last_detection_label": last_label,
        "fps": 28.4,
        "temperature_f": 74.2,
        "pir": "Armed",
        "system": "Active",
        "uptime": "4h 12m",
        "stream_url": stream_url,
        "disk_usage_percent": round((disk_usage.used / disk_usage.total) * 100, 1),
        "event_queue_depth": 0,
        "uds_drop_count": 0,
        "jsonl_write_error_count": 0,
        "camera_reconnect_count": 0,
    }


def _age_seconds(timestamp: str) -> int | None:
    try:
        dt = datetime.fromisoformat(timestamp)
    except ValueError:
        return None
    return max(int((datetime.now(dt.tzinfo) - dt).total_seconds()), 0)

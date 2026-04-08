"""Simple JSONL logger for per-step and per-wall-clock metrics."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .utils import to_serializable


class JsonlLogger:
    """Logger that records training events as line-delimited JSON."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._start_time = time.perf_counter()

    @property
    def elapsed(self) -> float:
        """Total elapsed wall-clock seconds since logger creation."""
        return time.perf_counter() - self._start_time

    def log(self, record: dict[str, Any]) -> None:
        payload = to_serializable(record)
        payload["wall_clock_sec"] = self.elapsed
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

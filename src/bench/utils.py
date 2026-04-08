"""Utility helpers for reproducibility and split generation."""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set python, numpy and torch seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def family_seed(base_seed: int, family: str) -> int:
    """Derive deterministic task-family-specific seed.

    This ensures all tasks in the same family share split behavior while
    keeping reproducibility from a global base seed.
    """
    digest = hashlib.sha256(f"{base_seed}:{family}".encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + base_seed) % (2**31 - 1)


def to_serializable(data: Any) -> Any:
    """Convert nested config/metrics objects into JSON-serializable values."""
    if is_dataclass(data):
        return {k: to_serializable(v) for k, v in asdict(data).items()}
    if isinstance(data, dict):
        return {k: to_serializable(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [to_serializable(v) for v in data]
    if isinstance(data, Path):
        return str(data)
    if isinstance(data, (np.float32, np.float64, np.int32, np.int64)):
        return data.item()
    return data

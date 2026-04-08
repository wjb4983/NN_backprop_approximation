"""Metric calculations for optimizer benchmarking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ThresholdResult:
    steps_to_threshold: int | None
    wall_clock_to_threshold: float | None


def threshold_metrics(
    values: list[float],
    steps: list[int],
    wall_secs: list[float],
    threshold: float,
    mode: str,
) -> ThresholdResult:
    """Compute first step/time at which metric crosses threshold."""
    for v, s, w in zip(values, steps, wall_secs):
        met = (v >= threshold) if mode == "max" else (v <= threshold)
        if met:
            return ThresholdResult(steps_to_threshold=s, wall_clock_to_threshold=w)
    return ThresholdResult(steps_to_threshold=None, wall_clock_to_threshold=None)


def auc_early(values: list[float], steps: list[int], early_window_steps: int) -> float:
    """Area under learning curve up to early_window_steps using trapezoid rule."""
    if not values:
        return 0.0
    xs, ys = [], []
    for s, v in zip(steps, values):
        if s <= early_window_steps:
            xs.append(s)
            ys.append(v)
    if len(xs) < 2:
        return float(ys[0]) if ys else 0.0
    return float(np.trapz(ys, xs))


def final_metric_at_budget(values: list[float]) -> float | None:
    """Return last available metric value within the run budget."""
    return values[-1] if values else None

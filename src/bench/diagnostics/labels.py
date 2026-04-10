"""Future-window label generation for diagnostics tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class LabelConfig:
    """Thresholds controlling target generation from trajectories."""

    horizon_steps: int = 5
    min_delta_for_progress: float = 1e-3
    noise_multiplier: float = 1.5
    instability_loss_jump: float = 0.5
    hp_mismatch_stall_windows: int = 3


@dataclass
class GeneratedLabels:
    step: int
    health_now: int
    stall_next_h: int
    meaningful_progress: int
    instability_risk: int
    hp_mismatch: int


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def generate_labels(records: list[dict[str, Any]], cfg: LabelConfig) -> list[GeneratedLabels]:
    """Generate labels for each record based on future-window behavior.

    Label semantics:
      - stall_next_h: no meaningful loss decrease over next horizon.
      - meaningful_progress: future decrease exceeds local noise-adjusted threshold.
      - instability_risk: upcoming fallback activity or abrupt loss jump.
      - health_now: currently healthy (non-stalled and not instability-prone).
      - hp_mismatch: repeated stall-like pattern with instability hints.
    """
    if not records:
        return []

    losses = [
        _safe_float((rec.get("val") or {}).get("loss"), _safe_float((rec.get("train") or {}).get("loss")))
        for rec in records
    ]
    fallback_events = [_safe_float((rec.get("optimizer_diagnostics") or {}).get("fallback_events")) for rec in records]

    out: list[GeneratedLabels] = []
    stall_rolling = 0
    n = len(records)

    for i, rec in enumerate(records):
        step = int(rec.get("step", i))
        cur_loss = losses[i]
        future_end = min(i + cfg.horizon_steps, n - 1)
        future_losses = losses[i + 1 : future_end + 1]

        if not future_losses:
            # Final window has no future context; mark as uncertain-negative by construction.
            meaningful_progress = 0
            stall_next = 0
            instability = 0
        else:
            best_future = float(np.min(np.asarray(future_losses, dtype=np.float64)))
            abs_improvement = cur_loss - best_future

            recent = losses[max(0, i - cfg.horizon_steps + 1) : i + 1]
            local_noise = float(np.std(np.asarray(recent, dtype=np.float64))) if len(recent) >= 2 else 0.0
            required_improvement = max(cfg.min_delta_for_progress, cfg.noise_multiplier * local_noise)

            meaningful_progress = int(abs_improvement > required_improvement)
            stall_next = int(abs_improvement <= required_improvement)

            future_fallback = max(fallback_events[i + 1 : future_end + 1]) if i + 1 <= future_end else 0.0
            future_loss_jump = float(np.max(np.asarray(future_losses, dtype=np.float64)) - cur_loss)
            instability = int((future_fallback > fallback_events[i]) or (future_loss_jump >= cfg.instability_loss_jump))

        stall_rolling = stall_rolling + 1 if stall_next else 0

        hp_mismatch = int(
            stall_rolling >= cfg.hp_mismatch_stall_windows
            and (instability or (_safe_float((rec.get("optimizer_diagnostics") or {}).get("trust_ratio_ema"), 1.0) > 2.5))
        )

        health_now = int((stall_next == 0) and (instability == 0))

        out.append(
            GeneratedLabels(
                step=step,
                health_now=health_now,
                stall_next_h=stall_next,
                meaningful_progress=meaningful_progress,
                instability_risk=instability,
                hp_mismatch=hp_mismatch,
            )
        )

    return out

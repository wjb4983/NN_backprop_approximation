"""Feature pipeline for training diagnostics.

We reuse optimizer diagnostic signals where possible so diagnostics and optimizer
policy can share representation assumptions without coupling model parameters.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FeatureSnapshot:
    """One featurized training step ready for downstream labeling/modeling."""

    step: int
    feature_names: list[str]
    values: list[float]


class DiagnosticsFeaturePipeline:
    """Build stable, low-dimensional diagnostics features from run logs.

    Features intentionally mix instantaneous values and short-window trends so
    event prediction (stall/instability) can be made with lead time.
    """

    def __init__(self, history_window: int = 8) -> None:
        self.history_window = max(int(history_window), 3)
        self._loss_hist: deque[float] = deque(maxlen=self.history_window)
        self._acc_hist: deque[float] = deque(maxlen=self.history_window)

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            casted = float(value)
        except (TypeError, ValueError):
            return default
        return casted if math.isfinite(casted) else default

    @staticmethod
    def _slope(values: deque[float]) -> float:
        if len(values) < 2:
            return 0.0
        xs = np.arange(len(values), dtype=np.float64)
        ys = np.asarray(values, dtype=np.float64)
        x_mean = float(xs.mean())
        y_mean = float(ys.mean())
        denom = float(np.sum((xs - x_mean) ** 2))
        if denom <= 1e-12:
            return 0.0
        return float(np.sum((xs - x_mean) * (ys - y_mean)) / denom)

    def build_from_record(self, record: dict[str, Any]) -> FeatureSnapshot:
        """Convert one training/eval log record into a feature vector."""
        step = int(record.get("step", 0))

        train = record.get("train", {})
        val = record.get("val", {})
        step_profile = record.get("step_profile", {})
        opt_diag = record.get("optimizer_diagnostics", {})

        train_loss = self._safe_float(train.get("loss"))
        val_loss = self._safe_float(val.get("loss"), train_loss)
        val_acc = self._safe_float(val.get("accuracy"))

        self._loss_hist.append(val_loss)
        self._acc_hist.append(val_acc)

        loss_slope = self._slope(self._loss_hist)
        acc_slope = self._slope(self._acc_hist)
        loss_std = float(np.std(np.asarray(self._loss_hist, dtype=np.float64))) if self._loss_hist else 0.0

        optimizer_share = self._safe_float(step_profile.get("optimizer_share"))
        optimizer_ms = self._safe_float(step_profile.get("optimizer_ms"))

        fallback_events = self._safe_float(opt_diag.get("fallback_events"))
        fallback_active = self._safe_float(opt_diag.get("fallback_active"))
        trust_ratio_ema = self._safe_float(opt_diag.get("trust_ratio_ema"), 1.0)
        control_drift_ema = self._safe_float(opt_diag.get("control_drift_ema"))

        feature_names = [
            "train_loss",
            "val_loss",
            "val_accuracy",
            "loss_slope",
            "acc_slope",
            "loss_std",
            "optimizer_share",
            "optimizer_ms",
            "fallback_events",
            "fallback_active",
            "trust_ratio_ema",
            "control_drift_ema",
            "history_len_frac",
        ]
        values = [
            train_loss,
            val_loss,
            val_acc,
            loss_slope,
            acc_slope,
            loss_std,
            optimizer_share,
            optimizer_ms,
            fallback_events,
            fallback_active,
            trust_ratio_ema,
            control_drift_ema,
            min(len(self._loss_hist) / float(self.history_window), 1.0),
        ]
        return FeatureSnapshot(step=step, feature_names=feature_names, values=values)

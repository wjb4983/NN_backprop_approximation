"""Evaluation metrics for diagnostics predictions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BinaryMetrics:
    auroc: float
    auprc: float
    ece: float


def _binary_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-scores)
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    tps = np.cumsum(sorted_labels)
    fps = np.cumsum(1 - sorted_labels)
    positives = max(np.sum(sorted_labels), 1)
    negatives = max(len(sorted_labels) - np.sum(sorted_labels), 1)

    tpr = tps / positives
    fpr = fps / negatives
    precision = tps / np.maximum(tps + fps, 1)
    recall = tpr

    return fpr, tpr, np.concatenate(([1.0], precision)), np.concatenate(([0.0], recall))


def _auc(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2:
        return 0.5
    return float(np.trapz(y, x))


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probs >= lo) & (probs < hi if i < bins - 1 else probs <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(probs[mask]))
        acc = float(np.mean(labels[mask]))
        ece += (np.sum(mask) / max(n, 1)) * abs(acc - conf)
    return float(ece)


def binary_metrics(probs: np.ndarray, labels: np.ndarray) -> BinaryMetrics:
    labels = labels.astype(np.int64)
    probs = probs.astype(np.float64)

    unique = np.unique(labels)
    if len(unique) < 2:
        # Degenerate label sets are common in tiny smoke runs.
        return BinaryMetrics(auroc=0.5, auprc=float(np.mean(labels)), ece=expected_calibration_error(probs, labels))

    fpr, tpr, precision, recall = _binary_curve(probs, labels)
    return BinaryMetrics(
        auroc=_auc(fpr, tpr),
        auprc=_auc(recall, precision),
        ece=expected_calibration_error(probs, labels),
    )


def lead_time_score(stall_probs: np.ndarray, stall_labels: np.ndarray, steps: np.ndarray, threshold: float = 0.6) -> float:
    """Average number of steps between first warning and true stall event."""
    event_steps = steps[stall_labels.astype(bool)]
    if len(event_steps) == 0:
        return 0.0
    warning_steps = steps[stall_probs >= threshold]
    if len(warning_steps) == 0:
        return 0.0

    leads: list[float] = []
    for ev in event_steps:
        prior = warning_steps[warning_steps <= ev]
        if len(prior) == 0:
            continue
        leads.append(float(ev - prior[-1]))
    return float(np.mean(leads)) if leads else 0.0


def decision_utility(
    stall_probs: np.ndarray,
    instability_probs: np.ndarray,
    hp_mismatch_probs: np.ndarray,
    labels: np.ndarray,
    stop_threshold: float = 0.8,
    tune_threshold: float = 0.65,
) -> float:
    """Simple practical utility score for stop/restart/tune triage."""
    stall_true = labels[:, 1]
    instability_true = labels[:, 3]
    hp_true = labels[:, 4]

    utility = 0.0
    for i in range(len(stall_probs)):
        if instability_probs[i] >= stop_threshold:
            utility += 2.0 if instability_true[i] else -1.0
        elif hp_mismatch_probs[i] >= tune_threshold:
            utility += 1.5 if hp_true[i] else -0.5
        elif stall_probs[i] >= tune_threshold:
            utility += 1.0 if stall_true[i] else -0.5
        else:
            utility += 0.1
    return float(utility / max(len(stall_probs), 1))

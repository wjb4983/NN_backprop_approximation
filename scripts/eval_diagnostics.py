#!/usr/bin/env python3
"""Evaluate diagnostics predictions and emit report-ready JSON metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from bench.diagnostics.eval_metrics import binary_metrics, decision_utility, lead_time_score
from bench.diagnostics.modeling import DiagnosticsMLP


def _load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    feat_cols = [c for c in rows[0] if c.startswith("feat_")]
    label_cols = [
        "label_health_now",
        "label_stall_next_h",
        "label_meaningful_progress",
        "label_instability_risk",
        "label_hp_mismatch",
    ]
    step = np.asarray([int(float(r.get("step", i))) for i, r in enumerate(rows)], dtype=np.int64)
    x = np.asarray([[float(r[c]) for c in feat_cols] for r in rows], dtype=np.float32)
    y = np.asarray([[float(r[c]) for c in label_cols] for r in rows], dtype=np.float32)
    return feat_cols, label_cols, step, x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate diagnostics model")
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="outputs/diagnostics_eval.json")
    args = parser.parse_args()

    feat_cols, label_cols, steps, x_np, y_np = _load_dataset(Path(args.dataset_csv))
    payload = torch.load(args.checkpoint, map_location="cpu")

    model = DiagnosticsMLP(input_dim=int(payload["input_dim"]))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    x = torch.tensor(x_np)
    with torch.no_grad():
        probs = model.predict_proba(x).numpy()

    per_task = {}
    for i, label in enumerate(label_cols):
        m = binary_metrics(probs[:, i], y_np[:, i])
        per_task[label] = {"auroc": m.auroc, "auprc": m.auprc, "ece": m.ece}

    metrics = {
        "per_task": per_task,
        "lead_time_steps": lead_time_score(probs[:, 1], y_np[:, 1], steps),
        "decision_utility": decision_utility(probs[:, 1], probs[:, 3], probs[:, 4], y_np),
        "n_rows": int(len(x_np)),
        "n_features": int(len(feat_cols)),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

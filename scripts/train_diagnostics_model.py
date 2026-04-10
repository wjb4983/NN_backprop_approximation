#!/usr/bin/env python3
"""Train diagnostics multi-task model from generated CSV labels."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

from bench.diagnostics.modeling import DiagnosticsMLP


def _load_dataset(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    feature_cols = [c for c in rows[0].keys() if c.startswith("feat_")]
    label_cols = [
        "label_health_now",
        "label_stall_next_h",
        "label_meaningful_progress",
        "label_instability_risk",
        "label_hp_mismatch",
    ]

    x = np.asarray([[float(r[c]) for c in feature_cols] for r in rows], dtype=np.float32)
    y = np.asarray([[float(r[c]) for c in label_cols] for r in rows], dtype=np.float32)
    return feature_cols, label_cols, x, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Train diagnostics model")
    parser.add_argument("--dataset-csv", required=True)
    parser.add_argument("--output-ckpt", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feature_cols, label_cols, x_np, y_np = _load_dataset(Path(args.dataset_csv))
    x = torch.tensor(x_np)
    y = torch.tensor(y_np)

    model = DiagnosticsMLP(input_dim=x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    n = x.shape[0]
    indices = np.arange(n)

    model.train()
    for epoch in range(args.epochs):
        np.random.shuffle(indices)
        for start in range(0, n, args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            xb, yb = x[batch_idx], y[batch_idx]
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % max(args.epochs // 5, 1) == 0:
            print(f"epoch={epoch+1} loss={float(loss.item()):.6f}")

    out = Path(args.output_ckpt)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "input_dim": int(x.shape[1]),
            "feature_names": feature_cols,
            "label_names": label_cols,
            "model_state_dict": model.state_dict(),
        },
        out,
    )
    print(f"saved checkpoint to {out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate diagnostics training dataset from metrics JSONL trajectories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from bench.diagnostics.feature_pipeline import DiagnosticsFeaturePipeline
from bench.diagnostics.labels import LabelConfig, generate_labels


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate diagnostics labels from training logs")
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--horizon-steps", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--noise-multiplier", type=float, default=1.5)
    parser.add_argument("--instability-loss-jump", type=float, default=0.5)
    parser.add_argument("--hp-mismatch-stall-windows", type=int, default=3)
    args = parser.parse_args()

    records = _read_jsonl(Path(args.metrics_jsonl))
    labels = generate_labels(
        records,
        LabelConfig(
            horizon_steps=args.horizon_steps,
            min_delta_for_progress=args.min_delta,
            noise_multiplier=args.noise_multiplier,
            instability_loss_jump=args.instability_loss_jump,
            hp_mismatch_stall_windows=args.hp_mismatch_stall_windows,
        ),
    )

    pipeline = DiagnosticsFeaturePipeline()
    rows = []
    for rec, lbl in zip(records, labels):
        snap = pipeline.build_from_record(rec)
        row = {"step": snap.step}
        for n, v in zip(snap.feature_names, snap.values):
            row[f"feat_{n}"] = v
        row.update(
            {
                "label_health_now": lbl.health_now,
                "label_stall_next_h": lbl.stall_next_h,
                "label_meaningful_progress": lbl.meaningful_progress,
                "label_instability_risk": lbl.instability_risk,
                "label_hp_mismatch": lbl.hp_mismatch,
            }
        )
        rows.append(row)

    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows generated from metrics JSONL")

    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()

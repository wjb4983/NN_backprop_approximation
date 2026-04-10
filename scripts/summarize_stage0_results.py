#!/usr/bin/env python3
"""Aggregate Stage 0 summary.json artifacts into a markdown report table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize Stage 0 benchmark outputs")
    parser.add_argument("outputs_dir", type=Path, help="Root outputs directory containing run subdirectories")
    parser.add_argument(
        "--write-md",
        type=Path,
        default=Path("docs/reports/stage0_benchmark_results_template.md"),
        help="Where to write markdown summary",
    )
    return parser.parse_args()


def load_stage0_rows(outputs_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for summary_path in sorted(outputs_dir.glob("stage0_*/summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        run_cfg = cfg.get("run", {})
        task_cfg = cfg.get("task", {})
        opt_cfg = cfg.get("optimizer", {})
        metrics = payload.get("metrics", {})

        rows.append(
            {
                "experiment": run_cfg.get("experiment_name", summary_path.parent.name),
                "seed": run_cfg.get("seed"),
                "task": task_cfg.get("name"),
                "optimizer": opt_cfg.get("name"),
                "final_metric": metrics.get("final_metric_at_budget"),
                "steps_to_threshold": metrics.get("steps_to_threshold"),
                "wall_clock_to_threshold": metrics.get("wall_clock_to_threshold"),
                "instability": metrics.get("instability_failure_count"),
            }
        )
    return rows


def to_markdown(rows: list[dict]) -> str:
    header = (
        "# Stage 0 Benchmark Results (Template)\n\n"
        "| experiment | seed | task | optimizer | final_metric_at_budget | "
        "steps_to_threshold | wall_clock_to_threshold | instability_failure_count |\n"
        "|---|---:|---|---|---:|---:|---:|---:|\n"
    )
    body = "\n".join(
        "| {experiment} | {seed} | {task} | {optimizer} | {final_metric} | {steps_to_threshold} | {wall_clock_to_threshold} | {instability} |".format(
            **row
        )
        for row in rows
    )

    go_no_go = (
        "\n\n## Stage 0 Go/No-Go\n"
        "- **Go** if at least one tuned baseline consistently beats default AdamW on final metric and wall-clock-to-threshold across seeds.\n"
        "- **No-Go** if metrics are too noisy (high seed variance) or instability count is elevated; retune protocol and budgets first.\n"
    )
    return header + (body + "\n" if body else "") + go_no_go


def main() -> None:
    args = parse_args()
    rows = load_stage0_rows(args.outputs_dir)
    report_md = to_markdown(rows)
    args.write_md.parent.mkdir(parents=True, exist_ok=True)
    args.write_md.write_text(report_md, encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.write_md}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 3 ID/OOD transfer evaluation runner.

This script executes a split protocol config that groups tasks into ID and OOD
buckets (family holdout, architecture holdout, and scale holdout) and writes a
transfer summary JSON with the required metrics:
- ID performance
- OOD performance drop
- stability rate
- overhead impact
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
from typing import Any

import yaml


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _run_case(case: dict[str, str], run_config: str, timeout_sec: int, seed: int) -> dict[str, Any]:
    run_cfg = _load_yaml(run_config)
    run_cfg["seed"] = int(seed)
    run_cfg["experiment_name"] = f"{run_cfg['experiment_name']}_{case['name']}"

    temp_run_cfg = Path("outputs") / f"_tmp_{case['name']}_run.yaml"
    temp_run_cfg.parent.mkdir(parents=True, exist_ok=True)
    with temp_run_cfg.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(run_cfg, handle, sort_keys=False)

    cmd = [
        "timeout",
        f"{int(timeout_sec)}s",
        "python",
        "-m",
        "bench.cli",
        "--task-config",
        case["task_config"],
        "--optimizer-config",
        case["optimizer_config"],
        "--run-config",
        str(temp_run_cfg),
    ]
    env = dict(**os.environ)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"src:{existing}" if existing else "src"
    result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
    if result.returncode != 0:
        return {
            "name": case["name"],
            "status": "failed",
            "returncode": result.returncode,
            "stderr_tail": result.stderr[-5000:],
        }

    summary_path = Path(run_cfg["log_dir"]) / run_cfg["experiment_name"] / "summary.json"
    if not summary_path.exists():
        return {
            "name": case["name"],
            "status": "failed",
            "returncode": -1,
            "stderr_tail": f"Missing summary at {summary_path}",
        }

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    metrics = summary.get("metrics", {})
    diag = (summary.get("val_eval", []) or [{}])[-1]
    return {
        "name": case["name"],
        "status": "ok",
        "summary_path": str(summary_path),
        "final_metric": float(metrics.get("final_metric_at_budget", 0.0)),
        "stability_failures": int(metrics.get("instability_failure_count", 0)),
        "fallback_events": int(metrics.get("fallback_events", 0)),
        "last_eval_wall_clock_sec": float(diag.get("wall_clock_sec", 0.0)),
    }


def _safe_mean(vals: list[float]) -> float:
    return float(statistics.fmean(vals)) if vals else 0.0


def _aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return {
            "id_performance": 0.0,
            "stability_rate": 0.0,
            "overhead_proxy_sec": 0.0,
        }
    perf = _safe_mean([r["final_metric"] for r in ok])
    fail = _safe_mean([float(r["stability_failures"]) for r in ok])
    stability_rate = float(max(0.0, 1.0 - fail))
    overhead = _safe_mean([r["last_eval_wall_clock_sec"] for r in ok])
    return {
        "id_performance": perf,
        "stability_rate": stability_rate,
        "overhead_proxy_sec": overhead,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 3 transfer evaluation protocol")
    parser.add_argument("--split-config", required=True, help="YAML split protocol config")
    parser.add_argument("--run-config", required=True, help="Run config YAML")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--output", default="outputs/stage3_transfer_summary.json")
    args = parser.parse_args()

    split = _load_yaml(args.split_config)

    groups = {
        "id": split.get("id_tasks", []),
        "ood_family": split.get("ood_family_tasks", []),
        "ood_arch": split.get("holdout_architecture_variants", []),
        "ood_scale": split.get("holdout_scale_tiers", []),
    }

    results: dict[str, list[dict[str, Any]]] = {k: [] for k in groups}
    for group_name, cases in groups.items():
        for case in cases:
            results[group_name].append(_run_case(case, args.run_config, args.timeout_sec, args.seed))

    id_aggr = _aggregate(results["id"])
    ood_cases = results["ood_family"] + results["ood_arch"] + results["ood_scale"]
    ood_aggr = _aggregate(ood_cases)

    output = {
        "protocol_name": split.get("protocol_name", "stage3_transfer"),
        "seed": args.seed,
        "split_config": args.split_config,
        "run_config": args.run_config,
        "results": results,
        "summary": {
            "id_performance": id_aggr["id_performance"],
            "ood_performance": ood_aggr["id_performance"],
            "ood_performance_drop": id_aggr["id_performance"] - ood_aggr["id_performance"],
            "id_stability_rate": id_aggr["stability_rate"],
            "ood_stability_rate": ood_aggr["stability_rate"],
            "overhead_impact_sec": ood_aggr["overhead_proxy_sec"] - id_aggr["overhead_proxy_sec"],
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    print(json.dumps(output["summary"], indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Stage 4 ablation matrix runner for transformer + residual experiments.

This script enforces bounded execution per case and computes go/no-go criteria
for continuing the direct-update residual line.
"""

from __future__ import annotations

import argparse
import copy
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


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _run_case(case: dict[str, Any], run_config: str, timeout_sec: int, seed: int) -> dict[str, Any]:
    run_cfg = _load_yaml(run_config)
    run_cfg["seed"] = int(seed)
    run_cfg["experiment_name"] = f"{run_cfg['experiment_name']}_{case['name']}"

    optimizer_cfg_path = case.get("optimizer_config")
    if optimizer_cfg_path:
        opt_cfg = _load_yaml(optimizer_cfg_path)
    else:
        # If no explicit config, use Stage 4 residual preset and apply overrides.
        opt_cfg = _load_yaml("configs/optimizers/learned_hybrid_stage4_transformer_residual.yaml")

    inline = case.get("optimizer_inline_overrides", {})
    if inline:
        opt_cfg = _deep_update(copy.deepcopy(opt_cfg), inline)

    tmp_dir = Path("outputs/stage4_tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    run_tmp = tmp_dir / f"run_{case['name']}.yaml"
    opt_tmp = tmp_dir / f"opt_{case['name']}.yaml"

    run_tmp.write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
    opt_tmp.write_text(yaml.safe_dump(opt_cfg, sort_keys=False), encoding="utf-8")

    cmd = [
        "timeout",
        f"{int(timeout_sec)}s",
        "python",
        "-m",
        "bench.cli",
        "--task-config",
        case["task_config"],
        "--optimizer-config",
        str(opt_tmp),
        "--run-config",
        str(run_tmp),
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
            "stdout_tail": result.stdout[-2000:],
        }

    summary_path = Path(run_cfg["log_dir"]) / run_cfg["experiment_name"] / "summary.json"
    if not summary_path.exists():
        return {
            "name": case["name"],
            "status": "failed",
            "returncode": -1,
            "stderr_tail": f"Missing summary: {summary_path}",
        }

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = summary.get("metrics", {})
    val_eval = summary.get("val_eval", [])
    last_eval = val_eval[-1] if val_eval else {}

    return {
        "name": case["name"],
        "status": "ok",
        "summary_path": str(summary_path),
        "final_metric": float(metrics.get("final_metric_at_budget", 0.0)),
        "instability": int(metrics.get("instability_failure_count", 0)),
        "fallback_events": int(metrics.get("fallback_events", 0)),
        "wall_clock_sec": float(last_eval.get("wall_clock_sec", 0.0)),
    }


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _classify_case(name: str) -> str:
    lower = name.lower()
    if "base_only" in lower:
        return "base_only"
    if "residual" in lower:
        return "base_plus_residual"
    return "other"


def _compute_go_no_go(results: list[dict[str, Any]]) -> dict[str, Any]:
    ok = [r for r in results if r.get("status") == "ok"]
    groups: dict[str, list[dict[str, Any]]] = {"base_only": [], "base_plus_residual": [], "other": []}
    for row in ok:
        groups[_classify_case(row["name"])].append(row)

    base = groups["base_only"]
    residual = groups["base_plus_residual"]

    base_metric = _mean([r["final_metric"] for r in base])
    residual_metric = _mean([r["final_metric"] for r in residual])
    base_wall = _mean([r["wall_clock_sec"] for r in base])
    residual_wall = _mean([r["wall_clock_sec"] for r in residual])
    base_instability = _mean([float(r["instability"]) for r in base])
    residual_instability = _mean([float(r["instability"]) for r in residual])

    decision = "no_go"
    reasons: list[str] = []
    if not residual:
        reasons.append("No base+residual cases completed successfully")
    else:
        if residual_metric <= base_metric:
            reasons.append("Residual variants do not beat base-only final metric")
        if residual_wall > base_wall:
            reasons.append("Residual variants lose wall-clock performance")
        if residual_instability > base_instability:
            reasons.append("Residual variants increase instability rate")

    if residual and not reasons:
        decision = "go"

    return {
        "decision": decision,
        "criteria": {
            "metric_delta_vs_base": residual_metric - base_metric,
            "wall_clock_delta_vs_base_sec": residual_wall - base_wall,
            "instability_delta_vs_base": residual_instability - base_instability,
        },
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 4 ablation matrix")
    parser.add_argument("--matrix-config", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-sec", type=int, default=600)
    parser.add_argument("--output", default="outputs/stage4_ablation_summary.json")
    args = parser.parse_args()

    matrix = _load_yaml(args.matrix_config)
    cases = matrix.get("cases", [])

    results = [_run_case(case, args.run_config, args.timeout_sec, args.seed) for case in cases]
    go_no_go = _compute_go_no_go(results)

    output = {
        "protocol_name": matrix.get("protocol_name", "stage4_ablation"),
        "seed": args.seed,
        "matrix_config": args.matrix_config,
        "run_config": args.run_config,
        "results": results,
        "go_no_go": go_no_go,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps({"go_no_go": go_no_go}, indent=2))


if __name__ == "__main__":
    main()

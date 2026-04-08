"""CLI for fair baseline tuning under equal budget constraints."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from .config import ExperimentConfig, OptimizerConfig, load_experiment_config
from .runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune optimizer hyperparameters on a fixed budget.")
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--search-config", required=True, help="YAML containing candidate optimizer settings")
    return parser.parse_args()


def _load_yaml(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _build_cfg(base: ExperimentConfig, opt: dict, exp_name: str) -> ExperimentConfig:
    cloned = copy.deepcopy(base)
    cloned.optimizer = OptimizerConfig(**opt)
    cloned.run.experiment_name = exp_name
    return cloned


def main() -> None:
    args = parse_args()
    search = _load_yaml(args.search_config)
    base_opt = search["base_optimizer_config"]
    candidates = search["candidates"]

    base = load_experiment_config(args.task_config, base_opt, args.run_config)

    results = []
    for idx, candidate in enumerate(candidates):
        exp_name = f"{base.run.experiment_name}_tune_{idx}"
        cfg = _build_cfg(base, candidate, exp_name)
        summary = run_experiment(cfg)
        results.append({"candidate": candidate, "metrics": summary["metrics"]})

    mode = base.run.threshold_mode
    key = "final_metric_at_budget"
    reverse = mode == "max"
    ranked = sorted(results, key=lambda r: (r["metrics"][key] is not None, r["metrics"][key]), reverse=reverse)

    print(json.dumps({"ranked": ranked, "selection_metric": key, "mode": mode}, indent=2))


if __name__ == "__main__":
    main()

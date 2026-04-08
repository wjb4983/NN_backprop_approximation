"""CLI entrypoint for running single benchmark experiments."""

from __future__ import annotations

import argparse
import json

from .config import load_experiment_config
from .runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single benchmark experiment.")
    parser.add_argument("--task-config", required=True, help="Path to YAML task config")
    parser.add_argument("--optimizer-config", required=True, help="Path to YAML optimizer config")
    parser.add_argument("--run-config", required=True, help="Path to YAML run config")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.task_config, args.optimizer_config, args.run_config)
    summary = run_experiment(cfg)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()

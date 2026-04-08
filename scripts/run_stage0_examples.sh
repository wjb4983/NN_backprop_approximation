#!/usr/bin/env bash
set -euo pipefail

# Non-interactive, bounded runs for reproducibility.
timeout 600s bench-run --task-config configs/tasks/tabular_synth.yaml --optimizer-config configs/optimizers/adamw.yaml --run-config configs/run.yaml

timeout 600s bench-run --task-config configs/tasks/tabular_synth.yaml --optimizer-config configs/optimizers/sgd_momentum.yaml --run-config configs/run.yaml

timeout 600s bench-run --task-config configs/tasks/tabular_synth.yaml --optimizer-config configs/optimizers/adamw_cosine.yaml --run-config configs/run.yaml

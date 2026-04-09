#!/usr/bin/env bash
set -euo pipefail

# Stage 1 reproducible experiment bundle with explicit bounded runtimes.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_CFG="configs/run.yaml"
TASK_TAB="configs/tasks/tabular_synth.yaml"
TASK_CNN="configs/tasks/cnn_mnist.yaml"
OPT_LEARNED="configs/optimizers/learned_hybrid.yaml"
OPT_ADAMW="configs/optimizers/adamw.yaml"
OPT_SGD="configs/optimizers/sgd_momentum.yaml"

# Smoke checks

timeout 180s bench-run --task-config "$TASK_TAB" --optimizer-config "$OPT_LEARNED" --run-config "$RUN_CFG"
timeout 180s bench-run --task-config "$TASK_TAB" --optimizer-config "$OPT_ADAMW" --run-config "$RUN_CFG"

# Bounded longer comparisons

timeout 600s bench-run --task-config "$TASK_CNN" --optimizer-config "$OPT_LEARNED" --run-config "$RUN_CFG"
timeout 600s bench-run --task-config "$TASK_CNN" --optimizer-config "$OPT_ADAMW" --run-config "$RUN_CFG"
timeout 600s bench-run --task-config "$TASK_CNN" --optimizer-config "$OPT_SGD" --run-config "$RUN_CFG"

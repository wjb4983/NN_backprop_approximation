#!/usr/bin/env bash
set -euo pipefail

# Stage 2 reproducible experiment bundle with explicit bounded runtimes.
# Runs in chunks so long suites can be resumed or parallelized externally.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_CFG="configs/run.yaml"
TASK_TAB="configs/tasks/tabular_synth.yaml"
TASK_CNN="configs/tasks/cnn_mnist.yaml"
TASK_CNN_LARGE="configs/tasks/cnn_mnist_large.yaml"
TASK_QUANT="configs/tasks/quant_structured.yaml"

OPT_STAGE2="configs/optimizers/learned_hybrid_stage2.yaml"
OPT_NO_TEMP="configs/optimizers/learned_hybrid_stage2_no_temporal.yaml"
OPT_NO_GATE="configs/optimizers/learned_hybrid_stage2_no_gating.yaml"
OPT_NO_TRUST="configs/optimizers/learned_hybrid_stage2_no_trust.yaml"
OPT_ADAMW="configs/optimizers/adamw.yaml"

run_group_1_smoke() {
  timeout 240s bench-run --task-config "$TASK_TAB" --optimizer-config "$OPT_STAGE2" --run-config "$RUN_CFG"
  timeout 240s bench-run --task-config "$TASK_QUANT" --optimizer-config "$OPT_STAGE2" --run-config "$RUN_CFG"
}

run_group_2_core() {
  timeout 600s bench-run --task-config "$TASK_CNN" --optimizer-config "$OPT_STAGE2" --run-config "$RUN_CFG"
  timeout 600s bench-run --task-config "$TASK_CNN_LARGE" --optimizer-config "$OPT_STAGE2" --run-config "$RUN_CFG"
  timeout 600s bench-run --task-config "$TASK_QUANT" --optimizer-config "$OPT_ADAMW" --run-config "$RUN_CFG"
}

run_group_3_ablations() {
  timeout 600s bench-run --task-config "$TASK_CNN_LARGE" --optimizer-config "$OPT_NO_TEMP" --run-config "$RUN_CFG"
  timeout 600s bench-run --task-config "$TASK_CNN_LARGE" --optimizer-config "$OPT_NO_GATE" --run-config "$RUN_CFG"
  timeout 600s bench-run --task-config "$TASK_CNN_LARGE" --optimizer-config "$OPT_NO_TRUST" --run-config "$RUN_CFG"
}

GROUP="${1:-all}"
case "$GROUP" in
  group1) run_group_1_smoke ;;
  group2) run_group_2_core ;;
  group3) run_group_3_ablations ;;
  all)
    run_group_1_smoke
    run_group_2_core
    run_group_3_ablations
    ;;
  *)
    echo "Unknown group '$GROUP'. Use: group1|group2|group3|all" >&2
    exit 2
    ;;
esac

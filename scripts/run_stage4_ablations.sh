#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MATRIX_CFG="${1:-configs/stage4/ablation/transformer_residual_matrix.yaml}"
RUN_CFG="${2:-configs/stage4/runs/ablation_default.yaml}"
TIMEOUT_SEC="${3:-600}"
SEED="${4:-42}"
OUT_JSON="${5:-outputs/stage4_ablation_summary.json}"

timeout "${TIMEOUT_SEC}s" python scripts/eval_stage4_ablation.py \
  --matrix-config "$MATRIX_CFG" \
  --run-config "$RUN_CFG" \
  --seed "$SEED" \
  --timeout-sec "$TIMEOUT_SEC" \
  --output "$OUT_JSON"

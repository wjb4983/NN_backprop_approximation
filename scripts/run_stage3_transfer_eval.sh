#!/usr/bin/env bash
set -euo pipefail

# Reproducible Stage 3 ID/OOD transfer protocol runner with bounded execution.
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SPLIT_CFG="${1:-configs/stage3/splits/family_scale_ood.yaml}"
RUN_CFG="${2:-configs/stage3/runs/transfer_default.yaml}"
TIMEOUT_SEC="${3:-600}"
SEED="${4:-42}"
OUT_JSON="${5:-outputs/stage3_transfer_summary.json}"

timeout "${TIMEOUT_SEC}s" python scripts/eval_stage3_transfer.py \
  --split-config "$SPLIT_CFG" \
  --run-config "$RUN_CFG" \
  --seed "$SEED" \
  --timeout-sec "$TIMEOUT_SEC" \
  --output "$OUT_JSON"

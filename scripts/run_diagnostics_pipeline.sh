#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/diagnostics/default.yaml}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "missing config: $CONFIG_PATH" >&2
  exit 1
fi

read_cfg() {
  python - <<'PY' "$CONFIG_PATH" "$1"
from pathlib import Path
import sys
import yaml
cfg = yaml.safe_load(Path(sys.argv[1]).read_text())
keys = sys.argv[2].split('.')
val = cfg
for k in keys:
    val = val[k]
print(val)
PY
}

METRICS_JSONL="$(read_cfg metrics_jsonl)"
DATASET_CSV="$(read_cfg dataset_csv)"
CHECKPOINT="$(read_cfg checkpoint)"
EVAL_JSON="$(read_cfg eval_json)"
H="$(read_cfg horizon_steps)"
MIN_DELTA="$(read_cfg min_delta)"
NOISE="$(read_cfg noise_multiplier)"
JUMP="$(read_cfg instability_loss_jump)"
STALL_W="$(read_cfg hp_mismatch_stall_windows)"
EPOCHS="$(read_cfg train.epochs)"
BATCH="$(read_cfg train.batch_size)"
LR="$(read_cfg train.lr)"
SEED="$(read_cfg train.seed)"

export PYTHONPATH="src:${PYTHONPATH:-}"

timeout 120s python scripts/generate_diagnostics_labels.py \
  --metrics-jsonl "$METRICS_JSONL" \
  --output-csv "$DATASET_CSV" \
  --horizon-steps "$H" \
  --min-delta "$MIN_DELTA" \
  --noise-multiplier "$NOISE" \
  --instability-loss-jump "$JUMP" \
  --hp-mismatch-stall-windows "$STALL_W"

timeout 300s python scripts/train_diagnostics_model.py \
  --dataset-csv "$DATASET_CSV" \
  --output-ckpt "$CHECKPOINT" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --lr "$LR" \
  --seed "$SEED"

timeout 120s python scripts/eval_diagnostics.py \
  --dataset-csv "$DATASET_CSV" \
  --checkpoint "$CHECKPOINT" \
  --output "$EVAL_JSON"

printf 'diagnostics pipeline complete\n'

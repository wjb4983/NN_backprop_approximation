# Learned-Optimizer Benchmark Harness (Stage 0)

This repository contains a reproducible Stage 0 harness for benchmarking optimizer baselines under a fair protocol before integrating learned optimizers.

## Stage 0 deliverables

- Modular experiment runner with:
  - task configs
  - optimizer configs
  - deterministic seed control
  - per-step + wall-clock metric logging
- Baselines:
  - AdamW
  - SGD + Momentum
  - AdamW + Cosine schedule
- Metrics:
  - steps-to-threshold
  - wall-clock-to-threshold
  - AULC over early window
  - final metric at fixed budget
  - instability/failure count
- Train/val/test split support at task-family level
- Fair tuning protocol docs
- Markdown report template

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run single experiments (non-interactive)

```bash
timeout 600s bench-run --task-config configs/tasks/cnn_mnist.yaml --optimizer-config configs/optimizers/adamw.yaml --run-config configs/run.yaml

timeout 600s bench-run --task-config configs/tasks/tabular_synth.yaml --optimizer-config configs/optimizers/sgd_momentum.yaml --run-config configs/run.yaml
```

## Run fair tuning search

```bash
timeout 600s bench-tune --task-config configs/tasks/tabular_synth.yaml --run-config configs/run.yaml --search-config configs/optimizers/adamw_search.yaml
```

Outputs are written to `outputs/<experiment_name>/` with:
- `metrics.jsonl` (step + wall-clock logs)
- `summary.json` (aggregate benchmark metrics and failure status)

See `docs/tuning_protocol.md`, `docs/reports/benchmark_report_template.md`, and `docs/assumptions_and_open_questions.md`.


## Stage 1 learned optimizer prototype

Stage 1 adds a simplest-working **learned hybrid optimizer** that modulates AdamW with bounded per-layer controller outputs and fallback safety guards.

Quick start:

```bash
timeout 180s bench-run --task-config configs/tasks/tabular_synth.yaml --optimizer-config configs/optimizers/learned_hybrid.yaml --run-config configs/run.yaml
timeout 900s bash scripts/run_stage1_learned_optimizer.sh
```

See `docs/stage1_learned_optimizer.md` and `docs/reports/stage1_result_table_template.md`.

## Stage 3 cross-family generalization

Stage 3 adds a unified optimizer feature interface and an adapter-capable learned
optimizer controller for transfer across model families and scale tiers.

Quick start (bounded):

```bash
timeout 300s bash scripts/run_stage3_transfer_eval.sh \
  configs/stage3/splits/family_scale_ood.yaml \
  configs/stage3/runs/transfer_smoke.yaml \
  300 42 outputs/stage3_transfer_smoke_summary.json
```

See `docs/stage3_generalization.md` and
`docs/reports/stage3_transfer_report_template.md`.


## Training diagnostics subsystem

A standalone diagnostics module predicts whether the model is learning and near-term risk signals (stall/instability/hyperparameter mismatch) from training logs.

Quick start (bounded non-interactive pipeline):

```bash
timeout 600s bash scripts/run_diagnostics_pipeline.sh configs/diagnostics/default.yaml
```

Live integration into benchmark logs is controlled via run-config keys:
- `diagnostics_enabled`
- `diagnostics_checkpoint`
- `diagnostics_mc_samples`

See `docs/training_diagnostics_subsystem.md` and `docs/reports/diagnostics_report_template.md`.

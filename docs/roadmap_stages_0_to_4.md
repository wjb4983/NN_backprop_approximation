# Learned Optimizer Roadmap (Stages 0-4 + Diagnostics)

This note provides reproducible command entrypoints, stage deliverables, and go/no-go criteria.

## Stage 0 — Benchmark harness + fair baseline protocol

Deliverables:
- Fixed-budget harness with per-step/wall-clock logging.
- Baseline matrix across tasks/optimizers/seeds.
- Optional tuned baseline search runs.
- Stage 0 markdown table generation.

Commands (bounded):
```bash
timeout 1800s bash scripts/run_stage0_fair_protocol.sh \
  configs/stage0/fair_baseline_matrix.yaml \
  configs/run.yaml

timeout 60s python scripts/summarize_stage0_results.py outputs
```

Go/No-Go:
- **Go** when tuned baseline protocol is stable across seeds and at least one baseline is clearly strongest on both final metric and wall-clock proxy.
- **No-Go** when seed variance or instability dominates signal; increase budget or adjust search space before Stage 1 comparisons.

## Stage 1 — Minimal hybrid layerwise AdamW modulator

Commands:
```bash
timeout 180s bench-run --task-config configs/tasks/tabular_synth.yaml --optimizer-config configs/optimizers/learned_hybrid.yaml --run-config configs/run.yaml
timeout 900s bash scripts/run_stage1_learned_optimizer.sh
```

Go/No-Go:
- **Go** if learned hybrid matches or beats strongest Stage 0 baseline early in training while preserving stability/fallback safety.
- **No-Go** if fallback events or instability offset any early-step gains.

## Stage 2 — Temporal + gated stronger hybrid

Commands:
```bash
timeout 900s bash scripts/run_stage2_learned_optimizer.sh group1
timeout 1800s bash scripts/run_stage2_learned_optimizer.sh group2
timeout 1800s bash scripts/run_stage2_learned_optimizer.sh group3
```

Go/No-Go:
- **Go** if temporal/gating/trust additions improve early-step and/or wall-clock performance with no instability regression.
- **No-Go** if added overhead cancels gains or ablations show no reliable contribution.

## Stage 3 — Cross-family generalization with shared schema + adapters

Commands:
```bash
timeout 300s bash scripts/run_stage3_transfer_eval.sh \
  configs/stage3/splits/family_scale_ood.yaml \
  configs/stage3/runs/transfer_smoke.yaml \
  300 42 outputs/stage3_transfer_smoke_summary.json
```

Go/No-Go:
- **Go** if OOD drop remains bounded while maintaining stability and acceptable overhead.
- **No-Go** if transfer collapses or family adapters overfit to ID families.

## Stage 4 — Transformer/residual direct-update branch

Commands:
```bash
timeout 300s bash scripts/run_stage4_ablations.sh \
  configs/stage4/ablation/transformer_residual_matrix.yaml \
  configs/stage4/runs/ablation_smoke.yaml \
  300 42 outputs/stage4_ablation_smoke_summary.json
```

Go/No-Go:
- Reuse the strict Stage 4 criteria from `docs/stage4_advanced_architecture.md`.

## Diagnostics subsystem (parallel track)

Command:
```bash
timeout 600s bash scripts/run_diagnostics_pipeline.sh configs/diagnostics/default.yaml
```

Go/No-Go:
- **Go** if diagnostics model is calibrated and improves decision utility (e.g., actionable stall-risk warnings).
- **No-Go** if predictions are uncalibrated or fail to provide lead-time.

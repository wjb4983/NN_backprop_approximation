# Stage 4: Advanced Optimizer Architecture Experiments

## Scope
Stage 4 introduces ambitious architecture experiments while preserving strict optimizer safety constraints.

## Branch A: Transformer over layer/tensor tokens
- Controller architecture option: `controller_arch: transformer_tokens`.
- Input representation: one token per selected scalar feature from the unified layer feature vector.
- Ablation knobs:
  - `transformer_layers`
  - `transformer_model_dim`
  - `transformer_heads`
  - `transformer_ff_dim`
  - `token_feature_subset` (`all`, `stats_only`, `stats_meta`, `stats_family`, `meta_only`)

## Branch B: Learned-optimizer prototype (bounded residual direct-update)
- Enabled with `enable_residual_update: true`.
- Residual is **always** bounded and applied on top of the safe base update:
  - `residual = residual_scale * base_update`
  - `residual_scale` is bounded via `tanh` and `residual_bound`.
- Trust-region constraint:
  - residual norm clipped by `residual_trust_radius * ||base_update||`.
- Fallback trigger:
  - if residual ratio exceeds `residual_fallback_ratio`, optimizer falls back to base-safe behavior.
- Initial Stage 4 variant does **not** support pure unconstrained raw direct parameter updates.

## Safety-first constraints (hard)
- Strict bounded output mapping for all learned controls.
- Layer update norm clipping remains active.
- Trust-ratio modulation remains clipped.
- Fallback-to-base mode remains available and is triggered for non-finite/unsafe behavior.

## Ablation coverage matrix
The matrix config at `configs/stage4/ablation/transformer_residual_matrix.yaml` includes:
1. Stage 2 base-only reference.
2. Transformer base (no residual).
3. Transformer + residual.
4. Transformer depth/width/token-subset variants.

## Go/No-Go criteria for direct-update line
Computed by `scripts/eval_stage4_ablation.py`:
- **Go** only if base+residual variants simultaneously:
  1. improve final metric vs base-only,
  2. do not lose wall-clock performance,
  3. do not increase instability rate.
- Otherwise: **No-Go** for continuation of direct-update line in this cycle.

## Repro commands (bounded)
```bash
# Smoke matrix
timeout 300s bash scripts/run_stage4_ablations.sh \
  configs/stage4/ablation/transformer_residual_matrix.yaml \
  configs/stage4/runs/ablation_smoke.yaml \
  300 42 outputs/stage4_ablation_smoke_summary.json

# Default matrix
timeout 1200s bash scripts/run_stage4_ablations.sh \
  configs/stage4/ablation/transformer_residual_matrix.yaml \
  configs/stage4/runs/ablation_default.yaml \
  1200 42 outputs/stage4_ablation_summary.json
```

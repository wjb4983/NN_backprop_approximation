# Stage 3: Generalization Across Model Families and Scales

## Goal
Stage 3 introduces a shared optimizer input interface and adapter-capable controller
that can transfer across CNN/vision-like models and structured quant-style models.

## Implemented Components

1. **Unified feature extraction module**
   - File: `src/bench/optimizers/feature_interface.py`
   - Schema includes:
     - normalized statistics (EMA-normalized gradient/parameter dynamics)
     - metadata tokens (depth, shape descriptors, param count, role/type)
     - optional modality/task-family token

2. **Adapter-capable learned optimizer architecture**
   - File: `src/bench/optimizers/learned_hybrid.py`
   - Shared backbone + family-specific residual adapters (`vision`, `structured`, `tabular`, `unknown`)
   - Control outputs remain safety-bounded and keep Stage 2 stability/fallback logic.

3. **Configurable family holdout protocol**
   - Split config: `configs/stage3/splits/family_scale_ood.yaml`
   - Run configs:
     - `configs/stage3/runs/transfer_default.yaml`
     - `configs/stage3/runs/transfer_smoke.yaml`
   - Stage 3 optimizer presets:
     - `configs/optimizers/learned_hybrid_stage3_vision.yaml`
     - `configs/optimizers/learned_hybrid_stage3_structured.yaml`

4. **ID/OOD transfer evaluation scripts**
   - Python driver: `scripts/eval_stage3_transfer.py`
   - Shell wrapper: `scripts/run_stage3_transfer_eval.sh`

## OOD Protocol

The protocol supports three holdout axes:
- **Family holdout**: exclude full task families (e.g., quant structured)
- **Architecture holdout**: exclude architecture variants (e.g., larger CNN variant)
- **Scale holdout**: exclude scale tiers (e.g., xlarge CNN / large structured model)

## Metrics Produced
- ID performance (`final_metric_at_budget`, averaged over ID tasks)
- OOD performance
- OOD performance drop (`ID - OOD`)
- Stability rate proxy (derived from instability failure count)
- Overhead impact proxy (OOD minus ID wall-clock to latest eval)

## Repro Commands (bounded)

```bash
# Smoke transfer run
timeout 300s bash scripts/run_stage3_transfer_eval.sh \
  configs/stage3/splits/family_scale_ood.yaml \
  configs/stage3/runs/transfer_smoke.yaml \
  300 42 outputs/stage3_transfer_smoke_summary.json

# Default transfer run
timeout 1800s bash scripts/run_stage3_transfer_eval.sh \
  configs/stage3/splits/family_scale_ood.yaml \
  configs/stage3/runs/transfer_default.yaml \
  1800 42 outputs/stage3_transfer_summary.json
```

## Assumptions and open questions
See `docs/assumptions_and_open_questions.md` for Stage 3 additions.

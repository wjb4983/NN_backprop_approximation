# Stage 2 Learned Optimizer: Temporal Hybrid Controller

## Goal
Stage 2 strengthens Stage 1 by improving temporal reasoning and transfer while keeping
optimizer overhead small and safeguards conservative.

## Key Enhancements

1. **Temporal encoder (ablatable)**
   - Feature windows are tracked per parameter tensor.
   - Controller can use `off`, `gru`, or `mlp` temporal encoding.
   - Switches:
     - `extra_params.temporal_on` (bool)
     - `extra_params.temporal_encoder_mode` (`gru|mlp|off`)

2. **Mode gating AdamW-like vs momentum-SGD-like (ablatable)**
   - Controller predicts a gate in `[0,1]`.
   - Update direction is blended:
     - `gate * adam_update + (1-gate) * sgd_momentum_update`
   - Switch:
     - `extra_params.gating_on` (bool)

3. **Trust-ratio style modulation (ablatable)**
   - Uses LAMB-like trust ratio scalar per layer:
     - `||param|| / (||update|| + eps)` with clipping.
   - Multiplies by controller trust multiplier and safety caps.
   - Switch:
     - `extra_params.trust_modulation_on` (bool)

4. **Extended stability regularization**
   - Penalty for erratic multiplier swings (control-jump shrinkage toward neutral).
   - Smoothness constraint via EMA over controls.
   - Diagnostics expose mean swing/smoothness penalties.

5. **Profiling hooks for optimizer overhead**
   - Runner logs `step_profile` with forward/backward/optimizer timings.
   - Optimizer diagnostics include per-step and EMA optimizer step time.

## Stage 2 Configs
- Full Stage 2: `configs/optimizers/learned_hybrid_stage2.yaml`
- Ablations:
  - `configs/optimizers/learned_hybrid_stage2_no_temporal.yaml`
  - `configs/optimizers/learned_hybrid_stage2_no_gating.yaml`
  - `configs/optimizers/learned_hybrid_stage2_no_trust.yaml`

## Evaluation Coverage
- Stage 1 tasks retained:
  - `configs/tasks/tabular_synth.yaml`
  - `configs/tasks/cnn_mnist.yaml`
- Added larger CNN variant:
  - `configs/tasks/cnn_mnist_large.yaml`
- Added non-image structured quant-style proxy task:
  - `configs/tasks/quant_structured.yaml`

## Repro Commands (bounded)
```bash
# group 1: smoke
timeout 900s bash scripts/run_stage2_learned_optimizer.sh group1

# group 2: core comparisons
timeout 1800s bash scripts/run_stage2_learned_optimizer.sh group2

# group 3: ablations
timeout 1800s bash scripts/run_stage2_learned_optimizer.sh group3
```

## Reporting checklist
- Report both **step efficiency** and **wall-clock efficiency**.
- Include ablation table for temporal/gating/trust switches.
- Include failure-case notes (fallback triggers, non-finite failures, regressions).

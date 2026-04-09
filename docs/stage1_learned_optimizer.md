# Stage 1 Learned Optimizer Prototype

## Architecture
- **Base optimizer**: AdamW-style update with standard first/second moments.
- **Controller**: tiny MLP (`TinyLayerwiseController`) producing 3 per-layer controls:
  1. LR scale multiplier
  2. Momentum correction multiplier
  3. Trust/clipping multiplier
- **Feature inputs (per layer)**:
  - gradient norm statistics
  - parameter norm
  - moment summaries (`m` mean-abs, `v` mean)
  - update-to-weight proxy ratio
  - loss trend features from short window
  - metadata: depth, parameter count, layer type one-hot

## Safety and fallback behavior
- Output is bounded with `tanh` and configurable clamp ranges.
- Per-layer update norm is clipped by `max_layer_update_norm * trust_multiplier`.
- Any instability trigger (non-finite grad or update norm) arms fallback mode.
- Fallback mode temporarily uses plain AdamW-style controls (`1.0` multipliers).
- Runner logs `fallback_events` and controller statistics for interpretability.

## Training approach implemented in Stage 1
1. **Imitation pretraining (lightweight)**
   - `scripts/stage1_imitation_pretrain.py` trains controller on conservative synthetic teacher targets around AdamW-like behavior.
   - Produces `controller_imitation.pt` checkpoint.
2. **Short-horizon meta-training scaffold**
   - `scripts/stage1_meta_train_stub.py` writes bounded-run scaffolding artifact and explicit TODOs for differentiable truncated unroll.

## Config and plugin usage
- Optimizer config: `configs/optimizers/learned_hybrid.yaml`
- Switch in existing loop by setting `name: learned_hybrid`.
- Optional pretrained controller path can be provided via:
  - `extra_params.controller_pretrained_path`

## Repro commands
```bash
# smoke learned optimizer run
timeout 180s bench-run --task-config configs/tasks/tabular_synth.yaml --optimizer-config configs/optimizers/learned_hybrid.yaml --run-config configs/run.yaml

# imitation pretrain
timeout 180s python scripts/stage1_imitation_pretrain.py

# bounded stage1 bundle
timeout 900s bash scripts/run_stage1_learned_optimizer.sh
```

## Logging outputs
- `metrics.jsonl` includes `optimizer_diagnostics` with controller mean outputs.
- `summary.json` includes `fallback_events` and updated instability count.

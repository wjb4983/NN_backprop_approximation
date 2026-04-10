# Training Diagnostics Subsystem: "Is this model actually learning?"

This module adds a **separate diagnostics model** to estimate training health and near-term risks without directly controlling optimizer updates.

## Scope

The diagnostics subsystem predicts:
1. training health now
2. stall likelihood in next `H` logged evaluation steps
3. whether recent loss reduction is meaningful versus local noise
4. conditioning/instability risk
5. hyperparameter mismatch likelihood

## Architecture

- `src/bench/diagnostics/feature_pipeline.py`
  - Reuses logged optimizer diagnostics (`fallback_events`, `trust_ratio_ema`, etc.)
  - Combines these with trend/noise signals from train/val losses.
- `src/bench/diagnostics/modeling.py`
  - Multi-task probabilistic MLP (`sigmoid` outputs).
- `src/bench/diagnostics/inference.py`
  - Runtime hook for integration into benchmark logging.
- `src/bench/diagnostics/labels.py`
  - Future-window label generation from trajectory logs.
- `src/bench/diagnostics/eval_metrics.py`
  - AUROC/AUPRC, calibration (ECE), lead-time, and practical decision utility.

## Reproducible pipeline

```bash
timeout 600s bash scripts/run_diagnostics_pipeline.sh configs/diagnostics/default.yaml
```

This executes:
1. label generation
2. diagnostics model training
3. evaluation report export

## Runtime integration

Enable diagnostics in run config by setting:

```yaml
diagnostics_enabled: true
diagnostics_checkpoint: outputs/diagnostics/model.pt
diagnostics_mc_samples: 8
```

When enabled, each logged training point includes `training_diagnostics` alongside curves.

## Assumptions

- Diagnostics labels are derived from logged eval cadence; smaller `eval_every` gives richer signals.
- Stall/progress thresholds are noise-adjusted heuristics and should be tuned per task family.

## Open questions

- Should uncertainty be upgraded from MC-dropout to deep ensembles?
- Should labels use step-level training batches (higher frequency) in addition to eval points?
- For practical utility, do we need asymmetric costs per deployment scenario (e.g., false-stop penalties)?

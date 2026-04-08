# Baseline Tuning Protocol (Fairness)

## Objective
Compare optimizer families fairly under fixed compute and data budgets.

## Protocol
1. **Equal step budget**: all optimizers use the same `max_steps`.
2. **Equal data splits**: train/val/test are deterministic by `(seed, family)`.
3. **Equal eval cadence**: all runs use the same `eval_every`.
4. **Selection metric**: choose best hyperparameter candidate by `final_metric_at_budget` on validation split.
5. **No test leakage**: test metrics are reported only after selecting candidates on val.
6. **Multi-seed extension** (recommended next): keep same seed list for all optimizers; aggregate mean/std and failure count.

## Required reporting fields
- Optimizer config + scheduler config
- Fixed budgets (`max_steps`, wall-clock cap if used)
- Selection metric and mode (max/min)
- Number of failed runs

## Notes for reproducibility
- Use `timeout` wrappers for all CLI calls.
- Store exact command lines in experiment reports.
- Keep config files version-controlled and immutable per benchmark run.

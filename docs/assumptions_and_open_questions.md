# Assumptions and Open Questions

## Assumptions
- Stage 0 prioritizes harness correctness and reproducibility over SOTA model quality.
- CPU-only execution is acceptable for sanity checks and CI-style validation.
- Deterministic family-based splitting is sufficient for fair optimizer comparisons.
- Validation metric threshold (`threshold_metric`, `threshold_value`) is task-specific and configurable.

## Open questions
- Should Stage 1 add robust multi-seed aggregation and confidence intervals by default?
- Should wall-clock normalization include data-loading warmup exclusion?
- Do we require a strict wall-clock budget in addition to step budget for fairness?
- Which canonical task suite (vision, tabular, time-series) should be frozen for longitudinal tracking?

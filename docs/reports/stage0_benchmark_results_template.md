# Stage 0 Benchmark Results (Template)

| experiment | seed | task | optimizer | final_metric_at_budget | steps_to_threshold | wall_clock_to_threshold | instability_failure_count |
|---|---:|---|---|---:|---:|---:|---:|


## Stage 0 Go/No-Go
- **Go** if at least one tuned baseline consistently beats default AdamW on final metric and wall-clock-to-threshold across seeds.
- **No-Go** if metrics are too noisy (high seed variance) or instability count is elevated; retune protocol and budgets first.

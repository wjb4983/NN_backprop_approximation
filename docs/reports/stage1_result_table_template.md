# Stage 1 Result Table Template

| Task | Optimizer | Seed | Early learning speed (AULC) | Steps-to-threshold | Wall-clock-to-threshold (s) | Final metric @ budget | Fallback events | Failure |
|---|---|---:|---:|---:|---:|---:|---:|---|
| tabular_synth | AdamW | 123 |  |  |  |  | 0 | no |
| tabular_synth | SGD+Momentum | 123 |  |  |  |  | 0 | no |
| tabular_synth | LearnedHybridAdamW | 123 |  |  |  |  |  |  |
| cnn_mnist | AdamW | 123 |  |  |  |  | 0 | no |
| cnn_mnist | SGD+Momentum | 123 |  |  |  |  | 0 | no |
| cnn_mnist | LearnedHybridAdamW | 123 |  |  |  |  |  |  |

## Notes
- Report stability across at least 3 seeds once runtime budget allows.
- Include any fallback triggers and instability reasons for learned optimizer runs.

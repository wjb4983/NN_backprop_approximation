# Stage 2 Ablation Table Template

| Task | Optimizer variant | Temporal | Gating | Trust mod | AUC early | Steps-to-threshold | Wall-clock-to-threshold (s) | Final metric @ budget | Mean optimizer share | Fallback events | Failure notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|
| tabular_synth | Stage2 full | on | on | on |  |  |  |  |  |  |  |
| tabular_synth | Stage2 no temporal | off | on | on |  |  |  |  |  |  |  |
| tabular_synth | Stage2 no gating | on | off | on |  |  |  |  |  |  |  |
| tabular_synth | Stage2 no trust | on | on | off |  |  |  |  |  |  |  |
| cnn_mnist_large | Stage2 full | on | on | on |  |  |  |  |  |  |  |
| cnn_mnist_large | Stage2 no temporal | off | on | on |  |  |  |  |  |  |  |
| cnn_mnist_large | Stage2 no gating | on | off | on |  |  |  |  |  |  |  |
| cnn_mnist_large | Stage2 no trust | on | on | off |  |  |  |  |  |  |  |
| quant_structured | Stage2 full | on | on | on |  |  |  |  |  |  |  |
| quant_structured | AdamW baseline | n/a | n/a | n/a |  |  |  |  |  | 0 |  |

## Failure-case notes
- Record fallback triggers and repeated instability patterns.
- Note if any ablation reduces wall-clock efficiency despite improving step efficiency.

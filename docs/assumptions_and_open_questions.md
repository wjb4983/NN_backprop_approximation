# Assumptions and Open Questions

## Assumptions
- Stage 1 prioritizes a stable prototype over aggressive meta-optimization performance.
- The learned controller starts close to AdamW behavior via bounded outputs around 1.0.
- Fallback-to-AdamW behavior is preferable to hard failure for early experimentation.
- Lightweight imitation pretraining is acceptable as initialization before full trajectory imitation is implemented.

## Open questions
- Should fallback trigger on additional criteria (e.g., moving-average loss spikes)?
- What is the best trust-multiplier bound schedule across training phases?
- Should layer metadata include richer structural features (residual block ids, attention heads)?
- When implementing full truncated unroll, should we optimize for validation loss or threshold-oriented reward proxies?
- What minimum multi-seed count is required before claiming stability improvements?

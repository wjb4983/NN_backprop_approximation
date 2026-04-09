# Assumptions and Open Questions

## Assumptions
- Stage 2 keeps Stage 1's conservative safety posture (bounded controls + fallback) while adding temporal and hybrid control paths.
- Temporal windows of length 4 are a practical default balancing signal quality and overhead.
- Trust-ratio modulation should be clipped aggressively to avoid over-amplifying sparse layers.
- Per-step overhead profiling is useful for wall-clock accountability and should be always-on during R&D.
- Synthetic structured quant-style tasks are acceptable proxies before integrating proprietary or market datasets.

## Open questions
- Should temporal mode default to GRU for all tasks, or should we auto-select GRU vs temporal MLP by model size?
- Does the gate collapse toward a single mode in longer runs (mode collapse), and should entropy regularization be added?
- Should trust-ratio bounds be global or per-module-type to improve transfer across architectures?
- Are control smoothness coefficients stable across seeds and tasks, or should they be scheduled over training?
- What minimum multi-seed suite is required to claim Stage 2 transfer improvements vs Stage 1?
- Should evaluation add regression-style quant tasks (e.g., directional return regression) in addition to regime classification?

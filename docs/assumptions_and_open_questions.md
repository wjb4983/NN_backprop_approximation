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

## Stage 3 assumptions
- A shared normalized feature schema can reduce optimizer-input distribution shift across vision and structured models.
- Lightweight family adapters (small residual MLPs) are sufficient to capture family-specific quirks without duplicating the full controller.
- Role/type/shape metadata tokens improve transfer to unseen architecture variants and scale tiers.
- Family-token assignment from config should be explicit in Stage 3 optimizer presets to keep runs reproducible.

## Stage 3 open questions
- Should family adapters be selected hard (current) or via soft routing across adapters for mixed-modality models?
- Is the current stability-rate proxy (from instability/fallback counts) sufficient, or should we add gradient explosion diagnostics?
- Do we need per-family normalization statistics instead of global EMA normalization when running mixed-family training curricula?
- For scale holdouts, should we normalize by parameter count tier more explicitly (e.g., learned scale embeddings)?
- Should overhead impact report pure optimizer step time from diagnostics in addition to end-to-end wall-clock proxy?

## Stage 4 assumptions
- Transformer-token controllers can capture richer cross-feature interactions than compact MLP adapters for some model/task niches.
- Residual direct-update predictors must stay as bounded corrections to safe base updates during initial R&D.
- Token feature subset ablations are necessary to identify whether metadata tokens justify overhead.
- Wall-clock overhead must be treated as first-class; step-efficiency gains alone are not sufficient.

## Stage 4 open questions
- Should transformer controllers be conditioned with explicit layer-position embeddings beyond scalar token indices?
- Is residual trust radius best as a global scalar or should it be parameterized by module type/family?
- Which instability metric should gate promotion to Stage 5: fallback events, hard run failures, or both?
- Should go/no-go criteria include variance across seeds rather than mean-only deltas?
- Are there specific task families where transformer overhead can be amortized enough to justify productionization?

## Diagnostics subsystem assumptions
- A separate diagnostics model is preferable initially so optimizer-control policy behavior remains stable and independently testable.
- Future-window labels from trajectory logs are a valid starting proxy before adding richer run metadata.
- Probabilistic outputs plus calibration checks are required for actionable stop/restart/tune recommendations.

## Diagnostics subsystem open questions
- Should stall labeling horizon be fixed globally or adapted by task family and eval cadence?
- Does practical decision utility improve with cost-sensitive thresholding per experiment budget?
- Should hyperparameter-mismatch targets be augmented with explicit search-space context from tuning logs?
- What criteria should gate future joint multi-task training with optimizer-control policy?

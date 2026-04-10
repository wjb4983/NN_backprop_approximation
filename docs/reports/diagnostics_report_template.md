# Training Diagnostics Evaluation Report Template

## Experiment metadata
- Date:
- Commit SHA:
- Dataset / task family:
- Run config(s):
- Diagnostics config:

## Label-generation settings
- Horizon `H`:
- Min improvement threshold:
- Noise multiplier:
- Instability jump threshold:
- HP mismatch stall-window requirement:

## Core event metrics
| Task | AUROC | AUPRC | ECE |
|---|---:|---:|---:|
| health_now | | | |
| stall_next_h | | | |
| meaningful_progress | | | |
| instability_risk | | | |
| hp_mismatch | | | |

## Lead-time utility
- Mean lead-time to stall warning (steps):
- Distribution notes (P50/P90):

## Practical decision utility
- Decision utility score (stop/restart/tune):
- Suggested thresholds:
  - stop_threshold:
  - tune_threshold:

## Calibration notes
- Reliability observations:
- Tasks with over/under-confidence:

## Integration observations
- Was `training_diagnostics` emitted during live runs?
- Do diagnostics align with visible curve behavior?

## Risks / TODO
- Risk 1:
- Risk 2:
- TODO 1:
- TODO 2:

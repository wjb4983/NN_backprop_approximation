#!/usr/bin/env bash
set -euo pipefail

MATRIX_CONFIG="${1:-configs/stage0/fair_baseline_matrix.yaml}"
RUN_CONFIG="${2:-configs/run.yaml}"

mkdir -p outputs

timeout 60s python - <<'PY' "$MATRIX_CONFIG"
import sys
import yaml

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
required = ["task_configs", "optimizer_configs", "seeds", "timeout_seconds"]
missing = [k for k in required if k not in cfg]
if missing:
    raise SystemExit(f"Missing required keys in {path}: {missing}")
print(f"[stage0] matrix={path}")
print(f"[stage0] tasks={len(cfg['task_configs'])} optimizers={len(cfg['optimizer_configs'])} seeds={len(cfg['seeds'])}")
PY

readarray -t RUN_COMMANDS < <(timeout 60s python - <<'PY' "$MATRIX_CONFIG" "$RUN_CONFIG"
import copy
import shlex
import sys
import tempfile
from pathlib import Path

import yaml

matrix_path, run_config_path = sys.argv[1], sys.argv[2]
with open(matrix_path, "r", encoding="utf-8") as f:
    matrix = yaml.safe_load(f)
with open(run_config_path, "r", encoding="utf-8") as f:
    run_base = yaml.safe_load(f)

timeout_s = int(matrix["timeout_seconds"]["bench_run"])
for task_cfg in matrix["task_configs"]:
    task_tag = Path(task_cfg).stem
    for opt_cfg in matrix["optimizer_configs"]:
        opt_tag = Path(opt_cfg).stem
        for seed in matrix["seeds"]:
            run_cfg = copy.deepcopy(run_base)
            run_cfg["seed"] = int(seed)
            run_cfg["experiment_name"] = f"stage0_{task_tag}_{opt_tag}_seed{seed}"
            tmp = tempfile.NamedTemporaryFile(prefix="stage0_run_", suffix=".yaml", delete=False)
            tmp_path = Path(tmp.name)
            tmp.close()
            with tmp_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(run_cfg, handle, sort_keys=False)
            cmd = (
                f"timeout {timeout_s}s bench-run --task-config {shlex.quote(task_cfg)} "
                f"--optimizer-config {shlex.quote(opt_cfg)} --run-config {shlex.quote(str(tmp_path))}"
            )
            print(cmd)
PY
)

for cmd in "${RUN_COMMANDS[@]}"; do
  echo "[stage0] $cmd"
  bash -lc "$cmd" || true
done

# run optional tuning jobs (validation-model selection protocol)
timeout 120s python - <<'PY' "$MATRIX_CONFIG" "$RUN_CONFIG"
import subprocess
import sys
from pathlib import Path

import yaml

matrix_path, run_config_path = sys.argv[1], sys.argv[2]
with open(matrix_path, "r", encoding="utf-8") as f:
    matrix = yaml.safe_load(f)

jobs = matrix.get("search_jobs", [])
timeout_s = int(matrix["timeout_seconds"].get("bench_tune", 600))
for idx, job in enumerate(jobs):
    cmd = [
        "timeout", f"{timeout_s}s", "bench-tune",
        "--task-config", job["task_config"],
        "--run-config", run_config_path,
        "--search-config", job["search_config"],
    ]
    print("[stage0]", " ".join(cmd))
    out = subprocess.run(cmd, check=False, capture_output=True, text=True)
    Path("outputs").mkdir(parents=True, exist_ok=True)
    Path(f"outputs/stage0_tune_job{idx}.json").write_text(out.stdout or "", encoding="utf-8")
    if out.returncode != 0:
        print(f"[stage0] warning: tuning job {idx} exit={out.returncode}")
PY

echo "[stage0] complete"
echo "[stage0] next: timeout 60s python scripts/summarize_stage0_results.py outputs"

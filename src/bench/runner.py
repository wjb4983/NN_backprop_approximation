"""Experiment runner that executes one optimizer-task configuration."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import ExperimentConfig
from .logger import JsonlLogger
from .metrics import auc_early, final_metric_at_budget, threshold_metrics
from .optimizers import build_optimizer
from .tasks import build_task
from .tasks.base import evaluate_model
from .utils import set_seed, to_serializable


class RunFailure(RuntimeError):
    """Raised when a run becomes unstable (NaN/Inf) or otherwise fails."""


def run_experiment(cfg: ExperimentConfig) -> dict:
    """Run a single experiment and return summarized metrics."""
    set_seed(cfg.run.seed)
    device = torch.device(cfg.run.device)

    task_params = dict(cfg.task.params)
    task_params.setdefault("batch_size", cfg.run.batch_size)
    task_params.setdefault("val_batch_size", cfg.run.val_batch_size)
    task_params.setdefault("num_workers", cfg.run.num_workers)
    task_params.setdefault("train_fraction", cfg.run.train_fraction)
    task_params.setdefault("val_fraction", cfg.run.val_fraction)
    task_params.setdefault("test_fraction", cfg.run.test_fraction)

    bundle = build_task(cfg.task.name, task_params, cfg.run.seed, cfg.task.family)
    model = bundle.model.to(device)
    optimizer, scheduler = build_optimizer(model, cfg.optimizer, cfg.run.max_steps)

    run_dir = Path(cfg.run.log_dir) / cfg.run.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = JsonlLogger(run_dir / "metrics.jsonl")

    summary: dict = {
        "failure": False,
        "failure_reason": None,
        "train_eval": [],
        "val_eval": [],
        "test_eval": [],
        "config": to_serializable(asdict(cfg)),
    }

    step = 0
    try:
        while step < cfg.run.max_steps:
            model.train()
            for x, y in bundle.train_loader:
                x = x.to(device)
                y = y.to(device)

                step_t0 = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)

                fw_t0 = time.perf_counter()
                logits = model(x)
                loss = bundle.criterion(logits, y)
                fw_ms = (time.perf_counter() - fw_t0) * 1000.0

                loss_value = float(loss.item())
                if not math.isfinite(loss_value):
                    raise RunFailure(f"Non-finite loss at step={step}: {loss_value}")

                if hasattr(optimizer, "update_loss"):
                    optimizer.update_loss(loss_value)

                bw_t0 = time.perf_counter()
                loss.backward()
                bw_ms = (time.perf_counter() - bw_t0) * 1000.0

                opt_t0 = time.perf_counter()
                optimizer.step()
                opt_ms = (time.perf_counter() - opt_t0) * 1000.0
                if scheduler is not None:
                    scheduler.step()

                step_ms = (time.perf_counter() - step_t0) * 1000.0

                step += 1

                if step % cfg.run.eval_every == 0 or step == cfg.run.max_steps:
                    train_metrics = evaluate_model(model, bundle.train_loader, bundle.criterion, device)
                    val_metrics = evaluate_model(model, bundle.val_loader, bundle.criterion, device)
                    test_metrics = evaluate_model(model, bundle.test_loader, bundle.criterion, device)

                    optimizer_diagnostics = optimizer.get_diagnostics() if hasattr(optimizer, "get_diagnostics") else {}
                    step_profile = {
                        "forward_ms": float(fw_ms),
                        "backward_ms": float(bw_ms),
                        "optimizer_ms": float(opt_ms),
                        "step_ms": float(step_ms),
                        "optimizer_share": float(opt_ms / max(step_ms, 1e-12)),
                    }
                    record = {
                        "step": step,
                        "train": train_metrics,
                        "val": val_metrics,
                        "test": test_metrics,
                        "lr": optimizer.param_groups[0]["lr"],
                        "step_profile": step_profile,
                        "optimizer_diagnostics": optimizer_diagnostics,
                    }
                    logger.log(record)
                    summary["train_eval"].append({"step": step, **train_metrics, "wall_clock_sec": logger.elapsed})
                    summary["val_eval"].append({"step": step, **val_metrics, "wall_clock_sec": logger.elapsed})
                    summary["test_eval"].append({"step": step, **test_metrics, "wall_clock_sec": logger.elapsed})

                if step >= cfg.run.max_steps:
                    break
    except Exception as exc:  # intentional broad catch so failures are counted explicitly.
        summary["failure"] = True
        summary["failure_reason"] = str(exc)

    metric_name = cfg.run.threshold_metric
    val_values = [m[metric_name] for m in summary["val_eval"] if metric_name in m]
    val_steps = [m["step"] for m in summary["val_eval"] if metric_name in m]
    wall_secs = [m["wall_clock_sec"] for m in summary["val_eval"] if metric_name in m]

    threshold = threshold_metrics(
        values=val_values,
        steps=val_steps,
        wall_secs=wall_secs,
        threshold=cfg.run.threshold_value,
        mode=cfg.run.threshold_mode,
    )
    fallback_events = 0
    if hasattr(optimizer, "get_diagnostics"):
        fallback_events = int(optimizer.get_diagnostics().get("fallback_events", 0))

    summary["metrics"] = {
        "steps_to_threshold": threshold.steps_to_threshold,
        "wall_clock_to_threshold": threshold.wall_clock_to_threshold,
        "auc_early_window": auc_early(val_values, val_steps, cfg.run.early_window_steps),
        "final_metric_at_budget": final_metric_at_budget(val_values),
        "instability_failure_count": int(bool(summary["failure"])) + fallback_events,
        "fallback_events": fallback_events,
    }

    summary_path = run_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(to_serializable(summary), handle, indent=2)

    return summary

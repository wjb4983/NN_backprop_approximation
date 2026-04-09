"""Configuration loaders for benchmark tasks, optimizers, and runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any



@dataclass
class TaskConfig:
    """Task-specific configuration.

    Attributes:
        name: Registered task name.
        family: Family identifier used to make deterministic family-level splits.
        params: Arbitrary task parameters passed to task factory.
    """

    name: str
    family: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    """Optimizer-specific configuration."""

    name: str
    lr: float
    weight_decay: float = 0.0
    momentum: float = 0.0
    schedule: str | None = None
    schedule_params: dict[str, Any] = field(default_factory=dict)
    extra_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunConfig:
    """Top-level run configuration."""

    experiment_name: str
    seed: int
    device: str
    max_steps: int
    eval_every: int
    threshold_metric: str
    threshold_value: float
    threshold_mode: str
    early_window_steps: int
    log_dir: str
    batch_size: int
    val_batch_size: int
    num_workers: int
    train_fraction: float
    val_fraction: float
    test_fraction: float


@dataclass
class ExperimentConfig:
    """Combined experiment configuration object."""

    run: RunConfig
    task: TaskConfig
    optimizer: OptimizerConfig


def _load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml  # local import so core runtime works without YAML unless CLI loader is used.

    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_experiment_config(task_cfg: str | Path, optimizer_cfg: str | Path, run_cfg: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config triplet."""
    task_data = _load_yaml(task_cfg)
    opt_data = _load_yaml(optimizer_cfg)
    run_data = _load_yaml(run_cfg)

    return ExperimentConfig(
        run=RunConfig(**run_data),
        task=TaskConfig(**task_data),
        optimizer=OptimizerConfig(**opt_data),
    )

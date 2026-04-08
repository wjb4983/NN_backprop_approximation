"""Task registry for benchmark harness."""

from __future__ import annotations

from .cnn_mnist import build_task as build_cnn_mnist
from .tabular_synth import build_task as build_tabular_synth

TASK_REGISTRY = {
    "cnn_mnist": build_cnn_mnist,
    "tabular_synth": build_tabular_synth,
}


def build_task(name: str, params: dict, seed: int, family: str):
    """Dispatch to task-specific bundle builders."""
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task '{name}'. Available: {sorted(TASK_REGISTRY)}")
    return TASK_REGISTRY[name](params=params, base_seed=seed, family=family)

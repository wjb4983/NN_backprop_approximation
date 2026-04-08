"""Synthetic tabular classification task for fast reproducible baselines."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils import family_seed
from .base import TaskBundle


class MLP(nn.Module):
    """Simple MLP baseline for tabular data."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _make_data(seed: int, n_samples: int, n_features: int, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    w = rng.normal(size=(n_features, n_classes)).astype(np.float32)
    logits = x @ w + 0.1 * rng.normal(size=(n_samples, n_classes)).astype(np.float32)
    y = logits.argmax(axis=1).astype(np.int64)
    return x, y


def build_task(params: dict, base_seed: int, family: str) -> TaskBundle:
    """Build synthetic tabular task with deterministic family split."""
    seed = family_seed(base_seed, family)
    x, y = _make_data(
        seed=seed,
        n_samples=params.get("n_samples", 10_000),
        n_features=params.get("n_features", 32),
        n_classes=params.get("n_classes", 4),
    )
    n = len(x)
    train_n = int(n * params["train_fraction"])
    val_n = int(n * params["val_fraction"])

    train_ds = TensorDataset(torch.from_numpy(x[:train_n]), torch.from_numpy(y[:train_n]))
    val_ds = TensorDataset(torch.from_numpy(x[train_n : train_n + val_n]), torch.from_numpy(y[train_n : train_n + val_n]))
    test_ds = TensorDataset(torch.from_numpy(x[train_n + val_n :]), torch.from_numpy(y[train_n + val_n :]))

    train_loader = DataLoader(train_ds, batch_size=params["batch_size"], shuffle=True, num_workers=params.get("num_workers", 0))
    val_loader = DataLoader(val_ds, batch_size=params.get("val_batch_size", params["batch_size"]), shuffle=False, num_workers=params.get("num_workers", 0))
    test_loader = DataLoader(test_ds, batch_size=params.get("val_batch_size", params["batch_size"]), shuffle=False, num_workers=params.get("num_workers", 0))

    return TaskBundle(
        model=MLP(
            in_dim=params.get("n_features", 32),
            hidden=params.get("hidden_dim", 128),
            out_dim=params.get("n_classes", 4),
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        metric_name="accuracy",
        metric_mode="max",
    )

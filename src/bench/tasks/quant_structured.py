"""Structured non-image task inspired by quant-style regime classification.

We synthesize multi-factor return sequences with latent volatility regimes and ask
models to predict the next-step return regime from recent engineered features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..utils import family_seed
from .base import TaskBundle


class QuantMLP(nn.Module):
    """Small MLP for structured factor features."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _simulate_factor_returns(
    seed: int,
    n_steps: int,
    n_factors: int,
    lookback: int,
    noise_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic synthetic factor features with regime labels.

    Regimes are defined by quantiles of next-step aggregated return to make this a
    multi-class classification problem (down/flat/up).
    """
    rng = np.random.default_rng(seed)

    regime = rng.integers(0, 3, size=n_steps)
    vol = np.choose(regime, [0.3, 0.8, 1.4]).astype(np.float32)

    factors = np.zeros((n_steps, n_factors), dtype=np.float32)
    for t in range(1, n_steps):
        ar = 0.7 * factors[t - 1]
        shock = vol[t] * rng.normal(size=n_factors).astype(np.float32)
        macro = 0.05 * np.sin(0.01 * t) + 0.03 * np.cos(0.02 * t)
        factors[t] = ar + shock + macro

    agg_ret = factors.mean(axis=1) + noise_scale * rng.normal(size=n_steps).astype(np.float32)

    feats: list[np.ndarray] = []
    targets: list[int] = []
    for t in range(lookback, n_steps - 1):
        window = factors[t - lookback : t]
        feat = np.concatenate(
            [
                window.mean(axis=0),
                window.std(axis=0),
                window[-1],
            ],
            axis=0,
        )
        nxt = float(agg_ret[t + 1])
        if nxt < -0.25:
            y = 0
        elif nxt > 0.25:
            y = 2
        else:
            y = 1
        feats.append(feat.astype(np.float32))
        targets.append(y)

    return np.stack(feats, axis=0), np.array(targets, dtype=np.int64)


def build_task(params: dict, base_seed: int, family: str) -> TaskBundle:
    """Build synthetic structured task with deterministic family split."""
    seed = family_seed(base_seed, family)
    x, y = _simulate_factor_returns(
        seed=seed,
        n_steps=params.get("n_steps", 14_000),
        n_factors=params.get("n_factors", 12),
        lookback=params.get("lookback", 12),
        noise_scale=params.get("noise_scale", 0.15),
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
        model=QuantMLP(
            in_dim=x.shape[1],
            hidden_dim=params.get("hidden_dim", 160),
            out_dim=3,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        metric_name="accuracy",
        metric_mode="max",
    )

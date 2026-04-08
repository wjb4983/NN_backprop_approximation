"""Optimizer and scheduler factory functions."""

from __future__ import annotations

from torch import nn, optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..config import OptimizerConfig


def build_optimizer(model: nn.Module, cfg: OptimizerConfig, max_steps: int):
    """Create optimizer and optional scheduler from config."""
    params = model.parameters()
    name = cfg.name.lower()

    if name == "adamw":
        optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif name in {"sgd_momentum", "sgd+momentum", "sgd"}:
        optimizer = optim.SGD(params, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    elif name in {"adamw_cosine", "adamw+cosine"}:
        optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.name}")

    scheduler = None
    if cfg.schedule == "cosine" or name in {"adamw_cosine", "adamw+cosine"}:
        t_max = int(cfg.schedule_params.get("t_max", max_steps))
        eta_min = float(cfg.schedule_params.get("eta_min", 0.0))
        scheduler = CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    return optimizer, scheduler

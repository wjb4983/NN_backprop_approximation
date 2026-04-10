"""Diagnostics prediction models.

Current baseline is a compact multi-task MLP with probabilistic binary outputs.
It is deliberately modular so we can later co-train with optimizer-control policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DiagnosticsTaskSpec:
    """Task naming contract for diagnostics outputs."""

    names: tuple[str, ...] = (
        "health_now",
        "stall_next_h",
        "meaningful_progress",
        "instability_risk",
        "hp_mismatch",
    )


class DiagnosticsMLP(nn.Module):
    """Multi-task probabilistic diagnostics head with optional MC-dropout."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.15) -> None:
        super().__init__()
        self.task_spec = DiagnosticsTaskSpec()
        out_dim = len(self.task_spec.names)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(x)
        return self.head(hidden)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))

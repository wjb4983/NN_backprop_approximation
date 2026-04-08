"""Task interface for benchmark task families."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader


@dataclass
class TaskBundle:
    """Container for model, criterion, and split loaders."""

    model: nn.Module
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    criterion: nn.Module
    metric_name: str
    metric_mode: str  # "min" or "max"


def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    """Evaluate model and return loss + accuracy-like metric when possible."""
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            total += x.size(0)
            if logits.ndim == 2 and y.ndim == 1:
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == y).sum().item())

    avg_loss = total_loss / max(total, 1)
    metrics = {"loss": avg_loss}
    if total > 0 and correct > 0:
        metrics["accuracy"] = correct / total
    return metrics

"""CNN baseline task on MNIST."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ..utils import family_seed
from .base import TaskBundle


class SmallCNN(nn.Module):
    """Compact CNN used for quick benchmark iterations."""

    def __init__(self, channels: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear((channels * 2) * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SplitConfig:
    train_fraction: float
    val_fraction: float
    test_fraction: float


def _split_indices(total: int, seed: int, split: SplitConfig) -> tuple[list[int], list[int], list[int]]:
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(total, generator=g).tolist()
    train_n = int(total * split.train_fraction)
    val_n = int(total * split.val_fraction)
    train_idx = order[:train_n]
    val_idx = order[train_n : train_n + val_n]
    test_idx = order[train_n + val_n :]
    return train_idx, val_idx, test_idx


def build_task(params: dict, base_seed: int, family: str) -> TaskBundle:
    """Create MNIST task bundle with family-level split determinism."""
    split = SplitConfig(
        train_fraction=params["train_fraction"],
        val_fraction=params["val_fraction"],
        test_fraction=params["test_fraction"],
    )
    dataset_root = params.get("dataset_root", "./data")
    transform = transforms.ToTensor()
    full = datasets.MNIST(root=dataset_root, train=True, download=True, transform=transform)

    split_seed = family_seed(base_seed, family)
    train_idx, val_idx, test_idx = _split_indices(len(full), split_seed, split)

    train_loader = DataLoader(
        Subset(full, train_idx),
        batch_size=params["batch_size"],
        shuffle=True,
        num_workers=params.get("num_workers", 0),
    )
    val_loader = DataLoader(
        Subset(full, val_idx),
        batch_size=params.get("val_batch_size", params["batch_size"]),
        shuffle=False,
        num_workers=params.get("num_workers", 0),
    )
    test_loader = DataLoader(
        Subset(full, test_idx),
        batch_size=params.get("val_batch_size", params["batch_size"]),
        shuffle=False,
        num_workers=params.get("num_workers", 0),
    )

    return TaskBundle(
        model=SmallCNN(channels=params.get("channels", 16)),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        metric_name="accuracy",
        metric_mode="max",
    )

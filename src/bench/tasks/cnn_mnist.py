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
    """Compact CNN used for quick benchmark iterations.

    Stage 2 adds optional depth/width scaling for a larger CNN variant while
    keeping the same task API and dataset.
    """

    def __init__(self, channels: int = 16, num_blocks: int = 2, fc_hidden: int = 128) -> None:
        super().__init__()
        if num_blocks < 2:
            raise ValueError("num_blocks must be >= 2 to preserve final feature map size assumptions")

        blocks: list[nn.Module] = []
        in_ch = 1
        cur_ch = channels
        for block_idx in range(num_blocks):
            blocks.extend(
                [
                    nn.Conv2d(in_ch, cur_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(cur_ch, cur_ch, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_ch = cur_ch
            if block_idx < num_blocks - 1:
                cur_ch = min(cur_ch * 2, channels * 4)

        spatial = 28 // (2**num_blocks)
        if spatial <= 0:
            raise ValueError("num_blocks too large for 28x28 MNIST input")

        self.net = nn.Sequential(
            *blocks,
            nn.Flatten(),
            nn.Linear(in_ch * spatial * spatial, fc_hidden),
            nn.ReLU(),
            nn.Linear(fc_hidden, 10),
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
        model=SmallCNN(
            channels=params.get("channels", 16),
            num_blocks=params.get("num_blocks", 2),
            fc_hidden=params.get("fc_hidden", 128),
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=nn.CrossEntropyLoss(),
        metric_name="accuracy",
        metric_mode="max",
    )

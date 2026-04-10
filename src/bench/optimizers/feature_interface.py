"""Unified optimizer feature interface for cross-family generalization (Stage 3).

This module standardizes per-parameter feature extraction so the learned optimizer
can consume a common schema across model families (vision/CNN-like, tabular, and
structured quant-style models).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class LayerMetadata:
    """Static layer metadata used as optimizer-side tokens."""

    depth: int
    param_count: int
    ndim: int
    shape_mean: float
    shape_max: float
    role_id: int
    type_id: int


class UnifiedFeatureExtractor:
    """Extract normalized statistics + metadata tokens for optimizer control.

    Feature schema is split into three blocks:
      1) normalized_stats: gradient/parameter dynamics normalized by running stats,
      2) metadata_tokens: depth/shape/param-count/role/type one-hot tokens,
      3) family_token: optional one-hot token for task-family/modality.

    Normalization uses running EMA moments for each scalar feature index to make
    features more stable across architectures and scales.
    """

    ROLE_VOCAB = ["weight", "bias", "norm", "embed", "other"]

    def __init__(
        self,
        model: nn.Module,
        model_family: str | None = None,
        family_vocab: list[str] | None = None,
        norm_ema_alpha: float = 0.05,
    ) -> None:
        self.model = model
        self.norm_ema_alpha = float(min(max(norm_ema_alpha, 1e-4), 1.0))

        self.layer_meta = self._build_layer_meta(model)
        self.type_count = max((meta.type_id for meta in self.layer_meta.values()), default=0) + 1

        self.family_vocab = family_vocab or ["vision", "structured", "tabular", "unknown"]
        self.family_to_idx = {name: idx for idx, name in enumerate(self.family_vocab)}
        self.model_family = self._resolve_family_token(model_family)

        self.stats_dim = 8
        self.meta_scalar_dim = 5
        self.meta_one_hot_dim = len(self.ROLE_VOCAB) + self.type_count
        self.family_dim = len(self.family_vocab)

        self._running_mean = torch.zeros(self.stats_dim, dtype=torch.float32)
        self._running_var = torch.ones(self.stats_dim, dtype=torch.float32)
        self._norm_bootstrapped = False

    @property
    def output_dim(self) -> int:
        """Return total feature dimension."""
        return self.stats_dim + self.meta_scalar_dim + self.meta_one_hot_dim + self.family_dim

    def _resolve_family_token(self, override: str | None) -> str:
        if override:
            key = override.lower().strip()
            return key if key in self.family_to_idx else "unknown"

        has_conv = any(isinstance(mod, nn.Conv2d) for mod in self.model.modules())
        has_linear = any(isinstance(mod, nn.Linear) for mod in self.model.modules())
        if has_conv:
            return "vision"
        if has_linear:
            return "structured"
        return "unknown"

    @staticmethod
    def _infer_role(param_name: str, module_type: str) -> int:
        lower_name = param_name.lower()
        lower_type = module_type.lower()
        if "bias" in lower_name:
            return 1
        if "norm" in lower_name or "norm" in lower_type:
            return 2
        if "embed" in lower_name or "embedding" in lower_type:
            return 3
        if "weight" in lower_name:
            return 0
        return 4

    def _build_layer_meta(self, model: nn.Module) -> dict[str, LayerMetadata]:
        module_type_to_id: dict[str, int] = {}
        param_to_meta: dict[str, LayerMetadata] = {}

        for module_name, module in model.named_modules():
            layer_type = module.__class__.__name__
            if layer_type not in module_type_to_id:
                module_type_to_id[layer_type] = len(module_type_to_id)

            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                shape = list(param.shape)
                shape_mean = float(sum(shape) / max(len(shape), 1)) if shape else 1.0
                role_id = self._infer_role(param_name=param_name, module_type=layer_type)
                param_to_meta[full_name] = LayerMetadata(
                    depth=full_name.count(".") + 1,
                    param_count=param.numel(),
                    ndim=param.ndim,
                    shape_mean=shape_mean,
                    shape_max=float(max(shape) if shape else 1.0),
                    role_id=role_id,
                    type_id=module_type_to_id[layer_type],
                )
        return param_to_meta

    def _normalize_stats(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply running-feature normalization with EMA moments."""
        if not self._norm_bootstrapped:
            self._running_mean = raw.detach().clone()
            self._running_var = torch.ones_like(raw)
            self._norm_bootstrapped = True
            return torch.zeros_like(raw)

        alpha = self.norm_ema_alpha
        delta = raw - self._running_mean
        self._running_mean = (1.0 - alpha) * self._running_mean + alpha * raw
        self._running_var = (1.0 - alpha) * self._running_var + alpha * delta.pow(2)
        return (raw - self._running_mean) / torch.sqrt(self._running_var + 1e-6)

    def _meta_scalars(self, meta: LayerMetadata) -> torch.Tensor:
        return torch.tensor(
            [
                float(meta.depth),
                math.log1p(float(meta.param_count)),
                float(meta.ndim),
                math.log1p(float(meta.shape_mean)),
                math.log1p(float(meta.shape_max)),
            ],
            dtype=torch.float32,
        )

    def _meta_one_hot(self, meta: LayerMetadata) -> torch.Tensor:
        token = torch.zeros(self.meta_one_hot_dim, dtype=torch.float32)
        token[meta.role_id] = 1.0
        token[len(self.ROLE_VOCAB) + meta.type_id] = 1.0
        return token

    def _family_one_hot(self) -> torch.Tensor:
        token = torch.zeros(self.family_dim, dtype=torch.float32)
        token[self.family_to_idx.get(self.model_family, self.family_to_idx["unknown"])] = 1.0
        return token

    def build_feature(self, name: str, stats: dict[str, Any]) -> torch.Tensor:
        """Create one unified feature vector for a parameter tensor."""
        meta = self.layer_meta.get(name)
        if meta is None:
            meta = LayerMetadata(depth=1, param_count=1, ndim=1, shape_mean=1.0, shape_max=1.0, role_id=4, type_id=0)

        raw_stats = torch.tensor(
            [
                math.log1p(float(stats["grad_norm"])),
                math.log1p(float(stats["param_norm"])),
                math.log1p(float(stats["m_mean_abs"])),
                math.log1p(float(stats["v_mean"])),
                float(stats["grad_to_param_ratio"]),
                float(stats["loss_delta"]),
                float(stats["loss_trend"]),
                float(stats["update_norm_hint"]),
            ],
            dtype=torch.float32,
        )

        normalized = self._normalize_stats(raw_stats)
        return torch.cat([normalized, self._meta_scalars(meta), self._meta_one_hot(meta), self._family_one_hot()], dim=0)

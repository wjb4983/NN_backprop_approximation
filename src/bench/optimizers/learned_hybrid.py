"""Hybrid AdamW optimizer modulated by a tiny learned layerwise controller.

Stage 1 goal is a stable and inspectable prototype, so this implementation prioritizes:
- bounded controller outputs,
- explicit stability guardrails,
- deterministic fallback behavior to AdamW-style updates,
- lightweight diagnostics for interpretability.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


@dataclass
class LayerMeta:
    """Metadata and static descriptors associated with one parameter tensor."""

    name: str
    depth: int
    param_count: int
    type_id: int


class TinyLayerwiseController(nn.Module):
    """Small MLP that maps per-layer features to bounded modulation coefficients."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class LearnedHybridAdamW(Optimizer):
    """AdamW update with learned per-layer modulation and strong safety clamps.

    The controller outputs three multiplicative knobs per layer:
    1) LR scale
    2) Momentum correction factor
    3) Trust/clipping multiplier

    If instability is detected (non-finite values or over-large updates), this
    optimizer temporarily falls back to unmodulated AdamW-style behavior.
    """

    def __init__(self, model: nn.Module, defaults: dict[str, Any]) -> None:
        params = [p for p in model.parameters() if p.requires_grad]
        super().__init__(params, defaults)

        self.model = model
        self.beta1 = float(defaults.get("beta1", 0.9))
        self.beta2 = float(defaults.get("beta2", 0.999))
        self.eps = float(defaults.get("eps", 1e-8))
        self.lr = float(defaults.get("lr", 1e-3))
        self.weight_decay = float(defaults.get("weight_decay", 0.0))

        self.lr_scale_bound = float(defaults.get("lr_scale_bound", 0.5))
        self.mom_correction_bound = float(defaults.get("mom_correction_bound", 0.3))
        self.trust_bound = float(defaults.get("trust_bound", 0.5))
        self.max_layer_update_norm = float(defaults.get("max_layer_update_norm", 1.0))
        self.fallback_steps = int(defaults.get("fallback_steps", 20))
        self.loss_window = int(defaults.get("loss_window", 5))

        self.loss_history: deque[float] = deque(maxlen=max(self.loss_window, 2))
        self._fallback_remaining = 0
        self._fallback_events = 0
        self._total_steps = 0
        self._last_diag: dict[str, Any] = {}

        self.layer_meta = self._build_layer_meta(model)
        self._param_order = [name for name, p in model.named_parameters() if p.requires_grad]

        self.type_count = max((meta.type_id for meta in self.layer_meta.values()), default=0) + 1
        input_dim = 9 + self.type_count
        self.controller = TinyLayerwiseController(input_dim=input_dim, hidden_dim=int(defaults.get("controller_hidden_dim", 32)))

        pretrained_path = defaults.get("controller_pretrained_path")
        if pretrained_path:
            payload = torch.load(pretrained_path, map_location="cpu")
            self.controller.load_state_dict(payload)

    @staticmethod
    def _build_layer_meta(model: nn.Module) -> dict[str, LayerMeta]:
        module_type_to_id: dict[str, int] = {}
        param_to_meta: dict[str, LayerMeta] = {}

        for module_name, module in model.named_modules():
            layer_type = module.__class__.__name__
            if layer_type not in module_type_to_id:
                module_type_to_id[layer_type] = len(module_type_to_id)
            for param_name, param in module.named_parameters(recurse=False):
                full_name = f"{module_name}.{param_name}" if module_name else param_name
                depth = full_name.count(".") + 1
                param_to_meta[full_name] = LayerMeta(
                    name=full_name,
                    depth=depth,
                    param_count=param.numel(),
                    type_id=module_type_to_id[layer_type],
                )
        return param_to_meta

    def update_loss(self, loss_value: float) -> None:
        """Inject most recent scalar loss for trend feature computation."""
        if math.isfinite(loss_value):
            self.loss_history.append(float(loss_value))

    def _loss_features(self) -> tuple[float, float]:
        if len(self.loss_history) < 2:
            return 0.0, 0.0
        recent = list(self.loss_history)
        diff = recent[-1] - recent[-2]
        trend = (recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        return diff, trend

    def _build_features(
        self,
        name: str,
        grad_norm: float,
        param_norm: float,
        m_mean_abs: float,
        v_mean: float,
        ratio: float,
        loss_delta: float,
        loss_trend: float,
    ) -> torch.Tensor:
        meta = self.layer_meta.get(name)
        if meta is None:
            meta = LayerMeta(name=name, depth=1, param_count=1, type_id=0)

        type_one_hot = torch.zeros(self.type_count, dtype=torch.float32)
        if 0 <= meta.type_id < self.type_count:
            type_one_hot[meta.type_id] = 1.0

        scalars = torch.tensor(
            [
                math.log1p(grad_norm),
                math.log1p(param_norm),
                math.log1p(m_mean_abs),
                math.log1p(v_mean),
                ratio,
                loss_delta,
                loss_trend,
                float(meta.depth),
                math.log1p(float(meta.param_count)),
            ],
            dtype=torch.float32,
        )
        return torch.cat([scalars, type_one_hot], dim=0)

    def _bounded_controls(self, raw: torch.Tensor) -> tuple[float, float, float]:
        # tanh keeps outputs finite; bounds define conservative modulation ranges.
        bounded = torch.tanh(raw)
        lr_mult = 1.0 + self.lr_scale_bound * float(bounded[0].item())
        mom_mult = 1.0 + self.mom_correction_bound * float(bounded[1].item())
        trust_mult = 1.0 + self.trust_bound * float(bounded[2].item())
        return lr_mult, mom_mult, trust_mult

    @torch.no_grad()
    def step(self, closure=None):
        del closure

        self._total_steps += 1
        loss_delta, loss_trend = self._loss_features()

        used_fallback = self._fallback_remaining > 0
        if used_fallback:
            self._fallback_remaining -= 1

        lr_mult_values: list[float] = []
        mom_mult_values: list[float] = []
        trust_mult_values: list[float] = []
        max_update_norm = 0.0

        fallback_triggered_this_step = False

        for group in self.param_groups:
            lr = float(group.get("lr", self.lr))
            for param_name, p in zip(self._param_order, group["params"]):
                if p.grad is None:
                    continue
                grad = p.grad
                if not torch.isfinite(grad).all():
                    fallback_triggered_this_step = True
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                grad_norm = float(torch.linalg.vector_norm(grad).item())
                param_norm = float(torch.linalg.vector_norm(p).item())
                m_mean_abs = float(exp_avg.abs().mean().item())
                v_mean = float(exp_avg_sq.mean().item())
                ratio = grad_norm / (param_norm + 1e-12)

                features = self._build_features(
                    name=param_name,
                    grad_norm=grad_norm,
                    param_norm=param_norm,
                    m_mean_abs=m_mean_abs,
                    v_mean=v_mean,
                    ratio=ratio,
                    loss_delta=loss_delta,
                    loss_trend=loss_trend,
                )

                lr_mult = 1.0
                mom_mult = 1.0
                trust_mult = 1.0

                if not used_fallback:
                    raw = self.controller(features)
                    lr_mult, mom_mult, trust_mult = self._bounded_controls(raw)

                eff_beta1 = min(max(self.beta1 * mom_mult, 0.0), 0.9999)
                exp_avg.mul_(eff_beta1).add_(grad, alpha=1 - eff_beta1)
                exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)

                bias_correction1 = 1 - eff_beta1**t
                bias_correction2 = 1 - self.beta2**t
                denom = (exp_avg_sq.sqrt() / math.sqrt(max(bias_correction2, 1e-12))).add_(self.eps)
                update = (exp_avg / max(bias_correction1, 1e-12)) / denom

                # Trust multiplier acts as an additional clamp on update size.
                layer_update_norm = float(torch.linalg.vector_norm(update).item())
                allowed_norm = self.max_layer_update_norm * max(trust_mult, 1e-6)
                if layer_update_norm > allowed_norm > 0:
                    update.mul_(allowed_norm / (layer_update_norm + 1e-12))
                    layer_update_norm = allowed_norm

                max_update_norm = max(max_update_norm, layer_update_norm)
                if not math.isfinite(layer_update_norm):
                    fallback_triggered_this_step = True

                if self.weight_decay:
                    p.add_(p, alpha=-lr * self.weight_decay)
                p.add_(update, alpha=-lr * lr_mult)

                lr_mult_values.append(lr_mult)
                mom_mult_values.append(mom_mult)
                trust_mult_values.append(trust_mult)

        if fallback_triggered_this_step:
            self._fallback_remaining = self.fallback_steps
            self._fallback_events += 1

        self._last_diag = {
            "step": self._total_steps,
            "used_fallback": bool(used_fallback),
            "fallback_triggered": bool(fallback_triggered_this_step),
            "fallback_remaining": int(self._fallback_remaining),
            "fallback_events": int(self._fallback_events),
            "max_layer_update_norm": max_update_norm,
            "controller": {
                "lr_mult_mean": float(sum(lr_mult_values) / max(len(lr_mult_values), 1)),
                "mom_mult_mean": float(sum(mom_mult_values) / max(len(mom_mult_values), 1)),
                "trust_mult_mean": float(sum(trust_mult_values) / max(len(trust_mult_values), 1)),
                "num_layers": int(len(lr_mult_values)),
            },
        }

    def get_diagnostics(self) -> dict[str, Any]:
        """Return latest diagnostic snapshot for logging/debugging."""
        return dict(self._last_diag)


def build_learned_hybrid_optimizer(model: nn.Module, cfg: Any) -> LearnedHybridAdamW:
    """Factory for config-driven learned hybrid optimizer creation."""
    extras = dict(getattr(cfg, "extra_params", {}) or {})
    defaults = {
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "beta1": float(extras.get("beta1", 0.9)),
        "beta2": float(extras.get("beta2", 0.999)),
        "eps": float(extras.get("eps", 1e-8)),
        "controller_hidden_dim": int(extras.get("controller_hidden_dim", 32)),
        "lr_scale_bound": float(extras.get("lr_scale_bound", 0.5)),
        "mom_correction_bound": float(extras.get("mom_correction_bound", 0.3)),
        "trust_bound": float(extras.get("trust_bound", 0.5)),
        "max_layer_update_norm": float(extras.get("max_layer_update_norm", 1.0)),
        "fallback_steps": int(extras.get("fallback_steps", 20)),
        "loss_window": int(extras.get("loss_window", 5)),
        "controller_pretrained_path": extras.get("controller_pretrained_path"),
    }
    return LearnedHybridAdamW(model=model, defaults=defaults)

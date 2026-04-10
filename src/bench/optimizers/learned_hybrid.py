"""Stage 2 hybrid learned optimizer with lightweight temporal reasoning.

This module keeps Stage 1's conservative safety-first behavior while adding:
- optional temporal encoding over recent per-layer feature windows,
- optional gating between AdamW-like and momentum-SGD-like update modes,
- optional trust-ratio style modulation,
- stability regularization on control trajectories,
- lightweight profiling hooks for optimizer overhead diagnostics.
"""

from __future__ import annotations

from collections import deque
import math
import time
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from .feature_interface import UnifiedFeatureExtractor


class TemporalFeatureEncoder(nn.Module):
    """Encode a short feature history using GRU/MLP/off modes.

    Modes:
      - off: use the latest feature vector directly.
      - gru: run a tiny GRU over the window and return last hidden state.
      - mlp: flatten the whole window and pass through a tiny MLP.
    """

    def __init__(
        self,
        input_dim: int,
        window: int,
        mode: str = "off",
        hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.window = max(int(window), 1)
        self.mode = mode.lower()
        self.hidden_dim = int(hidden_dim)

        if self.mode == "off":
            self.output_dim = input_dim
            self.gru = None
            self.mlp = None
        elif self.mode == "gru":
            self.gru = nn.GRU(input_dim, self.hidden_dim, batch_first=True)
            self.mlp = None
            self.output_dim = self.hidden_dim
        elif self.mode == "mlp":
            self.gru = None
            self.mlp = nn.Sequential(
                nn.Linear(input_dim * self.window, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
            )
            self.output_dim = self.hidden_dim
        else:
            raise ValueError(f"Unsupported temporal encoder mode: {mode}")

    def forward(self, feature_window: torch.Tensor) -> torch.Tensor:
        """Encode a [T, D] window into a fixed-size vector."""
        if self.mode == "off":
            return feature_window[-1]
        if self.mode == "gru":
            out, _ = self.gru(feature_window.unsqueeze(0))
            return out[0, -1]

        # MLP mode: flatten the full window and project.
        return self.mlp(feature_window.reshape(1, -1)).squeeze(0)


class AdapterBackboneController(nn.Module):
    """Shared backbone controller with lightweight family adapters.

    The architecture uses:
      - shared backbone MLP on unified tensor features,
      - a small residual adapter per model family,
      - a shared output head that predicts optimizer controls.
    """

    def __init__(
        self,
        encoded_dim: int,
        hidden_dim: int = 64,
        adapter_dim: int = 16,
        family_keys: list[str] | None = None,
    ) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(encoded_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        keys = family_keys or ["vision", "structured", "tabular", "unknown"]
        self.adapters = nn.ModuleDict(
            {
                key: nn.Sequential(
                    nn.Linear(hidden_dim, adapter_dim),
                    nn.ReLU(),
                    nn.Linear(adapter_dim, hidden_dim),
                )
                for key in keys
            }
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, encoded_features: torch.Tensor, family_key: str) -> torch.Tensor:
        trunk = self.backbone(encoded_features)
        adapter = self.adapters[family_key] if family_key in self.adapters else self.adapters["unknown"]
        adapted = trunk + adapter(trunk)
        return self.head(adapted)


class LearnedHybridAdamW(Optimizer):
    """Stage 2 hybrid optimizer with safety clamps and optional learned controls."""

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

        # Stage 2 ablations.
        self.temporal_on = bool(defaults.get("temporal_on", True))
        self.temporal_encoder_mode = str(defaults.get("temporal_encoder_mode", "gru"))
        self.temporal_window = int(defaults.get("temporal_window", 4))
        self.gating_on = bool(defaults.get("gating_on", True))
        self.trust_modulation_on = bool(defaults.get("trust_modulation_on", True))

        # Stability regularization controls.
        self.multiplier_swing_penalty = float(defaults.get("multiplier_swing_penalty", 0.1))
        self.control_smoothness = float(defaults.get("control_smoothness", 0.25))

        # Trust-ratio style modulation guardrails.
        self.trust_ratio_min = float(defaults.get("trust_ratio_min", 0.25))
        self.trust_ratio_max = float(defaults.get("trust_ratio_max", 4.0))

        # Optional profiling for optimizer-only overhead.
        self.profile_overhead = bool(defaults.get("profile_overhead", True))
        self._overhead_ema_alpha = float(defaults.get("overhead_ema_alpha", 0.05))

        self.loss_history: deque[float] = deque(maxlen=max(self.loss_window, 2))
        self._fallback_remaining = 0
        self._fallback_events = 0
        self._total_steps = 0
        self._last_diag: dict[str, Any] = {}

        self._overhead_last_ms = 0.0
        self._overhead_ema_ms = 0.0

        self._param_order = [name for name, p in model.named_parameters() if p.requires_grad]
        self.model_family = str(defaults.get("model_family", "unknown")).strip().lower()
        family_vocab = list(defaults.get("family_vocab", ["vision", "structured", "tabular", "unknown"]))
        self.feature_extractor = UnifiedFeatureExtractor(
            model=model,
            model_family=self.model_family,
            family_vocab=family_vocab,
            norm_ema_alpha=float(defaults.get("feature_norm_ema_alpha", 0.05)),
        )
        self.input_dim = self.feature_extractor.output_dim

        encoder_mode = self.temporal_encoder_mode if self.temporal_on else "off"
        self.temporal_encoder = TemporalFeatureEncoder(
            input_dim=self.input_dim,
            window=self.temporal_window,
            mode=encoder_mode,
            hidden_dim=int(defaults.get("temporal_hidden_dim", 32)),
        )
        self.controller = AdapterBackboneController(
            encoded_dim=self.temporal_encoder.output_dim,
            hidden_dim=int(defaults.get("controller_hidden_dim", 64)),
            adapter_dim=int(defaults.get("adapter_hidden_dim", 16)),
            family_keys=family_vocab,
        )

        # Per-parameter control and feature histories for temporal reasoning/smoothing.
        self._feature_history: dict[str, deque[torch.Tensor]] = {
            name: deque(maxlen=self.temporal_window) for name in self._param_order
        }
        self._prev_controls: dict[str, tuple[float, float, float, float]] = {}

        pretrained_path = defaults.get("controller_pretrained_path")
        if pretrained_path:
            payload = torch.load(pretrained_path, map_location="cpu")
            self.controller.load_state_dict(payload)

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
        return self.feature_extractor.build_feature(
            name=name,
            stats={
                "grad_norm": grad_norm,
                "param_norm": param_norm,
                "m_mean_abs": m_mean_abs,
                "v_mean": v_mean,
                "grad_to_param_ratio": ratio,
                "loss_delta": loss_delta,
                "loss_trend": loss_trend,
                "update_norm_hint": grad_norm,
            },
        )

    def _encode_features(self, name: str, feature: torch.Tensor) -> torch.Tensor:
        history = self._feature_history[name]
        history.append(feature)
        if not history:
            return feature

        # Left-pad with earliest observation so temporal encoder always sees a fixed window length.
        first = history[0]
        padded = [first for _ in range(max(self.temporal_window - len(history), 0))] + list(history)
        window_tensor = torch.stack(padded[-self.temporal_window :], dim=0)
        return self.temporal_encoder(window_tensor)

    def _bounded_controls(self, raw: torch.Tensor) -> tuple[float, float, float, float]:
        bounded = torch.tanh(raw)
        lr_mult = 1.0 + self.lr_scale_bound * float(bounded[0].item())
        mom_mult = 1.0 + self.mom_correction_bound * float(bounded[1].item())
        trust_mult = 1.0 + self.trust_bound * float(bounded[2].item())
        gate = 0.5 * (float(bounded[3].item()) + 1.0)
        return lr_mult, mom_mult, trust_mult, gate

    def _regularize_controls(
        self,
        name: str,
        controls: tuple[float, float, float, float],
    ) -> tuple[tuple[float, float, float, float], float, float]:
        """Apply Stage 2 stability regularization on control trajectories.

        Returns regularized controls plus two penalties for diagnostics:
          - swing penalty (erratic multiplier jumps)
          - smoothness penalty (distance from smoothed outputs)
        """
        prev = self._prev_controls.get(name, (1.0, 1.0, 1.0, 0.5))
        swing = sum(abs(c - p) for c, p in zip(controls[:3], prev[:3])) / 3.0
        swing_penalty = self.multiplier_swing_penalty * swing

        # Penalize erratic swings by shrinking deviations from neutral controls.
        shrink = 1.0 / (1.0 + max(swing_penalty, 0.0))
        shrunk = (
            1.0 + (controls[0] - 1.0) * shrink,
            1.0 + (controls[1] - 1.0) * shrink,
            1.0 + (controls[2] - 1.0) * shrink,
            controls[3],
        )

        # Smoothness constraint: EMA with previous controls.
        alpha = min(max(self.control_smoothness, 0.0), 1.0)
        smooth = tuple(alpha * p + (1.0 - alpha) * c for p, c in zip(prev, shrunk))
        smoothness_penalty = sum(abs(c - s) for c, s in zip(shrunk, smooth)) / 4.0

        self._prev_controls[name] = smooth
        return smooth, swing_penalty, smoothness_penalty

    @torch.no_grad()
    def step(self, closure=None):
        del closure

        step_start = time.perf_counter()
        self._total_steps += 1
        loss_delta, loss_trend = self._loss_features()

        used_fallback = self._fallback_remaining > 0
        if used_fallback:
            self._fallback_remaining -= 1

        lr_mult_values: list[float] = []
        mom_mult_values: list[float] = []
        trust_mult_values: list[float] = []
        gate_values: list[float] = []
        max_update_norm = 0.0

        swing_penalties: list[float] = []
        smoothness_penalties: list[float] = []

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
                    state["sgd_momentum"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                sgd_buf = state["sgd_momentum"]

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
                gate = 1.0

                if not used_fallback:
                    encoded = self._encode_features(param_name, features)
                    raw = self.controller(encoded, family_key=self.feature_extractor.model_family)
                    controls = self._bounded_controls(raw)
                    controls, swing_penalty, smoothness_penalty = self._regularize_controls(param_name, controls)
                    lr_mult, mom_mult, trust_mult, gate = controls
                    swing_penalties.append(swing_penalty)
                    smoothness_penalties.append(smoothness_penalty)

                if not self.gating_on:
                    gate = 1.0
                if not self.trust_modulation_on:
                    trust_mult = 1.0

                eff_beta1 = min(max(self.beta1 * mom_mult, 0.0), 0.9999)

                # AdamW-like path.
                exp_avg.mul_(eff_beta1).add_(grad, alpha=1 - eff_beta1)
                exp_avg_sq.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                bias_correction1 = 1 - eff_beta1**t
                bias_correction2 = 1 - self.beta2**t
                denom = (exp_avg_sq.sqrt() / math.sqrt(max(bias_correction2, 1e-12))).add_(self.eps)
                adam_update = (exp_avg / max(bias_correction1, 1e-12)) / denom

                # Momentum-SGD-like path.
                sgd_buf.mul_(eff_beta1).add_(grad)
                sgd_update = sgd_buf

                # Gated blend between modes.
                update = gate * adam_update + (1.0 - gate) * sgd_update

                # Optional trust-ratio modulation (LAMB-like scalar trust ratio).
                layer_update_norm = float(torch.linalg.vector_norm(update).item())
                if self.trust_modulation_on:
                    trust_ratio = param_norm / (layer_update_norm + self.eps)
                    trust_ratio = min(max(trust_ratio, self.trust_ratio_min), self.trust_ratio_max)
                    update = update * (trust_mult * trust_ratio)
                    layer_update_norm = float(torch.linalg.vector_norm(update).item())

                # Absolute safety cap remains in place.
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
                gate_values.append(gate)

        if fallback_triggered_this_step:
            self._fallback_remaining = self.fallback_steps
            self._fallback_events += 1

        step_elapsed_ms = (time.perf_counter() - step_start) * 1000.0 if self.profile_overhead else 0.0
        self._overhead_last_ms = step_elapsed_ms
        if self._total_steps == 1:
            self._overhead_ema_ms = step_elapsed_ms
        else:
            a = self._overhead_ema_alpha
            self._overhead_ema_ms = a * step_elapsed_ms + (1.0 - a) * self._overhead_ema_ms

        self._last_diag = {
            "step": self._total_steps,
            "used_fallback": bool(used_fallback),
            "fallback_triggered": bool(fallback_triggered_this_step),
            "fallback_remaining": int(self._fallback_remaining),
            "fallback_events": int(self._fallback_events),
            "max_layer_update_norm": max_update_norm,
            "regularization": {
                "swing_penalty_mean": float(sum(swing_penalties) / max(len(swing_penalties), 1)),
                "smoothness_penalty_mean": float(sum(smoothness_penalties) / max(len(smoothness_penalties), 1)),
            },
            "profiling": {
                "optimizer_step_ms": float(self._overhead_last_ms),
                "optimizer_step_ema_ms": float(self._overhead_ema_ms),
            },
            "controller": {
                "lr_mult_mean": float(sum(lr_mult_values) / max(len(lr_mult_values), 1)),
                "mom_mult_mean": float(sum(mom_mult_values) / max(len(mom_mult_values), 1)),
                "trust_mult_mean": float(sum(trust_mult_values) / max(len(trust_mult_values), 1)),
                "gate_mean": float(sum(gate_values) / max(len(gate_values), 1)),
                "num_layers": int(len(lr_mult_values)),
                "ablations": {
                    "temporal_on": self.temporal_on,
                    "gating_on": self.gating_on,
                    "trust_modulation_on": self.trust_modulation_on,
                },
                "feature_interface": {
                    "model_family": self.feature_extractor.model_family,
                    "feature_dim": int(self.input_dim),
                },
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
        "controller_hidden_dim": int(extras.get("controller_hidden_dim", 64)),
        "adapter_hidden_dim": int(extras.get("adapter_hidden_dim", 16)),
        "temporal_hidden_dim": int(extras.get("temporal_hidden_dim", 32)),
        "model_family": str(extras.get("model_family", "unknown")),
        "family_vocab": list(extras.get("family_vocab", ["vision", "structured", "tabular", "unknown"])),
        "feature_norm_ema_alpha": float(extras.get("feature_norm_ema_alpha", 0.05)),
        "temporal_on": bool(extras.get("temporal_on", True)),
        "temporal_encoder_mode": str(extras.get("temporal_encoder_mode", "gru")),
        "temporal_window": int(extras.get("temporal_window", 4)),
        "gating_on": bool(extras.get("gating_on", True)),
        "trust_modulation_on": bool(extras.get("trust_modulation_on", True)),
        "multiplier_swing_penalty": float(extras.get("multiplier_swing_penalty", 0.1)),
        "control_smoothness": float(extras.get("control_smoothness", 0.25)),
        "trust_ratio_min": float(extras.get("trust_ratio_min", 0.25)),
        "trust_ratio_max": float(extras.get("trust_ratio_max", 4.0)),
        "profile_overhead": bool(extras.get("profile_overhead", True)),
        "overhead_ema_alpha": float(extras.get("overhead_ema_alpha", 0.05)),
        "lr_scale_bound": float(extras.get("lr_scale_bound", 0.5)),
        "mom_correction_bound": float(extras.get("mom_correction_bound", 0.3)),
        "trust_bound": float(extras.get("trust_bound", 0.5)),
        "max_layer_update_norm": float(extras.get("max_layer_update_norm", 1.0)),
        "fallback_steps": int(extras.get("fallback_steps", 20)),
        "loss_window": int(extras.get("loss_window", 5)),
        "controller_pretrained_path": extras.get("controller_pretrained_path"),
    }
    return LearnedHybridAdamW(model=model, defaults=defaults)

"""Runtime inference hook to log diagnostics alongside training curves."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .feature_pipeline import DiagnosticsFeaturePipeline
from .modeling import DiagnosticsMLP


@dataclass
class RuntimeDiagnosticsConfig:
    enabled: bool = False
    checkpoint_path: str | None = None
    mc_dropout_samples: int = 8


class RuntimeDiagnosticsHook:
    """Lightweight adapter to produce diagnostics from live training records."""

    def __init__(self, cfg: RuntimeDiagnosticsConfig) -> None:
        self.cfg = cfg
        self.pipeline = DiagnosticsFeaturePipeline()
        self.model: DiagnosticsMLP | None = None
        self.feature_names: list[str] | None = None

        if self.cfg.enabled and self.cfg.checkpoint_path:
            self._load_checkpoint(self.cfg.checkpoint_path)

    def _load_checkpoint(self, path: str | Path) -> None:
        payload = torch.load(path, map_location="cpu")
        input_dim = int(payload["input_dim"])
        self.feature_names = list(payload.get("feature_names", []))
        self.model = DiagnosticsMLP(input_dim=input_dim)
        self.model.load_state_dict(payload["model_state_dict"])
        self.model.eval()

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        if not self.cfg.enabled:
            return {}

        snap = self.pipeline.build_from_record(record)
        x = torch.tensor(snap.values, dtype=torch.float32).unsqueeze(0)

        if self.model is None:
            # Boot mode: emit only features so logs can still be inspected even
            # before a trained diagnostics model exists.
            return {
                "status": "feature_only",
                "feature_names": snap.feature_names,
                "features": snap.values,
            }

        probs = self.model.predict_proba(x).squeeze(0)
        tasks = self.model.task_spec.names

        # MC-dropout uncertainty estimate.
        samples = []
        self.model.train()
        for _ in range(max(self.cfg.mc_dropout_samples, 1)):
            samples.append(self.model.predict_proba(x).detach())
        self.model.eval()
        stack = torch.stack(samples, dim=0).squeeze(1)
        epistemic_std = stack.std(dim=0)

        out = {tasks[i]: float(probs[i].item()) for i in range(len(tasks))}
        out["epistemic_std"] = {tasks[i]: float(epistemic_std[i].item()) for i in range(len(tasks))}
        return out

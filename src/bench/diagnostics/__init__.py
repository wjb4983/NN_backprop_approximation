"""Training diagnostics subsystem for learned-optimizer experiments.

This package is intentionally separate from optimizer-control policy code so it can
be evolved independently, then optionally connected via future joint multi-task
training.
"""

from .feature_pipeline import DiagnosticsFeaturePipeline
from .inference import RuntimeDiagnosticsHook
from .modeling import DiagnosticsMLP

__all__ = [
    "DiagnosticsFeaturePipeline",
    "RuntimeDiagnosticsHook",
    "DiagnosticsMLP",
]

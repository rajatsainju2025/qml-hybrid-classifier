"""qml_hybrid — hybrid quantum-classical classifier package.

Exports the primary model and training utilities for external use.
"""

from .model import HybridQClassifier
from .circuits import make_vqc_circuit, weight_shape, meyer_wallach_expressibility
from .train import run_training
from .evaluate import compute_metrics, compare_to_baseline

__version__ = "0.1.0"
__all__ = [
    "HybridQClassifier",
    "make_vqc_circuit",
    "weight_shape",
    "meyer_wallach_expressibility",
    "run_training",
    "compute_metrics",
    "compare_to_baseline",
]

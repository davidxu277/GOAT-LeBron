"""Public evaluation API used by models and agents."""

from .evaluator import evaluate, format_metrics
from .metrics import binary_auc

__all__ = ["binary_auc", "evaluate", "format_metrics"]


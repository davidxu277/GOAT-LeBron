"""KuaiRand-Pure ↔ GOAT-LeBron stable integration boundary."""

from .dataset import DatasetBundle, SplitView, load_dataset
from .evaluator import evaluate_predictions
from .runner import run_trainer
from .goat_executor import BridgeRunResult, KuaiRandGoatExecutor, assert_goat_compatible

__all__ = [
    "DatasetBundle", "SplitView", "load_dataset", "evaluate_predictions", "run_trainer",
    "BridgeRunResult", "KuaiRandGoatExecutor", "assert_goat_compatible",
]

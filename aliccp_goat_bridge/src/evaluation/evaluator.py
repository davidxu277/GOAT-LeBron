"""Simple evaluate(predictions, labels) interface for collaborators."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from aliccp_tools.evaluation import evaluate_predictions


def evaluate(
    predictions: str | Path,
    labels: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate predictions and return CTR/CVR AUC metrics.

    ``predictions`` must contain sample_id, ctr, cvr and ctcvr.
    ``labels`` must contain sample_id, click and conversion. Both arguments may
    point to Parquet files/directories, CSV, JSON or JSONL.
    """
    output = Path(output_path) if output_path is not None else None
    return evaluate_predictions(Path(labels), Path(predictions), output)


def format_metrics(metrics: dict[str, Any]) -> str:
    """Format metrics in the concise form used in the team interface."""
    def display(value: object) -> str:
        return "N/A" if value is None else f"{float(value):.4f}"

    return (f"CTR AUC: {display(metrics.get('ctr_auc'))}\n"
            f"CVR AUC (clicked): {display(metrics.get('cvr_auc'))}\n"
            f"CVR AUC (all): {display(metrics.get('cvr_auc_all'))}")


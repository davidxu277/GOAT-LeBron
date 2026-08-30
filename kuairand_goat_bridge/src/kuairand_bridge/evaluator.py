"""Official validation scorer and GOAT-friendly result envelope."""

from __future__ import annotations

import json
import pathlib

import numpy as np

from .dataset import DatasetBundle
from .official import module
from .predictions import normalize_predictions


def _plain(value):
    """Convert NumPy scalars returned by model/official code into JSON values."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def evaluate_predictions(dataset: DatasetBundle, predictions, split="valid", output_dir="output"):
    target = dataset.split(split)
    output_dir = pathlib.Path(output_dir)
    official_csv = normalize_predictions(predictions, output_dir / f"{split}_submission.csv", target.rows)
    scores = module("submit").read_submission(str(official_csv), target.rows)
    if split == "test":
        result = {"status": "checked", "split": "test", "rows": len(scores),
                  "submission": str(official_csv),
                  "message": "Test 只做格式检查，不向 Agent 返回标签或分数"}
    else:
        metrics = _plain(module("evaluate").evaluate(target.user_ids, target.labels, scores))
        result = {"status": "scored", "split": split, "metrics": metrics,
                  "submission": str(official_csv)}
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{split}_metrics.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["metrics_file"] = str(result_path)
    return result

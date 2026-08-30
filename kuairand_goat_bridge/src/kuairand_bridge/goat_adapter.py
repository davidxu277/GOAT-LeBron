"""Thin result adapter for GOAT-LeBron's orchestration/logging layer."""

from __future__ import annotations

from .evaluator import evaluate_predictions


class KuaiRandOfficialEvaluator:
    """Accept a teammate prediction artifact and return a GOAT-friendly report."""

    def __init__(self, dataset, output_dir="output"):
        self.dataset = dataset
        self.output_dir = output_dir

    def score(self, prediction_path):
        result = evaluate_predictions(self.dataset, prediction_path, "valid", self.output_dir)
        m = result["metrics"]
        return {
            "数据集": "KuaiRand-Pure",
            "任务": "用户内 long_view 排序",
            "验证集": {
                "GAUC": m["GAUC"], "nDCG@5": m["nDCG@5"],
                "主分": m["primary"], "用户数": m["users"], "总行数": m["rows"],
            },
            "官方结果文件": result["metrics_file"],
        }

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import KFold


class TargetEncoding:
    """目标编码 —— 把高基数 ID 替换成它历史上的平滑点击率。

    对每个指定字段，分别计算该字段每个取值在训练集上的平滑点击率：
        encode_value = (sum_y + alpha * global_mean) / (count + alpha)
    其中 sum_y 是该取值下的正样本数（点击），count 是该取值出现次数，
    global_mean 是训练集全局点击率，alpha 是平滑强度（从配置读）。

    ⚠️ 折外纪律（CLAUDE.md R2）：训练集内部用 K 折，算第 k 折的编码值时
    只用其余 K−1 折的统计量，杜绝标签泄漏。验证集 / 测试集用整个训练集
    的统计量。没见过的 ID 一律回退到全局先验。

    ⚠️ 本节零件不把 click 列加入特征输出。目标列名由配置 target_col 指定，
    默认使用 ctr_label，该列在训练开始前由主流程生成，不会进入模型输入。
    """

    def __init__(self, config: dict[str, Any]):
        cfg = config["features"]["目标编码"]
        self.fields: list[str] = cfg["fields"]
        self.smoothing: int = cfg["smoothing"]
        self.n_folds: int = cfg["n_folds"]
        self.target_col: str = cfg.get("target_col", "ctr_label")

        # fit 阶段学习到的统计量，只在训练集上计算
        self.global_mean: float = 0.0
        # 每个字段 -> 每个取值 -> (sum_y, count)，用整个训练集算，供验证集用
        self.full_stats: dict[str, dict[str, tuple[float, int]]] = {}
        # 每个字段 -> 每个取值 -> 编码值（由 full_stats 算出的最终编码）
        self.full_encoding: dict[str, dict[str, float]] = {}
        # 训练集折外编码时暂存每折统计量，key: (field, fold_idx)
        self._fold_stats: dict[tuple[str, int], dict[str, tuple[float, int]]] = {}

    # ── FeatureOp 接口 ──────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> None:
        """只在训练集上统计。绝不能读验证集（R2）。"""
        if self.target_col not in train_df.columns:
            raise ValueError(f"训练集缺少目标列 {self.target_col}")

        # 全局点击率（使用训练集的目标列）
        self.global_mean = train_df[self.target_col].mean()

        # 对整个训练集统计（供验证集用）
        for field in self.fields:
            stats = self._compute_stats(train_df, field, train_df[self.target_col])
            self.full_stats[field] = stats
            self.full_encoding[field] = {
                key: self._smooth_encode(count, sum_y) for key, (sum_y, count) in stats.items()
            }

        # 训练集内部折外：生成每一折的统计量，供 transform_train_with_fold 使用
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        for fold_idx, (train_idx, _) in enumerate(kf.split(train_df)):
            fold_train = train_df.iloc[train_idx]
            for field in self.fields:
                stats = self._compute_stats(fold_train, field, fold_train[self.target_col])
                self._fold_stats[(field, fold_idx)] = stats

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """套用训练集学到的统计量。行为一致于训练集 / 开发集 / 锁定集。

        只新增 target_enc_ 前缀的列，绝不包含禁用字段（R1）。
        """
        if not self.full_encoding:
            raise RuntimeError("必须先 fit 再 transform")

        for i, field in enumerate(self.fields):
            # 基础编码值（用全训练集统计量）
            encoded = df[field].map(self.full_encoding[field]).fillna(self.global_mean)
            col_name = f"target_enc_{i}_{field}"
            df[col_name] = encoded

        return df

    def transform_train_with_fold(self, df: pd.DataFrame, fold_idx: int) -> pd.DataFrame:
        """对训练集中的某一折做折外编码。

        调用时机：主流程在训练集上 fit 之后，会把训练集按同样折号划分，
        对每个 fold_idx 调用本方法，把该折样本用对应折的统计量编码，
        而不是用包含自己的统计量。
        """
        if not self._fold_stats:
            raise RuntimeError("必须先 fit 再 transform_train_with_fold")

        for i, field in enumerate(self.fields):
            stats = self._fold_stats.get((field, fold_idx), {})
            fold_encoding = {
                key: self._smooth_encode(count, sum_y) for key, (sum_y, count) in stats.items()
            }
            encoded = df[field].map(fold_encoding).fillna(self.global_mean)
            col_name = f"target_enc_{i}_{field}"
            df[col_name] = encoded

        return df

    # ── 内部 ────────────────────────────────────────────────────

    def _compute_stats(
        self, df: pd.DataFrame, field: str, y: pd.Series
    ) -> dict[str, tuple[float, int]]:
        """统计 df 中每个取值的正样本数和出现次数。"""
        temp = pd.DataFrame({"field": df[field], "y": y})
        grouped = temp.groupby("field")["y"].agg(["sum", "count"])
        return {
            str(key): (float(row["sum"]), int(row["count"]))
            for key, row in grouped.iterrows()
        }

    def _smooth_encode(self, count: int, sum_y: float) -> float:
        """平滑编码： (sum_y + smoothing * global_mean) / (count + smoothing)"""
        return (sum_y + self.smoothing * self.global_mean) / (count + self.smoothing)

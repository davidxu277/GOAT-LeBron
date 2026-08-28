"""多值字段接回来 —— 把未使用的历史行为字段和用户×商品交叉字段接回特征表。

历史行为字段（109_14/110_14/127_14/150_14）是"一串 ID + 对应权重"的多值字段，
不能只取最后一个值（CLAUDE.md 第六节）。本零件在 fit 阶段只基于训练集统计每个
行为 ID 的出现频次（R2），然后在 transform 阶段把每个多值字段聚合为固定维度的
统计向量：条目数、权重和、加权平均池化频率、频率加权标准差、最大频率。

交叉字段（508/509/702/853）是单值带权重字段，直接作为类别特征接入，
对应的 D508/D509/D702/D853 权重列作为稠密特征接入。

所有可调参数（字段名、池化方式、权重列前缀、缺失值填充）都从 config 读取（R7）。
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd


class MultiValuePooling:
    """把多值行为字段池化成统计特征，并把交叉字段接回特征表。

    配置项（features.多值字段接回来）：
        fields               多值行为字段名列表，如 ["109_14", "110_14", ...]
        cross_fields         单值交叉字段名列表，如 ["508", "509", ...]
        pooling              "weighted_mean" 或 "mean"
        cross_weight_prefix  交叉字段权重列前缀，默认 "D"
        missing_id           缺失 ID 的填充值，默认 -1
        missing_weight       缺失权重的填充值，默认 0.0
    """

    def __init__(self, config: dict[str, Any]):
        cfg = config["features"]["多值字段接回来"]
        self.fields: list[str] = list(cfg.get("fields", []))
        self.cross_fields: list[str] = list(cfg.get("cross_fields", []))
        self.pooling: str = cfg.get("pooling", "weighted_mean")
        self.cross_weight_prefix: str = cfg.get("cross_weight_prefix", "D")
        self.missing_id: int = int(cfg.get("missing_id", -1))
        self.missing_weight: float = float(cfg.get("missing_weight", 0.0))

        self.freq: dict[str, pd.Series] = {}
        self.freq_dict: dict[str, dict[str, int]] = {}

    # ── FeatureOp 接口 ─────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> None:
        """只在训练集上统计每个多值行为字段里 ID 的出现次数（R2）。"""
        for field in self.fields:
            counts: Counter = Counter()
            if field in train_df.columns:
                for items in train_df[field].apply(self._parse_items):
                    for id_, _w in items:
                        counts[id_] += 1
            self.freq[field] = pd.Series(counts, dtype=int)
            self.freq_dict[field] = dict(counts)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """把 fit 学到的统计量套用到 df 上，返回加工后的 DataFrame。"""
        idx = df.index
        n = len(df)

        for field in self.fields:
            if field in df.columns:
                parsed = df[field].apply(self._parse_items)
            else:
                parsed = pd.Series([[] for _ in range(n)], index=idx)

            counts = parsed.apply(len)
            sum_weights = parsed.apply(lambda items: sum(w for _, w in items))

            if self.pooling == "weighted_mean":
                pooled = parsed.apply(
                    lambda items: self._weighted_mean_freq(items, field)
                )
                pooled_std = parsed.apply(
                    lambda items: self._weighted_std_freq(items, field)
                )
            else:  # "mean"
                pooled = parsed.apply(
                    lambda items: self._mean_freq(items, field)
                )
                pooled_std = parsed.apply(
                    lambda items: self._std_freq(items, field)
                )

            max_freq = parsed.apply(
                lambda items: self._max_freq(items, field)
            )

            df[f"{field}_count"] = counts.astype(int)
            df[f"{field}_sum_weight"] = sum_weights.astype(float)
            df[f"{field}_{self.pooling}_freq"] = pooled.astype(float)
            df[f"{field}_freq_std"] = pooled_std.astype(float)
            df[f"{field}_freq_max"] = max_freq.astype(float)

        for field in self.cross_fields:
            if field not in df.columns:
                df[field] = self.missing_id
            df[field] = df[field].fillna(self.missing_id).astype("category")

            weight_col = self.cross_weight_prefix + field
            if weight_col not in df.columns:
                df[weight_col] = self.missing_weight
            df[weight_col] = pd.to_numeric(
                df[weight_col], errors="coerce"
            ).fillna(self.missing_weight).astype(float)

        return df

    # ── 内部工具 ───────────────────────────────────────────────────

    @staticmethod
    def _parse_items(val: Any) -> list[tuple[str, float]]:
        """把多值字段的一个单元格解析成 [(id, weight), ...] 列表。

        支持多种常见格式：
          - 字符串 "id1:0.5,id2:1.2"
          - 字符串 "id1,id2"（无权重，默认 1.0）
          - 列表 ["id1:0.5", "id2:1.2"]
          - 列表 [(id1, w1), (id2, w2)]
          - 单个标量 ID
          - NaN / None（返回空列表）
        """
        if val is None:
            return []
        if isinstance(val, float) and np.isnan(val):
            return []

        def _one(item: Any) -> tuple[str, float]:
            if isinstance(item, tuple) and len(item) == 2:
                return str(item[0]), float(item[1])
            if isinstance(item, str) and ":" in item:
                id_part, w_part = item.split(":", 1)
                return id_part.strip(), float(w_part)
            return str(item), 1.0

        if isinstance(val, list):
            return [_one(item) for item in val]
        if isinstance(val, tuple):
            return [_one(val)]
        if isinstance(val, str):
            parts = [p.strip() for p in val.split(",") if p.strip()]
            return [_one(p) for p in parts]
        return [(str(val), 1.0)]

    def _freq_of(self, field: str, id_: str) -> float:
        """返回某个 ID 在训练集里的出现次数；未出现过返回 0。"""
        return float(self.freq_dict.get(field, {}).get(id_, 0))

    def _weighted_mean_freq(self, items: list[tuple[str, float]], field: str) -> float:
        total_w = sum(w for _, w in items)
        if total_w == 0:
            return 0.0
        total = sum(w * self._freq_of(field, id_) for id_, w in items)
        return total / total_w

    def _mean_freq(self, items: list[tuple[str, float]], field: str) -> float:
        if not items:
            return 0.0
        return sum(self._freq_of(field, id_) for id_, _ in items) / len(items)

    def _weighted_std_freq(
        self, items: list[tuple[str, float]], field: str
    ) -> float:
        if not items:
            return 0.0
        mean = self._weighted_mean_freq(items, field)
        total_w = sum(w for _, w in items)
        if total_w == 0:
            return 0.0
        var = sum(
            w * (self._freq_of(field, id_) - mean) ** 2 for id_, w in items
        ) / total_w
        return float(np.sqrt(var))

    def _std_freq(
        self, items: list[tuple[str, float]], field: str
    ) -> float:
        if not items:
            return 0.0
        mean = self._mean_freq(items, field)
        if len(items) <= 1:
            return 0.0
        var = sum(
            (self._freq_of(field, id_) - mean) ** 2 for id_, _ in items
        ) / len(items)
        return float(np.sqrt(var))

    def _max_freq(
        self, items: list[tuple[str, float]], field: str
    ) -> float:
        if not items:
            return 0.0
        return max(self._freq_of(field, id_) for id_, _ in items)

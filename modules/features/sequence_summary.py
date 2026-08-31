"""把 AliCCP 多值字段压成稳定、定长的类别摘要。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class SequenceSummary:
    """为每个 ID/权重序列生成长度、首项和最高权重项。"""

    def __init__(self, config: dict[str, Any]):
        cfg = config["features"]["多值字段接回来"]
        self.fields: list[str] = [str(value) for value in cfg["fields"]]
        self.weight_prefix: str = str(cfg["weight_prefix"])
        self.summaries: list[str] = [str(value) for value in cfg["summaries"]]
        self.missing_token: str = str(cfg["missing_token"])
        self.drop_source_columns: bool = bool(cfg["drop_source_columns"])

        supported = {"length", "first_id", "max_weight_id"}
        unknown = set(self.summaries) - supported
        if unknown:
            raise ValueError(f"不支持的序列摘要：{sorted(unknown)}")

    def needs(self) -> list[str]:
        """要读哪些列。本来就是为了省内存写的零件，不声明反而会整表加载。"""
        return list(self.fields) if hasattr(self, "fields") else []

    def fit(self, train_df: pd.DataFrame) -> None:
        """只校验训练集字段；本零件不从验证集学习任何统计量。"""
        missing = self._missing_columns(train_df)
        if missing:
            raise ValueError(f"训练集缺少多值字段或权重列：{missing}")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = self._missing_columns(df)
        if missing:
            raise ValueError(f"数据缺少多值字段或权重列：{missing}")

        for field in self.fields:
            weight_field = f"{self.weight_prefix}{field}"
            ids_and_weights = zip(df[field], df[weight_field])

            if "length" in self.summaries:
                df[f"{field}_seq_length"] = np.fromiter(
                    (len(self._sequence(ids)) for ids, _ in ids_and_weights),
                    dtype=np.int32,
                    count=len(df),
                )
                ids_and_weights = zip(df[field], df[weight_field])

            if "first_id" in self.summaries:
                df[f"{field}_seq_first"] = [
                    self._first_id(ids) for ids, _ in ids_and_weights
                ]
                ids_and_weights = zip(df[field], df[weight_field])

            if "max_weight_id" in self.summaries:
                df[f"{field}_seq_top"] = [
                    self._max_weight_id(ids, weights)
                    for ids, weights in ids_and_weights
                ]

            if self.drop_source_columns:
                df.drop(columns=[field, weight_field], inplace=True)

        return df

    def _missing_columns(self, df: pd.DataFrame) -> list[str]:
        required = {
            name
            for field in self.fields
            for name in (field, f"{self.weight_prefix}{field}")
        }
        return sorted(required - set(map(str, df.columns)))

    @staticmethod
    def _sequence(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return list(value)
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        try:
            if pd.isna(value):
                return []
        except (TypeError, ValueError):
            return []
        return [value]

    def _first_id(self, ids_raw: Any) -> str:
        ids = self._sequence(ids_raw)
        return str(ids[0]) if ids else self.missing_token

    def _max_weight_id(self, ids_raw: Any, weights_raw: Any) -> str:
        ids = self._sequence(ids_raw)
        weights = self._sequence(weights_raw)
        if not ids or len(ids) != len(weights):
            return self.missing_token
        try:
            position = max(range(len(weights)), key=lambda index: float(weights[index]))
        except (TypeError, ValueError):
            return self.missing_token
        return str(ids[position])

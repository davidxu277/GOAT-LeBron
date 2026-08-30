"""商品出现次数特征 —— 特征类零件。

在特征列表中添加商品（field 206）的出现次数，基于训练集统计。
该特征可帮助模型感知商品流行度，减少冷门商品的过拟合。

配置项（全部从 config 读，不许写死 —— R7）：
    features.item_frequency.enabled   是否启用
    features.item_frequency.source_field   要统计的字段，默认 '206'
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class ItemFrequency:
    """统计指定字段的取值出现次数，作为新特征。

    生命周期：
        fit(train_df)        在训练集上统计次数
        transform(df)        对每个数据集加上出现次数列

    新特征列名：<source_field>_freq
    该特征会自动进入特征表，无需手动添加 base_fields。
    """

    def __init__(self, config: dict[str, Any]):
        cfg = config["features"]["item_frequency"]
        self.source_field: str = cfg.get("source_field", "206")
        self.freq_map: dict[Any, int] = {}
        self.freq_col: str = f"{self.source_field}_freq"

    def fit(self, train_df: Any) -> None:
        """在训练集上统计出现次数，保存到 self.freq_map。"""
        if self.source_field not in train_df.columns:
            raise KeyError(f"字段 {self.source_field} 不在训练集中，无法统计频率。")
        # 只统计源字段，不涉及任何禁用字段
        self.freq_map = train_df[self.source_field].value_counts().to_dict()

    def transform(self, df: Any) -> Any:
        """把出现次数映射到 df 的新列上。未出现过的取值记 0。"""
        if not self.freq_map:
            raise RuntimeError("必须先调用 fit() 再调用 transform()。")
        if self.source_field not in df.columns:
            raise KeyError(f"字段 {self.source_field} 不在输入 DataFrame 中。")
        df = df.copy()
        df[self.freq_col] = df[self.source_field].map(self.freq_map).fillna(0).astype(int)
        return df

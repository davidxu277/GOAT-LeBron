"""出现次数分桶 —— 加特征类零件的范文。

给工兵看的：这是"加特征"这一类零件该长什么样。
另外两类的范文在 modules/train/（改训练过程）和 modules/models/（改模型）。

做的事：把高基数 ID（商品、用户）的出现次数分成几档，当作一个新的类别特征。
低频 ID 自己那份 embedding 学不出信息，但"它是个冷门商品"这件事本身就有信息量。

⚠️ 出现次数只能在训练集上统计（CLAUDE.md R2）—— fit 只看 train_df，
transform 对验证集套用同一份统计量。读验证集来算统计量是作弊。
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class FrequencyBucket:
    """把某个字段的出现次数分桶，产出一个新的类别特征列。

    配置项（全部从 config 读，不许写死 —— R7）：
        features.频次分桶.field    对哪个字段统计出现次数，如 "205"（商品ID）
        features.频次分桶.edges    分桶边界，如 [10, 100, 1000]
    """

    def __init__(self, config: dict[str, Any]):
        cfg = config["features"]["频次分桶"]
        self.field: str = cfg["field"]
        self.edges: list[int] = cfg["edges"]
        self.counts: pd.Series | None = None

    # ── FeatureOp 接口（见 modules/base.py）──────────────────────

    def fit(self, train_df: pd.DataFrame) -> None:
        """只在训练集上统计出现次数。绝不许读验证集（R2）。"""
        self.counts = train_df[self.field].value_counts()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """套用训练集的统计量。训练集、开发集、锁定集走的都是这一个方法。"""
        if self.counts is None:
            raise RuntimeError("必须先 fit 再 transform")
        # 验证集里没见过的 ID → 出现次数记为 0，落进最低那一档
        freq = df[self.field].map(self.counts).fillna(0).to_numpy()
        df[f"{self.field}_freq_bucket"] = pd.Series(
            freq, index=df.index
        ).apply(self._bucket).astype("category")
        return df

    # ── 内部 ────────────────────────────────────────────────────

    def _bucket(self, n: float) -> int:
        """出现次数 → 档位序号。边界从配置来，不写死。"""
        for i, edge in enumerate(self.edges):
            if n < edge:
                return i
        return len(self.edges)

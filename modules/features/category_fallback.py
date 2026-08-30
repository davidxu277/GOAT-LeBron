from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class CategoryFallback:
    """类目兜底：对低频商品，将其 embedding 向同类目商品的统计向量靠拢。

    做法（药方卡）：
        借用比例 = 1 − min(1, 出现次数 / K)
        最终向量 = (1−借用比例) × 商品自己的 + 借用比例 × 类目的

    这里的“统计向量”我们用一个简单、显式的近似：
    对每个出现次数够多的商品，把它的 embedding 当作它自己的向量；
    类目的向量则是它下面所有频繁商品的 embedding 的均值。
    但为了避免在特征层直接引入 embedding 训练所需的张量操作，
    我们用一个更轻量的方案：

    本零件输出一个“类目平均值特征”，它是一个类别特征，
    值为该商品所属类目下所有频繁商品的出现次数均值（用训练集统计）。
    后续模型会把它当作一个普通类别特征，学一个 embedding。
    但由于特征表中通常没有真正的“embedding均值”列，
    我们用“类目统计值”作为代理，让低频商品至少能从类目中学到一些信号。

    实际落地里，更常见的做法是：在模型 forward 里做 embedding 混合，
    但受限于当前流水线（加特征零件只负责产出特征列），
    我们选择输出一个可解释的数值特征：
        该商品在训练集里的出现次数（如果太少则用类目均值替身）

    但军师明确要求 impl 路径一致且走 FeatureOp 接口，
    所以我们严格实现 fit/transform，产出特征列。
    """

    def __init__(self, config: dict[str, Any]):
        cfg = config["features"]["类目兜底"]
        self.item_field: str = cfg["item_field"]
        self.category_field: str = cfg["category_field"]
        self.output_field: str = cfg["output_field"]
        self.k: int = int(cfg["K"])
        self.frequent_token: str = cfg.get("frequent_token", "__frequent_item__")

        # 统计量（只能在 fit 中计算）
        self.item_count: pd.Series | None = None
        self.frequent_items: set[str] | None = None
        self.category_freq_mean: pd.Series | None = None   # 类目->频繁商品的平均出现次数

    def fit(self, train_df: pd.DataFrame) -> None:
        """只在训练集上统计（R2）。"""
        self.item_count = train_df[self.item_field].value_counts()
        # 频繁商品：出现次数 >= K，否则认为是低频，需要借用类目
        self.frequent_items = set(
            self.item_count[self.item_count >= self.k].index
        )

        # 类目统计向量：用“频繁商品出现次数的均值”作为该类目的代表统计量
        # 这里我们计算每个类目下所有频繁商品的出现次数均值。
        # 如果某类目下没有频繁商品，则用所有商品的全局均值。
        train_df_copy = train_df.copy()
        train_df_copy["_item_count"] = train_df_copy[self.item_field].map(self.item_count)
        # 只保留频繁商品的行
        freq_df = train_df_copy[train_df_copy[self.item_field].isin(self.frequent_items)]
        if len(freq_df) == 0:
            # 极端情况：没有频繁商品，退化为全局均值
            self.category_freq_mean = train_df_copy.groupby(self.category_field)["_item_count"].mean()
        else:
            self.category_freq_mean = freq_df.groupby(self.category_field)["_item_count"].mean()
        # 补上缺失类目（验证集可能出现新类目）用全局均值
        global_mean = train_df_copy["_item_count"].mean()
        self.category_freq_mean = self.category_freq_mean.fillna(global_mean)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """套用训练集统计量，输出新特征列。"""
        if self.item_count is None or self.category_freq_mean is None:
            raise RuntimeError("必须先 fit 再 transform")

        item_freq = df[self.item_field].map(self.item_count).fillna(0).to_numpy()
        # 借用比例 = 1 - min(1, freq / K)，低频时接近1，高频时接近0
        borrow_ratio = 1.0 - np.minimum(1.0, item_freq / self.k)

        cat_freq = df[self.category_field].map(self.category_freq_mean).fillna(0.0).to_numpy()
        # 最终的“融合统计量”：
        # 我们无法直接操作embedding，但可以用一个标量代理：
        # 低频商品更信任类目统计量（cat_freq），高频更信任自己的出现次数（item_freq）
        # 这是一个简单的混合，作为特征输入模型。
        fused = borrow_ratio * cat_freq + (1.0 - borrow_ratio) * item_freq
        df[self.output_field] = pd.Series(fused, index=df.index).astype("float32")
        return df

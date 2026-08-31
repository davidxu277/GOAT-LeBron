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
        # 目标列不给默认值 —— 原来默认 "ctr_label"，那一列在任何数据里都不存在，
        # 一启用就 ValueError。不同任务的标签列不一样（AliCCP 是 click，
        # KuaiRand 是 label），所以要么配置里写死，要么 fit 时从数据里认。
        self.target_col: str | None = cfg.get("target_col")

        # 折分种子从配置读（R7）。原来写死 42 —— 跑三个种子做稳定性实验时，
        # 目标编码的折分始终不变，等于把这部分随机性人为抹掉了。
        self.seed: int = int(
            (config.get("train") or {}).get("seed", cfg.get("seed", 0)))

        # fit 阶段学习到的统计量，只在训练集上计算
        self.global_mean: float = 0.0
        # 每个字段 -> 每个取值 -> (sum_y, count)，用整个训练集算，供验证集用
        self.full_stats: dict[str, dict[str, tuple[float, int]]] = {}
        # 每个字段 -> 每个取值 -> 编码值（由 full_stats 算出的最终编码）
        self.full_encoding: dict[str, dict[str, float]] = {}
        # 训练集折外编码时暂存每折统计量，key: (field, fold_idx)
        self._fold_stats: dict[tuple[str, int], dict[str, tuple[float, int]]] = {}
        # 每折要被编码的行下标。没有它，调用方拿不到折的划分
        self._fold_rows: dict[int, Any] = {}

    # ── FeatureOp 接口 ──────────────────────────────────────────

    def needs(self) -> list[str]:
        """要读哪些列。不声明的话执行器只能读整张表，内存会炸。"""
        cols = list(self.fields)
        if self.target_col:
            cols.append(self.target_col)
        return cols


    #: 没在配置里指定 target_col 时，按顺序在数据里找这几个
    LABEL_CANDIDATES = ("label", "click")

    def fit(self, train_df: pd.DataFrame) -> None:
        """只在训练集上统计。绝不能读验证集（R2）。"""
        if self.target_col is None:
            self.target_col = next(
                (c for c in self.LABEL_CANDIDATES if c in train_df.columns), None)
            if self.target_col is None:
                raise ValueError(
                    f"配置里没写 features.目标编码.target_col，数据里也找不到 "
                    f"{list(self.LABEL_CANDIDATES)} 中的任何一列")
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
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        self._fold_rows = {}
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df)):
            # 记住这一折**要被编码**的行 —— 不记的话调用方无从知道折的划分，
            # 也就没法调 transform_train_with_fold，只能退回会泄漏的 transform
            self._fold_rows[fold_idx] = val_idx
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

    def transform_train(self, df: pd.DataFrame) -> pd.DataFrame:
        """对**训练集**做折外编码 —— 训练集必须走这条，不能走 transform。

        为什么：transform 用的是全训练集统计量，而训练集自己的标签也在里面。
        每一行都在"用包含自己答案的统计量"给自己编码 —— 这是目标泄漏，
        分数会虚高，而且日志上完全看不出来，到测试集才现原形。

        折外的意思：把训练集分成 N 折，第 k 折的行只用**其余 N-1 折**
        算出来的统计量编码。这样任何一行的编码里都不含自己的标签。

        调用方只要 `hasattr(op, "transform_train")` 就该走这条 ——
        不需要知道折是怎么分的（那是本零件的内部实现）。
        """
        if not self._fold_rows:
            raise RuntimeError("必须先 fit 再 transform_train")
        out = df.copy()
        for fold_idx, rows in self._fold_rows.items():
            块 = self.transform_train_with_fold(df.iloc[rows].copy(), fold_idx)
            for col in 块.columns:
                if col.startswith("target_enc_"):
                    if col not in out.columns:
                        out[col] = self.global_mean
                    out.iloc[rows, out.columns.get_loc(col)] = 块[col].to_numpy()
        return out

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

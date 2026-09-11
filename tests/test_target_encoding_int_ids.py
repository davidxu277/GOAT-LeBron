"""目标编码在 KuaiRand 上输出的是一个常数 —— 表里存的键是字符串，查的却是整数。

_compute_stats 把每个取值存成 str(key)，而 transform / transform_train_with_fold
直接拿 df[field] 去查。AliCCP 的 ID 列读进来是字符串，查得到；KuaiRand 的
video_id / author_id 是整数，**一个都查不到**，全部 fillna 成全局均值 ——
整列只有一个值，模型从这个特征里什么也学不到，而且不报任何错。

实测（2026-09-11，500 个视频 × 20000 行的造数据，走 apply_feature_ops）：
训练集、验证集的编码列都只有 1 个取值。换任务之后「目标编码」这张卡一直是空转。
"""

import numpy as np
import pandas as pd
import pytest

from modules.features.target_encoding import TargetEncoding

COL = "target_enc_0_video_id"
SMOOTHING = 20


def _data(dtype: str) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    vid = rng.integers(0, 50, 5000)
    label = (rng.random(5000) < vid / 50).astype(int)     # 视频号越大越容易看完
    return pd.DataFrame({"video_id": vid.astype(dtype), "label": label})


def _op() -> TargetEncoding:
    return TargetEncoding({
        "features": {"目标编码": {"fields": ["video_id"], "smoothing": SMOOTHING,
                                  "n_folds": 5, "target_col": "label"}},
        "train": {"seed": 0},
    })


@pytest.mark.parametrize("dtype", ["int64", "str"])
def test_验证集的编码值就是这个ID的平滑看完率(dtype):
    train = _data(dtype)
    op = _op()
    op.fit(train)
    out = op.transform(train.copy())

    some_id = train["video_id"].iloc[0]
    rows = train["video_id"] == some_id
    gm = train["label"].mean()
    expected = (train.loc[rows, "label"].sum() + SMOOTHING * gm) / (rows.sum() + SMOOTHING)
    assert out.loc[rows, COL].iloc[0] == pytest.approx(expected)
    assert out[COL].nunique() > 1


@pytest.mark.parametrize("dtype", ["int64", "str"])
def test_训练集的折外编码也随ID变化(dtype):
    train = _data(dtype)
    op = _op()
    op.fit(train)
    out = op.transform_train(train)
    assert out[COL].nunique() > 1
    # 看完率高的那半边视频，编码平均值也应该更高 —— 编码真的带上了标签信息
    ids = train["video_id"].astype(int)
    assert out.loc[ids >= 25, COL].mean() > out.loc[ids < 25, COL].mean()

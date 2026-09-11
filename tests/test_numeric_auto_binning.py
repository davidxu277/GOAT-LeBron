"""连续值特征按训练集分位数自动分档。

Vocab 把每一列的每个不同取值登记成一个类别 ID —— 对 ID 列是对的，对 Agent 写的
统计类特征是错的：目标编码、看完率这种小数，训练集（折外）和验证集（全量统计）
算出来的数几乎对不上。实测：造数据走目标编码，验证集 37% 的行落进 OOV，
剩下的也只是数值恰好撞上。

以前靠卡片提醒工兵"要先分档"—— 靠自觉。现在 Vocab 从 dtype 自己认：
浮点列、且取值多于档数，就按训练集分位数切档（R2），档数从配置读（R7）。
"""

import numpy as np
import pandas as pd
import pytest

from harness.deep import OOV, Vocab
from harness.ops import apply_feature_ops, load_feature_ops


def _te_frames():
    rng = np.random.default_rng(0)

    def make(n):
        vid = rng.integers(0, 500, n)
        return pd.DataFrame({"video_id": vid,
                             "label": (rng.random(n) < vid / 500).astype(int)})

    cfg = {"features": {"目标编码": {
        "enabled": True, "impl": "modules/features/target_encoding.py",
        "fields": ["video_id"], "smoothing": 20, "n_folds": 5, "target_col": "label"}}}
    tr, (va,), new = apply_feature_ops(load_feature_ops(cfg), make(20000), [make(3000)])
    return tr, va, new[0]


def test_目标编码的连续值分档后验证集几乎都认得():
    tr, va, col = _te_frames()
    before = (Vocab().fit(tr, [col]).encode(va)[:, 0] == OOV).mean()
    v = Vocab(numeric_bins=32).fit(tr, [col])
    after = (v.encode(va)[:, 0] == OOV).mean()
    assert before > 0.2          # 不分档：实测 37%
    assert after < 0.01
    assert v.binned_fields == [col]


def test_整数列照旧按取值编号():
    df = pd.DataFrame({"n": np.arange(1000)})
    v = Vocab(numeric_bins=32).fit(df, ["n"])
    assert v.binned_fields == []
    assert v.sizes()["n"] == 1001


def test_取值少的小数列不分档():
    df = pd.DataFrame({"x": [0.1, 0.2, 0.3] * 10})
    v = Vocab(numeric_bins=32).fit(df, ["x"])
    assert v.binned_fields == []
    assert v.sizes()["x"] == 4


def test_超出训练范围落到两端的档_NaN落OOV():
    tr = pd.DataFrame({"x": np.linspace(0, 1, 1000)})
    v = Vocab(numeric_bins=8).fit(tr, ["x"])
    codes = v.encode(pd.DataFrame({"x": [-5.0, 0.5, 99.0, np.nan]}))[:, 0]
    assert codes[0] == 1                        # 最低档
    assert codes[2] == v.sizes()["x"] - 1       # 最高档
    assert codes[3] == OOV
    assert OOV not in codes[:3]


def test_不传档数时行为不变():
    tr, _, col = _te_frames()
    v = Vocab().fit(tr, [col])
    assert v.binned_fields == []
    assert v.sizes()[col] == tr[col].nunique() + 1

"""自动分档接进训练循环：档数从配置读（R7），训练记录里看得到哪些列被分档。

看得见这一条是为医生和复盘官准备的：runner 会把整份训练记录放进成绩单
「训练诊断 → 每轮训练记录」。一个统计特征加了没效果时，第一件要确认的是
它有没有被当成连续值分档 —— 不写出来，这一步就只能靠猜。
"""

import numpy as np
import pandas as pd
import pytest

from harness.deep import deep_kwargs, train_deep


def test_档数走白名单():
    assert deep_kwargs({})["numeric_bins"] == 32
    assert deep_kwargs({"numeric_bins": 100000})["numeric_bins"] == 256
    assert deep_kwargs({"numeric_bins": 1})["numeric_bins"] == 2


def test_训练记录写明哪些列被自动分档():
    pytest.importorskip("torch")
    rng = np.random.default_rng(0)

    def make(n):
        return pd.DataFrame({"user_id": rng.integers(0, 20, n),
                             "rate": rng.random(n),
                             "label": rng.integers(0, 2, n)})

    cfg = {"model": {"impl": "modules/models/mlp.py",
                     "mlp": {"hidden": [8], "tower": [4], "dropout": 0.0},
                     "deep": {"epochs": 1, "batch_size": 64, "numeric_bins": 8}},
           "train": {}}

    def loss(out, batch, torch):
        y = torch.tensor(batch["label"].to_numpy(), dtype=torch.float32,
                         device=out["ctr"].device)
        return torch.nn.functional.binary_cross_entropy(
            out["ctr"].clamp(1e-7, 1 - 1e-7), y)

    _, _, 记录 = train_deep(cfg, make(300), make(100), ["user_id", "rate"], seed=0,
                          task_loss=loss, task_metric=lambda *a: {"primary": 0.5})
    assert 记录["自动分档的列"] == ["rate"]
    assert 记录["_vocab"].sizes()["rate"] <= 8 + 1

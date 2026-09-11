"""模型零件写的损失函数，训练循环真的要用上。

train_deep 原来的优先级是「任务损失 > 零件的 loss > 双塔默认」，而 KuaiRand 总是
传任务损失（逐条 BCE）—— 零件里写的 loss **被静默忽略**，不报错也不提示。
配对排序（pairwise_loss，预计提升最大的一张卡）和时间衰减两张卡因此一直落不了地。
旧签名 loss(out, click, conv) 也只给两个标签张量，配对要的 user_id、
衰减要的 date 根本拿不到。

现在：零件写了 loss(out, batch, torch) 就用它，batch 是这一批的整张表；
训练记录写出「损失来源」，成绩单上一眼能看出这一轮到底用的是谁的打分规则。
"""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from harness import deep  # noqa: E402
from modules.models.mlp import MLPModel  # noqa: E402

CFG = {"model": {"impl": "modules/models/mlp.py",
                 "mlp": {"hidden": [8], "tower": [4], "dropout": 0.0},
                 "deep": {"epochs": 1, "batch_size": 64}},
       "train": {}}


def _frames():
    rng = np.random.default_rng(0)

    def make(n):
        return pd.DataFrame({"user_id": rng.integers(0, 20, n),
                             "date": rng.integers(20220409, 20220421, n),
                             "label": rng.integers(0, 2, n)})

    return make(300), make(100)


def _task_loss(out, batch, torch):
    y = torch.tensor(batch["label"].to_numpy(), dtype=torch.float32,
                     device=out["ctr"].device)
    return torch.nn.functional.binary_cross_entropy(out["ctr"].clamp(1e-7, 1 - 1e-7), y)


class _记账的零件(MLPModel):
    def __init__(self, config):
        super().__init__(config)
        self.见过的批 = []

    def loss(self, out, batch, torch):
        self.见过的批.append(batch)
        return _task_loss(out, batch, torch) * 2


def _train(monkeypatch, op):
    monkeypatch.setattr(deep, "load_model_op", lambda cfg: op)
    tr, va = _frames()
    return deep.train_deep(CFG, tr, va, ["user_id"], seed=0, task_loss=_task_loss,
                           task_metric=lambda *a: {"primary": 0.5})


def test_零件写了loss_任务也给了损失_用零件的(monkeypatch):
    op = _记账的零件(CFG)
    _, _, 记录 = _train(monkeypatch, op)
    assert op.见过的批, "零件的 loss 一次都没被调用"
    assert 记录["损失来源"] == "模型零件 _记账的零件.loss"


def test_零件拿到的是这一批的整张表(monkeypatch):
    op = _记账的零件(CFG)
    _train(monkeypatch, op)
    批 = op.见过的批[0]
    assert isinstance(批, pd.DataFrame)
    assert {"user_id", "date", "label"} <= set(批.columns)
    assert len(批) == 64


def test_零件没写loss时照旧用任务损失(monkeypatch):
    _, _, 记录 = _train(monkeypatch, MLPModel(CFG))
    assert 记录["损失来源"] == "任务默认"

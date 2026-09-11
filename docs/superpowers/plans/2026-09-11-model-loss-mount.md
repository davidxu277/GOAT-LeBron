# 模型零件损失挂载点 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 模型零件实现 `loss(out, batch, torch)` 时，训练循环真的用它，而不是被任务默认损失静默盖掉。

**Architecture:** 只改 `harness/deep.py` 的 `train_deep`：损失优先级改成「零件 loss > task_loss > 双塔默认」，零件拿到这一批的 DataFrame；训练记录写出 `损失来源`。接口说明写进 `modules/base.py`，两张卡去掉"目前不生效"。

**Tech Stack:** Python 3.14、pandas、torch、pytest

Spec: `docs/superpowers/specs/2026-09-11-model-loss-mount-design.md`

## Global Constraints

- 签名：`loss(self, out, batch, torch) -> 标量张量`，`batch` 是 `train.iloc[idx]`
- `损失来源` 取值：`"模型零件 <类名>.loss"` / `"任务默认"` / `"双塔默认"`
- 提交信息不加 Co-Authored-By
- 测试：`~/Projects/GOAT-LeBron/.venv/bin/python -m pytest -p no:cacheprovider`

---

### Task 1: train_deep 用零件的 loss，并写出损失来源

**Files:**
- Modify: `harness/deep.py`（`_default_loss` 的 docstring、`train_deep` 的损失选择与返回记录）
- Modify: `modules/base.py`（ModelOp 协议加 `loss` 说明）
- Test: `tests/test_model_loss_mount.py`（新建）

**Interfaces:**
- Produces: `train_deep(...)` 返回的记录里 `"损失来源": str`

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from harness import deep
from modules.models.mlp import MLPModel

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
    y = torch.tensor(batch["label"].to_numpy(), dtype=torch.float32, device=out["ctr"].device)
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
```

- [ ] **Step 2: 跑，确认失败**（零件的 loss 没被调用；记录里没有 `损失来源`）
- [ ] **Step 3: 实现**

`train_deep` 里把 `loss_fn = getattr(op, "loss", None)` 之后加：

```python
    if callable(loss_fn):
        损失来源 = f"模型零件 {type(op).__name__}.loss"
    elif task_loss is not None:
        损失来源 = "任务默认"
    else:
        损失来源 = "双塔默认"
```

循环里的损失选择改成：

```python
            if callable(loss_fn):          # 零件明确写了打分规则 —— 听它的
                loss = loss_fn(out, train.iloc[idx.numpy()], torch)
            elif task_loss is not None:    # 任务默认（KuaiRand：逐条 BCE）
                loss = task_loss(out, train.iloc[idx.numpy()], torch)
            else:
                loss = _default_loss(out, batch_click, batch_conv, torch)
```

返回的记录加 `"损失来源": 损失来源`。`_default_loss` 的 docstring 里
"自己实现 `loss` 方法覆盖它"改成新签名 `loss(out, batch, torch)`。
`modules/base.py` 的 ModelOp 加可选方法 `loss(self, out, batch, torch)` 的说明：
batch 是这一批的 DataFrame（带 user_id、date、label 和全部特征列），写了就优先于任务默认损失。

- [ ] **Step 4: 跑新测试 + 全量测试**
- [ ] **Step 5: 提交** `feat(harness): 模型零件写的损失函数真的会被用上`

### Task 2: 两张卡去掉"目前不生效"

**Files:**
- Modify: `knowledge/cards/pairwise_loss.yaml`、`knowledge/cards/time_decay.yaml`

- [ ] **Step 1**：pairwise_loss —— 删掉"流水线要让模型零件自带的损失生效 —— 目前不生效……"那条前提；
  「怎么实现」开头"挂载点就位之后："改成"新写一个模型零件（modules/models/ 下的新文件），
  继承 modules/models/mlp.py 的 MLPModel，只加一个 loss(self, out, batch, torch) 方法；
  batch 是这一批的表，带 user_id 和 label。model.impl 指向这个新文件。"
- [ ] **Step 2**：time_decay —— 删掉"流水线要能给每条样本的损失乘一个权重 —— 目前不能……"那条前提；
  「怎么实现」同样改成继承 MLPModel、在 loss 里用 batch 的 date 算权重，
  逐条 BCE（reduction="none"）乘权重后取平均
- [ ] **Step 3**：跑守卫测试 + `agent.cli check` + 全量测试 + 离线演习
- [ ] **Step 4**：提交 `docs(knowledge): 损失挂载点就位，配对排序和时间衰减两张卡能落地了`，fetch 后 push

# 数值特征自动分档 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agent 写的连续值特征在 `Vocab` 里按训练集分位数自动分档，验证集不再大面积落进 OOV。

**Architecture:** `Vocab` 是所有深度训练路径唯一的编码入口（`train_deep` 建一次，验证/测试预测复用）。在它的 `fit` 里按 dtype 判定连续列、存分位数边界，`encode` 用 `searchsorted`。档数走 `DEEP_PARAMS` 白名单，训练记录里写出被分档的列。

**Tech Stack:** Python 3.14、pandas、numpy、torch、pytest

Spec: `docs/superpowers/specs/2026-09-11-numeric-auto-binning-design.md`

## Global Constraints

- 边界只用训练集算（CLAUDE.md R2）
- 档数从配置读：`model.deep.numeric_bins`，区间 2~256，默认 32（R7）
- `Vocab()` 不传档数时行为与改动前完全一致
- 提交信息不加 Co-Authored-By
- 测试用 `~/Projects/GOAT-LeBron/.venv/bin/python -m pytest -p no:cacheprovider`

---

### Task 1: Vocab 支持连续列分档

**Files:**
- Modify: `harness/deep.py`（`Vocab` 类）
- Test: `tests/test_numeric_auto_binning.py`（新建）

**Interfaces:**
- Produces: `Vocab(numeric_bins: int | None = None)`；属性 `Vocab.binned_fields -> list[str]`；`fit` / `sizes` / `encode` 签名不变

- [ ] **Step 1: 写失败测试**

```python
import numpy as np
import pandas as pd
import pytest

from harness.deep import OOV, Vocab
from harness.executor import apply_feature_ops, load_feature_ops


def _te_frames():
    rng = np.random.default_rng(0)
    def make(n):
        vid = rng.integers(0, 500, n)
        return pd.DataFrame({"video_id": vid, "label": (rng.random(n) < vid / 500).astype(int)})
    cfg = {"features": {"目标编码": {
        "enabled": True, "impl": "modules/features/target_encoding.py",
        "fields": ["video_id"], "smoothing": 20, "n_folds": 5, "target_col": "label"}}}
    tr, (va,), new = apply_feature_ops(load_feature_ops(cfg), make(20000), [make(3000)])
    return tr, va, new[0]


def test_目标编码的连续值分档后验证集几乎都认得():
    tr, va, col = _te_frames()
    before = (Vocab().fit(tr, [col]).encode(va)[:, 0] == OOV).mean()
    after = (Vocab(numeric_bins=32).fit(tr, [col]).encode(va)[:, 0] == OOV).mean()
    assert before > 0.2          # 不分档：实测 37%
    assert after < 0.01
    assert Vocab(numeric_bins=32).fit(tr, [col]).binned_fields == [col]


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
    assert codes[0] == 1                       # 最低档
    assert codes[2] == v.sizes()["x"] - 1       # 最高档
    assert codes[3] == OOV
    assert OOV not in codes[:3]


def test_不传档数时行为不变():
    tr, va, col = _te_frames()
    v = Vocab().fit(tr, [col])
    assert v.binned_fields == []
    assert v.sizes()[col] == tr[col].nunique() + 1
```

- [ ] **Step 2: 跑，确认失败**（`Vocab()` 不收 `numeric_bins` → TypeError；`binned_fields` 不存在）

Run: `.venv/bin/python -m pytest -p no:cacheprovider tests/test_numeric_auto_binning.py -q`

- [ ] **Step 3: 实现**

```python
class Vocab:
    def __init__(self, numeric_bins: int | None = None) -> None:
        self.maps: dict[str, dict[Any, int]] = {}
        self.edges: dict[str, np.ndarray] = {}
        self.numeric_bins = numeric_bins

    @property
    def binned_fields(self) -> list[str]:
        return list(self.edges)

    def _is_continuous(self, col: pd.Series) -> bool:
        return (self.numeric_bins is not None
                and pd.api.types.is_float_dtype(col)
                and col.nunique(dropna=True) > self.numeric_bins)

    def fit(self, df, fields):
        for f in fields:
            col = df[f]
            if self._is_continuous(col):
                qs = np.linspace(0, 1, self.numeric_bins + 1)[1:-1]
                self.edges[f] = np.unique(np.nanquantile(col.to_numpy(dtype=float), qs))
                self.maps[f] = {b: b + 1 for b in range(len(self.edges[f]) + 1)}
            else:
                self.maps[f] = {v: i + 1 for i, v in enumerate(col.dropna().unique())}
        return self

    def encode(self, df):
        cols = []
        for f in self.maps:
            if f in self.edges:
                v = df[f].to_numpy(dtype=float)
                code = np.searchsorted(self.edges[f], v, side="right") + 1
                code[np.isnan(v)] = OOV
                cols.append(code.astype("int64"))
            else:
                cols.append(df[f].astype("object").map(self.maps[f]).fillna(OOV)
                            .astype("int64").to_numpy())
        return np.stack(cols, axis=1) if cols else np.zeros((len(df), 0), dtype="int64")
```

- [ ] **Step 4: 跑，确认通过**
- [ ] **Step 5: 提交** `feat(harness): 连续值特征按训练集分位数自动分档`

### Task 2: train_deep 接上档数配置、训练记录写出被分档的列

**Files:**
- Modify: `harness/deep.py`（`DEEP_PARAMS`、`train_deep`）
- Test: `tests/test_numeric_auto_binning.py`（追加）

**Interfaces:**
- Consumes: `Vocab(numeric_bins=...)`、`Vocab.binned_fields`（Task 1）
- Produces: `deep_kwargs(cfg)["numeric_bins"]`；`train_deep` 返回的记录里 `"自动分档的列": list[str]`

- [ ] **Step 1: 写失败测试**

```python
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
        y = torch.tensor(batch["label"].to_numpy(), dtype=torch.float32)
        return torch.nn.functional.binary_cross_entropy(out["ctr"].clamp(1e-7, 1 - 1e-7), y)

    _, _, 记录 = train_deep(cfg, make(300), make(100), ["user_id", "rate"], seed=0,
                          task_loss=loss, task_metric=lambda *a: {"primary": 0.5})
    assert 记录["自动分档的列"] == ["rate"]
    assert 记录["_vocab"].sizes()["rate"] <= 8 + 1
```

- [ ] **Step 2: 跑，确认失败**（`numeric_bins` 不在白名单；记录里没有这个键）
- [ ] **Step 3: 实现**：`DEEP_PARAMS` 加 `"numeric_bins": (2, 256, 32)`；
  `vocab = Vocab(numeric_bins=kw["numeric_bins"]).fit(train, features)`；
  返回的记录加 `"自动分档的列": vocab.binned_fields`
- [ ] **Step 4: 跑全量测试，确认通过、不退步**
- [ ] **Step 5: 提交** `feat(harness): 档数从配置读，训练诊断里看得到哪些列被分档`

### Task 3: 卡片上"要先分档"的提醒改成"流水线会自动分档"

**Files:**
- Modify: `knowledge/cards/target_encoding.yaml`、`knowledge/cards/user_history.yaml`、`knowledge/cards/din_pooling.yaml`

- [ ] **Step 1**：目标编码卡 —— 删掉"不能直接这样挂 / 要写新零件包一层分档"，改成
  "直接挂就行：连续值由流水线按训练集分位数自动分档（`model.deep.numeric_bins`），
  成绩单「训练诊断 → 每轮训练记录 → 自动分档的列」里能看到"；失败信号里
  "先确认编码值分档了没有"改成"先看「自动分档的列」里有没有它"
- [ ] **Step 2**：用户历史序列卡 —— "看完率这种连续值要先分档……再作为新列输出"
  改成"看完率这种连续值直接输出，流水线会自动分档"
- [ ] **Step 3**：DIN 卡退路那句 —— "分档后当普通类别特征进模型"改成"当普通特征进模型"
- [ ] **Step 4**：跑守卫测试 + `agent.cli check` + 全量测试
- [ ] **Step 5**：提交 `docs(knowledge): 分档交给流水线，卡片不再让工兵自己分`，fetch 后 push

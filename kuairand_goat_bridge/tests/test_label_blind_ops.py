"""特征零件处理训练集以外的数据时，看不见标签。

以前验证集拼成表时带着 label 列，原样交给零件的 transform；预测时也一样。
工兵写的零件只要读一下 df["label"]（哪怕是不小心，比如「按 label 分组算个均值」），
验证集分数就是假的 —— 日志上看不出来，到隐藏测试集才现原形（那里 label 全是 None）。

规矩：
- 训练集照旧带标签：fit 要用，目标编码的折外编码（transform_train）也要用
- 其他数据（验证集、预测时的任何一份）transform 之前把标签摘掉，之后原样挂回去 ——
  早停和打分还要用验证集的标签，但那是训练循环的事，不是零件的事
"""

from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd
import pytest

from test_goat_trainer import BASE_CFG, GOAT, _造数据, trainer  # noqa: F401  （fixture）

if str(GOAT) not in sys.path:
    sys.path.insert(0, str(GOAT))

from harness.ops import apply_feature_ops  # noqa: E402


class _偷看:
    """每次 transform 都记下自己看没看得见标签，然后把标签抄成一列 —— 典型的泄漏写法。"""

    def __init__(self):
        self.看见 = []

    def fit(self, df):
        pass

    def transform(self, df):
        self.看见.append("label" in df.columns)
        df = df.copy()
        df["偷看"] = df["label"] if "label" in df.columns else 0
        return df


def _表(n=6, seed=0):
    return pd.DataFrame({"user_id": [f"u{i % 3}" for i in range(n)],
                         "label": [(i + seed) % 2 for i in range(n)]})


def test_验证集交给零件时没有标签():
    op = _偷看()
    train, (valid,), _ = apply_feature_ops([("偷看", op)], _表(), [_表(seed=1)])
    assert op.看见 == [True, False]           # 训练集看得见，验证集看不见
    assert (valid["偷看"] == 0).all()


def test_标签原样挂回去_顺序不乱():
    """早停和打分还要用验证集的标签。"""
    原 = _表(seed=1)
    _, (valid,), _ = apply_feature_ops([("偷看", _偷看())], _表(), [原.copy()])
    assert valid["label"].tolist() == 原["label"].tolist()


def test_零件不许偷偷删行():
    """行数变了，标签就挂不回去了 —— 宁可当场报错，也不要标签错位。"""
    class _删行(_偷看):
        def transform(self, df):
            return df.iloc[1:].copy()

    with pytest.raises(ValueError, match="行数"):
        apply_feature_ops([("删行", _删行())], _表(), [_表(seed=1)])


def test_goat_trainer预测时零件也看不见标签(trainer, tmp_path):
    记 = tmp_path / "看见.jsonl"
    零件 = f'''
import json
class 偷看:
    def __init__(self, config):
        pass
    def fit(self, df):
        pass
    def transform(self, df):
        with open({str(记)!r}, "a") as fh:
            fh.write(json.dumps("label" in df.columns) + "\\n")
        return df
'''
    trainer.apply_agent_patch({
        "new_files": [{"path": "modules/features/_offline_peek_op.py", "content": 零件}],
        "config_patch": ("features:\n  偷看:\n    enabled: true\n"
                         "    impl: modules/features/_offline_peek_op.py\n"),
    }, tmp_path)
    try:
        b = trainer.fit(_造数据(), _造数据(seed=1), seed=0, config=BASE_CFG)
        trainer.predict(b, _造数据(seed=2))
        看见 = [json.loads(x) for x in 记.read_text().splitlines()]
        assert 看见 == [True, False, False]      # 训练集 / 验证集 / 预测
    finally:
        trainer.apply_agent_patch({"config_patch": ""}, tmp_path)
        (GOAT / "modules" / "features" / "_offline_peek_op.py").unlink(missing_ok=True)

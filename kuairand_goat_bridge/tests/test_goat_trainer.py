"""GOAT Trainer —— AliCCP 那套「Agent 自己写代码」的能力接到 KuaiRand 上。

换个数据集就把整套框架扔掉没有道理：路径守卫、R2 纪律、embedding 表、
epoch 循环、TrainOp 回调、最佳权重回滚 —— 全部复用 harness/，
这份 trainer 只是把 KuaiRand 的数据翻译成它们吃的形状。
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

BRIDGE = pathlib.Path(__file__).resolve().parents[1]
GOAT = BRIDGE.parent
for p in (str(GOAT), str(BRIDGE / "official_starter_kit")):
    if p not in sys.path:
        sys.path.insert(0, p)


def _need(*mods):
    import importlib
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as exc:                       # noqa: BLE001
            pytest.skip(f"{m} 不可用：{type(exc).__name__}: {exc}")


@pytest.fixture
def trainer():
    _need("torch", "pandas")
    spec = importlib.util.spec_from_file_location(
        "goat_trainer", BRIDGE / "examples" / "goat_trainer.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class _Split:
    def __init__(self, rows): self.rows = rows
    def __len__(self): return len(self.rows)


def _造数据(n_users=60, per_user=12, seed=0):
    """带真实信号的合成数据 —— 没信号 GAUC 恒等 0.5，测不出训练有没有起作用。"""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        for _ in range(per_user):
            v = int(rng.integers(0, 40))
            a = v % 7
            dur = float(rng.integers(1000, 60000))
            p = 1 / (1 + np.exp(-(v / 40 + a / 7 - 1)))
            rows.append(("20220408", f"u{u}", f"v{v}", f"a{a}", "1", dur,
                         int(rng.random() < p)))
    return _Split(rows)


BASE_CFG = {"model": {"name": "goat_mlp", "impl": "modules/models/mlp.py",
                      "mlp": {"hidden": [16], "tower": [8], "dropout": 0.0},
                      "deep": {"epochs": 2, "batch_size": 256,
                               "learning_rate": 0.05}},
            "train": {}}


def test_能训能评能预测(trainer):
    """指标走官方 evaluate，不自己实现 —— 自己实现等于造一个跟官方差一点的分数。"""
    import numpy as np
    b = trainer.fit(_造数据(), _造数据(seed=1), seed=0, config=BASE_CFG)
    记录 = b["训练记录"]
    assert 记录["训练轮数"] == 2
    第一轮 = 记录["每轮"][0]
    assert {"primary", "GAUC", "nDCG@5"} <= set(第一轮)
    # train_deep 拿 metrics 里第一个非 loss 的键当主指标挑最佳轮次 ——
    # task_metric 必须把 primary 放第一位（官方主分就是它）
    指标 = trainer.task_metric.__doc__ and list(
        trainer.task_metric(b["op"], b["model"], b["vocab"],
                            trainer.rows_to_frame(_造数据(seed=1).rows,
                                                  b["duration_edges"])[0]))
    assert 指标[0] == "primary"

    scores = trainer.predict(b, _造数据(seed=1))
    assert len(scores) == len(_造数据(seed=1))
    assert np.isfinite(scores).all()


def test_Agent写的特征代码真的会被加载并进特征表(trainer, tmp_path):
    """这是整件事的核心：换了数据集，Agent 依然能自己写代码。"""
    零件 = '''
class Doubler:
    def __init__(self, config):
        self.k = config["features"]["演示"]["k"]
        self.seen = None
    def fit(self, train_df):
        self.seen = int(train_df["video_id"].nunique()) * self.k
    def transform(self, df):
        df = df.copy(); df["演示列"] = self.seen; return df
'''
    trainer.apply_agent_patch({
        "new_files": [{"path": "modules/features/_offline_demo_op.py",
                       "content": 零件}],
        "config_patch": ("features:\n  演示:\n    enabled: true\n"
                         "    impl: modules/features/_offline_demo_op.py\n    k: 2\n"),
    }, tmp_path)
    try:
        b = trainer.fit(_造数据(), _造数据(seed=1), seed=0, config=BASE_CFG)
        assert b["装上的零件"] == ["演示"]
        assert "演示列" in b["fields"]           # 新列自动进特征表，工兵不用改 base_fields
    finally:
        trainer.apply_agent_patch({"config_patch": ""}, tmp_path)
        (GOAT / "modules" / "features" / "_offline_demo_op.py").unlink(missing_ok=True)


def test_不许写到modules之外(trainer, tmp_path):
    """R5：放开一寸就等于让 Agent 改主程序。"""
    for 坏路径 in ("harness/executor.py", "modules/../harness/x.py"):
        with pytest.raises(ValueError, match="非法写入路径"):
            trainer.apply_agent_patch(
                {"new_files": [{"path": 坏路径, "content": ""}]}, tmp_path)


def test_只准动三棵子树(trainer, tmp_path):
    """改了 evaluation 等于偷偷换考卷，前几轮的分数全没法比。"""
    with pytest.raises(ValueError, match="只准动"):
        trainer.apply_agent_patch(
            {"config_patch": "evaluation:\n  primary: 随便\n"}, tmp_path)


def test_补丁累积重放(trainer, tmp_path):
    trainer.apply_agent_patch({"history": [
        {"config_patch": "model:\n  deep:\n    epochs: 3\n"},
        {"config_patch": "model:\n  deep:\n    learning_rate: 0.01\n"},
    ]}, tmp_path)
    b = trainer.fit(_造数据(), _造数据(seed=1), seed=0, config=BASE_CFG)
    assert b["训练记录"]["训练轮数"] == 3                      # 两条都生效
    assert b["config"]["model"]["deep"]["learning_rate"] == 0.01
    trainer.apply_agent_patch({"config_patch": ""}, tmp_path)


def test_duration分桶边界只从训练集拟合(trainer):
    """R2：拿验证集的分布去定边界 = 偷看，分数虚高、测试集必掉。"""
    train = _造数据()
    _, edges = trainer.rows_to_frame(train.rows)
    df2, edges2 = trainer.rows_to_frame(_造数据(seed=9).rows, edges)
    assert (edges2 == edges).all()                    # 沿用训练集的边界，没重新拟合
    assert "duration_bucket" in df2.columns

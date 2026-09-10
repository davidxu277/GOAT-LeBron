"""train.<名字> 块写残了必须当场炸，不能静默跳过。

真实事故（2026-09-01 本地全跑，第 2 轮）：配置模板里的块长这样 ——

    train:
      early_stopping:
        inner_holdout_frac: 0.1     # 没有 enabled，没有 impl

`load_train_ops` 看到没有 `enabled` 就 `continue`，零件从头到尾没被加载。
军师于是往一个没通电的盒子里塞了 `patience: 3`，训练照常跑，124,909 个
预测值跟改动前 **bit 级完全相同**。复盘官只能判「说不清」。

（更坑的是 `inner_holdout_frac` 只有 harness/executor.py 那条 LightGBM
路径读，deep.py 根本不读它 —— 那个块是三重失效。）

这正是仓库自己在 config/pipeline.yaml 里写下的那句话要防的：
「以前没有加载机制时，改这里等于没改：训练结果纹丝不动，却被记成
『这个方案没用』。现在宁可当场炸，也不假装跑过。」

闸门只建了一半：`enabled: true` 却没写 impl 会被挡，而**压根没写
enabled** 这一种直接漏过去了。
"""

import pytest

from tests.conftest import need


def _load(config):
    need("torch", "pandas")
    from harness.deep import load_train_ops

    return load_train_ops(config)


def test_块带了参数却没写_enabled_要报错():
    with pytest.raises(ValueError, match="early_stopping"):
        _load({"train": {"early_stopping": {"inner_holdout_frac": 0.1}}})


def test_写了_enabled_却没指路_impl_要报错():
    with pytest.raises(ValueError, match="early_stopping"):
        _load({"train": {"early_stopping": {"enabled": True}}})


def test_明确写了_enabled_false_是合法的关闭开关():
    assert _load({"train": {"early_stopping": {"enabled": False}}}) == []


def test_没有_train_段不报错():
    assert _load({}) == []


def test_train_下的标量不当成零件声明():
    """official_fm_trainer 那套配置把 epochs / batch_size 直接放在 train 下。"""
    assert _load({"train": {"epochs": 40, "batch_size": 8192}}) == []


def test_不是零件的普通配置段不要误伤():
    """`train.loss_weight` 是 config/pipeline.yaml 里的普通配置段，
    modules/train/ 下没有同名零件，不该因为缺 enabled 就报错。

    判据必须能分开两件事：
      · early_stopping —— modules/train/early_stopping.py 存在 = 你在配一个零件
      · loss_weight    —— 没有同名零件 = 普通配置，执行器自己读
    """
    assert _load({
        "train": {"loss_weight": {"strategy": "fixed", "ctr": 1.0, "cvr": 1.0}}
    }) == []

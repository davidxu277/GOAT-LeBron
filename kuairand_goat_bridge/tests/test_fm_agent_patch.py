"""官方 FM Trainer 的 Agent 补丁钩子。

没有这个钩子时，Bridge 的 `_apply_agent_patch` 会抛 NotImplementedError
（它不肯静默忽略修改，这个设计是对的）—— 于是工兵只要产出任何补丁，
整轮就作废，这条路一轮都跑不成。
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

BRIDGE = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BRIDGE / "official_starter_kit"))


@pytest.fixture
def trainer():
    spec = importlib.util.spec_from_file_location(
        "official_fm_trainer", BRIDGE / "examples" / "official_fm_trainer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = {"model": {"k": 8, "learning_rate": 0.05},
        "train": {"epochs": 5, "batch_size": 1024,
                  "early_stopping_patience": 3, "min_delta": 0.0}}


def test_钩子存在(trainer):
    """Bridge 靠 getattr 找它；不存在就整轮作废。"""
    assert callable(getattr(trainer, "apply_agent_patch", None))


def test_工兵改的参数真的到得了训练(trainer, tmp_path):
    trainer.apply_agent_patch(
        {"config_patch": "model:\n  k: 16\ntrain:\n  epochs: 8\n"}, tmp_path)
    got = trainer._read_config(BASE)
    assert got["k"] == 16 and got["epochs"] == 8
    assert got["learning_rate"] == 0.05        # 没碰的键保持原样


def test_补丁是累积重放的(trainer, tmp_path):
    """每轮拿 history 从初始配置重叠一遍 —— 任何一轮都能从日志完整复现，
    也不会因为中间某轮被回滚而留下脏状态。"""
    trainer.apply_agent_patch({"history": [
        {"config_patch": "model:\n  k: 16\n"},
        {"config_patch": "model:\n  learning_rate: 0.01\n"},
    ]}, tmp_path)
    got = trainer._read_config(BASE)
    assert got["k"] == 16 and got["learning_rate"] == 0.01   # 两条都在，没互相冲掉


def test_深度合并不冲掉兄弟键(trainer, tmp_path):
    """浅层赋值会把整棵 model 子树换掉，工兵只想改 k 却把 learning_rate 抹了。"""
    trainer.apply_agent_patch({"config_patch": "model:\n  k: 32\n"}, tmp_path)
    assert trainer._read_config(BASE)["learning_rate"] == 0.05


@pytest.mark.parametrize("补丁,坏键", [
    ("train:\n  epochs: 5000\n", "epochs"),
    ("model:\n  k: 999\n", "k"),
    ("model:\n  learning_rate: 3.0\n", "learning_rate"),
])
def test_越界当场报错而不是静默夹回(trainer, tmp_path, 补丁, 坏键):
    """epochs 一个 500 就能把整场算力烧光，而 Agent 自己看不出是它干的。

    报错而不是夹回：让复盘官知道这是「提案不合法」，不是「这个方法没用」——
    两者对卡片信任分的处置完全不同。
    """
    with pytest.raises(ValueError, match=坏键):
        trainer.apply_agent_patch({"config_patch": 补丁}, tmp_path)


def test_只准动model和train两棵子树(trainer, tmp_path):
    """改了 evaluation 就等于偷偷换考卷，前几轮的分数全没法比了。"""
    with pytest.raises(ValueError, match="只准动"):
        trainer.apply_agent_patch(
            {"config_patch": "evaluation:\n  primary: 随便\n"}, tmp_path)


def test_写新代码文件要说清楚为什么不行(trainer, tmp_path):
    with pytest.raises(NotImplementedError, match="只接受配置实验"):
        trainer.apply_agent_patch(
            {"new_files": [{"path": "modules/x.py", "content": ""}]}, tmp_path)


def test_生效的覆盖值要落盘(trainer, tmp_path):
    """日志里看得见，才查得出问题。"""
    trainer.apply_agent_patch({"config_patch": "model:\n  k: 12\n"}, tmp_path)
    written = (tmp_path / "agent_overrides.yaml").read_text(encoding="utf-8")
    assert "k: 12" in written


def test_空补丁不留脏状态(trainer, tmp_path):
    """上一轮改过、这一轮没改 —— 不该还带着上一轮的覆盖值。"""
    trainer.apply_agent_patch({"config_patch": "model:\n  k: 64\n"}, tmp_path)
    trainer.apply_agent_patch({"config_patch": ""}, tmp_path)
    assert trainer._read_config(BASE)["k"] == 8      # 回到任务配置的值

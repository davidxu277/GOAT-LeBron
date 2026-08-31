"""配置守卫 —— 盯住那些「改一行就把整个能力关掉」的地方。

08-31 23:39 `kuairand_task.yaml` 的 trainer 从 `goat_trainer.py` 被换成
`official_fm_trainer.py`。那一行是**静默**的：跑得起来、有分数、日志正常，
唯一的区别是 Agent 从「能写特征能写模型」退化成「只能调 6 个超参数」。

真跑日志里的代价：第 14~18 轮全废，军师连烧 5 轮才自己摸出
「带 new_files 的提案会被当场拒」。每一轮 = 一次真训练 + 四次大模型调用。

这类改动不该靠人记得，得有测试盯着。
"""

from __future__ import annotations

import pathlib

import pytest
import yaml


仓库根 = pathlib.Path(__file__).resolve().parent.parent
配置目录 = 仓库根 / "kuairand_goat_bridge" / "configs"
范例目录 = 仓库根 / "kuairand_goat_bridge" / "examples"


def _读(名字: str) -> dict:
    return yaml.safe_load((配置目录 / 名字).read_text(encoding="utf-8"))


def _支持写代码(trainer_相对路径: str) -> bool:
    """从 Trainer 源码里读它自己声明的 AGENT_SUPPORTS_NEW_FILES。"""
    源码 = (仓库根 / "kuairand_goat_bridge" / trainer_相对路径).read_text(
        encoding="utf-8")
    for 行 in 源码.splitlines():
        if 行.startswith("AGENT_SUPPORTS_NEW_FILES"):
            return 行.split("=", 1)[1].strip() == "True"
    return False


def test_正式跑的配置必须让Agent能写代码():
    """赛题奖励「超越简单基线调参」。

    kuairand_task.yaml 是正式跑那条路。它的 trainer 一旦换成只认超参数的，
    Agent 就退化成调参器 —— 而这件事不会报错、不会掉分，
    只会让整场跑下来什么代码都没产出。
    """
    trainer = _读("kuairand_task.yaml")["trainer"]
    assert _支持写代码(trainer), (
        f"kuairand_task.yaml 的 trainer 是 {trainer}，它不支持 new_files —— "
        "Agent 写的任何零件都会被当场拒掉、整轮作废。\n"
        "要复现官方基线请用 configs/fm_baseline.yaml，别改这一份。"
    )


def test_基线复现的配置强制校验第0轮():
    """基线复现那份的全部意义就是「跑出来必须跟官方对得上」。

    不强制校验的话，它跟随便跑一次没有区别。
    """
    基线 = _读("fm_baseline.yaml")
    assert 基线["require_baseline_reproduction"] is True
    assert not _支持写代码(基线["trainer"]), (
        "基线复现要的是一条**纯净**的官方 FM，不能让 Agent 往里加零件"
    )


def test_两份配置不许写到同一个输出目录():
    """一份是 FM、一份是深度 MLP，混在同一个目录里，

    best_report.json / 快照 / 补丁历史会互相覆盖 —— 而这些正是
    「最终交哪一版」依赖的东西。
    """
    甲 = _读("kuairand_task.yaml")["output_dir"]
    乙 = _读("fm_baseline.yaml")["output_dir"]
    assert 甲 != 乙


@pytest.mark.parametrize("名字", ["kuairand_task.yaml", "fm_baseline.yaml"])
def test_官方基线数值两份都得对得上(名字):
    """基线数字散在多个文件里，改一处忘一处就会悄悄走岔。"""
    validation = _读(名字)["official_baseline"]["validation"]
    assert validation["GAUC"] == 0.6674
    assert validation["nDCG@5"] == 0.5357
    assert abs(validation["primary"] - (0.6674 + 0.5357) / 2) < 1e-9


def test_两份配置都要真的能被加载器接受():
    """守卫自己用 yaml.safe_load 直接读，绕过了 load_task 的必填项校验 ——
    于是 7741a60 拆配置时删掉的 data_dir 一直没人发现，两份配置**都加载不了**，
    正式跑和基线跑同时起不来。这条改用真正的加载器，把这类缺口关上。
    """
    import sys
    sys.path.insert(0, str(配置目录.parent / "src"))
    from kuairand_bridge.goat_run import load_task

    for 名字 in ("kuairand_task.yaml", "fm_baseline.yaml"):
        配置 = load_task(str(配置目录 / 名字))
        assert pathlib.Path(配置["data_dir"]).is_dir(), (
            f"{名字} 的 data_dir 指向的目录不存在：{配置['data_dir']}"
        )

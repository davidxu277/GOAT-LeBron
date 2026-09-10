"""new_files 只能放仓库里还不存在的文件；复用已有零件靠 config_patch。

真实事故（2026-09-01，两场跑各中一次）：军师开 SWA 的方，how_to 写的是
"新增 modules/train/swa.py" —— 但那个文件仓库里**本来就有**。工兵照办，
落地器的路径守卫抛 FileExistsError，整轮作废。第一场里那条补丁还留在
patch history 里，把后面 43 轮全部毒死。

两层问题：

1. implementer.md 第 1 条写着「文件路径以军师为准」，工兵没有别的选择。
   提示词里缺一条「复用已有零件时 new_files 留空」，而且它压根不知道
   modules/ 下已经有哪些零件 —— 那份清单从来没进过它的视野。

2. 就算工兵想对，这个错也要等训练子进程启动、落地补丁时才炸，白白消耗
   50 次训练配额里的一次。校验阶段拦下来是免费的，而且能重试。

这条测试盯的是第 2 层：路径已存在 = 当场打回，并在报错信息里告诉它正确
做法。报错信息会被喂回给工兵，所以它必须写清楚怎么改。
"""

import pathlib

import pytest

from agent.llm import Ledger, SchemaViolation
from agent import roles


BASE_CHECK = ["未使用禁用字段 conversion", "统计量只用了训练集", "参数从配置读取"]


def _validate(data):
    captured = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            captured["validate"] = kw["validate"]
            return data

    roles.implement(_FakeLLM(), {"card_id": "", "how_to": ""}, None, "", "", "")
    captured["validate"](data)


def _patch(path, config_patch=""):
    return {
        "change_type": "加新零件",
        "config_patch": config_patch,
        "new_files": [{"path": path, "content": "# 零件\n"}],
        "self_check": BASE_CHECK,
    }


@pytest.mark.parametrize("path", [
    "modules/train/swa.py",
    "modules/train/early_stopping.py",
    "modules/models/mlp.py",
])
def test_把仓库里已有的零件塞进_new_files_要当场打回(path):
    with pytest.raises(SchemaViolation, match="已经存在"):
        _validate(_patch(path))


def test_打回信息要告诉工兵正确做法():
    """这段文字会被喂回给工兵当修正提示，必须说清楚怎么改才对。"""
    with pytest.raises(SchemaViolation) as e:
        _validate(_patch("modules/train/swa.py"))

    msg = str(e.value)
    assert "new_files" in msg and "config_patch" in msg, msg


def test_真正的新文件照常放行():
    _validate(_patch("modules/train/我肯定不存在_zzz.py"))


def test_下划线开头的演习草稿不受这条限制():
    """`modules/**/_offline_*.py` 是演习每轮重建的临时零件（.gitignore 挡着，
    进不了仓库）。这条守卫要保护的是**仓库里的用户代码**，不是磁盘上任何
    一个文件 —— 否则离线演习自己就跑不动了。

    判据跟 existing_modules_block() 对齐：`_` 开头的不算"现成零件"。
    """
    pathlib.Path("modules/features/_offline_scratch.py")  # 演习会真的创建它
    _validate(_patch("modules/features/_offline_scratch.py"))


def test_只发_config_patch_启用已有零件是合法的():
    _validate({
        "change_type": "只改配置",
        "config_patch": ("train:\n  swa:\n    enabled: true\n"
                         "    impl: modules/train/swa.py\n"),
        "new_files": [],
        "self_check": BASE_CHECK,
    })

"""测试套件的全局设置。

唯一的事：**别让测试往真事件流里写东西**。

测试会造假数据、故意触发报错（「配置要的模型是 deepfm」「特征一个都不在数据里」），
这些事件跟真跑的混在同一个 logs/live_events.jsonl 里，看板上分不清哪条是哪来的。
不是假想 —— 今天排查的时候就被自己的测试事件误导过两次。

放在 conftest.py 而不是每个测试里手动 monkeypatch：
新加的测试不用记得这件事，忘了也不会污染。
"""

import os

# 空字符串 = 谁也不写（见 agent/events.py 的 _target）
os.environ.setdefault("AGENT_EVENTS_PATH", "")


import pytest


def need(*模块名: str):
    """这些库装不上就跳过，别报红。

    `pytest.importorskip` 只兜 ImportError。lightgbm 在没装 libomp 的 mac 上
    抛的是 `OSError: dlopen ... libomp.dylib`，它兜不住 —— 结果是 12 个测试
    集体飘红，而 README 把测试套件宣传成"谁都能跑的零成本三条"之一。
    装不上不是测试失败，是这台机器上跑不了这条。
    """
    import importlib

    for name in 模块名:
        try:
            importlib.import_module(name)
        except Exception as exc:                     # noqa: BLE001
            pytest.skip(f"{name} 在这台机器上不可用：{type(exc).__name__}: {exc}")

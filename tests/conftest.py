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

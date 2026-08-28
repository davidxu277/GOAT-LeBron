"""实时事件流 —— 看板的数据源。

Agent 进程只干一件事：往 logs/live_events.jsonl 追加一行事件（写完即 flush）。
看板服务器 tail 这个文件推给浏览器。两边零耦合：看板崩了不影响跑分，
Agent 不装看板照样跑。事件契约见 docs/前端页面计划.md 第二节。

安全：只放行白名单字段，任何值里都不该出现密钥 —— 这里不做内容审查，
但调用方永远不要把 headers / api_key 塞进事件。
"""

from __future__ import annotations

import json
import os
import pathlib
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVENTS_PATH = ROOT / "logs" / "live_events.jsonl"


def _target() -> pathlib.Path | None:
    """这一条写到哪去。

    `AGENT_EVENTS_PATH=` （空字符串）= 谁也不写。测试套件就设成这个 ——
    否则 pytest 会往真事件流里灌几十条假事件（造的假数据、故意触发的报错），
    看板上跟真跑的事件混在一起，排查时根本分不清哪条是哪来的。
    这不是假想：今天调试时就被自己的测试事件误导过两次。
    """
    override = os.environ.get("AGENT_EVENTS_PATH")
    if override is None:
        return EVENTS_PATH
    return pathlib.Path(override) if override.strip() else None

_ALLOWED = {
    "type", "round", "role", "model", "text", "kind", "name", "detail",
    "tokens_in", "tokens_out", "retries", "verdict", "seconds", "fidelity",
    "ctr_auc", "cvr_auc", "error",
}


def emit(event_type: str, **fields: Any) -> None:
    """追加一条事件。永远不抛异常 —— 看板挂了不能影响主循环。"""
    try:
        path = _target()
        if path is None:
            return
        event = {"ts": time.strftime("%H:%M:%S"), "type": event_type}
        event.update({k: v for k, v in fields.items() if k in _ALLOWED})
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            fh.flush()
    except Exception:
        pass  # 看板是旁观者，绝不反过来弄崩主循环

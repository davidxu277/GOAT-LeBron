"""实时事件流 —— 看板的数据源。

Agent 进程只干一件事：往 logs/live_events.jsonl 追加一行事件（写完即 flush）。
看板服务器 tail 这个文件推给浏览器。两边零耦合：看板崩了不影响跑分，
Agent 不装看板照样跑。事件契约见 docs/实时看板计划.md 第二节。

安全：只放行白名单字段，任何值里都不该出现密钥 —— 这里不做内容审查，
但调用方永远不要把 headers / api_key 塞进事件。
"""

from __future__ import annotations

import json
import pathlib
import time
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
EVENTS_PATH = ROOT / "logs" / "live_events.jsonl"

_ALLOWED = {
    "type", "round", "role", "model", "text", "kind", "name", "detail",
    "tokens_in", "tokens_out", "retries", "verdict", "seconds", "fidelity",
    "ctr_auc", "cvr_auc", "error",
}


def emit(event_type: str, **fields: Any) -> None:
    """追加一条事件。永远不抛异常 —— 看板挂了不能影响主循环。"""
    try:
        event = {"ts": time.strftime("%H:%M:%S"), "type": event_type}
        event.update({k: v for k, v in fields.items() if k in _ALLOWED})
        EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with EVENTS_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
            fh.flush()
    except Exception:
        pass  # 看板是旁观者，绝不反过来弄崩主循环

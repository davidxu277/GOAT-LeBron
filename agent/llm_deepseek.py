"""DeepSeek 版的 LLM 封装 —— 对外接口与 llm.LLM 完全一致，roles.py 一行不用改。

与 Anthropic 版的三处差异：
  1. 结构化输出：DeepSeek 走 OpenAI 兼容接口，只有 json_object 模式
     （保证合法 JSON，不保证字段齐全），所以把 schema 原文拼进 system，
     靠现有的 validate + 重试机制兜底。
  2. 流式：stream=True 逐块接收 —— 一边打印到终端（实时可见），
     一边通过 events.emit 推给看板。若模型返回独立的 reasoning_content
     （思维链），单独标 kind=reasoning，绝不与正式回答混在一起。
  3. 不分大小模型：只有一个型号，big 参数被接受但忽略。

密钥只从环境变量 DEEPSEEK_API_KEY 读，绝不写进任何文件或日志。
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI

from .events import emit
from .llm import Ledger, SchemaViolation

DEEPSEEK_MODEL = os.getenv("AGENT_DEEPSEEK_MODEL", "deepseek-v4-flash")
BASE_URL = "https://api.deepseek.com"


class DeepSeekLLM:
    """结构化调用封装（DeepSeek 版）。call() 的契约与 llm.LLM.call 相同：
    要么返回通过校验的 dict，要么抛 SchemaViolation。"""

    def __init__(self, ledger: Ledger | None = None):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("缺少环境变量 DEEPSEEK_API_KEY")
        self.client = OpenAI(api_key=api_key, base_url=BASE_URL)
        self.ledger = ledger or Ledger()

    def call(
        self,
        *,
        role: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        big: bool = True,          # 接受但忽略：DeepSeek 侧不分大小模型
        effort: str = "high",      # 接受但忽略：无对应参数
        # 默认值给医生/军师/复盘官用：输出的 JSON 不长，但带思维链的模型
        # 推理过程也计入输出预算，所以留足余量。工兵单独设更大值（见 roles.py）。
        max_tokens: int = 32000,
        validate=None,
    ) -> dict[str, Any]:
        # json_object 模式不认识 schema，把 schema 原文拼进 system 靠提示词遵守，
        # 字段错漏由下面的 validate + 重试兜底
        system_full = (
            f"{system}\n\n"
            f"## 输出格式（必须是符合下面 JSON Schema 的单个 JSON 对象，不要任何其它文字）\n"
            f"```json\n{json.dumps(schema, ensure_ascii=False)}\n```"
        )

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_full},
            {"role": "user", "content": user},
        ]
        last_error = ""

        for attempt in range(2):
            text, reasoning, inp, out, truncated = self._stream_once(
                role, messages, max_tokens)

            # 被 max_tokens 掐断时 JSON 必然不完整，报错会是「Unterminated string」
            # 这种看起来像模型不听话的信息 —— 说清楚真实原因，别让人和模型都误判。
            if truncated:
                self.ledger.add(role, DEEPSEEK_MODEL, inp, out, retries=1)
                emit("recovery", role=role,
                     text=f"{role}输出被 max_tokens={max_tokens} 截断（已输出 {out} token），"
                          f"JSON 不完整。这是预算不够，不是模型不听话。")
                raise SchemaViolation(
                    f"{role}：输出超过 max_tokens={max_tokens} 被截断。"
                    f"调大限制，或让方案产出更短的代码。")

            try:
                data = json.loads(text)
                if validate is not None:
                    validate(data)
                self.ledger.add(role, DEEPSEEK_MODEL, inp, out, retries=attempt)
                emit("llm_end", role=role, model=DEEPSEEK_MODEL,
                     tokens_in=inp, tokens_out=out, retries=attempt)
                return data
            except (json.JSONDecodeError, SchemaViolation) as exc:
                last_error = str(exc)
                self.ledger.add(role, DEEPSEEK_MODEL, inp, out, retries=1)
                emit("recovery", role=role, text=f"{role}输出不合格，重试：{last_error[:120]}")
                messages = messages[:2] + [
                    {"role": "assistant", "content": text or "(空)"},
                    {"role": "user", "content": (
                        f"上一次输出不合格：{last_error}\n"
                        f"请严格按 schema 重新输出，只输出 JSON。"
                    )},
                ]

        raise SchemaViolation(f"{role}：重试后仍不合格 —— {last_error}")

    # ── 内部：流式收一整条回复，边收边打印边发事件 ──

    def _stream_once(
        self, role: str, messages: list[dict[str, str]], max_tokens: int
    ) -> tuple[str, str, int, int, bool]:
        """返回 (正式输出, 推理过程, 输入token, 输出token, 是否被截断)。"""
        emit("llm_start", role=role, model=DEEPSEEK_MODEL)
        print(f"\n┌─[{role}]{'─' * 40}", file=sys.stderr)

        stream = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            stream=True,
            stream_options={"include_usage": True},
        )

        answer_parts: list[str] = []
        reasoning_parts: list[str] = []
        inp = out = 0
        in_reasoning = False
        finish_reason = None

        for chunk in stream:
            if chunk.usage is not None:           # 最后一个 chunk 带用量
                inp = chunk.usage.prompt_tokens
                out = chunk.usage.completion_tokens
            if not chunk.choices:
                continue
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason
            delta = chunk.choices[0].delta

            # 思维链（模型不一定提供；有就单独收、单独标）
            rc = getattr(delta, "reasoning_content", None)
            if rc:
                if not in_reasoning:
                    print("│ (推理过程) ", end="", file=sys.stderr, flush=True)
                    in_reasoning = True
                reasoning_parts.append(rc)
                print(rc, end="", file=sys.stderr, flush=True)
                emit("llm_delta", role=role, text=rc, kind="reasoning")

            if delta.content:
                if in_reasoning:
                    print(f"\n│ (正式输出) ", end="", file=sys.stderr, flush=True)
                    in_reasoning = False
                answer_parts.append(delta.content)
                print(delta.content, end="", file=sys.stderr, flush=True)
                emit("llm_delta", role=role, text=delta.content, kind="answer")

        print(f"\n└{'─' * 46}", file=sys.stderr)
        # finish_reason="length" 就是撞上 max_tokens 上限的信号
        return ("".join(answer_parts), "".join(reasoning_parts), inp, out,
                finish_reason == "length")


def make_llm(ledger: Ledger | None = None):
    """按环境变量选提供商：AGENT_PROVIDER=deepseek 用 DeepSeek，默认 Anthropic。"""
    if os.getenv("AGENT_PROVIDER", "").lower() == "deepseek":
        return DeepSeekLLM(ledger=ledger)
    from .llm import LLM
    return LLM(ledger=ledger)

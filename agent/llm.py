"""调用 Claude 的唯一入口。

三件事全在这里：
  1. 结构化输出 —— 模型只能按给定 JSON Schema 回答
  2. 重试 —— 解析失败重试一次，再失败记一条恢复事件
  3. 记账 —— 按角色统计 token 与花费（占评分 15%，必须报出来）

分级用模型：动脑的活用大模型，照着范文写代码的活用小模型。
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import anthropic

# ── 模型分级 ────────────────────────────────────────────────────────
# 医生 / 军师 / 复盘官要判断和推理 → 大模型
# 工兵只是照着范文把草图写成代码 → 小模型
BIG_MODEL = os.getenv("AGENT_BIG_MODEL", "claude-opus-5")
SMALL_MODEL = os.getenv("AGENT_SMALL_MODEL", "claude-haiku-4-5")

# 每 100 万 token 的价格（美元），用于成本估算
# DeepSeek 用非高峰的 cache-miss 价（2026-08 官方价）。实际账单会更低：
# 每次请求自带上下文缓存，system prompt 这类重复前缀按 cache-hit 计价（便宜约 31 倍），
# 所以这里的估算是保守上限。高峰时段（周一至周五 UTC 01-04、06-10 点）价格翻倍。
PRICING = {
    "claude-opus-5": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "deepseek-v4-flash": (0.22, 0.66),
    "deepseek-v4-pro": (0.55, 2.19),
}

# effort 只在 Opus / Sonnet 5 系列上可用，Haiku 4.5 传了会报错
_EFFORT_CAPABLE = ("claude-opus-", "claude-sonnet-5", "claude-fable-")

# 超过这个 max_tokens 就必须走流式，否则 SDK 直接抛 ValueError
# （"Streaming is required for operations that may take longer than 10 minutes"）
_STREAM_THRESHOLD = 20000

<<<<<<< HEAD
# 每个模型的输出上限。要超了不是被截断，是整个请求 400 ——
# 真撞过：工兵按 roles.py 给的 96000 去调 Haiku 4.5（上限 64000），
# 主方案和两个备胎连着三次全挂，整轮作废，而 token 一个没花、
# 报错也只说"96000 > 64000"，不看这张表根本不知道该填多少。
_MAX_OUTPUT = {
    "claude-haiku-4-5": 64_000,
    "claude-opus-5": 128_000,
    "claude-sonnet-5": 128_000,
    "claude-fable-5": 128_000,
}
_MAX_OUTPUT_DEFAULT = 64_000       # 不认识的模型按保守值来，宁可短也不要 400


def cap_max_tokens(model: str, want: int) -> int:
    """把请求的输出预算夹到这个模型真能给的上限内。

    夹住而不是报错：调用方给 96000 的本意是"给我尽量多"，
    而不是"少于 96000 就别跑"。真被截断了另有 stop_reason 会说。
    """
    return min(want, _MAX_OUTPUT.get(model, _MAX_OUTPUT_DEFAULT))
=======
# Claude Haiku 4.5 rejects requests above 64k output tokens.  Implementer calls
# ask for a larger generic budget, so clamp only the API request for this model.
_HAIKU_45_MAX_OUTPUT_TOKENS = 64000

def _schema_for_anthropic(value: Any) -> Any:
    """Return the SDK's compatible wire copy without mutating our source schema.

    Claude structured outputs supports only a JSON Schema subset.  The SDK
    removes unsupported constraints (array lengths, numeric bounds, and so on)
    and records them in descriptions while role validators keep enforcing our
    local semantic rules.
    """
    # A few unit-test clients intentionally use an empty placeholder schema.
    # Production role schemas always have a root type.
    if not value:
        return value
    return anthropic.transform_schema(value)
>>>>>>> ea775fc92de615fd942806d60582a14a414979b0


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    retries: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class Ledger:
    """按角色记账。跑完之后 report() 出来的表直接进交付材料。"""

    by_role: dict[str, Usage] = field(default_factory=lambda: defaultdict(Usage))
    by_model: dict[str, Usage] = field(default_factory=lambda: defaultdict(Usage))

    def add(self, role: str, model: str, inp: int, out: int, retries: int = 0) -> None:
        for bucket in (self.by_role[role], self.by_model[model]):
            bucket.input_tokens += inp
            bucket.output_tokens += out
            bucket.calls += 1
            bucket.retries += retries

    @property
    def total_tokens(self) -> int:
        return sum(u.total_tokens for u in self.by_role.values())

    @property
    def total_cost_usd(self) -> float:
        total = 0.0
        for model, u in self.by_model.items():
            inp_price, out_price = PRICING.get(model, (0.0, 0.0))
            total += u.input_tokens / 1e6 * inp_price
            total += u.output_tokens / 1e6 * out_price
        return total

    def report(self) -> str:
        lines = ["角色           调用  输入token  输出token   合计"]
        for role, u in self.by_role.items():
            lines.append(
                f"{role:<12} {u.calls:>5} {u.input_tokens:>10} "
                f"{u.output_tokens:>10} {u.total_tokens:>8}"
            )
        lines.append(f"{'总计':<12} {'':>5} {'':>10} {'':>10} {self.total_tokens:>8}")
        lines.append(f"估算花费：${self.total_cost_usd:.4f}")
        return "\n".join(lines)


class SchemaViolation(RuntimeError):
    """模型的输出过不了我们自己的校验。"""


class LLM:
    """结构化调用封装。

    call() 保证：要么返回一个通过校验的 dict，要么抛 SchemaViolation。
    绝不会把半成品往下游传。
    """

    def __init__(self, client: anthropic.Anthropic | None = None, ledger: Ledger | None = None):
        self.client = client or anthropic.Anthropic()
        self.ledger = ledger or Ledger()

    def call(
        self,
        *,
        role: str,
        system: str,
        user: str,
        schema: dict[str, Any],
        big: bool = True,
        effort: str = "high",
        # 默认值给医生/军师/复盘官用：输出的 JSON 不长，但带思维链的模型
        # 推理过程也计入输出预算，所以留足余量。工兵单独设更大值（见 roles.py）。
        max_tokens: int = 32000,
        validate=None,
    ) -> dict[str, Any]:
        """跑一次结构化调用。

        validate: 可选的额外校验函数，接收解析后的 dict，
                  不合格时抛 SchemaViolation 并附带说明，我们会把说明喂回去重试一次。
        """
        model = BIG_MODEL if big else SMALL_MODEL
<<<<<<< HEAD
        max_tokens = cap_max_tokens(model, max_tokens)
=======
        request_max_tokens = max_tokens
        if model.startswith("claude-haiku-4-5"):
            request_max_tokens = min(max_tokens, _HAIKU_45_MAX_OUTPUT_TOKENS)
>>>>>>> ea775fc92de615fd942806d60582a14a414979b0
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": request_max_tokens,
            "system": system,
            "output_config": {
                "format": {
                    "type": "json_schema",
                    "schema": _schema_for_anthropic(schema),
                }
            },
        }
        if model.startswith(_EFFORT_CAPABLE):
            kwargs["output_config"]["effort"] = effort

        messages = [{"role": "user", "content": user}]
        last_error = ""

        for attempt in range(2):
            # SDK 规定：预估耗时可能超 10 分钟的请求必须走流式，
            # 而 max_tokens 一大就会撞上这条线（工兵要写整个文件，给的是 96k）。
            # 非流式在那种情况下会直接抛 ValueError，一个字都拿不到。
            if request_max_tokens > _STREAM_THRESHOLD:
                with self.client.messages.stream(**kwargs, messages=messages) as stream:
                    resp = stream.get_final_message()
            else:
                resp = self.client.messages.create(**kwargs, messages=messages)
            inp, out = resp.usage.input_tokens, resp.usage.output_tokens

            if resp.stop_reason == "refusal":
                self.ledger.add(role, model, inp, out, retries=attempt)
                raise SchemaViolation(f"{role}：模型拒绝了这次请求")

            # 撞上 max_tokens 时 JSON 必然不完整，报错会是「Unterminated string」
            # 这种看起来像模型不听话的信息 —— 说清真实原因，别让人和模型都误判。
            if resp.stop_reason == "max_tokens":
                self.ledger.add(role, model, inp, out, retries=1)
                raise SchemaViolation(
                    f"{role}：输出撞上 max_tokens={request_max_tokens} 被截断"
                    f"（已输出 {out} token），JSON 不完整。"
                    f"这是输出预算不够，不是模型不听话 —— 调大限制，"
                    f"或让方案产出更短的代码。"
                )

            text = next((b.text for b in resp.content if b.type == "text"), "")
            try:
                data = json.loads(text)
                if validate is not None:
                    validate(data)
                self.ledger.add(role, model, inp, out, retries=attempt)
                return data
            except (json.JSONDecodeError, SchemaViolation) as exc:
                last_error = str(exc)
                self.ledger.add(role, model, inp, out, retries=1)
                messages = [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": text or "(空)"},
                    {
                        "role": "user",
                        "content": (
                            f"上一次输出不合格：{last_error}\n"
                            f"请严格按 schema 重新输出，只输出 JSON。"
                        ),
                    },
                ]

        raise SchemaViolation(f"{role}：重试后仍不合格 —— {last_error}")

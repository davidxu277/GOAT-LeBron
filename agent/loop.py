"""一轮循环，以及与队友代码之间的接口。

四个 AI 角色之间永远隔着一段普通代码 —— 筛卡片、选方案、跑校验都在这里，
一个 token 都不花。

Scheduler 和 Executor 是成员4 的地盘。这里给出可运行的参考实现，
成员3 可以在数据和模型都还没好的时候先把整条链路跑通。
"""

from __future__ import annotations

import json
import pathlib
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .knowledge import Card, CardLibrary, SymptomVocab
from .llm import LLM, SchemaViolation
from . import roles

ROOT = pathlib.Path(__file__).resolve().parent.parent

# 代码难度 → 力气分
DIFFICULTY = {"改配置": 0.5, "简单": 1.0, "中等": 2.0, "难": 3.0}
FIDELITY_LADDER = ["小份", "中份", "大份", "全量"]


# ────────────────────────────── 接口 ──────────────────────────────


@dataclass
class RunResult:
    """跑完一次训练返回什么。字段由成员4 最终确定。"""

    ok: bool
    health_report: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    seconds: float = 0.0
    fidelity: str = "小份"


class Executor(Protocol):
    """成员4：跑代码、超时、错误恢复、红线校验。"""

    def run(self, patch: dict[str, Any], fidelity: str) -> RunResult: ...


class Scheduler(Protocol):
    """成员4：从军师给的 3 个方案里选 1 个，并决定跑在哪个数据尺寸上。"""

    def pick(
        self, proposals: list[dict[str, Any]], cards: CardLibrary, budget_left: str
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]: ...


# ────────────────────────────── 参考实现 ──────────────────────────────


class CostAwareScheduler:
    """成本感知调度器 —— 纯代码，不调用任何模型。

    先淘汰，再算性价比：

        性价比 = 预计提升 × 卡片靠谱度 ÷ 力气
        力气   = 代码难度 × 训练时间倍数

    没被选中的方案存为备胎：工兵写不出代码时直接换，
    不用重新去问军师，省一次大模型调用。
    """

    def __init__(self, tried_cards: set[str] | None = None):
        self.tried_cards = tried_cards or set()

    def score(self, proposal: dict[str, Any], cards: CardLibrary) -> float:
        gain = sum(max(0.0, v) for v in proposal["expected"].values())
        card = cards.get(proposal["card_id"]) if proposal["card_id"] else None
        prior = card.prior if card else 0.5     # 自创方案给中性先验
        effort = (
            DIFFICULTY.get(proposal["cost"]["代码难度"], 2.0)
            * max(0.1, float(proposal["cost"]["训练时间倍数"]))
        )
        return gain * prior / effort

    def pick(
        self, proposals: list[dict[str, Any]], cards: CardLibrary, budget_left: str
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
        alive = [p for p in proposals if p["card_id"] not in self.tried_cards]
        if not alive:
            alive = list(proposals)
        if budget_left == "紧张":
            alive = [p for p in alive if p["cost"]["代码难度"] in ("改配置", "简单")] or alive

        ranked = sorted(alive, key=lambda p: self.score(p, cards), reverse=True)
        chosen, *backups = ranked
        # 全新的招一律从最小的数据尺寸起步（CLAUDE.md：先小份，有效才升级）
        return chosen, FIDELITY_LADDER[0], backups


class FakeExecutor:
    """假执行器：不训练，直接回放一份预置的成绩单。

    用来在数据和模型都还没好的时候，把整条 Agent 链路先跑通。
    成员4 的真实现替换掉它即可。
    """

    def __init__(self, next_report: dict[str, Any], seconds: float = 0.0):
        self.next_report = next_report
        self.seconds = seconds

    def run(self, patch: dict[str, Any], fidelity: str) -> RunResult:
        return RunResult(
            ok=True, health_report=self.next_report, seconds=self.seconds, fidelity=fidelity
        )


# ────────────────────────────── 一轮 ──────────────────────────────


@dataclass
class RoundLog:
    """一轮的完整记录。这就是交付物 #3。"""

    round_id: int
    started_at: str
    diagnosis: dict[str, Any] | None = None
    proposals: dict[str, Any] | None = None
    chosen: dict[str, Any] | None = None
    fidelity: str = ""
    patch_summary: dict[str, Any] | None = None
    run_ok: bool = False
    reflection: dict[str, Any] | None = None
    recoveries: list[str] = field(default_factory=list)
    interventions: int = 0
    tokens: int = 0
    seconds: float = 0.0

    def dump(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(self), ensure_ascii=False) + "\n")


def run_round(
    *,
    round_id: int,
    llm: LLM,
    vocab: SymptomVocab,
    cards: CardLibrary,
    health_report: dict[str, Any],
    parent_result: dict[str, Any],
    executor: Executor,
    scheduler: Scheduler,
    module_interface: str,
    example_module: str,
    current_config: str,
    history_brief: list[dict[str, Any]] | None = None,
    budget_left: str = "一般",
) -> RoundLog:
    """跑完整的一轮：诊断 → 筛卡 → 提案 → 调度 → 实现 → 执行 → 复盘。"""

    t0 = time.time()
    log = RoundLog(round_id=round_id, started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    tokens_before = llm.ledger.total_tokens

    # ① 医生
    log.diagnosis = roles.diagnose(llm, vocab, health_report, history_brief)
    findings = log.diagnosis["findings"]
    if log.diagnosis["no_finding"]:
        log.recoveries.append("医生未发现明显问题，本轮跳过")
        log.tokens = llm.ledger.total_tokens - tokens_before
        log.seconds = time.time() - t0
        return log

    # ── 筛卡片：纯代码，不花钱 ──
    symptom_ids = [f["symptom"] for f in findings]
    candidates = cards.match(symptom_ids, limit=5)

    # ② 军师
    log.proposals = roles.propose(
        llm, vocab, findings, candidates,
        budget_left=budget_left, pipeline_state=current_config,
    )

    # ── 调度：纯代码，不花钱 ──
    chosen, fidelity, backups = scheduler.pick(log.proposals["proposals"], cards, budget_left)
    log.chosen, log.fidelity = chosen, fidelity

    # ③ 工兵（失败可重试，再失败换备胎）
    queue = [chosen, *backups]
    patch = None
    last_error = ""
    for candidate in queue:
        card = cards.get(candidate["card_id"]) if candidate["card_id"] else None
        try:
            patch = roles.implement(
                llm, candidate, card, module_interface, example_module,
                current_config, last_error=last_error,
            )
            log.chosen = candidate
            break
        except SchemaViolation as exc:
            last_error = str(exc)
            log.recoveries.append(f"工兵实现失败（{candidate['card_id'] or '自创'}）：{exc}")

    if patch is None:
        log.recoveries.append("所有方案都实现失败，本轮放弃")
        log.tokens = llm.ledger.total_tokens - tokens_before
        log.seconds = time.time() - t0
        return log

    log.patch_summary = {
        "change_type": patch["change_type"],
        "new_files": [f["path"] for f in patch["new_files"]],
        "self_check": patch["self_check"],
    }

    # ── 执行：成员4 的地盘 ──
    result = executor.run(patch, fidelity)
    log.run_ok = result.ok
    if not result.ok:
        log.recoveries.append(f"执行失败：{result.error}")
        log.tokens = llm.ledger.total_tokens - tokens_before
        log.seconds = time.time() - t0
        return log

    # ④ 复盘官
    card = cards.get(log.chosen["card_id"]) if log.chosen["card_id"] else None
    log.reflection = roles.reflect(
        llm, vocab, log.chosen, result.health_report, parent_result, card
    )

    log.tokens = llm.ledger.total_tokens - tokens_before
    log.seconds = time.time() - t0
    return log

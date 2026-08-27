"""一轮循环，以及与队友代码之间的接口。

四个 AI 角色之间永远隔着一段普通代码 —— 筛卡片、选方案、跑校验都在这里，
一个 token 都不花。

Scheduler 和 Executor 是成员4 的地盘。这里给出可运行的参考实现，
成员3 可以在数据和模型都还没好的时候先把整条链路跑通。
"""

from __future__ import annotations

import json
import pathlib
import shutil
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import yaml

from .events import emit
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


@runtime_checkable
class Executor(Protocol):
    """成员4：跑代码、超时、错误恢复、红线校验。

    标了 runtime_checkable，别人的实现可以用 isinstance 当场自检对不对得上。
    """

    def run(self, patch: dict[str, Any], fidelity: str) -> RunResult: ...


@runtime_checkable
class Scheduler(Protocol):
    """成员4：从军师给的 3 个方案里选 1 个，并决定跑在哪个数据尺寸上。"""

    def pick(
        self, proposals: list[dict[str, Any]], cards: CardLibrary, budget_left: str
    ) -> tuple[dict[str, Any], str, list[dict[str, Any]]]: ...


# ────────────────────────────── 参考实现 ──────────────────────────────


class TimeLedger:
    """训练耗时记账 —— 实测倍数覆盖拍出来的「训练时间倍数」。

    卡片上的倍数是人估的；一张卡真跑过一次之后，真实耗时就摆在那里，不用再猜。
    倍数的分母用全部已记录运行的中位耗时 —— 自我校准，不需要人工指定基准配置。

    只对确定性的资源消耗做记账（跑一次就可信）。效果类的数字（提升、靠谱度）
    不归它管 —— 那些有噪声，得走复盘官那条路（见 docs/方法库进度.md 的交接说明）。
    """

    def __init__(self, records: dict[str, list[float]] | None = None):
        self.records: dict[str, list[float]] = records or {}

    def record(self, card_id: str, seconds: float) -> None:
        if not card_id or seconds <= 0:
            return
        self.records.setdefault(card_id, []).append(float(seconds))

    def multiplier(self, card_id: str, default: float) -> float:
        """该卡的实测倍数。没跑过、或没有别的卡可对比时，退回给定的猜测值。"""
        runs = self.records.get(card_id)
        others = [s for cid, secs in self.records.items() if cid != card_id for s in secs]
        if not runs or not others:
            return default
        all_secs = [s for secs in self.records.values() for s in secs]
        return statistics.median(runs) / statistics.median(all_secs)

    # ── 持久化：每轮结束后 dump 一次，重启不丢账 ──

    def dump(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.records, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> "TimeLedger":
        if not path.exists():
            return cls()
        return cls(records=json.loads(path.read_text(encoding="utf-8")))


class PriorLedger:
    """卡片靠谱度记账 —— 复盘官的结论累积在这里，卡片 yaml 原文不动。

    为什么不直接改 yaml：那是成员2 的知识库，是人写的先验。
    实验结论是另一回事，两者分开存，出了问题能各查各的。

    规则（docs/方法库进度.md 第五节）：
        猜对了且超过噪声带   +0.15
        目标毛病确实改善了   +0.05（可与上一条叠加）
        猜错了               -0.10
        没跑起来             -0.15
        说不清                0
    限幅 [0.05, 0.95] —— 一张卡永远不该被彻底判死或彻底封神。
    """

    HIT = 0.15
    SYMPTOM_IMPROVED = 0.05
    MISS = -0.10
    CRASHED = -0.15
    FLOOR, CEIL = 0.05, 0.95

    def __init__(self, values: dict[str, float] | None = None):
        self.values: dict[str, float] = values or {}

    def value(self, card_id: str, default: float) -> float:
        """这张卡当前的靠谱度。没有实验记录就用卡上的先验。"""
        return float(self.values.get(card_id, default))

    def apply(
        self,
        card_id: str,
        verdict: str,
        base_prior: float,
        *,
        symptom_improved: bool = False,
        beat_noise: bool = True,
    ) -> float:
        """按一次复盘结论更新靠谱度，返回更新后的值。"""
        if not card_id:                      # 自创方案没有卡可更新
            return base_prior
        delta = 0.0
        if verdict == "猜对了" and beat_noise:
            delta += self.HIT
        if symptom_improved:
            delta += self.SYMPTOM_IMPROVED
        if verdict == "猜错了":
            delta += self.MISS
        if verdict == "没跑起来":
            delta += self.CRASHED

        updated = self.value(card_id, base_prior) + delta
        updated = max(self.FLOOR, min(self.CEIL, updated))
        self.values[card_id] = updated
        return updated

    def apply_to(self, cards: CardLibrary) -> None:
        """把账本里的靠谱度盖到内存里的卡片上。每轮开始调一次。"""
        for card in cards.cards:
            card.prior = self.value(card.id, card.prior)

    def dump(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.values, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> "PriorLedger":
        if not path.exists():
            return cls()
        return cls(values=json.loads(path.read_text(encoding="utf-8")))


class InterventionLog:
    """人工干预记录 —— 交付物 #3 要报的那个数。

    赛题按「达到收敛所需的人工干预次数」给自主性打分，越少越高。

    但一个**只能是 0** 的数字不是观测值，是常量 —— 评委翻一眼代码就知道。
    所以必须让"非零"随手可得：跑的过程中任何人插了手，敲一条命令记下来

        python -m agent.cli intervene "第 7 轮撞 OOM，手动把 batch 调小了"

    有了这个口子，报出来的 0 才是一个真实的观测结果。

    边界（README 里也要写同一份）：
      不算干预 —— 跑之前的数据准备、环境搭建、写卡片写提示词、决定跑几轮
      算干预   —— 跑起来之后改配置改代码、手动杀掉某轮、手动挑提交版本
    """

    def __init__(self, path: pathlib.Path):
        self.path = path
        self._seen = len(self._read())

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in
                self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    @classmethod
    def record(cls, path: pathlib.Path, reason: str, round_id: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "时间": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "第几轮": round_id,
                "干了什么": reason,
            }, ensure_ascii=False) + "\n")

    def drain(self) -> list[dict[str, Any]]:
        """返回上次检查之后新增的干预。跑之前就存在的那些属于准备工作，不算。"""
        entries = self._read()
        fresh, self._seen = entries[self._seen:], len(entries)
        return fresh


SHELF_KEEP = 5          # 待议架最多留几条 —— 喂给军师的上下文不能越滚越大


class Shelf:
    """待议架 —— 军师提过、但调度器没挑中的方案。

    军师每轮想 3 个，只有 1 个会被执行。剩下两个以前除了当"实现失败的备胎"
    就直接扔了，下一轮它又从头把同样的推理做一遍 —— 那部分思维链是要花钱的。

    存下来，下一轮把它们摆回军师面前：还对症就直接复用，条件变了就明确放弃。

    三条过期规则（都在 relevant 里）：
      · 这张卡已经试过（失败拉黑，或已生效并入流水线）→ 丢
      · 目标毛病这一轮医生没再报 → 丢（病都没了，药自然不用留）
      · 同一张卡只留最近一次，最多留 SHELF_KEEP 条

    过期规则比"存下来"这件事本身更重要：陈旧的方案会把军师往回带，
    让它照着三轮前的诊断开药。
    """

    def __init__(self, entries: list[dict[str, Any]] | None = None):
        self.entries: list[dict[str, Any]] = entries or []

    @staticmethod
    def _key(entry: dict[str, Any]) -> tuple:
        # 自创方案没有 card_id，用目标病组合区分
        return (entry.get("card_id") or "", tuple(entry.get("targets") or []))

    def shelve(self, round_id: int, proposals: list[dict[str, Any]],
               chosen: dict[str, Any] | None) -> None:
        """把这一轮没被挑中的方案收进架子。"""
        chosen_key = self._key(chosen or {})
        for p in proposals:
            if self._key(p) == chosen_key:
                continue
            entry = {
                "提出于第几轮": round_id,
                "card_id": p.get("card_id", ""),
                "targets": list(p.get("targets") or []),
                "expected": p.get("expected", {}),
                "cost": p.get("cost", {}),
                # 只留个引子。军师需要的是"我想过这个"，不是把整段推理再读一遍
                "当时的理由": (p.get("rationale") or "")[:120],
            }
            self.entries = [e for e in self.entries if self._key(e) != self._key(entry)]
            self.entries.append(entry)
        self.entries = self.entries[-(SHELF_KEEP * 3):]     # 粗剪，精剪在 relevant

    def relevant(self, symptom_ids: list[str],
                 exclude_ids: set[str] | None = None) -> list[dict[str, Any]]:
        """挑出这一轮还说得通的存货。"""
        wanted = set(symptom_ids)
        exclude = exclude_ids or set()
        alive = [
            e for e in self.entries
            if (not e["card_id"] or e["card_id"] not in exclude)
            and wanted & set(e["targets"])
        ]
        return alive[-SHELF_KEEP:]

    def dump(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.entries, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: pathlib.Path) -> "Shelf":
        if not path.exists():
            return cls()
        return cls(entries=json.loads(path.read_text(encoding="utf-8")))


# 成绩单里两个 AUC 可能挂在哪 —— 真执行器写「验证集」，假成绩单写「总分」
_SCORE_SECTIONS = ("验证集", "总分")


def read_scores(report: dict[str, Any]) -> dict[str, float]:
    """从成绩单里取出两个 AUC。取不到的返回空 dict。

    收敛判定、挑最佳版本、算 delta 全靠它，所以两种成绩单格式都要认。
    """
    for section in _SCORE_SECTIONS:
        block = report.get(section)
        if isinstance(block, dict) and block.get("点击分") is not None:
            return {
                "点击AUC": float(block["点击分"]),
                "购买AUC": float(block.get("购买分") or 0.0),
            }
    return {}


def total_score(report: dict[str, Any]) -> float:
    """两个 AUC 之和 —— 排名按两项 delta 等权平均，和与均值同序。"""
    scores = read_scores(report)
    return sum(scores.values()) if scores else float("-inf")


class CostAwareScheduler:
    """成本感知调度器 —— 纯代码，不调用任何模型。

    先淘汰，再算性价比：

        性价比 = 预计提升 × 卡片靠谱度 ÷ 力气
        力气   = 代码难度 × 训练时间倍数

    训练时间倍数优先用 TimeLedger 里的实测值；这张卡没跑过才用军师报的数。
    没被选中的方案存为备胎：工兵写不出代码时直接换，
    不用重新去问军师，省一次大模型调用。
    """

    def __init__(
        self,
        tried_cards: set[str] | None = None,
        time_ledger: TimeLedger | None = None,
    ):
        self.tried_cards = tried_cards or set()
        self.time_ledger = time_ledger

    def score(self, proposal: dict[str, Any], cards: CardLibrary) -> float:
        gain = sum(max(0.0, v) for v in proposal["expected"].values())
        card = cards.get(proposal["card_id"]) if proposal["card_id"] else None
        prior = card.prior if card else 0.5     # 自创方案给中性先验

        multiplier = max(0.1, float(proposal["cost"]["训练时间倍数"]))
        if self.time_ledger is not None and proposal["card_id"]:
            multiplier = max(0.1, self.time_ledger.multiplier(proposal["card_id"], multiplier))

        effort = DIFFICULTY.get(proposal["cost"]["代码难度"], 2.0) * multiplier
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
    patch_files: dict[str, str] = field(default_factory=dict)   # 路径 → 完整代码，交付物要的 code diff
    run_ok: bool = False
    metrics: dict[str, Any] | None = None                       # 本轮成绩单原文，交付物要的 metrics
    reflection: dict[str, Any] | None = None
    recoveries: list[str] = field(default_factory=list)
    interventions: int = 0
    intervention_notes: list[str] = field(default_factory=list)   # 人到底干了什么，光有次数没用
    tokens: int = 0
    seconds: float = 0.0        # 整轮耗时（含 LLM 调用）
    train_seconds: float = 0.0  # 其中训练耗时。逐轮累加就是交付物要求的 GPU 总时长

    def dump(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(self), ensure_ascii=False) + "\n")


def _guard(log: RoundLog, what: str, fn, *args, **kwargs):
    """跑一个可能炸的步骤。炸了就记一笔恢复事件，返回 None，让上层决定怎么办。

    只吞 Exception —— KeyboardInterrupt 是 BaseException，Ctrl-C 照样能停。
    这是挂机跑一整夜的保险丝：一个角色抽风，作废这一轮，不是作废整场。
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:                     # noqa: BLE001 —— 就是要兜住全部
        log.recoveries.append(f"{what}失败：{type(exc).__name__}: {exc}")
        emit("recovery", text=f"{what}失败：{type(exc).__name__}: {exc}")
        return None


def _crashed_reflection(chosen: dict[str, Any] | None, error: str) -> dict[str, Any]:
    """执行失败时的复盘结论 —— 纯代码合成，不花一分钱去问大模型。

    跑都没跑起来，没有任何结果可复盘。这时候调大模型是纯浪费。
    """
    chosen = chosen or {}
    targets = chosen.get("targets") or []
    return {
        "verdict": "没跑起来",
        "actual": {"点击AUC": 0.0, "购买AUC": 0.0},
        "vs_expected": f"代码没跑通，拿不到结果：{error}",
        # 方案声称要治的病，逐个记「否」—— 跑都没跑起来，一个都没治
        "symptom_resolved": [
            {"symptom": t, "before": 0.0, "after": 0.0, "resolved": "否"}
            for t in targets
        ] or [{"symptom": "", "before": 0.0, "after": 0.0, "resolved": "否"}],
        "card_update": {
            "card_id": chosen.get("card_id", ""),
            "prior_delta": PriorLedger.CRASHED,
            "note": f"这一版没跑起来：{error}",
        },
        "next_hint": "换一个方案，或者先修这个报错",
        "promote": False,
        "由代码合成": True,          # 标记：不是复盘官说的，是代码填的
    }


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
    example_module: str | Callable[[str], str],
    current_config: str,
    history_brief: list[dict[str, Any]] | None = None,
    budget_left: str = "一般",
    time_ledger: TimeLedger | None = None,
    prior_ledger: PriorLedger | None = None,
    tried_before: list[dict[str, Any]] | None = None,
    exclude_ids: set[str] | None = None,
    shelf: "Shelf | None" = None,
    fidelity_override: str | None = None,
    noise_floor: float = roles.MIN_REAL_GAIN,
) -> RoundLog:
    """跑完整的一轮：诊断 → 筛卡 → 提案 → 调度 → 实现 → 执行 → 复盘。

    任何一个角色炸掉，本轮作废并返回，外层循环继续下一轮 —— 绝不把异常抛出去。
    """

    t0 = time.time()
    log = RoundLog(round_id=round_id, started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    tokens_before = llm.ledger.total_tokens

    def finish() -> RoundLog:
        # 收架子放在这里而不是选完方案时：工兵实现失败会换备胎，
        # log.chosen 到最后才定下来，那个才是真正被用掉的方案。
        if shelf is not None and log.proposals:
            shelf.shelve(round_id, log.proposals["proposals"], log.chosen)
        log.tokens = llm.ledger.total_tokens - tokens_before
        log.seconds = time.time() - t0
        return log

    # ① 医生
    log.diagnosis = _guard(log, "医生", roles.diagnose, llm, vocab, health_report, history_brief)
    if log.diagnosis is None:
        return finish()
    findings = log.diagnosis["findings"]
    if log.diagnosis["no_finding"]:
        log.recoveries.append("医生未发现明显问题，本轮跳过")
        return finish()

    # ── 筛卡片：纯代码，不花钱。试过且失败的卡在这里就被排除 ──
    symptom_ids = [f["symptom"] for f in findings]
    # 医生给的严重度直接进筛卡权重：治一个重病的卡，排在治两个轻病的卡前面
    severity = {f["symptom"]: f.get("severity", 1.0) for f in findings}
    candidates = cards.match(symptom_ids, exclude_ids=exclude_ids, limit=5, severity=severity)

    # ② 军师。把架子上还对症的存货一并摆给它，省得重新推导一遍
    shelved = shelf.relevant(symptom_ids, exclude_ids) if shelf is not None else None
    log.proposals = _guard(
        log, "军师", roles.propose,
        llm, vocab, findings, candidates,
        tried_before=tried_before, shelved=shelved,
        budget_left=budget_left, pipeline_state=current_config,
    )
    if log.proposals is None:
        return finish()

    # ── 调度：纯代码，不花钱 ──
    picked = _guard(log, "调度器", scheduler.pick, log.proposals["proposals"], cards, budget_left)
    if picked is None:
        return finish()
    chosen, fidelity, backups = picked
    log.chosen, log.fidelity = chosen, fidelity_override or fidelity

    # ③ 工兵（失败可重试，再失败换备胎）
    queue = [chosen, *backups]
    patch = None
    last_error = ""
    for candidate in queue:
        card = cards.get(candidate["card_id"]) if candidate["card_id"] else None
        # example_module 可以是字符串，也可以是「按环节取范文」的函数 ——
        # 改训练过程的方案看训练类范文，加特征的看特征类范文，产出质量差很多
        example = (example_module(card.stage if card else "")
                   if callable(example_module) else example_module)
        try:
            patch = roles.implement(
                llm, candidate, card, module_interface, example,
                current_config, last_error=last_error,
            )
            log.chosen = candidate
            break
        except SchemaViolation as exc:
            last_error = str(exc)
            log.recoveries.append(f"工兵实现失败（{candidate['card_id'] or '自创'}）：{exc}")
        except Exception as exc:                 # noqa: BLE001 —— 网络抖动等
            last_error = str(exc)
            log.recoveries.append(f"工兵调用出错（{candidate['card_id'] or '自创'}）：{exc}")

    if patch is None:
        log.recoveries.append("所有方案都实现失败，本轮放弃")
        return finish()

    log.patch_summary = {
        "change_type": patch["change_type"],
        "new_files": [f["path"] for f in patch["new_files"]],
        "config_patch": patch.get("config_patch", ""),
        "self_check": patch["self_check"],
    }
    # 交付物 #3 要的 code diff：完整代码，不是文件名列表
    log.patch_files = {f["path"]: f["content"] for f in patch["new_files"]}

    # ── 执行：成员4 的地盘。协议说返回 ok=False，但真实现难保不抛 ──
    result = _guard(log, "执行器", executor.run, patch, log.fidelity)
    if result is None:
        result = RunResult(ok=False, error="执行器抛异常，详见恢复记录", fidelity=log.fidelity)
    log.run_ok = result.ok
    log.train_seconds = result.seconds
    # 指标先落盘，再去复盘 —— 复盘官挂了也不能把这一轮的成绩单弄丢
    log.metrics = result.health_report or None

    # 耗时记账：实测值覆盖拍出来的倍数，下一轮调度就用真数
    if time_ledger is not None and result.ok and log.chosen.get("card_id"):
        time_ledger.record(log.chosen["card_id"], result.seconds)

    card = cards.get(log.chosen["card_id"]) if log.chosen["card_id"] else None

    if not result.ok:
        log.recoveries.append(f"执行失败：{result.error}")
        log.reflection = _crashed_reflection(log.chosen, result.error)
        if prior_ledger is not None and card is not None:
            prior_ledger.apply(card.id, "没跑起来", card.prior)
        return finish()

    # ④ 复盘官
    log.reflection = _guard(
        log, "复盘官", roles.reflect,
        llm, vocab, log.chosen, result.health_report, parent_result, card,
        noise_floor=noise_floor,
    )
    if log.reflection is not None and prior_ledger is not None and card is not None:
        gains = log.reflection["actual"].values()
        prior_ledger.apply(
            card.id, log.reflection["verdict"], card.prior,
            # 多个目标里只要有一个真的好转，这张卡就该加分
            symptom_improved=any(item["resolved"] in ("是", "部分")
                                 for item in log.reflection["symptom_resolved"]),
            beat_noise=max((abs(v) for v in gains), default=0.0) >= noise_floor,
        )
    return finish()


# ────────────────────────────── 一整场 ──────────────────────────────

# 收敛判定与预算。Starter Kit 公布官方 ε / N 之后改这里，别散落到各处（R7）。
DEFAULT_ROUNDS = 10
DEFAULT_EPSILON = 0.0005        # 提升小于它不算提升（CLAUDE.md R11）
DEFAULT_PATIENCE = 3            # 连续这么多轮没有真提升 = 收敛
DEFAULT_TOKEN_BUDGET = 2_000_000
BUDGET_TIERS = ((0.60, "宽裕"), (0.85, "一般"))   # 用超 85% → 紧张
NO_FINDING_ESCAPE = 2           # 连续几轮查不出病，就升一档数据再看


def _budget_tier(used: int, budget: int) -> str:
    ratio = used / budget if budget > 0 else 0.0
    for cap, name in BUDGET_TIERS:
        if ratio < cap:
            return name
    return "紧张"


def effective_config(executor: Any, fallback: str) -> str:
    """当前真正生效的配置文本。

    执行器把工兵的改动深度合并进自己内存里的 config，磁盘上那份 pipeline.yaml
    一直是初始状态。以前每轮都把初始文本喂给军师和工兵 —— 它们看到的流水线
    从第 2 轮起就是过期的，可能重复启用已经开着的零件。

    执行器没有 config 属性（假执行器）时退回传进来的文本。
    """
    cfg = getattr(executor, "config", None)
    if not isinstance(cfg, dict) or not cfg:
        return fallback
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)


def snapshot_round(logs_dir: pathlib.Path, log: RoundLog, config_text: str,
                   module_owner: dict[str, int]) -> None:
    """存下这一轮跑完之后流水线长什么样。

    为什么必须存：工兵的改动是**叠加**在同一份配置和同一个 modules/ 目录上的。
    跑到第 20 轮时，磁盘上是 1~20 轮所有改动叠在一起的样子 ——
    哪怕日志写着"第 5 轮最好"，第 5 轮那个状态已经被后面 15 轮盖掉了，重跑都回不去。
    而交付物 #4 要交的正是那一版。

    只存配置文本 + 一张"哪个文件是哪一轮写的"清单；文件内容本来就在
    rounds.jsonl 的 patch_files 里，不重复存。`agent.cli restore N` 照着还原。
    """
    path = logs_dir / "snapshots" / f"round_{log.round_id:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "轮次": log.round_id,
        "保真度": log.fidelity,
        "分数": read_scores(log.metrics or {}),
        "配置": config_text,
        "零件": dict(module_owner),      # 路径 → 哪一轮写的这一版
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def write_narrative(logs_dir: pathlib.Path, history: list[dict[str, Any]],
                    summary: "SessionSummary") -> None:
    """把一整场压成一条故事线。

    rounds.jsonl 里全是事实，但评委翻的时候想看的是一条线：
    第 3 轮发现什么 → 第 4 轮针对它试了什么 → 结论 → 第 5 轮顺着往哪走。
    这条线我们本来就攒着（history_brief，每轮喂给下一轮的医生），
    以前只给模型看，没写进交付物 —— 白瞎了。
    """
    lines = [
        "# 这一场发生了什么",
        "",
        f"> 跑了 {summary.rounds_run} 轮 · {summary.stopped_because}",
        f"> 人工干预 {summary.interventions} 次 · 错误恢复 {summary.recoveries} 次",
        f"> 最终提交第 {summary.best_round} 轮（{summary.best_fidelity}数据）",
        "",
        "| 轮 | 数据 | 试了什么 | 结论 | 实际变化 |",
        "|---|---|---|---|---|",
    ]
    for h in history:
        gains = h.get("实际变化") or {}
        delta = " / ".join(f"{k} {v:+.4f}" for k, v in gains.items()) or "—"
        lines.append(
            f"| {h['轮次']} | {h.get('数据') or '—'} | {h.get('选了')} "
            f"| {h.get('结论')} | {delta} |"
        )
    lines += ["", "## 每轮的下一步判断", ""]
    for h in history:
        note = (h.get("备注") or "").strip()
        if note:
            lines.append(f"- **第 {h['轮次']} 轮** → {note}")
    (logs_dir / "narrative.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _with_bands(report: dict[str, Any], bands: dict[str, Any] | None) -> dict[str, Any]:
    """把实测噪声带挂到成绩单上，医生才知道哪些差距是真的。

    只挂给医生看的那一份，不动执行器产出的原文（原文要进日志）。
    """
    if not bands:
        return report
    return {**report, "噪声带": {
        "单指标": bands.get("单指标噪声带"),
        "分组": {g: {k: v.get("购买分", {}).get("噪声带") for k, v in rows.items()}
                for g, rows in bands.get("分组", {}).items()},
        "怎么用": ("这是同配置换随机种子跑出来的抖动幅度。"
                 "任何小于它的差距都是噪声，不许当成病；"
                 "分桶差距要跟对应那个桶的噪声带比，别用统一阈值。"),
    }}


def _brief(log: RoundLog) -> dict[str, Any]:
    """把一轮压成一两行，喂给下一轮的医生和军师。全文喂过去会烧光预算。"""
    ref = log.reflection or {}
    return {
        "轮次": log.round_id,
        "选了": (log.chosen or {}).get("card_id") or "（自创或未选中）",
        "数据": log.fidelity,
        "结论": ref.get("verdict", "本轮作废"),
        "实际变化": ref.get("actual", {}),
        "备注": ref.get("next_hint", "") or "；".join(log.recoveries[:2]),
    }


@dataclass
class SessionSummary:
    """一整场跑完的汇总 —— 直接对应交付物 #5 的结果表。"""

    rounds_run: int = 0
    stopped_because: str = ""
    best_round: int = 0
    best_fidelity: str = ""
    best_scores: dict[str, float] = field(default_factory=dict)
    # 锁定集上的分数（R3：整场只评一次）。空 = 没配锁定集，或裁决失败。
    holdout_scores: dict[str, float] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)
    total_tokens: int = 0
    total_train_seconds: float = 0.0
    interventions: int = 0
    recoveries: int = 0

    @property
    def generalization_gap(self) -> dict[str, float]:
        """开发集分 − 锁定集分。正数越大，说明「涨的分」里迎合开发集的成分越多。

        这个数字本身就是交付材料 —— 诚实报告泛化落差，比藏着一个虚高的
        开发集分数有说服力得多。
        """
        if not self.holdout_scores or not self.best_scores:
            return {}
        return {m: v - self.holdout_scores.get(m, 0.0)
                for m, v in self.best_scores.items()
                if m in self.holdout_scores}

    @property
    def deltas(self) -> dict[str, float]:
        """相对官方基线的绝对差值 —— 排名真正用的数字。"""
        if not self.baseline or not self.best_scores:
            return {}
        return {m: self.best_scores.get(m, 0.0) - v for m, v in self.baseline.items()}

    def as_table(self) -> str:
        lines = [
            "══════════ 结果表（交付物 #5）══════════",
            f"跑了几轮        {self.rounds_run}（{self.stopped_because}）",
            f"最终提交        第 {self.best_round} 轮，{self.best_fidelity or '—'}数据",
        ]
        for metric, value in self.best_scores.items():
            lines.append(f"  {metric:<12} {value:.4f}")
        for metric, delta in self.deltas.items():
            lines.append(f"  {metric} 相对基线  {delta:+.4f}")
        if not self.baseline:
            lines.append("  ⚠️ 官方基线分未填，delta 算不出来（--baseline-ctr / --baseline-cvr）")
        if self.holdout_scores:
            lines.append("锁定集裁决      （整场只读一次，从未参与任何决策）")
            for metric, value in self.holdout_scores.items():
                gap = self.generalization_gap.get(metric)
                tail = f"   泛化落差 {gap:+.4f}" if gap is not None else ""
                lines.append(f"  {metric:<12} {value:.4f}{tail}")
        else:
            lines.append("锁定集裁决      未做（没配锁定集）—— 开发集分数可能偏乐观")
        lines += [
            f"LLM token 总量  {self.total_tokens:,}",
            f"训练总时长      {self.total_train_seconds / 3600:.3f} GPU-小时"
            f"（{self.total_train_seconds:.0f} 秒）",
            f"人工干预        {self.interventions} 次",
            f"错误恢复        {self.recoveries} 次",
        ]
        return "\n".join(lines)

    def dump(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        data["deltas"] = self.deltas
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def run_session(
    *,
    llm: LLM,
    vocab: SymptomVocab,
    cards: CardLibrary,
    executor: Executor,
    initial_report: dict[str, Any],
    initial_train_seconds: float = 0.0,   # 第 0 轮体检也烧了算力，要计进 GPU 小时
    module_interface: str,
    example_module: str | Callable[[str], str],   # 字符串，或按环节取范文的函数
    current_config: str,
    rounds: int = DEFAULT_ROUNDS,
    start_fidelity: str = FIDELITY_LADDER[0],   # 从哪一档数据起步
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    epsilon: float = DEFAULT_EPSILON,
    patience: int = DEFAULT_PATIENCE,
    noise_floor: float = roles.MIN_REAL_GAIN,
    noise_bands: dict[str, Any] | None = None,
    baseline: dict[str, float] | None = None,
    logs_dir: pathlib.Path | None = None,
    on_round=None,
) -> SessionSummary:
    """自主迭代 —— 一轮接一轮，中途不需要人碰键盘。

    这个函数就是"自主性"那 20% 分的实现。它管四件事，每件都不调用大模型：

      1. 接线    —— 这一轮跑出的成绩单，变成下一轮医生的输入
      2. 记忆    —— 试过且失败的卡拉黑；每轮压成一行喂给下一轮
      3. 升降级  —— 连着查不出病、或复盘官说值得复查，就升一档数据重测
      4. 判停    —— 跑满轮数 / 预算耗尽 / 连续 patience 轮没有真提升

    最后挑出"验证集最佳"的那一版作为最终提交（赛题：收敛时的最佳 checkpoint）。
    """
    logs_dir = logs_dir or (ROOT / "logs")
    time_ledger = TimeLedger.load(logs_dir / "time_ledger.json")
    prior_ledger = PriorLedger.load(logs_dir / "prior_ledger.json")
    shelf = Shelf.load(logs_dir / "shelf.json")
    interventions = InterventionLog(logs_dir / "interventions.jsonl")
    module_owner: dict[str, int] = {}      # 零件路径 → 哪一轮写的这一版

    if start_fidelity not in FIDELITY_LADDER:
        raise ValueError(f"没有「{start_fidelity}」这一档，只能是 {FIDELITY_LADDER}")

    cur = parent = initial_report
    history: list[dict[str, Any]] = []
    blacklist: set[str] = set()          # 试过且失败的卡 —— 调度器硬性跳过
    applied: set[str] = set()            # 已经生效、并入流水线的卡 —— 再上一次没意义
    tried: list[dict[str, Any]] = []     # 喂给军师看的「已经试过的」（含结论）
    rung = FIDELITY_LADDER.index(start_fidelity)   # 当前数据档位
    no_finding_streak = 0
    stale = 0                            # 连续多少轮没有超过 epsilon 的提升

    best_score = total_score(cur)
    best = {"round": 0, "report": cur, "fidelity": FIDELITY_LADDER[rung]}
    summary = SessionSummary(baseline=dict(baseline or {}),
                             total_train_seconds=initial_train_seconds)

    def escalate(round_id: int, reason: str) -> bool:
        """升一档数据，并在新档位上原样重测一次，拿到可比的新基准。

        重测不调用任何角色 —— 只是换个数据量把当前流水线再跑一遍。
        跨档的分数不可比，所以升档后收敛计数与最佳分全部重置。

        已经在最高档上时返回 False —— 没地方可升了，交给调用方判停。
        """
        nonlocal rung, cur, parent, best_score, best, stale, no_finding_streak
        if rung >= len(FIDELITY_LADDER) - 1:
            return False
        rung += 1
        fidelity = FIDELITY_LADDER[rung]
        print(f"  ↑ 升到{fidelity}数据（{reason}），原样重测一次")
        try:
            result = executor.run({"new_files": [], "config_patch": ""}, fidelity)
        except Exception as exc:                 # noqa: BLE001
            print(f"  ⚠️ 重测失败：{exc}，档位已升，下一轮直接用新档位")
            result = None
        if result is not None and result.ok:
            cur = parent = result.health_report
            summary.total_train_seconds += result.seconds
        elif result is not None:
            # 重测没跑起来：档位照升，但沿用旧成绩单，并记一笔恢复事件
            print(f"  ⚠️ {fidelity}上的重测失败：{result.error}，沿用上一档的成绩单")
            emit("recovery", text=f"升档重测失败：{result.error}")
            summary.recoveries += 1
            summary.total_train_seconds += result.seconds
        best_score = total_score(cur)
        # 升档后跨档分数不可比：当前流水线就是新档位上的最佳，轮次记为刚跑完那轮
        best = {"round": round_id, "report": cur, "fidelity": fidelity}
        stale = 0
        no_finding_streak = 0
        return True

    for rid in range(1, rounds + 1):
        used = llm.ledger.total_tokens
        if used >= token_budget:
            summary.stopped_because = "预算耗尽"
            break

        prior_ledger.apply_to(cards)     # 实验积累的靠谱度盖到卡片上
        emit("round_start", round=rid, fidelity=FIDELITY_LADDER[rung])

        log = run_round(
            round_id=rid,
            llm=llm, vocab=vocab, cards=cards,
            health_report=_with_bands(cur, noise_bands), parent_result=parent,
            executor=executor,
            scheduler=CostAwareScheduler(tried_cards=blacklist | applied, time_ledger=time_ledger),
            module_interface=module_interface,
            example_module=example_module,
            # 每轮都用真正生效的配置，不是磁盘上那份初始文本
            current_config=effective_config(executor, current_config),
            history_brief=history[-5:],
            budget_left=_budget_tier(used, token_budget),
            time_ledger=time_ledger,
            prior_ledger=prior_ledger,
            tried_before=tried,
            exclude_ids=blacklist | applied,
            shelf=shelf,
            fidelity_override=FIDELITY_LADDER[rung] if rung else None,
            noise_floor=noise_floor,
        )

        # 这一轮有人插手了吗 —— 跑之前就有的那些属于准备工作，不算。
        # 必须在 log.dump 之前算，否则日志里那条永远是 0
        fresh = interventions.drain()
        log.interventions = len(fresh)
        log.intervention_notes = [e["干了什么"] for e in fresh]

        # 快照：这一轮跑完之后流水线长什么样，交付物 #4 靠它还原
        for path_ in log.patch_files:
            module_owner[path_] = rid
        snapshot_round(logs_dir, log, effective_config(executor, current_config), module_owner)

        # ── 落盘：日志、账本、待议架。每轮都写，中途断电也不丢 ──
        log.dump(logs_dir / "rounds.jsonl")
        time_ledger.dump(logs_dir / "time_ledger.json")
        prior_ledger.dump(logs_dir / "prior_ledger.json")
        shelf.dump(logs_dir / "shelf.json")

        summary.rounds_run = rid
        summary.total_tokens = llm.ledger.total_tokens
        summary.total_train_seconds += log.train_seconds
        summary.interventions += log.interventions
        summary.recoveries += len(log.recoveries)
        history.append(_brief(log))
        ref = log.reflection or {}
        emit("round_end", round=rid, verdict=ref.get("verdict", "作废"),
             seconds=log.seconds)
        if on_round is not None:
            on_round(log, summary)

        # ── 记忆：试过什么、哪些别再试 ──
        card_id = (log.chosen or {}).get("card_id")
        if card_id:
            verdict = ref.get("verdict", "本轮作废")
            tried.append({"card_id": card_id, "结论": verdict})
            if verdict in ("猜错了", "没跑起来"):
                blacklist.add(card_id)      # 失败的招，别再提
            elif verdict == "猜对了":
                applied.add(card_id)        # 已经并进流水线了，再上一次是空转
            # 「说不清」两边都不进：换个数据量或换个条件还值得再试一次

        # ── 逃生舱：连着查不出病，说明这个数据量已经看不出差别了 ──
        if log.diagnosis and log.diagnosis.get("no_finding"):
            no_finding_streak += 1
        else:
            no_finding_streak = 0
        if no_finding_streak >= NO_FINDING_ESCAPE:
            if escalate(rid, f"连续 {no_finding_streak} 轮查不出病"):
                continue
            # 已经在最大数据上了还查不出问题 —— 这就是收敛的定义，别再空转烧医生
            summary.stopped_because = (
                f"在{FIDELITY_LADDER[rung]}数据上连续 {no_finding_streak} 轮查不出问题，视为收敛")
            break

        if not log.run_ok:
            continue                        # 这一轮没跑出结果，状态不动

        # ── 接线：本轮成绩单成为下一轮的输入 ──
        parent, cur = cur, log.metrics or cur

        score = total_score(cur)
        if score > best_score + epsilon:
            best_score = score
            best = {"round": rid, "report": cur, "fidelity": log.fidelity}
            stale = 0
        else:
            stale += 1

        if ref.get("promote") and escalate(
                rid, f"第 {rid} 轮复盘官判「猜对了」，值得在更大数据上复查"):
            continue

        if stale >= patience:
            summary.stopped_because = f"连续 {stale} 轮提升小于 {epsilon}，视为收敛"
            break
    else:
        summary.stopped_because = summary.stopped_because or f"跑满 {rounds} 轮"

    summary.best_round = best["round"]
    summary.best_fidelity = best["fidelity"]
    summary.best_scores = read_scores(best["report"])

    # ── 锁定集裁决（R3）：整场唯一一次，就在这里 ──
    # 开发集被反复看了几十轮，挑出来的改动里混着「恰好迎合它」的部分。
    # 这份从没被任何决策看过的数据，是唯一能说清泛化落差有多大的裁判。
    judge = getattr(executor, "final_judge", None)
    if callable(judge):
        verdict = judge(summary.best_fidelity or "全量")
        if verdict.ok:
            summary.holdout_scores = read_scores(verdict.health_report)
            summary.total_train_seconds += verdict.seconds
            (logs_dir / "holdout_report.json").write_text(
                json.dumps(verdict.health_report, ensure_ascii=False, indent=1),
                encoding="utf-8")
        elif verdict.error and "没有配锁定集" not in verdict.error:
            summary.recoveries += 1

    summary.dump(logs_dir / "session_summary.json")
    write_narrative(logs_dir, history, summary)
    (logs_dir / "best_report.json").write_text(
        json.dumps(best["report"], ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return summary

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
from . import noise, roles, schemas

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
    # 「执行器兑现不了」和「代码写错了/训练崩了」要分开记账：
    # 前者是我们的流水线缺能力，不该扣这张卡的信任分（见 PriorLedger）。
    unsupported: bool = False


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

    def __init__(self, *paths: pathlib.Path):
        """可以盯**好几个**文件。

        为什么必须盯多个：`agent.cli intervene` 写的是仓库根的
        `logs/interventions.jsonl`，而每一场跑的 `logs_dir` 是各自独立的
        （bridge 那条路是 `kuairand_goat_bridge/output/<场次>/logs/`，
        离线演习是 `logs/offline/`）。只盯自己那一份的话，人在跑的过程中
        敲多少次 intervene 都读不到 —— 结果表上永远印「人工干预 0 次」，
        而那正是这个类存在的全部意义要避免的东西。
        """
        seen: set[pathlib.Path] = set()
        self.paths: list[pathlib.Path] = []
        for path in paths:
            resolved = pathlib.Path(path).expanduser().resolve()
            if resolved not in seen:            # 两个路径指向同一份就别数两遍
                seen.add(resolved)
                self.paths.append(pathlib.Path(path))
        self.path = self.paths[0]               # 兼容：老代码读 .path
        self._seen = len(self._read())

    def _read(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for path in self.paths:
            if not path.exists():
                continue
            out += [json.loads(line) for line in
                    path.read_text(encoding="utf-8").splitlines() if line.strip()]
        return out

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


def beats_noise(gains: dict[str, float],
                floors: dict[str, float] | float | None) -> bool:
    """这次的变化里，有没有哪一项真的越过了**它自己**的噪声带。

    为什么必须分指标：点击分和购买分的抖动差一个数量级（实测验证集里
    点击正样本 8,950 个、转化正样本只有 38 个）。用一个标量门槛管两个指标，
    两头都会错 —— 真实的点击提升被购买分的抖动淹掉判成「说不清」，
    购买分自己抖一下又越过门槛被记成「猜对了」，白送 +0.15 信任分。

    floors 给 None 或标量时退回旧行为（R11 的 0.0005 兜底）。
    """
    if not gains:
        return False
    if not isinstance(floors, dict):
        floor = max(roles.MIN_REAL_GAIN, float(floors or 0.0))
        return max((abs(v) for v in gains.values()), default=0.0) >= floor
    return any(
        abs(v) >= max(roles.MIN_REAL_GAIN, float(floors.get(k, 0.0) or 0.0))
        for k, v in gains.items()
    )


# 指标叫什么由 schemas.METRIC_PAIRS 一处说了算 —— 复盘官的 actual、
# 军师的 expected、结果表、噪声带门槛全按同一套名字对齐。
# 两处各维护一张表，迟早会走岔，而走岔时不报错。
_METRIC_NAMING = schemas.METRIC_PAIRS


def read_scores(report: dict[str, Any]) -> dict[str, float]:
    """从成绩单里取出两个分指标，键名用这套任务自己的叫法。

    挑最佳版本、算 delta、写结果表全靠它，所以两种成绩单格式都要认。
    取不到返回空 dict。
    """
    for section in _SCORE_SECTIONS:
        block = report.get(section)
        if not isinstance(block, dict):
            continue
        for (first, second), (out_first, out_second) in _METRIC_NAMING:
            if block.get(first) is not None:
                return {
                    out_first: float(block[first]),
                    out_second: float(block.get(second) or 0.0),
                }
    return {}


def total_score(report: dict[str, Any]) -> float:
    """返回任务正式主分。

    KuaiRand 成绩单明确提供 ``主分``，必须直接使用它，因为官方
    epsilon=0.002 是针对 primary，而不是 GAUC+nDCG@5。

    旧 AliCCP 成绩单没有 ``主分`` 时，继续退回原来的双指标求和。
    """
    for section in _SCORE_SECTIONS:
        block = report.get(section)
        if isinstance(block, dict) and block.get("主分") is not None:
            return float(block["主分"])

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
    run_id: str = ""       # 哪一场跑的。日志是追加的，轮次每场都从 1 重数，没这个分不清
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


def _crashed_reflection(chosen: dict[str, Any] | None, error: str,
                        metrics: list[str] | None = None) -> dict[str, Any]:
    """执行失败时的复盘结论 —— 纯代码合成，不花一分钱去问大模型。

    跑都没跑起来，没有任何结果可复盘。这时候调大模型是纯浪费。
    """
    chosen = chosen or {}
    targets = chosen.get("targets") or []
    return {
        "verdict": "没跑起来",
        # 指标名跟着当前任务走 —— 写死 AliCCP 那两个名字的话，
        # 这条合成结论跟真实复盘的结论字段对不上，结果表拼不起来
        "actual": {m: 0.0 for m in (metrics or schemas.METRICS)},
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
        # ⚠️ 这里以前是写死的一句「换一个方案，或者先修这个报错」。
        # 它非空，于是 _brief 的 `备注` 永远取到它，`or` 后面的 recoveries
        # 永远轮不到 —— 执行器说的「你要的那些列读不进内存」在这一步就被扔了。
        # 下一轮只知道"没跑起来"，不知道为什么，只能换个说法再撞一次同一堵墙。
        "next_hint": f"上一版没跑起来。执行器的原话：{error[:300]}",
        "promote": False,
        "由代码合成": True,          # 标记：不是复盘官说的，是代码填的
    }


def run_round(
    *,
    round_id: int,
    run_id: str = "",
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
    # 分指标噪声带 {指标名: 门槛}。给了就按指标各判各的；
    # 不给就退回上面那个标量（旧行为）。
    noise_bands_by_metric: dict[str, float] | None = None,
) -> RoundLog:
    """跑完整的一轮：诊断 → 筛卡 → 提案 → 调度 → 实现 → 执行 → 复盘。

    任何一个角色炸掉，本轮作废并返回，外层循环继续下一轮 —— 绝不把异常抛出去。
    """

    t0 = time.time()
    log = RoundLog(round_id=round_id, run_id=run_id,
                   started_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
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
        history_brief=history_brief,
        metrics=schemas.metric_names(health_report),
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
        log.reflection = _crashed_reflection(
            log.chosen, result.error, schemas.metric_names(health_report))
        # 执行器兑现不了 ≠ 这张卡不靠谱 —— 前者是我们的流水线缺能力。
        # 照扣的话，一场跑下来会把 ESMM、DeepFM 这些好方法全扣成低信任分，
        # 下一场军师就再也不提它们了，一个自己造成的错误结论被固化进账本。
        # （仍然进黑名单：这一场里它确实跑不了，再提就是空转烧钱。）
        if prior_ledger is not None and card is not None and not result.unsupported:
            prior_ledger.apply(card.id, "没跑起来", card.prior)
        return finish()

    # ④ 复盘官
    log.reflection = _guard(
        log, "复盘官", roles.reflect,
        llm, vocab, log.chosen, result.health_report, parent_result, card,
        noise_floor=noise_floor,
        noise_bands_by_metric=noise_bands_by_metric,
    )
    if log.reflection is not None and prior_ledger is not None and card is not None:
        prior_ledger.apply(
            card.id, log.reflection["verdict"], card.prior,
            # 多个目标里只要有一个真的好转，这张卡就该加分
            symptom_improved=any(item["resolved"] in ("是", "部分")
                                 for item in log.reflection["symptom_resolved"]),
            # 分指标判定：点击和购买的抖动差一个数量级，一个标量门槛管两个，
            # 真实的点击提升会被购买分的抖动淹掉，购买分自己抖一下又白拿 +0.15
            beat_noise=beats_noise(log.reflection["actual"],
                                   noise_bands_by_metric or noise_floor),
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
    path = logs_dir / "snapshots" / (log.run_id or "unknown") / f"round_{log.round_id:03d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "轮次": log.round_id,
        "保真度": log.fidelity,
        "分数": read_scores(log.metrics or {}),
        "配置": config_text,
        "零件": dict(module_owner),      # 路径 → 哪一轮写的这一版
    }, ensure_ascii=False, indent=1), encoding="utf-8")


def read_snapshot(logs_dir: pathlib.Path, run_id: str, round_id: int
                  ) -> tuple[dict[str, Any], dict[str, str]] | None:
    """读出某一轮的流水线：(配置字典, {零件路径: 完整代码})。读不出返回 None。

    零件内容不在快照里 —— 快照只记「这个文件是哪一轮写的」，
    内容从 rounds.jsonl 的 patch_files 取，避免同一份代码存两遍。
    """
    snap_path = logs_dir / "snapshots" / (run_id or "unknown") / f"round_{round_id:03d}.json"
    rounds_path = logs_dir / "rounds.jsonl"
    if not snap_path.exists() or not rounds_path.exists():
        return None
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    config = yaml.safe_load(snap.get("配置") or "") or {}
    if not isinstance(config, dict):
        return None

    rows = [json.loads(l) for l in
            rounds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    # 只认同一场 —— 轮次编号每场都从 1 重数，跨场取会拿到别人那一轮的代码
    by_round = {r["round_id"]: r for r in rows if (r.get("run_id") or "") == run_id}
    files: dict[str, str] = {}
    for rel, owner in (snap.get("零件") or {}).items():
        content = (by_round.get(owner, {}).get("patch_files") or {}).get(rel)
        if content is None:
            return None                     # 缺件就别装了，半个流水线比没有更糟
        files[rel] = content
    return config, files


def snapshot_modules(logs_dir: pathlib.Path, run_id: str, round_id: int
                     ) -> dict[str, int]:
    """某一轮的快照里记着「哪个零件是哪一轮写的」。回滚时账本要跟着拨回去，
    否则后面几轮的快照会继续claim 已经被退掉的那些文件。"""
    p = logs_dir / "snapshots" / (run_id or "unknown") / f"round_{round_id:03d}.json"
    if not p.exists():
        return {}
    try:
        return dict(json.loads(p.read_text(encoding="utf-8")).get("零件") or {})
    except Exception:                        # noqa: BLE001
        return {}


def install_pipeline(logs_dir: pathlib.Path, run_id: str, round_id: int,
                     executor: Any) -> bool:
    """把某一轮的流水线装回执行器，供最终裁决使用。

    为什么必须装：工兵的改动是**叠加**的，跑到最后磁盘上只剩末态。
    而收敛条件是「连续 patience 轮没进步」—— 最佳轮之后必然还跑了至少
    patience 轮，**末态永远不等于最佳轮**，不是偶发情况。
    拿末态去考锁定集，考的是另一个模型，而锁定集只许读一次，机会就烧掉了。
    """
    if not hasattr(executor, "config"):
        return False
    got = read_snapshot(logs_dir, run_id, round_id)
    if got is None:
        return False
    config, files = got
    for rel, content in files.items():
        target = ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    executor.config = config
    return True


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
        f"> 第 `{summary.run_id}` 场 · 跑了 {summary.rounds_run} 轮 · {summary.stopped_because}",
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
        # 分指标：点击和购买的抖动差一个数量级（实测验证集里点击正样本
        # 8,950 个、转化正样本只有 38 个），合成一个数会被购买带主导或压低，
        # 两头都错。这是医生判断"这个分组差距算不算病"该看的数字。
        "分指标": bands.get("分指标噪声带"),
        "单指标": bands.get("单指标噪声带"),   # 旧字段，判分组差距请用上面的「分指标」
        # 逐桶取「这个桶该用的那把尺子」：缩放后的有效带 → 实测带 → 理论带。
        # 直接读实测带是不行的 —— 保真度抽样只抽负样本，click=1 子集每个种子
        # 完全相同，扰动不到购买塔，**每个桶实测出来都是 0.0000**。
        # 门槛是 0 意味着任何分桶差距都算病，而「冷门商品学不动」「新用户不会做」
        # 正是拿分桶差距判的。详见 agent/noise.py: bucket_band
        "分组": {g: {k: noise.bucket_band(v) for k, v in rows.items()}
                for g, rows in bands.get("分组", {}).items()},
        "怎么用": ("这是同配置换随机种子跑出来的抖动幅度。"
                 "任何小于它的差距都是噪声，不许当成病；"
                 "点击和购买要分别对自己的噪声带比，别用同一个数字判两个指标；"
                 "分桶差距要跟对应那个桶的噪声带比，别用统一阈值。"),
    }}


def _brief(log: RoundLog) -> dict[str, Any]:
    """把一轮压成一两行，喂给下一轮的医生和军师。全文喂过去会烧光预算。"""
    ref = log.reflection or {}
    brief: dict[str, Any] = {
        "轮次": log.round_id,
        "选了": (log.chosen or {}).get("card_id") or "（自创或未选中）",
        "数据": log.fidelity,
        "结论": ref.get("verdict", "本轮作废"),
        "实际变化": ref.get("actual", {}),
        "备注": ref.get("next_hint", "") or "；".join(log.recoveries[:2]),
    }
    # 出错原文单独占一格，不跟 `备注` 抢位置 —— 这是下一轮唯一能知道
    # 「为什么没跑起来」的渠道。复盘官成功、但中途有恢复事件的轮次
    # （比如工兵换了备胎）以前也传不出去，一并补上。
    if log.recoveries:
        brief["出了什么错"] = "；".join(log.recoveries)[:400]
    return brief


@dataclass
class SessionSummary:
    """一整场跑完的汇总 —— 直接对应交付物 #5 的结果表。"""

    run_id: str = ""
    rounds_run: int = 0
    stopped_because: str = ""
    best_round: int = 0
    best_fidelity: str = ""
    best_scores: dict[str, float] = field(default_factory=dict)
    # 锁定集上的分数（R3：整场只评一次）。空 = 没配锁定集，或裁决失败。
    holdout_scores: dict[str, float] = field(default_factory=dict)
    holdout_note: str = ""      # 锁定集大考的说明（跳过时写明为什么）
    # 这一场用的噪声带是怎么来的：量在哪个档位、有没有缩放过、有没有过期。
    # 必须进结果表 —— 「这次算不算真提升」全靠这把尺子，尺子刻度不对不会报错，
    # 只会让结论慢慢错，读交付材料的人有权知道它当时对不对得上。
    noise_note: str = ""
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
        if self.holdout_note:
            lines.append(f"锁定集          {self.holdout_note}")
        if self.holdout_scores:
            lines.append("锁定集裁决      （整场只读一次，从未参与任何决策）")
            for metric, value in self.holdout_scores.items():
                gap = self.generalization_gap.get(metric)
                tail = f"   泛化落差 {gap:+.4f}" if gap is not None else ""
                lines.append(f"  {metric:<12} {value:.4f}{tail}")
        else:
            lines.append("锁定集裁决      未做（没配锁定集）—— 开发集分数可能偏乐观")
        if self.noise_note:
            lines.append(f"噪声带          {self.noise_note}")
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
    run_id: str | None = None,                  # 这一场的编号，不给就用时间戳
    start_fidelity: str = FIDELITY_LADDER[0],   # 从哪一档数据起步
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    epsilon: float = DEFAULT_EPSILON,
    patience: int = DEFAULT_PATIENCE,
    # ── 爬山 ──
    # 走差了要不要退回当前最好的那一版。关掉 = 老行为（只累加、从不回头）：
    # 一次坏改动会永久留在流水线上，之后每一轮都建立在更差的基础上，
    # 而且医生看到的是退化后的指标，可能去治我们自己造成的病。
    rollback: bool = True,
    # 分数比历史最佳低多少才算「走差了」。
    #   None  → 用实测噪声带（比它小的下降分不清是真变差还是抖动）
    #   0     → 严格爬山：只要没超过历史最佳就退
    # 具体调多少是实验口径问题，留给跑实验的人定。
    rollback_margin: float | None = None,
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
    # 两处都盯：这一场自己的 logs_dir，以及 `agent.cli intervene` 默认写的
    # 仓库根 logs/。人在跑的过程中敲那条命令时，不该还要先想清楚
    # 「这一场的日志目录在哪」—— 那一刻他正忙着处理刚出的问题。
    interventions = InterventionLog(
        logs_dir / "interventions.jsonl",
        ROOT / "logs" / "interventions.jsonl",
    )
    module_owner: dict[str, int] = {}      # 零件路径 → 哪一轮写的这一版

    run_id = run_id or time.strftime("%Y%m%d-%H%M%S")
    if start_fidelity not in FIDELITY_LADDER:
        raise ValueError(f"没有「{start_fidelity}」这一档，只能是 {FIDELITY_LADDER}")

    cur = initial_report          # 当前这一版的成绩单。复盘官要的「改动之前那一版」就是它
    history: list[dict[str, Any]] = []
    blacklist: set[str] = set()          # 试过且失败的卡 —— 调度器硬性跳过
    applied: set[str] = set()            # 已经生效、并入流水线的卡 —— 再上一次没意义
    tried: list[dict[str, Any]] = []     # 喂给军师看的「已经试过的」（含结论）
    rung = FIDELITY_LADDER.index(start_fidelity)   # 当前数据档位
    no_finding_streak = 0
    stale = 0                            # 连续多少轮没有超过 epsilon 的提升

    best_score = total_score(cur)
    best = {"round": 0, "report": cur, "fidelity": FIDELITY_LADDER[rung]}
    summary = SessionSummary(run_id=run_id, baseline=dict(baseline or {}),
                             total_train_seconds=initial_train_seconds)

    # ── 开跑前先对一次尺子 ──
    #
    # 噪声带是在**某一个档位**上量出来的，只对那个档位有效。正样本一多抖动就小，
    # 拿小档位量的带子去卡大档位的结果，是一把过松的尺子：真实提升会被判成
    # 「说不清」；反过来拿窄带子卡小档位，又会把纯抖动奖励成「猜对了」，
    # 白送信任分。
    #
    # 最坑的地方是它**不会报错**，只会让结论慢慢错，然后顺着信任分、黑名单、
    # 升档决策一路传染。所以这里要么按样本量缩过去，要么把「没对上」写进
    # 结果表，别让它默默溜过去。
    if noise_bands:
        量在, 起步 = noise_bands.get("保真度") or "?", FIDELITY_LADDER[rung]
        if 量在 != 起步:
            noise_bands = noise.rescale(noise_bands, cur)
            noise_floor = float(noise_bands.get("单指标噪声带") or noise_floor)
            summary.noise_note = (f"带子量在「{量在}」档、这一场从「{起步}」起步："
                                  f"{noise_bands.get('缩放说明', '未能缩放')}")
            print(f"  ↳ {summary.noise_note}")
        else:
            summary.noise_note = f"带子量在「{量在}」档，与起步档位一致"

        # 带子的指标名要跟这一场的成绩单对得上。对不上时 reflect 里
        # `noise_bands_by_metric.get(k, 0)` 会静默退回 R11 兜底门槛 ——
        # 结果不会错，但"我们量过带子了"这个印象是错的：分指标门槛
        # 一条都没生效。这种事必须写进结果表，不能只在代码里默默降级。
        本场指标 = set(schemas.metric_names(cur))
        带子指标 = set((noise_bands.get("分指标噪声带") or {}))
        if 带子指标 and not (本场指标 & 带子指标):
            summary.noise_note += (
                f"；⚠️ 带子量的是 {sorted(带子指标)}，这一场的指标是 "
                f"{sorted(本场指标)} —— 对不上，分指标门槛全部失效，"
                f"实际用的是 R11 兜底 {max(noise_floor, roles.MIN_REAL_GAIN):.4f}。"
                f"`agent.cli noise` 还没在这个任务上跑过")
            noise_bands = {**noise_bands, "分指标噪声带": {}}
            print(f"  ⚠️ {summary.noise_note}")
    else:
        summary.noise_note = (
            f"没测过噪声带，全程用 R11 的兜底门槛 {max(noise_floor, roles.MIN_REAL_GAIN):.4f}"
            f"（一个拍出来的数，不是这份数据上量的）"
            f" —— 正式跑之前应该先 `agent.cli noise --seeds 3 "
            f"--fidelity {FIDELITY_LADDER[rung]}`，否则「算不算真提升」没有依据")
        print(f"⚠️ {summary.noise_note}\n")

    def escalate(round_id: int, reason: str) -> bool:
        """升一档数据，并在新档位上原样重测一次，拿到可比的新基准。

        重测不调用任何角色 —— 只是换个数据量把当前流水线再跑一遍。
        跨档的分数不可比，所以升档后收敛计数与最佳分全部重置。

        已经在最高档上时返回 False —— 没地方可升了，交给调用方判停。
        """
        nonlocal rung, cur, best_score, best, stale, no_finding_streak
        nonlocal noise_bands, noise_floor
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
            cur = result.health_report
            summary.total_train_seconds += result.seconds
            # 换档位之后噪声带必须跟着变 —— 正样本一多，抖动就小。
            # 沿用起步档位的带子是一把过松的尺子：真实提升会被当噪声抹掉。
            # 不重测（那要再烧 N 次训练），按样本量解析缩放。
            if noise_bands:
                noise_bands = noise.rescale(noise_bands, cur)
                # 标量门槛也得跟着走：分指标缺某个指标时，复盘官退回的正是它
                noise_floor = float(noise_bands.get("单指标噪声带") or noise_floor)
                summary.noise_note = f"升到{fidelity}时{noise_bands.get('缩放说明', '按样本量缩放')}"
                print(f"  ↳ 噪声带{noise_bands.get('缩放说明', '')}")
        else:
            if result is not None:
                # 重测没跑起来：档位照升，但沿用旧成绩单，并记一笔恢复事件
                print(f"  ⚠️ {fidelity}上的重测失败：{result.error}，沿用上一档的成绩单")
                emit("recovery", text=f"升档重测失败：{result.error}")
                summary.recoveries += 1
                summary.total_train_seconds += result.seconds
            if noise_bands:
                # 拿不到新档位的样本量就缩不过去。之后几轮是在用上一档的尺子量
                # 新档位的结果 —— 门槛偏松，真提升会被当噪声抹掉。
                # 这件事不会抛异常，所以必须自己喊出来并写进交付材料。
                summary.noise_note = (
                    f"⚠️ 升到{fidelity}时重测没跑起来，噪声带仍停在"
                    f"「{noise_bands.get('保真度') or '起步档位'}」档，"
                    f"这之后的「算不算真提升」判定偏松")
                emit("recovery", text=summary.noise_note)
        best_score = total_score(cur)
        # 升档后跨档分数不可比：当前流水线就是新档位上的最佳，轮次记为刚跑完那轮
        best = {"round": round_id, "report": cur, "fidelity": fidelity}
        stale = 0
        no_finding_streak = 0
        return True

    # 第 0 轮也留一份快照：best 一开始指向第 0 轮，第 1 轮就走差的话
    # 得退得回去，而循环里的 snapshot_round 只覆盖第 1 轮起。
    try:
        snapshot_round(logs_dir,
                       RoundLog(round_id=0, run_id=run_id, fidelity=FIDELITY_LADDER[rung],
                                started_at=time.strftime("%Y-%m-%dT%H:%M:%S"), metrics=cur),
                       effective_config(executor, current_config), {})
    except Exception as exc:                 # noqa: BLE001
        emit("recovery", text=f"第 0 轮快照写失败：{exc}")

    for rid in range(1, rounds + 1):
        used = llm.ledger.total_tokens
        if used >= token_budget:
            summary.stopped_because = "预算耗尽"
            break

        prior_ledger.apply_to(cards)     # 实验积累的靠谱度盖到卡片上
        emit("round_start", round=rid, fidelity=FIDELITY_LADDER[rung])

        log = run_round(
            round_id=rid, run_id=run_id,
            llm=llm, vocab=vocab, cards=cards,
            # 两个都传 cur：医生看当前状态做诊断，复盘官拿它当「改动之前那一版」。
            # 这里曾经传过一个落后两轮的 parent —— 第 3 轮会拿第 1 轮来比，
            # 中间两轮的进步全算在这一次头上，虚报的收益还会传染到
            # 卡片信任分、黑名单和升档决策。
            health_report=_with_bands(cur, noise_bands), parent_result=cur,
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
            # 分指标门槛：点击和购买的抖动差一个数量级，用一个标量管两个，
            # 真实的点击提升会被购买分的抖动淹掉，购买分自己抖一下又白拿 +0.15
            noise_bands_by_metric=(noise_bands or {}).get("分指标噪声带"),
        )

        # 这一轮有人插手了吗 —— 跑之前就有的那些属于准备工作，不算。
        # 必须在 log.dump 之前算，否则日志里那条永远是 0
        fresh = interventions.drain()
        log.interventions = len(fresh)
        log.intervention_notes = [e["干了什么"] for e in fresh]

        # ── 落盘：快照、日志、账本、待议架。每轮都写，中途断电也不丢 ──
        #
        # 整段包保险丝：四个角色都有保护，记账反而没有，那是最讽刺的死法 ——
        # 这段跑在**昂贵的训练之后**，磁盘满、某个字段序列化不了、
        # yaml dump 打个嗝，整场就挂在这里，连刚跑完那一轮的成果一起丢。
        # 写失败只是少一份记录，不该让整场停下来。
        for path_ in log.patch_files:
            module_owner[path_] = rid
        for 名字, 动作 in (
            ("快照", lambda: snapshot_round(
                logs_dir, log, effective_config(executor, current_config), module_owner)),
            ("逐轮日志", lambda: log.dump(logs_dir / "rounds.jsonl")),
            ("耗时账本", lambda: time_ledger.dump(logs_dir / "time_ledger.json")),
            ("靠谱度账本", lambda: prior_ledger.dump(logs_dir / "prior_ledger.json")),
            ("待议架", lambda: shelf.dump(logs_dir / "shelf.json")),
        ):
            try:
                动作()
            except Exception as exc:                 # noqa: BLE001
                log.recoveries.append(f"{名字}写失败：{type(exc).__name__}: {exc}")
                emit("recovery", text=f"{名字}写失败：{exc}")

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
        选中 = log.chosen or {}
        card_id = 选中.get("card_id")
        verdict = ref.get("verdict", "本轮作废")
        if card_id:
            tried.append({"card_id": card_id, "结论": verdict})
            if verdict in ("猜错了", "没跑起来"):
                blacklist.add(card_id)      # 失败的招，别再提
            elif verdict == "猜对了":
                applied.add(card_id)        # 已经并进流水线了，再上一次是空转
            # 「说不清」两边都不进：换个数据量或换个条件还值得再试一次
        elif 选中:
            # 自创方案没有 card_id，以前 `if card_id:` 整段跳过 —— 于是它
            # 不进「已经试过的」，军师下一轮完全不知道自己提过、更不知道
            # 被拒了，只能换个说法再提一遍。
            # 只记进 tried（给军师看），不进 blacklist：黑名单是按 card_id
            # 硬过滤的，而两个自创方案本来就不是同一个东西，硬拦会误伤。
            tried.append({
                "card_id": "（自创）",
                "治": 选中.get("targets") or [],
                "做法": (选中.get("how_to") or "")[:100],
                "结论": verdict,
            })
        tried[:] = tried[-12:]              # 只留最近的，别让上下文越滚越大

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
        cur = log.metrics or cur

        score = total_score(cur)
        if score > best_score + epsilon:
            best_score = score
            best = {"round": rid, "report": cur, "fidelity": log.fidelity}
            stale = 0
        else:
            stale += 1

        # ── 爬山：这一步走差了就退回当前最好的那一版 ──
        #
        # 不退的后果不是"少涨一点"，是**整条搜索轨迹一去不回头**：
        # 工兵的改动是深度合并进配置的，判「猜错了」只会把卡片拉黑、
        # 扣信任分，**改动本身留在流水线上**。于是后面每一轮都建立在更差的
        # 基础上；`cur` 跟着退化，医生看到的是退化后的指标，可能去诊断
        # 我们自己造成的病，复盘官的「改动之前那一版」也一路往下漂。
        #
        # 门槛用噪声带而不是 0：比噪声还小的下降分不清是真变差还是抖动，
        # 为它回滚只会来回颠簸。要严格爬山就把 rollback_margin 设成 0。
        门槛 = (max(0.0, float(rollback_margin)) if rollback_margin is not None
              else max(noise_floor, roles.MIN_REAL_GAIN))
        if rollback and best["round"] != rid and score < best_score - 门槛:
            退回 = best["round"]
            if install_pipeline(logs_dir, run_id, 退回, executor):
                cur = best["report"]                 # 比较基准也要跟着回去
                module_owner = snapshot_modules(logs_dir, run_id, 退回)
                说明 = (f"第 {rid} 轮分数 {score:.4f} 比最好的第 {退回} 轮 "
                       f"{best_score:.4f} 低了超过 {门槛:.4f}，已退回第 {退回} 轮的流水线")
                print(f"  ↩ {说明}")
                emit("recovery", text=说明)
                summary.recoveries += 1
                if history:
                    history[-1]["备注"] = (history[-1].get("备注") or "") + f"；{说明}"
            else:
                # 装不回去就别硬来：宁可带着这个坏改动往下跑，
                # 也不要让流水线停在一个说不清是哪一版的状态。
                emit("recovery", text=f"想退回第 {退回} 轮但快照装不回来，只能带着这一版继续")

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
    if callable(judge) and not install_pipeline(logs_dir, run_id, summary.best_round, executor):
        # 装不回最佳轮就别考 —— 锁定集只许读一次，宁可没有这个数，也不要一个错的数
        summary.holdout_note = (
            f"跳过锁定集大考：没能把第 {summary.best_round} 轮的流水线装回来。"
            f"末态跟最佳轮不是同一个模型，拿末态去考等于白烧唯一一次机会。")
        judge = None
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

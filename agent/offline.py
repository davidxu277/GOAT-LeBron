"""离线自检 —— 不花一分钱，把整条链路从头跑到尾。

真跑一场要 API key、要数据、要一个小时。可是"外层循环会不会在第 7 轮崩掉"
这种问题，等到挂机那天晚上才发现就太晚了。

所以这里有两个替身：

    ScriptedLLM      —— 照着 schema 生成合法回答的假模型，不联网
    DriftingExecutor —— 分数一轮轮慢慢涨的假执行器，可以按需在某轮失败

它们只用来验证**接线对不对**（状态有没有传下去、失败能不能恢复、
账本有没有在变），不用来验证提示词好不好 —— 那个只能拿真模型跑。

假执行器出的是 KuaiRand 成绩单（跟 goat_executor._health_report + diagnostics
同样的块名），假模型的指标名从 schema 里读 —— 换任务时这两个替身不用跟着改。

    python -m agent.cli run --offline --rounds 6 --fail-round 3
"""

from __future__ import annotations

import copy
from typing import Any

from .llm import Ledger, SchemaViolation
from .loop import RunResult


class ScriptedLLM:
    """假模型：从 schema 里读出合法取值，拼一个必定通过校验的回答。

    关键做法是**从 schema 反推答案**（病名 enum、卡片 id enum、指标名都在 schema 里），
    所以词表、卡片库、任务的指标怎么变，这个替身都不用跟着改。
    """

    def __init__(self, faults: dict[str, list[int]] | None = None,
                 promote_on: tuple[int, ...] = (2,)):
        self.ledger = Ledger()
        self.faults = faults or {}          # 角色 → 第几次调用要炸
        self.promote_on = set(promote_on)   # 第几次复盘建议升档
        self.calls: dict[str, int] = {}
        self._last_findings: list[dict[str, Any]] = []
        self._round = 0

    # ── 内部 ──

    def _count(self, role: str) -> int:
        self.calls[role] = self.calls.get(role, 0) + 1
        return self.calls[role]

    @staticmethod
    def _node(schema: dict, *path: str) -> dict:
        node = schema
        for key in path:
            node = node[key]
        return node

    @classmethod
    def _enum(cls, schema: dict, *path: str) -> list[str]:
        return cls._node(schema, *path)["enum"]

    # ── 对外：跟 LLM.call 同签名 ──

    def call(self, *, role: str, system: str, user: str, schema: dict,
             big: bool = True, effort: str = "high", max_tokens: int = 16000,
             validate=None, **_: Any) -> dict[str, Any]:
        n = self._count(role)
        model = "假模型-大" if big else "假模型-小"
        self.ledger.add(role, model, inp=3000, out=600)

        if n in self.faults.get(role, []):
            raise SchemaViolation(f"（演习）{role} 第 {n} 次调用被人为弄失败")

        data = getattr(self, f"_{role}")(schema)
        if validate is not None:
            validate(data)                  # 假答案也要过真校验，否则这个替身没意义
        return data

    # ── 四个角色的假答案 ──

    def _医生(self, schema: dict) -> dict[str, Any]:
        finding = ("properties", "findings", "items", "properties")
        symptoms = self._enum(schema, *finding, "symptom")
        metrics = self._enum(schema, *finding, "affects", "items")
        # 轮流报不同的病，好让筛卡和拉黑逻辑真的被走到
        picked = symptoms[self._round % len(symptoms)]
        self._round += 1
        self._last_findings = [{
            "symptom": picked,
            "severity": 0.7,
            "confidence": "高",
            "evidence": (f"按视频曝光次数分组：最低桶 GAUC 0.612，最高桶 0.681，"
                         f"差 0.069（第 {self._round} 轮）"),
            "affects": list(metrics),
        }]
        return {"findings": copy.deepcopy(self._last_findings),
                "no_finding": False, "reason_if_none": ""}

    def _军师(self, schema: dict) -> dict[str, Any]:
        proposal = ("properties", "proposals", "items", "properties")
        card_ids = [c for c in self._enum(schema, *proposal, "card_id") if c]
        metrics = list(self._node(schema, *proposal, "expected", "properties"))
        targets = [f["symptom"] for f in self._last_findings] or ["冷门视频排不上去"]

        def _one(rank: int, card_id: str, gain: float, 难度: str) -> dict[str, Any]:
            return {
                "rank": rank,
                "card_id": card_id,
                "targets": targets,
                "rationale": f"最低曝光桶 GAUC 0.612 比最高桶 0.681 低 0.069（方案 {rank}）",
                # 最后一个指标报 gain，其余报一个小正数 —— 调度器要有差别可排
                "expected": {m: (gain if i == len(metrics) - 1 else 0.001)
                             for i, m in enumerate(metrics)},
                "cost": {"代码难度": 难度, "训练时间倍数": 1.0},
                "risk": "热门视频可能被作者信息稀释",
                "novel": not card_id,
                # 用现成卡片时也要写 —— 写的是"在当前流水线上怎么落"，不是重述卡片
                "how_to": (f"在 features 下新开一块，item_field 用 video_id、"
                           f"category_field 用 author_id、K 取 20；"
                           f"新零件写 modules/features/fallback_{rank}.py，"
                           f"base_fields 不用动（新列自动进特征表）"),
            }

        if not card_ids:
            return {"proposals": [_one(1, "", 0.004, "简单")]}
        # 多提几个，调度器才有得挑，没挑中的才会进待议架
        picks = card_ids[:3]
        return {"proposals": [
            _one(i + 1, cid, 0.004 - i * 0.001, "简单" if i == 0 else "中等")
            for i, cid in enumerate(picks)
        ]}

    def _工兵(self, schema: dict) -> dict[str, Any]:
        return {
            "change_type": "加新零件",
            # 名字带 _offline_ 前缀：.gitignore 认这个前缀，
            # 演习产出的假零件就再也溜不进仓库了（真发生过一次）
            "config_patch": "features:\n  _offline_scratch:\n    enabled: false\n",
            "new_files": [{
                "path": "modules/features/_offline_scratch.py",
                "content": "# 离线演习用的空零件\nVALUE = 1\n",
            }],
            "self_check": [
                "未使用禁用字段（long_view、play_time_ms 等）",
                "统计量只用了训练集",
                "参数全部从配置读取",
            ],
        }

    def _复盘官(self, schema: dict) -> dict[str, Any]:
        # 方案声称要治哪几个病，就得逐个交代 —— 跟真复盘官一样的规矩
        symptoms = [f["symptom"] for f in self._last_findings] or ["冷门视频排不上去"]
        metrics = list(self._node(schema, "properties", "actual", "properties"))
        promote = self.calls.get("复盘官", 0) in self.promote_on
        return {
            "verdict": "猜对了",
            "actual": {m: (0.0031 if i == len(metrics) - 1 else 0.0008)
                       for i, m in enumerate(metrics)},
            "vs_expected": "比预计的 0.004 略低",
            "symptom_resolved": [
                {"symptom": s, "before": 0.069, "after": 0.051, "resolved": "部分"}
                for s in symptoms
            ],
            "card_update": {"card_id": "", "prior_delta": 0.1, "note": "小份数据上有效"},
            "next_hint": "去看新用户那一组",
            "promote": promote,
        }


class DriftingExecutor:
    """假执行器：每跑一次，分数往上挪一点点，直到挪不动为止。

    这样外层循环的收敛判定、最佳版本挑选、升档逻辑才有东西可判 ——
    FakeExecutor 每次回放同一份成绩单，第二轮就"收敛"了，测不出什么。

    起点取官方 FM 基线在验证集上的量级（GAUC 0.6674 / nDCG@5 0.5357 附近）。
    """

    def __init__(self, base_gauc: float = 0.6638, base_ndcg: float = 0.5357,
                 gain: float = 0.003, decay: float = 0.45,
                 fail_rounds: tuple[int, ...] = (), seconds: float = 12.0):
        self.gauc, self.ndcg = base_gauc, base_ndcg
        self.gain, self.decay = gain, decay
        self.fail_rounds = set(fail_rounds)
        self.seconds = seconds
        self.runs = 0

    def run(self, patch: dict[str, Any], fidelity: str) -> RunResult:
        self.runs += 1
        if self.runs in self.fail_rounds:
            return RunResult(ok=False, error="（演习）训练进程被人为杀掉",
                             seconds=1.0, fidelity=fidelity)
        self.gauc += self.gain
        self.ndcg += self.gain * 1.4
        self.gain *= self.decay              # 越往后越难涨，最终自然收敛
        return RunResult(ok=True, seconds=self.seconds, fidelity=fidelity,
                         health_report=self.report(fidelity))

    @staticmethod
    def _scores(gauc: float, ndcg: float) -> dict[str, float]:
        return {"GAUC": round(gauc, 4), "nDCG@5": round(ndcg, 4),
                "主分": round((gauc + ndcg) / 2, 4)}

    def report(self, fidelity: str = "小份") -> dict[str, Any]:
        """当前这一版的成绩单。外层循环要一份第 0 轮的做起点。"""
        g, n = self.gauc, self.ndcg
        return {
            "数据集": "KuaiRand-Pure",
            "保真度": fidelity,
            "随机种子": 20260827,
            "验证集": {**self._scores(g, n), "总行数": 124_909, "用户数": 25_800},
            "训练集": {**self._scores(g + 0.015, n + 0.015), "用户数": 5_000},
            "训练诊断": {
                "实际特征": ["user_id", "video_id", "author_id", "tab", "duration_bucket"],
                "装上的零件": [],
            },
            "按视频曝光次数分组": [
                {"分组": "曝光<10次", "行数": 27_480, "占比": 0.22, "正样本数": 7_900,
                 **self._scores(g - 0.052, n - 0.06), "用户数": 9_100},
                {"分组": "曝光>1000次", "行数": 18_740, "占比": 0.15, "正样本数": 6_300,
                 **self._scores(g + 0.017, n + 0.02), "用户数": 11_400},
            ],
        }

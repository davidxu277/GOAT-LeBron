"""离线自检 —— 不花一分钱，把整条链路从头跑到尾。

真跑一场要 API key、要数据、要一个小时。可是"外层循环会不会在第 7 轮崩掉"
这种问题，等到挂机那天晚上才发现就太晚了。

所以这里有两个替身：

    ScriptedLLM      —— 照着 schema 生成合法回答的假模型，不联网
    DriftingExecutor —— 分数一轮轮慢慢涨的假执行器，可以按需在某轮失败

它们只用来验证**接线对不对**（状态有没有传下去、失败能不能恢复、
账本有没有在变），不用来验证提示词好不好 —— 那个只能拿真模型跑。

    python -m agent.cli run --offline --rounds 6 --fail-round 3
"""

from __future__ import annotations

import copy
from typing import Any

from .llm import Ledger, SchemaViolation
from .loop import RunResult


class ScriptedLLM:
    """假模型：从 schema 里读出合法取值，拼一个必定通过校验的回答。

    关键做法是**从 schema 反推答案**（病名 enum、卡片 id enum 都在 schema 里），
    所以词表和卡片库怎么变，这个替身都不用跟着改。
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
    def _enum(schema: dict, *path: str) -> list[str]:
        node = schema
        for key in path:
            node = node[key]
        return node["enum"]

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
        symptoms = self._enum(
            schema, "properties", "findings", "items", "properties", "symptom")
        # 轮流报不同的病，好让筛卡和拉黑逻辑真的被走到
        picked = symptoms[self._round % len(symptoms)]
        self._round += 1
        self._last_findings = [{
            "symptom": picked,
            "severity": 0.7,
            "confidence": "高",
            "evidence": f"冷门桶 0.552 对热门桶 0.638，差 0.086（第 {self._round} 轮）",
            "affects": ["购买AUC"],
        }]
        return {"findings": copy.deepcopy(self._last_findings),
                "no_finding": False, "reason_if_none": ""}

    def _军师(self, schema: dict) -> dict[str, Any]:
        card_ids = [c for c in self._enum(
            schema, "properties", "proposals", "items", "properties", "card_id") if c]
        targets = [f["symptom"] for f in self._last_findings] or ["冷门商品学不动"]
        novel = not card_ids
        return {"proposals": [{
            "rank": 1,
            "card_id": card_ids[0] if card_ids else "",
            "targets": targets,
            "rationale": "冷门桶 0.552 比热门桶 0.638 低 0.086，是曝光次数不足导致的欠拟合",
            "expected": {"点击AUC": 0.001, "购买AUC": 0.004},
            "cost": {"代码难度": "简单", "训练时间倍数": 1.0},
            "risk": "热门桶可能被稀释",
            "novel": novel,
            "how_to": "按类目做兜底编码" if novel else "",
        }]}

    def _工兵(self, schema: dict) -> dict[str, Any]:
        return {
            "change_type": "加新零件",
            "config_patch": "features:\n  demo_op:\n    enabled: true\n",
            "new_files": [{
                "path": "modules/features/demo_op.py",
                "content": "# 离线演习用的空零件\nVALUE = 1\n",
            }],
            "self_check": [
                "未使用禁用字段 conversion 等五个",
                "统计量只用了训练集",
                "参数全部从配置读取",
            ],
        }

    def _复盘官(self, schema: dict) -> dict[str, Any]:
        # 方案声称要治哪几个病，就得逐个交代 —— 跟真复盘官一样的规矩
        symptoms = [f["symptom"] for f in self._last_findings] or ["冷门商品学不动"]
        promote = self.calls.get("复盘官", 0) in self.promote_on
        return {
            "verdict": "猜对了",
            "actual": {"点击AUC": 0.0008, "购买AUC": 0.0031},
            "vs_expected": "比预计的 0.004 略低",
            "symptom_resolved": [
                {"symptom": s, "before": 0.086, "after": 0.061, "resolved": "部分"}
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
    """

    def __init__(self, base_ctr: float = 0.6108, base_cvr: float = 0.5942,
                 gain: float = 0.003, decay: float = 0.45,
                 fail_rounds: tuple[int, ...] = (), seconds: float = 12.0):
        self.ctr, self.cvr = base_ctr, base_cvr
        self.gain, self.decay = gain, decay
        self.fail_rounds = set(fail_rounds)
        self.seconds = seconds
        self.runs = 0

    def run(self, patch: dict[str, Any], fidelity: str) -> RunResult:
        self.runs += 1
        if self.runs in self.fail_rounds:
            return RunResult(ok=False, error="（演习）训练进程被人为杀掉",
                             seconds=1.0, fidelity=fidelity)
        self.ctr += self.gain
        self.cvr += self.gain * 1.4
        self.gain *= self.decay              # 越往后越难涨，最终自然收敛
        return RunResult(ok=True, seconds=self.seconds, fidelity=fidelity,
                         health_report=self.report(fidelity))

    def report(self, fidelity: str = "小份") -> dict[str, Any]:
        """当前这一版的成绩单。外层循环要一份第 0 轮的做起点。"""
        return {
            "保真度": fidelity,
            "随机种子": 20260827,
            "验证集": {
                "总行数": 218_000, "点击数": 8_400, "转化数": 320,
                "点击分": round(self.ctr, 4),
                "购买分": round(self.cvr, 4),
                "购买分_全曝光口径": round(self.cvr + 0.02, 4),
            },
            "训练集": {"总行数": 1_910_000,
                      "点击分": round(self.ctr + 0.02, 4),
                      "购买分": round(self.cvr + 0.03, 4)},
            "当前特征": ["101", "205", "206", "301"],
            "未使用的字段": ["109_14", "110_14", "508", "509"],
            "按商品出现次数分组": [
                {"区间": "<10次", "样本占比": 0.34, "转化正样本数": 41,
                 "点击分": round(self.ctr - 0.02, 4), "购买分": round(self.cvr - 0.04, 4)},
                {"区间": ">1000次", "样本占比": 0.06, "转化正样本数": 96,
                 "点击分": round(self.ctr + 0.02, 4), "购买分": round(self.cvr + 0.04, 4)},
            ],
        }

"""改动没生效时，不许给卡片记负分。

这是「改动是否生效」那条机械判定的**用处**所在。harness 已经能证明这一轮
的验证预测跟上一轮逐位相同（改动压根没进训练），那么：

  · 假设当然不可能成立 —— 不许判「猜对了」
  · 更要紧的是：**不许给卡片扣信任分**

第二条才是真正保护的东西。靠谱度账本决定军师以后还提不提这张卡：扣到底
就等于把它拉黑。而这一轮什么都没执行，卡片是被冤枉的 —— 该修的是 harness
（零件缺 enabled、挂载点不存在、config_patch 落在没人读的键上），不是卡片。

2026-09-01 那场真跑里，复盘官靠自己推理躲过了这个坑两次（都判了「说不清」、
prior_delta 记 0）。但那是运气 —— 它得先想明白「减均值不改用户内次序」这种
事。现在有铁证了，就用代码强制，不再靠提示词自觉。
"""

import pytest

from agent.llm import Ledger, SchemaViolation
from agent import roles
from agent.knowledge import SymptomVocab


ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent


@pytest.fixture
def vocab():
    return SymptomVocab.load(ROOT / "knowledge" / "symptoms.yaml")


未生效的成绩单 = {
    "验证集": {"GAUC": 0.6363, "nDCG@5": 0.5236, "主分": 0.5800},
    "改动是否生效": {
        "结论": "未生效",
        "依据": "验证集预测与上一轮**逐位相同** —— 这次改动没有进入训练。",
    },
}

生效的成绩单 = {
    "验证集": {"GAUC": 0.6363, "nDCG@5": 0.5236, "主分": 0.5800},
    "改动是否生效": {"结论": "生效", "依据": "验证集预测与上一轮不同"},
}


def _reflect(vocab, data, result):
    captured = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            captured["validate"] = kw["validate"]
            return data

    roles.reflect(_FakeLLM(), vocab, {}, result, {}, None)
    captured["validate"](data)


def _reflection(verdict, prior_delta):
    return {
        "verdict": verdict,
        "actual": {"GAUC": 0.0, "nDCG@5": 0.0},
        "vs_expected": "预期 +0.002，实际 0.000",
        "symptom_resolved": [],
        "card_update": {
            "card_id": "早停与训练轮次",
            "prior_delta": prior_delta,
            "note": "验证集分数与改动前完全相同",
        },
        "next_hint": "检查零件是不是真的被加载了",
        "promote": False,
    }


def test_未生效时不许给卡片扣信任分(vocab):
    with pytest.raises(SchemaViolation, match="未生效"):
        _reflect(vocab, _reflection("说不清", -0.15), 未生效的成绩单)


def test_未生效时不许判猜对了(vocab):
    with pytest.raises(SchemaViolation, match="未生效"):
        _reflect(vocab, _reflection("猜对了", 0.0), 未生效的成绩单)


def test_未生效时判说不清且不动信任分是对的(vocab):
    _reflect(vocab, _reflection("说不清", 0.0), 未生效的成绩单)


def test_未生效时判没跑起来也可以(vocab):
    _reflect(vocab, _reflection("没跑起来", 0.0), 未生效的成绩单)


def test_改动生效时扣分照旧允许(vocab):
    """真的跑了、真的没效果 —— 该扣就扣，这条闸门不该挡住正常的记账。"""
    _reflect(vocab, _reflection("猜错了", -0.15), 生效的成绩单)


def test_没有这个块时行为不变(vocab):
    """老成绩单（或首轮判不了）不受影响。"""
    _reflect(vocab, _reflection("猜错了", -0.15), {"验证集": {"GAUC": 0.6}})

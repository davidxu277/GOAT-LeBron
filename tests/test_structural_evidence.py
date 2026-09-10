"""有些病天生没有数字可引 —— 证据规则不能把它们锁死。

医生的证据校验原本只有一条硬规则：**必须包含数字**。对「在背题」这种
（训练 0.6456 vs 验证 0.5800，差 0.0656）完全合适，但对另一类病是死结：

    历史行为没用上
      detect: 用户历史行为特征完全未使用……看「训练诊断 → 实际特征」
              里到底有哪些列进了模型。

它的证据是「特征清单里只有 user_id / video_id / author_id / tab /
duration_bucket，没有任何历史特征」—— 一个**关于清单的事实**，不是数值
比较。没有数字可引。

实测（2026-09-01，两场跑一共三次）：医生每次想报这个病都被打回，必须重
试一次才能过；有一次军师也因为同一条规则（方案理由必须引用数字）被连打
两次，整轮作废。而这个病指向的正是官方 Starter Kit 排第 2 的 headroom
方向（用户历史序列建模）—— **校验器在给最值钱的那条路加收过路费。**

规则改成：证据要么带数字，**要么点名它读的是成绩单的哪一块**。
"哪一块"不是写死的名单，是从病名自己的 `needs` 字段里抽出来的（那里本来
就用「」标好了要看什么），所以加新病名时不用回来改校验器。
"""

import pytest

from agent.llm import Ledger, SchemaViolation
from agent import roles
from agent.knowledge import SymptomVocab


@pytest.fixture
def vocab():
    return SymptomVocab.load()


def _diagnose(vocab, findings, report=None):
    data = {"findings": findings, "no_finding": False, "reason_if_none": ""}
    captured = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            captured["validate"] = kw["validate"]
            return data

    roles.diagnose(_FakeLLM(), vocab, report or {"验证集": {"GAUC": 0.63}})
    captured["validate"](data)


def _finding(symptom, evidence):
    return {"symptom": symptom, "severity": 0.5, "confidence": "高",
            "evidence": evidence, "affects": ["GAUC", "nDCG@5"]}


def test_点名了成绩单哪一块的结构型证据要放行(vocab):
    """「历史行为没用上」的 needs 写着要看「实际特征」和「装上的零件」。"""
    _diagnose(vocab, [_finding(
        "历史行为没用上",
        "训练诊断里的实际特征只有 user_id、video_id、author_id、tab、"
        "duration_bucket，没有任何用户历史行为列。",
    )])


def test_只点名不带数字也可以(vocab):
    """这类病本来就没有数字可引 —— 不该因此被打回。"""
    _diagnose(vocab, [_finding(
        "历史行为没用上",
        "看实际特征这一块，进模型的全是当次曝光的 id 类字段，"
        "用户过去看过什么完全没有进来。",
    )])


def test_既没数字也没点名的空话照旧打回(vocab):
    """放宽不等于放弃 —— 「感觉特征不太够」这种还是不算证据。"""
    with pytest.raises(SchemaViolation, match="证据"):
        _diagnose(vocab, [_finding(
            "历史行为没用上", "感觉用户侧的信息不太够，可以再加点特征。")])


def test_数值型的病带数字仍然照旧放行(vocab):
    _diagnose(vocab, [_finding(
        "在背题", "训练集主分 0.6456，验证集主分 0.57999，差 0.0656 > 0.04。")])


def test_数值型的病说空话也照旧打回(vocab):
    with pytest.raises(SchemaViolation, match="证据"):
        _diagnose(vocab, [_finding("在背题", "看起来有点过拟合。")])


def test_每个病名都有可用的证据锚点(vocab):
    """如果某个病的 needs 里一个「」都没有，医生就只剩"必须带数字"一条路 ——
    对结构型的病等于还是死结。这条测试盯着别再出现这种病名。"""
    缺锚点 = [
        s.id for s in vocab.all()
        if not roles.evidence_anchors(s)
    ]
    assert not 缺锚点, f"这些病名的 needs 里没有「」标出要看哪一块：{缺锚点}"

"""离线测试：不调用模型，不花一分钱。

覆盖的是"代码强制"的那一层 —— 它比提示词可靠得多，
所以必须有测试兜住。
"""

from __future__ import annotations

import pathlib
import sys

import pytest
import yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent.knowledge import Card, CardLibrary, SymptomVocab
from agent.llm import Ledger, SchemaViolation
from agent.loop import CostAwareScheduler
from agent import roles, schemas


@pytest.fixture(scope="module")
def vocab():
    return SymptomVocab.load()


@pytest.fixture(scope="module")
def cards(vocab):
    return CardLibrary.load(vocab)


# ────────────────── 词表与卡片：对暗号那一层 ──────────────────


def test_词表加载且包含四个核心病(vocab):
    for sid in ["转化样本偏差", "冷门商品学不动", "在背题", "新用户不会做"]:
        assert sid in vocab
        assert vocab[sid].core


def test_卡片标签必须全部合法(cards):
    assert len(cards) >= 2


def test_贴了词表外的标签直接报错(vocab, tmp_path):
    (tmp_path / "bad.yaml").write_text(
        yaml.safe_dump(
            {"编号": "瞎写的", "治哪些毛病": ["不存在的病"], "为什么管用": "x"},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="词表里没有的病名"):
        CardLibrary.load(vocab, cards_dir=tmp_path)


def test_缺少为什么管用的卡片直接报错(vocab, tmp_path):
    (tmp_path / "bad.yaml").write_text(
        yaml.safe_dump(
            {"编号": "偷懒卡", "治哪些毛病": ["在背题"], "为什么管用": ""},
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="为什么管用"):
        CardLibrary.load(vocab, cards_dir=tmp_path)


def test_按病名筛卡片(cards):
    hits = cards.match(["冷门商品学不动"])
    assert [c.id for c in hits] == ["类目兜底"]
    assert cards.match(["训练太慢"]) == []


def test_筛卡片会排除已试过的(cards):
    """卡片库会一直长大，所以断言「被排除的那张不在结果里」，而不是断言结果为空。"""
    命中 = [c.id for c in cards.match(["转化样本偏差"], limit=99)]
    assert "ESMM" in 命中

    排除后 = [c.id for c in cards.match(["转化样本偏差"], exclude_ids={"ESMM"}, limit=99)]
    assert "ESMM" not in 排除后
    assert set(排除后) == set(命中) - {"ESMM"}   # 只少了那一张，别的没被误伤


# ────────────────── 医生：病名 enum 由词表生成 ──────────────────


def test_医生schema的病名枚举来自词表(vocab):
    enum = schemas.doctor_schema(vocab)["properties"]["findings"]["items"] \
        ["properties"]["symptom"]["enum"]
    assert enum == vocab.ids   # 模型在物理上说不出词表外的病名


def _doctor_validate(vocab, data):
    captured = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            captured["validate"] = kw["validate"]
            return data

    roles.diagnose(_FakeLLM(), vocab, {"总分": {}})
    captured["validate"](data)


def test_证据里没有数字会被打回(vocab):
    data = {
        "findings": [{
            "symptom": "在背题", "severity": 0.5, "confidence": "高",
            "evidence": "训练分明显高于验证分", "affects": ["点击AUC"],
        }],
        "no_finding": False, "reason_if_none": "",
    }
    with pytest.raises(SchemaViolation, match="没有任何数字"):
        _doctor_validate(vocab, data)


def test_证据里有数字就放行(vocab):
    data = {
        "findings": [{
            "symptom": "在背题", "severity": 0.5, "confidence": "高",
            "evidence": "训练 0.7412 vs 验证 0.6187，差 0.1225", "affects": ["点击AUC"],
        }],
        "no_finding": False, "reason_if_none": "",
    }
    _doctor_validate(vocab, data)


def test_没查出问题时findings必须为空(vocab):
    data = {
        "findings": [{
            "symptom": "在背题", "severity": 0.1, "confidence": "低",
            "evidence": "差 0.01", "affects": ["点击AUC"],
        }],
        "no_finding": True, "reason_if_none": "都正常",
    }
    with pytest.raises(SchemaViolation, match="findings 必须为空"):
        _doctor_validate(vocab, data)


# ────────────────── 工兵：红线 ──────────────────


def _impl_validate(data):
    captured = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            captured["validate"] = kw["validate"]
            return data

    roles.implement(_FakeLLM(), {"card_id": "", "how_to": ""}, None, "", "", "")
    captured["validate"](data)


BASE_CHECK = ["未使用禁用字段 conversion", "统计量只用了训练集", "参数从配置读取"]


def test_把答案字段当特征会被拦下():
    data = {
        "change_type": "加新零件",
        "config_patch": "",
        "new_files": [{
            "path": "modules/features/x.py",
            "content": "cols = ['user_id', 'conversion']",
        }],
        "self_check": BASE_CHECK,
    }
    with pytest.raises(SchemaViolation, match="禁用字段 conversion"):
        _impl_validate(data)


def test_写到modules之外会被拦下():
    data = {
        "change_type": "加新零件",
        "config_patch": "",
        "new_files": [{"path": "harness/runner.py", "content": "x = 1"}],
        "self_check": BASE_CHECK,
    }
    with pytest.raises(SchemaViolation, match="只能在 modules/ 里新建"):
        _impl_validate(data)


def test_合法的实现放行():
    data = {
        "change_type": "加新零件",
        "config_patch": "features:\n  x:\n    enabled: true\n",
        "new_files": [{
            "path": "modules/features/x.py",
            "content": "cols = ['user_id', 'item_id']",
        }],
        "self_check": BASE_CHECK,
    }
    _impl_validate(data)


# ────────────────── 复盘官：防自我欺骗 ──────────────────


def _reflect_validate(vocab, data):
    captured = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            captured["validate"] = kw["validate"]
            return data

    roles.reflect(_FakeLLM(), vocab, {}, {}, {}, None)
    captured["validate"](data)


def _reflection(verdict, resolved, gain, delta=0.1):
    return {
        "verdict": verdict,
        "actual": {"点击AUC": 0.0, "购买AUC": gain},
        "vs_expected": "",
        "symptom_resolved": {
            "symptom": "冷门商品学不动", "before": 0.07, "after": 0.07, "resolved": resolved,
        },
        "card_update": {"card_id": "类目兜底", "prior_delta": delta, "note": ""},
        "next_hint": "", "promote": False,
    }


def test_分数涨了但毛病没治好不许判猜对了(vocab):
    """整套系统里最重要的一条守则。"""
    with pytest.raises(SchemaViolation, match="必须判「说不清」"):
        _reflect_validate(vocab, _reflection("猜对了", "否", 0.004))


def test_毛病治好了才可以判猜对了(vocab):
    _reflect_validate(vocab, _reflection("猜对了", "是", 0.004))


def test_提升低于门槛不许判猜对了(vocab):
    with pytest.raises(SchemaViolation, match="低于 0.0005"):
        _reflect_validate(vocab, _reflection("猜对了", "是", 0.0002))


def test_说不清时不许大改卡片可信度(vocab):
    with pytest.raises(SchemaViolation, match="不应大幅调整"):
        _reflect_validate(vocab, _reflection("说不清", "否", 0.004, delta=0.2))


# ────────────────── 调度器：性价比公式 ──────────────────


def _proposal(card_id, gain, 难度, 倍数):
    return {
        "rank": 1, "card_id": card_id, "targets": ["冷门商品学不动"],
        "rationale": "", "expected": {"点击AUC": 0.0, "购买AUC": gain},
        "cost": {"代码难度": 难度, "训练时间倍数": 倍数},
        "risk": "", "novel": not card_id, "how_to": "",
    }


def test_预期最高的方案不一定胜出(cards):
    """又难写又跑得慢的方案，哪怕预期提升是两倍多，性价比也可能更低。

    这正是"成本感知"的意义：不是挑看起来能提最多的，是挑单位算力回报最高的。
    """
    小而准 = _proposal("类目兜底", 0.003, "简单", 1.0)   # 0.003*0.60/(1.0*1.0) = 0.00180
    大而贵 = _proposal("ESMM", 0.008, "难", 2.0)        # 0.008*0.85/(3.0*2.0) = 0.00113

    sched = CostAwareScheduler()
    assert 大而贵["expected"]["购买AUC"] > 小而准["expected"]["购买AUC"]   # 预期更高
    assert sched.score(大而贵, cards) < sched.score(小而准, cards)        # 性价比更低

    chosen, fidelity, backups = sched.pick([大而贵, 小而准], cards, "一般")
    assert chosen is 小而准
    assert fidelity == "小份"          # 全新的招一律从最小数据起步
    assert backups == [大而贵]         # 没选中的存为备胎，工兵失败时直接换


def test_预算紧张时只留便宜方案(cards):
    便宜 = _proposal("类目兜底", 0.001, "改配置", 1.0)
    昂贵 = _proposal("ESMM", 0.02, "难", 3.0)
    chosen, _, _ = CostAwareScheduler().pick([昂贵, 便宜], cards, "紧张")
    assert chosen is 便宜


def test_已经试过的卡不会被再选(cards):
    a = _proposal("类目兜底", 0.003, "简单", 1.0)
    b = _proposal("ESMM", 0.001, "难", 3.0)
    chosen, _, _ = CostAwareScheduler(tried_cards={"类目兜底"}).pick([a, b], cards, "一般")
    assert chosen is b


# ────────────────── 记账 ──────────────────


def test_按角色记账并估算花费():
    led = Ledger()
    led.add("医生", "claude-opus-5", 3000, 500)
    led.add("工兵", "claude-haiku-4-5", 6000, 1500)
    assert led.total_tokens == 11000
    assert led.by_role["医生"].calls == 1
    # 3000/1e6*5 + 500/1e6*25 + 6000/1e6*1 + 1500/1e6*5
    assert led.total_cost_usd == pytest.approx(0.0410, abs=1e-4)

"""离线测试：不调用模型，不花一分钱。

覆盖的是"代码强制"的那一层 —— 它比提示词可靠得多，
所以必须有测试兜住。
"""

from __future__ import annotations

import json
import pathlib
import sys

import pytest
import yaml

from conftest import need

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent.knowledge import Card, CardLibrary, SymptomVocab
from agent.llm import Ledger, SchemaViolation
from agent.loop import (SHELF_KEEP, CostAwareScheduler, InterventionLog, PriorLedger,
                        RunResult, Shelf, TimeLedger, _with_bands, effective_config,
                        run_round, run_session)
from agent.offline import DriftingExecutor, ScriptedLLM
from agent import noise, roles, schemas


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
    """同上：卡片库会一直长大，所以断言「对症的在、不对症的不在」，不锁死具体名单。"""
    命中 = [c.id for c in cards.match(["冷门商品学不动"], limit=99)]
    assert "类目兜底" in 命中
    assert "ESMM" not in 命中          # ESMM 不治这个病，不该被筛出来

    # 每一张被筛出来的卡，标签里都必须真的有这个病
    for card in cards.match(["冷门商品学不动"], limit=99):
        assert "冷门商品学不动" in card.treats


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


def _reflect_validate(vocab, data, targets=None):
    captured = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            captured["validate"] = kw["validate"]
            return data

    hypothesis = {"targets": targets} if targets else {}
    roles.reflect(_FakeLLM(), vocab, hypothesis, {}, {}, None)
    captured["validate"](data)


def _resolved(symptom="冷门商品学不动", resolved="是", before=0.07, after=None):
    # resolved 与 before/after 必须自洽：说治好了，那两个数就得真的变了
    if after is None:
        after = before if resolved == "否" else 0.03
    return {"symptom": symptom, "before": before, "after": after, "resolved": resolved}


def _reflection(verdict, resolved, gain, delta=0.1, promote=False, after=None, items=None):
    return {
        "verdict": verdict,
        "actual": {"点击AUC": 0.0, "购买AUC": gain},
        "vs_expected": "",
        "symptom_resolved": items or [_resolved(resolved=resolved, after=after)],
        "card_update": {"card_id": "类目兜底", "prior_delta": delta, "note": ""},
        "next_hint": "", "promote": promote,
    }


def test_分数涨了但毛病没治好不许判猜对了(vocab):
    """整套系统里最重要的一条守则。"""
    with pytest.raises(SchemaViolation, match="必须判「说不清」"):
        _reflect_validate(vocab, _reflection("猜对了", "否", 0.004))


def test_毛病治好了才可以判猜对了(vocab):
    _reflect_validate(vocab, _reflection("猜对了", "是", 0.004))


def test_提升低于门槛不许判猜对了(vocab):
    with pytest.raises(SchemaViolation, match="0.0005"):
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


def test_耗时账本_没跑过就退回猜测值():
    ledger = TimeLedger()
    assert ledger.multiplier("类目兜底", 1.2) == 1.2          # 空账本
    ledger.record("类目兜底", 300)
    assert ledger.multiplier("类目兜底", 1.2) == 1.2          # 只有自己，没得比
    assert ledger.multiplier("没跑过的卡", 2.0) == 2.0


def test_耗时账本_实测倍数是相对全部运行的中位数():
    ledger = TimeLedger()
    ledger.record("类目兜底", 100)
    ledger.record("ESMM", 300)
    # 全部运行的中位耗时 = 200 → 类目兜底 0.5 倍，ESMM 1.5 倍
    assert ledger.multiplier("类目兜底", 1.0) == pytest.approx(0.5)
    assert ledger.multiplier("ESMM", 1.2) == pytest.approx(1.5)


def test_耗时账本_坏输入不入账():
    ledger = TimeLedger()
    ledger.record("", 100)        # 自创方案没有 card_id
    ledger.record("ESMM", 0.0)    # 假执行器的 0 耗时
    ledger.record("ESMM", -5)
    assert ledger.records == {}


def test_调度器用实测耗时而不是军师的报价(cards):
    """军师说两个方案一样快（倍数都报 1.0），但账本知道 ESMM 实测慢 6 倍。

    实测倍数：ESMM = 600/350 ≈ 1.71，类目兜底 = 100/350 ≈ 0.29。
    尽管 ESMM 的靠谱度更高（0.85 vs 0.60），实测成本一除就翻盘了。
    """
    ledger = TimeLedger()
    ledger.record("ESMM", 600)
    ledger.record("类目兜底", 100)

    p_esmm = _proposal("ESMM", 0.003, "简单", 1.0)
    p_fallback = _proposal("类目兜底", 0.003, "简单", 1.0)

    有账本 = CostAwareScheduler(time_ledger=ledger)
    没账本 = CostAwareScheduler()
    assert 没账本.score(p_esmm, cards) > 没账本.score(p_fallback, cards)   # 只看报价：ESMM 靠谱度高，胜
    assert 有账本.score(p_esmm, cards) < 有账本.score(p_fallback, cards)   # 看实测：ESMM 太慢，败


def test_耗时账本_落盘再读回(tmp_path):
    ledger = TimeLedger()
    ledger.record("ESMM", 600)
    path = tmp_path / "time_ledger.json"
    ledger.dump(path)
    assert TimeLedger.load(path).records == {"ESMM": [600.0]}
    assert TimeLedger.load(tmp_path / "不存在.json").records == {}


def test_按角色记账并估算花费():
    led = Ledger()
    led.add("医生", "claude-opus-5", 3000, 500)
    led.add("工兵", "claude-haiku-4-5", 6000, 1500)
    assert led.total_tokens == 11000
    assert led.by_role["医生"].calls == 1
    # 3000/1e6*5 + 500/1e6*25 + 6000/1e6*1 + 1500/1e6*5
    assert led.total_cost_usd == pytest.approx(0.0410, abs=1e-4)


# ────────────────── 复盘官：新补的四道墙 ──────────────────


def test_自我申报必须跟数字一致(vocab):
    """更隐蔽的一种自欺：before / after 一模一样，却填 resolved=是。

    resolved 是模型自己写的，不拿它自己给的数字对一遍，这条就是白纸。
    """
    with pytest.raises(SchemaViolation, match="自我申报必须跟数字一致"):
        _reflect_validate(vocab, _reflection("猜对了", "是", 0.004, after=0.07))


def test_两个指标都没涨不许判猜对了(vocab):
    data = _reflection("猜对了", "是", -0.002)
    with pytest.raises(SchemaViolation, match="都没有上涨"):
        _reflect_validate(vocab, data)


def test_判猜错了还给卡片加分会被打回(vocab):
    with pytest.raises(SchemaViolation, match="方向反了"):
        _reflect_validate(vocab, _reflection("猜错了", "否", -0.004, delta=0.1))


def test_只有猜对了才准升到更大数据(vocab):
    with pytest.raises(SchemaViolation, match="才准 promote"):
        _reflect_validate(vocab, _reflection("说不清", "否", 0.0004, delta=0.0, promote=True))
    _reflect_validate(vocab, _reflection("猜对了", "是", 0.004, promote=True))


def test_噪声带顶掉默认门槛(vocab):
    """实测抖动 0.006 时，0.004 的"提升"就是噪声，不许判猜对了。"""
    captured = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            captured["validate"] = kw["validate"]
            return {}

    roles.reflect(_FakeLLM(), vocab, {}, {}, {}, None, noise_floor=0.006)
    with pytest.raises(SchemaViolation, match="0.006"):
        captured["validate"](_reflection("猜对了", "是", 0.004))


# ────────────────── 工兵：路径穿越与配置越权 ──────────────────


def test_路径里带两个点会被拦下():
    data = {
        "change_type": "加新零件",
        "config_patch": "",
        "new_files": [{"path": "modules/../harness/runner.py", "content": "x = 1"}],
        "self_check": BASE_CHECK,
    }
    with pytest.raises(SchemaViolation, match="`..`"):
        _impl_validate(data)


def test_配置补丁只准动三棵子树():
    data = {
        "change_type": "只改配置",
        "config_patch": "eval:\n  cvr_space: all\n",     # 偷偷改评估口径
        "new_files": [],
        "self_check": BASE_CHECK,
    }
    with pytest.raises(SchemaViolation, match="只准动"):
        _impl_validate(data)


def test_配置补丁语法错了会被拦下():
    data = {
        "change_type": "只改配置",
        "config_patch": "features:\n  - a\n bad indent:\n",
        "new_files": [],
        "self_check": BASE_CHECK,
    }
    with pytest.raises(SchemaViolation, match="不是合法 YAML"):
        _impl_validate(data)


def test_合法的配置补丁放行():
    data = {
        "change_type": "只改配置",
        "config_patch": "train:\n  early_stopping:\n    patience: 2\n",
        "new_files": [],
        "self_check": BASE_CHECK,
    }
    _impl_validate(data)


# ────────────────── 军师预计提升限幅 ──────────────────


def test_预计提升限幅不在schema层_因为接口不支持(vocab):
    """这条测试原本断言限幅写在 schema 的 maximum 里 —— 而结构化输出接口
    **根本不支持数值约束**，写了整个请求 400。也就是说那道闸门从来没生效过，
    而测试是绿的（它只检查了字典里有没有那个键，没检查接口认不认）。

    限幅现在由 roles.propose 的 validate 强制，见
    test_军师_预计提升超过上限会被打回。这里只钉住"别再写回 schema 里"。
    """
    prop = schemas.strategist_schema(vocab, ["ESMM"])["properties"]["proposals"]["items"]
    for m in ("点击AUC", "购买AUC"):
        字段 = prop["properties"]["expected"]["properties"][m]
        assert "maximum" not in 字段 and "minimum" not in 字段
        assert str(schemas.EXPECTED_CAP) in 字段["description"]   # 但要告诉模型上限是多少


# ────────────────── 靠谱度账本 ──────────────────


def test_靠谱度账本_按规则加减():
    led = PriorLedger()
    assert led.value("ESMM", 0.85) == 0.85                       # 空账本用卡上的先验
    assert led.apply("ESMM", "猜对了", 0.85, symptom_improved=True) == pytest.approx(0.95)  # 限幅
    assert led.apply("类目兜底", "猜错了", 0.60) == pytest.approx(0.50)
    assert led.apply("AITM", "没跑起来", 0.50) == pytest.approx(0.35)
    assert led.apply("DCNv2", "说不清", 0.50) == pytest.approx(0.50)


def test_靠谱度账本_猜对了但没超噪声带不加分():
    led = PriorLedger()
    assert led.apply("ESMM", "猜对了", 0.50, beat_noise=False) == pytest.approx(0.50)


def test_靠谱度账本_限幅在0点05到0点95():
    led = PriorLedger()
    for _ in range(20):
        led.apply("ESMM", "猜错了", 0.5)
    assert led.values["ESMM"] == pytest.approx(PriorLedger.FLOOR)


def test_靠谱度账本_自创方案没有卡可更新():
    led = PriorLedger()
    assert led.apply("", "猜对了", 0.5) == 0.5
    assert led.values == {}


def test_靠谱度账本_盖到卡片上而不动yaml(cards):
    led = PriorLedger(values={"ESMM": 0.21})
    led.apply_to(cards)
    assert cards.get("ESMM").prior == pytest.approx(0.21)
    assert CardLibrary.load(SymptomVocab.load()).get("ESMM").prior == pytest.approx(0.85)


def test_靠谱度账本_落盘再读回(tmp_path):
    led = PriorLedger(values={"ESMM": 0.7})
    path = tmp_path / "prior_ledger.json"
    led.dump(path)
    assert PriorLedger.load(path).values == {"ESMM": 0.7}
    assert PriorLedger.load(tmp_path / "没有.json").values == {}


# ────────────────── 卡片：失败信号 ──────────────────


def test_失败信号读进来了但不给军师看(cards):
    card = cards.get("ESMM")
    assert "损失接错" in card.failure_signals          # 复盘官要用
    assert card.failure_signals not in card.as_prompt_block()   # 军师不许看到


# ────────────────── 外层循环：整场跑通 ──────────────────


def test_一整场_状态在轮与轮之间传下去(tmp_path):
    """最关键的一条：这一轮跑出的成绩单，必须变成下一轮医生的输入。"""
    llm, ex = ScriptedLLM(), DriftingExecutor()
    summary = run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=5, logs_dir=tmp_path,
    )
    rows = [json.loads(l) for l in (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == summary.rounds_run >= 2
    分数 = [r["metrics"]["验证集"]["点击分"] for r in rows if r["metrics"]]
    assert 分数 == sorted(分数) and 分数[0] < 分数[-1]     # 一轮比一轮高 = 状态真的传下去了
    assert summary.best_round > 0
    assert summary.total_tokens > 0
    assert (tmp_path / "session_summary.json").exists()
    assert (tmp_path / "best_report.json").exists()


def test_一整场_每轮日志都符合交付物要求(tmp_path):
    llm, ex = ScriptedLLM(), DriftingExecutor()
    run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, logs_dir=tmp_path,
    )
    rows = [json.loads(l) for l in (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    first = rows[0]
    assert first["diagnosis"]["findings"]          # 假设
    assert first["patch_files"]                    # 代码改动全文
    assert first["metrics"]["验证集"]["点击分"]      # 指标
    assert first["interventions"] == 0             # 人工干预
    assert "recoveries" in first                   # 错误与恢复


def test_一整场_角色炸了不会拖垮整场(tmp_path):
    """医生第 2 次调用抛异常：那一轮作废，后面的轮次照跑。"""
    llm = ScriptedLLM(faults={"医生": [2]})
    ex = DriftingExecutor()
    summary = run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=4, logs_dir=tmp_path,
    )
    rows = [json.loads(l) for l in (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert summary.rounds_run == 4
    assert rows[1]["diagnosis"] is None and rows[1]["recoveries"]
    assert rows[2]["run_ok"]                       # 第 3 轮照常跑
    assert summary.recoveries >= 1


def test_一整场_训练失败不调复盘官(tmp_path):
    """跑都没跑起来就没什么可复盘的，这时候调大模型是纯浪费。"""
    llm = ScriptedLLM()
    ex = DriftingExecutor(fail_rounds=(2,))
    run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, logs_dir=tmp_path,
    )
    rows = [json.loads(l) for l in (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    炸掉那轮 = rows[1]
    assert 炸掉那轮["reflection"]["verdict"] == "没跑起来"
    assert 炸掉那轮["reflection"]["由代码合成"] is True
    assert llm.calls["复盘官"] == llm.calls["工兵"] - 1     # 少调了一次


def test_一整场_失败的卡会被拉黑(tmp_path):
    llm = ScriptedLLM()
    ex = DriftingExecutor(fail_rounds=(1,))
    run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=4, logs_dir=tmp_path,
    )
    rows = [json.loads(l) for l in (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    炸掉的卡 = rows[0]["chosen"]["card_id"]
    后面选的卡 = [r["chosen"]["card_id"] for r in rows[1:] if r["chosen"]]
    assert 炸掉的卡 and 炸掉的卡 not in 后面选的卡


def test_一整场_不涨了就自己停(tmp_path):
    """分数不动的执行器：跑满 patience 轮就该判收敛，不该把 20 轮全烧掉。"""
    llm = ScriptedLLM(promote_on=())
    ex = DriftingExecutor(gain=0.0, decay=1.0)
    summary = run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=20, patience=2, logs_dir=tmp_path,
    )
    assert summary.rounds_run <= 4
    assert "收敛" in summary.stopped_because


def test_一整场_预算耗尽就停(tmp_path):
    llm = ScriptedLLM(promote_on=())
    ex = DriftingExecutor()
    summary = run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=20, token_budget=20_000, logs_dir=tmp_path,
    )
    assert summary.stopped_because == "预算耗尽"
    assert summary.total_tokens >= 20_000


def test_一整场_结果表算得出相对基线的差值(tmp_path):
    llm, ex = ScriptedLLM(), DriftingExecutor()
    summary = run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, baseline={"点击AUC": 0.6000, "购买AUC": 0.5900}, logs_dir=tmp_path,
    )
    assert summary.deltas["点击AUC"] == pytest.approx(
        summary.best_scores["点击AUC"] - 0.6000)
    assert "相对基线" in summary.as_table()


# ────────────────── 噪声带 ──────────────────


def test_噪声带_正样本越少抖得越厉害():
    """47 条正样本的 AUC 和 4 万条正样本的 AUC，不是一回事。"""
    少 = noise.hanley_mcneil_se(0.70, n_pos=47, n_neg=940)
    多 = noise.hanley_mcneil_se(0.70, n_pos=41_200, n_neg=824_000)
    assert 少 > 多 * 10


def test_噪声带_从多次运行算出来():
    reports = [
        {"保真度": "小份", "验证集": {"点击分": 0.610, "购买分": 0.590},
         "按商品出现次数分组": [{"区间": "<10次", "点击分": 0.59, "购买分": 0.55,
                            "转化正样本数": 47}]},
        {"保真度": "小份", "验证集": {"点击分": 0.614, "购买分": 0.596},
         "按商品出现次数分组": [{"区间": "<10次", "点击分": 0.60, "购买分": 0.57,
                            "转化正样本数": 45}]},
        {"保真度": "小份", "验证集": {"点击分": 0.612, "购买分": 0.602},
         "按商品出现次数分组": [{"区间": "<10次", "点击分": 0.58, "购买分": 0.53,
                            "转化正样本数": 49}]},
    ]
    bands = noise.summarize(reports, seeds=[1, 2, 3])
    assert bands["单指标噪声带"] > 0
    assert bands["单指标噪声带"] == pytest.approx(bands["购买分"]["噪声带"])   # 购买分抖得更厉害
    assert bands["分组"]["按商品出现次数分组"]["<10次"]["转化正样本数"] == 47
    assert "噪声带" in bands["表格"]


def test_噪声带_挂到成绩单上给医生看():
    bands = {"单指标噪声带": 0.006, "分组": {}}
    报告 = {"验证集": {"点击分": 0.61}}
    带了 = _with_bands(报告, bands)
    assert 带了["噪声带"]["单指标"] == 0.006
    assert "验证集" in 带了 and "噪声带" not in 报告       # 不动原文


def test_噪声带_医生看到的也是分指标的():
    """医生和复盘官必须看同一套数字 —— 只修复盘官那边，医生还在用合成的
    单一门槛，会重演"点击真提升被购买抖动淹没"那类误判，只是换个角色犯错。"""
    bands = {"单指标噪声带": 0.09, "分指标噪声带": {"点击AUC": 0.001, "购买AUC": 0.09},
            "分组": {}}
    带了 = _with_bands({"验证集": {"点击分": 0.61}}, bands)
    assert 带了["噪声带"]["分指标"] == {"点击AUC": 0.001, "购买AUC": 0.09}


# ────────────────── 真执行器：落地补丁那一段 ──────────────────


def test_真执行器_能吃下YAML文本的配置补丁(tmp_path, monkeypatch):
    """工兵产出的 config_patch 是 YAML **文本**，不是 dict。

    执行器早先直接对它 .items()，任何一个带配置改动的补丁都会当场
    AttributeError —— 而这正是最常见的那种补丁。
    """
    pytest.importorskip("pandas")
    from harness import executor as ex_mod

    monkeypatch.setattr(ex_mod, "ROOT", tmp_path)
    ex = ex_mod.RealExecutor.__new__(ex_mod.RealExecutor)
    ex.config = {}
    ex._apply_patch({
        "config_patch": "train:\n  seed: 7\nfeatures:\n  x:\n    enabled: true\n",
        "new_files": [{"path": "modules/features/x.py", "content": "V = 1\n"}],
    })
    assert ex.config["train"] == {"seed": 7}
    assert (tmp_path / "modules" / "features" / "x.py").read_text(encoding="utf-8") == "V = 1\n"


def test_真执行器_不许写到modules之外(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    from harness import executor as ex_mod

    monkeypatch.setattr(ex_mod, "ROOT", tmp_path)
    ex = ex_mod.RealExecutor.__new__(ex_mod.RealExecutor)
    ex.config = {}
    with pytest.raises(ValueError, match="非法写入路径"):
        ex._apply_patch({"config_patch": "",
                         "new_files": [{"path": "harness/x.py", "content": "V = 1"}]})


def test_真执行器_满足外层循环要的协议():
    pytest.importorskip("pandas")
    from harness.executor import RealExecutor
    from agent.loop import Executor

    assert isinstance(RealExecutor("a", "b"), Executor)   # Protocol 运行时检查


def test_一整场_最大数据上还查不出病就算收敛(tmp_path):
    """升到顶了还连着查不出问题 —— 那就是收敛，不该继续空转烧医生。"""
    class _没病医生(ScriptedLLM):
        def _医生(self, schema):
            return {"findings": [], "no_finding": True, "reason_if_none": "都在噪声带内"}

    llm, ex = _没病医生(), DriftingExecutor()
    summary = run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=30, logs_dir=tmp_path,
    )
    # 4 档 × 每档 2 轮 = 8 轮左右就该停，不会把 30 轮烧完
    assert summary.rounds_run <= 10
    assert "收敛" in summary.stopped_because
    assert llm.calls.get("军师") is None          # 一次都没查出病，后面三个角色一次没调


def test_一整场_范文可以是按环节取的函数(tmp_path):
    """成员2 的 example_for：改训练过程的看训练类范文，加特征的看特征类范文。

    外层循环必须把这个函数原样传下去，不能在半路被当成字符串。
    """
    看到的环节 = []

    def 取范文(stage: str) -> str:
        看到的环节.append(stage)
        return "# 范文\n"

    llm, ex = ScriptedLLM(), DriftingExecutor()
    run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module=取范文, current_config="",
        rounds=2, logs_dir=tmp_path,
    )
    assert 看到的环节                      # 真的被调用了
    assert all(isinstance(s, str) for s in 看到的环节)
# ────────────────── 输出被 max_tokens 截断 ──────────────────


class _FakeResp:
    """假的 Anthropic 回复：只需要 stop_reason / usage / content 三样。"""

    def __init__(self, stop_reason, text=""):
        self.stop_reason = stop_reason
        self.usage = type("U", (), {"input_tokens": 100, "output_tokens": 96000})()
        self.content = [type("B", (), {"type": "text", "text": text})()]


class _FakeClient:
    """假的 Anthropic 客户端。两条路都要有 —— 大请求走 stream，小请求走 create。"""

    def __init__(self, resp):
        class _Stream:
            def __enter__(self_inner):
                return self_inner
            def __exit__(self_inner, *a):
                return False
            def get_final_message(self_inner):
                return resp

        self.messages = type("M", (), {
            "create": lambda *a, **k: resp,
            "stream": lambda *a, **k: _Stream(),
        })()


def test_撞上token上限要报预算不够而不是JSON坏了():
    """截断时 JSON 必然不完整，若照原样报「Unterminated string」，
    人和重试时的模型都会以为是模型不听话，往错误方向诊断。"""
    from agent.llm import LLM

    llm = LLM(client=_FakeClient(_FakeResp("max_tokens", '{"findings": [')))
    with pytest.raises(SchemaViolation, match="被截断"):
        llm.call(role="工兵", system="", user="", schema={})

    # 这次调用照样要记账 —— 烧掉的 token 不能因为失败就漏记
    assert llm.ledger.total_tokens == 96100


def test_模型拒绝与截断是两种不同的错():
    from agent.llm import LLM

    llm = LLM(client=_FakeClient(_FakeResp("refusal")))
    with pytest.raises(SchemaViolation, match="拒绝"):
        llm.call(role="医生", system="", user="", schema={})


def test_max_tokens大时必须走流式():
    """SDK 规定预估超 10 分钟的请求必须流式，非流式会直接抛 ValueError。

    max_tokens 提到 96k 后踩过这个坑：医生每轮都失败，整场空转两轮。
    """
    from agent.llm import LLM, _STREAM_THRESHOLD

    用了流式 = []

    class _Stream:
        def __enter__(self):
            用了流式.append(True)
            return self
        def __exit__(self, *a):
            return False
        def get_final_message(self):
            return _FakeResp("end_turn", '{"ok": 1}')

    class _Client:
        def __init__(self):
            self.messages = type("M", (), {
                "create": lambda *a, **k: _FakeResp("end_turn", '{"ok": 1}'),
                "stream": lambda *a, **k: _Stream(),
            })()

    llm = LLM(client=_Client())
    llm.call(role="工兵", system="", user="", schema={},
             max_tokens=_STREAM_THRESHOLD + 1)
    assert 用了流式, "max_tokens 超阈值时应该走流式，否则 SDK 会抛 ValueError"

    用了流式.clear()
    llm.call(role="医生", system="", user="", schema={},
             max_tokens=_STREAM_THRESHOLD - 1)
    assert not 用了流式, "小请求不必流式，非流式更简单"


# ────────────────── 配置：白名单与深度合并 ──────────────────


def test_特征清单在白名单内可改():
    """医生最常诊断出的就是「特征没用上」，改特征却被白名单拦住的话，
    等于一边让它诊断、一边堵死修复的路。base_fields 必须在 features 下。"""
    import yaml as _yaml
    from agent.roles import _check_config_patch

    cfg = _yaml.safe_load(
        (pathlib.Path(__file__).resolve().parent.parent
         / "config" / "pipeline.yaml").read_text(encoding="utf-8"))
    assert "base_fields" in cfg["features"], "特征清单要放在 features 下，工兵才改得动"
    assert "fidelity" not in cfg.get("train", {}), "数据规模归调度器管，不能让工兵改"

    _check_config_patch("features:\n  base_fields: ['101', '205']\n")   # 不该抛


def test_改配置不会冲掉同级的其他键():
    """浅层赋值的坑：工兵只想改一个 K，结果把 features 下其他零件全抹了。"""
    from harness.executor import _deep_set

    cfg = {"features": {"类目兜底": {"enabled": True, "K": 20},
                        "目标编码": {"enabled": False}}}
    _deep_set(cfg, ["features", "类目兜底", "K"], 50)
    assert cfg["features"]["类目兜底"]["K"] == 50
    assert cfg["features"]["类目兜底"]["enabled"] is True      # 同级键还在
    assert "目标编码" in cfg["features"]                       # 兄弟零件还在


def test_执行器默认读配置文件():
    """不读的话，工兵改配置类的方案永远等于没改，复盘官只会一直判「猜错了」。"""
    from harness.executor import _load_pipeline_config

    cfg = _load_pipeline_config()
    assert cfg.get("features", {}).get("base_fields"), "应该读到特征清单"
# ────────────────── 复盘官：多个目标毛病 ──────────────────


def test_方案打了几个病就得逐个交代(vocab):
    """26 张卡里 11 张是多病卡。一个方案打三个病、复盘只报一个，
    剩下两个就永远没人验证 —— 这是最容易漏掉的一种"没做完"。
    """
    data = _reflection("猜对了", "是", 0.004,
                       items=[_resolved("冷门商品学不动")])
    with pytest.raises(SchemaViolation, match="没有交代"):
        _reflect_validate(vocab, data, targets=["冷门商品学不动", "新用户不会做"])


def test_逐个交代了就放行(vocab):
    data = _reflection("猜对了", "是", 0.004, items=[
        _resolved("冷门商品学不动", "是"),
        _resolved("新用户不会做", "部分", before=0.05, after=0.04),
    ])
    _reflect_validate(vocab, data, targets=["冷门商品学不动", "新用户不会做"])


def test_多个目标里有一个好转就算数(vocab):
    """两个目标，一个治好了一个没有 —— 这仍然可以判「猜对了」。"""
    data = _reflection("猜对了", "是", 0.004, items=[
        _resolved("冷门商品学不动", "是"),
        _resolved("新用户不会做", "否", before=0.05),
    ])
    _reflect_validate(vocab, data, targets=["冷门商品学不动", "新用户不会做"])


def test_全部目标都没好转就不许判猜对了(vocab):
    data = _reflection("猜对了", "否", 0.004, items=[
        _resolved("冷门商品学不动", "否"),
        _resolved("新用户不会做", "否", before=0.05),
    ])
    with pytest.raises(SchemaViolation, match="所有目标毛病都没有改善"):
        _reflect_validate(vocab, data, targets=["冷门商品学不动", "新用户不会做"])


def test_多个目标里任何一个自我申报对不上数字都打回(vocab):
    data = _reflection("猜对了", "是", 0.004, items=[
        _resolved("冷门商品学不动", "是"),
        _resolved("新用户不会做", "是", before=0.05, after=0.05),   # 没动却说治好了
    ])
    with pytest.raises(SchemaViolation, match="自我申报必须跟数字一致"):
        _reflect_validate(vocab, data, targets=["冷门商品学不动", "新用户不会做"])


# ────────────────── 筛卡：按严重度加权 ──────────────────


def test_筛卡_一个重病优先于两个轻病(cards):
    """医生本来就给了 severity，以前这一步只做集合求交，把它扔了。

    同样三个病，只是权重不同，选出来的第一张卡就该不一样：
      治「冷门商品学不动 + 新用户不会做」的卡 → 0.2 + 0.2 = 0.4
      治「在背题」的卡                      → 0.9
    """
    症状 = ["冷门商品学不动", "新用户不会做", "在背题"]

    加权 = cards.match(症状, severity={"冷门商品学不动": 0.2,
                                     "新用户不会做": 0.2,
                                     "在背题": 0.9})
    assert "在背题" in 加权[0].treats                      # 重病的卡排第一

    不加权 = cards.match(症状)                              # 退化成"命中几个病"
    assert len(set(不加权[0].treats) & set(症状)) == 2      # 命中两个的排第一
    assert "在背题" not in 不加权[0].treats                 # 严重度被忽略了


def test_筛卡_不给severity跟以前完全一致(cards):
    症状 = ["冷门商品学不动", "新用户不会做"]
    assert [c.id for c in cards.match(症状)] == [c.id for c in cards.match(症状, severity={})]


def test_一整场_可以指定起步档位(tmp_path):
    """控制台上选了「中份」就该真的从中份起步，不能嘴上说中份、实际跑小份。"""
    llm = ScriptedLLM(promote_on=())
    ex = DriftingExecutor()
    run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("中份"),
        module_interface="", example_module="", current_config="",
        rounds=2, start_fidelity="中份", logs_dir=tmp_path,
    )
    rows = [json.loads(l) for l in (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert all(r["fidelity"] == "中份" for r in rows)


def test_一整场_起步档位写错当场报错(tmp_path):
    ex = DriftingExecutor()
    with pytest.raises(ValueError, match="没有「超大份」这一档"):
        run_session(
            llm=ScriptedLLM(), vocab=SymptomVocab.load(),
            cards=CardLibrary.load(SymptomVocab.load()),
            executor=ex, initial_report=ex.report(), module_interface="",
            example_module="", current_config="", start_fidelity="超大份",
            logs_dir=tmp_path,
        )


# ────────────────── 待议架：军师提过但没轮到的方案 ──────────────────


def _prop(card_id, targets, rank=1):
    return {"rank": rank, "card_id": card_id, "targets": targets,
            "rationale": "冷门桶 0.552 比热门桶 0.638 低 0.086，" * 5,
            "expected": {"点击AUC": 0.0, "购买AUC": 0.003},
            "cost": {"代码难度": "简单", "训练时间倍数": 1.0},
            "risk": "", "novel": not card_id, "how_to": ""}


def test_待议架_只收没被挑中的():
    shelf = Shelf()
    a, b, c = _prop("类目兜底", ["冷门商品学不动"]), _prop("ESMM", ["转化样本偏差"]), _prop("AITM", ["转化样本偏差"])
    shelf.shelve(1, [a, b, c], chosen=a)
    assert {e["card_id"] for e in shelf.entries} == {"ESMM", "AITM"}


def test_待议架_理由只留个引子():
    """存整段推理没意义 —— 军师需要的是"我想过这个"，不是把当时的话再读一遍。"""
    shelf = Shelf()
    long = _prop("ESMM", ["转化样本偏差"])
    shelf.shelve(1, [long], chosen=None)
    assert len(shelf.entries[0]["当时的理由"]) <= 120
    assert len(long["rationale"]) > 120


def test_待议架_试过的卡不再摆出来():
    shelf = Shelf()
    shelf.shelve(1, [_prop("ESMM", ["转化样本偏差"]), _prop("AITM", ["转化样本偏差"])], chosen=None)
    活着的 = shelf.relevant(["转化样本偏差"], exclude_ids={"ESMM"})
    assert [e["card_id"] for e in 活着的] == ["AITM"]


def test_待议架_病没了药也不留():
    """最重要的过期规则：陈旧方案会把军师往回带，让它照着三轮前的诊断开药。"""
    shelf = Shelf()
    shelf.shelve(1, [_prop("ESMM", ["转化样本偏差"])], chosen=None)
    assert shelf.relevant(["转化样本偏差"])            # 这轮还在报这个病 → 留
    assert shelf.relevant(["在背题"]) == []            # 这轮不报了 → 丢


def test_待议架_同一张卡只留最近一次():
    shelf = Shelf()
    shelf.shelve(1, [_prop("ESMM", ["转化样本偏差"])], chosen=None)
    shelf.shelve(5, [_prop("ESMM", ["转化样本偏差"])], chosen=None)
    assert len(shelf.entries) == 1
    assert shelf.entries[0]["提出于第几轮"] == 5


def test_待议架_有上限():
    """喂给军师的上下文不能越滚越大，否则省下的 token 还不够多花的。"""
    shelf = Shelf()
    for i in range(20):
        shelf.shelve(i, [_prop(f"卡{i}", ["转化样本偏差"])], chosen=None)
    assert len(shelf.relevant(["转化样本偏差"])) == SHELF_KEEP


def test_待议架_落盘再读回(tmp_path):
    shelf = Shelf()
    shelf.shelve(1, [_prop("ESMM", ["转化样本偏差"])], chosen=None)
    path = tmp_path / "shelf.json"
    shelf.dump(path)
    assert Shelf.load(path).entries == shelf.entries
    assert Shelf.load(tmp_path / "没有.json").entries == []


def test_待议架_真的摆到军师面前了(tmp_path):
    """跑两轮，第二轮军师收到的材料里必须出现第一轮没被挑中的方案。"""
    看到的 = []

    class _记录军师(ScriptedLLM):
        def _医生(self, schema):
            # 每轮报同一个病 —— 病变了架子本来就该清空（见「病没了药也不留」）
            self._last_findings = [{
                "symptom": "转化样本偏差", "severity": 0.8, "confidence": "高",
                "evidence": "购买模型只用了 click=1 的样本，占比 3.4%", "affects": ["购买AUC"],
            }]
            return {"findings": [dict(self._last_findings[0])],
                    "no_finding": False, "reason_if_none": ""}

        def call(self, **kw):
            if kw["role"] == "军师":
                看到的.append(kw["user"])
            return super().call(**kw)

    llm, ex = _记录军师(promote_on=()), DriftingExecutor()
    run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, logs_dir=tmp_path,
    )
    assert "你以前提过、但还没轮到的方案" not in 看到的[0]     # 第一轮架子是空的
    assert "你以前提过、但还没轮到的方案" in 看到的[1]         # 第二轮摆出来了
    assert (tmp_path / "shelf.json").exists()


def test_待议架_工兵换了备胎就不算没轮到(tmp_path):
    """工兵第一个方案写失败、换备胎成功 —— 那个备胎是被用掉的，不该留在架子上。"""
    llm = ScriptedLLM(faults={"工兵": [1]})       # 第一次实现失败，逼它换备胎
    ex, shelf = DriftingExecutor(), Shelf()
    vocab = SymptomVocab.load()
    log = run_round(
        round_id=1, llm=llm, vocab=vocab, cards=CardLibrary.load(vocab),
        health_report=ex.report("小份"), parent_result=ex.report("小份"),
        executor=ex, scheduler=CostAwareScheduler(),
        module_interface="", example_module="", current_config="",
        shelf=shelf,
    )
    assert log.recoveries                                    # 确实换了备胎
    用掉的 = log.chosen["card_id"]
    assert 用掉的 not in {e["card_id"] for e in shelf.entries}


# ────────────────── 锁定集（R3）──────────────────


def test_锁定集只许读一次(tmp_path):
    """读第二次它就跟开发集一样被污染了 —— 一旦拿它的分数做过决策，
    它就不再是干净的裁判。这条靠代码硬拦，不靠自觉。"""
    from harness.executor import RealExecutor

    ex = RealExecutor.__new__(RealExecutor)      # 不碰真数据，只测守卫
    ex.holdout_path = tmp_path / "holdout"
    ex.holdout_reads = 1                          # 假装已经读过
    with pytest.raises(RuntimeError, match="只许读一次"):
        ex.final_judge("小份")


def test_没配锁定集不算错(tmp_path):
    """没有裁判只是少一份证据，不该让整场跑挂。"""
    from harness.executor import RealExecutor

    ex = RealExecutor.__new__(RealExecutor)
    ex.holdout_path = None
    r = ex.final_judge("小份")
    assert not r.ok and "没有配锁定集" in r.error


def test_泛化落差算的是开发集减锁定集():
    """落差为正 = 开发集分虚高，那部分是反复筛选筛出来的迎合。"""
    from agent.loop import SessionSummary

    s = SessionSummary()
    s.best_scores = {"点击AUC": 0.60, "购买AUC": 0.55}
    s.holdout_scores = {"点击AUC": 0.56, "购买AUC": 0.55}
    assert s.generalization_gap["点击AUC"] == pytest.approx(0.04)
    assert s.generalization_gap["购买AUC"] == pytest.approx(0.0)

    # 没做裁决时不该编一个落差出来
    assert SessionSummary().generalization_gap == {}
# ────────────────── 人工干预：让 0 成为观测值而不是常量 ──────────────────


def test_干预记录_跑之前就有的不算(tmp_path):
    """准备阶段的记录属于搭建，不是"跑起来之后插手"。"""
    path = tmp_path / "interventions.jsonl"
    InterventionLog.record(path, "跑之前改了数据路径")
    log = InterventionLog(path)              # 开跑时初始化，把已有的记成"已知"
    assert log.drain() == []

    InterventionLog.record(path, "第 3 轮撞 OOM，手动把 batch 调小", round_id=3)
    fresh = log.drain()
    assert len(fresh) == 1 and fresh[0]["第几轮"] == 3
    assert log.drain() == []                 # 取过就不再重复计


def test_干预记录_真的会进逐轮日志(tmp_path):
    """跑到一半有人插手，那一轮的日志里必须体现出来 —— 交付物 #3 的要求。"""
    llm, ex = ScriptedLLM(promote_on=()), DriftingExecutor()

    def 第一轮之后插一手(log, summary):
        if log.round_id == 1:
            InterventionLog.record(tmp_path / "interventions.jsonl",
                                   "手动改了学习率", round_id=2)

    summary = run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, logs_dir=tmp_path, on_round=第一轮之后插一手,
    )
    rows = [json.loads(l) for l in (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["interventions"] == 0
    assert rows[1]["interventions"] == 1
    assert rows[1]["intervention_notes"] == ["手动改了学习率"]
    assert summary.interventions == 1        # 结果表里的总数也对得上


# ────────────────── 快照与还原：交付物 #4 ──────────────────


def test_快照_每轮都留一份(tmp_path):
    llm, ex = ScriptedLLM(promote_on=()), DriftingExecutor()
    run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="model:\n  name: mlp\n",
        rounds=3, run_id="测试场", logs_dir=tmp_path,
    )
    snaps = sorted((tmp_path / "snapshots" / "测试场").glob("round_*.json"))
    # 第 0 轮（起点）也留一份 —— 第 1 轮就走差的话得退得回去
    assert [p_.stem for p_ in snaps] == ["round_000", "round_001", "round_002", "round_003"]
    起点 = json.loads(snaps[0].read_text(encoding="utf-8"))
    assert 起点["配置"] and 起点["零件"] == {}      # 起点还没有任何零件
    snap = json.loads(snaps[1].read_text(encoding="utf-8"))
    assert snap["配置"] and snap["零件"]           # 配置文本 + 哪个文件哪轮写的
    assert snap["分数"]["点击AUC"] > 0


def test_快照_记住每个零件是哪一轮写的(tmp_path):
    """同一个路径被后面的轮次覆盖时，早期那一版只能靠这张清单找回来。"""
    llm, ex = ScriptedLLM(promote_on=()), DriftingExecutor()
    run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, run_id="测试场", logs_dir=tmp_path,
    )
    第三轮 = json.loads(
        (tmp_path / "snapshots" / "测试场" / "round_003.json").read_text(encoding="utf-8"))
    # 假工兵每轮写同一个路径 → 第 3 轮的快照该指向第 3 轮那一版
    assert set(第三轮["零件"].values()) == {3}


def test_有效配置_用执行器里真正生效的那份():
    """执行器把改动合并进内存里的 config，磁盘上那份一直是初始状态。"""
    class _带配置的执行器(DriftingExecutor):
        config = {"model": {"name": "esmm"}, "train": {"seed": 7}}

    assert "esmm" in effective_config(_带配置的执行器(), "model:\n  name: mlp\n")
    # 执行器没有 config（假执行器）→ 退回传进来的文本
    assert effective_config(DriftingExecutor(), "model:\n  name: mlp\n") == "model:\n  name: mlp\n"


# ────────────────── 叙事：跨轮的故事线 ──────────────────


def test_叙事_把一整场压成一条线(tmp_path):
    llm, ex = ScriptedLLM(promote_on=()), DriftingExecutor()
    run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, logs_dir=tmp_path,
    )
    text = (tmp_path / "narrative.md").read_text(encoding="utf-8")
    assert "人工干预 0 次" in text
    assert text.count("| 1 |") == 1 and "| 3 |" in text     # 每轮一行
    assert "最终提交第" in text


def test_两个人各跑一次_日志能分得开(tmp_path):
    """日志是追加的、轮次每场都从 1 重数 —— 没有 run_id，两场会糊成
    [1,2,3,1,2,3,4] 这么一串，评委读到的就是一团乱麻。
    """
    v = SymptomVocab.load()
    for who, n in (("队友A", 3), ("队友B", 4)):
        run_session(
            llm=ScriptedLLM(promote_on=()), vocab=v, cards=CardLibrary.load(v),
            executor=DriftingExecutor(), initial_report=DriftingExecutor().report("小份"),
            module_interface="", example_module="", current_config="",
            rounds=n, run_id=who, logs_dir=tmp_path,
        )
    rows = [json.loads(l) for l in
            (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["round_id"] for r in rows] == [1, 2, 3, 1, 2, 3, 4]      # 编号确实会撞
    assert len([r for r in rows if r["run_id"] == "队友A"]) == 3        # 但分得开
    assert len([r for r in rows if r["run_id"] == "队友B"]) == 4
    # 快照也各存各的，不互相覆盖
    assert (tmp_path / "snapshots" / "队友A").is_dir()
    assert (tmp_path / "snapshots" / "队友B").is_dir()


# ────────────────── finalize：一条命令出齐提交包 ──────────────────


def test_整理提交包_只取一场(tmp_path):
    """几个人各跑几次混在一个日志里，提交包必须只含一场，否则轮次编号是乱的。"""
    import argparse as _ap
    from agent import cli

    v = SymptomVocab.load()
    for who, n in (("旧的一场", 2), ("要交的那场", 3)):
        run_session(
            llm=ScriptedLLM(promote_on=()), vocab=v, cards=CardLibrary.load(v),
            executor=DriftingExecutor(), initial_report=DriftingExecutor().report("小份"),
            module_interface="", example_module="", current_config="",
            rounds=n, run_id=who, logs_dir=tmp_path,
        )
    out = tmp_path / "deliverables"
    cli.cmd_finalize(_ap.Namespace(run="要交的那场", out=str(out), logs=str(tmp_path)))

    rows = [json.loads(l) for l in
            (out / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["round_id"] for r in rows] == [1, 2, 3]           # 编号连续可读
    assert {r["run_id"] for r in rows} == {"要交的那场"}
    for name in ("narrative.md", "session_summary.json", "dashboard.html"):
        assert (out / name).exists(), name
    assert (out / "best_pipeline" / "config" / "pipeline.yaml").exists()


def test_整理提交包_默认取最后一场(tmp_path):
    import argparse as _ap
    from agent import cli

    v = SymptomVocab.load()
    for who in ("先跑的", "后跑的"):
        run_session(
            llm=ScriptedLLM(promote_on=()), vocab=v, cards=CardLibrary.load(v),
            executor=DriftingExecutor(), initial_report=DriftingExecutor().report("小份"),
            module_interface="", example_module="", current_config="",
            rounds=2, run_id=who, logs_dir=tmp_path,
        )
    out = tmp_path / "deliverables"
    cli.cmd_finalize(_ap.Namespace(run=None, out=str(out), logs=str(tmp_path)))
    rows = [json.loads(l) for l in
            (out / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    assert {r["run_id"] for r in rows} == {"后跑的"}


def test_整理提交包_场次不存在时说清楚有哪些(tmp_path):
    import argparse as _ap
    from agent import cli

    v = SymptomVocab.load()
    run_session(
        llm=ScriptedLLM(promote_on=()), vocab=v, cards=CardLibrary.load(v),
        executor=DriftingExecutor(), initial_report=DriftingExecutor().report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=2, run_id="真有的那场", logs_dir=tmp_path,
    )
    with pytest.raises(SystemExit, match="真有的那场"):
        cli.cmd_finalize(_ap.Namespace(run="不存在", out=str(tmp_path / "x"),
                                       logs=str(tmp_path)))


def test_复盘官拿的是上一轮而不是上上轮(tmp_path):
    """曾经的 off-by-one：第 3 轮会拿第 1 轮来比，中间两轮的进步全算在这一次头上。

    虚报的收益会传染到卡片信任分、黑名单和升档决策 —— 一处错，全盘失真。
    """
    看到的 = []

    class _记录复盘官(ScriptedLLM):
        def call(self, **kw):
            if kw["role"] == "复盘官":
                看到的.append(kw["user"])
            return super().call(**kw)

    llm, ex = _记录复盘官(promote_on=()), DriftingExecutor()
    run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, logs_dir=tmp_path,
    )
    rows = [json.loads(l) for l in
            (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()]
    第2轮的分 = rows[1]["metrics"]["验证集"]["点击分"]
    第1轮的分 = rows[0]["metrics"]["验证集"]["点击分"]

    # 第 3 轮的复盘材料里，「改动之前那一版」必须是第 2 轮的分，不是第 1 轮的
    第3轮材料 = 看到的[2]
    assert str(第2轮的分) in 第3轮材料.split("## 改动之前那一版")[1]
    assert 第1轮的分 != 第2轮的分                       # 两个数确实不同，测试才有意义


# ────────────────── 锁定集大考：必须考最佳轮，不是末态 ──────────────────


class _带配置的执行器(DriftingExecutor):
    """会记账"最后一次大考用的是哪份配置"的假执行器。"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.config = {"model": {"name": "起始"}}
        self.judged_config = None

    def run(self, patch, fidelity):
        # 模拟工兵的改动叠加：每轮都往配置里盖一层
        self.config = {"model": {"name": f"第{self.runs + 1}轮"}}
        return super().run(patch, fidelity)

    def final_judge(self, fidelity):
        self.judged_config = dict(self.config)
        return RunResult(ok=True, seconds=1.0, fidelity=fidelity,
                         health_report=self.report(fidelity))


def test_大考用的是最佳轮的配置而不是末态(tmp_path):
    """收敛条件是「连续 patience 轮没进步」——最佳轮之后必然还跑了至少 patience 轮，
    末态永远不等于最佳轮。拿末态去考，考的是另一个模型，而机会只有一次。
    """
    ex = _带配置的执行器(gain=0.004, decay=0.05)      # 涨一轮就基本不动了
    summary = run_session(
        llm=ScriptedLLM(promote_on=()), vocab=SymptomVocab.load(),
        cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=8, patience=2, run_id="测试场", logs_dir=tmp_path,
    )
    assert summary.rounds_run > summary.best_round        # 最佳轮之后确实还跑了
    assert ex.judged_config == {"model": {"name": f"第{summary.best_round}轮"}}
    assert summary.holdout_scores                          # 考成了


def test_装不回最佳轮就跳过大考(tmp_path):
    """锁定集只许读一次 —— 宁可没有这个数，也不要一个错的数。"""
    ex = _带配置的执行器()
    summary = run_session(
        llm=ScriptedLLM(promote_on=()), vocab=SymptomVocab.load(),
        cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, run_id="测试场", logs_dir=tmp_path,
    )
    # 把快照删掉，模拟"装不回来"
    import shutil as _sh
    _sh.rmtree(tmp_path / "snapshots")
    ex2 = _带配置的执行器()
    s2 = run_session(
        llm=ScriptedLLM(promote_on=()), vocab=SymptomVocab.load(),
        cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex2, initial_report=ex2.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, run_id="另一场", logs_dir=tmp_path / "空的",
    )
    assert s2.holdout_scores or s2.holdout_note            # 要么考了，要么写明为什么没考


def test_读快照_缺件就整份不认(tmp_path):
    """半个流水线比没有更糟 —— 装一半会让大考考出一个谁也不认识的模型。"""
    from agent.loop import read_snapshot

    (tmp_path / "snapshots" / "某场").mkdir(parents=True)
    (tmp_path / "snapshots" / "某场" / "round_002.json").write_text(json.dumps({
        "轮次": 2, "配置": "model:\n  name: mlp\n",
        "零件": {"modules/features/x.py": 1},          # 说是第 1 轮写的
    }, ensure_ascii=False), encoding="utf-8")
    # 但日志里第 1 轮没有这个文件
    (tmp_path / "rounds.jsonl").write_text(json.dumps({
        "round_id": 1, "run_id": "某场", "patch_files": {}}, ensure_ascii=False) + "\n",
        encoding="utf-8")
    assert read_snapshot(tmp_path, "某场", 2) is None


# ────────────────── 超参数：工兵改了得真的生效，但不能想改多大改多大 ──────────────────


def test_超参数_默认值跟改之前一模一样():
    """接通配置不能顺手改变基线行为，否则历史分数全对不上了。"""
    pytest.importorskip("pandas")
    from harness.executor import lgbm_kwargs

    assert lgbm_kwargs(None) == {
        "n_estimators": 120, "num_leaves": 31, "learning_rate": 0.05,
        "min_child_samples": 20, "colsample_bytree": 1.0,
    }
    # 购买塔的覆盖值也照旧
    cvr = lgbm_kwargs(None, {"n_estimators": 60, "num_leaves": 15})
    assert cvr["n_estimators"] == 60 and cvr["num_leaves"] == 15


def test_超参数_工兵改了真的生效():
    pytest.importorskip("pandas")
    from harness.executor import lgbm_kwargs

    kw = lgbm_kwargs({"n_estimators": 300, "learning_rate": 0.02})
    assert kw["n_estimators"] == 300 and kw["learning_rate"] == 0.02


def test_超参数_越界会被夹回去():
    """护栏不是建议：n_estimators=100000 能让一轮跑到天亮，把整场预算烧光。"""
    pytest.importorskip("pandas")
    from harness.executor import lgbm_kwargs

    assert lgbm_kwargs({"n_estimators": 100000})["n_estimators"] == 2000
    assert lgbm_kwargs({"num_leaves": 1})["num_leaves"] == 4
    assert lgbm_kwargs({"learning_rate": 5.0})["learning_rate"] == 0.5


def test_超参数_写歪了退回默认而不是炸掉():
    """一个配置错字不该让整轮训练报废。"""
    pytest.importorskip("pandas")
    from harness.executor import lgbm_kwargs

    assert lgbm_kwargs({"n_estimators": "很多"})["n_estimators"] == 120
    assert lgbm_kwargs({"learning_rate": None})["learning_rate"] == 0.05


def test_超参数_没在白名单里的键被忽略():
    """工兵只能调这五个 —— 别的键写了也不会被传给 LightGBM。"""
    pytest.importorskip("pandas")
    from harness.executor import lgbm_kwargs

    kw = lgbm_kwargs({"n_jobs": 999, "device": "cuda", "n_estimators": 200})
    assert "n_jobs" not in kw and "device" not in kw
    assert kw["n_estimators"] == 200


# ────────────────── 加特征零件：写进去的文件必须真的被跑起来 ──────────────────


def _写个零件(tmp_path, monkeypatch, body: str) -> None:
    """在临时 ROOT 下放一个零件文件，并把执行器的 ROOT 指过去。"""
    from harness import executor as ex_mod

    monkeypatch.setattr(ex_mod, "ROOT", tmp_path)
    path = tmp_path / "modules" / "features" / "demo.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


零件范本 = '''
class Demo:
    def __init__(self, config):
        self.k = config["features"]["演示零件"]["k"]
        self.seen = None
    def fit(self, train_df):
        self.seen = float(train_df["a"].mean()) * self.k
    def transform(self, df):
        df = df.copy()
        df["新列"] = self.seen
        return df
'''


def test_零件_启用了就真的被加载并跑起来(tmp_path, monkeypatch):
    """这是 ① 的核心：以前文件写进去了，但没有任何机制去加载和运行它。"""
    pd = pytest.importorskip("pandas")
    from harness.executor import apply_feature_ops, load_feature_ops

    _写个零件(tmp_path, monkeypatch, 零件范本)
    cfg = {"features": {"演示零件": {
        "enabled": True, "impl": "modules/features/demo.py", "k": 2}}}

    ops = load_feature_ops(cfg)
    assert [n for n, _ in ops] == ["演示零件"]

    train = pd.DataFrame({"a": [1.0, 3.0]})
    val = pd.DataFrame({"a": [99.0]})
    train, (val,), 新列 = apply_feature_ops(ops, train, [val])
    assert 新列 == ["新列"]
    assert train["新列"].iloc[0] == 4.0          # (1+3)/2 * 2
    assert val["新列"].iloc[0] == 4.0            # 验证集套用训练集的统计量（R2）


def test_零件_没启用的不加载(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    from harness.executor import load_feature_ops

    _写个零件(tmp_path, monkeypatch, 零件范本)
    assert load_feature_ops({"features": {"演示零件": {
        "enabled": False, "impl": "modules/features/demo.py"}}}) == []


def test_零件_启用了却没写impl直接报错(tmp_path, monkeypatch):
    """以前这种情况是**静默无效**：配置改了、文件写了，训练纹丝不动，
    却被记成「这个方案没用」，工兵白挨一次负分。宁可当场炸。"""
    pytest.importorskip("pandas")
    from harness.executor import load_feature_ops

    _写个零件(tmp_path, monkeypatch, 零件范本)
    with pytest.raises(ValueError, match="没写 impl"):
        load_feature_ops({"features": {"演示零件": {"enabled": True}}})


def test_零件_不许从modules之外加载(tmp_path, monkeypatch):
    """放开一寸就等于让 Agent import 任意文件（R5）。"""
    pytest.importorskip("pandas")
    from harness.executor import load_feature_ops

    _写个零件(tmp_path, monkeypatch, 零件范本)
    for 坏路径 in ("harness/executor.py", "modules/../harness/x.py"):
        with pytest.raises(ValueError, match="非法零件路径"):
            load_feature_ops({"features": {"x": {"enabled": True, "impl": 坏路径}}})


def test_零件_文件里没有合格的类会说清楚(tmp_path, monkeypatch):
    pytest.importorskip("pandas")
    from harness.executor import load_feature_ops

    _写个零件(tmp_path, monkeypatch, "class 不合格:\n    pass\n")
    with pytest.raises(TypeError, match="没有实现 FeatureOp 接口"):
        load_feature_ops({"features": {"x": {
            "enabled": True, "impl": "modules/features/demo.py"}}})


def test_零件_fit只看训练集(tmp_path, monkeypatch):
    """R2：读验证集来算统计量 = 作弊，分数虚高，测试集必掉。"""
    pd = pytest.importorskip("pandas")
    from harness.executor import apply_feature_ops, load_feature_ops

    _写个零件(tmp_path, monkeypatch, '''
class Demo:
    def __init__(self, config): self.fit_rows = []
    def fit(self, train_df): self.fit_rows.append(len(train_df))
    def transform(self, df):
        df = df.copy(); df["新列"] = 1; return df
''')
    ops = load_feature_ops({"features": {"x": {
        "enabled": True, "impl": "modules/features/demo.py"}}})
    train = pd.DataFrame({"a": [1, 2, 3]})
    val = pd.DataFrame({"a": [9]})
    apply_feature_ops(ops, train, [val])
    assert ops[0][1].fit_rows == [3]             # 只 fit 过训练集，一次


# ── 执行器能不能兑现配置里承诺的东西 ────────────────────────────────
#
# 背景：26 张卡里只有 3 张是纯特征卡，其余 23 张落在 模型/损失函数/训练策略。
# 而执行器走的是 LightGBM 这条路 —— ModelOp / TrainOp 两类零件没有任何加载机制，
# TrainOp 的接口（按 epoch 回调 + state_dict）跟 LightGBM 结构上也对不上。
#
# 如果这些配置被"接受但无视"，就会重演那个最贵的 bug：
# 工兵改了、跑完了、分数纹丝不动，复盘官判「猜错了」，好方法被拉黑。
# 所以：能真做的真做，做不了的**当场炸**。


def test_能力check_默认配置跑得通():
    """仓库里那份 config/pipeline.yaml 必须是执行器兑现得了的。"""
    from harness.executor import check_supported

    cfg = (pathlib.Path(__file__).resolve().parent.parent
           / "config" / "pipeline.yaml")
    check_supported(yaml.safe_load(cfg.read_text(encoding="utf-8")))


def test_能力check_换深度模型要当场炸():
    """model.name 换成 deepfm 但执行器还是 LightGBM —— 静默无视等于骗自己。"""
    from harness.executor import check_supported

    with pytest.raises(ValueError, match="deepfm"):
        check_supported({"model": {"name": "deepfm"}})


def test_能力check_报错要说清为什么和怎么办():
    """错误信息得让人（和下一轮的军师）知道这不是"方法不行"，是"跑不了"。

    深度路径接通之后，esmm 不再是「整类不支持」—— 它只是还没人指路。
    所以报错要说清缺的是什么、怎么补，而不是笼统一句"只会 LightGBM"。
    """
    from harness.executor import check_supported

    with pytest.raises(ValueError) as e:
        check_supported({"model": {"name": "esmm"}})
    报错 = str(e.value)
    assert "impl" in 报错 and "modules/models/" in 报错      # 缺什么、放哪
    assert "ModelOp" in 报错                                 # 照哪个接口写

    # 指了路就放行 —— 剩下的错留给加载时报（文件不存在之类）
    check_supported({"model": {"name": "esmm", "impl": "modules/models/esmm.py"}})


def test_能力check_epoch类零件要当场炸():
    """SWA 要按 epoch 平权重，LightGBM 没有 epoch 循环可以挂。"""
    from harness.executor import check_supported

    with pytest.raises(ValueError, match="SWA|swa"):
        check_supported({"train": {"swa": {"enabled": True}}})


def test_能力check_关着的epoch类零件不算错():
    """enabled: false 就是没开，不该拦。"""
    from harness.executor import check_supported

    check_supported({"train": {"swa": {"enabled": False}}})


def test_能力check_多任务损失权重要当场炸():
    """loss_weight 是给「一个模型同时学两件事」用的；
    这里点击和购买是两个独立的 LightGBM，权重无处可施。"""
    from harness.executor import check_supported

    with pytest.raises(ValueError, match="uncertainty"):
        check_supported({"train": {"loss_weight": {"strategy": "uncertainty"}}})


def test_能力check_固定权重不算错():
    from harness.executor import check_supported

    check_supported({"train": {"loss_weight": {"strategy": "fixed",
                                               "ctr": 1.0, "cvr": 1.0}}})


# ── 负采样：概率要还原回真实尺度 ──────────────────────────────────


def test_负采样_没开就原样返回():
    from harness.executor import recalibrate

    p = [0.1, 0.5, 0.9]
    assert recalibrate(p, keep_ratio=1.0) == pytest.approx(p)


def test_负采样_还原后概率变小():
    """负样本被抽掉之后模型看到的正样本比例虚高，预测的概率整体偏大。
    还原就是把它压回真实尺度。"""
    from harness.executor import recalibrate

    out = recalibrate([0.5], keep_ratio=0.1)
    assert out[0] < 0.5
    # w*p / (1-p+w*p) = 0.05/0.55
    assert out[0] == pytest.approx(0.05 / 0.55)


def test_负采样_还原不改变排序():
    """AUC 只看排序 —— 还原是单调变换，所以 AUC 不该被它改动。
    它影响的是 logloss 和「预测均值对不对得上真实点击率」。"""
    from harness.executor import recalibrate

    out = recalibrate([0.2, 0.4, 0.8], keep_ratio=0.3)
    assert out == sorted(out)


def test_负采样_边界值不炸():
    from harness.executor import recalibrate

    assert recalibrate([0.0, 1.0], keep_ratio=0.5) == pytest.approx([0.0, 1.0])


# ── 「跑不了」和「方法不行」是两回事 ──────────────────────────────


def test_兑现不了的配置不该扣卡片的信任分():
    """执行器兑现不了 ≠ 这张卡不靠谱。

    前者是我们的流水线缺能力，后者是方法本身没用。混为一谈的话，
    一场跑下来会把 ESMM、DeepFM 这些真正的好方法全部扣成低信任分，
    下一场开跑时军师就再也不会提它们了 —— 错误结论被固化进账本。
    """
    from harness.executor import UnsupportedByExecutor, check_supported

    with pytest.raises(UnsupportedByExecutor):
        check_supported({"model": {"name": "deepfm"}})


def test_兑现不了也是一种跑不起来():
    """仍然是 ValueError 的子类 —— 老代码里 except ValueError 的地方不会漏接。"""
    from harness.executor import UnsupportedByExecutor

    assert issubclass(UnsupportedByExecutor, ValueError)


def test_跑挂了的结果会标出是不是兑现不了():
    from agent.loop import RunResult

    assert RunResult(ok=False).unsupported is False       # 默认不影响老代码
    assert RunResult(ok=False, unsupported=True).unsupported is True


# ── 执行器整条路跑一遍（造几百行假数据，秒级）────────────────────
#
# 这一组是补一个真实存在过的窟窿：_build_report 曾经引用了两个
# 属于**另一个函数**的局部变量（ctr_kw / cvr_kw），任何一次真训练都会
# NameError 当场崩 —— 但当时全部单元测试都是绿的，因为没有一个测试
# 真的走过 _train_and_score。纯函数测得再细也发现不了这类断裂。


def _造数据(tmp_path, n=400):
    """造一份最小可训练数据：两个类别特征 + 点击 + 转化。"""
    pd = pytest.importorskip("pandas")
    need("lightgbm")
    import numpy as np

    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "sample_id": range(n),
        "101": rng.integers(0, 7, n),
        "205": rng.integers(0, 5, n),
    })
    # 让标签跟特征有点关系，否则 AUC 恒等于 0.5，测不出东西
    df["click"] = ((df["101"] % 3 == 0) ^ (rng.random(n) < 0.25)).astype(int)
    df["conversion"] = (df["click"] & (df["205"] % 2 == 0)
                        & (rng.random(n) < 0.7)).astype(int)
    train_p, val_p = tmp_path / "train.parquet", tmp_path / "val.parquet"
    df.iloc[:300].to_parquet(train_p)
    df.iloc[300:].to_parquet(val_p)
    return str(train_p), str(val_p)


def _配置(**train_over):
    cfg = {"features": {"base_fields": ["101", "205"]},
           "model": {"name": "lightgbm",
                     "lightgbm": {"n_estimators": 20, "num_leaves": 7}},
           "train": {"seed": 1}}
    cfg["train"].update(train_over)
    return cfg


def test_整条路_基线跑得完并出成绩单(tmp_path):
    need("lightgbm")
    from harness.executor import RealExecutor

    tr, va = _造数据(tmp_path)
    r = RealExecutor(tr, va, seed=1, config=_配置()).run(
        {"new_files": [], "config_patch": {}}, "全量")
    assert r.ok, r.error
    assert r.health_report["验证集"]["点击分"] is not None
    # 这一条就是当初那个 NameError 的哨兵
    assert r.health_report["实际超参数"]["点击塔"]["n_estimators"] == 20


def test_整条路_改超参数真的会传到模型上(tmp_path):
    """R7：配置里写的数必须真的进模型，不是摆设。"""
    need("lightgbm")
    from harness.executor import RealExecutor

    tr, va = _造数据(tmp_path)
    cfg = _配置()
    cfg["model"]["lightgbm"]["n_estimators"] = 33
    r = RealExecutor(tr, va, seed=1, config=cfg).run(
        {"new_files": [], "config_patch": {}}, "全量")
    assert r.ok, r.error
    assert r.health_report["实际超参数"]["点击塔"]["n_estimators"] == 33


def test_整条路_开早停跑得完(tmp_path):
    """早停从训练集内部切裁判，不碰被评的那份数据（R2/R3）。"""
    need("lightgbm")
    from harness.executor import RealExecutor

    tr, va = _造数据(tmp_path)
    cfg = _配置(early_stopping={"enabled": True, "patience": 2,
                               "inner_holdout_frac": 0.2})
    cfg["model"]["lightgbm"]["n_estimators"] = 200
    r = RealExecutor(tr, va, seed=1, config=cfg).run(
        {"new_files": [], "config_patch": {}}, "全量")
    assert r.ok, r.error


def test_整条路_开负采样跑得完且训练集变小(tmp_path):
    need("lightgbm")
    from harness.executor import RealExecutor

    tr, va = _造数据(tmp_path)
    满 = RealExecutor(tr, va, seed=1, config=_配置()).run(
        {"new_files": [], "config_patch": {}}, "全量")
    抽 = RealExecutor(tr, va, seed=1, config=_配置(
        negative_sampling={"enabled": True, "keep_ratio": 0.3})).run(
        {"new_files": [], "config_patch": {}}, "全量")
    assert 满.ok and 抽.ok, (满.error, 抽.error)
    assert 抽.health_report["训练集"]["总行数"] < 满.health_report["训练集"]["总行数"]


def test_整条路_兑现不了的配置标成unsupported(tmp_path):
    """跑不了要跟跑崩了分开，否则会扣错卡片的信任分。"""
    need("lightgbm")
    from harness.executor import RealExecutor

    tr, va = _造数据(tmp_path)
    cfg = _配置()
    cfg["model"]["name"] = "deepfm"
    r = RealExecutor(tr, va, seed=1, config=cfg).run(
        {"new_files": [], "config_patch": {}}, "全量")
    assert not r.ok
    assert r.unsupported is True


def test_整条路_训练崩了不算unsupported(tmp_path):
    """普通的崩溃仍然该扣分 —— 别把两种失败混成一种。"""
    need("lightgbm")
    from harness.executor import RealExecutor

    tr, va = _造数据(tmp_path)
    cfg = _配置()
    cfg["features"]["base_fields"] = ["根本不存在的字段"]
    r = RealExecutor(tr, va, seed=1, config=cfg).run(
        {"new_files": [], "config_patch": {}}, "全量")
    assert not r.ok
    assert r.unsupported is False


# ── 噪声门槛必须分指标，不能一个标量管两个 ────────────────────────
#
# 实测：验证集里点击正样本 8,950 个，转化正样本只有 38 个。
# 购买 AUC 的抖动比点击大一个数量级 —— 一个转化样本换个排位，
# 购买分就能动 1/38 ≈ 0.026，而点击分动一下要 8,950 个样本一起使劲。
#
# 以前 summarize() 取 max(点击带, 购买带) 当唯一门槛，被购买带主导，
# 于是两头都错：真实的点击提升（+0.008 这种，已实测到过）被当噪声抹掉，
# 购买分的纯抖动（±0.05）反而越过门槛被记成"猜对了"，白送 +0.15 信任分。


def test_噪声带_分指标各给各的():
    from agent import noise

    reports = [{"验证集": {"点击分": 0.550 + i * 0.0005,
                          "购买分": 0.45 + i * 0.03}} for i in range(3)]
    bands = noise.summarize(reports, seeds=[1, 2, 3])
    分 = bands["分指标噪声带"]
    assert 分["点击AUC"] < 分["购买AUC"]          # 购买抖得多，门槛就该高
    assert 分["点击AUC"] == bands["点击分"]["噪声带"]
    assert 分["购买AUC"] == bands["购买分"]["噪声带"]


def test_噪声带_老的单指标字段还在():
    """别把已经在用它的地方弄挂了。"""
    from agent import noise

    reports = [{"验证集": {"点击分": 0.55, "购买分": 0.45}} for _ in range(3)]
    assert "单指标噪声带" in noise.summarize(reports, seeds=[1, 2, 3])


def test_越过噪声_点击涨了就该算数():
    """+0.008 的点击提升是真的（实测加 3 个交叉特征就有 +0.0075），
    不该因为购买分抖得凶而被一起否掉。"""
    from agent.loop import beats_noise

    assert beats_noise({"点击AUC": 0.008, "购买AUC": 0.0},
                       {"点击AUC": 0.001, "购买AUC": 0.05}) is True


def test_越过噪声_购买分的抖动不该算数():
    """38 个正样本上下抖 0.05 是常态，不是本事。"""
    from agent.loop import beats_noise

    assert beats_noise({"点击AUC": 0.0001, "购买AUC": 0.05},
                       {"点击AUC": 0.001, "购买AUC": 0.09}) is False


def test_越过噪声_没量过噪声就退回R11门槛():
    from agent.loop import beats_noise

    assert beats_noise({"点击AUC": 0.01}, None) is True
    assert beats_noise({"点击AUC": 0.0001}, None) is False


def test_噪声带_测出0要退回理论值():
    """保真度抽样只抽负样本，click=1 子集在每个种子下完全一样 ——
    换种子扰动不到购买塔，测出来的噪声带是 0.0000。
    照单全收的话购买分任何抖动都能越过门槛，比不分指标还糟。"""
    from agent import noise

    reports = [{"验证集": {"总行数": 217974, "点击数": 8950, "转化数": 38,
                          "点击分": 0.5507, "购买分": 0.4462}} for _ in range(3)]
    分 = noise.summarize(reports, seeds=[1, 2, 3])["分指标噪声带"]
    assert 分["购买AUC"] > 0.05, "38 个正样本的理论带该在 ±0.09 量级"


def test_噪声带_测得出来就用实测的():
    """理论带是兜底，不该盖掉真正测出来的抖动。"""
    from agent import noise

    reports = [{"验证集": {"总行数": 217974, "点击数": 8950, "转化数": 38,
                          "点击分": 0.55 + i * 0.001, "购买分": 0.45 + i * 0.002}}
               for i in range(3)]
    分 = noise.summarize(reports, seeds=[1, 2, 3])["分指标噪声带"]
    assert 分["购买AUC"] < 0.05          # 实测出了抖动，就不该退回 0.09 的理论带


def test_事件流_可以改道到别处(tmp_path, monkeypatch):
    """看板的事件流要能改道，否则测试会往真日志里灌假事件。"""
    from agent import events

    target = tmp_path / "ev.jsonl"
    monkeypatch.setenv("AGENT_EVENTS_PATH", str(target))
    events.emit("phase", name="测试用", detail="不该进真日志")
    assert "测试用" in target.read_text(encoding="utf-8")


def test_事件流_设成空就谁也不写(tmp_path, monkeypatch):
    """测试套件用的就是这一档（见 tests/conftest.py）。"""
    from agent import events

    monkeypatch.setenv("AGENT_EVENTS_PATH", "")
    before = events.EVENTS_PATH.stat().st_size if events.EVENTS_PATH.exists() else 0
    events.emit("phase", name="绝对不该出现在任何文件里")
    after = events.EVENTS_PATH.stat().st_size if events.EVENTS_PATH.exists() else 0
    assert after == before


# ── 导出预测（交付物 #4 的原料）────────────────────────────────


def _造无标签测试集(tmp_path, n=120):
    pd = pytest.importorskip("pandas")
    import numpy as np
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"sample_id": range(10000, 10000 + n),
                       "101": rng.integers(0, 7, n),
                       "205": rng.integers(0, 5, n)})
    p = tmp_path / "test_nolabel.parquet"
    df.to_parquet(p)
    return str(p)


def test_预测_测试集没有标签也能出预测(tmp_path):
    """真测试集就是没标签的 —— 这条要是不通，交付物 #4 根本交不出来。"""
    need("lightgbm")
    from harness.executor import RealExecutor

    tr, va = _造数据(tmp_path)
    te = _造无标签测试集(tmp_path)
    out = RealExecutor(tr, va, seed=1, config=_配置()).predict_frame(te, "全量")
    assert len(out) == 120
    assert list(out.columns) == ["sample_id", "ctr", "cvr", "ctcvr"]


def test_预测_概率都在0和1之间(tmp_path):
    need("lightgbm")
    from harness.executor import RealExecutor

    tr, va = _造数据(tmp_path)
    te = _造无标签测试集(tmp_path)
    out = RealExecutor(tr, va, seed=1, config=_配置()).predict_frame(te, "全量")
    for col in ("ctr", "cvr", "ctcvr"):
        assert out[col].between(0, 1).all(), f"{col} 跑到 [0,1] 外面去了"


def test_预测_ctcvr是两者相乘(tmp_path):
    """口径写死：cvr 是 P(购买|点击) 这个条件概率，ctcvr 才是 P(点击且购买)。
    两者混用是这个赛题里最容易犯的错（见 ESMM 卡片）。"""
    need("lightgbm")
    from harness.executor import RealExecutor

    tr, va = _造数据(tmp_path)
    te = _造无标签测试集(tmp_path)
    out = RealExecutor(tr, va, seed=1, config=_配置()).predict_frame(te, "全量")
    assert (out["ctcvr"] - out["ctr"] * out["cvr"]).abs().max() < 1e-12


def test_预测_sample_id原样带出来(tmp_path):
    """对不上行就等于没交 —— 顺序和取值都不许动。"""
    need("lightgbm")
    from harness.executor import RealExecutor

    tr, va = _造数据(tmp_path)
    te = _造无标签测试集(tmp_path)
    out = RealExecutor(tr, va, seed=1, config=_配置()).predict_frame(te, "全量")
    assert out["sample_id"].tolist() == list(range(10000, 10120))


def test_预测_没有sample_id要当场报错(tmp_path):
    need("lightgbm")
    from harness.executor import RealExecutor

    pd = pytest.importorskip("pandas")
    tr, va = _造数据(tmp_path)
    bad = tmp_path / "no_id.parquet"
    pd.DataFrame({"101": [1, 2], "205": [3, 4]}).to_parquet(bad)
    with pytest.raises(ValueError, match="sample_id"):
        RealExecutor(tr, va, seed=1, config=_配置()).predict_frame(str(bad), "全量")


def test_预测_跟评分走同一条路(tmp_path):
    """同一份数据，predict_frame 出的 ctr 必须跟评分时用的是同一个模型。

    这条是防"两条路慢慢走岔" —— 导出预测要是自己复制一份训练流程，
    加特征零件、数组拍平、负采样还原任何一步改了单边，
    交上去的预测就跟验证时评的不是同一个模型，而且**分数完全正常**，
    根本看不出来。
    """
    need("lightgbm")
    from harness.executor import RealExecutor
    from sklearn.metrics import roc_auc_score

    tr, va = _造数据(tmp_path)
    ex = RealExecutor(tr, va, seed=1, config=_配置())
    report = ex.run({"new_files": [], "config_patch": {}}, "全量").health_report
    out = ex.predict_frame(va, "全量")          # 拿验证集当"测试集"再走一遍

    import pandas as pd
    truth = pd.read_parquet(va)[["sample_id", "click"]]
    m = out.merge(truth, on="sample_id")
    assert roc_auc_score(m["click"], m["ctr"]) == pytest.approx(
        report["验证集"]["点击分"], abs=1e-9)


# ────────────────── 审查修掉的问题：回归锁 ──────────────────


def test_落盘失败不该拖垮整场(tmp_path, monkeypatch):
    """四个角色都有保险丝，记账反而没有 —— 那是最讽刺的死法。

    这段跑在昂贵的训练之后：磁盘满、某个字段序列化不了，
    整场挂在这里，连刚跑完那一轮的成果一起丢。
    """
    from agent import loop as loop_mod

    def 写不了(*a, **kw):
        raise OSError("磁盘满了")

    monkeypatch.setattr(loop_mod, "snapshot_round", 写不了)
    llm, ex = ScriptedLLM(promote_on=()), DriftingExecutor()
    summary = run_session(
        llm=llm, vocab=SymptomVocab.load(), cards=CardLibrary.load(SymptomVocab.load()),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, logs_dir=tmp_path,
    )
    assert summary.rounds_run == 3                      # 照跑不误
    assert summary.recoveries >= 3                      # 但每次都记了一笔


def test_噪声带_按样本量缩放到新档位():
    """升档只换数据量重训，不重测噪声带 —— 正样本一多抖动就小，
    沿用起步档位的带子是一把过松的尺子。"""
    from agent import noise

    小份 = [{"保真度": "小份", "验证集": {"总行数": 218000, "点击数": 8950,
                                    "转化数": 38, "点击分": 0.61, "购买分": p}}
           for p in (0.56, 0.60, 0.52)]
    bands = noise.summarize(小份, [1, 2, 3])
    assert bands["样本量"]["转化数"] == 38

    放大 = noise.rescale(bands, {"保真度": "全量",
                               "验证集": {"总行数": 2180000, "点击数": 89500, "转化数": 467}})
    # 转化样本从 38 涨到 467，购买带必须明显收窄
    assert 放大["分指标噪声带"]["购买AUC"] < bands["分指标噪声带"]["购买AUC"] / 2
    assert 放大["分指标噪声带"]["点击AUC"] < bands["分指标噪声带"]["点击AUC"]
    assert "缩放" in 放大["缩放说明"]


def test_噪声带_缺样本量时原样返回并说明():
    """悄悄用一把错的尺子，比没有尺子更糟。"""
    from agent import noise

    bands = {"分指标噪声带": {"点击AUC": 0.002}, "样本量": {}}
    out = noise.rescale(bands, {"验证集": {}})
    assert out["分指标噪声带"] == bands["分指标噪声带"]
    assert "没做缩放" in out["缩放说明"]


# ── 大数据集不能一次性读进内存（真撞过 OOM）──────────────────────
#
# large_25pct 的 train+public_test 加起来两千多万行，predict_frame 整份
# 读进 16GB 的机器直接被系统强制杀掉（退出码 137）。修法：按可用内存
# 分批读目标数据，模型只训一次，分批的只是"喂给它做预测的数据"。


def _造多分片数据(tmp_path, dirname, n_shards=5, rows_per_shard=20):
    """造一个"分片目录"——几个小 parquet 文件，模拟真实的 256 片布局。"""
    pd = pytest.importorskip("pandas")
    import numpy as np
    d = tmp_path / dirname
    d.mkdir()
    rng = np.random.default_rng(3)
    for i in range(n_shards):
        df = pd.DataFrame({
            "sample_id": range(i * rows_per_shard, (i + 1) * rows_per_shard),
            "101": rng.integers(0, 7, rows_per_shard),
            "205": rng.integers(0, 5, rows_per_shard),
        })
        df.to_parquet(d / f"part-{i:04d}.parquet")
    return d


def test_分批读取_内存预算够大时一批读完(tmp_path):
    from harness.executor import read_in_batches

    d = _造多分片数据(tmp_path, "big_enough")
    批次 = list(read_in_batches(d, budget_bytes=10 * 1024 ** 3))  # 10GB 预算，随便装
    assert len(批次) == 1
    assert len(批次[0]) == 5 * 20


def test_分批读取_预算小就拆成多批(tmp_path):
    from harness.executor import read_in_batches

    d = _造多分片数据(tmp_path, "tight")
    # 量出单个分片的真实大小，预算刚好够 2 个分片
    import pandas as pd
    one = pd.read_parquet(sorted(d.glob("*.parquet"))[0])
    per_shard = int(one.memory_usage(deep=True).sum())
    批次 = list(read_in_batches(d, budget_bytes=per_shard * 2))
    assert len(批次) > 1, "预算收紧了却还是一批读完，分批没生效"
    assert sum(len(b) for b in 批次) == 100          # 行一行没丢


def test_分批读取_合起来跟不分批结果一样(tmp_path):
    """分几批是内存策略，不该影响"读到了什么"——总行数、sample_id 集合必须一致。"""
    from harness.executor import read_in_batches
    import pandas as pd

    d = _造多分片数据(tmp_path, "consistency")
    整批 = pd.concat(list(read_in_batches(d, budget_bytes=10 * 1024 ** 3)), ignore_index=True)
    分批 = pd.concat(list(read_in_batches(d, budget_bytes=1024)), ignore_index=True)  # 预算小到逼近逐片读
    assert set(整批["sample_id"]) == set(分批["sample_id"])
    assert len(整批) == len(分批)


def test_分批读取_单个文件不是目录也能读(tmp_path):
    """不是分片数据集时（单个 parquet 文件），整份当一批——绕不开这个限制，
    但至少不能报错或漏数据。"""
    pd = pytest.importorskip("pandas")
    from harness.executor import read_in_batches

    p = tmp_path / "single.parquet"
    pd.DataFrame({"sample_id": range(10), "101": range(10)}).to_parquet(p)
    批次 = list(read_in_batches(p, budget_bytes=1))  # 预算给到 1 字节也没用，文件不可再拆
    assert len(批次) == 1
    assert len(批次[0]) == 10


def test_可用内存_装不上psutil就退回保守默认(monkeypatch):
    import builtins
    from harness import executor

    真实import = builtins.__import__

    def 假装没装(name, *a, **kw):
        if name == "psutil":
            raise ImportError("模拟这台机器没装 psutil")
        return 真实import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", 假装没装)
    assert executor.available_memory_bytes() == 2 * 1024 ** 3


def test_预测_大数据集分批也能出完整预测(tmp_path):
    """端到端验证：目标数据分成好几个小分片，跟单个大文件相比，
    分批处理不能丢行、不能变模型——这是真正防回归的那道关卡。"""
    need("lightgbm")
    from harness.executor import RealExecutor

    tr, va = _造数据(tmp_path)
    多分片测试集 = _造多分片数据(tmp_path, "predict_target", n_shards=4, rows_per_shard=15)

    ex = RealExecutor(tr, va, seed=1, config=_配置())
    # 逼它分很多批：预算小到只够一两个分片
    import harness.executor as _exec_mod
    原始函数 = _exec_mod.available_memory_bytes
    try:
        _exec_mod.available_memory_bytes = lambda: 200 * 1024   # 极小，逼出多批
        out = ex.predict_frame(多分片测试集, "全量")
    finally:
        _exec_mod.available_memory_bytes = 原始函数

    assert len(out) == 60                               # 4 片 × 15 行，一行不丢
    assert set(out["sample_id"]) == set(range(60))
    assert out["ctr"].between(0, 1).all()
# ────────────────── 深度模型训练路径 ──────────────────


def _造数据(tmp_path, n_train=1500, n_val=600):
    """带真实信号的合成数据 —— 没信号的话 AUC 恒等 0.5，测不出训练有没有起作用。"""
    pd = pytest.importorskip("pandas")
    import numpy as np

    rng = np.random.default_rng(0)

    def 造(n):
        u, i = rng.integers(0, 50, n), rng.integers(0, 30, n)
        p = 1 / (1 + np.exp(-(u / 50 + i / 30 - 1)))
        click = (rng.random(n) < p).astype(int)
        conv = ((click == 1) & (rng.random(n) < 0.3)).astype(int)
        return pd.DataFrame({"sample_id": np.arange(n), "101": u, "205": i,
                             "click": click, "conversion": conv})

    造(n_train).to_parquet(tmp_path / "train.parquet")
    造(n_val).to_parquet(tmp_path / "val.parquet")
    return str(tmp_path / "train.parquet"), str(tmp_path / "val.parquet")


def _深度配置(**deep):
    return {"features": {"base_fields": ["101", "205"]},
            "model": {"name": "mlp", "impl": "modules/models/mlp.py",
                      "mlp": {"hidden": [16], "tower": [8], "dropout": 0.0},
                      "deep": {"epochs": 3, "batch_size": 256,
                               "learning_rate": 0.02, **deep}}}


def test_深度路径_训得起来并出成绩单(tmp_path):
    """21 张卡卡在这条路上：模型类卡本身就是神经网络，
    损失函数类卡只有在梯度下降里才存在，训练策略类卡要挂在 epoch 边界上。"""
    need("torch")
    from harness.executor import RealExecutor

    train, val = _造数据(tmp_path)
    ex = RealExecutor(train, val, seed=7, config=_深度配置())
    r = ex.run({"new_files": [], "config_patch": ""}, "全量")
    assert r.ok, r.error
    rep = r.health_report
    assert 0.0 < rep["验证集"]["点击分"] < 1.0
    assert rep["训练集"]["点击分"] is not None        # 医生判「在背题」要用它
    dt = rep["深度训练"]
    assert dt["训练轮数"] == 3 and dt["最佳轮次"] >= 1
    assert [e["轮"] for e in dt["每轮"]] == [1, 2, 3]  # 每轮曲线进日志


def test_深度路径_不需要lightgbm(tmp_path, monkeypatch):
    """深度路径用不到 LightGBM，不该因为这台机器装不上它就跑不了。"""
    need("torch")
    from harness.executor import RealExecutor

    train, val = _造数据(tmp_path, 400, 200)
    ex = RealExecutor(train, val, seed=7, config=_深度配置(epochs=1))
    # 把 lightgbm 变成一 import 就炸，模拟缺 libomp 的机器
    import builtins
    真import = builtins.__import__

    def 拦(name, *a, **kw):
        if name.startswith("lightgbm"):
            raise OSError("（演习）libomp 缺失")
        return 真import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", 拦)
    assert ex.run({"new_files": [], "config_patch": ""}, "全量").ok


def test_深度路径_超参数从配置读(tmp_path):
    need("torch")
    from harness.executor import RealExecutor

    train, val = _造数据(tmp_path, 400, 200)
    ex = RealExecutor(train, val, seed=7, config=_深度配置(epochs=2))
    r = ex.run({"new_files": [], "config_patch": ""}, "全量")
    assert r.health_report["深度训练"]["训练轮数"] == 2


def test_深度路径_超参数越界被夹回():
    """epochs: 9999 能把一晚上的算力烧光，而 Agent 自己看不出是它干的。"""
    need("torch")
    from harness.deep import deep_kwargs

    assert deep_kwargs({"epochs": 9999})["epochs"] == 50
    assert deep_kwargs({"learning_rate": 10})["learning_rate"] == 0.1
    assert deep_kwargs({"epochs": "很多"})["epochs"] == 5       # 写歪了退回默认


def test_深度路径_缺impl当场炸而不是悄悄换模型(tmp_path):
    """静默无视是最坑的形态：一切看起来都跑了，结论却是假的。"""
    need("torch")
    from harness.executor import RealExecutor

    cfg = _深度配置()
    cfg["model"].pop("impl")
    train, val = _造数据(tmp_path, 300, 150)
    r = RealExecutor(train, val, seed=7, config=cfg).run(
        {"new_files": [], "config_patch": ""}, "全量")
    assert not r.ok and r.unsupported             # 是「跑不了」不是「方法不行」
    assert "impl" in r.error and "modules/models/" in r.error


def test_深度路径_ID词表只在训练集上建():
    """R2：验证集里没见过的 ID 落 OOV 槽，而不是给它一个新编号 ——
    后者等于偷看了验证集里有哪些 ID。"""
    need("torch")
    pd = pytest.importorskip("pandas")
    from harness.deep import OOV, Vocab

    train = pd.DataFrame({"101": [1, 2, 3]})
    val = pd.DataFrame({"101": [2, 999]})          # 999 训练集里没有
    v = Vocab().fit(train, ["101"])
    assert v.sizes() == {"101": 4}                 # 3 个 ID + 1 个 OOV 槽
    assert v.encode(val)[1][0] == OOV


def test_深度路径_导出预测走同一条路(tmp_path):
    need("torch")
    from harness.executor import RealExecutor

    train, val = _造数据(tmp_path, 500, 250)
    ex = RealExecutor(train, val, seed=7, config=_深度配置(epochs=1))
    out = ex.predict_frame(val, "全量")
    assert list(out.columns) == ["sample_id", "ctr", "cvr", "ctcvr"]
    assert len(out) == 250
    assert ((out.ctr >= 0) & (out.ctr <= 1)).all()


def test_深度路径_epoch类训练策略不再被拒():
    """早停、SWA 这些是挂在 epoch 边界的回调 —— LightGBM 那条路没有轮次，
    深度路径有，就该放行。"""
    need("torch")
    from harness.executor import check_supported

    深度 = {"model": {"name": "mlp", "impl": "modules/models/mlp.py"},
           "train": {"swa": {"enabled": True}}}
    check_supported(深度)                                   # 不该抛

    with pytest.raises(ValueError, match="没有 epoch 循环"):
        check_supported({"model": {"name": "lightgbm"}, "train": {"swa": {"enabled": True}}})


# ────────────────── 军师自己写落地草图 ──────────────────


def _提案(card_id="类目兜底", how_to="在 features 下新开一块，206 做兜底键"):
    return {"rank": 1, "card_id": card_id, "targets": ["冷门商品学不动"],
            "rationale": "冷门桶 0.552 比热门桶 0.638 低 0.086",
            "expected": {"点击AUC": 0.0, "购买AUC": 0.003},
            "cost": {"代码难度": "简单", "训练时间倍数": 1.0},
            "risk": "", "novel": not card_id, "how_to": how_to}


def _军师校验(vocab, cards, data):
    captured = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            captured["validate"] = kw["validate"]
            return data

    候选 = [cards.get("类目兜底")]
    roles.propose(_FakeLLM(), vocab, [{"symptom": "冷门商品学不动"}], 候选)
    captured["validate"](data)


def test_军师_用现成卡片也要写落地草图(vocab, cards):
    """卡片的 how_to 是从论文来的通用知识；这里要的是 Agent 针对当前流水线的适配 ——
    那正是评分标准里「识别出什么值得尝试的、以及为什么」要看的东西。"""
    with pytest.raises(SchemaViolation, match="没写 how_to"):
        _军师校验(vocab, cards, {"proposals": [_提案(how_to="  ")]})

    _军师校验(vocab, cards, {"proposals": [_提案()]})       # 写了就放行


def test_军师_照抄卡片原文会被打回(vocab, cards):
    """工兵本来就会看到那张卡，重复一遍只是白烧 token。"""
    原文 = cards.get("类目兜底").how_to
    with pytest.raises(SchemaViolation, match="照抄卡片"):
        _军师校验(vocab, cards, {"proposals": [_提案(how_to=原文[:60])]})


def test_工兵_两份草图都拿得到(cards):
    """以前是「军师的 or 卡片的」，二选一 —— 无论丢哪份都少了一半信息。"""
    看到的 = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            看到的["user"] = kw["user"]
            return {"change_type": "只改配置", "config_patch": "",
                    "new_files": [], "self_check": BASE_CHECK}

    卡 = cards.get("类目兜底")
    roles.implement(_FakeLLM(), _提案(how_to="用 206 当兜底键，K 取 20"),
                    卡, "", "", "")
    材料 = 看到的["user"]
    assert "这招通用的做法" in 材料 and 卡.how_to[:20] in 材料      # 卡片那份
    assert "军师要求在当前流水线上这样落地" in 材料                  # 军师那份
    assert "K 取 20" in 材料

# ── 分桶的带子：实测恒等于 0，医生却拿它当门槛 ──────────────────────


def test_噪声带_分桶实测为0时退回理论带():
    """保真度抽样只抽负样本，click=1 子集每个种子完全相同 —— 分桶购买分
    实测抖动必然是 0.0000。那不是「这个桶很稳」，是「这个测法扰动不到它」。
    照单全收的话门槛就是 0，任何分桶差距都算病。"""
    实测为0 = {"购买分": {"均值": 0.55, "噪声带": 0.0},
              "购买分_理论噪声带": 0.087, "转化正样本数": 41}
    assert noise.bucket_band(实测为0) == 0.087

    有实测 = {"购买分": {"均值": 0.55, "噪声带": 0.021},
             "购买分_理论噪声带": 0.087, "转化正样本数": 41}
    assert noise.bucket_band(有实测) == 0.021          # 实测优先于理论

    缩过的 = {**实测为0, "购买分_有效噪声带": 0.012}
    assert noise.bucket_band(缩过的) == 0.012          # 缩放后的有效带最优先


def test_噪声带_医生拿到的分桶门槛不是0():
    """医生判「冷门商品学不动」「新用户不会做」用的就是这个数。
    给他 0，等于告诉他任何差距都算病。"""
    bands = {"单指标噪声带": 0.09, "分指标噪声带": {"点击AUC": 0.001, "购买AUC": 0.09},
             "分组": {"按商品出现次数分组": {
                 "<10次": {"购买分": {"均值": 0.55, "噪声带": 0.0},
                          "购买分_理论噪声带": 0.087, "转化正样本数": 41}}}}
    带了 = _with_bands({"验证集": {"点击分": 0.61}}, bands)
    assert 带了["噪声带"]["分组"]["按商品出现次数分组"]["<10次"] == 0.087


def test_噪声带_分桶带子也跟着缩放到新档位():
    """全局带子缩窄了、桶带子还停在起步档位 = 医生手里两把刻度不一样的尺子。"""
    bands = {
        "保真度": "小份",
        "样本量": {"总行数": 218_000, "点击数": 8_950, "转化数": 38},
        "分指标噪声带": {"点击AUC": 0.001, "购买AUC": 0.09},
        "点击分": {"均值": 0.61}, "购买分": {"均值": 0.56},
        "分组": {"按商品出现次数分组": {
            "<10次": {"购买分": {"均值": 0.55, "噪声带": 0.0},
                     "购买分_理论噪声带": 0.087, "转化正样本数": 41}}},
    }
    新报告 = {"保真度": "全量",
             "验证集": {"总行数": 2_180_000, "点击数": 89_500, "转化数": 467},
             "按商品出现次数分组": [{"区间": "<10次", "转化正样本数": 502}]}
    放大 = noise.rescale(bands, 新报告)

    桶 = 放大["分组"]["按商品出现次数分组"]["<10次"]
    assert 桶["转化正样本数"] == 502
    # 41 → 502 个正样本，这个桶的带子必须明显收窄
    assert 0 < noise.bucket_band(桶) < 0.087 / 2
    # 合成的标量门槛也得跟着缩，否则「单指标」和「分指标」是两个档位的数
    assert 放大["单指标噪声带"] == pytest.approx(max(放大["分指标噪声带"].values()))
    assert 放大["单指标噪声带"] < bands["分指标噪声带"]["购买AUC"]


def test_噪声带_缩不过去的桶不会悄悄变成0():
    """新档位的成绩单里没有这个桶（比如分桶边界变了）—— 宁可留旧带子，
    也不能退回 0，那等于把门槛拆了。"""
    bands = {
        "保真度": "小份",
        "样本量": {"总行数": 218_000, "点击数": 8_950, "转化数": 38},
        "分指标噪声带": {"点击AUC": 0.001, "购买AUC": 0.09},
        "点击分": {"均值": 0.61}, "购买分": {"均值": 0.56},
        "分组": {"按商品出现次数分组": {
            "<10次": {"购买分": {"均值": 0.55, "噪声带": 0.0},
                     "购买分_理论噪声带": 0.087, "转化正样本数": 41}}},
    }
    放大 = noise.rescale(bands, {"保真度": "全量",
                               "验证集": {"总行数": 2_180_000, "点击数": 89_500,
                                        "转化数": 467}})   # 没有分组那一节
    assert noise.bucket_band(放大["分组"]["按商品出现次数分组"]["<10次"]) == 0.087


# ── 尺子对不对得上这一场的档位，必须写进结果表 ────────────────────


def _跑一场(tmp_path, **kw):
    ex = DriftingExecutor()
    return run_session(
        llm=ScriptedLLM(promote_on=()), vocab=SymptomVocab.load(),
        cards=CardLibrary.load(SymptomVocab.load()), executor=ex,
        initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=2, logs_dir=tmp_path, **kw)


def test_噪声带_档位对不上会自动缩并写进结果表(tmp_path):
    """拿小份量的带子去卡中份的结果，不会抛任何异常 —— 所以必须自己喊出来。"""
    bands = {"保真度": "小份",
             "样本量": {"总行数": 218_000, "点击数": 8_950, "转化数": 38},
             "分指标噪声带": {"点击AUC": 0.001, "购买AUC": 0.09},
             "单指标噪声带": 0.09,
             "点击分": {"均值": 0.61}, "购买分": {"均值": 0.56}, "分组": {}}
    summary = _跑一场(tmp_path, start_fidelity="中份", noise_bands=bands)
    assert "小份" in summary.noise_note and "中份" in summary.noise_note
    assert "噪声带" in summary.as_table()


def test_噪声带_档位对得上就照实说(tmp_path):
    bands = {"保真度": "小份",
             "样本量": {"总行数": 218_000, "点击数": 8_950, "转化数": 38},
             "分指标噪声带": {"点击AUC": 0.001, "购买AUC": 0.09},
             "单指标噪声带": 0.09, "分组": {}}
    summary = _跑一场(tmp_path, start_fidelity="小份", noise_bands=bands)
    assert "与起步档位一致" in summary.noise_note


def test_噪声带_没测过要在结果表里点名(tmp_path):
    """报一个没有依据的门槛，比报「没有门槛」更危险。"""
    summary = _跑一场(tmp_path)
    assert "没测过噪声带" in summary.noise_note
    assert "agent.cli noise" in summary.noise_note


def test_噪声带_升档重测失败要留下已过期的记录(tmp_path):
    """重测没跑起来 = 拿不到新档位的样本量 = 缩不过去。
    之后几轮是用上一档的尺子量新档位的结果，门槛偏松，必须留痕。"""
    # 门槛必须比假复盘官报的 actual（购买 +0.0031）小，否则它判不了「猜对了」，
    # promote 永远不会为真，这一轮压根走不到升档那一步
    bands = {"保真度": "小份",
             "样本量": {"总行数": 218_000, "点击数": 8_950, "转化数": 38},
             "分指标噪声带": {"点击AUC": 0.0005, "购买AUC": 0.002},
             "单指标噪声带": 0.002, "分组": {}}
    # 复盘官第 1 次就判 promote → 第 1 轮结束触发升档；让升档那次重测失败。
    # DriftingExecutor 的 run 计数：第 1 轮训练是第 1 次，升档重测是第 2 次。
    ex = DriftingExecutor(fail_rounds=(2,))
    summary = run_session(
        llm=ScriptedLLM(promote_on=(1,)), vocab=SymptomVocab.load(),
        cards=CardLibrary.load(SymptomVocab.load()), executor=ex,
        initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=2, logs_dir=tmp_path, start_fidelity="小份", noise_bands=bands)
    assert "仍停在" in summary.noise_note and "偏松" in summary.noise_note


# ── schema 必须落在结构化输出接口吃得下的那个子集里 ──────────────────
#
# 真跑那天撞出来的：medium 那次「真实 API 端到端跑通」走的是 DeepSeek
# （OpenAI 兼容的 json_object 模式，根本不校验 schema），Anthropic 这条
# 结构化输出的路第一次被调用就 400 了：
#   output_config.format.schema: For 'array' type, property 'maxItems' is not supported
# 而且报错只点名**第一个**违规项，得一个一个撞。本地跑一遍就能全揪出来。


def test_schema_只用接口吃得下的关键字(vocab, cards):
    """maxItems / minimum / maximum / oneOf 这些写了会让整个请求 400。"""
    for 名字, s in (
        ("医生", schemas.doctor_schema(vocab)),
        ("军师", schemas.strategist_schema(vocab, ["类目兜底", "ESMM"])),
        ("工兵", schemas.implementer_schema()),
        ("复盘官", schemas.reflector_schema(vocab)),
    ):
        try:
            schemas.assert_api_compatible(s)
        except ValueError as exc:
            pytest.fail(f"{名字} 的 schema 接口吃不下：{exc}")


def test_schema守卫_认得出越界的关键字():
    """守卫本身要真的会拦，否则它只是一句安慰。"""
    with pytest.raises(ValueError, match="maxItems"):
        schemas.assert_api_compatible(
            {"type": "array", "items": {"type": "string"}, "maxItems": 3})
    with pytest.raises(ValueError, match="minimum"):
        schemas.assert_api_compatible({"type": "number", "minimum": 0})
    with pytest.raises(ValueError, match="minItems"):
        schemas.assert_api_compatible(
            {"type": "array", "items": {"type": "string"}, "minItems": 3})
    # 嵌套在深处也要抓得到
    with pytest.raises(ValueError, match="maxItems"):
        schemas.assert_api_compatible(
            {"type": "object", "additionalProperties": False,
             "properties": {"a": {"type": "array", "maxItems": 2}}})


# ── schema 卡不住的，validate 必须接住 ────────────────────────────


def _医生校验(vocab, data):
    captured = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            captured["validate"] = kw["validate"]
            return data

    roles.diagnose(_FakeLLM(), vocab, {"验证集": {"点击分": 0.56}})
    captured["validate"](data)


def _一条诊断(**改):
    f = {"symptom": "在背题", "severity": 0.8, "confidence": "高",
         "evidence": "训练 0.8867 对验证 0.6135，差 0.2732", "affects": ["购买AUC"]}
    f.update(改)
    return f


def test_军师_预计提升超过上限会被打回(vocab, cards):
    """这道闸门以前写在 schema 的 minimum/maximum 里 —— 接口根本不支持数值约束，
    等于从来没生效过。报个 +0.3 会把调度器的性价比公式整个带偏。"""
    p = _提案()
    p["expected"] = {"点击AUC": 0.3, "购买AUC": 0.0}
    with pytest.raises(SchemaViolation, match="0.05"):
        _军师校验(vocab, cards, {"proposals": [p]})


def test_军师_训练时间倍数不能小于下限(vocab, cards):
    """倍数是性价比公式的分母，报个 0 会让这个方案的性价比变成无穷大。"""
    p = _提案()
    p["cost"] = {"代码难度": "简单", "训练时间倍数": 0.0}
    with pytest.raises(SchemaViolation, match="训练时间倍数"):
        _军师校验(vocab, cards, {"proposals": [p]})


def test_医生_严重度超出0到1会被打回(vocab):
    """severity 直接进筛卡的加权求和，超范围会把排序带歪。"""
    with pytest.raises(SchemaViolation, match="severity"):
        _医生校验(vocab, {"findings": [_一条诊断(severity=1.7)],
                       "no_finding": False, "reason_if_none": ""})


def test_医生_报太多条会被打回(vocab):
    with pytest.raises(SchemaViolation, match="最多"):
        _医生校验(vocab, {"findings": [_一条诊断()] * 4,
                       "no_finding": False, "reason_if_none": ""})


# ── 内存不够 = 我们兑现不了，不是这个方法不行 ────────────────────
#
# 第 1 轮真撞过：军师挑了「多值字段接回来」，工兵写的零件诚实声明它要
# 12 个多值列，953 万行 × 12 列约 60G，机器只剩 20G —— 读到一半 ArrowMemoryError。
# 崩本身不算意外，**记账错了才是**：那一轮走的是「没跑起来」，卡片被扣
# 0.15 信任分并拉黑，等于把「我们读不下」记成「这个方法不行」，
# 而那张卡治的是医生排第 2、第 3 的两个病。


def _造分片数据(tmp_path, 片数=3, 每片=40):
    """造一个多分片目录 —— 估算函数要靠"抽一片再外推"，单文件它会主动弃权。"""
    pd = pytest.importorskip("pandas")
    import numpy as np
    rng = np.random.default_rng(0)
    d = tmp_path / "train"
    d.mkdir()
    for i in range(片数):
        pd.DataFrame({
            "sample_id": range(i * 每片, (i + 1) * 每片),
            "101": rng.integers(0, 7, 每片),
            "205": rng.integers(0, 5, 每片),
            "click": rng.integers(0, 2, 每片),
            "conversion": np.zeros(每片, dtype=int),
        }).to_parquet(d / f"part-{i:03d}.parquet")
    return d


def test_内存不够时报的是兑现不了而不是跑崩了(tmp_path, monkeypatch):
    need("pandas")
    from harness import executor as ex_mod

    d = _造分片数据(tmp_path)
    ex = ex_mod.RealExecutor(str(d), str(d), seed=1,
                             config={"features": {"base_fields": ["101", "205"]},
                                     "model": {"name": "lightgbm"}, "train": {}})
    # 假装这台机器只剩 1KB —— 任何数据都装不下
    monkeypatch.setattr(ex_mod, "available_memory_bytes", lambda: 1024)
    with pytest.raises(ex_mod.UnsupportedByExecutor) as e:
        ex._guard_memory(d, ["sample_id", "101", "205", "click", "conversion"])
    说明 = str(e.value)
    assert "不是方法不行" in 说明          # 军师和复盘官都要读得懂这句
    assert "G" in 说明                      # 要给出具体数字，别只说"内存不够"


def test_内存够就放行(tmp_path, monkeypatch):
    need("pandas")
    from harness import executor as ex_mod

    d = _造分片数据(tmp_path)
    ex = ex_mod.RealExecutor(str(d), str(d), seed=1, config={})
    monkeypatch.setattr(ex_mod, "available_memory_bytes", lambda: 8 * 1024 ** 3)
    ex._guard_memory(d, ["sample_id", "101", "click"])      # 不该抛


def test_估不出来就不拦(tmp_path, monkeypatch):
    """单个文件没法便宜地抽样。不确定的时候放行，别把能跑的挡在门外。"""
    need("pandas")
    import pandas as pd
    from harness import executor as ex_mod

    f = tmp_path / "train.parquet"
    pd.DataFrame({"sample_id": [1, 2], "101": [3, 4], "click": [0, 1]}).to_parquet(f)
    assert ex_mod.estimate_read_bytes(f, ["101"]) == (0, 0)
    ex = ex_mod.RealExecutor(str(f), str(f), seed=1, config={})
    monkeypatch.setattr(ex_mod, "available_memory_bytes", lambda: 1)
    ex._guard_memory(f, ["101"])                            # 估不出来 → 放行


def test_多值列认得出来(tmp_path):
    """内存大头就是它们，报错信息要点名，否则看的人不知道该砍哪几列。"""
    need("pandas")
    import pandas as pd
    from harness import executor as ex_mod

    d = tmp_path / "shards"
    d.mkdir()
    pd.DataFrame({"101": [1, 2], "109_14": [[1, 2], [3]], "click": [0, 1]}
                 ).to_parquet(d / "part-000.parquet")
    assert ex_mod.list_columns_of(d) == {"109_14"}


def test_兑现不了的不扣卡片信任分():
    """这条已经有测试盖着记账逻辑，这里钉住「内存不够」也走同一条路。"""
    from agent.loop import PriorLedger, RunResult

    r = RunResult(ok=False, error="内存不够", unsupported=True)
    assert r.unsupported is True
    账 = PriorLedger()
    账.apply("多值字段接回来", "没跑起来", 0.6) if not r.unsupported else None
    assert 账.values == {}                  # 兑现不了 → 一分不扣


# ── 爬山：走差了要退回去 ──────────────────────────────────────
#
# 不退的后果不是"少涨一点"，是整条搜索轨迹一去不回头：工兵的改动是深度合并
# 进配置的，判「猜错了」只把卡片拉黑、扣信任分，**改动本身留在流水线上**。
# 之后每一轮都建立在更差的基础上，`cur` 跟着退化，医生看到的是退化后的
# 指标，可能去诊断我们自己造成的病。


class _会变差的执行器(DriftingExecutor):
    """第 1 轮涨，第 2 轮开始一路掉 —— 用来看退不退得回去。"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.config = {"model": {"name": "起点"}}

    def run(self, patch, fidelity):
        self.runs += 1
        self.config = {**self.config, f"第{self.runs}轮加的": True}
        self.ctr += 0.01 if self.runs == 1 else -0.05      # 第 2 轮起断崖
        return RunResult(ok=True, seconds=1.0, fidelity=fidelity,
                         health_report=self.report(fidelity))


def _爬山跑一场(tmp_path, **kw):
    ex = _会变差的执行器()
    summary = run_session(
        llm=ScriptedLLM(promote_on=()), vocab=SymptomVocab.load(),
        cards=CardLibrary.load(SymptomVocab.load()), executor=ex,
        initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, run_id="爬山场", logs_dir=tmp_path, **kw)
    return ex, summary


def test_爬山_走差了退回最好那一版(tmp_path):
    ex, summary = _爬山跑一场(tmp_path, rollback_margin=0.0)
    assert summary.best_round == 1
    # 第 2、3 轮加进去的东西必须已经被退掉
    assert "第1轮加的" in ex.config
    assert "第2轮加的" not in ex.config and "第3轮加的" not in ex.config


def test_爬山_关掉就是老行为_只累加不回头(tmp_path):
    ex, summary = _爬山跑一场(tmp_path, rollback=False)
    assert summary.best_round == 1
    # 坏改动全留在流水线上 —— 这正是不加回滚时的样子
    assert all(f"第{i}轮加的" in ex.config for i in (1, 2, 3))


def test_爬山_退回之后比较基准也跟着回去(tmp_path):
    """只把配置退回去、`cur` 不退，下一轮医生看的还是退化后的成绩单，
    复盘官的「改动之前那一版」也还停在坏的那一版。"""
    ex, summary = _爬山跑一场(tmp_path, rollback_margin=0.0)
    最好 = summary.best_scores["点击AUC"]
    行 = [json.loads(l) for l in
          (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    第一轮 = next(r for r in 行 if r["round_id"] == 1)
    assert 第一轮["metrics"]["验证集"]["点击分"] == pytest.approx(最好)


def test_爬山_门槛比噪声小的下降不值得退(tmp_path):
    """比噪声还小的下降分不清是真变差还是抖动，为它回滚只会来回颠簸。"""
    ex = _带配置的执行器(gain=0.0, decay=1.0)      # 分数基本不动
    summary = run_session(
        llm=ScriptedLLM(promote_on=()), vocab=SymptomVocab.load(),
        cards=CardLibrary.load(SymptomVocab.load()), executor=ex,
        initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=3, run_id="平场", logs_dir=tmp_path, rollback_margin=0.5)
    assert summary.recoveries == 0            # 门槛给得足够宽 → 一次都不该退


def test_爬山_第0轮有快照所以第1轮就变差也退得回去(tmp_path):
    """best 一开始指向第 0 轮。没有第 0 轮快照的话，第 1 轮走差就无处可退。"""
    class _一上来就变差(DriftingExecutor):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.config = {"model": {"name": "起点"}}

        def run(self, patch, fidelity):
            self.runs += 1
            self.config = {**self.config, f"第{self.runs}轮加的": True}
            self.ctr -= 0.05
            return RunResult(ok=True, seconds=1.0, fidelity=fidelity,
                             health_report=self.report(fidelity))

    ex = _一上来就变差()
    summary = run_session(
        llm=ScriptedLLM(promote_on=()), vocab=SymptomVocab.load(),
        cards=CardLibrary.load(SymptomVocab.load()), executor=ex,
        initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=2, run_id="开局就差", logs_dir=tmp_path, rollback_margin=0.0)
    assert summary.best_round == 0
    assert "第1轮加的" not in ex.config        # 退回了起点


# ── 失败原因必须传到下一轮 ────────────────────────────────────
#
# 真跑 10 轮 0 成功那次的根因：崩掉时 _crashed_reflection 写死一句
# 「换一个方案，或者先修这个报错」，而 _brief 的 `备注` 取的正是它、
# `or` 后面的 recoveries 永远轮不到。执行器说的「你要的那些列读不进内存」
# 在这一步被扔掉，Agent 只能靠猜，连着几轮撞同一堵墙。


def test_崩了之后报错原文要传下去():
    from agent.loop import RoundLog, _brief, _crashed_reflection

    log = RoundLog(round_id=3, started_at="", run_id="x")
    log.chosen = {"card_id": "", "targets": ["历史行为没用上"]}
    log.recoveries = ["执行失败：UnsupportedByExecutor: 这些多值列读不进内存"]
    log.reflection = _crashed_reflection(log.chosen, "UnsupportedByExecutor: 读不进内存")
    b = _brief(log)
    assert "读不进内存" in b["备注"]              # next_hint 带上了原话
    assert "读不进内存" in b["出了什么错"]         # 恢复事件也单独占一格


def test_没崩但有恢复事件也要传下去():
    """工兵换了备胎、某个角色重试过 —— 这些以前也传不出去。"""
    from agent.loop import RoundLog, _brief

    log = RoundLog(round_id=2, started_at="", run_id="x")
    log.recoveries = ["工兵实现失败（FinalMLP）：出现了禁用字段 click"]
    log.reflection = {"verdict": "猜对了", "actual": {}, "next_hint": "继续"}
    assert "禁用字段 click" in _brief(log)["出了什么错"]


def test_没出错就不占位():
    from agent.loop import RoundLog, _brief

    log = RoundLog(round_id=1, started_at="", run_id="x")
    log.reflection = {"verdict": "猜对了", "actual": {}, "next_hint": "继续"}
    assert "出了什么错" not in _brief(log)


# ── 自创方案也要记账 ──────────────────────────────────────────


class _只提自创的假模型(ScriptedLLM):
    def _军师(self, schema):
        out = super()._军师(schema)
        for p in out["proposals"]:
            p["card_id"], p["novel"] = "", True
            p["how_to"] = "自己想的：按类目做兜底编码，K 从 20 起"
        return out


def test_自创方案会进已经试过的清单(tmp_path):
    """以前 `if card_id:` 整段跳过 —— 自创方案不留痕，军师下一轮
    完全不知道自己提过、更不知道被拒了，只能换个说法再提一遍。"""
    看到的 = []
    真propose = roles.propose

    def 记一笔(*a, **kw):
        看到的.append(kw.get("tried_before") or [])
        return 真propose(*a, **kw)

    import agent.loop as loop_mod
    原 = loop_mod.roles.propose
    loop_mod.roles.propose = 记一笔
    try:
        ex = DriftingExecutor()
        run_session(
            llm=_只提自创的假模型(promote_on=()), vocab=SymptomVocab.load(),
            cards=CardLibrary.load(SymptomVocab.load()), executor=ex,
            initial_report=ex.report("小份"),
            module_interface="", example_module="", current_config="",
            rounds=3, logs_dir=tmp_path)
    finally:
        loop_mod.roles.propose = 原

    最后一轮看到的 = 看到的[-1]
    assert 最后一轮看到的, "军师应该看到前面试过的自创方案"
    assert any(t.get("card_id") == "（自创）" for t in 最后一轮看到的)
    assert any("按类目做兜底编码" in (t.get("做法") or "") for t in 最后一轮看到的)


def test_军师能拿到最近几轮的历史(vocab, cards):
    """开药的人得看得到上一副药为什么没煎成。"""
    看到的 = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            看到的["user"] = kw["user"]
            return {"proposals": [_提案()]}

    roles.propose(_FakeLLM(), vocab, [{"symptom": "冷门商品学不动"}],
                  [cards.get("类目兜底")],
                  history_brief=[{"轮次": 1, "结论": "没跑起来",
                                  "出了什么错": "这些多值列读不进内存"}])
    assert "最近几轮发生了什么" in 看到的["user"]
    assert "读不进内存" in 看到的["user"]


def test_没历史时不塞空块(vocab, cards):
    看到的 = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            看到的["user"] = kw["user"]
            return {"proposals": [_提案()]}

    roles.propose(_FakeLLM(), vocab, [{"symptom": "冷门商品学不动"}],
                  [cards.get("类目兜底")])
    assert "最近几轮发生了什么" not in 看到的["user"]


# ── 单值的多值列，在 Arrow 层就压成标量 ──────────────────────────
#
# pandas 里一个多值列是「每行一个 Python 对象」，光对象头就约 110 字节/行，
# 跟里面装几个元素无关 —— 一个 85% 都是空数组的列，代价跟装满的一样。
# 真数据上三个这样的列要吃十几 G，压成 int64 之后不到 1 G。


def _造带多值列的分片(tmp_path, 片数=2, 每片=50):
    pd = pytest.importorskip("pandas")
    import numpy as np
    rng = np.random.default_rng(0)
    d = tmp_path / "shards"
    d.mkdir()
    for i in range(片数):
        pd.DataFrame({
            "sample_id": range(i * 每片, (i + 1) * 每片),
            "101": rng.integers(0, 5, 每片),
            # 单值：有的行是空数组 —— 空的含义是「没有交集」，不是缺失
            "508": [[int(v)] if v % 3 else [] for v in rng.integers(0, 9, 每片)],
            # 真多值：长度不定，动不了
            "109_14": [list(range(int(k))) for k in rng.integers(0, 6, 每片)],
            "click": rng.integers(0, 2, 每片),
        }).to_parquet(d / f"part-{i:03d}.parquet")
    return d


def test_单值多值列被压成标量(tmp_path):
    need("pandas")
    from harness.data import EMPTY_TOKEN, read_any

    df = read_any(_造带多值列的分片(tmp_path))
    assert df["508"].map(lambda v: hasattr(v, "__len__")).sum() == 0, "508 应该已经是标量"
    assert EMPTY_TOKEN in set(df["508"]), "空数组要映射成专门的类别，不是丢掉"
    # 真多值不能动 —— 硬压成单值会丢信息，那是编码零件的活
    assert df["109_14"].map(lambda v: hasattr(v, "__len__")).all()


def test_压平之后行数和对齐都不变(tmp_path):
    """空数组那些行必须留在原位，不能被 list_flatten 悄悄挤掉。"""
    need("pandas")
    import pandas as pd
    from harness.data import read_any

    d = _造带多值列的分片(tmp_path)
    df = read_any(d)
    原 = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("*.parquet"))],
                   ignore_index=True)
    assert len(df) == len(原)
    assert list(df["sample_id"]) == list(原["sample_id"])
    非空 = 原["508"].map(len) > 0
    assert (df.loc[非空, "508"].to_numpy()
            == 原.loc[非空, "508"].map(lambda v: v[0]).to_numpy()).all()


def test_压平省下的内存是数量级的(tmp_path):
    need("pandas")
    import pandas as pd
    from harness.data import read_any

    d = _造带多值列的分片(tmp_path, 片数=4, 每片=2000)
    压过 = read_any(d, columns=["sample_id", "101", "508", "click"])
    原样 = pd.concat([pd.read_parquet(f, columns=["sample_id", "101", "508", "click"])
                     for f in sorted(d.glob("*.parquet"))], ignore_index=True)
    省 = 原样["508"].memory_usage(deep=True) / 压过["508"].memory_usage(deep=True)
    assert 省 > 5, f"只省了 {省:.1f} 倍，压平大概没生效"


def test_内存守卫也护验证集(tmp_path, monkeypatch):
    """以前只护训练集。但验证集可能比训练集还大（小训练集 + 大验证集
    是完全合理的配法），护一边等于没护。"""
    need("pandas")
    from harness import executor as ex_mod

    d = _造带多值列的分片(tmp_path, 片数=3, 每片=100)
    ex = ex_mod.RealExecutor(str(d), str(d), seed=1,
                             config={"features": {"base_fields": ["101"]},
                                     "model": {"name": "lightgbm"}, "train": {}})
    monkeypatch.setattr(ex_mod, "available_memory_bytes", lambda: 1024)
    with pytest.raises(ex_mod.UnsupportedByExecutor):
        ex._guard_memory(d, ["sample_id", "101", "109_14"])

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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from agent.knowledge import Card, CardLibrary, SymptomVocab
from agent.llm import Ledger, SchemaViolation
from agent.loop import CostAwareScheduler, PriorLedger, TimeLedger, _with_bands, run_session
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


def _reflect_validate(vocab, data):
    captured = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            captured["validate"] = kw["validate"]
            return data

    roles.reflect(_FakeLLM(), vocab, {}, {}, {}, None)
    captured["validate"](data)


def _reflection(verdict, resolved, gain, delta=0.1, promote=False, after=None):
    # resolved 与 before/after 必须自洽：说治好了，那两个数就得真的变了
    if after is None:
        after = 0.07 if resolved == "否" else 0.03
    return {
        "verdict": verdict,
        "actual": {"点击AUC": 0.0, "购买AUC": gain},
        "vs_expected": "",
        "symptom_resolved": {
            "symptom": "冷门商品学不动", "before": 0.07, "after": after, "resolved": resolved,
        },
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
    with pytest.raises(SchemaViolation, match="低于 0.006"):
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


def test_预计提升在schema层就限幅(vocab):
    prop = schemas.strategist_schema(vocab, ["ESMM"])["properties"]["proposals"]["items"]
    for m in ("点击AUC", "购买AUC"):
        assert prop["properties"]["expected"]["properties"][m]["maximum"] == schemas.EXPECTED_CAP


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
    def __init__(self, resp):
        self.messages = type("M", (), {"create": lambda *a, **k: resp})()


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

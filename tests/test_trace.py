"""一轮的经过写成人话 —— 终端一行行打，每轮另存一份全文。

以前跑起来终端什么都不打（真跑那条路连每轮一行都没有），四个角色的输出只进了
rounds.jsonl —— 一行几万字的 JSON，人读不了。排版只许有这一份，
终端和 markdown 共用同一批摘要函数。

这里的任何异常都不许往外冒：记录是旁观者，绝不反过来弄崩正在跑的训练。
"""

from agent import trace
from agent.loop import RoundLog


def _log() -> RoundLog:
    """一轮走完全程的记录：工兵第一次写错、第二次改对，最后判「猜对了」。"""
    log = RoundLog(round_id=3, started_at="2026-09-26T10:00:00", run_id="abc")
    log.diagnosis = {
        "findings": [{"symptom": "时间漂移", "severity": 0.7, "confidence": "高",
                      "evidence": "最低桶 GAUC 0.612，最高桶 0.681",
                      "affects": ["GAUC"]}],
        "no_finding": False, "reason_if_none": ""}
    log.candidates = [{"card_id": "时间衰减加权", "名字": "近期的行为更算数",
                       "治哪些病": ["时间漂移"], "信任分": 0.6}]
    log.proposals = {"proposals": [{
        "rank": 1, "card_id": "时间衰减加权", "targets": ["时间漂移"],
        "rationale": "最低桶比最高桶低 0.069", "expected": {"GAUC": 0.001},
        "risk": "热门视频被稀释", "cost": {"代码难度": "简单", "训练时间倍数": 1.0},
        "how_to": "在 features 下新开一块", "novel": False}]}
    log.chosen, log.fidelity = log.proposals["proposals"][0], "中份"
    log.backups = ["行为序列"]
    log.attempts = [
        {"候选": "时间衰减加权", "第几次": 1, "新文件": ["modules/features/time_decay.py"],
         "config_patch": "features:\n  time_decay:\n    enabled: true\n",
         "试跑": "挂了", "报错": "KeyError: 'play_time'\n子进程 traceback：…",
         "失败代码": {"modules/features/time_decay.py": "class 写错了: pass"}},
        {"候选": "时间衰减加权", "第几次": 2, "新文件": ["modules/features/time_decay.py"],
         "config_patch": "features:\n  time_decay:\n    enabled: true\n",
         "试跑": "过了", "报错": "", "失败代码": {}},
    ]
    log.patch_files = {"modules/features/time_decay.py": "class 改好了: pass"}
    log.run_ok = True
    log.metrics = {"验证集": {"GAUC": 0.6692, "nDCG@5": 0.5433, "主分": 0.6062}}
    log.reflection = {
        "verdict": "猜对了", "actual": {"GAUC": 0.0008, "nDCG@5": 0.0031},
        "vs_expected": "略低", "next_hint": "去看新用户那一组",
        "symptom_resolved": [{"symptom": "时间漂移", "before": 0.069,
                              "after": 0.051, "resolved": "部分"}]}
    log.steps = [{"角色": "医生", "秒": 12.3, "token": 4200}]
    log.seconds, log.tokens, log.train_seconds = 221.0, 38000, 218.0
    return log


def test_医生摘要带上病名和证据():
    出 = trace.步骤摘要("医生", _log().diagnosis)
    assert any("时间漂移" in x for x in 出)
    assert any("0.612" in x for x in 出)


def test_试跑挂了的摘要只留报错首行():
    出 = trace.步骤摘要("试跑", {"试跑": "挂了", "报错": "KeyError: 'x'\n第二行\n第三行"})
    assert "KeyError" in 出[0] and "第二行" not in 出[0]


def test_渲染全文八个段落都在():
    md = trace.render_round(_log())
    for 段 in ("医生", "筛卡", "军师", "调度", "工兵", "训练", "复盘官", "本轮"):
        assert f"## {段}" in md, 段


def test_失败那版代码在全文里_成功那版不重抄():
    md = trace.render_round(_log())
    assert "class 写错了" in md
    assert "class 改好了" not in md      # 成功那版在 rounds.jsonl 的 patch_files 里
    assert "patch_files" in md           # 给个去处


def test_半路死掉的一轮也能渲染():
    log = RoundLog(round_id=1, started_at="x")
    log.diagnosis = _log().diagnosis
    log.recoveries = ["军师失败：SchemaViolation: 瞎写"]
    md = trace.render_round(log)         # 不许抛
    assert "没走到这一步" in md
    assert "军师失败" in md


def test_tracer按顺序打印():
    行 = []
    t = trace.Tracer(printer=行.append)
    t.round_header(3, "中份")
    t.step("医生", _log().diagnosis)
    assert "第 3 轮" in 行[0] and "中份" in 行[0]
    assert "医生" in 行[1]


def test_摘要炸了不许把这一轮带崩(monkeypatch):
    def 炸(*a, **k):
        raise RuntimeError("排版写错了")

    monkeypatch.setattr(trace, "步骤摘要", 炸)
    行 = []
    trace.Tracer(printer=行.append).step("医生", {})     # 不许抛
    assert 行 == []


def test_关掉就什么都不打():
    行 = []
    t = trace.Tracer(printer=行.append, enabled=False)
    t.round_header(1, "小份")
    t.step("医生", _log().diagnosis)
    assert 行 == []


def test_写文件落在按场次分的目录里(tmp_path):
    p = trace.write_round_md(tmp_path, _log())
    assert p == tmp_path / "rounds" / "abc" / "round_003.md"
    assert "## 复盘官" in p.read_text(encoding="utf-8")

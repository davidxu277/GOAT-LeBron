"""接线：跑起来终端真的边跑边打，每轮真的落一份 markdown。

排版本身在 test_trace.py 里测过了，这里只管「有没有接上」：
run_round 会不会调 tracer、run_session 会不会给每轮建一个、markdown 有没有落盘。
"""

from agent import trace
from agent.knowledge import CardLibrary, SymptomVocab
from agent.loop import CostAwareScheduler, PriorLedger, RunResult, run_round, run_session
from agent.offline import DriftingExecutor, ScriptedLLM


class _试跑执行器(DriftingExecutor):
    def __init__(self, fail_times: int = 0, **kw):
        super().__init__(**kw)
        self.fail_times = fail_times
        self.smoke_calls = 0

    def smoke(self, patch):
        self.smoke_calls += 1
        if self.smoke_calls <= self.fail_times:
            return RunResult(ok=False, error="KeyError: '不存在的列'\n子进程 traceback：…",
                             seconds=0.1, fidelity="试跑")
        return RunResult(ok=True, seconds=0.1, fidelity="试跑")


def _一轮(ex, **kw):
    vocab = SymptomVocab.load()
    return run_round(
        round_id=1, llm=ScriptedLLM(promote_on=()), vocab=vocab,
        cards=CardLibrary.load(vocab),
        health_report=ex.report("小份"), parent_result=ex.report("小份"),
        executor=ex, scheduler=CostAwareScheduler(),
        module_interface="", example_module="", current_config="",
        prior_ledger=PriorLedger(), **kw)


def _一场(logs_dir, rounds=2, **kw):
    vocab = SymptomVocab.load()
    ex = _试跑执行器()
    return run_session(
        llm=ScriptedLLM(promote_on=()), vocab=vocab, cards=CardLibrary.load(vocab),
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=rounds, logs_dir=logs_dir, **kw)


def test_一轮把每个步骤都打出来():
    行 = []
    _一轮(_试跑执行器(fail_times=1), tracer=trace.Tracer(printer=行.append))
    文 = "\n".join(行)
    for 步 in ("医生", "筛卡", "军师", "调度", "工兵", "试跑", "训练", "复盘官", "本轮"):
        assert 步 in 文, 步
    assert 文.count("工兵") == 2          # 第一次写错、第二次改对，两次都看得见


def test_不传tracer就一行都不打(capsys):
    _一轮(_试跑执行器())
    assert capsys.readouterr().out == ""


def test_一整场每轮都落一份markdown(tmp_path):
    summary = _一场(tmp_path, rounds=2)
    出 = sorted((tmp_path / "rounds" / summary.run_id).glob("round_*.md"))
    assert len(出) == 2
    assert "## 复盘官" in 出[0].read_text(encoding="utf-8")


def test_写markdown失败只记一笔_整场照跑完(monkeypatch, tmp_path):
    def 炸(*a, **k):
        raise OSError("磁盘满了")

    monkeypatch.setattr(trace, "write_round_md", 炸)
    summary = _一场(tmp_path, rounds=1)
    assert summary.rounds_run == 1
    assert summary.recoveries >= 1          # 记一笔恢复事件，不是把整场带走


def test_关掉trace就不落文件也不打印(capsys, tmp_path):
    _一场(tmp_path, rounds=1, trace=False)
    assert not (tmp_path / "rounds").exists()
    out = capsys.readouterr().out
    assert "医生" not in out

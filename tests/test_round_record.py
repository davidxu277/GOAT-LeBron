"""一轮跑完，过程也要留下来 —— 不只是结论。

以前 RoundLog 只记「最后选了哪张卡、最后那版代码、最后的分数」。丢掉的那四样：

- 筛卡摆给军师的候选卡（军师当时只看得见这几张，不记就无从解释它为什么提这个）
- 调度选了谁、备胎是谁
- 工兵的每一次尝试（「第一次写错、拿到 traceback 后改对」是试跑功能唯一的证据）
- 每个角色花了多少时间、多少 token（哪个角色在烧钱，靠它看）
"""

from agent.knowledge import CardLibrary, SymptomVocab
from agent.loop import CostAwareScheduler, PriorLedger, RunResult, run_round
from agent.offline import DriftingExecutor, ScriptedLLM

TRACE = ("ChildRunnerError: 训练子进程失败：KeyError: '不存在的列'\n"
         "子进程 traceback：\n  File \"modules/features/x.py\", line 9, in fit\n"
         "KeyError: '不存在的列'")


class _试跑执行器(DriftingExecutor):
    """前 fail_times 次试跑挂掉，之后放行。"""

    def __init__(self, fail_times: int = 0, **kw):
        super().__init__(**kw)
        self.fail_times = fail_times
        self.smoke_calls = 0

    def smoke(self, patch):
        self.smoke_calls += 1
        if self.smoke_calls <= self.fail_times:
            return RunResult(ok=False, error=TRACE, seconds=0.1, fidelity="试跑")
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


def test_候选卡和备胎都记下来():
    log = _一轮(_试跑执行器())
    assert log.candidates, "筛卡结果没记"
    第一张 = log.candidates[0]
    assert 第一张["card_id"] and 第一张["名字"] and 第一张["治哪些病"]
    assert 0.0 <= 第一张["信任分"] <= 1.0
    assert isinstance(log.backups, list)          # 可能真的没有备胎，但字段要在


def test_工兵每一次尝试都记下来_含写错那版代码():
    log = _一轮(_试跑执行器(fail_times=1))
    assert [a["第几次"] for a in log.attempts] == [1, 2]
    assert log.attempts[0]["试跑"] == "挂了"
    assert "不存在的列" in log.attempts[0]["报错"]
    assert log.attempts[0]["失败代码"], "写错那一版的代码没留下来"
    assert log.attempts[1]["试跑"] == "过了"
    assert log.attempts[1]["失败代码"] == {}      # 跑通那版在 patch_files 里，不存两遍


def test_没有试跑功能的执行器记没试跑():
    log = _一轮(DriftingExecutor())
    assert log.attempts and log.attempts[0]["试跑"] == "没试跑"


def test_每步的时间和token都记下来():
    log = _一轮(_试跑执行器())
    角色 = [s["角色"] for s in log.steps]
    assert 角色[:2] == ["医生", "军师"]
    assert "训练" in 角色 and "复盘官" in 角色
    assert all(s["秒"] >= 0 and s["token"] >= 0 for s in log.steps)
    assert sum(s["token"] for s in log.steps) == log.tokens   # 每步加起来就是这一轮的花费

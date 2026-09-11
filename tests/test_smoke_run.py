"""每一轮在正式训练之前先试跑；挂了就把报错交回工兵改。

以前工兵的代码只有「没过校验」时才会带着报错重试；训练崩了的 traceback 从来没交回
过它 —— 那一轮直接作废，还占掉一个官方训练名额，卡片被记「没跑起来」扣信任分，
可写错代码的是工兵，不是这个方法。2026-09-01 真跑 7 轮里 4 轮就是这么没的。

现在：执行器有 smoke 就先试跑（极小一份数据、1 轮，不占名额）。
挂了 → traceback 尾巴交回工兵，同一个方案最多再改 SMOKE_RETRIES 次；
还挂 → 换备胎；全挂 → 本轮结束，不占训练名额、不扣卡片信任分。
"""

from agent import loop
from agent.knowledge import CardLibrary, SymptomVocab
from agent.loop import CostAwareScheduler, PriorLedger, RunResult, run_round
from agent.offline import DriftingExecutor, ScriptedLLM

TRACE = ("ChildRunnerError: 训练子进程失败：KeyError: '不存在的列'\n子进程 traceback：\n"
         + "\n".join(f'  File "modules/features/x.py", line {i}, in fit' for i in range(10))
         + "\nKeyError: '不存在的列'")


class _试跑执行器(DriftingExecutor):
    """前 fail_times 次试跑挂掉；正式训练照 DriftingExecutor 走，并数一数跑了几次。"""

    def __init__(self, fail_times=0, unsupported=False, **kw):
        super().__init__(**kw)
        self.fail_times, self.unsupported = fail_times, unsupported
        self.smoke_calls = self.run_calls = 0

    def smoke(self, patch):
        self.smoke_calls += 1
        if self.smoke_calls <= self.fail_times:
            return RunResult(ok=False, error=TRACE, seconds=0.1, fidelity="试跑",
                             unsupported=self.unsupported)
        return RunResult(ok=True, seconds=0.1, fidelity="试跑")

    def run(self, patch, fidelity):
        self.run_calls += 1
        return super().run(patch, fidelity)


class _只数正式训练(DriftingExecutor):
    """没有 smoke 的执行器（离线演习那种）。"""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.run_calls = 0

    def run(self, patch, fidelity):
        self.run_calls += 1
        return super().run(patch, fidelity)


class _记下工兵输入(ScriptedLLM):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.工兵输入 = []

    def call(self, **kw):
        if kw["role"] == "工兵":
            self.工兵输入.append(kw["user"])
        return super().call(**kw)


def _一轮(ex):
    vocab = SymptomVocab.load()
    llm, ledger = _记下工兵输入(promote_on=()), PriorLedger()
    log = run_round(
        round_id=1, llm=llm, vocab=vocab, cards=CardLibrary.load(vocab),
        health_report=ex.report("小份"), parent_result=ex.report("小份"),
        executor=ex, scheduler=CostAwareScheduler(),
        module_interface="", example_module="", current_config="",
        prior_ledger=ledger)
    return log, llm, ledger


def test_试跑挂两次第三次过_正式训练只跑一次():
    ex = _试跑执行器(fail_times=2)
    log, llm, _ = _一轮(ex)
    assert len(llm.工兵输入) == 3
    assert "不存在的列" not in llm.工兵输入[0]
    assert "不存在的列" in llm.工兵输入[1] and "不存在的列" in llm.工兵输入[2]
    assert ex.run_calls == 1
    assert log.run_ok
    assert sum("试跑" in r for r in log.recoveries) == 2


def test_试跑一直挂_不占训练名额也不扣卡片分():
    ex = _试跑执行器(fail_times=99)
    log, _, ledger = _一轮(ex)
    assert ex.run_calls == 0                  # 正式训练一次都没跑 = 一个名额都没占
    assert ledger.values == {}                # 写错代码的是工兵，不是这张卡
    assert not log.run_ok                     # 这一轮仍算跑不起来，连续失败判停照常生效
    assert any("没占训练名额" in r for r in log.recoveries)


def test_执行器兑现不了就不让工兵重试():
    """兑现不了是流水线缺能力，工兵改代码改不出来 —— 直接换备胎，别白烧 token。"""
    ex = _试跑执行器(fail_times=99, unsupported=True)
    _, llm, _ = _一轮(ex)
    assert len(llm.工兵输入) == ex.smoke_calls          # 每个方案只实现一次
    assert ex.run_calls == 0


def test_没有试跑的执行器行为不变():
    ex = _只数正式训练()
    log, llm, _ = _一轮(ex)
    assert len(llm.工兵输入) == 1
    assert ex.run_calls == 1
    assert log.run_ok


def test_交回工兵的报错只留尾巴():
    """traceback 可能几百行，最有用的是最后那几行；整段塞进去会把提示词撑爆。"""
    长 = "\n".join(f"line {i}" for i in range(200))
    fb = loop._smoke_feedback(长)
    assert "line 199" in fb and "line 160" in fb
    assert "line 159" not in fb
    assert len(loop._smoke_feedback("x" * 20000)) < 4500

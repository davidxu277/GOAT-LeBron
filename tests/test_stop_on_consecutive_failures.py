"""连续跑不起来也要判停 —— 一场已经废掉的跑不该烧到 50 轮上限。

判停原本只看一件事：验证分连续 patience 轮没有超过 epsilon 的提升。而失败
的轮次根本走不到那段逻辑 —— `if not log.run_ok: continue` 直接跳过了。

后果在 2026-09-01 那场真跑里看得很清楚：第 5 轮工兵想覆盖一个已有文件被
守卫拒掉，那条补丁留在 patch history 里（另见 test_patch_history_rollback），
于是第 6、7 轮无论提什么方案都撞同一个错。剩余 43 轮全部注定失败，而
**它不会自己停**：失败轮次不计入 stale，判停条件永远不满足，它会一直烧
LLM 调用直到 50 轮上限或 token 预算耗尽。

「没有进步」和「压根跑不起来」是两种不同的终止理由，但都该终止。
"""

import json

import pytest

from agent.knowledge import CardLibrary, SymptomVocab
from agent.loop import run_session
from agent.offline import DriftingExecutor, ScriptedLLM


@pytest.fixture
def 知识库():
    vocab = SymptomVocab.load()
    return vocab, CardLibrary.load(vocab)


def _跑一场(tmp_path, 知识库, fail_rounds, rounds=12):
    vocab, cards = 知识库
    ex = DriftingExecutor(fail_rounds=fail_rounds)
    return run_session(
        llm=ScriptedLLM(), vocab=vocab, cards=cards,
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=rounds, logs_dir=tmp_path,
    )


def test_连续失败到达阈值就停(tmp_path, 知识库):
    """从第 2 轮起全部跑不起来 —— 不该跑满 12 轮。"""
    summary = _跑一场(tmp_path, 知识库, fail_rounds=tuple(range(2, 40)))

    assert summary.rounds_run < 12, (
        f"连续失败没有触发判停，跑满了 {summary.rounds_run} 轮 —— "
        f"真跑里这意味着白烧 43 轮的 LLM 调用")
    assert "跑不起来" in summary.stopped_because, summary.stopped_because


def test_偶发失败不该误停(tmp_path, 知识库):
    """只有一轮失败，后面还能跑 —— 这是正常的错误恢复，不是该收摊。"""
    summary = _跑一场(tmp_path, 知识库, fail_rounds=(3,), rounds=6)

    assert "跑不起来" not in summary.stopped_because, summary.stopped_because
    assert summary.rounds_run >= 5


def test_查不出病的轮次不算跑不起来(tmp_path, 知识库):
    """医生连着查不出病 → 那是「查不出问题」，有它自己的收敛语义和升档逻辑。

    这两种轮次都没有训练结果，但原因完全不同，终止理由也该不同。混在一起
    的话，一场本来该走"升档再看"的跑会被误判成"已经废了"。
    """
    vocab, cards = 知识库

    class _没病医生(ScriptedLLM):
        def _医生(self, schema):
            return {"findings": [], "no_finding": True, "reason_if_none": "都在噪声带内"}

    ex = DriftingExecutor()
    summary = run_session(
        llm=_没病医生(), vocab=vocab, cards=cards,
        executor=ex, initial_report=ex.report("小份"),
        module_interface="", example_module="", current_config="",
        rounds=30, logs_dir=tmp_path,
    )

    assert "跑不起来" not in summary.stopped_because, summary.stopped_because
    assert "收敛" in summary.stopped_because, summary.stopped_because


def test_停下来的那一场仍然留下完整日志(tmp_path, 知识库):
    """判停不等于崩溃 —— 结果表和逐轮日志照样要有，不然没法复盘。"""
    _跑一场(tmp_path, 知识库, fail_rounds=tuple(range(2, 40)))

    assert (tmp_path / "session_summary.json").exists()
    rows = [
        json.loads(l)
        for l in (tmp_path / "rounds.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert rows, "一行日志都没有"

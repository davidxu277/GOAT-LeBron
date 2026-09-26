"""抽样种子和训练种子分开。

以前一个 seed 管两件事：抽哪部分训练集（小份 = 15%）、模型怎么初始化。
量噪声带时要的是「同一份数据、只换训练种子」—— 一场里轮与轮之间就是这种情况，
数据子集从头到尾不变，抖动只来自训练随机性。一个 seed 管两件事的话，
换种子把抽样的抖动也量进去，门槛虚高，真提升会被当成噪声抹掉。

sample_seed 不给 = 等于 seed，老行为一个字节都不变。
"""

import pathlib
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kuairand_bridge import runner as runner_mod  # noqa: E402
from kuairand_bridge.goat_executor import KuaiRandGoatExecutor  # noqa: E402


class _抽样喊停(Exception):
    """假数据包在抽样那一步记下种子就喊停，不往下真训练。"""

    def __init__(self, fidelity, seed):
        super().__init__(f"{fidelity} {seed}")
        self.fidelity, self.seed = fidelity, seed


class _假数据包:
    def with_train_fidelity(self, fidelity, seed):
        raise _抽样喊停(fidelity, seed)


def _抽样用的种子(monkeypatch, **kw):
    monkeypatch.setattr(runner_mod, "load_dataset", lambda *a, **k: _假数据包())
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(_抽样喊停) as 停:
        runner_mod.run_trainer(tmp, "不会走到加载 trainer 这一步.py", tmp,
                               fidelity="小份", **kw)
    return 停.value.seed


def test_不给抽样种子时_seed照旧决定抽样(monkeypatch):
    assert _抽样用的种子(monkeypatch, seed=7) == 7


def test_给了抽样种子_换训练种子不换数据子集(monkeypatch):
    assert _抽样用的种子(monkeypatch, seed=1, sample_seed=0) == 0
    assert _抽样用的种子(monkeypatch, seed=2, sample_seed=0) == 0


def _记下参数(data_dir, trainer_path, output_dir, seed, make_test, **kw):
    """模块顶层（子进程要能 pickle）。"""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.txt").write_text(f"{seed} {kw.get('sample_seed', '没传')}", encoding="utf-8")
    return {"validation": {"metrics": {}}}


def _老式runner(data_dir, trainer_path, output_dir, seed, make_test,
                agent_patch=None, trainer_config=None, fidelity="全量"):
    """不认 sample_seed 的 runner（测试里的假 runner 都长这样）。"""
    return {"validation": {"metrics": {}}}


def _执行器(tmp, runner, **kw):
    return KuaiRandGoatExecutor(
        data_dir=tmp, trainer_path=str(ROOT / "examples" / "goat_trainer.py"),
        output_dir=str(pathlib.Path(tmp) / "rounds"), runner=runner, **kw)


def test_执行器把抽样种子传给runner():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _执行器(tmp, _记下参数, seed=3, sample_seed=0)
        ex.run({"new_files": [], "config_patch": ""}, "小份")
        got = next(pathlib.Path(tmp, "rounds").rglob("args.txt")).read_text(encoding="utf-8")
        assert got == "3 0"


def test_不给抽样种子时_执行器不传这个参数():
    """老式 runner 不认 sample_seed，传了就 TypeError。"""
    with tempfile.TemporaryDirectory() as tmp:
        ex = _执行器(tmp, _老式runner, seed=3)
        r = ex.run({"new_files": [], "config_patch": ""}, "小份")
        assert "TypeError" not in (r.error or "")

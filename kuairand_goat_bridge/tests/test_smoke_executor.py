"""执行器的试跑：极小一份数据、1 轮，只回答"跑不跑得通"。

真数据实测（2026-09-12）：一个读了不存在的列的零件，正式训练 3 秒就挂，
报错里带着完整的 traceback —— 可它照样占掉一个官方训练名额（上限 50），
那一轮也跟着作废。试跑把这类"代码写错了"挡在花名额之前：

- 不占训练名额、不写进补丁历史（失败了不留痕迹）
- 只取训练集 2%（保真度档 试跑），goat_trainer 下训练轮数强制 1
- 报错原样带回子进程的 traceback，外层循环好交回工兵改
"""

import json
import pathlib
import sys
import tempfile

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kuairand_bridge.goat_executor import KuaiRandGoatExecutor  # noqa: E402

GOAT_TRAINER = ROOT / "examples" / "goat_trainer.py"
FM_TRAINER = ROOT / "examples" / "official_fm_trainer.py"
PATCH = {"new_files": [], "config_patch": "model:\n  deep:\n    learning_rate: 0.01\n"}


def recording_runner(data_dir, trainer_path, output_dir, seed, make_test,
                     agent_patch=None, trainer_config=None, fidelity="全量"):
    """模块顶层（子进程要能 pickle）。把收到的参数写下来，测试再读回去。"""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "runner_args.json").write_text(
        json.dumps({"fidelity": fidelity, "history": (agent_patch or {}).get("history")},
                   ensure_ascii=False),
        encoding="utf-8")
    return {"validation": {"metrics": {}}}


def crashing_runner(data_dir, trainer_path, output_dir, seed, make_test,
                    agent_patch=None, trainer_config=None, fidelity="全量"):
    import pandas as pd
    return pd.DataFrame({"a": [1]})["不存在的列"]


def _executor(tmp, runner, trainer=GOAT_TRAINER):
    return KuaiRandGoatExecutor(
        data_dir=tmp, trainer_path=str(trainer), output_dir=str(pathlib.Path(tmp) / "rounds"),
        seed=0, runner=runner)


def _runner_args(tmp):
    got = sorted((pathlib.Path(tmp) / "rounds" / "smoke").glob("*/runner_args.json"))
    assert got, "试跑没有写到 output_dir/smoke/ 下"
    return json.loads(got[-1].read_text(encoding="utf-8"))


def test_试跑不占训练名额也不进补丁历史():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _executor(tmp, recording_runner)
        r = ex.smoke(PATCH)
        assert r.ok, r.error
        assert r.fidelity == "试跑"
        assert ex.training_attempts == 0
        assert ex._patch_history == []

        args = _runner_args(tmp)
        assert args["fidelity"] == "试跑"
        # 这次的补丁在历史里，最后再压一条「只训 1 轮」—— 工兵自己改了轮数也盖得住
        assert args["history"][-2]["config_patch"] == PATCH["config_patch"]
        assert yaml.safe_load(args["history"][-1]["config_patch"]) == {
            "model": {"deep": {"epochs": 1}}}


def test_试跑挂了带回子进程的traceback():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _executor(tmp, crashing_runner)
        r = ex.smoke(PATCH)
        assert not r.ok
        assert "KeyError" in r.error and "不存在的列" in r.error
        assert "traceback" in r.error.lower()           # 工兵要的是这一段，不只是一行
        assert ex.training_attempts == 0
        assert ex._patch_history == []


def test_只能改配置的trainer不强加训练轮数():
    """官方 FM trainer 只认 6 个超参数，塞一个 model.deep.epochs 它会直接拒掉。"""
    with tempfile.TemporaryDirectory() as tmp:
        ex = _executor(tmp, recording_runner, trainer=FM_TRAINER)
        assert ex.smoke(PATCH).ok
        assert _runner_args(tmp)["history"][-1]["config_patch"] == PATCH["config_patch"]

"""锁定集：验证集按用户切出一块，整场只在最后考一次。

Agent 跑 N 轮都在看同一份验证集挑最好的一轮，「最好」里混着恰好迎合它的部分。
B1 实测单种子抖动就有 ±0.003~0.005，跑 20 轮挑最高的，光运气就能挑出好几个千分点。
只有一份从没被任何决策看过的数据，才说得清这部分有多大。

按用户切而不是按行切：评测是用户内排序（GAUC / nDCG@5 都按用户算），
同一个用户的曝光拆到两边，两边就不是独立的考卷了。
"""

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kuairand_bridge.dataset import DatasetBundle, SplitView  # noqa: E402


def _bundle(n_users=500, per_user=4) -> DatasetBundle:
    rows = [(20220422, str(u), str(i), "a", "0", 1000.0, (u + i) % 2)
            for u in range(n_users) for i in range(per_user)]
    return DatasetBundle(train=SplitView("train", rows[:10], True),
                         valid=SplitView("valid", rows, True), test=None,
                         data_dir=pathlib.Path("."))


def test_开发集和锁定集的用户不相交_行数加起来是全量():
    b = _bundle()
    开发 = b.with_valid_part("开发", 0.2).valid
    锁定 = b.with_valid_part("锁定", 0.2).valid
    assert not set(开发.user_ids.tolist()) & set(锁定.user_ids.tolist())
    assert len(开发) + len(锁定) == len(b.valid)


def test_锁定集大约占两成用户():
    锁定 = _bundle().with_valid_part("锁定", 0.2).valid
    占比 = len(set(锁定.user_ids.tolist())) / 500
    assert 0.15 <= 占比 <= 0.25


def test_每次切出来一模一样():
    a = _bundle().with_valid_part("锁定", 0.2).valid.rows
    b = _bundle().with_valid_part("锁定", 0.2).valid.rows
    assert a == b


def test_全部就是原样():
    b = _bundle()
    assert b.with_valid_part("全部", 0.2) is b


def test_训练集不受影响():
    b = _bundle()
    assert b.with_valid_part("开发", 0.2).train is b.train


def test_比例不合法就拒绝():
    for frac in (0, 1, -0.1):
        with pytest.raises(ValueError):
            _bundle().with_valid_part("开发", frac)
    with pytest.raises(ValueError):
        _bundle().with_valid_part("随便", 0.2)


# ── runner：跑的时候只在开发集上打分，大考时两边分开打分 ──

from kuairand_bridge import runner as runner_mod  # noqa: E402

_最小TRAINER = '''
import numpy as np

def fit(train, valid, seed=0, config=None):
    return {}

def predict(model, split):
    # 跟标签沾点边的分数，免得官方评测函数算出一堆 0.5
    return np.asarray([0.3 + 0.4 * ((int(r[2]) * 7 + int(r[1])) % 5) / 5 for r in split.rows])
'''


def _跑(tmp_path, monkeypatch, **kw):
    monkeypatch.setattr(runner_mod, "load_dataset", lambda *a, **k: _bundle())
    trainer = tmp_path / "trainer.py"
    trainer.write_text(_最小TRAINER, encoding="utf-8")
    return runner_mod.run_trainer(tmp_path, trainer, tmp_path / "out", fidelity="全量", **kw)


def _行数(part):
    return len(_bundle().with_valid_part(part, 0.2).valid) if part != "全部" else len(_bundle().valid)


def test_默认在全量验证集上打分(tmp_path, monkeypatch):
    r = _跑(tmp_path, monkeypatch)
    assert r["validation"]["metrics"]["rows"] == _行数("全部")
    assert "holdout" not in r


def test_开发模式只在开发集上打分(tmp_path, monkeypatch):
    r = _跑(tmp_path, monkeypatch, valid_part="开发", holdout_frac=0.2)
    assert r["validation"]["metrics"]["rows"] == _行数("开发")
    assert "holdout" not in r                     # 锁定集的分数不许出现在平时的成绩里


def test_大考时开发集和锁定集分开打分(tmp_path, monkeypatch):
    r = _跑(tmp_path, monkeypatch, valid_part="裁决", holdout_frac=0.2)
    assert r["validation"]["metrics"]["rows"] == _行数("开发")
    assert r["holdout"]["metrics"]["rows"] == _行数("锁定")


def test_训练里早停看到的也只是开发集(tmp_path, monkeypatch):
    """goat_trainer 拿 valid 做早停和最佳权重回滚 —— 它看到锁定集，锁定集就不干净了。"""
    monkeypatch.setattr(runner_mod, "load_dataset", lambda *a, **k: _bundle())
    记 = tmp_path / "fit_saw.txt"
    trainer = tmp_path / "trainer.py"
    trainer.write_text(_最小TRAINER.replace(
        "    return {}\n",
        f"    open({str(记)!r}, 'w').write(str(len(valid.rows)))\n    return {{}}\n", 1),
        encoding="utf-8")
    for part in ("开发", "裁决"):
        runner_mod.run_trainer(tmp_path, trainer, tmp_path / part, fidelity="全量",
                               valid_part=part, holdout_frac=0.2)
        assert int(记.read_text()) == _行数("开发"), part


# ── 执行器：跑的时候只看开发集，最后大考 ──

import json  # noqa: E402
import tempfile  # noqa: E402

from kuairand_bridge.goat_executor import KuaiRandGoatExecutor  # noqa: E402


def 记参数的runner(data_dir, trainer_path, output_dir, seed, make_test, **kw):
    """模块顶层（子进程要能 pickle）。记下收到什么，按模式回分数。"""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = (kw.get("agent_patch") or {}).get("history") or []
    (out / "收到.json").write_text(json.dumps({
        "valid_part": kw.get("valid_part", "没传"), "holdout_frac": kw.get("holdout_frac"),
        "make_test": make_test, "history_len": len(history)}, ensure_ascii=False),
        encoding="utf-8")

    def m(g, n):
        return {"metrics": {"GAUC": g, "nDCG@5": n, "primary": (g + n) / 2,
                            "rows": 10, "users": 2}}
    g = 0.66 + 0.002 * len(history)           # 补丁越多分越高，好看出截到了哪一轮
    r = {"validation": m(g, 0.53)}
    if kw.get("valid_part") == "裁决":
        r["holdout"] = m(g - 0.001, 0.52)
    if make_test:
        r["test"] = {"status": "checked"}
    return r


def _执行器(tmp, **kw):
    return KuaiRandGoatExecutor(
        data_dir=tmp, trainer_path=str(ROOT / "examples" / "goat_trainer.py"),
        output_dir=str(pathlib.Path(tmp) / "rounds"), runner=记参数的runner, **kw)


def _收到(tmp, 子目录):
    return [json.loads(p.read_text(encoding="utf-8"))
            for p in sorted(pathlib.Path(tmp, "rounds", 子目录).rglob("收到.json"))]


_空 = {"new_files": [], "config_patch": ""}
_改 = {"new_files": [], "config_patch": "model:\n  deep:\n    learning_rate: 0.01\n"}


def test_开了锁定集_平时每一轮都只看开发集():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _执行器(tmp, holdout_frac=0.2)
        r = ex.run(_空, "小份")
        assert r.ok, r.error
        ex.smoke(_改)
        收到 = _收到(tmp, "")
        assert {x["valid_part"] for x in 收到} == {"开发"}
        assert {x["holdout_frac"] for x in 收到} == {0.2}
        assert "锁定集" not in json.dumps(r.health_report, ensure_ascii=False)


def test_没开锁定集就什么都不传():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _执行器(tmp)
        ex.run(_空, "小份")
        assert {x["valid_part"] for x in _收到(tmp, "")} == {"没传"}


def test_大考_每个轮次各训一次_补丁历史截到那一轮():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _执行器(tmp, holdout_frac=0.2)
        ex.run(_空, "小份")
        ex.run(_改, "小份")
        ex.run(_改, "小份")
        用了 = ex.training_attempts
        v = ex.holdout_verdict([0, 2], "小份")
        assert v.ok, v.error
        assert ex.training_attempts == 用了 + 2            # 大考也是真训练，占名额
        各轮 = v.health_report["各轮"]
        assert [x["执行器轮次"] for x in 各轮] == [0, 2]
        assert 各轮[1]["开发集"]["GAUC"] > 各轮[0]["开发集"]["GAUC"]   # 截对了轮次
        assert "锁定集" in 各轮[0] and "主分" in 各轮[0]["锁定集"]
        大考 = _收到(tmp, "holdout")
        assert [x["history_len"] for x in 大考] == [1, 3]
        assert {x["valid_part"] for x in 大考} == {"裁决"}


def test_没开锁定集就不能大考():
    with tempfile.TemporaryDirectory() as tmp:
        ex = _执行器(tmp)
        ex.run(_空, "小份")
        v = ex.holdout_verdict([0], "小份")
        assert not v.ok and "锁定集" in v.error


def test_最终提交永远用全量验证集():
    """跟官方基线比的就是这个数 —— 切掉两成用户就不可比了。"""
    with tempfile.TemporaryDirectory() as tmp:
        ex = _执行器(tmp, holdout_frac=0.2)
        ex.run(_空, "小份")
        ex.make_final_submission()
        最终 = _收到(tmp, "final")
        assert 最终 and 最终[0]["valid_part"] == "没传" and 最终[0]["make_test"]

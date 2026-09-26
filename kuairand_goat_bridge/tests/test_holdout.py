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

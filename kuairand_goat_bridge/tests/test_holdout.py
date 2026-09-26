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

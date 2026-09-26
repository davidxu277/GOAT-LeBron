"""结果表里锁定集那一栏：有第 0 轮做锚点时，报「挑出来的运气」而不是「开发集分 − 锁定集分」。

开发集和锁定集是按用户切的两拨人，第 0 轮就差着一截（锁定集那 20% 用户恰好难一点、
或容易一点）。直接拿最佳轮的开发集分减锁定集分，减出来的有一大块是人群差异，
却会被读成「涨的分里有多少是迎合开发集」。
正确的比法是比**涨幅**：开发集从第 0 轮涨了多少 vs 锁定集涨了多少。
"""

from agent.loop import SessionSummary


def _s(**kw):
    return SessionSummary(best_scores={"GAUC": 0.660, "nDCG@5": 0.536},
                          holdout_scores={"GAUC": 0.644, "nDCG@5": 0.521}, **kw)


def test_有运气数字就报运气():
    表 = _s(holdout_luck={"GAUC": 0.006, "nDCG@5": 0.005}).as_table()
    assert "挑出来的运气 +0.0060" in 表
    assert "泛化落差" not in 表


def test_没有锚点时照旧报泛化落差():
    表 = _s().as_table()
    assert "泛化落差 +0.0160" in 表

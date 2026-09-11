"""真跑时，外层循环读出来的分数要叫 GAUC / nDCG@5 —— 跟官方基线同名。

goat_run 以前在每一场真跑开始时，把 agent.loop.read_scores 整个换成一个适配器：
它把 GAUC / nDCG@5 改名成 点击AUC / 购买AUC，理由是"旧 GOAT 的内部账本仍用这两个名字"。

那个理由早就不成立了 —— 外层循环自己就认 KuaiRand 的 GAUC / nDCG@5
（schemas.METRIC_PAIRS）。适配器唯一的作用变成了把对的名字改成错的：

- goat_run 传给外层循环的官方基线用的是 GAUC / nDCG@5，
  而读出来的分数叫 点击AUC / 购买AUC —— 两边没有交集，
  结果表上「相对官方基线」那一栏是空的（goat_run 自己的注释就警告过这件事）
- final_summary.json 里的 best_scores 标着另一个任务的指标名

不报错，只是交付物上的数悄悄对不上。
"""

import inspect

from agent import loop


def test_goat_run不再换掉外层循环读分数的函数():
    from kuairand_bridge import goat_run

    src = inspect.getsource(goat_run)
    assert "read_scores =" not in src, "goat_run 又在运行时换掉 agent.loop.read_scores"
    assert not hasattr(goat_run, "_track2_read_scores")


def test_外层循环读出的分数跟官方基线同名():
    report = {"验证集": {"GAUC": 0.6638, "nDCG@5": 0.5357, "主分": 0.59975}}
    baseline = {"GAUC": 0.6674, "nDCG@5": 0.5357}      # goat_run 传给 run_session 的那份
    scores = loop.read_scores(report)
    assert set(scores) == set(baseline)
    assert scores["GAUC"] == 0.6638

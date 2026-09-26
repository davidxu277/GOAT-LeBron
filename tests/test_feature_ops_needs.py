"""每个特征零件的 needs() 都要能调，而且返回的列名是它自己真会读的列。

needs() 是 FeatureOp 接口里「强烈建议实现」的方法：执行器靠它决定只把哪几列读进内存。
category_fallback 的 needs() 里写的是 self.field —— 这个零件根本没有 field 属性，
只有 item_field。现在的训练路径恰好不调 needs()，所以一直没炸；
哪天执行器照接口约定去调，这个零件就当场 AttributeError。
"""

import pandas as pd

from harness.ops import load_op_class

_类目兜底配置 = {"features": {"类目兜底": {
    "enabled": True, "impl": "modules/features/category_fallback.py",
    "item_field": "video_id", "category_field": "author_id",
    "output_field": "兜底热度", "K": 3}}}


def test_类目兜底的needs能调_而且列名是它真会读的():
    op = load_op_class("modules/features/category_fallback.py", ("fit", "transform"))(_类目兜底配置)
    要 = op.needs()
    assert 要 == ["video_id", "author_id"]
    df = pd.DataFrame({"video_id": ["a", "a", "b"], "author_id": ["x", "x", "y"]})
    op.fit(df[要])                 # 只给它声明过的列，它也得能干活
    assert "兜底热度" in op.transform(df[要]).columns

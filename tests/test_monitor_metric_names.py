"""早停盯的指标名必须跟训练循环真正产出的对齐 —— 而且要从一张表推出来。

真实事故（2026-09-01 本地全跑，第 2 轮）：

    工兵输出不合格，重试：train.early_stopping.monitor 写成了 'dev_primary'，
    但训练循环只产出 ['loss', '点击分', '购买分']。

工兵照着**报错信息**改成了 `monitor: 点击分`，顺利通过校验、进入训练，
然后在 `metrics['点击分']` 上 KeyError，整轮白烧。

`点击分 / 购买分` 是 AliCCP 时代的键名。KuaiRand 这条路每轮产出的是
`primary / GAUC / nDCG@5`。所以这条校验不只是"没拦住" —— 它列出的可选值
本身就是错的，而那段文字会被喂回给工兵当修正提示：**校验器主动把 Agent
教进了坑里。**

根因不是"这个常量写错了"，是它**另起了一张表**。指标名在 schemas.METRIC_PAIRS
里已经有唯一真相（医生、复盘官、外层循环都从那里取），roles.py 却写死了一份
自己的。loop.py:350 的注释早就写明白了：「两处各维护一张表，迟早会走岔，
而走岔时不报错。」这条测试盯的就是别再走岔。
"""

import pytest

from agent.llm import Ledger, SchemaViolation
from agent import roles, schemas


BASE_CHECK = ["未使用禁用字段 long_view", "统计量只用了训练集", "参数从配置读取"]

# KuaiRand-Pure 的成绩单长相 —— 用它让派生逻辑认出当前任务。
KUAIRAND_REPORT = {"验证集": {"GAUC": 0.6363, "nDCG@5": 0.5236, "主分": 0.5800}}


def _validate(monitor: str, report=KUAIRAND_REPORT) -> None:
    data = {
        "change_type": "只改配置",
        "config_patch": f"train:\n  early_stopping:\n    monitor: {monitor}\n",
        "new_files": [],
        "self_check": BASE_CHECK,
    }

    captured = {}

    class _FakeLLM:
        ledger = Ledger()

        def call(self, **kw):
            captured["validate"] = kw["validate"]
            return data

    roles.implement(
        _FakeLLM(), {"card_id": "", "how_to": ""}, None, "", "", "",
        health_report=report,
    )
    captured["validate"](data)


@pytest.mark.parametrize("name", ["primary", "GAUC", "nDCG@5", "loss"])
def test_KuaiRand_真正产出的指标名要放行(name):
    _validate(name)


@pytest.mark.parametrize("name", ["点击分", "购买分"])
def test_AliCCP_时代的指标名在_KuaiRand_成绩单下要被打回(name):
    with pytest.raises(SchemaViolation, match="monitor"):
        _validate(name)


def test_报错信息里列出的可选值必须是真实存在的():
    """这段文字会被喂回给工兵当修正提示，列错了就是在教它写错。"""
    with pytest.raises(SchemaViolation) as e:
        _validate("mean_auc")

    msg = str(e.value)
    assert "点击分" not in msg and "购买分" not in msg, f"还在推荐 AliCCP 的键名：{msg}"
    assert "primary" in msg


def test_指标名只有一张表():
    """roles.py 不许再私藏一份名单 —— 必须从 schemas 推。"""
    import inspect
    src = inspect.getsource(roles.implement)
    assert "点击分" not in src, "roles.implement 里又出现了写死的指标名"
    assert "epoch_metric_names" in src, "应当从 schemas 派生，而不是自己列"


def test_派生函数跟着成绩单走():
    assert set(schemas.epoch_metric_names(KUAIRAND_REPORT)) == {
        "GAUC", "nDCG@5", "primary", "loss"}
    # 认不出成绩单（比如 AliCCP 那种旧格式）时退回当前任务那一套，不再另认一套名字
    assert set(schemas.epoch_metric_names({"验证集": {"点击分": 1, "购买分": 1}})) == {
        "GAUC", "nDCG@5", "primary", "loss"}

"""分组证据测试 —— 医生判 6 个病靠的就是这些块。

这些块 08-31 之前完全不存在，医生一整场只能回「看不出来」。
测试要锁住的不只是"字段有没有"，更是三条纪律：

  · 分组依据只能从 train 统计（R2）
  · 样本量不够就明说不够，不给数字（给了医生就会拿噪声当病）
  · 抽样按整个用户抽，不撕碎用户内部（GAUC/nDCG 是用户内指标）
"""

from __future__ import annotations

import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from kuairand_bridge import diagnostics


def row(date, user, video, label):
    """行元组，结构见 official_starter_kit/data.py。"""
    return (date, user, video, "author", "1", 1000.0, label)


def 造一批(users, videos_per_user, *, date=20220422, video_prefix="v"):
    """每个用户一正一负，保证进得了 GAUC 统计。"""
    rows = []
    for user in users:
        for index in range(videos_per_user):
            rows.append(
                row(date, user, f"{video_prefix}{index}", index % 2)
            )
    return rows


class 分桶依据(unittest.TestCase):

    def test_曝光次数只从训练集数(self):
        """R2：验证集里出现很多次、训练集里没出现过的视频，必须落进最低桶。

        用验证集自己的频次分桶，等于让分桶带上验证集的信息 ——
        「冷门视频」这个病名当场失去意义。
        """
        train = [row(20220408, f"u{i}", "老视频", i % 2) for i in range(60)]
        # 这个视频在验证集里出现 400 次，但训练集里一次都没有
        valid = [row(20220422, f"u{i % 30}", "新视频", i % 2) for i in range(400)]

        report = diagnostics.build(
            train_rows=train,
            valid_rows=valid,
            valid_scores=np.linspace(0.1, 0.9, len(valid)),
        )

        桶 = {b["分组"]: b for b in report["按视频曝光次数分组"]}
        self.assertEqual(桶["曝光<10次"]["行数"], 400)
        self.assertEqual(桶["曝光10-100次"]["行数"], 0)

    def test_见没见过用户也只看训练集(self):
        train = 造一批(["老用户"], 60)
        valid = 造一批(["老用户", "新用户"], 200)

        report = diagnostics.build(
            train_rows=train,
            valid_rows=valid,
            valid_scores=np.linspace(0.1, 0.9, len(valid)),
        )

        堆 = {b["分组"]: b for b in report["按用户是否见过分组"]}
        self.assertEqual(堆["训练集里见过的"]["行数"], 200)
        self.assertEqual(堆["训练集里没见过的"]["行数"], 200)


class 样本量不够就明说(unittest.TestCase):

    def test_小分组不出分只给说明(self):
        """几十行算出来的 GAUC 是纯噪声。

        医生被要求「证据必须带数字」—— 给它一个噪声数字，
        它就会一本正经地拿噪声当病报出来。
        """
        train = 造一批(["u1"], 60)
        valid = 造一批(["u1"], 10)          # 只有 10 行，低于 200 行门槛

        report = diagnostics.build(
            train_rows=train,
            valid_rows=valid,
            valid_scores=np.linspace(0.1, 0.9, len(valid)),
        )

        堆 = {b["分组"]: b for b in report["按用户是否见过分组"]}
        见过 = 堆["训练集里见过的"]
        self.assertIn("说明", 见过)
        self.assertNotIn("GAUC", 见过)
        self.assertIn(str(diagnostics.MIN_GROUP_ROWS), 见过["说明"])

    def test_全是同一个标签的分组不出分(self):
        """正样本 0 个或全是正样本，AUC 没有定义。"""
        train = 造一批(["u1"], 60)
        valid = [row(20220422, "u1", f"v{i}", 1) for i in range(300)]

        report = diagnostics.build(
            train_rows=train,
            valid_rows=valid,
            valid_scores=np.linspace(0.1, 0.9, len(valid)),
        )

        堆 = {b["分组"]: b for b in report["按用户是否见过分组"]}
        self.assertIn("说明", 堆["训练集里见过的"])


class 用户构成(unittest.TestCase):

    def test_GAUC参与用户只数有正有负的(self):
        """官方 GAUC 只统计 0 < 正例数 < 曝光数 的用户。

        零正例和全正例用户不进 GAUC，却照样进 nDCG（零正例记 0）。
        这两类占比一高，两个指标会朝相反方向动，只看总分完全看不出来。
        """
        valid = (
            [row(20220422, "有正有负", f"a{i}", i % 2) for i in range(4)]
            + [row(20220422, "全是负的", f"b{i}", 0) for i in range(4)]
            + [row(20220422, "全是正的", f"c{i}", 1) for i in range(4)]
        )

        构成 = diagnostics._user_composition(
            [r[diagnostics.ROW_USER] for r in valid],
            [r[diagnostics.ROW_LABEL] for r in valid],
        )

        self.assertEqual(构成["总用户数"], 3)
        self.assertEqual(构成["GAUC参与用户数"], 1)
        self.assertAlmostEqual(构成["零正例用户占比"], 1 / 3, places=3)
        self.assertAlmostEqual(构成["全正例用户占比"], 1 / 3, places=3)


class 预测健康(unittest.TestCase):

    def test_行数对不上直接抛错(self):
        train = 造一批(["u1"], 60)
        valid = 造一批(["u1"], 100)

        with self.assertRaises(ValueError):
            diagnostics.build(
                train_rows=train,
                valid_rows=valid,
                valid_scores=[0.5] * (len(valid) - 1),
            )

    def test_NaN被数出来而不是被忽略(self):
        scores = np.array([0.1, np.nan, 0.3, np.inf])
        健康 = diagnostics._prediction_health(scores, 4)

        self.assertEqual(健康["NaN或Inf数"], 2)
        self.assertTrue(健康["行数一致"])

    def test_所有行同分时去重取值数是1(self):
        """模型对所有行输出同一个分 —— 排序完全无效，但总分不一定难看。"""
        健康 = diagnostics._prediction_health(np.full(500, 0.42), 500)
        self.assertEqual(健康["去重后取值数"], 1)


class 抽样纪律(unittest.TestCase):

    def test_按整个用户抽不撕碎用户内部(self):
        """随机抽行会把用户的曝光列表撕碎，算出来的分数跟验证集不可比。"""
        rows = 造一批([f"u{i}" for i in range(50)], 20)   # 50 用户 × 20 行

        抽中 = diagnostics.sample_rows_by_user(rows, 200, seed=0)

        每个用户的行数 = {}
        for r in 抽中:
            每个用户的行数[r[diagnostics.ROW_USER]] = (
                每个用户的行数.get(r[diagnostics.ROW_USER], 0) + 1
            )
        # 被抽中的用户，20 行必须一行不少
        self.assertTrue(all(n == 20 for n in 每个用户的行数.values()))
        self.assertLessEqual(len(抽中), 200 + 20)

    def test_行数没超上限就原样返回(self):
        rows = 造一批(["u1"], 10)
        self.assertEqual(len(diagnostics.sample_rows_by_user(rows, 999, 0)), 10)

    def test_同一个种子抽出同一批(self):
        rows = 造一批([f"u{i}" for i in range(50)], 20)
        甲 = diagnostics.sample_rows_by_user(rows, 200, seed=7)
        乙 = diagnostics.sample_rows_by_user(rows, 200, seed=7)
        self.assertEqual(甲, 乙)


class 口径提醒(unittest.TestCase):

    def test_分组口径提醒必须在(self):
        """实测：小份上验证集总主分 0.5800，按日期切开每天只有 0.50~0.53。

        看着像暴跌，其实一天都没跌 —— nDCG@5 是用户内指标，数据切小之后
        每个用户的曝光条数变少，它天然就低。没有这句提醒，医生会拿分组
        分数去比总分，每轮报一个不存在的「时间漂移」。
        """
        train = 造一批(["u1"], 60)
        valid = 造一批(["u1"], 300)

        report = diagnostics.build(
            train_rows=train,
            valid_rows=valid,
            valid_scores=np.linspace(0.1, 0.9, len(valid)),
        )

        self.assertIn("分组口径提醒", report)
        self.assertIn("总分", report["分组口径提醒"])

    def test_训练集自评拿不到时整块不出现(self):
        """缺证据要明说缺，不要填 0 冒充 —— 0 会被医生当成"训练分极低"。"""
        train = 造一批(["u1"], 60)
        valid = 造一批(["u1"], 300)

        report = diagnostics.build(
            train_rows=train,
            valid_rows=valid,
            valid_scores=np.linspace(0.1, 0.9, len(valid)),
            train_eval=None,
        )

        self.assertNotIn("训练集", report)


if __name__ == "__main__":
    unittest.main()


from kuairand_bridge.goat_executor import KuaiRandGoatExecutor      # noqa: E402


def 出一份成绩单(**extra):
    return KuaiRandGoatExecutor._health_report(
        {"GAUC": 0.6638, "nDCG@5": 0.5344, "primary": 0.5991,
         "rows": 124909, "users": 22377},
        "小份",
        pathlib.Path("."),
        seed=0,
        training_attempt=1,
        remaining_iterations=49,
        elapsed_seconds=31.0,
        remaining_seconds=21568.0,
        executor_round=0,
        **extra,
    )


class 成绩单要带上证据(unittest.TestCase):

    def test_分组证据摊平到顶层(self):
        """藏在一层嵌套里，医生读成绩单时容易整块略过。"""
        报告 = 出一份成绩单(group_evidence={
            "用户构成": {"GAUC参与用户占比": 0.5778},
            "训练集": {"主分": 0.6456},
        })

        self.assertIn("用户构成", 报告)
        self.assertIn("训练集", 报告)
        self.assertEqual(报告["训练集"]["主分"], 0.6456)
        # 跟「验证集」平级，医生一眼就能拿两边做差
        self.assertIn("验证集", 报告)

    def test_没有证据时成绩单照样出得来(self):
        """证据是诊断用的，不是成绩。它挂了不能把整轮拖垮。"""
        报告 = 出一份成绩单()
        self.assertEqual(报告["验证集"]["主分"], 0.5991)
        self.assertNotIn("训练集", 报告)

    def test_证据不许盖掉成绩单本身的字段(self):
        """摊平是方便医生读，不是给下游一个改写分数的口子。"""
        报告 = 出一份成绩单(group_evidence={"验证集": {"主分": 0.99}})
        self.assertEqual(报告["验证集"]["主分"], 0.5991)

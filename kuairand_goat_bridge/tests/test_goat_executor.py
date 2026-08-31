"""KuaiRand GOAT Executor 测试。

Windows multiprocessing 使用 spawn，因此传给 Executor 的测试 Runner
必须定义在模块顶层，不能定义成测试方法中的局部函数或 lambda。
"""

from __future__ import annotations

import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(
    __file__
).resolve().parents[1]

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from kuairand_bridge.goat_executor import (
    KuaiRandGoatExecutor,
    assert_goat_compatible,
)


def fake_runner(
    data_dir,
    trainer_path,
    output_dir,
    seed,
    make_test,
    agent_patch=None,
    trainer_config=None,
    fidelity="全量",
):
    """模块顶层成功Runner，兼容Windows spawn。"""
    del (
        data_dir,
        trainer_path,
        seed,
        agent_patch,
        trainer_config,
    )

    output_path = pathlib.Path(
        output_dir
    )
    output_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "training": {
            "fidelity": fidelity,
            "训练集抽样比例": 1.0 if fidelity == "全量" else 0.15,
            "训练行数": 171167 if fidelity == "小份" else 1141112,
            "全量训练行数": 1141112,
            "每轮训练记录": {"训练轮数": 3, "最佳轮次": 2},
        },
        "validation": {
            "metrics": {
                "GAUC": 0.6671,
                "nDCG@5": 0.5358,
                "primary": 0.60145,
                "users": 22377,
                "rows": 124909,
            }
        }
    }

    if make_test:
        submission = (
            output_path
            / "test_submission.csv"
        )
        submission.write_text(
            "row_id,user_id,video_id,score\n",
            encoding="utf-8",
        )

        result["test"] = {
            "status": "checked",
            "split": "test",
            "rows": 0,
            "submission": str(
                submission
            ),
            "message": (
                "Test只做格式检查"
            ),
        }

    return result


def broken_runner(
    data_dir,
    trainer_path,
    output_dir,
    seed,
    make_test,
    agent_patch=None,
    trainer_config=None,
    fidelity="全量",
):
    """模块顶层失败Runner，兼容Windows spawn。"""
    del (
        data_dir,
        trainer_path,
        output_dir,
        seed,
        make_test,
        agent_patch,
        trainer_config,
    )

    raise RuntimeError(
        "training failed"
    )


def unsupported_runner(
    data_dir,
    trainer_path,
    output_dir,
    seed,
    make_test,
    agent_patch=None,
    trainer_config=None,
    fidelity="全量",
):
    """模拟Trainer不支持Agent修改。"""
    del (
        data_dir,
        trainer_path,
        output_dir,
        seed,
        make_test,
        agent_patch,
        trainer_config,
    )

    raise NotImplementedError(
        "patch is unsupported"
    )


class GoatExecutorTests(
    unittest.TestCase
):
    """测试Executor协议、指标、预算和错误恢复。"""

    def test_protocol_and_track2_metrics(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            executor = KuaiRandGoatExecutor(
                tmp,
                __file__,
                tmp,
                runner=fake_runner,
            )

            assert_goat_compatible(
                executor
            )

            result = executor.run(
                {
                    "new_files": [],
                    "config_patch": "",
                },
                "全量",
            )

            self.assertTrue(
                result.ok,
                result.error,
            )

            validation = (
                result.health_report[
                    "验证集"
                ]
            )

            self.assertNotIn("点击分", validation)
            self.assertNotIn("购买分", validation)
            self.assertAlmostEqual(
                validation["主分"],
                0.60145,
            )
            # 官方基线**不进成绩单**。赛题按「相对官方基线的提升」排名，
            # 那是评委的尺子，不是 Agent 的输入 —— 给了它，它会退化成
            # 对着一个固定数字调参，而不去看训练/验证差、分桶、用户构成。
            # 基线仍然全程记录，在 final_summary.json 里（见 goat_run.run）。
            for 字段 in result.health_report:
                self.assertNotIn(
                    "基线",
                    str(字段),
                    f"成绩单里冒出了基线字段：{字段}",
                )

            # 具体数字也不许从别的字段漏进去
            成绩单原文 = json.dumps(
                result.health_report,
                ensure_ascii=False,
            )
            for 数字 in ("0.6674", "0.5357", "0.6016", "0.6610", "0.5946"):
                self.assertNotIn(数字, 成绩单原文)
            self.assertEqual(
                result.health_report["训练诊断"]["每轮训练记录"]["最佳轮次"],
                2,
            )

    def test_fidelity_is_forwarded_to_runner_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = KuaiRandGoatExecutor(
                tmp,
                __file__,
                tmp,
                runner=fake_runner,
            )
            result = executor.run({}, "小份")
            self.assertTrue(result.ok, result.error)
            self.assertEqual(result.health_report["保真度"], "小份")
            self.assertEqual(result.health_report["训练诊断"]["fidelity"], "小份")
            self.assertEqual(result.health_report["训练诊断"]["训练行数"], 171167)

    def test_errors_become_run_results(
        self,
    ):
        """子进程异常应转成ok=False，而不是炸掉主流程。"""
        with tempfile.TemporaryDirectory() as tmp:
            executor = KuaiRandGoatExecutor(
                tmp,
                __file__,
                tmp,
                runner=broken_runner,
            )

            result = executor.run(
                {},
                "全量",
            )

            self.assertFalse(
                result.ok
            )
            self.assertIn(
                "training failed",
                result.error,
            )

    def test_unsupported_error_is_marked(
        self,
    ):
        """NotImplementedError必须标记为unsupported。"""
        with tempfile.TemporaryDirectory() as tmp:
            executor = KuaiRandGoatExecutor(
                tmp,
                __file__,
                tmp,
                runner=unsupported_runner,
            )

            result = executor.run(
                {},
                "全量",
            )

            self.assertFalse(
                result.ok
            )
            self.assertTrue(
                result.unsupported
            )
            self.assertIn(
                "unsupported",
                result.error,
            )

    def test_total_training_attempts_are_capped(
        self,
    ):
        """所有训练尝试必须受统一上限约束。"""
        with tempfile.TemporaryDirectory() as tmp:
            executor = KuaiRandGoatExecutor(
                tmp,
                __file__,
                tmp,
                max_iterations=1,
                runner=fake_runner,
            )

            first = executor.run(
                {
                    "new_files": [],
                    "config_patch": "",
                },
                "全量",
            )

            self.assertTrue(
                first.ok,
                first.error,
            )
            self.assertEqual(
                executor.training_attempts,
                1,
            )
            self.assertEqual(
                executor.remaining_iterations,
                0,
            )

            second = executor.run(
                {
                    "new_files": [],
                    "config_patch": "",
                },
                "全量",
            )

            self.assertFalse(
                second.ok
            )
            self.assertIn(
                "上限",
                second.error,
            )
            self.assertEqual(
                executor.training_attempts,
                1,
            )
            self.assertEqual(
                executor.remaining_iterations,
                0,
            )

    def test_health_report_primary_is_official_mean(
        self,
    ):
        """成绩单主分必须保留官方primary。"""
        metrics = {
            "GAUC": 0.6674,
            "nDCG@5": 0.5357,
            "primary": 0.60155,
        }

        report = (
            KuaiRandGoatExecutor
            ._health_report(
                metrics,
                "全量",
                pathlib.Path("."),
                seed=0,
                training_attempt=1,
                remaining_iterations=49,
                elapsed_seconds=1.0,
                remaining_seconds=21599.0,
            )
        )

        self.assertAlmostEqual(
            report["验证集"]["主分"],
            0.60155,
        )
        self.assertAlmostEqual(
            report["验证集"]["GAUC"],
            0.6674,
        )
        self.assertAlmostEqual(
            report["验证集"]["nDCG@5"],
            0.5357,
        )

    def test_wrong_primary_is_rejected(
        self,
    ):
        """primary与两个官方指标均值不一致时必须失败。"""
        metrics = {
            "GAUC": 0.6674,
            "nDCG@5": 0.5357,
            "primary": 0.7000,
        }

        with self.assertRaisesRegex(
            ValueError,
            "不一致",
        ):
            (
                KuaiRandGoatExecutor
                ._health_report(
                    metrics,
                    "全量",
                    pathlib.Path("."),
                    seed=0,
                    training_attempt=1,
                    remaining_iterations=49,
                    elapsed_seconds=1.0,
                    remaining_seconds=21599.0,
                )
            )

    def test_float32_rounding_does_not_reject_primary(
        self,
    ):
        """跨平台 float32 舍入误差不能让正常的第0轮失败。"""
        metrics = {
            "GAUC": 0.6674000024795532,
            "nDCG@5": 0.5357000231742859,
            "primary": 0.6015500000000000,
        }

        report = (
            KuaiRandGoatExecutor
            ._health_report(
                metrics,
                "全量",
                pathlib.Path("."),
                seed=0,
                training_attempt=1,
                remaining_iterations=49,
                elapsed_seconds=1.0,
                remaining_seconds=21599.0,
            )
        )

        self.assertAlmostEqual(
            report["验证集"]["主分"],
            metrics["primary"],
        )

    def test_training_attempt_is_recorded(
        self,
    ):
        """每次成功训练后必须记录尝试编号。"""
        with tempfile.TemporaryDirectory() as tmp:
            executor = KuaiRandGoatExecutor(
                tmp,
                __file__,
                tmp,
                max_iterations=3,
                runner=fake_runner,
            )

            first = executor.run(
                {},
                "全量",
            )

            self.assertTrue(
                first.ok,
                first.error,
            )
            self.assertEqual(
                first.health_report[
                    "运行预算"
                ]["训练尝试编号"],
                1,
            )

            second = executor.run(
                {},
                "全量",
            )

            self.assertTrue(
                second.ok,
                second.error,
            )
            self.assertEqual(
                second.health_report[
                    "运行预算"
                ]["训练尝试编号"],
                2,
            )
            self.assertEqual(
                executor.training_attempts,
                2,
            )

    def test_final_submission_consumes_attempt(
        self,
    ):
        """最终重训和提交也必须占一次训练机会。"""
        with tempfile.TemporaryDirectory() as tmp:
            executor = KuaiRandGoatExecutor(
                tmp,
                __file__,
                tmp,
                max_iterations=2,
                runner=fake_runner,
            )

            baseline = executor.run(
                {},
                "全量",
            )

            self.assertTrue(
                baseline.ok,
                baseline.error,
            )

            executor.select_round(0)

            final = (
                executor.make_final_submission()
            )

            self.assertTrue(
                final.ok,
                final.error,
            )
            self.assertEqual(
                executor.training_attempts,
                2,
            )
            self.assertEqual(
                executor.remaining_iterations,
                0,
            )
            self.assertIn(
                "最终提交",
                final.health_report,
            )

    def test_final_submission_respects_limit(
        self,
    ):
        """训练预算耗尽后不得再启动最终重训。"""
        with tempfile.TemporaryDirectory() as tmp:
            executor = KuaiRandGoatExecutor(
                tmp,
                __file__,
                tmp,
                max_iterations=1,
                runner=fake_runner,
            )

            baseline = executor.run(
                {},
                "全量",
            )

            self.assertTrue(
                baseline.ok,
                baseline.error,
            )

            executor.select_round(0)

            final = (
                executor.make_final_submission()
            )

            self.assertFalse(
                final.ok
            )
            self.assertIn(
                "上限",
                final.error,
            )
            self.assertEqual(
                executor.training_attempts,
                1,
            )


if __name__ == "__main__":
    unittest.main()

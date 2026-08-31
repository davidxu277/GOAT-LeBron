"""KuaiRand GOAT 任务配置测试。"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import pytest
import yaml


ROOT = pathlib.Path(
    __file__
).resolve().parents[1]

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from kuairand_bridge.goat_run import (
    _track2_read_scores,
    load_task,
)


def _valid_task_config(
    data_dir: str,
    *,
    max_iterations: int = 50,
) -> dict:
    """生成一份满足新版 load_task() 契约的测试配置。"""
    return {
        "data_dir": data_dir,
        "trainer": (
            "examples/official_fm_trainer.py"
        ),
        "output_dir": "output/x",
        "seed": 0,
        "max_iterations": (
            max_iterations
        ),
        "epsilon": 0.002,
        "patience": 3,
        "max_wall_seconds": 21600,
        "token_budget": 2_000_000,
        "generate_test_after_convergence": (
            True
        ),
        "trainer_config": {
            "model": {
                "name": "fm",
                "k": 16,
                "learning_rate": 0.001,
            },
            "train": {
                "epochs": 40,
                "batch_size": 8192,
                "early_stopping_patience": 4,
                "min_delta": 0.00001,
            },
        },
        "official_baseline": {
            "validation": {
                "GAUC": 0.6674,
                "nDCG@5": 0.5357,
                "primary": 0.60155,
            },
            "hidden_test": {
                "GAUC": 0.6610,
                "nDCG@5": 0.5282,
                "primary": 0.5946,
            },
            "reproduction_tolerance": (
                0.003
            ),
        },
    }


class GoatRunConfigTests(
    unittest.TestCase
):
    """测试任务配置加载与官方限制。"""

    def test_track2_score_adapter_reads_official_names(self):
        scores = _track2_read_scores({
            "验证集": {"GAUC": 0.6674, "nDCG@5": 0.5357, "主分": 0.60155}
        })
        self.assertEqual(scores, {"点击AUC": 0.6674, "购买AUC": 0.5357})

    def test_official_iteration_limit_is_enforced(
        self,
    ):
        """超过官方50次训练上限必须被拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                pathlib.Path(tmp)
                / "task.yaml"
            )

            config = _valid_task_config(
                tmp,
                max_iterations=51,
            )

            path.write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "50",
            ):
                load_task(path)

    def test_max_iterations_is_loaded(
        self,
    ):
        """合法的 max_iterations 必须正确读入。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                pathlib.Path(tmp)
                / "task.yaml"
            )

            config = _valid_task_config(
                tmp,
                max_iterations=17,
            )

            path.write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            loaded = load_task(path)

            self.assertEqual(
                loaded["max_iterations"],
                17,
            )

    def test_relative_paths_are_resolved(
        self,
    ):
        """配置中的相对路径必须转换成绝对路径。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                pathlib.Path(tmp)
                / "task.yaml"
            )

            config = _valid_task_config(
                "data",
                max_iterations=3,
            )

            path.write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            loaded = load_task(path)

            self.assertTrue(
                pathlib.Path(
                    loaded["data_dir"]
                ).is_absolute()
            )
            self.assertTrue(
                pathlib.Path(
                    loaded["trainer"]
                ).is_absolute()
            )
            self.assertTrue(
                pathlib.Path(
                    loaded["output_dir"]
                ).is_absolute()
            )
            self.assertEqual(
                loaded["max_iterations"],
                3,
            )

    def test_data_dir_is_resolved_relative_to_task_file(
        self,
    ):
        """data_dir 应相对于任务配置文件所在目录解析。"""
        with tempfile.TemporaryDirectory() as tmp:
            task_directory = pathlib.Path(
                tmp
            )

            path = (
                task_directory
                / "task.yaml"
            )

            config = _valid_task_config(
                "relative_data"
            )

            path.write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            loaded = load_task(path)

            expected = (
                task_directory
                / "relative_data"
            ).resolve()

            self.assertEqual(
                pathlib.Path(
                    loaded["data_dir"]
                ),
                expected,
            )

    def test_trainer_config_is_preserved(
        self,
    ):
        """FM训练参数必须由配置完整传递。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                pathlib.Path(tmp)
                / "task.yaml"
            )

            config = _valid_task_config(
                tmp
            )

            path.write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            loaded = load_task(path)

            self.assertEqual(
                loaded[
                    "trainer_config"
                ]["model"]["k"],
                16,
            )
            self.assertEqual(
                loaded[
                    "trainer_config"
                ]["model"][
                    "learning_rate"
                ],
                0.001,
            )
            self.assertEqual(
                loaded[
                    "trainer_config"
                ]["train"]["epochs"],
                40,
            )
            self.assertEqual(
                loaded[
                    "trainer_config"
                ]["train"][
                    "batch_size"
                ],
                8192,
            )

    def test_missing_trainer_config_is_rejected(
        self,
    ):
        """缺少 Trainer 参数配置必须失败。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                pathlib.Path(tmp)
                / "task.yaml"
            )

            config = _valid_task_config(
                tmp
            )
            config.pop(
                "trainer_config"
            )

            path.write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "trainer_config",
            ):
                load_task(path)

    def test_wrong_epsilon_is_rejected(
        self,
    ):
        """正式运行必须使用官方 epsilon=0.002。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                pathlib.Path(tmp)
                / "task.yaml"
            )

            config = _valid_task_config(
                tmp
            )
            config["epsilon"] = 0.001

            path.write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "0.002",
            ):
                load_task(path)

    def test_wrong_patience_is_rejected(
        self,
    ):
        """正式运行必须使用官方 patience=3。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                pathlib.Path(tmp)
                / "task.yaml"
            )

            config = _valid_task_config(
                tmp
            )
            config["patience"] = 2

            path.write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "3",
            ):
                load_task(path)

    def test_wall_clock_limit_is_enforced(
        self,
    ):
        """墙钟时间不能超过官方6小时。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                pathlib.Path(tmp)
                / "task.yaml"
            )

            config = _valid_task_config(
                tmp
            )
            config[
                "max_wall_seconds"
            ] = 21601

            path.write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "21600",
            ):
                load_task(path)

    def test_official_primary_must_match_mean(
        self,
    ):
        """官方primary必须等于GAUC和nDCG@5均值。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                pathlib.Path(tmp)
                / "task.yaml"
            )

            config = _valid_task_config(
                tmp
            )
            config[
                "official_baseline"
            ]["validation"][
                "primary"
            ] = 0.7000

            path.write_text(
                yaml.safe_dump(
                    config,
                    allow_unicode=True,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "均值",
            ):
                load_task(path)


if __name__ == "__main__":
    unittest.main()


# ─────────── 09-01：最佳轮编号取错层，整场跑完才炸 ───────────


def test_最佳轮编号从运行预算里取():
    """`执行器轮次` 写在「运行预算」下面，不是顶层。

    在顶层读会 KeyError —— 而这一行跑在 run_session **之后**，
    也就是整场（最长 6 小时）训练全部跑完、就差生成提交文件时才炸。
    """
    from kuairand_bridge.goat_run import _best_executor_round

    报告 = {
        "验证集": {"GAUC": 0.6638, "nDCG@5": 0.5344, "主分": 0.5991},
        "运行预算": {"训练尝试编号": 9, "执行器轮次": 7},
    }
    assert _best_executor_round(报告, pathlib.Path("best_report.json")) == 7


def test_最佳轮编号取不到就当场报错():
    """下一步就是生成最终提交。宁可在这里停，

    也不能默默交一个不知道是哪一轮的版本 —— select_round 只检查下标
    越不越界，选错了不报错，照样跑完照样出文件。
    """
    from kuairand_bridge.goat_run import _best_executor_round

    for 坏报告 in ({"验证集": {}}, {"运行预算": {}}, {"运行预算": "不是字典"}):
        with pytest.raises(KeyError):
            _best_executor_round(坏报告, pathlib.Path("best_report.json"))


def test_最佳轮编号不能拿Agent轮次顶替():
    """Agent 轮次和执行器轮次会往**两个方向**漂：

      · 第 0 轮基线、升档重测 —— 调了 executor.run，但不是 Agent 轮次
      · 医生 no_finding 跳过、军师/工兵失败 —— 那一轮压根没调 executor.run
    """
    from kuairand_bridge.goat_run import _best_executor_round

    # Agent 第 5 轮最好，但它在执行器里是第 7 份补丁
    报告 = {"运行预算": {"执行器轮次": 7}}
    assert _best_executor_round(报告, pathlib.Path("x")) != 5
    assert _best_executor_round(报告, pathlib.Path("x")) == 7

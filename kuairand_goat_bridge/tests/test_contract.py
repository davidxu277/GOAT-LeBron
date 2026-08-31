"""KuaiRand Bridge 数据与提交契约测试。"""

from __future__ import annotations

import csv
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = pathlib.Path(
    __file__
).resolve().parents[1]

SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from kuairand_bridge.dataset import (
    DatasetBundle,
    SplitView,
    load_dataset,
)
from kuairand_bridge.predictions import (
    normalize_predictions,
)


class ContractTests(unittest.TestCase):
    """检查 test 标签隔离和官方 submission 契约。"""

    def test_train_fidelity_is_deterministic_and_keeps_validation_full(self):
        rows = [
            (20220408, f"u{i}", f"v{i}", "a", "1", 1.0, i % 2)
            for i in range(100)
        ]
        valid_rows = [
            (20220422, "uv", f"vv{i}", "a", "1", 1.0, i % 2)
            for i in range(20)
        ]
        bundle = DatasetBundle(
            train=SplitView("train", rows, True),
            valid=SplitView("valid", valid_rows, True),
            test=None,
            data_dir=pathlib.Path("."),
        )
        first = bundle.with_train_fidelity("小份", seed=7)
        second = bundle.with_train_fidelity("小份", seed=7)
        self.assertEqual(first.train.rows, second.train.rows)
        self.assertEqual(len(first.train), 16)  # 两类各 round(50 * 0.15)=8
        self.assertEqual(len(first.valid), 20)
        self.assertEqual(int(first.train.labels.sum()), 8)

    def test_unknown_fidelity_is_rejected(self):
        bundle = DatasetBundle(
            train=SplitView("train", [], True),
            valid=SplitView("valid", [], True),
            test=None,
            data_dir=pathlib.Path("."),
        )
        with self.assertRaisesRegex(ValueError, "fidelity"):
            bundle.with_train_fidelity("随便", seed=0)

    def test_test_labels_are_locked(self):
        """test SplitView 不允许通过 labels 属性读取标签。"""
        split = SplitView(
            name="test",
            rows=[
                (
                    20220429,
                    "u",
                    "v",
                    "a",
                    "t",
                    1.0,
                    None,
                )
            ],
            expose_labels=False,
        )

        with self.assertRaises(
            PermissionError
        ):
            _ = split.labels

    def test_loaded_test_rows_can_be_sanitized(
        self,
    ):
        """本地带标签 test 行必须能在边界处清除标签。"""
        raw = (
            20220429,
            "u",
            "v",
            "a",
            "t",
            1.0,
            1,
        )

        sanitized = (
            tuple(raw[:6]) + (None,)
        )

        split = SplitView(
            name="test",
            rows=[sanitized],
            expose_labels=False,
        )

        self.assertIsNone(
            split.rows[0][6]
        )

        with self.assertRaises(
            PermissionError
        ):
            _ = split.labels

    def test_development_loader_does_not_request_test(
        self,
    ):
        """开发阶段必须明确要求官方加载器跳过 test。"""
        fake_data = {
            "train": [
                (
                    20220408,
                    "u1",
                    "v1",
                    "a1",
                    "t1",
                    1.0,
                    0,
                )
            ],
            "valid": [
                (
                    20220422,
                    "u2",
                    "v2",
                    "a2",
                    "t2",
                    1.0,
                    1,
                )
            ],
        }

        with patch(
            "kuairand_bridge.dataset.module"
        ) as official_module:
            data_module = (
                official_module.return_value
            )
            data_module.load.return_value = (
                fake_data
            )

            bundle = load_dataset(
                "unused",
                include_test=False,
            )

            self.assertIsNone(
                bundle.test
            )

            self.assertEqual(
                len(bundle.train),
                1,
            )
            self.assertEqual(
                len(bundle.valid),
                1,
            )

            official_module.assert_called_once_with(
                "data"
            )

            data_module.load.assert_called_once_with(
                str(
                    pathlib.Path(
                        "unused"
                    ).resolve()
                ),
                include_test=False,
                expose_test_labels=False,
            )

    def test_final_loader_requests_test_without_labels(
        self,
    ):
        """最终提交阶段可以加载 test 特征，但不得请求 test 标签。"""
        fake_data = {
            "train": [
                (
                    20220408,
                    "u1",
                    "v1",
                    "a1",
                    "t1",
                    1.0,
                    0,
                )
            ],
            "valid": [
                (
                    20220422,
                    "u2",
                    "v2",
                    "a2",
                    "t2",
                    1.0,
                    1,
                )
            ],
            "test": [
                (
                    20220429,
                    "u3",
                    "v3",
                    "a3",
                    "t3",
                    1.0,
                    None,
                )
            ],
        }

        with patch(
            "kuairand_bridge.dataset.module"
        ) as official_module:
            data_module = (
                official_module.return_value
            )
            data_module.load.return_value = (
                fake_data
            )

            bundle = load_dataset(
                "unused",
                include_test=True,
            )

            self.assertIsNotNone(
                bundle.test
            )
            self.assertFalse(
                bundle.test.expose_labels
            )
            self.assertIsNone(
                bundle.test.rows[0][6]
            )

            official_module.assert_called_once_with(
                "data"
            )

            data_module.load.assert_called_once_with(
                str(
                    pathlib.Path(
                        "unused"
                    ).resolve()
                ),
                include_test=True,
                expose_test_labels=False,
            )

    def test_final_test_rows_have_no_label(
        self,
    ):
        """最终提交使用的 test 行必须以 None 作为标签占位符。"""
        split = SplitView(
            name="test",
            rows=[
                (
                    20220429,
                    "u",
                    "v",
                    "a",
                    "t",
                    1.0,
                    None,
                )
            ],
            expose_labels=False,
        )

        self.assertIsNone(
            split.rows[0][6]
        )

        with self.assertRaises(
            PermissionError
        ):
            _ = split.labels

    def test_test_records_do_not_expose_label(
        self,
    ):
        """records() 生成的 test 字典不能包含 long_view。"""
        split = SplitView(
            name="test",
            rows=[
                (
                    20220429,
                    "u",
                    "v",
                    "a",
                    "t",
                    1.0,
                    None,
                )
            ],
            expose_labels=False,
        )

        records = list(
            split.records()
        )

        self.assertEqual(
            len(records),
            1,
        )
        self.assertNotIn(
            "long_view",
            records[0],
        )
        self.assertEqual(
            records[0]["user_id"],
            "u",
        )
        self.assertEqual(
            records[0]["video_id"],
            "v",
        )

    def test_score_only_csv_becomes_official_submission(
        self,
    ):
        """只有 score 的文件应转换成官方四列表头。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(
                tmp
            )

            source = (
                tmp_path / "scores.csv"
            )

            source.write_text(
                "score\n0.2\n0.8\n",
                encoding="utf-8",
            )

            rows = [
                (
                    1,
                    "u1",
                    "v1",
                    "a",
                    "t",
                    1,
                    0,
                ),
                (
                    1,
                    "u2",
                    "v2",
                    "a",
                    "t",
                    1,
                    1,
                ),
            ]

            output = normalize_predictions(
                source,
                tmp_path / "submission.csv",
                rows,
            )

            with output.open(
                newline="",
                encoding="utf-8",
            ) as file_handle:
                reader = csv.reader(
                    file_handle
                )

                self.assertEqual(
                    next(reader),
                    [
                        "row_id",
                        "user_id",
                        "video_id",
                        "score",
                    ],
                )

                records = list(reader)

            self.assertEqual(
                len(records),
                2,
            )
            self.assertEqual(
                records[0][0],
                "0",
            )
            self.assertEqual(
                records[1][0],
                "1",
            )

    def test_nan_is_rejected(self):
        """submission score 不允许包含 NaN。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(
                tmp
            )

            source = (
                tmp_path / "scores.npy"
            )

            np.save(
                source,
                np.array(
                    [np.nan],
                    dtype=float,
                ),
            )

            rows = [
                (
                    1,
                    "u",
                    "v",
                    "a",
                    "t",
                    1,
                    0,
                )
            ]

            with self.assertRaisesRegex(
                ValueError,
                "NaN",
            ):
                normalize_predictions(
                    source,
                    tmp_path / "submission.csv",
                    rows,
                )

    def test_inf_is_rejected(self):
        """submission score 不允许包含 Inf。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(
                tmp
            )

            source = (
                tmp_path / "scores.npy"
            )

            np.save(
                source,
                np.array(
                    [np.inf],
                    dtype=float,
                ),
            )

            rows = [
                (
                    1,
                    "u",
                    "v",
                    "a",
                    "t",
                    1,
                    0,
                )
            ]

            with self.assertRaisesRegex(
                ValueError,
                "Inf",
            ):
                normalize_predictions(
                    source,
                    tmp_path / "submission.csv",
                    rows,
                )

    def test_prediction_row_count_must_match(
        self,
    ):
        """预测行数必须与目标 split 完全一致。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(
                tmp
            )

            source = (
                tmp_path / "scores.npy"
            )

            np.save(
                source,
                np.array(
                    [0.1, 0.2],
                    dtype=float,
                ),
            )

            rows = [
                (
                    1,
                    "u",
                    "v",
                    "a",
                    "t",
                    1,
                    0,
                )
            ]

            with self.assertRaisesRegex(
                ValueError,
                "预测有",
            ):
                normalize_predictions(
                    source,
                    tmp_path / "submission.csv",
                    rows,
                )

    def test_row_id_misalignment_is_rejected(
        self,
    ):
        """CSV 中 row_id 跳号必须被拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(
                tmp
            )

            source = (
                tmp_path / "scores.csv"
            )

            source.write_text(
                "row_id,user_id,video_id,score\n"
                "1,u1,v1,0.5\n",
                encoding="utf-8",
            )

            rows = [
                (
                    1,
                    "u1",
                    "v1",
                    "a",
                    "t",
                    1,
                    0,
                )
            ]

            with self.assertRaisesRegex(
                ValueError,
                "row_id",
            ):
                normalize_predictions(
                    source,
                    tmp_path / "submission.csv",
                    rows,
                )

    def test_user_video_misalignment_is_rejected(
        self,
    ):
        """CSV 中 user_id/video_id 错位必须被拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(
                tmp
            )

            source = (
                tmp_path / "scores.csv"
            )

            source.write_text(
                "row_id,user_id,video_id,score\n"
                "0,wrong_user,v1,0.5\n",
                encoding="utf-8",
            )

            rows = [
                (
                    1,
                    "u1",
                    "v1",
                    "a",
                    "t",
                    1,
                    0,
                )
            ]

            with self.assertRaisesRegex(
                ValueError,
                "错位",
            ):
                normalize_predictions(
                    source,
                    tmp_path / "submission.csv",
                    rows,
                )


if __name__ == "__main__":
    unittest.main()

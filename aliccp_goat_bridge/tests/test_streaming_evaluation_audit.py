import sqlite3
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from aliccp_tools.audit import audit_processed
from aliccp_tools.evaluation import _AucGroups, binary_auc, evaluate_predictions


class StreamingEvaluationTest(unittest.TestCase):
    def test_binary_auc_handles_ties(self):
        self.assertEqual(binary_auc([0, 1, 0, 1], [0.1, 0.5, 0.5, 0.9]), 0.875)

    def test_auc_groups_spill_to_disk(self):
        connection = sqlite3.connect(":memory:")
        groups = _AucGroups(connection, "scores", max_in_memory_scores=2)
        groups.add(pa.array([0.1, 0.2, 0.3, 0.4]), pa.array([0, 0, 1, 1]))
        self.assertTrue(groups.spilled)
        self.assertEqual(groups.auc(), 1.0)
        connection.close()

    def test_evaluate_streams_across_different_row_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = pa.table({
                "sample_id": ["a", "b", "c", "d"],
                "click": [0, 1, 0, 1],
                "conversion": [0, 0, 0, 1],
            })
            predictions = pa.table({
                "sample_id": ["a", "b", "c", "d"],
                "ctr": [0.1, 0.9, 0.2, 0.8],
                "cvr": [0.1, 0.2, 0.3, 0.9],
                "ctcvr": [0.01, 0.18, 0.06, 0.72],
            })
            pq.write_table(labels, root / "labels.parquet", row_group_size=3)
            pq.write_table(predictions, root / "predictions.parquet", row_group_size=2)

            metrics = evaluate_predictions(
                root / "labels.parquet", root / "predictions.parquet", batch_size=2,
            )

            self.assertTrue(metrics["streaming"])
            self.assertEqual(metrics["rows"], 4)
            self.assertEqual(metrics["ctr_auc"], 1.0)
            self.assertEqual(metrics["cvr_auc"], 1.0)
            self.assertEqual(metrics["cvr_auc_all"], 1.0)

    def test_evaluate_rejects_out_of_order_predictions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq.write_table(
                pa.table({"sample_id": ["a", "b"], "click": [0, 1], "conversion": [0, 1]}),
                root / "labels.parquet",
            )
            pq.write_table(
                pa.table({
                    "sample_id": ["b", "a"],
                    "ctr": [0.8, 0.2],
                    "cvr": [0.5, 0.5],
                    "ctcvr": [0.4, 0.1],
                }),
                root / "predictions.parquet",
            )
            with self.assertRaisesRegex(ValueError, "sample_id order mismatch"):
                evaluate_predictions(root / "labels.parquet", root / "predictions.parquet")


class StreamingAuditTest(unittest.TestCase):
    @staticmethod
    def _write_split(root: Path, split: str, sample_ids, users):
        directory = root / split
        directory.mkdir(parents=True)
        pq.write_table(
            pa.table({
                "sample_id": sample_ids,
                "click": [0, 1][:len(sample_ids)],
                "conversion": [0, 1][:len(sample_ids)],
                "101": users,
                "205": list(range(len(sample_ids))),
            }),
            directory / "part.parquet",
        )

    def test_audit_uses_disk_index_and_detects_global_problems(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_split(root, "train", ["1", "1"], [10, 11])
            self._write_split(root, "val", ["2", "3"], [10, 12])
            self._write_split(root, "public_test", ["4", "5"], [13, 14])

            report = audit_processed(root, batch_size=1)

            self.assertTrue(report["streaming"])
            self.assertEqual(report["duplicate_check"], "temporary_disk_index")
            self.assertEqual(report["splits"]["train"]["duplicate_sample_ids"], 1)
            self.assertEqual(report["train_val_user_overlap"], 1)
            self.assertEqual(report["status"], "failed")


if __name__ == "__main__":
    unittest.main()

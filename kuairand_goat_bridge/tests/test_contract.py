import csv
import pathlib
import sys
import tempfile
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuairand_bridge.dataset import SplitView
from kuairand_bridge.predictions import normalize_predictions


class ContractTests(unittest.TestCase):
    def test_test_labels_are_locked(self):
        split = SplitView("test", [(20220429, "u", "v", "a", "t", 1.0, 1)], False)
        with self.assertRaises(PermissionError):
            _ = split.labels

    def test_loaded_test_rows_can_be_sanitized(self):
        raw = (20220429, "u", "v", "a", "t", 1.0, 1)
        sanitized = tuple(raw[:6]) + (None,)
        split = SplitView("test", [sanitized], False)
        self.assertIsNone(split.rows[0][6])

    def test_score_only_csv_becomes_official_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            source = tmp_path / "scores.csv"
            source.write_text("score\n0.2\n0.8\n", encoding="utf-8")
            rows = [(1, "u1", "v1", "a", "t", 1, 0), (1, "u2", "v2", "a", "t", 1, 1)]
            out = normalize_predictions(source, tmp_path / "submission.csv", rows)
            with out.open() as fh:
                self.assertEqual(next(csv.reader(fh)), ["row_id", "user_id", "video_id", "score"])

    def test_nan_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            source = tmp_path / "scores.npy"; np.save(source, np.array([np.nan]))
            with self.assertRaisesRegex(ValueError, "NaN"):
                normalize_predictions(source, tmp_path / "submission.csv",
                                      [(1, "u", "v", "a", "t", 1, 0)])


if __name__ == "__main__":
    unittest.main()

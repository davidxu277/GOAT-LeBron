import pathlib
import sys
import tempfile
import unittest

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuairand_bridge.goat_run import load_task


class GoatRunConfigTests(unittest.TestCase):
    def test_official_limits_are_enforced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "task.yaml"
            path.write_text(yaml.safe_dump({
                "data_dir": tmp, "trainer": "examples/tunable_popularity_trainer.py",
                "output_dir": "output/x", "max_rounds": 51,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "50"):
                load_task(path)

    def test_relative_paths_are_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "task.yaml"
            path.write_text(yaml.safe_dump({
                "data_dir": "data", "trainer": "examples/tunable_popularity_trainer.py",
                "output_dir": "output/x", "max_rounds": 3,
            }), encoding="utf-8")
            cfg = load_task(path)
            self.assertTrue(pathlib.Path(cfg["data_dir"]).is_absolute())
            self.assertTrue(pathlib.Path(cfg["trainer"]).is_absolute())


if __name__ == "__main__":
    unittest.main()

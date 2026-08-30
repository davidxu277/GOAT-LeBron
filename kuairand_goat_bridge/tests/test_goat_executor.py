import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kuairand_bridge.goat_executor import KuaiRandGoatExecutor, assert_goat_compatible


class GoatExecutorTests(unittest.TestCase):
    @staticmethod
    def fake_runner(data_dir, trainer_path, output_dir, seed, make_test,
                    agent_patch=None):
        del data_dir, trainer_path, seed, make_test, agent_patch
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        return {"validation": {"metrics": {
            "GAUC": 0.6671, "nDCG@5": 0.5358, "primary": 0.60145,
            "users": 22377, "rows": 124909,
        }}}

    def test_protocol_and_metric_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            ex = KuaiRandGoatExecutor(tmp, __file__, tmp, runner=self.fake_runner)
            assert_goat_compatible(ex)
            got = ex.run({"new_files": [], "config_patch": ""}, "全量")
            self.assertTrue(got.ok)
            valid = got.health_report["验证集"]
            self.assertEqual(valid["GAUC"], valid["点击分"])
            self.assertEqual(valid["nDCG@5"], valid["购买分"])
            self.assertAlmostEqual(valid["主分"], 0.60145)

    def test_errors_become_run_results(self):
        def broken(*args, **kwargs):
            raise RuntimeError("training failed")
        with tempfile.TemporaryDirectory() as tmp:
            ex = KuaiRandGoatExecutor(tmp, __file__, tmp, runner=broken)
            got = ex.run({}, "全量")
            self.assertFalse(got.ok)
            self.assertIn("training failed", got.error)


if __name__ == "__main__":
    unittest.main()

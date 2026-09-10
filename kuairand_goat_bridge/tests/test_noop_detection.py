"""改动没生效，成绩单必须明说 —— 不能让它伪装成一次正常实验。

2026-09-01 那场真跑里，连续三轮的验证预测**逐位完全相同**（124,909 个
浮点数，最大差 0.0），而系统把它们当成三次有效实验记录了下来：

  第 1 轮  原样起步                    primary 0.57999
  第 2 轮  早停 patience: 3            primary 0.57999   ← 零件根本没加载
  第 3 轮  用户内归一化                 primary 0.57999   ← 挂载点不存在，
                                                          且数学上是恒等变换

复盘官靠推理各抓到一次（判「说不清」而不是「这招没用」，很漂亮），但那是
**运气**：它得自己想明白「减均值不改用户内次序」这种事。harness 本来就
握着铁证 —— 两轮的预测数组一模一样 —— 却没人去比一下。

一个哈希就能把这一整类问题抓住，而且不用 LLM 推理：
  · 零件没通电（缺 enabled/impl）
  · 挂载点不存在（工兵自创了一个 config 键）
  · 方案数学上是恒等变换
  · config_patch 落在了没人读的键上

这条检查的价值在于**归因**：把「这招没用」和「这招压根没执行」分开。
前者该给卡片记负分，后者该去修 harness —— 混在一起，账本就被污染了。
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kuairand_bridge.goat_executor import KuaiRandGoatExecutor


FIXED_SCORES = np.array([0.1, 0.9, 0.4, 0.7], dtype=np.float32)


def _result(fidelity: str) -> dict:
    return {
        "training": {
            "fidelity": fidelity,
            "训练集抽样比例": 0.15,
            "训练行数": 171167,
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
        },
    }


def frozen_runner(
    data_dir, trainer_path, output_dir, seed, make_test,
    agent_patch=None, trainer_config=None, fidelity="全量",
):
    """无论 Agent 改什么，预测都一模一样 —— 复现"改了等于没改"。"""
    del data_dir, trainer_path, seed, make_test, agent_patch, trainer_config

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "valid_scores.npy", FIXED_SCORES)
    return _result(fidelity)


def moving_runner(
    data_dir, trainer_path, output_dir, seed, make_test,
    agent_patch=None, trainer_config=None, fidelity="全量",
):
    """每轮预测都不一样 —— 正常情况，不该被误报成空转。"""
    del data_dir, trainer_path, seed, make_test, trainer_config

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    nth = len(list(out.parent.glob("round_*")))
    np.save(out / "valid_scores.npy", FIXED_SCORES + nth * 0.01)
    return _result(fidelity)


def _run(executor, patch=None):
    return executor.run(patch or {"new_files": [], "config_patch": ""}, "小份")


class NoopDetectionTests(unittest.TestCase):

    def test_第一轮没有上一轮可比_不下结论(self):
        with tempfile.TemporaryDirectory() as tmp:
            ex = KuaiRandGoatExecutor(tmp, __file__, tmp, runner=frozen_runner)
            r = _run(ex)
            self.assertTrue(r.ok, r.error)
            块 = r.health_report["改动是否生效"]
            self.assertEqual(块["结论"], "判不了")

    def test_预测跟上一轮逐位相同就判未生效(self):
        with tempfile.TemporaryDirectory() as tmp:
            ex = KuaiRandGoatExecutor(tmp, __file__, tmp, runner=frozen_runner)
            _run(ex)
            r = _run(ex, {"new_files": [],
                          "config_patch": "train:\n  swa:\n    enabled: true\n"})

            self.assertTrue(r.ok, r.error)
            块 = r.health_report["改动是否生效"]
            self.assertEqual(块["结论"], "未生效")
            self.assertIn("逐位相同", 块["依据"])

    def test_预测变了就是生效(self):
        with tempfile.TemporaryDirectory() as tmp:
            ex = KuaiRandGoatExecutor(tmp, __file__, tmp, runner=moving_runner)
            _run(ex)
            r = _run(ex)

            块 = r.health_report["改动是否生效"]
            self.assertEqual(块["结论"], "生效")

    def test_空改动不该被当成未生效(self):
        """本轮什么都没改（config_patch 和 new_files 都空），预测当然一样。
        那不是"改动没生效"，是"根本没有改动" —— 别把它算成事故。"""
        with tempfile.TemporaryDirectory() as tmp:
            ex = KuaiRandGoatExecutor(tmp, __file__, tmp, runner=frozen_runner)
            _run(ex)
            r = _run(ex)                      # 又一轮空补丁

            块 = r.health_report["改动是否生效"]
            self.assertEqual(块["结论"], "本轮无改动")

    def test_失败轮次不参与比较(self):
        """没跑成的那轮没有预测结果，不能拿它当"上一轮"。"""
        with tempfile.TemporaryDirectory() as tmp:
            ex = KuaiRandGoatExecutor(tmp, __file__, tmp, runner=frozen_runner)
            _run(ex)
            ex.run({"new_files": [{"path": "harness/坏路径.py", "content": ""}],
                    "config_patch": ""}, "小份")     # 会失败
            r = _run(ex, {"new_files": [], "config_patch": "model:\n  mlp:\n    dropout: 0.3\n"})

            块 = r.health_report["改动是否生效"]
            self.assertEqual(块["结论"], "未生效")


if __name__ == "__main__":
    unittest.main()

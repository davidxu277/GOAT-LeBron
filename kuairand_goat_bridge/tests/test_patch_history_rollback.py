"""失败轮次不能把自己的补丁留在 history 里。

真实事故（2026-09-01 本地全跑）：第 5 轮工兵想启用仓库里已有的
modules/train/swa.py，被 apply_agent_patch 的路径守卫以 FileExistsError
拒掉。那一轮的补丁在训练**之前**就 append 进了 _patch_history，失败后
没有回滚，于是后面每一轮重放 history 都会再撞一次同一个守卫 —— 第 6、7
轮提的方案跟 swa 毫无关系（第 6 轮只改超参数、连新文件都没有），照样报
一模一样的 FileExistsError。剩余 43 轮全部注定失败。

而且判停只看验证分、不看连续失败，所以它不会自己停下来 —— 会一路烧到
50 轮上限。**一次踩坑 = 整场报废。**
"""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kuairand_bridge.goat_executor import KuaiRandGoatExecutor


EXISTING_FILE = "modules/train/swa.py"


def guarded_runner(
    data_dir,
    trainer_path,
    output_dir,
    seed,
    make_test,
    agent_patch=None,
    trainer_config=None,
    fidelity="全量",
):
    """复刻 apply_agent_patch 的路径守卫：重放 history，撞到已有文件就抛。

    真 Trainer 从初始配置开始把整份 history 重放一遍，因此只要 history 里
    还留着那条补丁，哪一轮重放都会再抛一次 —— 与本轮提了什么无关。
    """
    del data_dir, trainer_path, seed, make_test, trainer_config

    patch = agent_patch or {}
    for item in patch.get("history") or [patch]:
        for f in item.get("new_files") or []:
            if f["path"] == EXISTING_FILE:
                raise FileExistsError(
                    f"Agent 想覆盖已有文件：{EXISTING_FILE}；只允许新建零件"
                )

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

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


def _executor(tmp):
    return KuaiRandGoatExecutor(tmp, __file__, tmp, runner=guarded_runner)


def _bad_round(executor):
    """一轮会被守卫拒掉的改动（想复用仓库里已有的零件）。"""
    return executor.run(
        {
            "new_files": [
                {"path": EXISTING_FILE, "content": "# 会被守卫拒掉\n"}
            ],
            "config_patch": "train:\n  swa:\n    enabled: true\n",
        },
        "小份",
    )


def _innocent_round(executor):
    """一轮完全无辜的改动：纯超参数，没有新文件。"""
    return executor.run(
        {
            "new_files": [],
            "config_patch": "model:\n  mlp:\n    dropout: 0.3\n",
        },
        "小份",
    )


class PatchHistoryRollbackTests(unittest.TestCase):

    def test_失败轮次之后无辜的一轮仍然能跑成(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = _executor(tmp)

            self.assertFalse(_bad_round(executor).ok, "这一轮本来就该失败")

            innocent = _innocent_round(executor)

            self.assertTrue(
                innocent.ok,
                f"无辜的一轮被上一轮的失败补丁毒死了：{innocent.error}",
            )

    def test_连续两轮无辜改动都不受失败轮次影响(self):
        with tempfile.TemporaryDirectory() as tmp:
            executor = _executor(tmp)

            _bad_round(executor)

            self.assertTrue(_innocent_round(executor).ok)
            self.assertTrue(_innocent_round(executor).ok)

    def test_成功轮次的补丁必须留在history里(self):
        """回滚只能针对失败轮次 —— 成功的改动是累积的，删掉就等于没做过。"""
        with tempfile.TemporaryDirectory() as tmp:
            executor = _executor(tmp)

            self.assertTrue(_innocent_round(executor).ok)
            self.assertTrue(_innocent_round(executor).ok)

            self.assertEqual(
                len(executor._patch_history), 2,
                "两轮都成功，history 应该有两条",
            )


if __name__ == "__main__":
    unittest.main()

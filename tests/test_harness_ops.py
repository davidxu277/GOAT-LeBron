"""KuaiRand 这条路不许再依赖旧的 AliCCP 执行器。

harness/executor.py 一千多行，几乎全是 AliCCP 专用（RealExecutor、内存守卫、
LightGBM、概率还原……），只有「按配置装零件」那一小块是 KuaiRand 也在用的：
harness/deep.py 装模型零件和训练零件、goat_trainer 装加特征零件，都从它借。

所以要拆旧流水线，先得把这一小块搬出来。搬完之后，KuaiRand 真跑时
harness.executor 这个模块压根不会被 import —— 这条测试盯的就是这件事。
"""

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

_EARLY_STOP = {"train": {"early_stopping": {
    "enabled": True, "impl": "modules/train/early_stopping.py",
    "monitor": "primary", "mode": "max", "patience": 3, "min_delta": 0.0005}}}


def test_KuaiRand路径装零件不再碰旧执行器():
    code = (
        "import sys; import harness.deep as d; "
        f"d.load_train_ops({_EARLY_STOP!r}); "
        "print('harness.executor' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_goat_trainer不从旧执行器导入():
    src = (ROOT / "kuairand_goat_bridge/examples/goat_trainer.py").read_text(encoding="utf-8")
    assert "harness.executor" not in src
    assert "from harness.ops import" in src

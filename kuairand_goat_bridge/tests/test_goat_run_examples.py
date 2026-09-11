"""真跑时工兵看到的范文，要跟着方案的环节走 —— 不能是一份写死的热度 trainer。

goat_run 在 08-30 把 examples/tunable_popularity_trainer.py 写死成工兵的范文。
那时 KuaiRand 只有「只能调配置」的热度 trainer，范文确实是它。后来换成能写零件的
goat_trainer，同一个函数里的「流水线说明」修好了，范文却一直停在热度 trainer ——
工兵提示词里「范文：一个现成的零件」那一节，放的是一个只有 model.prior 旋钮、
跟 FeatureOp / ModelOp / TrainOp 毫无关系的脚本。

而 agent/cli.py 早就有按环节取范文的 example_for（改模型看 mlp.py、加特征看
frequency_bucket.py、改训练过程看 early_stopping.py），只是真跑从来没用上。
同一件事两份手写的设置、真跑用的是过期的那份。现在只留一份，两个入口都用它。
"""

from kuairand_bridge.goat_run import _goat_root, _implementer_materials


def test_真跑时工兵按方案环节拿范文():
    _, example = _implementer_materials(_goat_root())
    assert callable(example), "范文要按方案的环节取，不能是一份固定文本"
    assert "def build" in example("模型 + 训练策略")        # modules/models/mlp.py
    assert "def transform" in example("特征")               # modules/features/frequency_bucket.py
    assert "def on_epoch_end" in example("训练策略")        # modules/train/early_stopping.py


def test_真跑时工兵不再拿到热度trainer当范文():
    _, example = _implementer_materials(_goat_root())
    for stage in ("模型", "特征", "训练策略", ""):
        assert "apply_agent_patch" not in example(stage)   # 热度 trainer 的标志函数


def test_真跑时的接口说明是仓库里真的那份():
    interface, _ = _implementer_materials(_goat_root())
    assert "class ModelOp" in interface and "class FeatureOp" in interface

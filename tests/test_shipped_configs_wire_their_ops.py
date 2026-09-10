"""仓库里出厂的配置，零件必须真的接上电。

`load_train_ops` 现在会对写残的块报错，但那是**运行时**才炸 —— 而且只在
真跑到那一步才炸。配置模板是给人和 Agent 抄的范本，抄一份坏的出去，
下一场跑又是「改了等于没改」。

这条测试把出厂配置整个扫一遍，让坏模板在 CI 就被拦住，不用等到烧掉一次
真训练。2026-09-01 那场就是 kuairand_task.yaml 的 early_stopping 块只写了
inner_holdout_frac（而且那个键还只有 LightGBM 那条路径读，deep.py 根本不
读），三重失效，没有任何东西拦得住。
"""

import pathlib

import pytest
import yaml

from tests.conftest import need


ROOT = pathlib.Path(__file__).resolve().parent.parent

SHIPPED_CONFIGS = sorted(
    p for p in ROOT.glob("**/*.yaml")
    if ".venv" not in p.parts and "dataset" not in p.parts
)


def _train_blocks(doc):
    """配置里所有 train.<名字> 的字典块（顶层和 trainer_config 下都要看）。"""
    for root in (doc, (doc or {}).get("trainer_config")):
        if not isinstance(root, dict):
            continue
        for name, block in (root.get("train") or {}).items():
            if isinstance(block, dict):
                yield name, block


@pytest.mark.parametrize("path", SHIPPED_CONFIGS, ids=lambda p: p.name)
def test_出厂配置里的零件声明都接上电了(path):
    need("torch", "pandas")
    from harness.deep import _is_train_op_block

    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        pytest.fail(f"{path} 不是合法 YAML：{exc}")

    if not isinstance(doc, dict):
        return

    for name, block in _train_blocks(doc):
        if not _is_train_op_block(name, block):
            continue                      # 普通配置段，不是零件

        assert "enabled" in block, (
            f"{path.relative_to(ROOT)} 的 train.{name} 对应 "
            f"modules/train/{name}.py，但没写 enabled —— 零件不会被加载，"
            f"任何人抄这份模板都会得到「改了等于没改」")

        if block["enabled"]:
            assert block.get("impl"), (
                f"{path.relative_to(ROOT)} 的 train.{name} 写了 enabled: true "
                f"却没写 impl，没有东西可加载")


def test_确实扫到了配置文件():
    """防止 glob 写错导致这条测试空转 —— 空集合上的断言永远通过。"""
    assert len(SHIPPED_CONFIGS) >= 4, SHIPPED_CONFIGS

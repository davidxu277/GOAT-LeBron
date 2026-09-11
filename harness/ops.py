"""按配置装零件 —— FeatureOp / ModelOp / TrainOp 共用的加载规则。

从 harness/executor.py 搬出来：那个文件一千多行几乎全是 AliCCP 专用的旧执行器，
只有这一小块是 KuaiRand 也在用的（harness/deep.py 装模型零件和训练零件、
goat_trainer 装加特征零件）。放在一起的后果是：想拆旧流水线，就得先认出
哪几行不能拆。现在它自己一个文件，KuaiRand 路径不再 import 旧执行器。
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any

import pandas as pd

from agent.events import emit

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_op_class(rel_path: str, methods: tuple[str, ...],
                  接口名: str = "") -> Any:
    """从 modules/ 下的一个文件里取出实现了指定方法的类。

    三种零件（FeatureOp / ModelOp / TrainOp）共用这一套：路径守卫一处、
    找类的规则一处，免得三份实现各自长歪。

    只认 modules/ 下的路径（R5）—— 这是 Agent 唯一被允许写入的地方，
    放开一寸就等于让它 import 任意文件。
    """
    rel = str(rel_path).replace("\\", "/")
    if not rel.startswith("modules/") or ".." in rel.split("/"):
        raise ValueError(f"非法零件路径：{rel}（只能在 modules/ 下，R5）")
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"配置里指的零件文件不存在：{rel}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for obj in vars(module).values():
        if (isinstance(obj, type) and obj.__module__ == module.__name__
                and all(callable(getattr(obj, m, None)) for m in methods)):
            return obj
    raise TypeError(f"{rel} 里没有实现 {接口名 or '零件'} 接口"
                    f"（{' + '.join(methods)}）的类")


def _load_feature_op_class(rel_path: str) -> Any:
    """加特征零件（FeatureOp）。"""
    return load_op_class(rel_path, ("fit", "transform"), "FeatureOp")


def load_feature_ops(config: dict[str, Any]) -> list[tuple[str, Any]]:
    """按配置实例化启用了的加特征零件，返回 [(名字, 实例)]。

    约定：配置块里的 `impl` 指向实现文件。零件类自己去 config 里挖自己那一块
    （见 modules/features/frequency_bucket.py 的范文），所以这里把整份 config 传给它。

    `enabled: true` 却没写 `impl` —— 直接报错。以前这种情况是**静默无效**：
    文件写进去了、配置也改了，但没有任何东西去加载它，训练结果纹丝不动，
    却被记成"这个方案没用"，工兵白挨一次负分。宁可当场炸，也不要假装跑过。
    """
    ops: list[tuple[str, Any]] = []
    for name, block in (config.get("features") or {}).items():
        if not isinstance(block, dict) or not block.get("enabled"):
            continue
        impl = block.get("impl")
        if not impl:
            raise ValueError(
                f"features.{name} 启用了但没写 impl —— 不知道该加载哪个文件。"
                f"配置块里加一行 impl: modules/features/xxx.py")
        ops.append((name, _load_feature_op_class(impl)(config)))
    return ops


def apply_feature_ops(ops: list[tuple[str, Any]], train: pd.DataFrame,
                      others: list[pd.DataFrame]
                      ) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
    """先在训练集上 fit，再对每份数据 transform。

    返回 (加工后的训练集, 加工后的其他数据集, 新长出来的列名)。

    ⚠️ fit **只看训练集**（R2）。读验证集算统计量 = 作弊，分数虚高，测试集必掉。
    新列会自动接进特征列表 —— 工兵不用再记得去改 base_fields，
    忘了改就等于零件白装，那正是这次要修掉的那类"静默无效"。
    """
    before = set(train.columns)
    for name, op in ops:
        emit("phase", name="装零件", detail=name)
        op.fit(train)
        # 训练集优先走零件自己的折外通道 —— 目标编码这类零件，
        # 用 transform 会让每一行"用包含自己标签的统计量"给自己编码，
        # 那是目标泄漏：分数虚高、日志上看不出来、到测试集才现原形。
        # 零件提供了 transform_train 就说明它知道这件事，让它自己处理。
        训练变换 = getattr(op, "transform_train", None)
        train = 训练变换(train) if callable(训练变换) else op.transform(train)
        others = [op.transform(df) for df in others]
        if not isinstance(train, pd.DataFrame):
            raise TypeError(f"零件「{name}」的 transform 没有返回 DataFrame")
    return train, others, [c for c in train.columns if c not in before]

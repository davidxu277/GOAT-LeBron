"""配置驱动的官方 KuaiRand FM Trainer。

接口：

    fit(train, valid, seed=0, config=None)
    predict(model_bundle, split)

所有参数从 config 读取。词表、duration 分桶边界等统计量只从
train 拟合，validation/test 只应用 train 统计量。
"""

from __future__ import annotations

import copy
import pathlib
import sys
from typing import Any

import numpy as np
import yaml


# 工兵能调的参数：键路径 → (下限, 上限)。
#
# 上限是**护栏不是建议**：epochs 一个 500 就能把整场的算力预算烧光，
# 而 Agent 自己看不出这是它干的 —— 它只会看到"这一轮特别久"。
# 越界不静默夹回，直接报错说清楚，让复盘官知道这是"提案不合法"
# 而不是"这个方法没用"（两者对卡片信任分的处置完全不同）。
AGENT_PARAM_BOUNDS = {
    ("model", "k"): (1, 64),
    ("model", "learning_rate"): (1e-5, 0.5),
    ("train", "epochs"): (1, 60),
    ("train", "batch_size"): (32, 65536),
    ("train", "early_stopping_patience"): (1, 20),
    ("train", "min_delta"): (0.0, 0.05),
}

# 工兵只准动这两棵子树。data / evaluation 之类一旦被改，
# 跑出来的分数就没法跟前几轮比了 —— 那等于偷偷换了考卷。
AGENT_CONFIG_ROOTS = {"model", "train"}

# 这个 Trainer 训的就是官方 baseline 那个 FM，没有加载 Agent 自写代码的机制。
# ⚠️ 这是**我们这份文件**的限制，不是主办方的规定 —— 赛题恰恰奖励
# 「超越简单基线调参」。要让 Agent 写特征/写模型，换 goat_trainer.py。
AGENT_SUPPORTS_NEW_FILES = False

# apply_agent_patch 攒下来的覆盖值。fit() 会把它合并到任务配置之上。
_AGENT_OVERRIDES: dict[str, Any] = {}


BRIDGE_ROOT = pathlib.Path(
    __file__
).resolve().parents[1]

STARTER_KIT = (
    BRIDGE_ROOT / "official_starter_kit"
)

if str(STARTER_KIT) not in sys.path:
    sys.path.insert(
        0,
        str(STARTER_KIT),
    )

import baseline
import data
import evaluate


REQUIRED_MODEL_CONFIG = (
    "k",
    "learning_rate",
)

REQUIRED_TRAIN_CONFIG = (
    "epochs",
    "batch_size",
    "early_stopping_patience",
    "min_delta",
)


def _require_section(
    config: dict[str, Any],
    section_name: str,
) -> dict[str, Any]:
    """读取并检查配置分区。"""
    section = config.get(
        section_name
    )

    if not isinstance(section, dict):
        raise ValueError(
            f"Trainer 配置缺少 {section_name} 对象"
        )

    return section


def _require_keys(
    section: dict[str, Any],
    keys: tuple[str, ...],
    section_name: str,
) -> None:
    """检查配置必填项。"""
    missing = [
        key
        for key in keys
        if section.get(key) is None
    ]

    if missing:
        raise ValueError(
            f"Trainer 配置 {section_name} "
            f"缺少参数：{missing}"
        )


def _merge(dst: dict, src: dict) -> None:
    """深度合并 —— 浅层赋值会把整棵子树冲掉，
    工兵只想改一个 learning_rate，结果把 k 也抹了。"""
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value


def _check_bounds(config: dict[str, Any]) -> None:
    """越界当场报错，别等训练跑到天亮才发现。"""
    for (root, key), (lo, hi) in AGENT_PARAM_BOUNDS.items():
        value = (config.get(root) or {}).get(key)
        if value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            raise ValueError(
                f"{root}.{key} 写成了 {value!r}，不是数字"
            ) from None
        if not lo <= number <= hi:
            raise ValueError(
                f"{root}.{key} = {value}，超出允许区间 [{lo}, {hi}]。"
                f"这是提案不合法，不是这个方法没用。"
            )


def apply_agent_patch(patch, output_dir) -> None:
    """接收 Agent 的配置补丁 —— Bridge 在每轮训练前调用它。

    没有这个函数时，Bridge 会抛 NotImplementedError（它不肯静默忽略修改），
    于是工兵只要产出任何补丁，整轮就作废 —— 这条路以前一轮都跑不成。

    补丁是**累积重放**的：每轮拿 history 从初始配置重新叠一遍，
    而不是在上一轮结果上继续改。这样任何一轮的配置都能从日志完整复现，
    也不会因为中间某轮被回滚而留下脏状态。
    """
    global _AGENT_OVERRIDES
    _AGENT_OVERRIDES = {}

    for item in patch.get("history") or [patch]:
        if item.get("new_files"):
            raise NotImplementedError(
                "官方 FM Trainer 只接受配置实验 —— 它训的是 baseline.FM，"
                "没有加载 Agent 自己写的模型/特征代码的机制。"
                "要做代码实验，得换一个实现了 new_files 的 Trainer。"
            )
        raw = item.get("config_patch") or ""
        parsed = yaml.safe_load(raw) if isinstance(raw, str) else raw
        if not parsed:
            continue
        if not isinstance(parsed, dict):
            raise ValueError("config_patch 必须是 YAML 键值对")
        unknown = set(parsed) - AGENT_CONFIG_ROOTS
        if unknown:
            raise ValueError(
                f"config_patch 想改 {sorted(unknown)}，"
                f"只准动 {sorted(AGENT_CONFIG_ROOTS)} 这两棵子树。"
            )
        _merge(_AGENT_OVERRIDES, parsed)

    _check_bounds(_AGENT_OVERRIDES)

    # 把这一轮真正生效的覆盖值落盘 —— 日志里看得见，才查得出问题
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent_overrides.yaml").write_text(
        yaml.safe_dump(_AGENT_OVERRIDES, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _read_config(
    config: dict[str, Any] | None,
) -> dict[str, Any]:
    """读取并验证 FM 配置。"""
    # 任务配置打底，Agent 的补丁盖在上面 —— 这一步以前不存在，
    # 所以工兵改的东西根本到不了训练那里
    config = copy.deepcopy(dict(config or {}))
    _merge(config, _AGENT_OVERRIDES)
    _check_bounds(config)

    model_config = _require_section(
        config,
        "model",
    )
    train_config = _require_section(
        config,
        "train",
    )

    _require_keys(
        model_config,
        REQUIRED_MODEL_CONFIG,
        "model",
    )
    _require_keys(
        train_config,
        REQUIRED_TRAIN_CONFIG,
        "train",
    )

    parsed = {
        "k": int(
            model_config["k"]
        ),
        "learning_rate": float(
            model_config["learning_rate"]
        ),
        "epochs": int(
            train_config["epochs"]
        ),
        "batch_size": int(
            train_config["batch_size"]
        ),
        "early_stopping_patience": int(
            train_config[
                "early_stopping_patience"
            ]
        ),
        "min_delta": float(
            train_config["min_delta"]
        ),
    }

    if parsed["k"] < 1:
        raise ValueError(
            "model.k 必须大于等于 1"
        )

    if parsed["learning_rate"] <= 0:
        raise ValueError(
            "model.learning_rate 必须大于 0"
        )

    if parsed["epochs"] < 1:
        raise ValueError(
            "train.epochs 必须大于等于 1"
        )

    if parsed["batch_size"] < 1:
        raise ValueError(
            "train.batch_size 必须大于等于 1"
        )

    if (
        parsed["early_stopping_patience"]
        < 1
    ):
        raise ValueError(
            "train.early_stopping_patience "
            "必须大于等于 1"
        )

    if parsed["min_delta"] < 0:
        raise ValueError(
            "train.min_delta 不能小于 0"
        )

    return parsed


def _safe_rows(rows) -> list[tuple]:
    """隐藏目标 split 标签，防止编码阶段读取 test 标签。"""
    return [
        tuple(row[:6]) + (0,)
        for row in rows
    ]


def _encode_target(
    train_rows: list[tuple],
    target_rows,
):
    """只用 train 拟合词表，并转换目标 split。"""
    encoded, dimension = data.encode({
        "train": train_rows,
        "valid": _safe_rows(
            target_rows
        ),
        "test": [],
    })

    target_features = encoded[
        "valid"
    ][0]

    return target_features, dimension


def fit(
    train,
    valid,
    seed: int = 0,
    config: dict[str, Any] | None = None,
):
    """训练官方 FM，并按 validation primary 早停。"""
    parameters = _read_config(
        config
    )

    k = parameters["k"]
    learning_rate = parameters[
        "learning_rate"
    ]
    epochs = parameters["epochs"]
    batch_size = parameters[
        "batch_size"
    ]
    early_stopping_patience = parameters[
        "early_stopping_patience"
    ]
    min_delta = parameters[
        "min_delta"
    ]

    train_rows = list(
        train.rows
    )
    valid_rows = list(
        valid.rows
    )

    if not train_rows:
        raise ValueError(
            "训练集为空"
        )

    if not valid_rows:
        raise ValueError(
            "验证集为空"
        )

    # data.encode() 中的词表和分桶只从 train 构建。
    encoded, dimension = data.encode({
        "train": train_rows,
        "valid": valid_rows,
        "test": [],
    })

    train_x, train_y, _ = encoded[
        "train"
    ]
    valid_x, valid_y, valid_users = (
        encoded["valid"]
    )

    if train_y is None:
        raise RuntimeError(
            "训练集标签不可用"
        )

    if valid_y is None:
        raise RuntimeError(
            "验证集标签不可用"
        )

    model = baseline.FM(
        dimension,
        k=k,
        lr=learning_rate,
        seed=int(seed),
    )

    rng = np.random.default_rng(
        int(seed)
    )

    best_primary = float("-inf")
    best_state = None
    stale_epochs = 0

    for _epoch in range(epochs):
        order = rng.permutation(
            len(train_y)
        )

        for start in range(
            0,
            len(order),
            batch_size,
        ):
            indices = order[
                start:start + batch_size
            ]

            model.step(
                train_x[indices],
                train_y[indices],
            )

        valid_scores = model.predict(
            valid_x
        )

        valid_metrics = evaluate.evaluate(
            valid_users,
            valid_y,
            valid_scores,
        )

        primary = float(
            valid_metrics["primary"]
        )

        if (
            primary
            > best_primary + min_delta
        ):
            best_primary = primary
            stale_epochs = 0

            best_state = (
                model.V.copy(),
                model.W.copy(),
                np.float32(model.b),
            )
        else:
            stale_epochs += 1

            if (
                stale_epochs
                >= early_stopping_patience
            ):
                break

    if best_state is None:
        raise RuntimeError(
            "FM 训练完成但没有生成有效 checkpoint"
        )

    model.V, model.W, model.b = (
        best_state
    )

    return {
        "model": model,
        "train_rows": train_rows,
        "seed": int(seed),
        "config": parameters,
        "best_validation_primary": (
            best_primary
        ),
    }


def predict(
    model_bundle,
    split,
) -> np.ndarray:
    """使用 train-only 编码器逻辑预测任意 split。"""
    if not isinstance(
        model_bundle,
        dict,
    ):
        raise TypeError(
            "model_bundle 必须是 dict"
        )

    if "model" not in model_bundle:
        raise ValueError(
            "model_bundle 缺少 model"
        )

    if "train_rows" not in model_bundle:
        raise ValueError(
            "model_bundle 缺少 train_rows"
        )

    target_features, _ = _encode_target(
        model_bundle["train_rows"],
        list(split.rows),
    )

    scores = np.asarray(
        model_bundle["model"].predict(
            target_features
        ),
        dtype=float,
    ).reshape(-1)

    if len(scores) != len(split):
        raise ValueError(
            "FM 预测行数与目标 split 不一致"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "FM 预测包含 NaN 或 Inf"
        )

    return scores
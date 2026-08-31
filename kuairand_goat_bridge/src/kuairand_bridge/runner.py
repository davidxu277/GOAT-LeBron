"""KuaiRand Trainer 插件运行器。

负责：

1. 加载 Trainer；
2. 应用 Agent patch；
3. 把 trainer_config 传给 Trainer.fit()；
4. 开发阶段只加载 train/valid；
5. 最终提交阶段才加载无标签 test；
6. 使用官方评估代码计算 validation；
7. test 只生成和检查 submission，不返回分数。
"""

from __future__ import annotations

import importlib.util
import inspect
import pathlib
from types import ModuleType
from typing import Any

import numpy as np

from .dataset import DatasetBundle, load_dataset
from .evaluator import evaluate_predictions


def _load_trainer(
    path: str | pathlib.Path,
) -> ModuleType:
    """加载并检查 Trainer 模块。"""
    trainer_path = pathlib.Path(
        path
    ).expanduser().resolve()

    if not trainer_path.is_file():
        raise FileNotFoundError(
            f"Trainer 文件不存在：{trainer_path}"
        )

    spec = importlib.util.spec_from_file_location(
        "teammate_trainer",
        trainer_path,
    )

    if spec is None or spec.loader is None:
        raise ImportError(
            f"无法加载 Trainer：{trainer_path}"
        )

    trainer = importlib.util.module_from_spec(
        spec
    )
    spec.loader.exec_module(trainer)

    missing = [
        name
        for name in ("fit", "predict")
        if not callable(
            getattr(trainer, name, None)
        )
    ]

    if missing:
        raise TypeError(
            "Trainer 缺少必要函数："
            f"{', '.join(missing)}"
        )

    return trainer


def _normalize_patch(
    agent_patch: dict[str, Any] | None,
) -> dict[str, Any]:
    """补全并复制 Agent patch。"""
    agent_patch = agent_patch or {}

    return {
        "new_files": list(
            agent_patch.get("new_files") or []
        ),
        "config_patch": (
            agent_patch.get("config_patch") or ""
        ),
        "history": list(
            agent_patch.get("history") or []
        ),
    }


def _patch_has_changes(
    patch: dict[str, Any],
) -> bool:
    """检查当前 patch 或历史中是否有实际修改。"""
    if patch.get("new_files"):
        return True

    if patch.get("config_patch"):
        return True

    for item in patch.get("history") or []:
        if item.get("new_files"):
            return True

        if item.get("config_patch"):
            return True

    return False


def _apply_agent_patch(
    trainer: ModuleType,
    patch: dict[str, Any],
    output_dir: pathlib.Path,
) -> None:
    """让 Trainer 应用 Agent 修改。"""
    if not _patch_has_changes(patch):
        return

    hook = getattr(
        trainer,
        "apply_agent_patch",
        None,
    )

    if not callable(hook):
        raise NotImplementedError(
            "本轮包含 Agent 修改，但 Trainer "
            "没有实现 apply_agent_patch(patch, output_dir)。"
            "Bridge 不会静默忽略修改。"
        )

    hook(
        patch,
        output_dir,
    )


def _fit_trainer(
    trainer: ModuleType,
    dataset: DatasetBundle,
    *,
    seed: int,
    trainer_config: dict[str, Any],
):
    """调用 Trainer.fit() 并传入配置。"""
    fit_function = trainer.fit
    signature = inspect.signature(
        fit_function
    )
    parameters = signature.parameters

    if "config" not in parameters:
        if trainer_config:
            raise TypeError(
                "任务配置提供了 trainer_config，"
                "但 Trainer.fit() 没有 config 参数。"
                "接口必须为："
                "fit(train, valid, seed=0, config=None)"
            )

        return fit_function(
            dataset.train,
            dataset.valid,
            seed=int(seed),
        )

    return fit_function(
        dataset.train,
        dataset.valid,
        seed=int(seed),
        config=dict(trainer_config),
    )


def _predict(
    trainer: ModuleType,
    model: Any,
    split,
    *,
    split_name: str,
) -> np.ndarray:
    """调用 predict() 并检查结果。"""
    scores = np.asarray(
        trainer.predict(
            model,
            split,
        ),
        dtype=float,
    ).reshape(-1)

    if len(scores) != len(split):
        raise ValueError(
            f"{split_name} 预测行数错误："
            f"预测={len(scores):,}，"
            f"数据={len(split):,}"
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            f"{split_name} 预测包含 NaN 或 Inf"
        )

    return scores


def _save_scores(
    path: pathlib.Path,
    scores: np.ndarray,
) -> pathlib.Path:
    """保存预测分数。"""
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    np.save(path, scores)
    return path.resolve()


def run_trainer(
    data_dir: str | pathlib.Path,
    trainer_path: str | pathlib.Path,
    output_dir: str | pathlib.Path = "output",
    seed: int = 0,
    make_test: bool = False,
    agent_patch: dict[str, Any] | None = None,
    trainer_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """训练并评估一个 Trainer。"""
    work_dir = pathlib.Path(
        output_dir
    ).expanduser().resolve()

    work_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # make_test=False 时完全不加载 test。
    dataset = load_dataset(
        data_dir,
        include_test=bool(make_test),
    )

    trainer = _load_trainer(
        trainer_path
    )

    patch = _normalize_patch(
        agent_patch
    )

    _apply_agent_patch(
        trainer,
        patch,
        work_dir,
    )

    model = _fit_trainer(
        trainer,
        dataset,
        seed=int(seed),
        trainer_config=dict(
            trainer_config or {}
        ),
    )

    valid_scores = _predict(
        trainer,
        model,
        dataset.valid,
        split_name="validation",
    )

    valid_path = _save_scores(
        work_dir / "valid_scores.npy",
        valid_scores,
    )

    result: dict[str, Any] = {
        "validation": evaluate_predictions(
            dataset,
            valid_path,
            split="valid",
            output_dir=work_dir,
        )
    }

    if not make_test:
        return result

    if dataset.test is None:
        raise RuntimeError(
            "make_test=True，但数据加载器没有返回 test"
        )

    if dataset.test.expose_labels:
        raise PermissionError(
            "test SplitView 不得暴露标签"
        )

    for row_index, row in enumerate(
        dataset.test.rows
    ):
        if len(row) <= 6:
            raise ValueError(
                f"test 第 {row_index} 行结构不完整"
            )

        if row[6] is not None:
            raise PermissionError(
                "test 标签没有清除："
                f"第 {row_index} 行标签不是 None"
            )

    test_scores = _predict(
        trainer,
        model,
        dataset.test,
        split_name="test",
    )

    test_path = _save_scores(
        work_dir / "test_scores.npy",
        test_scores,
    )

    result["test"] = evaluate_predictions(
        dataset,
        test_path,
        split="test",
        output_dir=work_dir,
    )

    if "metrics" in result["test"]:
        raise RuntimeError(
            "test 结果意外包含 metrics；"
            "隐藏测试集不能在本地评分"
        )

    return result
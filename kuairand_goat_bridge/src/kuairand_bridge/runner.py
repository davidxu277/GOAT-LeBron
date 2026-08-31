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

from . import diagnostics
from .dataset import DatasetBundle, FIDELITY_FRACTIONS, SplitView, load_dataset
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


def validate_trainer(path: str | pathlib.Path) -> None:
    """在 dry-run 阶段真实导入 Trainer，提前暴露缺失依赖和接口错误。"""
    _load_trainer(path)


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


def _plain(value: Any) -> Any:
    """把 Trainer 诊断中的 NumPy 值整理成可跨进程、可写 JSON 的值。"""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _training_diagnostics(model: Any, dataset: DatasetBundle, fidelity: str,
                          full_train_rows: int) -> dict[str, Any]:
    """从不同 Trainer 的模型包中提取统一、轻量的训练诊断。"""
    diagnostics: dict[str, Any] = {
        "fidelity": fidelity,
        "训练集抽样比例": FIDELITY_FRACTIONS[fidelity],
        "训练行数": len(dataset.train),
        "全量训练行数": int(full_train_rows),
        "验证行数": len(dataset.valid),
    }
    if not isinstance(model, dict):
        return diagnostics

    record = model.get("训练记录")
    if isinstance(record, dict):
        diagnostics["每轮训练记录"] = _plain(record)
    if model.get("best_validation_primary") is not None:
        diagnostics["最佳验证Primary"] = float(model["best_validation_primary"])
    if isinstance(model.get("config"), dict):
        diagnostics["实际生效配置"] = _plain(model["config"])
    if isinstance(model.get("fields"), (list, tuple)):
        diagnostics["实际特征"] = [str(value) for value in model["fields"]]
    if isinstance(model.get("装上的零件"), (list, tuple)):
        diagnostics["装上的零件"] = [str(value) for value in model["装上的零件"]]
    return diagnostics



def _train_self_evaluation(
    trainer: ModuleType,
    model: Any,
    train: SplitView,
    *,
    seed: int,
    max_rows: int = diagnostics.DEFAULT_TRAIN_EVAL_ROWS,
) -> dict[str, Any] | None:
    """在训练集上再出一次预测，用来判「在背题」。

    没有这一块，医生手里只有验证集一个数字，训练/验证差算不出来，
    CLAUDE.md 写死的那道「差值 > 0.15 必须报病」的闸门也永远不会响。

    只做 predict，绝不重新 fit。按整个用户抽样（GAUC/nDCG 是用户内指标，
    随机抽行会把用户的曝光列表撕碎，算出来跟验证集不可比）。

    失败不许把整轮拖垮 —— 它是诊断证据，不是成绩。拿不到就返回 None，
    成绩单里这一块直接不出现，医生看到"没有"比看到一个假数字安全。
    """
    try:
        rows = diagnostics.sample_rows_by_user(
            train.rows,
            max_rows,
            seed,
        )
        view = SplitView("train", rows, True)

        scores = _predict(
            trainer,
            model,
            view,
            split_name="train-self-eval",
        )

        metrics = diagnostics._evaluate(
            [row[diagnostics.ROW_USER] for row in rows],
            [int(row[diagnostics.ROW_LABEL]) for row in rows],
            scores,
        )

    except Exception as exc:                      # noqa: BLE001
        return {
            "取不到": f"{type(exc).__name__}: {exc}",
            "说明": "训练集自评失败，本轮无法判断「在背题」",
        }

    return {
        "GAUC": round(float(metrics["GAUC"]), 4),
        "nDCG@5": round(float(metrics["nDCG@5"]), 4),
        "主分": round(float(metrics["primary"]), 4),
        "总行数": len(rows),
        "用户数": int(metrics["users"]),
        "抽样说明": (
            f"训练集全部 {len(rows):,} 行；只做预测，没有重新训练"
            if len(rows) == len(train.rows) else
            f"从训练集 {len(train.rows):,} 行里按整个用户抽了 {len(rows):,} 行；"
            "只做预测，没有重新训练"
        ),
        "口径提醒": (
            "走的是推理路径（零件的 transform）。训练时若用过折外编码，"
            "这个分数会比模型真正见过的略高"
        ),
    }


def run_trainer(
    data_dir: str | pathlib.Path,
    trainer_path: str | pathlib.Path,
    output_dir: str | pathlib.Path = "output",
    seed: int = 0,
    make_test: bool = False,
    agent_patch: dict[str, Any] | None = None,
    trainer_config: dict[str, Any] | None = None,
    fidelity: str = "全量",
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
    full_dataset = load_dataset(
        data_dir,
        include_test=bool(make_test),
    )
    # 最终提交必须用全量训练集；开发轮次才按 GOAT 的 fidelity 梯子缩放。
    effective_fidelity = "全量" if make_test else str(fidelity)
    dataset = full_dataset.with_train_fidelity(effective_fidelity, int(seed))

    trainer = _load_trainer(
        trainer_path
    )

    patch = _normalize_patch(
        agent_patch
    )

    try:
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
            ),
            "training": _training_diagnostics(
                model,
                dataset,
                effective_fidelity,
                len(full_dataset.train),
            ),
        }

        # 医生判 6 个病靠的是分组之后的数字，不是验证集总分。
        # 这一块失败不能拖垮整轮 —— 分数已经拿到了，证据缺就缺，明说。
        try:
            result["diagnostics"] = diagnostics.build(
                train_rows=dataset.train.rows,
                valid_rows=dataset.valid.rows,
                valid_scores=valid_scores,
                train_eval=_train_self_evaluation(
                    trainer,
                    model,
                    dataset.train,
                    seed=int(seed),
                ),
            )
        except Exception as exc:                  # noqa: BLE001
            result["diagnostics"] = {
                "取不到": f"{type(exc).__name__}: {exc}",
                "说明": "分组证据生成失败，本轮医生只能看验证集总分",
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
    finally:
        cleanup = getattr(trainer, "cleanup_agent_patch", None)
        if callable(cleanup):
            cleanup()

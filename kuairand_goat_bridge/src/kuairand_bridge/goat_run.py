"""从一个YAML配置启动完整KuaiRand GOAT运行。"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import pathlib
import sys
import time
from typing import Any

import yaml

from .goat_executor import (
    KuaiRandGoatExecutor,
    assert_goat_compatible,
)


BRIDGE_ROOT = pathlib.Path(
    __file__
).resolve().parents[2]

OFFICIAL_EPSILON = 0.002
OFFICIAL_PATIENCE = 3
OFFICIAL_MAX_ITERATIONS = 50
OFFICIAL_MAX_SECONDS = 21600


def _resolve(
    value: str,
    base: pathlib.Path,
) -> pathlib.Path:
    path = pathlib.Path(
        value
    ).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        base / path
    ).resolve()


def _mapping(
    value: Any,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{name}必须是YAML对象"
        )

    return value


def _official_validation(
    config: dict[str, Any],
) -> dict[str, float]:
    baseline = _mapping(
        config.get("official_baseline"),
        "official_baseline",
    )
    validation = _mapping(
        baseline.get("validation"),
        "official_baseline.validation",
    )

    required = (
        "GAUC",
        "nDCG@5",
        "primary",
    )

    missing = [
        key
        for key in required
        if validation.get(key) is None
    ]

    if missing:
        raise ValueError(
            "official baseline缺少："
            f"{missing}"
        )

    result = {
        key: float(validation[key])
        for key in required
    }

    expected = (
        result["GAUC"]
        + result["nDCG@5"]
    ) / 2.0

    if abs(
        result["primary"] - expected
    ) > 1e-10:
        raise ValueError(
            "official baseline primary "
            "必须等于GAUC和nDCG@5的均值"
        )

    return result


def load_task(
    path: str | pathlib.Path,
) -> dict[str, Any]:
    task_path = pathlib.Path(
        path
    ).expanduser().resolve()

    if not task_path.is_file():
        raise FileNotFoundError(
            f"任务配置不存在：{task_path}"
        )

    config = yaml.safe_load(
        task_path.read_text(
            encoding="utf-8"
        )
    ) or {}

    if not isinstance(config, dict):
        raise ValueError(
            "任务配置顶层必须是对象"
        )

    for key in (
        "data_dir",
        "trainer",
        "output_dir",
    ):
        if not config.get(key):
            raise ValueError(
                f"任务配置缺少：{key}"
            )

    config["data_dir"] = str(
        _resolve(
            config["data_dir"],
            task_path.parent,
        )
    )
    config["trainer"] = str(
        _resolve(
            config["trainer"],
            BRIDGE_ROOT,
        )
    )
    config["output_dir"] = str(
        _resolve(
            config["output_dir"],
            BRIDGE_ROOT,
        )
    )

    config["seed"] = int(
        config.get("seed", 0)
    )
    config["max_iterations"] = int(
        config.get("max_iterations", 50)
    )
    config["epsilon"] = float(
        config.get("epsilon", 0.002)
    )
    config["patience"] = int(
        config.get("patience", 3)
    )
    config["max_wall_seconds"] = int(
        config.get(
            "max_wall_seconds",
            21600,
        )
    )
    config["token_budget"] = int(
        config.get(
            "token_budget",
            2_000_000,
        )
    )
    config[
        "generate_test_after_convergence"
    ] = bool(
        config.get(
            "generate_test_after_convergence",
            True,
        )
    )

    if not (
        1
        <= config["max_iterations"]
        <= OFFICIAL_MAX_ITERATIONS
    ):
        raise ValueError(
            "max_iterations必须在1到50之间"
        )

    if (
        config["epsilon"]
        != OFFICIAL_EPSILON
    ):
        raise ValueError(
            "epsilon必须是0.002"
        )

    if (
        config["patience"]
        != OFFICIAL_PATIENCE
    ):
        raise ValueError(
            "patience必须是3"
        )

    if not (
        1
        <= config["max_wall_seconds"]
        <= OFFICIAL_MAX_SECONDS
    ):
        raise ValueError(
            "max_wall_seconds必须在1到21600之间"
        )

    trainer_config = config.get(
        "trainer_config"
    )

    if not isinstance(
        trainer_config,
        dict,
    ):
        raise ValueError(
            "任务配置必须包含trainer_config对象"
        )

    config["trainer_config"] = (
        trainer_config
    )

    _official_validation(config)

    return config


def _goat_root() -> pathlib.Path:
    root = BRIDGE_ROOT.parent

    if not (
        root / "agent" / "loop.py"
    ).is_file():
        raise FileNotFoundError(
            "未找到agent/loop.py"
        )

    if str(root) not in sys.path:
        sys.path.insert(
            0,
            str(root),
        )

    return root


def validate_task(
    config_path: str | pathlib.Path,
) -> dict[str, Any]:
    config = load_task(
        config_path
    )

    if not pathlib.Path(
        config["data_dir"]
    ).is_dir():
        raise FileNotFoundError(
            f"data_dir不存在："
            f"{config['data_dir']}"
        )

    if not pathlib.Path(
        config["trainer"]
    ).is_file():
        raise FileNotFoundError(
            f"trainer不存在："
            f"{config['trainer']}"
        )

    _goat_root()
    return config


def run(
    config_path: str | pathlib.Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    config = validate_task(
        config_path
    )
    official = _official_validation(
        config
    )

    baseline_config = config[
        "official_baseline"
    ]
    tolerance = float(
        baseline_config.get(
            "reproduction_tolerance",
            0.003,
        )
    )

    if dry_run:
        return {
            "status": "ready",
            "config": config,
            "official_baseline_validation": (
                official
            ),
        }

    run_started = time.monotonic()

    _goat_root()

    from agent.cli import make_llm
    from agent.knowledge import (
        CardLibrary,
        SymptomVocab,
    )
    from agent.loop import run_session

    # 病名词表与药方卡用主仓库那一套（knowledge/），不再单独维护一份。
    # 08-31 之前这里指向 bridge 内的 goat_profile/，那份只有 9 病 3 卡，
    # 而主仓库那套是从 AliCCP 一路积累下来、按新任务改过口径的 12 病 14 卡。
    # 两份并存的代价是：改一边、忘一边，军师看到的永远是没人维护的那份。
    profile = _goat_root()
    vocab = SymptomVocab.load(
        profile / "knowledge" / "symptoms.yaml"
    )
    cards = CardLibrary.load(
        vocab,
        profile / "knowledge" / "cards",
    )

    output = pathlib.Path(
        config["output_dir"]
    )
    logs = output / "logs"

    output.mkdir(
        parents=True,
        exist_ok=True,
    )
    logs.mkdir(
        parents=True,
        exist_ok=True,
    )

    executor = KuaiRandGoatExecutor(
        data_dir=config["data_dir"],
        trainer_path=config["trainer"],
        output_dir=(
            output / "rounds"
        ),
        seed=config["seed"],
        max_seconds=(
            config["max_wall_seconds"]
        ),
        max_iterations=(
            config["max_iterations"]
        ),
        trainer_config=config.get(
            "trainer_config",
            {},
        ),
    )

    assert_goat_compatible(executor)

    first = executor.run(
        {
            "new_files": [],
            "config_patch": "",
        },
        "全量",
    )

    if not first.ok:
        raise RuntimeError(
            f"第0轮baseline失败：{first.error}"
        )

    actual_primary = float(
        first.health_report[
            "验证集"
        ]["主分"]
    )
    expected_primary = float(
        official["primary"]
    )

    baseline_passed = (
        abs(
            actual_primary
            - expected_primary
        )
        <= tolerance
    )

    if not baseline_passed:
        raise RuntimeError(
            "官方baseline复现失败："
            f"expected={expected_primary:.5f}, "
            f"actual={actual_primary:.5f}, "
            f"tolerance={tolerance:.5f}"
        )

    pipeline = (
        BRIDGE_ROOT
        / "configs"
        / "pipeline.yaml"
    ).read_text(
        encoding="utf-8"
    )
    interface = (
        profile
        / "model_interface.py.txt"
    ).read_text(
        encoding="utf-8"
    )
    example = (
        BRIDGE_ROOT
        / "examples"
        / "tunable_popularity_trainer.py"
    ).read_text(
        encoding="utf-8"
    )

    llm = make_llm()

    reserve_final = (
        1
        if config[
            "generate_test_after_convergence"
        ]
        else 0
    )

    research_rounds = max(
        0,
        config["max_iterations"]
        - 1
        - reserve_final,
    )

    summary = run_session(
        llm=llm,
        vocab=vocab,
        cards=cards,
        executor=executor,
        initial_report=(
            first.health_report
        ),
        initial_train_seconds=(
            first.seconds
        ),
        module_interface=interface,
        example_module=example,
        current_config=pipeline,
        rounds=research_rounds,
        token_budget=(
            config["token_budget"]
        ),
        epsilon=config["epsilon"],
        patience=config["patience"],
        baseline={
            "点击AUC": official["GAUC"],
            "购买AUC": official["nDCG@5"],
        },
        logs_dir=logs,
    )

    executor.select_round(
        int(summary.best_round)
    )

    final = None

    if config[
        "generate_test_after_convergence"
    ]:
        final = (
            executor.make_final_submission()
        )

        if not final.ok:
            raise RuntimeError(
                f"最终提交失败：{final.error}"
            )

    summary_data = (
        asdict(summary)
        if is_dataclass(summary)
        else dict(summary)
    )

    result = {
        "status": "complete",
        "best_round": summary.best_round,
        "best_scores": summary.best_scores,
        "stopped_because": (
            summary.stopped_because
        ),
        "rounds_run": summary.rounds_run,
        "total_tokens": summary.total_tokens,
        "wall_seconds": (
            time.monotonic()
            - run_started
        ),
        "training_attempts": (
            executor.training_attempts
        ),
        "max_iterations": (
            executor.max_iterations
        ),
        "convergence": {
            "metric": "primary",
            "epsilon": config["epsilon"],
            "patience": config["patience"],
        },
        "official_baseline_validation": (
            official
        ),
        "baseline_reproduction": {
            "expected_primary": (
                expected_primary
            ),
            "actual_primary": (
                actual_primary
            ),
            "tolerance": tolerance,
            "passed": baseline_passed,
        },
        "submission": (
            final.health_report.get(
                "最终提交"
            )
            if final is not None
            else None
        ),
        "goat_session_summary": (
            summary_data
        ),
    }

    summary_path = (
        output / "final_summary.json"
    )
    summary_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return result

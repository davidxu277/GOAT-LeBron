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


def _track2_read_scores(report: dict[str, Any]) -> dict[str, float]:
    """把 Track 2 正式指标翻译成旧 GOAT 账本键；不污染健康报告。"""
    validation = report.get("验证集") or {}
    if validation.get("GAUC") is None:
        return {}
    return {
        "点击AUC": float(validation["GAUC"]),
        "购买AUC": float(validation.get("nDCG@5") or 0.0),
    }


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
    config["require_baseline_reproduction"] = bool(
        config.get("require_baseline_reproduction", False)
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

    # dry-run 不能只检查文件存在：真实导入一次，提前暴露 pandas/torch 等
    # 运行依赖缺失，以及 fit/predict 接口不完整的问题。
    from .runner import validate_trainer
    validate_trainer(config["trainer"])

    _goat_root()
    return config


def _best_executor_round(
    best_report: dict[str, Any],
    source: pathlib.Path,
) -> int:
    """最佳那一轮在执行器里的编号。

    为什么不能用 ``summary.best_round``：那是 **Agent 的轮次编号**，
    而执行器数的是自己 ``_patch_history`` 的下标，两者会往两个方向漂 ——

      · 第 0 轮基线、升档重测：调了 executor.run，但不是 Agent 轮次
      · 医生 no_finding 直接跳过、军师或工兵失败：那一轮压根没调 executor.run

    拿 Agent 的编号去 select_round，选中的是**另一轮**，而且
    select_round 只检查下标越不越界 —— 选错了不报错，照样跑完、
    照样出提交文件，交上去的却是另一个模型。

    取不到就**当场报错**。这一步之后就是生成最终提交，宁可在这里停下，
    也不能默默交一个不知道是哪一轮的版本。
    """
    budget = best_report.get("运行预算")

    if not isinstance(budget, dict) or budget.get("执行器轮次") is None:
        raise KeyError(
            f"{source} 里没有「运行预算 → 执行器轮次」，"
            "无法确定最佳轮在执行器里的编号。"
            "这份 best_report.json 可能是旧版本执行器写的；"
            "请重新跑一场，或手动 select_round 后再生成提交。"
        )

    return int(budget["执行器轮次"])


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
    from agent import loop as goat_loop

    # 旧 GOAT 的内部账本仍把两个分量命名为点击AUC/购买AUC。只在本次
    # Bridge 进程内部安装读取适配器，不把这些错误业务名写进 Doctor 成绩单。
    # 这样核心调度/总结保持兼容，LLM 看到的始终只有 Track 2 正式语义。
    goat_loop.read_scores = _track2_read_scores
    run_session = goat_loop.run_session

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
        official_baseline=official,
    )

    assert_goat_compatible(executor)

    initial_fidelity = (
        "全量" if config["require_baseline_reproduction"] else "小份"
    )
    first = executor.run(
        {
            "new_files": [],
            "config_patch": "",
        },
        initial_fidelity,
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

    if config["require_baseline_reproduction"] and not baseline_passed:
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
        / "modules"
        / "base.py"
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
        start_fidelity=initial_fidelity,
        # 键名必须跟 read_scores 读出来的一致，否则「相对官方基线」
        # 这一栏取不到交集，结果表上是一片空白。
        baseline={
            "GAUC": official["GAUC"],
            "nDCG@5": official["nDCG@5"],
        },
        logs_dir=logs,
    )

    best_report_path = (
        logs / "best_report.json"
    )

    best_report = json.loads(
        best_report_path.read_text(
            encoding="utf-8"
        )
    )

    executor.select_round(
        _best_executor_round(
            best_report,
            best_report_path,
        )
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

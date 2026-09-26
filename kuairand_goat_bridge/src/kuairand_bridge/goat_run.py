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
    config["require_baseline_reproduction"] = bool(
        config.get("require_baseline_reproduction", False)
    )
    # 验证集按用户切多少当锁定集（见 kuairand_bridge/dataset.py with_valid_part）。
    # 0 = 不切。开着时 Agent 所有决策只看开发集，整场结束才考一次锁定集。
    config["holdout_users"] = float(config.get("holdout_users", 0.0) or 0.0)
    if not 0.0 <= config["holdout_users"] <= 0.5:
        raise ValueError(
            f"holdout_users 要在 [0, 0.5] 之间，收到 {config['holdout_users']}"
            "（切太多开发集就太小，每一轮的分数抖得没法用）")
    if config["holdout_users"] and config["require_baseline_reproduction"]:
        raise ValueError(
            "锁定集和 require_baseline_reproduction 不能同时开：复现官方基线比的是"
            "全量验证集上的主分，开了锁定集之后每一轮只在开发集上打分，两个数不可比。"
            "要复现基线用 configs/fm_baseline.yaml，Agent 正式跑再开锁定集")

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


def _implementer_materials(profile: pathlib.Path) -> tuple[str, Any]:
    """工兵看到的两份材料：零件接口说明 + 按方案环节取的范文。

    以前范文写死成 examples/tunable_popularity_trainer.py —— 那是 KuaiRand 只能
    调配置时的热度 trainer。换成能写零件的 goat_trainer 之后，同一个函数里的
    「流水线说明」修好了，范文却没人回来改：工兵提示词里「范文：一个现成的零件」
    那一节放的是一个只有 model.prior 旋钮、跟零件接口毫无关系的脚本。
    现在跟 agent.cli 用同一个 example_for，只有一份。
    """
    from agent.roles import example_for

    interface = (profile / "modules" / "base.py").read_text(encoding="utf-8")
    return interface, example_for


def _render_pipeline(
    config: dict[str, Any],
    executor: "KuaiRandGoatExecutor",
) -> str:
    """军师和工兵看到的「当前流水线」。

    三块都来自**真实来源**，没有一处是手写的副本：

      任务      —— 任务配置里的固定口径
      现在的配置 —— 执行器里真正生效的那份（初始 + 历轮已接受的改动）
      能改什么   —— Trainer 自己声明的（能不能写新零件、哪些子树、参数范围）

    最后一块尤其重要。真跑里军师连烧 5 轮才自己摸出「写新文件会被拒」，
    每一轮都是一次真训练加四次大模型调用。这是**已知事实**，
    不该让它花钱试出来。
    """
    能力 = executor.agent_capabilities()

    block = {
        "任务": {
            "数据集": "KuaiRand-Pure",
            "标签": "long_view",
            "排序范围": "用户内（within_user）",
            "正式指标": ["GAUC", "nDCG@5"],
            "主分": "(GAUC + nDCG@5) / 2",
        },
        "现在真正生效的配置": (
            executor.effective_trainer_config()
        ),
        "这一场你能改什么": {
            **能力,
            "说明": (
                "「可以写新零件=False」意味着任何带 new_files 的提案都会被"
                "当场拒掉、整轮作废 —— 这一场只能提配置改动。"
                "这是当前 Trainer 的限制，不是方法本身不行。"
                if 能力 and not 能力.get("可以写新零件")
                else "可以往 modules/ 下写 FeatureOp / ModelOp / TrainOp，"
                     "写完自动进流水线。"
            ),
        },
        "收敛判定": {
            "epsilon": config["epsilon"],
            "patience": config["patience"],
        },
    }

    return yaml.safe_dump(
        block,
        allow_unicode=True,
        sort_keys=False,
    )


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


def _holdout_summary(verdict: dict[str, Any]) -> dict[str, Any]:
    """大考结果 → 开发集涨了多少、锁定集涨了多少、差出来多少。

    「挑出来的运气」= 开发集涨幅 − 锁定集涨幅。开发集被反复看了几十轮，
    最佳轮是在它上面挑出来的，所以开发集上的涨幅里混着恰好迎合它的部分；
    锁定集从没被任何决策看过，它上面的涨幅才是能带走的那部分。
    """
    各轮 = verdict["各轮"]
    起点, 终点 = 各轮[0], 各轮[-1]
    指标 = list(起点["开发集"])

    def 涨(集: str) -> dict[str, float]:
        return {m: round(终点[集][m] - 起点[集][m], 6) for m in 指标}

    开发涨, 锁定涨 = 涨("开发集"), 涨("锁定集")
    return {
        "保真度": verdict.get("保真度"),
        "锁定集比例": verdict.get("锁定集比例"),
        "比的是": f"执行器第 {起点['执行器轮次']} 轮 → 第 {终点['执行器轮次']} 轮",
        "开发集涨了": 开发涨,
        "锁定集涨了": 锁定涨,
        "挑出来的运气": {m: round(开发涨[m] - 锁定涨[m], 6) for m in 指标},
        "最佳轮锁定集分数": dict(终点["锁定集"]),
        "各轮": 各轮,
        "怎么读": ("开发集被反复看了几十轮，最佳轮是在它上面挑出来的；锁定集从没参与任何决策。"
                "锁定集涨了多少才是能带走的提升，「挑出来的运气」越大，说明涨的分里迎合开发集的成分越多。"
                "两边都只训了一次，单次抖动见 noise_bands.json —— 小于噪声带的差距别当真。"),
    }


def _noise_bands_for(logs: pathlib.Path, seed: int) -> dict[str, Any] | None:
    """这一场用哪份噪声带。没量过 = None，run_session 会退回兜底门槛并写进结果表。

    档位对不上由 run_session 自己核对（那种一定作废）。这里只管抽样种子：
    带子量的是训练随机性，换一份同规模的数据子集幅度差不多，照用 —— 但要说出来。
    量法见 kuairand_bridge/noise.py，命令是 `python -m kuairand_bridge noise`。
    """
    from .noise import load_bands

    bands = load_bands(logs)
    if bands is None:
        return None
    量时 = bands.get("数据抽样种子")
    if 量时 is not None and int(量时) != int(seed):
        print(f"  ↳ 噪声带量在抽样种子 {量时} 的数据子集上，这一场是 {seed}。"
              "带子量的是训练随机性，同规模的子集幅度差不多，照用")
    return bands


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

    # 以前这里把 goat_loop.read_scores 换成一个改名适配器（GAUC/nDCG@5 → 点击AUC/购买AUC），
    # 理由是"旧 GOAT 的内部账本仍用这两个名字"。外层循环早就直接认 GAUC/nDCG@5 了，
    # 那个适配器只剩副作用：下面传进去的官方基线用 GAUC/nDCG@5，读出来的分数却叫
    # 点击AUC/购买AUC，结果表的「相对官方基线」按 0 分去减，每一场都报 −0.67。
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
        holdout_frac=config["holdout_users"],
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

    # ⚠️ 以前这里读的是 configs/pipeline.yaml —— 那是**另一个文件**，
    # 跟真正在跑的 trainer_config 毫无关系。实测那份写着
    # `model: item_popularity, prior: 20.0`，而真跑的是 FM(k=16, lr=0.001)。
    # 军师于是整轮整轮地推理一个根本不存在的模型：该不该给
    # item_popularity 上 SWA、prior 能不能调到 200 —— 而那个键在 FM Trainer
    # 里压根没有，提上去必被拒。开药的人拿到的是别人的病历。
    interface, example = _implementer_materials(profile)

    pipeline = _render_pipeline(
        config,
        executor,
    )

    llm = make_llm()

    reserve_final = (
        1
        if config[
            "generate_test_after_convergence"
        ]
        else 0
    )

    # 锁定集大考要训两次（第 0 轮 + 最佳轮），名额先留出来
    reserve_holdout = 2 if config["holdout_users"] else 0

    research_rounds = max(
        0,
        config["max_iterations"]
        - 1
        - reserve_final
        - reserve_holdout,
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
        noise_bands=_noise_bands_for(logs, config["seed"]),
    )

    best_report_path = (
        logs / "best_report.json"
    )

    best_report = json.loads(
        best_report_path.read_text(
            encoding="utf-8"
        )
    )

    best_executor_round = _best_executor_round(
        best_report,
        best_report_path,
    )
    executor.select_round(best_executor_round)

    # ── 锁定集大考：整场只在这里考一次，结果不回流任何决策 ──
    holdout = None
    if config["holdout_users"]:
        verdict = executor.holdout_verdict(
            sorted({0, best_executor_round}),
            summary.best_fidelity or initial_fidelity,
        )
        if verdict.ok:
            holdout = _holdout_summary(verdict.health_report)
            summary.holdout_scores = {
                k: v for k, v in holdout["最佳轮锁定集分数"].items() if k != "主分"}
            运气 = holdout["挑出来的运气"]
            summary.holdout_luck = {k: v for k, v in 运气.items() if k != "主分"}
            summary.holdout_note = (
                f"锁定集 {config['holdout_users']:.0%} 用户，{holdout['比的是']}："
                + "；".join(f"{m} 开发集 {holdout['开发集涨了'][m]:+.4f} / "
                           f"锁定集 {holdout['锁定集涨了'][m]:+.4f} / 运气 {运气[m]:+.4f}"
                           for m in 运气))
            (logs / "holdout_report.json").write_text(
                json.dumps(holdout, ensure_ascii=False, indent=1), encoding="utf-8")
        else:
            summary.holdout_note = f"锁定集大考没考成：{verdict.error}"
        summary.dump(logs / "session_summary.json")

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
        "holdout": holdout,
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

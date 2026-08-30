"""One-config/one-command launcher connecting Bridge to GOAT's run_session."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import os
import pathlib
import signal
import sys
import time
from typing import Any

import yaml

from .goat_executor import KuaiRandGoatExecutor, assert_goat_compatible


BRIDGE_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _resolve(value: str, base: pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_task(path: str | pathlib.Path) -> dict[str, Any]:
    task_path = pathlib.Path(path).expanduser().resolve()
    raw = yaml.safe_load(task_path.read_text(encoding="utf-8")) or {}
    required = ["data_dir", "trainer", "output_dir"]
    missing = [key for key in required if not raw.get(key)]
    if missing:
        raise ValueError(f"任务配置缺少：{missing}")
    raw["data_dir"] = str(_resolve(raw["data_dir"], task_path.parent))
    raw["trainer"] = str(_resolve(raw["trainer"], BRIDGE_ROOT))
    raw["output_dir"] = str(_resolve(raw["output_dir"], BRIDGE_ROOT))
    rounds = int(raw.get("max_rounds", 50))
    if not 1 <= rounds <= 50:
        raise ValueError("max_rounds必须在1到官方上限50之间")
    if float(raw.get("epsilon", 0.002)) != 0.002:
        raise ValueError("正式KuaiRand任务的epsilon必须是官方0.002")
    if int(raw.get("patience", 3)) != 3:
        raise ValueError("正式KuaiRand任务的patience必须是官方3")
    if int(raw.get("max_wall_seconds", 21600)) > 21600:
        raise ValueError("max_wall_seconds不能超过官方6小时（21600秒）")
    return raw


def _goat_root() -> pathlib.Path:
    # Expected layout: GOAT-LeBron/kuairand_goat_bridge/...
    root = BRIDGE_ROOT.parent
    if not (root / "agent" / "loop.py").exists():
        raise FileNotFoundError(
            "Bridge必须位于GOAT-LeBron仓库内，旁边应存在agent/loop.py")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def validate_task(config_path: str | pathlib.Path) -> dict[str, Any]:
    cfg = load_task(config_path)
    for key in ("data_dir", "trainer"):
        if not pathlib.Path(cfg[key]).exists():
            raise FileNotFoundError(f"{key}不存在：{cfg[key]}")
    _goat_root()
    return cfg


def run(config_path: str | pathlib.Path, dry_run: bool = False) -> dict[str, Any]:
    cfg = validate_task(config_path)
    if dry_run:
        return {"status": "ready", "config": cfg,
                "message": "路径、官方参数和GOAT目录检查通过；未调用LLM或训练"}

    _goat_root()
    from agent.cli import make_llm
    from agent.knowledge import CardLibrary, SymptomVocab
    from agent.loop import run_session

    profile = BRIDGE_ROOT / "goat_profile"
    vocab = SymptomVocab.load(profile / "symptoms.yaml")
    cards = CardLibrary.load(vocab, profile / "cards")
    output = pathlib.Path(cfg["output_dir"])
    logs = output / "logs"
    output.mkdir(parents=True, exist_ok=True)

    executor = KuaiRandGoatExecutor(
        cfg["data_dir"], cfg["trainer"], output / "rounds",
        seed=int(cfg.get("seed", 0)),
        max_seconds=int(cfg.get("max_wall_seconds", 21600)),
    )
    assert_goat_compatible(executor)
    first = executor.run({"new_files": [], "config_patch": ""}, "全量")
    if not first.ok:
        raise RuntimeError(f"第0轮基线失败：{first.error}")

    pipeline = (BRIDGE_ROOT / "configs" / "pipeline.yaml").read_text(encoding="utf-8")
    interface = (profile / "model_interface.py.txt").read_text(encoding="utf-8")
    example = (BRIDGE_ROOT / "examples" / "tunable_popularity_trainer.py").read_text(
        encoding="utf-8")
    llm = make_llm()

    old_handler = None
    wall = int(cfg.get("max_wall_seconds", 21600))
    session_started = time.monotonic()
    try:
        if hasattr(signal, "SIGALRM"):
            old_handler = signal.signal(
                signal.SIGALRM,
                lambda _s, _f: (_ for _ in ()).throw(TimeoutError("达到官方6小时上限")),
            )
            signal.alarm(wall)
        summary = run_session(
            llm=llm, vocab=vocab, cards=cards, executor=executor,
            initial_report=first.health_report,
            initial_train_seconds=first.seconds,
            module_interface=interface, example_module=example,
            current_config=pipeline,
            rounds=int(cfg.get("max_rounds", 50)),
            token_budget=int(cfg.get("token_budget", 2_000_000)),
            epsilon=0.002, patience=3,
            baseline={"点击AUC": 0.6674, "购买AUC": 0.5357},
            logs_dir=logs,
        )
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)
            if old_handler is not None:
                signal.signal(signal.SIGALRM, old_handler)

    executor.select_round(int(summary.best_round))
    final = None
    if bool(cfg.get("generate_test_after_convergence", True)):
        final = executor.make_final_submission()
        if not final.ok:
            raise RuntimeError(f"最终提交生成失败：{final.error}")

    summary_data = asdict(summary) if is_dataclass(summary) else dict(summary)
    result = {
        "status": "complete",
        "best_round": summary.best_round,
        "best_scores": summary.best_scores,
        "stopped_because": summary.stopped_because,
        "rounds_run": summary.rounds_run,
        "total_tokens": summary.total_tokens,
        "agent_wall_seconds": time.monotonic() - session_started,
        "manual_interventions": summary.interventions,
        "recoveries": summary.recoveries,
        "official_baseline_validation": {
            "GAUC": 0.6674, "nDCG@5": 0.5357, "primary": 0.6016,
        },
        "submission": ((final.health_report or {}).get("最终提交") if final else None),
        "goat_session_summary": summary_data,
    }
    (output / "final_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result

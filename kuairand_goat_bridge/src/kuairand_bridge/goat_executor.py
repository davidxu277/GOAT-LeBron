"""GOAT Executor-compatible boundary backed by the official KuaiRand bridge.

This module deliberately lives inside the bridge.  It does not modify GOAT's
agent/, harness/, config/ or modules/ trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import pathlib
import signal
import time
from typing import Any, Callable

from .runner import run_trainer


@dataclass
class BridgeRunResult:
    """Duck-compatible with agent.loop.RunResult."""

    ok: bool
    health_report: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    seconds: float = 0.0
    fidelity: str = "全量"
    unsupported: bool = False


class _RunTimeout(TimeoutError):
    pass


def _alarm_handler(_signum, _frame):
    raise _RunTimeout("达到本场6小时运行上限")


class KuaiRandGoatExecutor:
    """Adapter implementing GOAT's ``run(patch, fidelity)`` protocol.

    Canonical metrics remain GAUC/nDCG@5/primary.  ``点击分`` and ``购买分``
    are emitted only as compatibility aliases because the current GOAT loop's
    score reader still expects two numeric slots.  They do *not* mean CTR/CVR.
    """

    OFFICIAL_EPSILON = 0.002
    OFFICIAL_PATIENCE = 3
    OFFICIAL_MAX_ROUNDS = 50
    OFFICIAL_MAX_SECONDS = 6 * 60 * 60

    def __init__(self, data_dir: str, trainer_path: str,
                 output_dir: str = "kuairand_goat_bridge/output/goat_runs",
                 seed: int = 0, max_seconds: int = OFFICIAL_MAX_SECONDS,
                 runner: Callable[..., dict[str, Any]] = run_trainer):
        self.data_dir = str(pathlib.Path(data_dir).expanduser().resolve())
        self.trainer_path = str(pathlib.Path(trainer_path).expanduser().resolve())
        self.output_dir = pathlib.Path(output_dir).expanduser().resolve()
        self.seed = int(seed)
        self.max_seconds = int(max_seconds)
        self._runner = runner
        self._started = time.monotonic()
        self._run_no = 0
        self._patch_history: list[dict[str, Any]] = []
        self._selected_round: int | None = None

    def _run_dir(self) -> pathlib.Path:
        self._run_no += 1
        path = self.output_dir / f"round_{self._run_no:03d}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def _health_report(metrics: dict[str, Any], fidelity: str,
                       result_dir: pathlib.Path) -> dict[str, Any]:
        gauc = float(metrics["GAUC"])
        ndcg = float(metrics["nDCG@5"])
        primary = float(metrics["primary"])
        return {
            "数据集": "KuaiRand-Pure",
            "任务": {
                "标签": "long_view",
                "形式": "用户内排序",
                "正式指标": ["GAUC", "nDCG@5"],
            },
            "保真度": fidelity,
            "验证集": {
                "GAUC": gauc,
                "nDCG@5": ndcg,
                "主分": primary,
                "总行数": int(metrics.get("rows", 0)),
                "用户数": int(metrics.get("users", 0)),
                # Only for the unchanged GOAT score reader. Never present these
                # aliases as CTR/CVR in reports or official submissions.
                "点击分": gauc,
                "购买分": ndcg,
                "兼容字段说明": "点击分=GAUC、购买分=nDCG@5，仅供旧GOAT读取两个槽位",
            },
            "官方结果目录": str(result_dir),
        }

    def run(self, patch: dict[str, Any], fidelity: str) -> BridgeRunResult:
        started = time.monotonic()
        run_dir = self._run_dir()
        patch = patch or {"new_files": [], "config_patch": ""}
        self._patch_history.append(patch)
        effective_patch = {
            "new_files": patch.get("new_files", []),
            "config_patch": patch.get("config_patch", ""),
            "history": list(self._patch_history),
        }
        (run_dir / "agent_patch.json").write_text(
            json.dumps(effective_patch, ensure_ascii=False, indent=2), encoding="utf-8")

        remaining = self.max_seconds - (started - self._started)
        if remaining <= 0:
            return BridgeRunResult(False, error="达到本场6小时运行上限",
                                   fidelity=fidelity, seconds=0.0)
        old_handler = None
        try:
            if hasattr(signal, "SIGALRM"):
                old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
                signal.alarm(max(1, int(remaining)))
            result = self._runner(
                self.data_dir, self.trainer_path, run_dir, self.seed,
                False, agent_patch=effective_patch)
            metrics = result["validation"]["metrics"]
            report = self._health_report(metrics, fidelity, run_dir)
            return BridgeRunResult(True, health_report=report,
                                   seconds=time.monotonic() - started,
                                   fidelity=fidelity)
        except Exception as exc:  # error becomes a recovery event in GOAT
            unsupported = isinstance(exc, NotImplementedError)
            error = f"{type(exc).__name__}: {exc}"
            (run_dir / "error.json").write_text(json.dumps({
                "error": error, "fidelity": fidelity,
                "seconds": time.monotonic() - started,
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            return BridgeRunResult(False, error=error,
                                   seconds=time.monotonic() - started,
                                   fidelity=fidelity, unsupported=unsupported)
        finally:
            if hasattr(signal, "SIGALRM"):
                signal.alarm(0)
                if old_handler is not None:
                    signal.signal(signal.SIGALRM, old_handler)

    def select_round(self, round_id: int) -> None:
        """Select the validation-best cumulative patch state for final output."""
        if round_id < 0 or round_id >= len(self._patch_history):
            raise ValueError(f"不存在第 {round_id} 轮；已经运行 {len(self._patch_history)} 轮")
        self._selected_round = round_id

    def make_final_submission(self) -> BridgeRunResult:
        """Train once and create the checked test submission without scoring it."""
        started = time.monotonic()
        run_dir = self.output_dir / "final"
        try:
            upto = (self._selected_round + 1
                    if self._selected_round is not None else len(self._patch_history))
            history = self._patch_history[:upto]
            final_patch = {
                "new_files": [], "config_patch": "", "history": history,
            }
            result = self._runner(
                self.data_dir, self.trainer_path, run_dir, self.seed,
                True, agent_patch=final_patch)
            valid = result["validation"]["metrics"]
            report = self._health_report(valid, "全量", run_dir)
            report["最终提交"] = result["test"]["submission"]
            report["Test状态"] = "只做格式与对齐检查；不向Agent返回Test分数"
            return BridgeRunResult(True, report, seconds=time.monotonic() - started,
                                   fidelity="全量")
        except Exception as exc:
            return BridgeRunResult(False, error=f"{type(exc).__name__}: {exc}",
                                   seconds=time.monotonic() - started,
                                   fidelity="全量")


def assert_goat_compatible(executor: Any) -> None:
    """Fail fast before an expensive run if the object misses GOAT's contract."""
    if not callable(getattr(executor, "run", None)):
        raise TypeError("执行器必须实现 run(patch, fidelity)")

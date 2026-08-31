"""KuaiRand GOAT Executor。

统一管理：

- Trainer配置；
- 训练次数；
- 六小时deadline；
- Windows跨平台子进程超时；
- Agent patch历史；
- validation成绩；
- 最终test submission。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import pathlib
import time
from typing import Any, Callable

from .process_runner import (
    ChildRunnerError,
    run_with_timeout,
)
from .runner import run_trainer


@dataclass
class BridgeRunResult:
    ok: bool
    health_report: dict[str, Any] = field(
        default_factory=dict
    )
    error: str = ""
    seconds: float = 0.0
    fidelity: str = "全量"
    unsupported: bool = False


class KuaiRandGoatExecutor:
    OFFICIAL_EPSILON = 0.002
    OFFICIAL_PATIENCE = 3
    OFFICIAL_MAX_ROUNDS = 50
    OFFICIAL_MAX_SECONDS = 21600
    DEFAULT_VALIDATION_BASELINE = {
        "GAUC": 0.6674,
        "nDCG@5": 0.5357,
        "primary": 0.60155,
    }

    def __init__(
        self,
        data_dir: str,
        trainer_path: str,
        output_dir: str = (
            "kuairand_goat_bridge/output/goat_runs"
        ),
        seed: int = 0,
        max_seconds: int = OFFICIAL_MAX_SECONDS,
        max_iterations: int = OFFICIAL_MAX_ROUNDS,
        trainer_config: dict[str, Any] | None = None,
        official_baseline: dict[str, Any] | None = None,
        runner: Callable[..., dict[str, Any]] = run_trainer,
    ) -> None:
        self.data_dir = str(
            pathlib.Path(data_dir)
            .expanduser()
            .resolve()
        )
        self.trainer_path = str(
            pathlib.Path(trainer_path)
            .expanduser()
            .resolve()
        )
        self.output_dir = (
            pathlib.Path(output_dir)
            .expanduser()
            .resolve()
        )

        self.seed = int(seed)
        self.max_seconds = int(max_seconds)
        self.max_iterations = int(
            max_iterations
        )
        self.trainer_config = dict(
            trainer_config or {}
        )
        self.official_baseline = {
            key: float(value)
            for key, value in (
                official_baseline or self.DEFAULT_VALIDATION_BASELINE
            ).items()
            if key in {"GAUC", "nDCG@5", "primary"}
        }
        missing_baseline = {
            "GAUC", "nDCG@5", "primary"
        } - set(self.official_baseline)
        if missing_baseline:
            raise ValueError(f"官方 Validation 基线缺少：{sorted(missing_baseline)}")

        if not callable(runner):
            raise TypeError(
                "runner 必须可调用"
            )

        if not (
            1
            <= self.max_seconds
            <= self.OFFICIAL_MAX_SECONDS
        ):
            raise ValueError(
                "max_seconds 必须在1到21600之间"
            )

        if not (
            1
            <= self.max_iterations
            <= self.OFFICIAL_MAX_ROUNDS
        ):
            raise ValueError(
                "max_iterations 必须在1到50之间"
            )

        self._runner = runner
        self._started = time.monotonic()
        self._deadline = (
            self._started + self.max_seconds
        )
        self._training_attempts = 0
        self._run_no = 0
        self._patch_history: list[
            dict[str, Any]
        ] = []
        self._selected_round: int | None = None

    @property
    def training_attempts(self) -> int:
        return self._training_attempts

    @property
    def remaining_iterations(self) -> int:
        return max(
            0,
            self.max_iterations
            - self._training_attempts,
        )

    @property
    def elapsed_seconds(self) -> float:
        return max(
            0.0,
            time.monotonic() - self._started,
        )

    @property
    def remaining_seconds(self) -> float:
        return max(
            0.0,
            self._deadline - time.monotonic(),
        )

    def _reserve_training_attempt(
        self,
    ) -> int:
        if (
            self._training_attempts
            >= self.max_iterations
        ):
            raise RuntimeError(
                "达到官方训练尝试上限"
            )

        if self.remaining_seconds <= 0:
            raise TimeoutError(
                "达到本场6小时运行上限"
            )

        self._training_attempts += 1
        return self._training_attempts

    def _run_dir(self) -> pathlib.Path:
        self._run_no += 1
        path = (
            self.output_dir
            / f"round_{self._run_no:03d}"
        )
        path.mkdir(
            parents=True,
            exist_ok=True,
        )
        return path

    def _call_runner(
        self,
        run_dir: pathlib.Path,
        *,
        make_test: bool,
        agent_patch: dict[str, Any],
        fidelity: str,
    ) -> dict[str, Any]:
        return run_with_timeout(
            self._runner,
            args=(
                self.data_dir,
                self.trainer_path,
                run_dir,
                self.seed,
                bool(make_test),
            ),
            kwargs={
                "agent_patch": agent_patch,
                "trainer_config": dict(
                    self.trainer_config
                ),
                "fidelity": str(fidelity),
            },
            timeout_seconds=(
                self.remaining_seconds
            ),
        )

    @staticmethod
    def _normalize_patch(
        patch: dict[str, Any] | None,
    ) -> dict[str, Any]:
        patch = patch or {}

        return {
            "new_files": list(
                patch.get("new_files") or []
            ),
            "config_patch": (
                patch.get("config_patch") or ""
            ),
        }

    @staticmethod
    def _copy_history(
        history: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "new_files": list(
                    item.get("new_files") or []
                ),
                "config_patch": (
                    item.get("config_patch") or ""
                ),
            }
            for item in history
        ]

    @staticmethod
    def _health_report(
        metrics: dict[str, Any],
        fidelity: str,
        result_dir: pathlib.Path,
        *,
        seed: int,
        training_attempt: int,
        remaining_iterations: int,
        elapsed_seconds: float,
        remaining_seconds: float,
        executor_round: int,
        official_baseline: dict[str, Any] | None = None,
        training: dict[str, Any] | None = None,
        group_evidence: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        gauc = float(metrics["GAUC"])
        ndcg = float(metrics["nDCG@5"])
        primary = float(metrics["primary"])

        expected = (
            gauc + ndcg
        ) / 2.0

        # 官方评估链路里部分值会经过 NumPy float32；Windows/Linux/macOS
        # 的序列化与加法顺序可能留下 1e-8 量级误差。这里只拒绝真正的口径
        # 不一致，不应把正常的浮点舍入误判成第 0 轮失败。
        if not math.isclose(primary, expected, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "primary 与GAUC/nDCG均值不一致："
                f"primary={primary:.12f}, expected={expected:.12f}"
            )

        baseline = {
            key: float(value)
            for key, value in (
                official_baseline
                or KuaiRandGoatExecutor.DEFAULT_VALIDATION_BASELINE
            ).items()
        }
        deltas = {
            "GAUC": gauc - baseline["GAUC"],
            "nDCG@5": ndcg - baseline["nDCG@5"],
            "Primary": primary - baseline["primary"],
        }
        primary_delta = deltas["Primary"]
        comparison = (
            "显著高于官方基线" if primary_delta > KuaiRandGoatExecutor.OFFICIAL_EPSILON
            else "显著低于官方基线" if primary_delta < -KuaiRandGoatExecutor.OFFICIAL_EPSILON
            else "与官方基线差异未超过 epsilon"
        )

        # 医生判 12 个病里有 6 个靠分组之后的数字（训练集自评、曝光分桶、
        # 新老用户、用户构成、日期分段、预测健康）。摊平到顶层，跟「验证集」
        # 平级 —— 藏在一层嵌套里，医生读成绩单时容易整块略过。
        # 摊平是为了让医生一眼看到，不是给下游一个改写分数的口子 ——
        # 证据块里如果冒出一个「验证集」，它会把真分数盖掉，而且不报错。
        保留字段 = {
            "数据集", "任务", "保真度", "随机种子", "验证集",
            "官方Validation基线", "相对官方基线", "训练诊断",
            "运行预算", "官方结果目录", "最终提交",
        }
        evidence = {
            key: value
            for key, value in (group_evidence or {}).items()
            if key not in 保留字段
        }

        return {
            "数据集": "KuaiRand-Pure",
            "任务": {
                "标签": "long_view",
                "形式": "用户内排序",
                "正式指标": [
                    "GAUC",
                    "nDCG@5",
                    "primary",
                ],
                "Primary定义": "(GAUC + nDCG@5) / 2",
                "不存在购买/CVR任务": True,
            },
            "保真度": fidelity,
            "随机种子": int(seed),
            "验证集": {
                "GAUC": gauc,
                "nDCG@5": ndcg,
                "主分": primary,
                "总行数": int(
                    metrics.get("rows", 0)
                ),
                "用户数": int(
                    metrics.get("users", 0)
                ),
            },
            "官方Validation基线": {
                "GAUC": baseline["GAUC"],
                "nDCG@5": baseline["nDCG@5"],
                "Primary": baseline["primary"],
                "来源": "Track 2 Starter Kit 官方 FM (k=16, lr=0.001)",
            },
            "相对官方基线": {
                **deltas,
                "epsilon": KuaiRandGoatExecutor.OFFICIAL_EPSILON,
                "判断": comparison,
            },
            "训练诊断": dict(training or {}),
            **evidence,
            "运行预算": {
                "训练尝试编号": (
                    training_attempt
                ),
                "剩余训练尝试": (
                    remaining_iterations
                ),
                "累计墙钟秒数": (
                    elapsed_seconds
                ),
                "剩余墙钟秒数": (
                    remaining_seconds
                ),
                "执行器轮次": int(
                    executor_round
                ),
            },
            "官方结果目录": str(
                result_dir.resolve()
            ),
        }

    @staticmethod
    def _unsupported(
        exc: Exception,
    ) -> bool:
        if isinstance(
            exc,
            NotImplementedError,
        ):
            return True

        return (
            isinstance(exc, ChildRunnerError)
            and exc.exception_type
            == "NotImplementedError"
        )

    def _write_error(
        self,
        run_dir: pathlib.Path,
        error: str,
        fidelity: str,
        seconds: float,
        attempt: int | None,
    ) -> None:
        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        try:
            (
                run_dir / "error.json"
            ).write_text(
                json.dumps(
                    {
                        "error": error,
                        "fidelity": fidelity,
                        "seconds": seconds,
                        "seed": self.seed,
                        "training_attempt": attempt,
                        "training_attempts_used": (
                            self.training_attempts
                        ),
                        "remaining_iterations": (
                            self.remaining_iterations
                        ),
                        "remaining_seconds": (
                            self.remaining_seconds
                        ),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError:
            pass

    def run(
        self,
        patch: dict[str, Any],
        fidelity: str,
    ) -> BridgeRunResult:
        started = time.monotonic()
        run_dir: pathlib.Path | None = None
        attempt: int | None = None

        try:
            attempt = (
                self._reserve_training_attempt()
            )
            run_dir = self._run_dir()

            normalized = self._normalize_patch(
                patch
            )
            self._patch_history.append(
                normalized
            )
            executor_round = (
                len(self._patch_history) - 1
            )

            effective_patch = {
                "new_files": list(
                    normalized["new_files"]
                ),
                "config_patch": (
                    normalized["config_patch"]
                ),
                "history": self._copy_history(
                    self._patch_history
                ),
            }

            (
                run_dir / "agent_patch.json"
            ).write_text(
                json.dumps(
                    effective_patch,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = self._call_runner(
                run_dir,
                make_test=False,
                agent_patch=effective_patch,
                fidelity=fidelity,
            )

            metrics = result[
                "validation"
            ]["metrics"]

            report = self._health_report(
                metrics,
                fidelity,
                run_dir,
                seed=self.seed,
                training_attempt=attempt,
                remaining_iterations=(
                    self.remaining_iterations
                ),
                elapsed_seconds=(
                    self.elapsed_seconds
                ),
                remaining_seconds=(
                    self.remaining_seconds
                ),
                executor_round=executor_round,
                official_baseline=self.official_baseline,
                training=result.get("training") or {},
                group_evidence=result.get("diagnostics") or {},
            )

            return BridgeRunResult(
                ok=True,
                health_report=report,
                seconds=(
                    time.monotonic() - started
                ),
                fidelity=fidelity,
            )

        except Exception as exc:
            if run_dir is None:
                run_dir = (
                    self.output_dir
                    / "budget_or_timeout_failure"
                )

            error = (
                f"{type(exc).__name__}: {exc}"
            )

            self._write_error(
                run_dir,
                error,
                fidelity,
                time.monotonic() - started,
                attempt,
            )

            return BridgeRunResult(
                ok=False,
                error=error,
                seconds=(
                    time.monotonic() - started
                ),
                fidelity=fidelity,
                unsupported=self._unsupported(
                    exc
                ),
            )

    def select_round(
        self,
        round_id: int,
    ) -> None:
        round_id = int(round_id)

        if not (
            0
            <= round_id
            < len(self._patch_history)
        ):
            raise ValueError(
                f"不存在第 {round_id} 轮"
            )

        self._selected_round = round_id

    def make_final_submission(
        self,
    ) -> BridgeRunResult:
        started = time.monotonic()
        run_dir = self.output_dir / "final"
        attempt: int | None = None

        try:
            attempt = (
                self._reserve_training_attempt()
            )

            if not self._patch_history:
                raise RuntimeError(
                    "尚未运行第0轮baseline"
                )

            selected = (
                self._selected_round
                if self._selected_round
                is not None
                else len(
                    self._patch_history
                ) - 1
            )

            final_patch = {
                "new_files": [],
                "config_patch": "",
                "history": self._copy_history(
                    self._patch_history[
                        : selected + 1
                    ]
                ),
            }

            result = self._call_runner(
                run_dir,
                make_test=True,
                agent_patch=final_patch,
                fidelity="全量",
            )

            validation = result[
                "validation"
            ]
            test_result = result["test"]

            if "metrics" in test_result:
                raise RuntimeError(
                    "test结果不得包含metrics"
                )

            if (
                test_result.get("status")
                != "checked"
            ):
                raise RuntimeError(
                    "test提交未通过检查"
                )

            report = self._health_report(
                validation["metrics"],
                "全量",
                run_dir,
                seed=self.seed,
                training_attempt=attempt,
                remaining_iterations=(
                    self.remaining_iterations
                ),
                elapsed_seconds=(
                    self.elapsed_seconds
                ),
                remaining_seconds=(
                    self.remaining_seconds
                ),
                official_baseline=self.official_baseline,
                training=result.get("training") or {},
                group_evidence=result.get("diagnostics") or {},
            )

            report["最终提交"] = (
                test_result["submission"]
            )
            report["Test状态"] = (
                "只检查格式与对齐；"
                "不返回隐藏测试集分数"
            )

            return BridgeRunResult(
                ok=True,
                health_report=report,
                seconds=(
                    time.monotonic() - started
                ),
                fidelity="全量",
            )

        except Exception as exc:
            error = (
                f"{type(exc).__name__}: {exc}"
            )

            self._write_error(
                run_dir,
                error,
                "全量",
                time.monotonic() - started,
                attempt,
            )

            return BridgeRunResult(
                ok=False,
                error=error,
                seconds=(
                    time.monotonic() - started
                ),
                fidelity="全量",
                unsupported=self._unsupported(
                    exc
                ),
            )


def assert_goat_compatible(
    executor: Any,
) -> None:
    for method in (
        "run",
        "select_round",
        "make_final_submission",
    ):
        if not callable(
            getattr(executor, method, None)
        ):
            raise TypeError(
                f"Executor缺少方法：{method}"
            )

"""跨平台 Runner 子进程与硬超时控制。

Windows 不支持 SIGALRM，因此 Trainer 必须放在独立子进程中运行。
主进程等待指定时间，超时后终止并清理子进程。

本模块使用 multiprocessing 的 spawn 模式，确保 Windows、Linux 和
macOS 行为尽可能一致。
"""

from __future__ import annotations

import multiprocessing as mp
import queue
import traceback
from typing import Any, Callable


class ChildRunnerError(RuntimeError):
    """Trainer 子进程执行失败。"""

    def __init__(
        self,
        exception_type: str,
        message: str,
        child_traceback: str = "",
    ) -> None:
        self.exception_type = str(
            exception_type
        )
        self.child_message = str(message)
        self.child_traceback = str(
            child_traceback
        )

        detail = (
            f"训练子进程失败："
            f"{self.exception_type}: "
            f"{self.child_message}"
        )

        if self.child_traceback:
            detail += (
                "\n子进程 traceback：\n"
                f"{self.child_traceback}"
            )

        super().__init__(detail)


def _worker(
    result_queue,
    runner: Callable[..., dict[str, Any]],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    """在训练子进程中调用 Runner。"""
    try:
        result = runner(
            *args,
            **kwargs,
        )

        result_queue.put({
            "status": "ok",
            "result": result,
        })

    except BaseException as exc:
        result_queue.put({
            "status": "error",
            "exception_type": (
                type(exc).__name__
            ),
            "message": str(exc),
            "traceback": traceback.format_exc(),
        })


def _stop_process(
    process: mp.Process,
    *,
    grace_seconds: float = 5.0,
) -> None:
    """终止子进程，必要时升级为 kill。"""
    if not process.is_alive():
        process.join()
        return

    process.terminate()
    process.join(
        timeout=max(
            0.1,
            float(grace_seconds),
        )
    )

    if process.is_alive():
        process.kill()
        process.join(
            timeout=max(
                0.1,
                float(grace_seconds),
            )
        )

    if process.is_alive():
        raise RuntimeError(
            "训练子进程在 terminate/kill 后仍未退出"
        )


def run_with_timeout(
    runner: Callable[..., dict[str, Any]],
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    timeout_seconds: float,
) -> dict[str, Any]:
    """在独立子进程运行 Runner，并执行硬超时。

    参数
    ----
    runner:
        要执行的模块顶层函数。

        Windows spawn 模式下不能使用 lambda、局部函数或闭包。

    args:
        Runner 的位置参数。

    kwargs:
        Runner 的关键字参数。

    timeout_seconds:
        本次训练可使用的剩余墙钟秒数。

    返回
    ----
    dict:
        Runner 返回的字典。

    异常
    ----
    TimeoutError:
        训练超过剩余墙钟时间，子进程已被终止。

    ChildRunnerError:
        Trainer 或 Runner 在子进程中抛出异常。
    """
    timeout = float(timeout_seconds)

    if timeout <= 0:
        raise TimeoutError(
            "达到本场6小时运行上限"
        )

    if not callable(runner):
        raise TypeError(
            "runner 必须是可调用对象"
        )

    context = mp.get_context("spawn")
    result_queue = context.Queue(
        maxsize=1
    )

    process = context.Process(
        target=_worker,
        args=(
            result_queue,
            runner,
            tuple(args),
            dict(kwargs or {}),
        ),
        daemon=False,
        name="kuairand-trainer",
    )

    try:
        process.start()
    except Exception:
        result_queue.close()
        result_queue.join_thread()
        raise

    process.join(timeout=timeout)

    if process.is_alive():
        try:
            _stop_process(process)
        finally:
            result_queue.close()
            result_queue.join_thread()

        raise TimeoutError(
            "训练超过剩余墙钟时间 "
            f"{timeout:.1f} 秒，"
            "已终止训练子进程"
        )

    try:
        payload = result_queue.get(
            timeout=2.0
        )
    except queue.Empty as exc:
        raise RuntimeError(
            "训练子进程已经退出，但没有返回结果；"
            f"exitcode={process.exitcode}"
        ) from exc
    finally:
        result_queue.close()
        result_queue.join_thread()

    if not isinstance(payload, dict):
        raise TypeError(
            "训练子进程返回协议错误："
            f"实际类型={type(payload).__name__}"
        )

    status = payload.get("status")

    if status == "error":
        raise ChildRunnerError(
            exception_type=payload.get(
                "exception_type",
                "Exception",
            ),
            message=payload.get(
                "message",
                "",
            ),
            child_traceback=payload.get(
                "traceback",
                "",
            ),
        )

    if status != "ok":
        raise RuntimeError(
            "训练子进程返回未知状态："
            f"{status!r}"
        )

    result = payload.get("result")

    if not isinstance(result, dict):
        raise TypeError(
            "Runner 返回值必须是 dict，"
            f"实际是 {type(result).__name__}"
        )

    return result
"""带全程留痕的跑法 —— 把四个角色的每一次对话原样存下来。

现有日志已经存了不少，但差两块：

  · `logs/live_events.jsonl` 有模型的**回答**（连思维链一起，kind=reasoning），
    但没有**提示词**，而且是散成一条条 delta 的，事后拼起来才能读
  · `logs/rounds.jsonl` 有各角色**解析后**的结构化产物（诊断/提案/补丁全文/
    成绩单/复盘），但没有原始往返

交付物 #3 要的是「每轮的假设、代码 diff、指标、错误与恢复」，评委还要靠它
判自主性。原始往返丢了就补不回来 —— 一场六小时、五十轮，重跑一次的代价
是整场。所以这里在 LLM 外面套一层，把每一次调用**完整**落盘。

这一层是旁观者：只读不改，任何异常都不许影响主循环（跑分优先于留痕）。
不修改 agent/ 下任何文件（R5）。

用法：
    python3 tools/run_with_transcript.py configs/kuairand_task.yaml
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kuairand_goat_bridge" / "src"))


class TranscriptLLM:
    """把任何 LLM 对象包一层，每次 call 原样落盘。

    对外的属性访问一律转发给内层，所以主循环拿它当原来那个用，
    ledger（token 账本）也照常工作。
    """

    def __init__(self, inner, path: pathlib.Path):
        self._inner = inner
        self._path = path
        self._n = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def call(self, *, role: str, system: str, user: str, schema, **kw):
        self._n += 1
        started = time.time()
        try:
            out = self._inner.call(
                role=role, system=system, user=user, schema=schema, **kw)
            self._write(role, system, user, out, None, time.time() - started)
            return out
        except Exception as exc:                                  # noqa: BLE001
            self._write(role, system, user, None,
                        f"{type(exc).__name__}: {exc}", time.time() - started)
            raise

    def _write(self, role, system, user, answer, error, seconds) -> None:
        """留痕失败绝不能弄崩主循环 —— 跑分优先于留痕。"""
        try:
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "序号": self._n,
                    "时刻": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "角色": role,
                    "耗时秒": round(seconds, 2),
                    "系统提示词": system,
                    "用户提示词": user,
                    "回答": answer,
                    "错误": error,
                }, ensure_ascii=False) + "\n")
                fh.flush()
        except Exception:                                         # noqa: BLE001
            pass


def main() -> int:
    config = sys.argv[1] if len(sys.argv) > 1 else "configs/kuairand_task.yaml"
    if not config.startswith("configs/"):
        config = f"configs/{config}"

    stamp = time.strftime("%Y%m%d-%H%M%S")
    transcript = ROOT / "logs" / f"transcript-{stamp}.jsonl"

    from agent import cli as agent_cli
    原始 = agent_cli.make_llm

    def 带留痕的(*a, **kw):
        return TranscriptLLM(原始(*a, **kw), transcript)

    agent_cli.make_llm = 带留痕的                      # goat_run.run() 从这里取

    print(f"全程留痕 → {transcript.relative_to(ROOT)}")
    print(f"事件流   → logs/live_events.jsonl（模型原文，含思维链）")
    print(f"每轮记录 → 见配置里 output_dir 下的 logs/rounds.jsonl")
    print()

    from kuairand_bridge.goat_run import run
    try:
        result = run(str(ROOT / "kuairand_goat_bridge" / config))
        print("\n跑完了：", json.dumps(
            {k: v for k, v in result.items() if k != "config"},
            ensure_ascii=False, indent=1)[:2000])
        return 0
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C —— 已跑的轮次都留在日志里，没丢。")
        return 130
    except Exception:                                             # noqa: BLE001
        traceback.print_exc()
        print("\n跑挂了，但留痕文件是完整的：", transcript)
        return 1
    finally:
        agent_cli.make_llm = 原始
        if transcript.exists():
            n = sum(1 for _ in transcript.open(encoding="utf-8"))
            print(f"\n留痕共 {n} 次角色调用 → {transcript}")


if __name__ == "__main__":
    raise SystemExit(main())

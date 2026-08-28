"""本地控制台 —— 填数据路径、预检、启动真实训练、看实时进度。

    python web/server.py          # 打开 http://127.0.0.1:8000

只绑 127.0.0.1，不对外。用标准库 http.server，零依赖。

四个接口：
    POST /api/pick        弹出系统文件选择器（Finder），返回选中文件的真实路径
    POST /api/preflight   读数据、报规模与质量、亮红绿灯（不训练）
    POST /api/run         真的跑一轮：训练 → 预测 → 评分 → 成绩单
    GET  /api/events      读 logs/live_events.jsonl 的增量（轮询）
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PORT = 8000
_state: dict = {"running": False, "last": None}


def _json(handler: BaseHTTPRequestHandler, code: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode()
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # 静音访问日志
        pass

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            html = (ROOT / "web" / "console.html").read_text(encoding="utf-8")
            body = html.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/api/events"):
            offset = 0
            if "?" in self.path:
                q = self.path.split("?", 1)[1]
                for kv in q.split("&"):
                    if kv.startswith("offset="):
                        offset = int(kv.split("=", 1)[1] or 0)
            path = ROOT / "logs" / "live_events.jsonl"
            lines = []
            if path.exists():
                all_lines = path.read_text(encoding="utf-8").splitlines()
                lines = [json.loads(x) for x in all_lines[offset:] if x.strip()]
                offset = len(all_lines)
            _json(self, 200, {"events": lines, "offset": offset,
                              "running": _state["running"]})
        elif self.path.startswith("/api/detail"):
            # 取某次 LLM 调用的完整内容：把 llm_delta 拼回去，
            # 推理过程和正式输出分开返回（前端折叠显示）
            q = dict(kv.split("=", 1) for kv in self.path.split("?", 1)[1].split("&")) \
                if "?" in self.path else {}
            start, end = int(q.get("start", 0)), int(q.get("end", 0))
            path = ROOT / "logs" / "live_events.jsonl"
            reasoning, answer = [], []
            if path.exists():
                lines = path.read_text(encoding="utf-8").splitlines()[start:end]
                for line in lines:
                    if not line.strip():
                        continue
                    e = json.loads(line)
                    if e.get("type") != "llm_delta":
                        continue
                    (reasoning if e.get("kind") == "reasoning" else answer).append(
                        e.get("text", ""))
            _json(self, 200, {"reasoning": "".join(reasoning), "answer": "".join(answer)})

        elif self.path.startswith("/api/status"):
            _json(self, 200, {"running": _state["running"], "last": _state["last"]})
        else:
            _json(self, 404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or "{}")

        if self.path == "/api/pick":
            _json(self, 200, {"path": _pick_path(payload.get("title", "选择数据"),
                                                 payload.get("kind", "file"))})

        elif self.path == "/api/preflight":
            try:
                from harness.data import preflight
                report = preflight(payload["train"], payload["val_features"],
                                   payload.get("val_labels") or None)
                _json(self, 200, report.to_dict())
            except Exception as exc:
                _json(self, 200, {"level": "bad", "rows": {}, "stats": {},
                                  "checks": [{"level": "bad",
                                              "text": f"{type(exc).__name__}: {exc}"}]})

        elif self.path == "/api/run":
            if _state["running"]:
                _json(self, 409, {"error": "已经有任务在跑"})
                return
            _state["running"] = True
            _state["last"] = None
            threading.Thread(target=_run_job, args=(payload,), daemon=True).start()
            _json(self, 202, {"started": True})
        else:
            _json(self, 404, {"error": "not found"})


def _pick_path(title: str, kind: str = "file") -> str:
    """弹出 macOS 原生选择器，返回真实路径。kind="folder" 时选目录。

    浏览器出于安全拿不到本地文件的真实路径，但服务器就跑在本机上，
    直接调系统的选择器即可。用户点取消返回空串。

    选目录是为了支持分片数据集 —— 一个 split 常常是几百个
    part-xxxx.parquet，选目录比逐个选文件合理。
    """
    if kind == "folder":
        script = f'set f to choose folder with prompt "{title}"\nPOSIX path of f'
    else:
        script = (f'set f to choose file with prompt "{title}" '
                  f'of type {{"csv", "parquet", "txt"}}\nPOSIX path of f')
    try:
        out = subprocess.run(["osascript", "-e", script],
                             capture_output=True, text=True, timeout=300)
        return out.stdout.strip().rstrip("/")   # 取消时 stdout 为空
    except Exception:
        return ""


def _run_job(payload: dict) -> None:
    """后台线程里跑任务，结果放进 _state。

    两种模式：
      train  —— 只跑一次训练（不经过四个角色），用来验证数据管线通不通
      agent  —— 完整的自主迭代 N 轮，医生/军师/工兵/复盘官全程参与
    """
    mode = payload.get("mode", "train")
    try:
        from agent.events import emit
        if mode == "agent":
            _run_agent(payload, emit)
        else:
            _run_train_only(payload, emit)
    except Exception as exc:
        traceback.print_exc()
        _state["last"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                          "traceback": traceback.format_exc()[-2000:]}
        try:
            from agent.events import emit as _e
            _e("recovery", text=f"整场中断：{type(exc).__name__}: {exc}")
        except Exception:
            pass
    finally:
        _state["running"] = False


def _run_train_only(payload: dict, emit) -> None:
    from harness.executor import RealExecutor
    emit("phase", name="启动", detail=f"只跑训练 · 保真度 {payload.get('fidelity', '全量')}")
    ex = RealExecutor(payload["train"], payload["val_features"],
                      payload.get("val_labels") or None,
                      seed=int(payload.get("seed", 20260827)))
    result = ex.run({"new_files": [], "config_patch": {}},
                    payload.get("fidelity", "全量"))
    _state["last"] = {"mode": "train", "ok": result.ok, "seconds": result.seconds,
                      "error": result.error, "report": result.health_report}
    emit("round_end", seconds=result.seconds,
         verdict="完成" if result.ok else "失败")


def _run_agent(payload: dict, emit) -> None:
    """完整自主迭代 —— 四个角色的调用会自己发事件，控制台照单全收。"""
    import time
    import json as _json
    from agent.cli import (INTERFACE_SPEC, PIPELINE_CONFIG, _noise_bands,
                           _noise_floor, example_for, make_llm)
    from agent.knowledge import CardLibrary, SymptomVocab
    from agent.loop import run_session
    from harness.executor import RealExecutor

    rounds = int(payload.get("rounds", 5))
    fidelity = payload.get("fidelity", "小份")
    emit("phase", name="启动自主迭代", detail=f"最多 {rounds} 轮 · 起步 {fidelity}数据")

    vocab = SymptomVocab.load()
    executor = RealExecutor(payload["train"], payload["val_features"],
                            payload.get("val_labels") or None,
                            seed=int(payload.get("seed", 20260827)),
                            holdout_path=payload.get("holdout") or None)
    # 第一份成绩单：先跑一次基线，医生要看着它做第一次诊断
    emit("phase", name="跑基线", detail="给医生第一份成绩单")
    base = executor.run({"new_files": [], "config_patch": ""}, fidelity)
    if not base.ok:
        raise RuntimeError(f"基线就没跑起来：{base.error}")

    t0 = time.time()
    bands = _noise_bands()
    summary = run_session(
        llm=make_llm(), vocab=vocab, cards=CardLibrary.load(vocab),
        executor=executor, initial_report=base.health_report,
        # 体检那一跑也烧了算力，要计进 GPU 小时（交付物 #5）
        initial_train_seconds=base.seconds,
        module_interface=INTERFACE_SPEC, example_module=example_for,
        current_config=PIPELINE_CONFIG, rounds=rounds,
        start_fidelity=fidelity,        # 界面选了哪档就从哪档起步
        noise_floor=_noise_floor(bands), noise_bands=bands,
        logs_dir=ROOT / "logs",
    )
    # 展示最终提交的那一版，不是第 0 轮的基线 —— 跑完十轮还给人看起步分数是误导
    best_path = ROOT / "logs" / "best_report.json"
    best_report = (_json.loads(best_path.read_text(encoding="utf-8"))
                   if best_path.exists() else base.health_report)
    _state["last"] = {
        "mode": "agent", "ok": True, "seconds": time.time() - t0,
        "rounds_run": getattr(summary, "rounds_run", None),
        "stopped_because": getattr(summary, "stopped_because", ""),
        "summary_text": summary.as_table(),
        "total_tokens": summary.total_tokens,
        "best_round": summary.best_round,
        "report": best_report,
    }


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"控制台已启动 → http://127.0.0.1:{PORT}")
    print("按 Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
        elif self.path.startswith("/api/status"):
            _json(self, 200, {"running": _state["running"], "last": _state["last"]})
        else:
            _json(self, 404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or "{}")

        if self.path == "/api/pick":
            _json(self, 200, {"path": _pick_file(payload.get("title", "选择数据文件"))})

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


def _pick_file(title: str) -> str:
    """弹出 macOS 原生文件选择器，返回选中文件的真实路径。

    浏览器出于安全拿不到本地文件的真实路径，但服务器就跑在本机上，
    直接调系统的选择器即可。用户点取消返回空串。
    """
    script = (
        f'set f to choose file with prompt "{title}" '
        f'of type {{"csv", "parquet", "txt"}}\n'
        f'POSIX path of f'
    )
    try:
        out = subprocess.run(["osascript", "-e", script],
                             capture_output=True, text=True, timeout=300)
        return out.stdout.strip()          # 取消时 osascript 报错，stdout 为空
    except Exception:
        return ""


def _run_job(payload: dict) -> None:
    """后台线程里跑训练，结果放进 _state。"""
    try:
        from agent.events import emit
        from harness.executor import RealExecutor
        emit("phase", name="启动", detail=f"保真度 {payload.get('fidelity', '全量')}")
        ex = RealExecutor(payload["train"], payload["val_features"],
                          payload["val_labels"],
                          seed=int(payload.get("seed", 20260827)))
        result = ex.run({"new_files": [], "config_patch": {}},
                        payload.get("fidelity", "全量"))
        _state["last"] = {"ok": result.ok, "seconds": result.seconds,
                          "error": result.error, "report": result.health_report}
        emit("round_end", seconds=result.seconds,
             verdict="完成" if result.ok else "失败")
    except Exception as exc:
        traceback.print_exc()
        _state["last"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        _state["running"] = False


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

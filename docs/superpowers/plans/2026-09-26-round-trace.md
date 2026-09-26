# 每轮看得见 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 跑的时候每做完一步在终端打一小段人话，每轮另存一份 markdown 全文；顺带把现在丢掉的四样过程数据记进 `RoundLog`。

**Architecture:** 新文件 `agent/trace.py` 只管排版（纯函数 + 一个会吞异常的 `Tracer`）；
`agent/loop.py` 的 `RoundLog` 加四个字段，`run_round` 每步记一笔并调 `tracer.step`；
`run_session` 默认造一个 `Tracer`，并把 markdown 写进已有的落盘循环。

**Tech Stack:** Python 3.14、pytest

Spec: `docs/superpowers/specs/2026-09-26-round-trace-design.md`

## Global Constraints

- 排版逻辑只有一份：终端和 markdown 共用 `agent/trace.py` 的摘要函数
- `Tracer.step` 与 markdown 落盘全程吞异常，绝不让记录弄崩正在跑的一轮
- `run_round(tracer=None)` 时行为与现在完全一致（只多记字段，不打印）
- 成功那一版代码不在 `attempts` 里重抄，指向 `patch_files`
- 不动 `agent/events.py` 白名单和 `web/console.html`
- 提交信息不加 Co-Authored-By；每个 Task 结束跑全量测试，0 失败

---

### Task 1: `agent/trace.py`

**Files:**
- Create: `agent/trace.py`
- Test: `tests/test_trace.py`

**Interfaces:**
- Produces:
  - `步骤摘要(步骤: str, 内容: Any) -> list[str]` —— 步骤名取
    `医生 / 筛卡 / 军师 / 调度 / 工兵 / 试跑 / 训练 / 复盘官 / 本轮`
  - `class Tracer(printer: Callable[[str], None] = print, enabled: bool = True)`，
    方法 `step(步骤: str, 内容: Any) -> None`、`round_header(round_id: int, fidelity: str) -> None`
  - `render_round(log) -> str`
  - `write_round_md(logs_dir: pathlib.Path, log) -> pathlib.Path`

- [ ] **Step 1: 写失败测试** `tests/test_trace.py`

```python
from agent import trace
from agent.loop import RoundLog

def _log() -> RoundLog:
    log = RoundLog(round_id=3, started_at="2026-09-26T10:00:00", run_id="abc")
    log.diagnosis = {"findings": [{"symptom": "时间漂移", "severity": 0.7, "confidence": "高",
                                   "evidence": "最低桶 GAUC 0.612，最高桶 0.681",
                                   "affects": ["GAUC"]}],
                     "no_finding": False, "reason_if_none": ""}
    log.candidates = [{"card_id": "时间衰减加权", "名字": "近期的行为更算数",
                       "治哪些病": ["时间漂移"], "信任分": 0.6}]
    log.proposals = {"proposals": [{"rank": 1, "card_id": "时间衰减加权", "targets": ["时间漂移"],
                                    "rationale": "最低桶比最高桶低 0.069",
                                    "expected": {"GAUC": 0.001}, "risk": "热门视频被稀释",
                                    "cost": {"代码难度": "简单", "训练时间倍数": 1.0},
                                    "how_to": "在 features 下新开一块", "novel": False}]}
    log.chosen, log.fidelity = log.proposals["proposals"][0], "中份"
    log.backups = ["行为序列"]
    log.attempts = [
        {"候选": "时间衰减加权", "第几次": 1, "新文件": ["modules/features/time_decay.py"],
         "config_patch": "features:\n  time_decay:\n    enabled: true\n",
         "试跑": "挂了", "报错": "KeyError: 'play_time'\n子进程 traceback：…",
         "失败代码": {"modules/features/time_decay.py": "class 写错了: pass"}},
        {"候选": "时间衰减加权", "第几次": 2, "新文件": ["modules/features/time_decay.py"],
         "config_patch": "features:\n  time_decay:\n    enabled: true\n",
         "试跑": "过了", "报错": "", "失败代码": {}},
    ]
    log.patch_files = {"modules/features/time_decay.py": "class 改好了: pass"}
    log.run_ok = True
    log.metrics = {"验证集": {"GAUC": 0.6692, "nDCG@5": 0.5433, "主分": 0.6062}}
    log.reflection = {"verdict": "猜对了", "actual": {"GAUC": 0.0008, "nDCG@5": 0.0031},
                      "vs_expected": "略低", "next_hint": "去看新用户那一组",
                      "symptom_resolved": [{"symptom": "时间漂移", "before": 0.069,
                                            "after": 0.051, "resolved": "部分"}]}
    log.steps = [{"角色": "医生", "秒": 12.3, "token": 4200}]
    log.seconds, log.tokens, log.train_seconds = 221.0, 38000, 218.0
    return log

def test_医生摘要带上病名和证据():
    出 = trace.步骤摘要("医生", _log().diagnosis)
    assert any("时间漂移" in x for x in 出)
    assert any("0.612" in x for x in 出)

def test_试跑挂了的摘要只留报错首行():
    出 = trace.步骤摘要("试跑", {"试跑": "挂了", "报错": "KeyError: 'x'\n第二行\n第三行"})
    assert "KeyError" in 出[0] and "第二行" not in 出[0]

def test_渲染全文八个段落都在():
    md = trace.render_round(_log())
    for 段 in ("医生", "筛卡", "军师", "调度", "工兵", "训练", "复盘官", "本轮"):
        assert f"## {段}" in md, 段

def test_失败那版代码在全文里_成功那版不重抄():
    md = trace.render_round(_log())
    assert "class 写错了" in md
    assert "class 改好了" not in md          # 成功那版在 rounds.jsonl 的 patch_files 里
    assert "patch_files" in md               # 指个路

def test_半路死掉的一轮也能渲染():
    log = RoundLog(round_id=1, started_at="x")
    log.diagnosis = _log().diagnosis
    log.recoveries = ["军师失败：SchemaViolation: 瞎写"]
    md = trace.render_round(log)             # 不许抛
    assert "没走到这一步" in md
    assert "军师失败" in md

def test_tracer_按顺序打印():
    行 = []
    t = trace.Tracer(printer=行.append)
    t.round_header(3, "中份")
    t.step("医生", _log().diagnosis)
    assert "第 3 轮" in 行[0] and "中份" in 行[0]
    assert "医生" in 行[1]

def test_摘要炸了不许把这一轮带崩(monkeypatch):
    def 炸(*a, **k):
        raise RuntimeError("排版写错了")
    monkeypatch.setattr(trace, "步骤摘要", 炸)
    行 = []
    trace.Tracer(printer=行.append).step("医生", {})   # 不许抛
    assert 行 == []

def test_关掉就什么都不打():
    行 = []
    t = trace.Tracer(printer=行.append, enabled=False)
    t.round_header(1, "小份")
    t.step("医生", _log().diagnosis)
    assert 行 == []

def test_写文件落在按场次分的目录里(tmp_path):
    p = trace.write_round_md(tmp_path, _log())
    assert p == tmp_path / "rounds" / "abc" / "round_003.md"
    assert "## 复盘官" in p.read_text(encoding="utf-8")
```

- [ ] **Step 2: 跑，确认失败**

Run: `.venv/bin/python -m pytest tests/test_trace.py -q`
Expected: FAIL（`No module named 'agent.trace'`）

- [ ] **Step 3: 实现 `agent/trace.py`**

结构（照 spec）：

```python
"""把一轮的经过写成人话 —— 终端一行行打，每轮另存一份全文。

排版只有这一份：终端和 markdown 共用下面这些摘要函数。
这里的任何异常都不许往外冒 —— 记录是旁观者，绝不反过来弄崩正在跑的训练。
"""
```

- `_数(x, 位=4)`：数字统一格式，`None` → `—`
- 每个步骤一个私有摘要函数（`_医生 / _筛卡 / _军师 / _调度 / _工兵 / _试跑 / _训练 / _复盘官 / _本轮`），
  返回 `list[str]`；`步骤摘要` 按名字分发，不认识的步骤返回 `[str(内容)[:200]]`
- `Tracer.step`：`for i, 行 in enumerate(步骤摘要(...))`，第一行 `f"  {步骤:<6}{行}"`，
  续行缩进对齐（`" " * 8 + 行`）；整个方法体包 `try/except Exception: pass`
- `render_round`：按八段拼 markdown，每段缺数据就写 `（没走到这一步）`；
  工兵那段按 `attempts` 逐条列，失败代码用 ```` ```python ```` 围栏，
  成功那版写一句「完整代码见 `rounds.jsonl` 的 `patch_files`」
- `write_round_md(logs_dir, log)`：`logs_dir / "rounds" / (log.run_id or "unknown") / f"round_{log.round_id:03d}.md"`，
  `mkdir(parents=True, exist_ok=True)` 后写入，返回路径

- [ ] **Step 4: 跑到全绿**

Run: `.venv/bin/python -m pytest tests/test_trace.py -q` → 9 passed
Run: `.venv/bin/python -m pytest -q` → 0 失败

- [ ] **Step 5: 提交并推送**

```bash
git add agent/trace.py tests/test_trace.py
git commit -m "feat(trace): 一轮的经过写成人话（终端摘要 + 每轮 markdown 全文）"
git push origin main
```

---

### Task 2: `RoundLog` 记下现在丢掉的四样东西

**Files:**
- Modify: `agent/loop.py`（`RoundLog` 加字段；`run_round` 每步记一笔）
- Test: `tests/test_round_record.py`

**Interfaces:**
- Consumes: 无
- Produces: `RoundLog.candidates / backups / attempts / steps`（四个 list，默认空）

- [ ] **Step 1: 写失败测试** `tests/test_round_record.py`

用 `tests/test_smoke_run.py` 里那套假执行器的写法（`DriftingExecutor` + `smoke`），
跑一轮真的 `run_round`：

```python
def test_候选卡和备胎都记下来():
    log = _一轮(_试跑执行器())
    assert log.candidates and "card_id" in log.candidates[0] and "信任分" in log.candidates[0]
    assert isinstance(log.backups, list)

def test_工兵每一次尝试都记下来_含失败那版代码():
    log = _一轮(_试跑执行器(fail_times=1))
    assert [a["第几次"] for a in log.attempts] == [1, 2]
    assert log.attempts[0]["试跑"] == "挂了" and "不存在的列" in log.attempts[0]["报错"]
    assert log.attempts[0]["失败代码"]            # 写错那一版的代码留着
    assert log.attempts[1]["试跑"] == "过了" and log.attempts[1]["失败代码"] == {}

def test_没有试跑的执行器记没试跑():
    log = _一轮(_只数正式训练())
    assert log.attempts[0]["试跑"] == "没试跑"

def test_每步的时间和token都记下来():
    log = _一轮(_试跑执行器())
    角色 = [s["角色"] for s in log.steps]
    assert 角色[:2] == ["医生", "军师"] and "复盘官" in 角色
    assert all(s["秒"] >= 0 and s["token"] >= 0 for s in log.steps)
```

- [ ] **Step 2: 跑，确认失败**

Run: `.venv/bin/python -m pytest tests/test_round_record.py -q`
Expected: FAIL（`AttributeError: 'RoundLog' object has no attribute 'candidates'`）

- [ ] **Step 3: 实现**

`RoundLog` 加四个字段（都 `field(default_factory=list)`），紧跟现有字段后面，
各带一行注释说明为什么要记。

`run_round` 里：

```python
    def _记一步(角色: str, t: float, tok: int) -> None:
        log.steps.append({"角色": 角色, "秒": round(time.time() - t, 2),
                          "token": llm.ledger.total_tokens - tok})
```

- 医生 / 军师 / 复盘官：调用前存 `t = time.time()`、`tok = llm.ledger.total_tokens`，调用后 `_记一步`
- 筛卡后：`log.candidates = [{"card_id": c.id, "名字": c.name, "治哪些病": c.treats, "信任分": round(c.prior, 2)} for c in candidates]`
- 调度后：`log.backups = [b["card_id"] for b in backups]`
- 工兵循环里每次 implement 之后追加一条 `attempts`，字段照 spec；
  试跑结果回填 `试跑` / `报错`；失败时 `失败代码 = {f["path"]: f["content"] for f in patch["new_files"]}`，
  成功时留空 dict
- 执行器成功/失败之后各记一步（角色写 `训练`）

- [ ] **Step 4: 跑到全绿**

Run: `.venv/bin/python -m pytest tests/test_round_record.py tests/test_smoke_run.py -q`
Run: `.venv/bin/python -m pytest -q` → 0 失败

- [ ] **Step 5: 提交并推送**

```bash
git add agent/loop.py tests/test_round_record.py
git commit -m "feat(loop): 记下候选卡、备胎、工兵每一次尝试、每步的时间和 token"
git push origin main
```

---

### Task 3: 接线 —— 终端边跑边打，每轮落一份 markdown

**Files:**
- Modify: `agent/loop.py`（`run_round(tracer=...)`、`run_session(trace=True)`、落盘循环）
- Test: `tests/test_trace_wiring.py`

**Interfaces:**
- Consumes: `agent.trace.Tracer`、`agent.trace.write_round_md`
- Produces: `run_round(..., tracer: "Tracer | None" = None)`、`run_session(..., trace: bool = True)`

- [ ] **Step 1: 写失败测试** `tests/test_trace_wiring.py`

```python
def test_一轮把八个步骤都打出来():
    行 = []
    log = _一轮(_试跑执行器(fail_times=1), tracer=trace.Tracer(printer=行.append))
    文 = "\n".join(行)
    for 步 in ("医生", "筛卡", "军师", "调度", "工兵", "试跑", "训练", "复盘官", "本轮"):
        assert 步 in 文, 步
    assert 文.count("工兵") == 2                  # 写错一次、改对一次

def test_不传tracer就一行都不打(capsys):
    _一轮(_试跑执行器())
    assert capsys.readouterr().out == ""

def test_一整场每轮都落一份markdown(tmp_path):
    summary = _一场(logs_dir=tmp_path, rounds=2)
    出 = sorted((tmp_path / "rounds" / summary.run_id).glob("round_*.md"))
    assert len(出) == 2 and "## 复盘官" in 出[0].read_text(encoding="utf-8")

def test_写markdown失败只记一笔恢复事件_整场不停(monkeypatch, tmp_path):
    def 炸(*a, **k):
        raise OSError("磁盘满了")
    monkeypatch.setattr(trace, "write_round_md", 炸)
    summary = _一场(logs_dir=tmp_path, rounds=1)
    assert summary.rounds_run == 1                # 照跑完
```

- [ ] **Step 2: 跑，确认失败**

Run: `.venv/bin/python -m pytest tests/test_trace_wiring.py -q`
Expected: FAIL（`run_round` 不认 `tracer`）

- [ ] **Step 3: 实现**

- `run_round` 签名末尾加 `tracer: "trace.Tracer | None" = None`；
  开头 `tracer = tracer or trace.Tracer(enabled=False)`，这样底下直接调不用判空
- 每个 `_记一步` 旁边跟一句 `tracer.step("医生", log.diagnosis)`（内容照 spec 的步骤名）
- `run_session` 加 `trace: bool = True`，循环里：

```python
        tracer = trace.Tracer(enabled=trace_on)
        tracer.round_header(rid, FIDELITY_LADDER[rung])
```

  （参数名与模块名会撞，`run_session` 内部把参数收成 `trace_on`）
- 落盘循环追加一项：`("角色日志", lambda: trace.write_round_md(logs_dir, log))`
- 离线演习那个 `on_round` 一行摘要保留不动

- [ ] **Step 4: 跑到全绿 + 离线演习肉眼看一遍**

Run: `.venv/bin/python -m pytest -q` → 0 失败
Run: `.venv/bin/python -m agent.cli run --offline --rounds 3 --fresh`
看终端每轮八个步骤都在，`logs/offline/rounds/<场次>/round_001.md` 能打开

- [ ] **Step 5: 提交并推送**

```bash
git add agent/loop.py tests/test_trace_wiring.py
git commit -m "feat(loop): 每一步边跑边打，每轮落一份 markdown 全文"
git push origin main
```

---

### Task 4: 真数据验证

**Files:**
- 无源码改动（发现问题就回到对应 Task）

- [ ] **Step 1: 真执行器跑一轮**（不调大模型，用写好剧本的假模型，工兵第一次故意写错）
  —— 复用上次那个探针脚本，加 `tracer`
- [ ] **Step 2: 把终端输出和生成的 `round_001.md` 发给用户看**
- [ ] **Step 3: 在 `docs/待做清单.md` 里把 A1 划掉，写一句怎么验的**
- [ ] **Step 4: 提交并推送**

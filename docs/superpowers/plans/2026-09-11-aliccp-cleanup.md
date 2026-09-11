# AliCCP 大扫除 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 仓库里只剩 KuaiRand 一条路：旧 AliCCP 流水线、旧格式练习材料、agent 核心的兼容分支全部拆掉，KuaiRand 在用的共享代码不受影响。

**Architecture:** 先把共享的零件加载代码搬到 `harness/ops.py`，切断 KuaiRand 路径对 `harness/executor.py` 的依赖；再把离线演习、假成绩单、通用测试换成 KuaiRand 成绩单格式，删掉 agent 核心的兼容分支；然后整块删除 AliCCP 专用代码；最后重写文档。每个 Task 单独提交、全量测试绿。

**Tech Stack:** Python 3.14、pytest、PyYAML

Spec: `docs/superpowers/specs/2026-09-11-aliccp-cleanup-design.md`

## Global Constraints

- 每个 Task 结束：`.venv/bin/python -m pytest -p no:cacheprovider -q tests kuairand_goat_bridge/tests` 0 失败；
  `.venv/bin/python -m agent.cli check` 通过；`.venv/bin/python -m agent.cli run --offline --rounds 6 --fail-round 3` 跑完
- KuaiRand 成绩单格式以 `goat_executor._health_report` + `diagnostics.py` 为准：
  `验证集: {GAUC, nDCG@5, 主分, 总行数, 用户数}`，分组块摊平在顶层
  （分组条目：`分组 / 行数 / 占比 / 正样本数 / GAUC / nDCG@5 / 主分 / 用户数`）
- 退役的字段与指标名：`click conversion ctcvr sample_id common_id`、数字字段（`206`、`109_14`…）、
  `点击分 购买分 点击AUC 购买AUC`
- `docs/` 下历史记录不改；提交信息不加 Co-Authored-By

---

### Task 1: 共享的零件加载代码搬到 harness/ops.py

**Files:**
- Create: `harness/ops.py`（`load_op_class`、`load_feature_ops`、`apply_feature_ops`，代码从 executor.py 原样搬）
- Modify: `harness/executor.py`（改为从 `.ops` 导入，保留 `_load_op_class_by = load_op_class` 别名到 Task 4 删除为止）
- Modify: `harness/deep.py`（两处 `from .executor import _load_op_class_by` → `from .ops import load_op_class`）
- Modify: `kuairand_goat_bridge/examples/goat_trainer.py`（`from harness.ops import ...`）
- Modify: 测试里所有 `from harness.executor import load_feature_ops / apply_feature_ops / _load_op_class_by`
- Test: `tests/test_harness_ops.py`（新建）

**Interfaces:**
- Produces: `harness.ops.load_op_class(rel_path, methods, kind)`、`load_feature_ops(config)`、`apply_feature_ops(ops, train, others)`

- [ ] **Step 1: 写失败测试**

```python
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

_EARLY_STOP = {"train": {"early_stopping": {
    "enabled": True, "impl": "modules/train/early_stopping.py",
    "monitor": "primary", "mode": "max", "patience": 3, "min_delta": 0.0005}}}


def test_KuaiRand路径装零件不再碰旧执行器():
    code = (
        "import sys; import harness.deep as d; "
        f"d.load_train_ops({_EARLY_STOP!r}); "
        "print('harness.executor' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_goat_trainer不从旧执行器导入():
    src = (ROOT / "kuairand_goat_bridge/examples/goat_trainer.py").read_text(encoding="utf-8")
    assert "harness.executor" not in src
    assert "from harness.ops import" in src
```

- [ ] **Step 2: 跑，确认失败**（装早停零件会 import harness.executor；goat_trainer 还从 executor 导入）
- [ ] **Step 3: 搬代码**：executor.py 第 56–137 行（`_load_op_class_by`、`_load_op_class`、`load_feature_ops`、`apply_feature_ops`）连同它们用到的 import（`importlib.util`、`pathlib`、`pandas`、`emit`、`ROOT`）原样搬进 ops.py，`_load_op_class_by` 改名 `load_op_class`；executor.py 顶部 `from .ops import load_op_class, load_feature_ops, apply_feature_ops` 并保留 `_load_op_class_by = load_op_class`；改 deep.py、goat_trainer.py、测试的 import
- [ ] **Step 4: 跑新测试 + 全量**
- [ ] **Step 5: 提交** `refactor(harness): 零件加载搬到 harness/ops.py，KuaiRand 路径不再依赖旧执行器`

### Task 2: 离线演习和假成绩单换成 KuaiRand 格式

**Files:**
- Modify: `agent/offline.py`（ScriptedLLM 从 schema 读指标名；DriftingExecutor 出 KuaiRand 成绩单）
- Modify: `agent/fixtures/health_reports.yaml`（5 份全部改成 KuaiRand 格式；`任务打架` → `前五名没跟上`）
- Delete: `agent/fixtures/假成绩单_5份.md`（yaml 的手抄副本，无人引用）
- Test: `tests/test_offline_speaks_kuairand.py`（新建）

**Interfaces:**
- Produces: `DriftingExecutor(base_gauc=0.6638, base_ndcg=0.5357, gain=0.003, decay=0.45, fail_rounds=(), seconds=12.0)`；
  `report()` 返回 KuaiRand 格式成绩单

- [ ] **Step 1: 写失败测试**

```python
import re
import yaml

from agent import schemas
from agent.cli import FIXTURES
from agent.knowledge import CardLibrary, SymptomVocab
from agent.loop import run_session
from agent.offline import DriftingExecutor, ScriptedLLM

RETIRED = re.compile(r"点击分|购买分|点击AUC|购买AUC|\bclick\b|conversion|ctcvr|\b\d{3}_\d{2}\b")


def test_假执行器出的是KuaiRand成绩单():
    r = DriftingExecutor().report()
    assert {"GAUC", "nDCG@5", "主分"} <= set(r["验证集"])
    assert schemas.metric_names(r) == list(schemas.METRICS)
    assert not RETIRED.search(yaml.safe_dump(r, allow_unicode=True))


def test_假成绩单全是KuaiRand格式():
    fixtures = yaml.safe_load(FIXTURES.read_text(encoding="utf-8"))
    for name, fx in fixtures.items():
        assert {"GAUC", "nDCG@5", "主分"} <= set(fx["report"]["验证集"]), name
    assert not RETIRED.search(FIXTURES.read_text(encoding="utf-8"))


def test_演习整场说的都是当前任务的指标名(tmp_path):
    v = SymptomVocab.load()
    ex = DriftingExecutor(fail_rounds=(3,))
    summary = run_session(
        llm=ScriptedLLM(), vocab=v, cards=CardLibrary.load(v), executor=ex,
        initial_report=ex.report(), module_interface="", example_module="",
        current_config="", rounds=4, log_dir=tmp_path)
    text = "\n".join(p.read_text(encoding="utf-8") for p in tmp_path.rglob("*.json"))
    assert not RETIRED.search(text)
```

（`run_session` 的实际参数名以 agent/loop.py 为准，写测试时对照一次）

- [ ] **Step 2: 跑，确认失败**
- [ ] **Step 3: 改 offline.py**：
  - 医生 `affects` 取 `self._enum(schema, "properties","findings","items","properties","affects","items")`；
    证据写「按视频曝光次数分组最低桶 GAUC 0.612 比最高桶 0.681 低 0.069（第 N 轮）」
  - 军师 `expected` 的键取 `schema["properties"]["proposals"]["items"]["properties"]["expected"]["properties"]`；
    `how_to` 改成 author_id 兜底；targets 兜底名改成 `冷门视频排不上去`
  - 工兵 self_check 改成「未使用禁用字段 long_view 等」
  - 复盘官 `actual` 的键取 reflector schema 的 `actual.properties`
  - DriftingExecutor：`gauc += gain`、`ndcg += gain * 1.4`、`主分 = (gauc + ndcg) / 2`；
    `report()` 出 `验证集`、`训练集`（都带 GAUC/nDCG@5/主分）、`训练诊断: {实际特征: [user_id, video_id, author_id, tab, duration_bucket], 装上的零件: []}`、
    `按视频曝光次数分组`（两条，字段同 Global Constraints）
- [ ] **Step 4: 改 fixtures**：5 份按 KuaiRand 块名重写，expect 用现行 12 个病名
- [ ] **Step 5: 跑新测试 + 全量 + check + 离线演习**
- [ ] **Step 6: 提交** `refactor(agent): 离线演习和假成绩单换成 KuaiRand 格式`

> **执行时调整顺序（2026-09-11）：先做 Task 4，再做 Task 3。** Task 3 的守卫测试点名的
> `harness/deep.py` 双塔默认路径和 `agent/noise.py` 本身就是 AliCCP 专用代码；先删它们会
> 让 RealExecutor 的测试全挂，Task 3 没法单独绿着收尾。所以先整块删旧代码（连同只测它的
> 测试），再删兼容分支、迁测试。Task 4 验证时单独跳过还是红的守卫测试文件，不提交它。

### Task 3: 通用测试迁到 KuaiRand 格式，删 agent 核心的兼容分支

**Files:**
- Modify: `tests/test_offline.py`（下面 61 个测试里的旧格式数据）、`tests/test_monitor_metric_names.py`、
  `tests/test_knowledge_speaks_current_task.py`、`tests/test_new_files_must_be_new.py`
- Modify: `agent/schemas.py`（`METRIC_PAIRS` 只留 KuaiRand 一套）、`agent/roles.py`（`_MAIN_METRIC_KEYS = ("主分",)`、
  `FORBIDDEN_FIELDS` 去掉 AliCCP 五个、危险信号文案「点击AUC差值」→「主分差值」）、`agent/loop.py`（`total_score` 去掉双指标退路、
  注释里描述现行行为的「点击分/购买分」改成 GAUC / nDCG@5）
- Test: `tests/test_no_retired_task_in_code.py`（新建）

- [ ] **Step 1: 写失败测试** —— agent/、harness/deep.py、harness/ops.py、modules/ 的**代码行**（去掉注释和 docstring 后）
  不许出现退役字段或指标名的字符串字面量：

```python
import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
RETIRED = re.compile(r"^(点击分|购买分|点击AUC|购买AUC|click|conversion|ctcvr|sample_id|common_id|\d{3}(_\d{2})?)$")
SCOPE = [*ROOT.glob("agent/*.py"), ROOT / "harness/deep.py", ROOT / "harness/ops.py",
         *ROOT.glob("modules/**/*.py")]


def _string_literals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                  and n.body and isinstance(n.body[0], ast.Expr)
                  and isinstance(n.body[0].value, ast.Constant)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings:
            yield n.value


def test_代码里不再有退役任务的字段和指标名():
    hits = [(p.relative_to(ROOT).as_posix(), s) for p in SCOPE if p.exists()
            for s in _string_literals(p) if RETIRED.match(s)]
    assert not hits, hits
```

- [ ] **Step 2: 跑，确认失败**（schemas / roles / offline 里还有）
- [ ] **Step 3: 删兼容分支**（schemas、roles、loop，见 Files）
- [ ] **Step 4: 迁测试**，规则：成绩单里 `点击分→GAUC`、`购买分→nDCG@5`，并补 `主分 = 两者平均`；
  `expected / actual / affects` 里 `点击AUC→GAUC`、`购买AUC→nDCG@5`；`冷门商品学不动→冷门视频排不上去`；
  专门测"两套任务并存"的测试直接删：`test_成绩单_AliCCP的叫法不许被顺手改掉`、`test_指标名_AliCCP成绩单还是老那套`、
  `test_危险信号_两边指标名对不上就不算`、test_monitor_metric_names 里 AliCCP 那几条；
  test_knowledge_speaks_current_task 的 `_retired_metric_names()` 改为显式列表（METRIC_PAIRS 不再保留旧任务）。
  需要迁的 61 个测试（行号为改动前）：120 137 148 181 206 270 287 507 624 643 660 677 694 709 723 736 778 786
  838 857 1163 1177 1256 1287 1329 1357 1382 1400 1415 1428 1442 1467 1492 1511 1527 1935 2095 2104 2112 2263
  2758 2825 2837 2853 2897 2938 2962 3049 3109 3132 3207 3256 3406 3444 3534 3566 3578 3584 3594 3700 3711
  （1467/1492/1511 三个测 finalize，随 Task 4 删）
- [ ] **Step 5: 跑新测试 + 全量 + check + 离线演习**
- [ ] **Step 6: 提交** `refactor(agent): 通用测试迁到 KuaiRand 格式，删掉兼容旧任务的分支`

### Task 4: 删 AliCCP 专用代码

**Files:**
- Delete: `harness/executor.py`、`harness/data.py`、`agent/noise.py`、`config/pipeline.yaml`、`config/pipeline.before-agent.yaml`、
  `modules/features/sequence_summary.py`、`modules/features/item_frequency.py`、`modules/models/mlp_aitm.py`、`modules/train/compliance_gate.py`
- Modify: `agent/cli.py`（只留 check / doctor / run --offline / intervene；`current_config` 改读
  `kuairand_goat_bridge/configs/kuairand_task.yaml` 的 `trainer_config`）
- Modify: `tests/test_offline.py`（删 74 个只测 AliCCP 专用代码的测试、3 个 finalize 测试、读 config/pipeline.yaml 的测试）

- [ ] **Step 1**：`tests/test_harness_ops.py` 追加：`harness/executor.py`、`harness/data.py`、`agent/noise.py` 不存在；
  `agent.cli` 的子命令集合 == {check, doctor, run, intervene}；跑，确认失败
- [ ] **Step 2**：删文件、删 cli 命令与 `_make_executor` 真执行器分支、删对应测试
- [ ] **Step 3**：全量 + check + 离线演习
- [ ] **Step 4**：提交 `chore: 删掉 AliCCP 旧流水线，只留 KuaiRand 一条路`

### Task 5: 文档

**Files:**
- Modify: `CLAUDE.md`（按 KuaiRand 重写：禁用字段 = roles.FORBIDDEN_FIELDS 那一份、R2/R3/R5/R7/R8/R11 纪律保留、
  评估口径 = 官方 evaluate 的 GAUC / nDCG@5 / primary、危险信号改成主分）、`README.md`、`agent/README.md`
  （真跑入口 `python -m kuairand_bridge`，演习入口 `agent.cli run --offline`）
- Modify: `knowledge/cards/seed_ensemble.yaml`（去掉「现成命令 agent.cli noise」）

- [ ] **Step 1**：`tests/test_no_retired_task_in_code.py` 追加：`CLAUDE.md`、`README.md`、`agent/README.md` 不含
  `AliCCP` 以外的退役字段（`click`、`conversion`、`109_14`、`点击AUC`…；提到 AliCCP 这个名字本身作为历史说明是允许的）；
  跑，确认失败
- [ ] **Step 2**：重写文档与卡片
- [ ] **Step 3**：全量 + check + 离线演习
- [ ] **Step 4**：提交 `docs: CLAUDE.md 和 README 按 KuaiRand 重写`，fetch 后 push

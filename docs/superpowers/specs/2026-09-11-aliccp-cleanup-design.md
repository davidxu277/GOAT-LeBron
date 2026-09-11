# AliCCP 大扫除 —— 设计

日期：2026-09-11 · 状态：用户已批准方案 A（分四步）

## 背景

比赛中途（08-31）数据集从 AliCCP 换成 KuaiRand-Pure。团队在旁边另接了一条
KuaiRand 的路（`kuairand_goat_bridge/`），旧的 AliCCP 流水线一直没拆。
两套并存的代价已经兑现过好几次：过期卡片、过期范文、过期的指标名单
都是旧任务的东西漏进了 agent。

比赛提交版由 tag `submission-techjam2026` 钉住，git 历史里旧代码都在。

## 盘点

| 类别 | 内容 | 处理 |
|---|---|---|
| 共用 | `harness/executor.py` 的 `_load_op_class_by`、`load_feature_ops`、`apply_feature_ops`（deep.py 和 goat_trainer 在用） | 挪到 `harness/ops.py` |
| AliCCP 专用代码 | `harness/executor.py` 其余部分（RealExecutor、内存守卫、LightGBM、recalibrate…）、`harness/data.py`、`agent/noise.py`；`agent/cli.py` 的 round / run 真跑路径 / predict / restore / finalize / noise；`config/pipeline.yaml`、`config/pipeline.before-agent.yaml`；`modules/features/sequence_summary.py`、`modules/features/item_frequency.py`、`modules/models/mlp_aitm.py`、`modules/train/compliance_gate.py` | 删 |
| AliCCP 格式的练习材料 | `agent/offline.py`（ScriptedLLM、DriftingExecutor）、`agent/fixtures/health_reports.yaml`、`agent/fixtures/假成绩单_5份.md` | 改成 KuaiRand 成绩单格式 |
| agent 核心的兼容分支 | `schemas.METRIC_PAIRS` 的 AliCCP 一套、`roles._MAIN_METRIC_KEYS` 的「点击分」、`roles.FORBIDDEN_FIELDS` 的 AliCCP 五个、loop 里"旧成绩单没有主分就退回双指标"一类退路 | 材料换完后删 |
| 测试（test_offline.py 244 个） | 74 个只测 AliCCP 专用代码；61 个测通用逻辑但用 AliCCP 格式假数据；109 个无关 | 跟着删 / 改格式保留 / 不动 |
| 文档 | `CLAUDE.md`（硬规则）、`README.md`、`agent/README.md`；多种子卡「现成命令 agent.cli noise」 | 按 KuaiRand 重写；`docs/` 下的开发日志等历史记录原样保留 |

## 四步（每步单独提交，每步全量测试绿）

1. **搬家**：新建 `harness/ops.py` 放三个共用函数；`executor.py` 从它转引，
   `deep.py`、goat_trainer、测试改为从 `harness.ops` 导入。不改任何行为。
2. **换材料**：离线演习、假成绩单、61 个通用测试换成 KuaiRand 格式
   （验证集 GAUC / nDCG@5 / 主分，分组块沿用 KuaiRand 成绩单的真实块名）。
   随后删 agent 核心的 AliCCP 兼容分支。
3. **删旧代码**：上表"AliCCP 专用代码"整列，连同只测它们的测试。
   `agent/cli.py` 保留 check / doctor / run --offline / intervene；
   真跑入口是 `python -m kuairand_bridge`。
4. **文档**：`CLAUDE.md` 按 KuaiRand 重写（禁用字段、R2 等纪律保留，
   评估口径改为官方 evaluate 的 GAUC / nDCG@5 / primary）；README、agent/README
   写清唯一的真跑入口；多种子卡去掉 `agent.cli noise` 的说法。

## 不做的事

- 不做 KuaiRand 版的量噪声工具 —— 第二阶段做消融实验时按新任务重写。
  删掉旧的 noise 之后，"涨多少才算真涨"退回默认门槛 roles.MIN_REAL_GAIN（0.0005）。
- 不改 `docs/` 下的历史记录（开发日志、baseline 笔记、交接文档……）。
- 不顺手重构 test_offline.py 的文件结构 —— 只删、只迁。

## 验收

- 每一步：全量测试 0 失败；`agent.cli check` 通过；`agent.cli run --offline` 跑完。
- 做完之后：代码、配置、提示词、卡片、测试里搜不到 AliCCP 专属的字段和指标名
  （`click` / `conversion` / `ctcvr` / `sample_id` / `common_id` / `109_14` 这类数字字段 /
  点击分 / 购买分 / 点击AUC / 购买AUC），只剩说明历史的注释。
- `python -m kuairand_bridge` 的真跑路径不受影响（kuairand_goat_bridge/tests 全绿）。

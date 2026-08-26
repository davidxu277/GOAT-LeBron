# Agent 大脑

负责人：成员3。这是四个 AI 角色以及把它们串起来的那段代码。

## 先跑起来

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# ① 不调用模型，零成本 —— 检查词表和卡片是否自洽
.venv/bin/python -m agent.cli check

# ② 需要凭据
export ANTHROPIC_API_KEY=sk-ant-...

# 用一份假成绩单跑医生
.venv/bin/python -m agent.cli doctor 一切正常

# 跑全部 5 份，对照标准答案
.venv/bin/python -m agent.cli doctor --all

# 用假执行器跑完整一轮：诊断 → 筛卡 → 提案 → 调度 → 实现 → 复盘
.venv/bin/python -m agent.cli round 正常起步

# 离线测试（不花钱）
.venv/bin/python -m pytest tests/ -q
```

**数据和模型都还没好，也能开发。** 假成绩单（`fixtures/health_reports.yaml`）和
假执行器（`loop.FakeExecutor`）让整条链路今天就能跑通。

## 文件

| 文件 | 干什么 |
|---|---|
| `knowledge.py` | 读病名词表和药方卡；**按病名筛卡片**就在这里，纯代码不花钱 |
| `schemas.py` | 四个角色的输出结构。医生的病名是 enum，**直接从 symptoms.yaml 生成** |
| `llm.py` | 唯一的 Claude 调用入口：结构化输出、重试、按角色记账 |
| `roles.py` | 四个角色 + 各自的额外校验 |
| `loop.py` | 一轮循环；与成员4 的接口（Scheduler / Executor）及参考实现 |
| `prompts/` | 四段提示词。改行为改这里，不要改代码 |
| `fixtures/` | 5 份假成绩单，含标准答案 |

## 两个设计要点

**① 病名说不出词表以外的词。**
医生输出的 `symptom` 字段是一个 JSON Schema enum，取值直接由 `symptoms.yaml`
生成。模型在物理上就说不出没定义过的病名 —— 这是"对暗号"最硬的实现方式，
比在提示词里叮嘱可靠得多。

**② 每个 AI 角色之间隔着一段普通代码。**

```
医生（花钱）→ 筛卡片（不花钱）→ 军师（花钱）→ 调度（不花钱）
            → 工兵（花钱，小模型）→ 校验+训练（不花钱）→ 复盘官（花钱）
```

查卡片、算性价比、跑校验都是查字典和算术，用 if 判断就够了。
省下的 token 直接计入评分的"资源消耗"（占 15%）。

## 三条最重要的校验（都有测试兜住）

这些是**代码强制**的，不靠提示词自觉：

1. **证据必须带数字** —— 医生说不出"明显偏低"这种没有数字的话
2. **禁用字段拦截** —— 工兵写的代码里出现 `conversion` 等五个字段直接打回（CLAUDE.md R1）
3. **分数涨了但毛病没治好 → 必须判「说不清」** —— 防止 Agent 沿着错误的因果链一路走下去

## 与队友的接口

| 谁 | 提供什么 | 现在用什么顶着 |
|---|---|---|
| 成员1 + 成员4 | 真实成绩单 | `fixtures/health_reports.yaml` |
| 成员2 | 20 张药方卡、零件接口、范文 | `knowledge/cards/` 里 2 张种子卡 + `cli.py` 里的 STUB |
| 成员4 | `Scheduler`、`Executor` | `loop.CostAwareScheduler`、`loop.FakeExecutor` |

`loop.py` 里的 `Scheduler` / `Executor` 是 Protocol，成员4 写自己的实现替换即可，
`run_round` 不用改。

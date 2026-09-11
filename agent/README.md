# Agent 大脑

四个 AI 角色（医生 → 军师 → 工兵 → 复盘官），以及把它们串起来的外层循环。

## 先跑起来

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

# ① 不调用模型，零成本 —— 检查病名词表、药方卡、假成绩单是否自洽
.venv/bin/python -m agent.cli check

# ② 不调用模型，零成本 —— 整场演习：假模型 + 假执行器把四个角色和外层循环跑一遍
.venv/bin/python -m agent.cli run --offline --rounds 6 --fail-round 3

# ③ 离线测试（不花钱）
.venv/bin/python -m pytest tests/ -q

# ④ 需要凭据：拿假成绩单调医生的提示词
export ANTHROPIC_API_KEY=...        # 或者 AGENT_PROVIDER=deepseek + DEEPSEEK_API_KEY
.venv/bin/python -m agent.cli doctor 一切正常
.venv/bin/python -m agent.cli doctor --all      # 5 份全跑，对照标准答案
```

真数据的自主迭代走 KuaiRand 那条路：
`python -m kuairand_bridge goat-run --config kuairand_goat_bridge/configs/kuairand_task.yaml`
（见仓库根目录的 README）。

## 文件

| 文件 | 干什么 |
|---|---|
| `knowledge.py` | 读病名词表和药方卡；**按病名筛卡片**就在这里，纯代码不花钱。卡片的指标名启动时就校验 |
| `schemas.py` | 四个角色的输出结构。医生的病名是 enum，**直接从 symptoms.yaml 生成**；指标名只有一张表 |
| `llm.py` / `llm_deepseek.py` | 大模型调用入口：结构化输出、重试、按角色记账 |
| `roles.py` | 四个角色 + 各自的校验（证据、禁用字段、复盘官不许自欺）；工兵的范文按方案环节取 |
| `loop.py` | 一轮、一整场、三本账（耗时 / 靠谱度 / 待议架）、爬山回滚、判停 |
| `offline.py` | 假模型 + 假执行器，出 KuaiRand 格式的成绩单 —— 演习和测试用 |
| `prompts/` | 四段提示词。改行为改这里 |
| `fixtures/` | 5 份 KuaiRand 格式的假成绩单，含标准答案（其中两份是陷阱题） |

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
省下的 token 直接计入评分的"资源消耗"。

## 三条最重要的校验（都有测试兜住）

这些是**代码强制**的，不靠提示词自觉：

1. **证据必须落到成绩单上** —— 要么带数字，要么点名它读的是成绩单的哪一块
2. **禁用字段拦截** —— 工兵写的代码里出现 `long_view`、`play_time_ms` 等禁用字段直接打回（CLAUDE.md R1）
3. **分数涨了但毛病没治好 → 必须判「说不清」** —— 防止 Agent 沿着错误的因果链一路走下去

# GOAT-LeBron

TikTok TechJam · 赛道二：面向推荐系统的自主机器学习研究智能体

---

## 我们在做什么

平时算法工程师干活是这样的：拿到数据和评分标准 → 试一个想法 → 写代码 → 训练 →
看分数 → 想下一步 → 再试一遍，一轮一轮直到分数不再涨。

**我们要造的，是一个能代替人干这件事的机器人。** 把数据和评分规则交给它，
然后不碰键盘，看它自己看数据、自己写代码、自己训练、自己复盘、自己决定下一步。

> 我们做的不是模型，是造模型的机器人。

### 和已有系统的差别

已有的同类系统（如 AIDE）本质是在代码空间里**盲目搜索** —— 不停改代码看分数涨没涨，
但它不知道模型到底哪里差。

我们的做法是**先诊断，再开药**：

1. **先体检** —— 分组看分数（冷门商品 vs 热门、新用户 vs 老用户）、看概率准不准、看有没有过拟合
2. **再开药** —— 根据诊断出的病，从方法库里找对症的方案，并写清楚为什么选它
3. **知道自己的快速实验有多可信** —— 专门校准"小份数据上的结论在大份数据上是否成立"

---

## 任务与评估

**数据集**：AliCCP（阿里巴巴电商，曝光 → 点击 → 转化漏斗）

**要预测**：每条曝光记录输出两个概率 —— 会不会点击、点击后会不会转化

**评估口径**（固定，不许改）：

| 指标 | 在哪些记录上算 | 正样本 |
|---|---|---|
| CTR AUC | 全部曝光 | click = 1 |
| CVR AUC | **仅 click = 1 的记录** | conversion = 1 |

**排名依据**：相对官方基线的绝对提升，两个指标等权平均。
最终只在隐藏测试集上评估一次，使用收敛时验证集最佳的 checkpoint。

---

## 仓库结构

```
CLAUDE.md / AGENTS.md   ← 红线清单。写任何代码前必读，AI 工具会自动读取
prep/                   ← 原始数据解析、分层采样、数据切分
harness/                ← 运行沙箱、评分脚本、超时与错误恢复
config/                 ← 流水线配置（Agent 通过改这里来做实验）
modules/                ← 可替换零件
  features/             ←   特征类
  models/               ←   模型类
agent/                  ← Agent 大脑（成员3）
  knowledge.py          ←   读词表与卡片；按病名筛卡片
  schemas.py            ←   四个角色的输出结构
  llm.py                ←   Claude 调用入口 + 按角色记账
  llm_deepseek.py       ←   DeepSeek 入口（OpenAI 兼容接口）
  roles.py              ←   医生 / 军师 / 工兵 / 复盘官
  loop.py               ←   一轮 run_round + 一整场 run_session + 两个账本
  noise.py              ←   噪声带：同配置换种子，量出测量误差有多大
  offline.py            ←   假模型 + 假执行器，不花钱演习整场
  prompts/              ←   四段提示词
  fixtures/             ←   假成绩单，用于离线调试
knowledge/              ← 方法知识库
  symptoms.yaml         ←   12 个病名（医生与卡片之间的"暗号"）
  卡片格式.md            ←   药方卡规范 + 样例
  cards/                ←   药方卡本体
docs/                   ← 接口约定
logs/                   ← 逐轮运行日志（内容不入库）
```

## 文档索引

| 文件 | 讲什么 | 谁必读 |
|---|---|---|
| [CLAUDE.md](CLAUDE.md) | **12 条红线 + 5 个危险信号 + 自查清单** | **全员，动手前** |
| [docs/开发日志.md](docs/开发日志.md) | 一行一条，谁改了什么。**每次提交顺手加一行** | 全员 |
| [docs/baseline笔记.md](docs/baseline笔记.md) | **baseline 长什么样、洞在哪、两个待确认的问题** | **全员，动手前** |
| [knowledge/symptoms.yaml](knowledge/symptoms.yaml) | 12 个病名及判定规则 | 成员1、2、3 |
| [knowledge/卡片格式.md](knowledge/卡片格式.md) | 药方卡的六个栏目 + 两张样例卡 | 成员2 |
| [docs/四个角色接口.md](docs/四个角色接口.md) | 医生/军师/工兵/复盘官的输入输出 | 成员3、4 |
| [agent/README.md](agent/README.md) | Agent 大脑怎么跑、怎么改 | 成员3、4 |
| [docs/评审说明.md](docs/评审说明.md) | 现状、已知弱点、待决策的设计 | **reviewer 从这里开始** |

---

## 分工

| | 角色 | 负责 |
|---|---|---|
| 成员1 | 数据与评估 | 原始数据解析、分层采样、数据切分、AUC 与分组指标 |
| 成员2 | 模型与方法库 | 基线复现、ESMM、可替换零件、20 张药方卡 |
| 成员3 | Agent 大脑 | 医生 / 军师 / 工兵 / 复盘官四个角色 |
| 成员4 | 执行与可靠性 | 运行沙箱、错误恢复、调度器、预算与收敛判定 |
| 全员 | 日志与交付 | 各自记录自己的实验，最后统一整理 |

---

## 跑起来

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

**不花钱的三条**（不联网、不调用模型）：

```bash
.venv/bin/python -m agent.cli check              # 病名词表与药方卡是否自洽
.venv/bin/python -m pytest tests/ -q             # 57 个离线测试
.venv/bin/python -m agent.cli run --offline --rounds 8   # 假模型演习整场
```

演习模式用假模型 + 假执行器把整条链路从头跑到尾，专门用来验证**接线**：
状态有没有在轮与轮之间传下去、角色炸了能不能恢复、不涨了会不会自己停。
可以按需制造事故：

```bash
.venv/bin/python -m agent.cli run --offline --rounds 8 --fail-round 3 --fail-role-call 2
```

演习日志写在 `logs/offline/`，不会污染交付用的 `logs/rounds.jsonl`。

**真跑**（需要凭据 + 数据）：

```bash
export AGENT_PROVIDER=deepseek DEEPSEEK_API_KEY=...      # 或 ANTHROPIC_API_KEY=sk-ant-...

# 先量一次噪声带：同配置换 3 个种子，看分数自己抖多少
.venv/bin/python -m agent.cli noise --seeds 3 --train data/train --val-features data/val

# 自主迭代，中途不需要人碰键盘
.venv/bin/python -m agent.cli run --rounds 20 \
    --train data/train --val-features data/val \
    --baseline-ctr 0.xxxx --baseline-cvr 0.xxxx
```

跑完会打印结果表（交付物 #5），并把逐轮日志、两个账本、最佳版本的成绩单
全部落在 `logs/` 下。中途任何一轮崩掉都不会中断整场。

调单个环节：

```bash
.venv/bin/python -m agent.cli doctor --all       # 5 份假成绩单对照标准答案
.venv/bin/python -m agent.cli round 正常起步      # 只跑一轮，看四个角色各说了什么
```

*（待补：数据下载与预处理命令、NISE 官方基线分数）*

## 什么算「人工干预」

赛题按「达到收敛所需的人工干预次数」评自主性。这个数只有在边界说清楚的前提下
才有意义，所以我们把线划在这里 —— **跑之前定好，跑完不改**：

| 不算干预（准备与搭建） | 算干预（跑起来之后插手） |
|---|---|
| 下载与预处理数据、切分数据集 | 中途改配置或改代码 |
| 写药方卡、写提示词、定病名词表 | 手动杀掉某一轮、手动重启进程 |
| 决定跑几轮、给多少预算、选起步数据档位 | 手动指定提交哪一版 |
| 装环境、修我们自己代码里的 bug（跑之前） | 跑到一半修 bug 然后接着跑 |

记录方式 —— 任何人插手，当场敲一条：

```bash
.venv/bin/python -m agent.cli intervene "第 7 轮撞 OOM，手动把 batch 调小了" --round 7
```

它会进 `logs/interventions.jsonl`，正在跑的那一场下一轮就会把它算进逐轮日志
和最终结果表。**"非零"随手可得，报出来的 0 才是一个观测值，不是写死的常量。**

## 交付物在哪

一场跑完，`logs/` 下就是全套：

| 文件 | 对应交付物 |
|---|---|
| `rounds.jsonl` | #3 逐轮日志：假设 / 代码全文 / 指标 / 错误恢复 / 干预 |
| `narrative.md` | #3 的人话版：一整场压成一条故事线 |
| `session_summary.json` | #5 结果表：最佳分、与基线的差值、token 总量、GPU 小时 |
| `snapshots/round_XX.json` | #4 的还原依据：每一轮的配置 + 零件清单 |
| `best_report.json` | 最终提交那一版的完整成绩单 |

要交第 5 轮那一版（假设它是验证集最佳）：

```bash
.venv/bin/python -m agent.cli restore 5 --out submission/
```

把那一轮的配置和零件原样还原出来，照着重跑一次即可产出提交用的预测结果。
**必须这样做** —— 工兵的改动是叠加在同一份配置上的，跑到第 20 轮时磁盘上
只剩最后那个叠加态，第 5 轮的样子早被盖掉了。

## 局限性与改进方向

*（待补：提交前填写）*

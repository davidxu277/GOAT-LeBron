# Agent 部分 TODO · 成员2 + 成员3

> 2026-08-27 · 今晚交付。目标：把循环打磨到"数据和真执行器一插上就能挂机"。
> 车道划分防 git 冲突 —— 成员3：`agent/loop.py` `roles.py` `schemas.py` `knowledge.py`；
> 成员2：`agent/prompts/` `agent/cli.py` `knowledge/cards/` `modules/` `config/` `docs/`。

---

## 第 0 段 · 两人一起：排最大的暗雷（半小时）

- [ ] 带真实 API key 跑 `python -m agent.cli doctor --all`（5 份假成绩单对照标准答案）
- [ ] 跑 `python -m agent.cli round 正常起步`（完整一轮走通）
- [ ] 炸了当场分诊：**输出内容不对 → 成员2 改 prompts；格式过不了 schema → 成员3 看 schema 是否过严**
- [ ] 重点看两道陷阱题：第 4 份（干净成绩单不许硬编病）、第 5 份（不许拿小样本数字当证据）

这条路一次都没跑过，是全项目最大的未验证风险。没跑通之前别开工别的。

---

## 成员3（室友）· 按顺序做 —— **08-27 晚全部完成**

> 验收方式见文末「实际验收记录」。真实 API 那条路仍未跑过（开发机没凭据）。

### ① 外层循环 `run --rounds N` —— 最大的缺口，自主迭代现在根本不存在

- [x] 新 cli 子命令，循环调 `run_round`，维护跨轮状态：
  - `parent_result`（只在成功轮更新）
  - `history_brief`（每轮压缩成 1-2 行：轮次/选卡/verdict/实际变化）
  - 已试卡黑名单（verdict = 猜错了 / 没跑起来 的卡；"说不清"不拉黑）
  - TimeLedger / PriorLedger 每轮落盘
  - token 花费 → `budget_left` 三档（<60% 宽裕 / 60-85% 一般 / >85% 紧张）
- [x] 逃生舱：连续 2 次 no_finding → 升 `FIDELITY_LADDER` 下一档
      （给 `run_round` 加 `fidelity_override` 可选参数即可，不用动调度器）
- [x] 停止条件：跑满 N 轮或预算耗尽

### ② 三个角色包 try —— 挂机保险丝

- [x] 医生（loop.py:167）、军师（:180）、复盘官（:229）各包
      `try/except SchemaViolation`（+ 网络异常）→ 记 recoveries → 本轮作废继续
- [x] `executor.run` 也包一层（协议说返回 ok=False，但真实现难保不 raise）
- [x] 别吞 `KeyboardInterrupt`
- [x] 复盘官失败时指标必须已落盘（见 ③ 的赋值顺序）

### ③ RoundLog 存全 —— 交付物 #3 合规

- [x] 加 `metrics` 字段（`result.health_report` 原文）
- [x] 加 `patch_files` 字段（`{路径: 完整代码}`）
- [x] **两个赋值都放在调用复盘官之前**（:219 拿到 result 就赋，:229 才 reflect）

### ④ PriorLedger + 三根断线接上 + 失败记账

- [x] `PriorLedger`：照 `docs/方法库进度.md` 第五节规格，仿 TimeLedger 写
      （规则算分：猜对了且超噪声带 +0.15 / 目标病改善 +0.05 / 猜错了 −0.10 /
      没跑起来 −0.15，限幅 [0.05, 0.95]）
- [x] 应用：每轮开始 `card.prior = ledger.value(card.id, card.prior)`（yaml 原文不动）
- [x] 断线 1：`cards.match(symptom_ids, exclude_ids=黑名单)`（loop.py:177 现在没传）
- [x] 断线 2：`CostAwareScheduler(tried_cards=黑名单)`（cli 现在传空）
- [x] 断线 3：`roles.propose(tried_before=已试清单)`（run_round 现在没传——
      军师提示词说"试过的不要重复提"，但它根本不知道试过什么）
- [x] 执行失败时纯代码合成 `{"verdict": "没跑起来"}` + `ledger.apply(-0.15)`，不调大模型

### ⑤ 六条一行流校验（合计 <20 行）

- [x] 复盘官（roles.py:198 附近）：目标指标全部 ≤0 → 禁判"猜对了"
- [x] 复盘官："猜错了" → prior_delta 不许为正
- [x] 复盘官：promote=true 仅当"猜对了"
- [x] 军师 schema：expected 数值限 [−0.05, +0.05]
- [x] 工兵（roles.py:148 附近）：路径含 `..` 直接打回
- [x] 工兵：config_patch 键只许 `features.` / `model.` / `train.` 前缀

### ⑥ knowledge.py 读「失败信号」

- [x] `Card` 加 `failure_signals: str = ""`，`from_dict` 读 `d.get("失败信号")`
- [x] 只喂复盘官：在 `reflect()` 拼 user 消息处追加，**别改 `as_prompt_block`**
      （那个军师也在用，会被负面信号带偏）

---

## 成员2 · 按顺序做

### ① 提示词迭代（主战场，贯穿全天）

- [ ] doctor.md：对照陷阱题 4/5 的表现改；"样本量偏少"的措辞和成绩单实际字段名对齐
- [ ] strategist.md：盯两段因果是否走完（尤其"病根→这招怎么对症"）；
      发现绕过禁用词表的同义词就往 `roles.py` 的 `_HEDGE_WORDS` 加
- [ ] implementer.md：盯完整文件输出（不是 diff）、接口是否严格照 base.py
- [ ] reflector.md：**室友 ⑤ 的新校验落地后，提示词同步补对应规矩**
      （校验是墙，提示词是墙上的标语，缺一个就白烧重试 token）
- [ ] 每改一版跑一版，一版一个 commit；重试率是提示词质量的温度计

### ② STUB 换真货（cli.py:27-55）

- [ ] `STUB_INTERFACE` → `read_text()` 读 `modules/base.py`
- [ ] `STUB_EXAMPLE` → 读 `modules/train/early_stopping.py`
- [ ] `STUB_CONFIG` → 新建 `config/pipeline.yaml`（键名和 25 张卡的「怎么实现」对上：
      `features.*` / `model.*` / `train.early_stopping.*` / `train.seed`），cli 改读文件。
      这份文件顺便成为全队的配置 schema 事实标准
- [ ] 可选加分：把原 STUB_EXAMPLE 的 FrequencyBucket 落成真文件
      `modules/features/frequency_bucket.py`（5 分钟，凑齐"加特征"类范文），
      然后按卡的「属于哪个环节」选对应范文（一行 if，接线跟室友商量放哪）

### ③ 修断链（1 分钟）

- [ ] CLAUDE.md 和 AGENTS.md 的 R6、R10：`docs/接口约定.md` → `docs/四个角色接口.md`
      （两个文件是复制关系，共 4 处）

### ④ 卡片现场修（被动触发）

- [ ] 工兵在某张卡上反复失败 → 多半是「怎么实现」太抽象，现场补伪代码
- [ ] 新写内容注意 yaml 裸冒号（出处栏沿用块标量写法）

### ⑤ 有余力：军师数字摘要

- [ ] 新文件 `agent/summary.py`：纯函数 `summarize_report(health_report) -> str`
      （两个总 AUC 及与上一版的差 / 分桶表含每桶正样本数 / 噪声带 / 当前配置一行，
      ≤300 token），拿 fixtures 直接写离线测试；接线让室友做

---

## 第 2 段 · 合拢验收（傍晚前，两人一起）

- [ ] 5 份假成绩单全部走完整 `round`，陷阱题不上当
- [ ] `run --rounds 5` 连跑 5 轮不炸，中途人为制造一次失败（改坏一个 fixture）看恢复
- [ ] `logs/rounds.jsonl` 每轮有：成绩单原文、patch 全文、干预数、恢复事件
- [ ] `logs/time_ledger.json` / `prior_ledger.json` 有内容且在变化
- [ ] 26+ 测试全绿
- [ ] push 一个"agent 部分完工"提交

## 等外部输入（到了立即换）

- 成员1 的完整 Parquet + 切分 → 成员2 立刻去跑 NISE 基线（环境已就绪 `NISE/.venv`）
- 成员4 的真执行器 → 换掉 FakeExecutor，进入真实挂机

## 明确不做

合成器 fallback · observations 新病 · paper_index · 阈值自动化 · learned_links ·
第 6 份陷阱成绩单 · **web 前端**（赛道二交付物不含前端和视频；
唯一可做的替代品：核心全绿后若有富余，写个 rounds.jsonl → 单页 HTML 的渲染脚本，
约 1 小时，评委可读性 + 答辩素材两用）


---

## 实际验收记录 · 08-27 晚 · 成员3

全部用 `agent/offline.py` 的假模型 + 假执行器跑的 —— 不联网、不花钱，
验证的是**接线**，不是提示词质量。

| 验收项 | 怎么验的 | 结果 |
|---|---|---|
| 状态在轮与轮之间传下去 | 分数一轮比一轮高，且逐轮日志里的成绩单在变 | ✅ |
| 角色炸了不拖垮整场 | `--fail-role-call 2` 让医生第 2 次调用抛异常 | ✅ 那轮作废，后面照跑 |
| 训练失败能恢复 | `--fail-round 3 --fail-round 5` | ✅ 复盘结论由代码合成，没多调一次大模型 |
| 失败的卡被拉黑 | 炸掉那张卡后面不再出现 | ✅ |
| 已生效的卡不重复上 | 判「猜对了」的卡进 applied，不再被选 | ✅ |
| 不涨了自己停 | 分数不动的执行器 + `patience=2` | ✅ 3 轮就停，不烧满 20 轮 |
| 预算耗尽自己停 | `token_budget=20000` | ✅ |
| 逐轮日志合规 | 假设 / 代码全文 / 指标 / 错误恢复 / 干预数 | ✅ 每轮齐 |
| 两个账本在变 | `time_ledger.json` / `prior_ledger.json` | ✅ |
| 测试 | `pytest tests/ -q` | ✅ 57 passed（原 26） |
| 看板能读新日志 | `web/build_report.py logs/offline/rounds.jsonl` | ✅ 5 轮正常渲染 |

一条命令复现全部：

```bash
.venv/bin/python -m agent.cli run --offline --rounds 8 --fail-round 3 --fail-role-call 2
```

### 还没验的（诚实记一笔）

1. **真模型那条路** —— 提示词好不好，假模型验证不了。要真 key 跑 `doctor --all`。
2. **真数据那条路** —— 开发机装不上 lightgbm（缺 libomp），`RealExecutor` 的
   训练与评分段没跑过。落地补丁那一段有测试兜住。
   装它需要：`brew install libomp`。
3. **噪声带的实测值** —— 公式与汇总有测试，还没在真数据上量过一次。

### 顺手修的跨车道小问题（成员2 知会一声）

- `harness/executor.py`：`config_patch` 是 YAML **文本**（见 `schemas.py`），
  原来直接 `.items()`，任何带配置改动的补丁都会当场 `AttributeError`。已改成先解析。
- `requirements.txt`：缺 openai / pandas / pyarrow / scikit-learn / lightgbm，
  照 README 装完跑不起来。已补齐。
- `agent/cli.py`：`make_llm` 改成延迟导入 —— `check` 和 `run --offline`
  不该因为没装某个 SDK 就跑不起来。

### 给成员2 的一个小请求

医生现在会在成绩单里多看到一个 `噪声带` 字段（测过之后才有），
里面写了「小于这个数的差距都是噪声」。字段自带说明，不改提示词也能用，
但在 `doctor.md` 里加一句「先看噪声带再判分组差距」会更稳。

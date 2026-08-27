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

## 成员3（室友）· 按顺序做

### ① 外层循环 `run --rounds N` —— 最大的缺口，自主迭代现在根本不存在

- [ ] 新 cli 子命令，循环调 `run_round`，维护跨轮状态：
  - `parent_result`（只在成功轮更新）
  - `history_brief`（每轮压缩成 1-2 行：轮次/选卡/verdict/实际变化）
  - 已试卡黑名单（verdict = 猜错了 / 没跑起来 的卡；"说不清"不拉黑）
  - TimeLedger / PriorLedger 每轮落盘
  - token 花费 → `budget_left` 三档（<60% 宽裕 / 60-85% 一般 / >85% 紧张）
- [ ] 逃生舱：连续 2 次 no_finding → 升 `FIDELITY_LADDER` 下一档
      （给 `run_round` 加 `fidelity_override` 可选参数即可，不用动调度器）
- [ ] 停止条件：跑满 N 轮或预算耗尽

### ② 三个角色包 try —— 挂机保险丝

- [ ] 医生（loop.py:167）、军师（:180）、复盘官（:229）各包
      `try/except SchemaViolation`（+ 网络异常）→ 记 recoveries → 本轮作废继续
- [ ] `executor.run` 也包一层（协议说返回 ok=False，但真实现难保不 raise）
- [ ] 别吞 `KeyboardInterrupt`
- [ ] 复盘官失败时指标必须已落盘（见 ③ 的赋值顺序）

### ③ RoundLog 存全 —— 交付物 #3 合规

- [ ] 加 `metrics` 字段（`result.health_report` 原文）
- [ ] 加 `patch_files` 字段（`{路径: 完整代码}`）
- [ ] **两个赋值都放在调用复盘官之前**（:219 拿到 result 就赋，:229 才 reflect）

### ④ PriorLedger + 三根断线接上 + 失败记账

- [ ] `PriorLedger`：照 `docs/方法库进度.md` 第五节规格，仿 TimeLedger 写
      （规则算分：猜对了且超噪声带 +0.15 / 目标病改善 +0.05 / 猜错了 −0.10 /
      没跑起来 −0.15，限幅 [0.05, 0.95]）
- [ ] 应用：每轮开始 `card.prior = ledger.value(card.id, card.prior)`（yaml 原文不动）
- [ ] 断线 1：`cards.match(symptom_ids, exclude_ids=黑名单)`（loop.py:177 现在没传）
- [ ] 断线 2：`CostAwareScheduler(tried_cards=黑名单)`（cli 现在传空）
- [ ] 断线 3：`roles.propose(tried_before=已试清单)`（run_round 现在没传——
      军师提示词说"试过的不要重复提"，但它根本不知道试过什么）
- [ ] 执行失败时纯代码合成 `{"verdict": "没跑起来"}` + `ledger.apply(-0.15)`，不调大模型

### ⑤ 六条一行流校验（合计 <20 行）

- [ ] 复盘官（roles.py:198 附近）：目标指标全部 ≤0 → 禁判"猜对了"
- [ ] 复盘官："猜错了" → prior_delta 不许为正
- [ ] 复盘官：promote=true 仅当"猜对了"
- [ ] 军师 schema：expected 数值限 [−0.05, +0.05]
- [ ] 工兵（roles.py:148 附近）：路径含 `..` 直接打回
- [ ] 工兵：config_patch 键只许 `features.` / `model.` / `train.` 前缀

### ⑥ knowledge.py 读「失败信号」

- [ ] `Card` 加 `failure_signals: str = ""`，`from_dict` 读 `d.get("失败信号")`
- [ ] 只喂复盘官：在 `reflect()` 拼 user 消息处追加，**别改 `as_prompt_block`**
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

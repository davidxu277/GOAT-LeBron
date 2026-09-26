# 噪声带实测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 同一份配置、同一份数据子集、只换训练种子跑 N 次，量出 GAUC / nDCG@5 各自的抖动，
写成 `run_session(noise_bands=...)` 已经认的格式；真跑时自动读进来。

**Architecture:**
- runner / 执行器加可选 `sample_seed`：不给就等于 `seed`（老行为），给了就只管「抽哪部分训练集」。
  这样量噪声时数据子集钉死，只有训练随机性在变 —— 跟一场里轮与轮之间的情况一致。
- 新文件 `kuairand_bridge/noise.py`：`measure_noise(...) -> dict` + `bands_from_scores(...)`（纯函数）。
- `python -m kuairand_bridge noise --config ... --fidelity 小份 --seeds 5` 落盘 `<output_dir>/logs/noise_bands.json`。
- `goat_run` 找到这份文件就传给 `run_session`；档位对不上 `run_session` 自己会作废并写进结果表。

**门槛怎么定：** 比的是**两次运行之间的差**，每次各带标准差 σ 的抖动，差的标准差是 √2·σ。
取 2 倍 → 噪声带 = 2·√2·σ ≈ 2.83σ：两次同配置跑，差距约 95% 落在带内。
（旧 AliCCP 工具用 2σ，那是单次数值的带子，拿来卡差值偏松。）

## Global Constraints

- `sample_seed=None` 时一切行为与现在完全一致（runner 与执行器）
- 执行器只在 `sample_seed` 不为 None 时才把它传给 runner —— 测试里的假 runner 没有这个参数
- 输出字段名沿用 `run_session` 已认的：`保真度`、`分指标噪声带`、`单指标噪声带`
- 种子数 < 3 直接拒绝：两个点算不出像样的标准差
- 提交不加 Co-Authored-By；每个 Task 结束全量测试 0 失败

---

### Task 1: `sample_seed` —— 数据子集和训练种子分开

**Files:** Modify `kuairand_bridge/runner.py`、`kuairand_bridge/goat_executor.py`；
Test `kuairand_goat_bridge/tests/test_sample_seed.py`

- [ ] 失败测试：`run_trainer(..., seed=1, sample_seed=0)` 与 `seed=2, sample_seed=0` 抽到的训练行完全相同；
  不给 `sample_seed` 时 `seed` 决定抽样（老行为）；执行器 `sample_seed=0` 会传给 runner，`None` 时不传
- [ ] 实现、全绿、提交

### Task 2: `noise.py` —— 算带子 + 量带子

**Files:** Create `kuairand_bridge/noise.py`；Test `kuairand_goat_bridge/tests/test_noise.py`

- [ ] 失败测试：
  - `bands_from_scores` 对已知数列给出 2·√2·样本标准差；少于 3 个拒绝
  - `measure_noise` 用假 runner：每个训练种子各跑一次、`sample_seed` 全程同一个、
    输出带 `保真度` / `分指标噪声带` / `单指标噪声带` / `逐次分数`；某次跑挂就整体报错（不拿残缺数据算）
- [ ] 实现、全绿、提交

### Task 3: 命令行 + 真跑时读进来

**Files:** Modify `kuairand_bridge/cli.py`、`kuairand_bridge/goat_run.py`；Test 追加到 `test_noise.py`

- [ ] 失败测试：`goat_run` 在 `<output_dir>/logs/noise_bands.json` 存在时把它传给 `run_session`；不存在时传 None
- [ ] 实现、全绿、提交

### Task 4: 真数据量一次

- [ ] 小份、5 个训练种子，看带子数值
- [ ] 跟官方 epsilon 0.002、R11 兜底 0.0005 对比，写进 `docs/待做清单.md` 并划掉 B1
- [ ] 提交推送

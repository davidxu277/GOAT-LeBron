# Baseline 笔记 · 读 NISE 源码得到的结论

> 2026-08-26 · 成员3 · 源码：https://github.com/Hjh233/NISE （分支 `master`）
> 论文：RecSys 2024《Utilizing Non-click Samples via Semi-supervised Learning
> for Conversion Rate Prediction》

这份笔记回答三件事：baseline 长什么样、它身上有哪些洞（= 我们的机会）、
以及**两个必须先问清楚才能动手的问题**。

---

## 零、先看这里：两个待确认的问题

这两条不确认，后面所有实验都可能白做。

### Q1 · 购买 AUC 到底在哪些记录上算？

NISE 的评测代码（`torch-rechub/torch_rechub/trainers/mtl_trainer.py:424`）：

```python
cvr_score = self.evaluate_fns[0](targets[:, 0], predicts[:, 0])
#   targets[:, 0] = cvr_label = 全部曝光记录的 purchase 标签
#   没有筛 click == 1
```

**它是在全部曝光上算的。** 而我们 CLAUDE.md 原来写的是「仅在 click=1 上算」，
还把「在全部曝光上算购买 AUC」列成了危险信号。

两种口径都有文献依据，但算出来的数完全不是一回事：

| 口径 | 分母 | 正样本比例 | 典型量级 |
|---|---|---|---|
| 全部曝光 | 全部记录 | 极低（约 0.02%） | 偏高，且和点击 AUC 高度相关 |
| 仅 click=1 | 点击过的记录 | 约 2%–5% | 偏低，是真正的"点了会不会买" |

排名按 **delta vs 官方 baseline** 算，所以口径必须跟**主办方的评测脚本**一致
—— 不一定是 NISE 的口径，也不一定是我们红线文件里写的那个。

**要做的事**：拿到主办方的评测脚本或问清楚。在此之前，
代码里评估范围必须是**配置项**，两种口径都算、都记进日志。

### Q2 · 官方 baseline 是 NISE 本身，还是它的对照组？

这个问题决定了下面 7 张卡是**机会还是陷阱**。

NISE 论文的对照组是 ESMM / MMoE / ESCM²-IPS / ESCM²-DR / DCMT。
论文的核心主张就是 **NISE 打赢了这五个**。

- 如果官方 baseline = **NISE** → 照搬那五个方法，delta **大概率是负的**。
  它们只能当零件用（借思路），不能当替换用。
- 如果官方 baseline = **ESMM**（经典设置）→ NISE 本身就是一张大卡。

**在没确认之前，不要把这五个方法当成"预计能涨分"的方案。**

⚠️ 顺带：`knowledge/cards/esmm.yaml` 上写着 `靠不靠谱: 0.85`、
`预计能提多少: {购买AUC: 0.01}`。如果 baseline 是 NISE，这两个数是**反的**。
成员2 复核时注意。

---

## 一、baseline 长什么样

共享 embedding → 两个塔（点击塔、购买塔）→ `ctcvr = ctr_pred × cvr_pred`。
ctcvr 是**乘出来的**，不是单独学的。骨架三选一：MLP / DeepFM / DCNv2。

超参全部硬编在 `baselines/mlp_model.py` 顶部，可直接照抄：

| 项 | 值 |
|---|---|
| epoch | 10 |
| learning_rate | 1e-3 |
| weight_decay | 1e-5 |
| batch_size | 2048 |
| embed_dim | 16（每个稀疏字段） |
| 塔结构 | `[160, 80]`（cvr / ctr / imputation 都一样） |
| MMoE 专家 / 塔 | `[80]` / `[40]`，2 个专家 |
| 早停 | patience=3，盯 **task 0 = 购买 AUC** |
| 标签顺序 | `['cvr_label', 'ctr_label', 'ctcvr_label']`（写死，不许改） |

NISE 自己的招（`mtl_trainer.py:269`）：

```python
loss_clicked_cvr   = BCE(cvr_pred, cvr_label, weight=click)        # 点过的：真标签
loss_unclicked_cvr = BCE(cvr_pred, cvr_pred,  weight=1 - click)    # 没点的：自己的预测当标签
loss_cvr = loss_clicked_cvr + loss_unclicked_cvr

w = min(loss_ctr.item() / loss_cvr.item(), 50)     # 自适应任务权重，上限 β=50
loss = loss_ctr + w * loss_cvr
```

两个动作：**把没点击的样本也拉进购买塔训练**（自训练/熵最小化），
外加**按损失比例自动平衡两个任务**。

---

## 二、baseline 身上的三个洞

### 洞 1 · 直接扔掉了 8 个字段 ★ 最大的机会

`baselines/mlp_model.py:69`：

```python
dense_cols = ['D109_14','D110_14','D127_14','D150_14','D508','D509','D702','D853']
sparse_cols = [col for col in col_names if col not in dense_cols and col not in label_cols]
used_cols = sparse_cols   # ESMM only for sparse features in origin paper
```

扔掉的正好是：

| 字段 | 是什么 |
|---|---|
| 109_14 / 110_14 / 127_14 / 150_14 | **全部 4 个用户历史行为字段**（多值带权重） |
| 508 / 509 / 702 / 853 | **全部 4 个用户 × 商品交叉特征** |

23 个特征字段里，**8 个没进模型**。

我们词表里的「历史行为没用上」和「特征没组合起来」在 baseline 上**字面成立**，
不是我们编出来的病。而且这是纯特征工程，不动模型结构，
风险最低、代码最简单、**跟 Q2 的不确定性完全无关** ——
无论官方 baseline 是 NISE 还是 ESMM，把扔掉的字段加回来都是正向的。

**结论：第一轮实验就该打这里，不要碰那五个多任务方法。**

### 洞 2 · 没有验证集，拿 test 做早停

`baselines/mlp_model.py:92` 和 `:135`：

```python
dg.generate_dataloader(x_val=x_train, y_val=y_train, x_test=x_test, ...)   # x_val = x_train
...
total_log = mtl_trainer.fit(train_dataloader, test_dataloader, ...)        # 早停盯 test
```

`x_val` 就是 `x_train`，而 `fit()` 的第二个参数（名字叫 `val_dataloader`）
传进去的是 **test**。每个 epoch 在 test 上评估、按 test 分数早停、存 test 最优权重。

**这意味着两件事**：

1. 我们的 R3（锁定集）不是多此一举，是在补这个洞。
2. **不能照抄它的训练脚本**。必须自己从训练集切出开发集和锁定集，
   早停只看开发集。照抄 = 我们的分数虚高，测试集上必掉。

### 洞 3 · 参数全硬编，没有配置文件

`epoch = 10` / `learning_rate = 1e-3` / `cvr_params = {"dims": [160, 80]}`
全部写死在模块顶层。这违反我们的 R7。

对我们是**好消息**：把这些抽成配置，Agent 才有旋钮可拧。
这件事本来就在成员2 的接口工作范围内。

---

## 三、顺手捡到的东西

### 药方卡的现成弹药

repo 里已经实现好的方法，每个都有论文出处，「为什么管用」不用瞎编：

| 方法 | 文件 | 对应我们的病 |
|---|---|---|
| ESMM | `models/multi_task/esmm.py` | 转化样本偏差 |
| MMoE | `models/multi_task/mmoe.py` | 两个任务打架 |
| ESCM²-IPS | `models/multi_task/ips.py` | 转化样本偏差 |
| ESCM²-DR | `models/multi_task/dr.py` | 转化样本偏差、结果不稳 |
| DCMT | `models/multi_task/dcmt.py` | 转化样本偏差 |
| NISE | `models/multi_task/ucvrlc.py` | 转化样本偏差、两个任务打架 |
| DTP / DWA | `mtl_trainer.py:319` / `:305` | 两个任务打架 |

骨架已写到 `knowledge/cards/`，成员2 补「怎么实现」那一栏。
**但先读第零节 Q2** —— 这些卡的预期增益方向还没确定。

### 成绩单不用另造

训练日志本来就输出 `auc / ks / log_loss` × `cvr / ctr / ctcvr` 共 9 个数
（`mlp_model.py:145` 的 `column_names`）。我们的健康报告可以直接从这里取。

特别是 **LogLoss 是现成的**。词表里「把握不准」原本要自己算校准误差，
现在可以直接用 LogLoss，少写一段代码、少一个自己拍的阈值。
**KS 也是现成的**（`mtl_trainer.py:19`），可以当第二个稳定性信号。

### 论文出发点印证了我们的核心病

NISE 整篇论文要解决的就是：只有 click=1 的样本能训购买塔
→ 样本选择偏差 + 数据极度稀疏。这正是我们词表里标了 `core: true` 的
「转化样本偏差」。把它当头号病是对的。

---

## 四、还没解决的问题

1. **README 没给任何 AUC 数字。** baseline 的绝对分只能自己跑出来。
   在跑出来之前，我们所有"提升 0.00X"的判断都没有参照系。
2. **数据格式对不上。** repo 假定输入是预处理好的
   `ali_ccp_train.csv` / `ali_ccp_test.csv`（torch-rechub 格式，
   列名形如 `101` / `D109_14`），**不是原始的两张表**。
   成员1 做预处理时要对齐这个格式，否则跑不了 baseline，也就没有基准。
3. **字段 `129` 归属存疑。** baseline 把它当**商品**字段
   （`item_cols = ['129','205','206','207','210','216']`，`mlp_model.py:75`），
   我们 CLAUDE.md 把它列在用户属性组。需要核对官方 schema。

---

## 五、对我们的行动影响

| 原计划 | 改成 |
|---|---|
| 评估口径写死「购买 AUC 只在 click=1 上算」 | 改成配置项，两种都算都记；等主办方确认 |
| 第一轮试 ESMM 类的多任务方法 | **改成先把扔掉的 8 个字段加回来**（无 Q2 风险） |
| 「把握不准」自己算校准误差 | 直接用现成的 LogLoss |
| 照抄 baseline 训练脚本 | 不能抄 —— 它拿 test 早停，必须自己切开发集/锁定集 |

# 模型零件的损失函数挂载点 —— 设计

日期：2026-09-11 · 状态：用户已批准方案 A

## 问题

`harness/deep.py` 的 `train_deep` 里，损失的优先级是：

    任务损失 task_loss  >  零件的 op.loss(out, click, conv)  >  双塔默认

KuaiRand 的 goat_trainer 总是传 task_loss（逐条 BCE），所以**模型零件自己写的
loss 被静默忽略**，不报错、不提示。这把两张卡锁死了：

- pairwise_loss（用户内配对排序，预计提升最大）—— 要换损失
- 时间衰减加权 —— 要给每条样本的损失乘权重

而且旧签名只给 `click` / `conv` 两个标签张量，就算被调用，也拿不到
配对需要的 user_id、衰减需要的 date。

## 方案 A

**一个挂载点、一个签名：** ModelOp 可以实现

    loss(self, out, batch, torch) -> 标量张量

`batch` 是这一批的 DataFrame（`train.iloc[idx]`），带全部列：
user_id、date、label（KuaiRand）或 click、conversion（AliCCP），以及所有特征列。
跟 task_loss 同一个签名。

**优先级改成"谁明确写了听谁的"：**

    零件的 op.loss  >  任务损失 task_loss  >  双塔默认

零件写了 loss 就是明确要换打分规则，任务默认不该盖掉它。

**看得见：** train_deep 的训练记录新增 `损失来源`：
`"模型零件 <类名>.loss"` / `"任务默认"` / `"双塔默认"`。
runner 已经把整份训练记录放进成绩单「训练诊断」，不用另改。

**兼容：** 仓库里没有任何零件实现过 loss（已 grep 确认），改签名不破坏现有零件。
`_default_loss` 的 docstring 里"自己实现 loss(out, click, conv)"的说法同步改掉。

## 文档与卡片

- `modules/base.py` 的 ModelOp 协议加上 `loss` 的说明（可选方法、签名、batch 里有什么、
  优先级）—— 工兵要知道有这个入口。
- pairwise_loss 卡、时间衰减卡：删掉"目前不生效 / 目前不能"的前提，
  「怎么实现」改成"新写一个模型零件，继承 modules/models/mlp.py 的 MLPModel，
  只加一个 loss(out, batch, torch) 方法"。

## 不做的事

- 不做 TrainOp 上可叠加的损失钩子（方案 B）—— 要先定"几个损失怎么叠"的规矩，现在用不上。
- 不在流水线里内置 pairwise / 时间衰减（方案 C）—— 代码该由 Agent 自己写。

## 验收

1. 传了 task_loss、零件也写了 loss 时，调用的是零件的 loss；`损失来源` 写明零件类名。
2. 零件的 loss 拿到的 batch 是 DataFrame，含 user_id、date、label 列，行数等于这一批的行数。
3. 零件没写 loss 时行为与改动前一致；`损失来源` 为 `任务默认`。
4. 全量测试不退步；守卫测试（配置样例真加载、不引用不存在的文件）照常通过。

# KuaiRand-Pure × GOAT-LeBron 接口包

第一次接入 GOAT-LeBron，请先阅读：[GOAT接入与使用说明.md](GOAT接入与使用说明.md)。

这个文件夹可以直接复制进 GOAT-LeBron 仓库。它负责：

1. 使用官方固定日期切分读取 KuaiRand-Pure；
2. 给同学的模型提供稳定的 `train / valid / test` 接口；
3. 锁住 Test 标签，避免 Agent 用测试集调参；
4. 接收 CSV、NPY 或 NPZ 预测；
5. 转换并严格检查官方 `row_id,user_id,video_id,score` 格式；
6. Validation 调用官方 `evaluate.py` 计算 GAUC、nDCG@5、Primary；
7. Test 只检查并生成提交文件，不返回分数。

`official_starter_kit/` 是官方代码的原样副本。不要修改其中的 `evaluate.py`。
原始数据没有复制进本包；每位同学通过 `--data-dir` 指向自己的
`KuaiRand-Pure/data` 即可。

## 1. 安装

在本文件夹执行：

```bash
cd /Users/huyaofu/Documents/GitHub/NISE/kuairand_goat_bridge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## 2. 检查数据与官方切分

```bash
python -m kuairand_bridge preflight \
  --data-dir /Users/huyaofu/Documents/GitHub/kuairand-starter-kit/raw_data/KuaiRand-Pure/data
```

正确行数应为：Train 1,141,112；Validation 124,909；Test 170,588。

## 3. 同学的模型如何读取数据

```python
from kuairand_bridge import load_dataset

dataset = load_dataset("/path/to/KuaiRand-Pure/data")
train = dataset.train
valid = dataset.valid
test = dataset.test

print(len(train), train.user_ids, train.video_ids, train.labels)
print(len(valid), valid.labels)
print(len(test), test.user_ids, test.video_ids)
```

`test.labels` 会直接报错，这是有意的数据泄漏保护。
`split.records()` 会逐条提供 `row_id/date/user_id/video_id/author_id/tab/duration_ms`；
Train/Validation 还会提供 `long_view`。

## 4. 同学的模型如何接入训练

训练文件只需实现两个函数：

```python
def fit(train, valid, seed=0):
    return trained_model

def predict(trained_model, split):
    return scores  # 一维数组，长度必须等于 split 行数
```

然后运行：

```bash
python -m kuairand_bridge run-trainer \
  --data-dir /path/to/KuaiRand-Pure/data \
  --trainer /path/to/teammate_trainer.py \
  --seed 0 \
  --output-dir output/my_model
```

只有最终模型确定后才加 `--make-test`。开发迭代时不要加。

可以先用包内的快速示例验证整条接线：

```bash
python -m kuairand_bridge run-trainer \
  --data-dir /path/to/KuaiRand-Pure/data \
  --trainer examples/popularity_trainer.py \
  --seed 0 \
  --output-dir output/popularity_smoke \
  --make-test
```

## 5. 接收已经训练好的预测结果并评分

支持：

- 官方四列 CSV：`row_id,user_id,video_id,score`；
- 只有 `score` 一列的 CSV；
- 一维 `.npy`；
- `.npz` 中名为 `score/scores/prediction/predictions` 的数组。

Validation 评分：

```bash
python -m kuairand_bridge evaluate \
  --data-dir /path/to/KuaiRand-Pure/data \
  --split valid \
  --predictions /path/to/classmate_valid_scores.npy \
  --output-dir output/classmate_model
```

会生成：

- `valid_submission.csv`：官方四列格式；
- `valid_metrics.json`：官方 GAUC、nDCG@5、Primary。

最终 Test 检查：

```bash
python -m kuairand_bridge evaluate \
  --data-dir /path/to/KuaiRand-Pure/data \
  --split test \
  --predictions /path/to/classmate_test_scores.npy \
  --output-dir output/final
```

会生成经过官方对齐检查的 `test_submission.csv`，但不会向 Agent 返回 Test 分数。

## 6. GOAT-LeBron 如何调用

GOAT 的训练/模型代码可以直接使用第 3、4 节接口。若 GOAT 已经自行训练并落盘预测，
其执行器只需调用：

```python
from kuairand_bridge import load_dataset
from kuairand_bridge.goat_adapter import KuaiRandOfficialEvaluator

dataset = load_dataset("/path/to/KuaiRand-Pure/data")
evaluator = KuaiRandOfficialEvaluator(dataset, "output/goat_run")
health_report = evaluator.score("/path/to/goat_valid_scores.npy")
```

注意：GOAT-LeBron 当前仓库的 AliCCP `RealExecutor` 不能直接用于新题，因为它写死了
CTR/CVR 双目标与 AUC。本包提供的是 KuaiRand 新题的替代边界，不应把旧成绩字段硬映射过去。

## 7. 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖 Test 标签锁定、预测格式转换及 NaN 拒绝。

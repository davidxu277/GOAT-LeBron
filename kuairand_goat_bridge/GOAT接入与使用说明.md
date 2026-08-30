# GOAT-LeBron 接入 KuaiRand Bridge 使用说明

> 最新推荐入口：编辑`configs/kuairand_task.yaml`，先运行`goat-run --dry-run`，
> 再运行`goat-run`。该入口会自动创建KuaiRand执行器、接入GOAT多轮循环、
> 按官方规则收敛并整理最终输出。完整命令见根目录`README.md`开头。

这份说明写给刚拿到 `kuairand_goat_bridge` 文件夹的合作者。目标是把自己电脑上的
GOAT-LeBron 或其他训练程序连接到官方 KuaiRand-Pure 数据和评分脚本。

## 一、开始前需要准备什么

电脑上需要有：

1. Python 3.9 或更高版本；
2. 一份 GOAT-LeBron 仓库；
3. `kuairand_goat_bridge` 文件夹；
4. KuaiRand-Pure 原始数据。

原始数据不放进 Git 仓库。数据目录中至少应有：

```text
KuaiRand-Pure/data/
├── log_standard_4_08_to_4_21_pure.csv
├── log_standard_4_22_to_5_08_pure.csv
└── video_features_basic_pure.csv
```

后面的命令用下面两个示例路径表示：

```text
/path/to/GOAT-LeBron
/path/to/KuaiRand-Pure/data
```

请把它们替换成自己电脑上的真实绝对路径。

## 二、把 Bridge 放到 GOAT-LeBron 里面

推荐目录结构：

```text
GOAT-LeBron/
├── agent/
├── harness/
├── modules/
├── kuairand_goat_bridge/
│   ├── src/kuairand_bridge/
│   ├── official_starter_kit/
│   ├── examples/
│   ├── output/
│   └── README.md
└── ...
```

也可以把 Bridge 放在 GOAT-LeBron 旁边。只要安装时进入正确的 Bridge 目录即可。

## 三、建立运行环境并安装 Bridge

打开终端后执行：

```bash
cd /path/to/GOAT-LeBron
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ./kuairand_goat_bridge
```

如果 Bridge 放在 GOAT-LeBron 外面，最后一条改成它的绝对路径：

```bash
python -m pip install -e /absolute/path/to/kuairand_goat_bridge
```

Apple 芯片不需要 CUDA。数据读取、官方评分、Popularity 和 NumPy FM 都能用 CPU。
只有同学的训练代码主动使用 CUDA 深度模型时，才需要 NVIDIA GPU。

## 四、先检查数据是否正确

```bash
python -m kuairand_bridge preflight \
  --data-dir /path/to/KuaiRand-Pure/data
```

正确结果应该包含：

```text
train: 1,141,112
valid: 124,909
test: 170,588
official_row_counts_match: true
test_labels_exposed: false
```

如果数量不一致，先检查数据版本和 `--data-dir`，不要继续训练。

## 五、GOAT 或模型程序如何读取数据

在 GOAT 的 Python 程序中：

```python
from kuairand_bridge import load_dataset

dataset = load_dataset("/path/to/KuaiRand-Pure/data")

train = dataset.train
valid = dataset.valid
test = dataset.test

print(len(train))
print(train.user_ids)
print(train.video_ids)
print(train.labels)
```

每个 split 可以提供：

```text
row_id
date
user_id
video_id
author_id
tab
duration_ms
long_view（只在 Train 和 Validation 中提供）
```

如果训练代码想逐条读取：

```python
for row in dataset.train.records():
    print(row["user_id"], row["video_id"], row["long_view"])
```

`dataset.test.labels` 会主动报错。这是防止 GOAT 使用 Test 标签调参，不是程序故障。

## 六、最推荐的训练连接方式

为 GOAT 的模型写一个训练入口，例如：

```text
GOAT-LeBron/teammate_trainer.py
```

这个文件只需要实现两个函数：

```python
def fit(train, valid, seed=0):
    # 只用 train 拟合词表、统计量和模型。
    # valid 可以用于早停和选择最佳 epoch，但不能拿它拟合统计特征。
    trained_model = ...
    return trained_model


def predict(trained_model, split):
    # 返回一维分数，顺序必须与 split 原始顺序完全一致。
    # 分数不用是概率，只要数值越大代表越应该排在前面。
    scores = ...
    return scores
```

重要要求：

- `fit()` 返回的模型必须保存从 Train 学到的词表、分桶边界等状态；
- `predict()` 不得重新拟合 Validation 或 Test；
- 输出必须恰好一条曝光对应一个分数；
- 不要重新排序数据；
- 分数中不能有 NaN 或 Inf。

## 七、让 Bridge 负责训练、预测和评分

开发阶段只跑 Validation：

```bash
cd /path/to/GOAT-LeBron
source .venv/bin/activate

python -m kuairand_bridge run-trainer \
  --data-dir /path/to/KuaiRand-Pure/data \
  --trainer /path/to/GOAT-LeBron/teammate_trainer.py \
  --seed 0 \
  --output-dir /path/to/GOAT-LeBron/runs/kuairand/run_001
```

模型最终确定后，才生成 Test 提交：

```bash
python -m kuairand_bridge run-trainer \
  --data-dir /path/to/KuaiRand-Pure/data \
  --trainer /path/to/GOAT-LeBron/teammate_trainer.py \
  --seed 0 \
  --output-dir /path/to/GOAT-LeBron/runs/kuairand/final \
  --make-test
```

`--output-dir` 决定结果保存在哪里。强烈建议每次实验使用不同目录，例如
`run_001`、`run_002`，避免覆盖。

## 八、如果 GOAT 已经自己训练完了

如果 GOAT 已经输出预测，Bridge 不需要再次训练，只负责接收和评分。

支持以下文件：

```text
valid_scores.npy
valid_scores.npz
valid_scores.csv
valid_submission.csv
```

最简单的 CSV 可以只有一列：

```csv
score
0.123
-0.472
0.806
```

Validation 评分命令：

```bash
python -m kuairand_bridge evaluate \
  --data-dir /path/to/KuaiRand-Pure/data \
  --split valid \
  --predictions /path/to/GOAT-LeBron/runs/run_001/valid_scores.npy \
  --output-dir /path/to/GOAT-LeBron/runs/kuairand/run_001
```

最终 Test 文件检查命令：

```bash
python -m kuairand_bridge evaluate \
  --data-dir /path/to/KuaiRand-Pure/data \
  --split test \
  --predictions /path/to/GOAT-LeBron/runs/final/test_scores.npy \
  --output-dir /path/to/GOAT-LeBron/runs/kuairand/final
```

## 九、在 GOAT 代码内部调用官方评分

如果 GOAT 希望直接得到一份成绩单：

```python
from kuairand_bridge import load_dataset
from kuairand_bridge.goat_adapter import KuaiRandOfficialEvaluator

dataset = load_dataset("/path/to/KuaiRand-Pure/data")

evaluator = KuaiRandOfficialEvaluator(
    dataset=dataset,
    output_dir="/path/to/GOAT-LeBron/runs/kuairand/run_001",
)

health_report = evaluator.score(
    "/path/to/GOAT-LeBron/runs/run_001/valid_scores.npy"
)

print(health_report)
```

返回内容包括：

```text
GAUC
nDCG@5
主分 Primary
用户数
曝光行数
官方结果文件位置
```

GOAT 原来的 AliCCP `RealExecutor` 计算的是 CTR/CVR AUC，不能直接用于新题。
KuaiRand 实验必须调用上面的 `KuaiRandOfficialEvaluator`，不能把新指标伪装成旧字段。

## 十、Output 到哪里找

结果位于运行命令指定的 `--output-dir`。

一次 Validation 实验通常包含：

```text
run_001/
├── valid_scores.npy
├── valid_submission.csv
└── valid_metrics.json
```

各文件含义：

- `valid_scores.npy`：模型原始预测；
- `valid_submission.csv`：Bridge 转换后的官方四列格式；
- `valid_metrics.json`：官方 Validation 分数。

`valid_metrics.json` 示例：

```json
{
  "status": "scored",
  "split": "valid",
  "metrics": {
    "GAUC": 0.6671,
    "nDCG@5": 0.5358,
    "primary": 0.6015,
    "users": 22377,
    "rows": 124909
  }
}
```

最终运行目录通常包含：

```text
final/
├── valid_scores.npy
├── valid_submission.csv
├── valid_metrics.json
├── test_scores.npy
├── test_submission.csv
└── test_metrics.json
```

真正交给官方的是：

```text
test_submission.csv
```

`test_metrics.json` 只记录 Test 文件是否通过格式和对齐检查，不包含 Test 分数。

如果运行命令没有写 `--output-dir`，默认输出到执行命令时所在目录下的：

```text
output/
```

为了避免找不到文件，建议始终填写绝对路径的 `--output-dir`。

## 十一、快速确认整条连接是否正常

Bridge 自带一个 CPU Popularity 示例，不需要 CUDA：

```bash
cd /path/to/GOAT-LeBron/kuairand_goat_bridge
source /path/to/GOAT-LeBron/.venv/bin/activate

python -m kuairand_bridge run-trainer \
  --data-dir /path/to/KuaiRand-Pure/data \
  --trainer examples/popularity_trainer.py \
  --seed 0 \
  --output-dir output/connection_test \
  --make-test
```

预期现象：

1. Validation 约有 124,909 条预测；
2. Validation Primary 约为 0.5807；
3. Test 提交约有 170,588 行；
4. Test 状态显示 `checked`；
5. `output/connection_test/` 中出现上述结果文件。

## 十二、常见问题

### `No module named kuairand_bridge`

说明当前虚拟环境还没有安装 Bridge。重新激活环境并执行：

```bash
python -m pip install -e /absolute/path/to/kuairand_goat_bridge
```

### 找不到数据文件

`--data-dir` 必须指向最后一级 `data` 文件夹，而不是只指向 `KuaiRand-Pure`。

### 预测行数不一致

模型可能漏掉、过滤或重新排列了曝光。预测时必须保留官方 split 的原始行序。

### user_id/video_id 对齐失败

不要用 `(user_id, video_id)` 作为唯一键，因为同一组合可以重复出现。必须以原始
`row_id` 顺序输出。

### Test 为什么没有分数

这是正确行为。开发阶段只能根据 Validation 选择模型，Test 只用于最终提交。

### Mac 没有 CUDA 能否运行

Bridge、官方评分和 CPU 示例都能运行。只有训练器本身写死 CUDA 时才需要改成
CPU/MPS，或者把训练交给带 NVIDIA GPU 的同学。

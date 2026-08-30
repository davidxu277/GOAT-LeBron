# GOAT-LeBron × KuaiRand Bridge 完整流程使用教程

这份教程说明如何用一个任务配置文件和一条命令，启动KuaiRand-Pure数据读取、
GOAT多轮实验、官方Validation评分、收敛判断和最终Test提交生成。

## 1. 运行前需要准备什么

需要以下内容：

```text
GOAT-LeBron/
├── agent/
├── knowledge/
├── kuairand_goat_bridge/
└── KUAIRAND完整流程使用教程.md

KuaiRand-Pure/
└── data/
    ├── log_standard_4_08_to_4_21_pure.csv
    ├── log_standard_4_22_to_5_08_pure.csv
    └── video_features_basic_pure.csv
```

KuaiRand数据不需要复制进GOAT仓库，只需在配置中填写`data/`的绝对路径。

## 2. 第一次安装

打开终端，进入GOAT-LeBron：

```bash
cd /Users/huyaofu/Documents/GitHub/GOAT-LeBron
```

在Bridge内部建立独立环境：

```bash
python3 -m venv kuairand_goat_bridge/.venv
```

安装Bridge以及GOAT调用LLM所需的依赖：

```bash
kuairand_goat_bridge/.venv/bin/python -m pip install --upgrade pip
kuairand_goat_bridge/.venv/bin/python -m pip install -e 'kuairand_goat_bridge[goat]'
```

这套基础环境可以运行数据处理、官方评分、CPU参考Trainer和GOAT大脑。
如果同学的真实Trainer依赖PyTorch、LightGBM或其他库，还需要按照该Trainer自己的
`requirements.txt`安装。

## 3. 准备任务配置

### 本机已有配置

当前电脑可以使用：

```text
kuairand_goat_bridge/configs/local_task.yaml
```

内容包括数据、Trainer、Output路径和官方运行参数。

### 合作者第一次使用

复制模板：

```bash
cp kuairand_goat_bridge/configs/kuairand_task.yaml \
   kuairand_goat_bridge/configs/my_task.yaml
```

打开`my_task.yaml`，至少修改下面三项：

```yaml
data_dir: /absolute/path/to/KuaiRand-Pure/data
trainer: /absolute/path/to/teammate_trainer.py
output_dir: /absolute/path/to/output/full_run
```

完整配置示例：

```yaml
data_dir: /absolute/path/to/KuaiRand-Pure/data
trainer: examples/tunable_popularity_trainer.py
output_dir: output/full_run

seed: 0
max_rounds: 50
epsilon: 0.002
patience: 3
max_wall_seconds: 21600
token_budget: 2000000
generate_test_after_convergence: true
```

正式规则说明：

- `max_rounds`不能超过50；
- `epsilon`固定为0.002；
- `patience`固定为3；
- `max_wall_seconds`不能超过21600秒，也就是6小时；
- 开发阶段只使用Train和Validation；
- Test只在最终生成提交时使用，并且不向Agent返回标签或成绩。

## 4. 先进行零成本预检

本机运行：

```bash
cd /Users/huyaofu/Documents/GitHub/GOAT-LeBron

kuairand_goat_bridge/.venv/bin/python -m kuairand_bridge goat-run \
  --config kuairand_goat_bridge/configs/local_task.yaml \
  --dry-run
```

合作者将`local_task.yaml`换成自己的`my_task.yaml`。

正确结果应该包含：

```json
{
  "status": "ready",
  "message": "路径、官方参数和GOAT目录检查通过；未调用LLM或训练"
}
```

`--dry-run`不会调用LLM、不会训练，也不会消耗Token。它会检查：

- 数据路径是否存在；
- Trainer是否存在；
- Bridge是否位于GOAT仓库中；
- 轮数、收敛参数和时间上限是否合法；
- Output路径如何解析。

预检不通过时不要正式启动。

## 5. 先用CPU参考Trainer测试接线

任务配置中的Trainer填写：

```yaml
trainer: examples/tunable_popularity_trainer.py
```

如果只想测试一次训练和评分，不调用GOAT多轮Agent：

```bash
kuairand_goat_bridge/.venv/bin/python -m kuairand_bridge run-trainer \
  --data-dir /Users/huyaofu/Documents/GitHub/kuairand-starter-kit/raw_data/KuaiRand-Pure/data \
  --trainer /Users/huyaofu/Documents/GitHub/GOAT-LeBron/kuairand_goat_bridge/examples/tunable_popularity_trainer.py \
  --seed 0 \
  --output-dir /Users/huyaofu/Documents/GitHub/GOAT-LeBron/kuairand_goat_bridge/output/connection_test \
  --make-test
```

参考结果的Validation Primary约为0.58。它的目的只是证明数据、Trainer、评分和
提交格式已经连接，不代表最终参赛模型。

## 6. 配置LLM凭据

GOAT需要调用大语言模型完成诊断、提出方案、生成修改和复盘。

使用DeepSeek时：

```bash
export AGENT_PROVIDER=deepseek
export DEEPSEEK_API_KEY='你的API密钥'
```

不要把API密钥写进YAML、Python文件或Git仓库。

关闭终端后环境变量通常会失效，下次正式运行前需要重新设置。

## 7. 正式启动完整任务

本机启动命令：

```bash
cd /Users/huyaofu/Documents/GitHub/GOAT-LeBron

kuairand_goat_bridge/.venv/bin/python -m kuairand_bridge goat-run \
  --config kuairand_goat_bridge/configs/local_task.yaml
```

合作者使用：

```bash
kuairand_goat_bridge/.venv/bin/python -m kuairand_bridge goat-run \
  --config kuairand_goat_bridge/configs/my_task.yaml
```

程序会自动执行：

```text
读取任务配置
→ 检查官方数据
→ 运行第0轮基线
→ GOAT读取成绩单
→ 诊断值得改进的问题
→ 提出并选择方案
→ 产生配置或代码patch
→ Trainer应用累计patch
→ 在Train上训练
→ 为Validation生成score
→ 官方evaluate.py计算GAUC和nDCG@5
→ 结果返回GOAT复盘
→ 继续下一轮
→ 达到官方收敛条件
→ 选择Validation最佳轮
→ 恢复最佳轮累计修改
→ 生成并检查test_submission.csv
→ 写出final_summary.json
```

正式运行过程中尽量不要手动修改代码、配置或挑选结果。若必须干预，应在交付材料中
记录干预原因和发生轮次。

## 8. 同学的真实Trainer需要实现什么

Trainer文件需要实现：

```python
def fit(train, valid, seed=0):
    return trained_model


def predict(trained_model, split):
    return scores


def apply_agent_patch(patch, output_dir):
    # 把GOAT本轮配置或代码修改应用到Trainer自己的受控实验环境。
    ...
```

要求：

- `fit()`中的词表、统计量和分桶边界只能由Train拟合；
- `predict()`返回一维score，长度与split曝光行数完全一致；
- score越大代表该视频应该排得越靠前；
- 不得输出NaN或Inf；
- 不得打乱split的原始顺序；
- `apply_agent_patch()`不得静默忽略修改；
- 不支持的修改必须抛出清楚的异常；
- Test标签不会提供给Trainer。

Bridge自带的`tunable_popularity_trainer.py`只支持配置实验。Agent若要新建复杂模型、
排序损失或历史序列模块，需要同学的真实Trainer支持这些代码插件。

## 9. Output到哪里找

结果位于任务配置的`output_dir`。本机默认是：

```text
/Users/huyaofu/Documents/GitHub/GOAT-LeBron/kuairand_goat_bridge/output/full_run/
```

运行后通常包含：

```text
output/full_run/
├── logs/
│   ├── rounds.jsonl
│   ├── session_summary.json
│   ├── best_report.json
│   ├── narrative.md
│   └── snapshots/
├── rounds/
│   ├── round_001/
│   │   ├── agent_patch.json
│   │   ├── effective_config.yaml
│   │   ├── valid_scores.npy
│   │   ├── valid_submission.csv
│   │   └── valid_metrics.json
│   ├── round_002/
│   └── final/
│       ├── valid_scores.npy
│       ├── valid_metrics.json
│       ├── test_scores.npy
│       ├── test_submission.csv
│       └── test_metrics.json
└── final_summary.json
```

重点文件：

| 文件 | 用途 |
|---|---|
| `logs/rounds.jsonl` | 每轮假设、patch、指标、错误和恢复 |
| `logs/narrative.md` | 整场实验故事线 |
| `rounds/round_xxx/valid_metrics.json` | 某轮官方Validation分数 |
| `final_summary.json` | 最佳轮、停止原因、Token、时间、干预、提交路径 |
| `rounds/final/test_submission.csv` | 最终官方提交文件 |

正式指标始终是：

```text
GAUC
nDCG@5
Primary = (GAUC + nDCG@5) / 2
```

成绩单中的`点击分=GAUC`和`购买分=nDCG@5`只是让旧GOAT读取两个指标槽位的
兼容别名，不表示KuaiRand存在CTR购买任务。

## 10. 单独评估同学已经生成的预测

如果同学已经输出`valid_scores.npy`：

```bash
kuairand_goat_bridge/.venv/bin/python -m kuairand_bridge evaluate \
  --data-dir /absolute/path/to/KuaiRand-Pure/data \
  --split valid \
  --predictions /absolute/path/to/valid_scores.npy \
  --output-dir /absolute/path/to/output/evaluation
```

会生成：

```text
valid_submission.csv
valid_metrics.json
```

最终Test检查：

```bash
kuairand_goat_bridge/.venv/bin/python -m kuairand_bridge evaluate \
  --data-dir /absolute/path/to/KuaiRand-Pure/data \
  --split test \
  --predictions /absolute/path/to/test_scores.npy \
  --output-dir /absolute/path/to/output/final
```

Test只返回`checked`状态，不返回分数。

## 11. 运行自动测试

```bash
cd /Users/huyaofu/Documents/GitHub/GOAT-LeBron

kuairand_goat_bridge/.venv/bin/python -m unittest discover \
  -s kuairand_goat_bridge/tests -v
```

测试覆盖：

- Test标签隔离；
- 行数和提交格式；
- NaN/Inf拒绝；
- GOAT Executor协议；
- 错误转为可恢复结果；
- 官方轮数和收敛参数；
- 配置路径解析。

## 12. 常见问题

### `No module named kuairand_bridge`

重新安装Bridge：

```bash
kuairand_goat_bridge/.venv/bin/python -m pip install -e 'kuairand_goat_bridge[goat]'
```

### `DEEPSEEK_API_KEY`不存在

重新设置：

```bash
export AGENT_PROVIDER=deepseek
export DEEPSEEK_API_KEY='你的API密钥'
```

### 数据路径错误

`data_dir`必须指向最后一级`KuaiRand-Pure/data`，而不是只指向
`KuaiRand-Pure`。

### Trainer返回`unsupported=True`

说明GOAT提出了一个Trainer当前无法应用的修改。查看该轮目录中的：

```text
agent_patch.json
error.json
```

然后由Trainer负责人补充相应插件能力。不要把不支持的修改当成模型方法本身无效。

### Mac没有CUDA

Bridge、官方评分和CPU参考Trainer不需要CUDA。真实深度Trainer如果写死CUDA，需由
Trainer负责人增加CPU/MPS支持，或在带NVIDIA GPU的电脑运行。

### 如何停止

终端中按`Control+C`。强制中断前已经完成的轮次仍保存在`logs/`和`rounds/`中，
但当前版本的一键入口不会自动把中断中的那一轮当成正式完成结果。

## 13. 最短操作清单

第一次：

```bash
cd /Users/huyaofu/Documents/GitHub/GOAT-LeBron
python3 -m venv kuairand_goat_bridge/.venv
kuairand_goat_bridge/.venv/bin/python -m pip install -e 'kuairand_goat_bridge[goat]'
```

每次正式运行：

```bash
cd /Users/huyaofu/Documents/GitHub/GOAT-LeBron
export AGENT_PROVIDER=deepseek
export DEEPSEEK_API_KEY='你的API密钥'

kuairand_goat_bridge/.venv/bin/python -m kuairand_bridge goat-run \
  --config kuairand_goat_bridge/configs/local_task.yaml \
  --dry-run

kuairand_goat_bridge/.venv/bin/python -m kuairand_bridge goat-run \
  --config kuairand_goat_bridge/configs/local_task.yaml
```

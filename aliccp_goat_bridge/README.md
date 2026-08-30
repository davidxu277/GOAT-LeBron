# NISE Medium 数据接口包

这是AliCCP Medium 5%数据的代码与元数据交付包，抽样seed为`12345678`。

## 数据没有包含在Git目录中

本包不复制Parquet，以免在本机重复占用约716MB空间，也避免把大型数据提交到Git。合作者仍然必须通过共享盘、压缩包或其他传输方式获得以下两个目录：

```text
data/train/*.parquet
data/val/*.parquet
```

如需最终统一测试，再单独提供：

```text
data/public_test/*.parquet
```

在原NISE项目中，源数据位于：

```text
/Users/huyaofu/Documents/GitHub/Original_raw/data_processed/medium_5pct/
```

不要把这个本机绝对路径当作合作者电脑上的路径。

## 安装

```bash
cd /path/to/nise_medium_seed_12345678
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## 读取

将Parquet放入`data/train`和`data/val`后：

```python
import pyarrow.dataset as ds

train = ds.dataset("data/train", format="parquet")
val = ds.dataset("data/val", format="parquet")
```

公共读取接口也可以直接读取交付包内的`data`目录：

```python
from src.data import load_dataset

data = load_dataset(size="medium", processed_dir="data")
train = data.dataset("train")
val = data.dataset("val")
```

## 评估

预测文件必须包含`sample_id`、`ctr`、`cvr`和`ctcvr`：

```python
from src.evaluation import evaluate, format_metrics

metrics = evaluate(
    predictions="results/predictions.parquet",
    labels="data/val",
    output_path="results/metrics.json",
)
print(format_metrics(metrics))
```

CTR AUC在全部曝光上计算；CVR AUC (clicked)在`click=1`子集上计算；CVR AUC (all)在全部曝光上计算。当前评估器尚未与比赛官方Evaluation Script核对。

评估器按批流式读取，适用于Full AliCCP。Prediction必须保留标签数据的`sample_id`行顺序；Parquet文件数、row group和batch边界可以不同。命令行示例：

```bash
python -m aliccp_tools.cli evaluate \
  --labels data/public_test \
  --predictions results/public_test_predictions.parquet \
  --output results/public_test_metrics.json \
  --batch-size 262144
```

Audit同样按批读取。全局重复ID和Train/Val用户泄漏使用自动清理的临时磁盘索引，不再把整份数据转成Python列表：

```bash
python -m aliccp_tools.cli audit \
  --root data \
  --output results/dataset_audit.json \
  --batch-size 262144
```

## 元数据

- `metadata/dataset_manifest.json`：seed、数据规模和特征列表。
- `metadata/split_manifest.json`：用户级Train/Val切分规则。
- `metadata/quality_report.json`：完整原始Train/Test处理质量。
- `metadata/medium_statistics.json`：Medium各分区曝光、点击和转化统计。
- `metadata/aliccp_schema.json`：所有字段编号及含义。

Baseline和Agent必须使用同一份Train与Val，并核对manifest中的seed为`12345678`。

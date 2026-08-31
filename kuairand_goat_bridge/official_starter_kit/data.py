"""KuaiRand-Pure 数据加载、官方日期划分和特征编码。

开发阶段推荐：

    splits = load(
        data_dir,
        include_test=False,
        expose_test_labels=False,
    )

最终生成提交文件时推荐：

    splits = load(
        data_dir,
        include_test=True,
        expose_test_labels=False,
    )

只有复现本地官方 Starter Kit、自检公开测试标签时，才允许使用：

    splits = load(
        data_dir,
        include_test=True,
        expose_test_labels=True,
    )

特征词表、duration 分桶边界等统计量始终只使用 train。
"""

from __future__ import annotations

import collections
import csv
import os
from typing import Any

import numpy as np


LABEL = "long_view"

SPLITS = {
    "train": (20220408, 20220421),
    "valid": (20220422, 20220428),
    "test": (20220429, 20220508),
}

# 官方 FM 使用的 5 个特征域。
FIELDS = [
    "user_id",
    "video_id",
    "author_id",
    "tab",
    "dur_bucket",
]

LOG_FILES = (
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
)

VIDEO_FEATURE_FILE = "video_features_basic_pure.csv"

UNKNOWN_AUTHOR = "UNK"

ROW_DATE_INDEX = 0
ROW_USER_INDEX = 1
ROW_VIDEO_INDEX = 2
ROW_AUTHOR_INDEX = 3
ROW_TAB_INDEX = 4
ROW_DURATION_INDEX = 5
ROW_LABEL_INDEX = 6

DEFAULT_DURATION_BUCKETS = 10


def _find_split(date: int) -> str | None:
    """根据官方日期边界返回记录所属的数据划分。"""
    for split_name, (start_date, end_date) in SPLITS.items():
        if start_date <= date <= end_date:
            return split_name

    return None


def _read_video_authors(data_dir: str) -> dict[str, str]:
    """读取 video_id 到 author_id 的映射。"""
    path = os.path.join(data_dir, VIDEO_FEATURE_FILE)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"缺少视频特征文件：{path}")

    video_to_author: dict[str, str] = {}

    with open(path, newline="", encoding="utf-8-sig") as file_handle:
        reader = csv.DictReader(file_handle)

        required_columns = {"video_id", "author_id"}
        actual_columns = set(reader.fieldnames or [])
        missing_columns = required_columns - actual_columns

        if missing_columns:
            raise ValueError(
                f"{VIDEO_FEATURE_FILE} 缺少字段："
                f"{sorted(missing_columns)}"
            )

        for record in reader:
            video_to_author[record["video_id"]] = record["author_id"]

    return video_to_author


def _parse_label(record: dict[str, str], source_path: str) -> int:
    """解析训练集或验证集标签。

    这个函数不能用于 expose_test_labels=False 的 test 记录。
    """
    if LABEL not in record:
        raise ValueError(
            f"数据文件 {source_path} 缺少标签字段 {LABEL!r}"
        )

    return 1 if record[LABEL] != "0" else 0


def load(
    data_dir: str,
    *,
    include_test: bool = True,
    expose_test_labels: bool = True,
) -> dict[str, list[tuple[Any, ...]]]:
    """按官方日期划分读取 KuaiRand-Pure 数据。

    参数
    ----
    data_dir:
        KuaiRand-Pure CSV 文件所在目录。

    include_test:
        False 时完全跳过 test 日期范围的数据。开发、调参和诊断阶段
        应设为 False。

        True 时读取 test 的非标签字段，可用于最终生成提交文件。

    expose_test_labels:
        只有 include_test=True 时有意义。

        False 时不会访问 test 记录中的 ``long_view`` 字段，返回的
        test 行会用 None 作为标签占位符。

        True 时保留 Starter Kit 原始行为，读取本地 test 标签。
        正式开发流程不得使用这个模式。

    返回
    ----
    dict:
        至少包含 ``train`` 和 ``valid``。当 include_test=True 时还
        包含 ``test``。

        每一行的结构为：

        (
            date,
            user_id,
            video_id,
            author_id,
            tab,
            duration_ms,
            label,
        )

        当 test 标签被隐藏时，test 行中的 label 为 None。
    """
    resolved_data_dir = os.path.abspath(
        os.path.expanduser(data_dir)
    )

    if not os.path.isdir(resolved_data_dir):
        raise FileNotFoundError(
            f"数据目录不存在：{resolved_data_dir}"
        )

    video_to_author = _read_video_authors(resolved_data_dir)

    output: dict[str, list[tuple[Any, ...]]] = {
        "train": [],
        "valid": [],
    }

    if include_test:
        output["test"] = []

    required_feature_columns = {
        "date",
        "user_id",
        "video_id",
        "tab",
        "duration_ms",
    }

    for filename in LOG_FILES:
        source_path = os.path.join(
            resolved_data_dir,
            filename,
        )

        if not os.path.isfile(source_path):
            raise FileNotFoundError(
                f"缺少日志文件：{source_path}"
            )

        with open(
            source_path,
            newline="",
            encoding="utf-8-sig",
        ) as file_handle:
            reader = csv.DictReader(file_handle)

            actual_columns = set(reader.fieldnames or [])
            missing_columns = (
                required_feature_columns - actual_columns
            )

            if missing_columns:
                raise ValueError(
                    f"{filename} 缺少特征字段："
                    f"{sorted(missing_columns)}"
                )

            for record in reader:
                date = int(record["date"])
                split_name = _find_split(date)

                if split_name is None:
                    continue

                # 开发阶段完全跳过 test，既不保存记录，也不解析标签。
                if split_name == "test" and not include_test:
                    continue

                video_id = record["video_id"]

                if (
                    split_name == "test"
                    and not expose_test_labels
                ):
                    # 这里故意不访问 record[LABEL]。
                    label = None
                else:
                    label = _parse_label(
                        record,
                        source_path,
                    )

                row = (
                    date,
                    record["user_id"],
                    video_id,
                    video_to_author.get(
                        video_id,
                        UNKNOWN_AUTHOR,
                    ),
                    record["tab"],
                    float(record["duration_ms"]),
                    label,
                )

                output[split_name].append(row)

    return output


def _bucket_edges(
    durations: list[float],
    n: int = DEFAULT_DURATION_BUCKETS,
) -> np.ndarray:
    """仅根据训练集 duration 计算分桶边界。"""
    if not durations:
        raise ValueError(
            "训练集为空，无法计算 duration 分桶边界"
        )

    if n < 1:
        raise ValueError("duration 分桶数量必须大于等于 1")

    values = np.asarray(durations, dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "训练集 duration_ms 中存在 NaN 或 Inf"
        )

    quantiles = np.linspace(
        0.0,
        1.0,
        n + 1,
    )[1:-1]

    return np.quantile(values, quantiles)


def encode(
    splits: dict[str, list[tuple[Any, ...]]],
) -> tuple[
    dict[str, tuple[np.ndarray, np.ndarray | None, list[Any]]],
    int,
]:
    """把类别特征映射成连续整数 ID。

    所有统计量只使用 train：

    - duration 分桶边界；
    - 各字段词表；
    - 未知值槽位；
    - field dimensions。

    valid 和 test 只使用 train 阶段拟合好的统计量进行转换。
    未见过的值统一映射到对应字段的 UNK 槽。

    对于标签为 None 的 test，返回的 y 也是 None。
    """
    if "train" not in splits:
        raise ValueError("splits 必须包含 train")

    train_rows = splits["train"]

    if not train_rows:
        raise ValueError("train 数据为空，无法拟合编码器")

    duration_edges = _bucket_edges([
        float(row[ROW_DURATION_INDEX])
        for row in train_rows
    ])

    def raw_features(
        row: tuple[Any, ...],
    ) -> list[str]:
        duration_bucket = int(
            np.searchsorted(
                duration_edges,
                float(row[ROW_DURATION_INDEX]),
            )
        )

        return [
            str(row[ROW_USER_INDEX]),
            str(row[ROW_VIDEO_INDEX]),
            str(row[ROW_AUTHOR_INDEX]),
            str(row[ROW_TAB_INDEX]),
            str(duration_bucket),
        ]

    vocabularies: list[dict[str, int]] = [
        {} for _ in FIELDS
    ]

    # R2：词表严格只从 train 构建。
    for row in train_rows:
        values = raw_features(row)

        for field_index, value in enumerate(values):
            vocabulary = vocabularies[field_index]

            if value not in vocabulary:
                vocabulary[value] = len(vocabulary)

    unknown_ids = [
        len(vocabulary)
        for vocabulary in vocabularies
    ]

    field_dimensions = [
        len(vocabulary) + 1
        for vocabulary in vocabularies
    ]

    offsets = np.cumsum(
        [0] + field_dimensions[:-1]
    ).astype(np.int32)

    encoded: dict[
        str,
        tuple[np.ndarray, np.ndarray | None, list[Any]],
    ] = {}

    for split_name, rows in splits.items():
        features = np.empty(
            (len(rows), len(FIELDS)),
            dtype=np.int32,
        )
        users: list[Any] = []

        labels_are_exposed = all(
            row[ROW_LABEL_INDEX] is not None
            for row in rows
        )

        labels: np.ndarray | None

        if labels_are_exposed:
            labels = np.empty(
                len(rows),
                dtype=np.float32,
            )
        else:
            labels = None

        for row_index, row in enumerate(rows):
            values = raw_features(row)

            for field_index, value in enumerate(values):
                encoded_value = vocabularies[
                    field_index
                ].get(
                    value,
                    unknown_ids[field_index],
                )

                features[row_index, field_index] = (
                    encoded_value + offsets[field_index]
                )

            if labels is not None:
                labels[row_index] = float(
                    row[ROW_LABEL_INDEX]
                )

            users.append(row[ROW_USER_INDEX])

        encoded[split_name] = (
            features,
            labels,
            users,
        )

    return encoded, int(sum(field_dimensions))

"""Stable, label-safe dataset interface for teammate models."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Iterator, Sequence

import numpy as np

from .official import module


FIDELITY_FRACTIONS = {
    "小份": 0.15,
    "中份": 0.40,
    "大份": 0.75,
    "全量": 1.00,
}


@dataclass(frozen=True)
class SplitView:
    name: str
    rows: Sequence[tuple]
    expose_labels: bool

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def row_ids(self) -> np.ndarray:
        return np.arange(len(self.rows), dtype=np.int64)

    @property
    def user_ids(self) -> np.ndarray:
        return np.asarray([r[1] for r in self.rows])

    @property
    def video_ids(self) -> np.ndarray:
        return np.asarray([r[2] for r in self.rows])

    @property
    def labels(self) -> np.ndarray:
        if not self.expose_labels:
            raise PermissionError("test 标签被适配层锁定；训练和调参只能使用 train/valid")
        return np.asarray([r[6] for r in self.rows], dtype=np.float32)

    def records(self) -> Iterator[dict]:
        for i, r in enumerate(self.rows):
            item = {
                "row_id": i, "date": r[0], "user_id": r[1],
                "video_id": r[2], "author_id": r[3], "tab": r[4],
                "duration_ms": r[5],
            }
            if self.expose_labels:
                item["long_view"] = r[6]
            yield item


@dataclass(frozen=True)
class DatasetBundle:
    train: SplitView
    valid: SplitView
    test: SplitView | None
    data_dir: pathlib.Path

    def with_train_fidelity(self, fidelity: str, seed: int) -> "DatasetBundle":
        """只缩放训练集；官方 validation/test 永远保持固定日期切分和全量。

        正负标签分别做确定性抽样，避免小份数据偶然改变 long_view 比例。
        选中后按原始行号排序，保证同一 seed/fidelity 在各平台可复现。
        """
        if fidelity not in FIDELITY_FRACTIONS:
            raise ValueError(
                f"fidelity 必须是 {sorted(FIDELITY_FRACTIONS)}，收到 {fidelity!r}"
            )
        fraction = FIDELITY_FRACTIONS[fidelity]
        if fraction >= 1.0:
            return self

        labels = self.train.labels.astype(np.int8)
        rng = np.random.default_rng(int(seed))
        selected: list[int] = []
        for label in (0, 1):
            candidates = np.flatnonzero(labels == label)
            if not len(candidates):
                continue
            count = max(1, int(round(len(candidates) * fraction)))
            selected.extend(rng.choice(candidates, size=count, replace=False).tolist())
        selected.sort()
        sampled_rows = [self.train.rows[index] for index in selected]
        return DatasetBundle(
            train=SplitView("train", sampled_rows, True),
            valid=self.valid,
            test=self.test,
            data_dir=self.data_dir,
        )

    def split(self, name: str) -> SplitView:
        if name not in {"train", "valid", "test"}:
            raise ValueError("split 必须是 train、valid 或 test")
        
        value = getattr(self, name)
        if value is None:
            raise PermissionError("开发运行没有加载 test；只能在最终提交阶段加载")
        return value

    def official_encoded(self):
        """Official FM-compatible arrays with the test label replaced by None."""
        safe_test = [tuple(r[:6]) + (0,) for r in self.test.rows]
        raw = {"train": self.train.rows, "valid": self.valid.rows, "test": safe_test}
        encoded, dim = module("data").encode(raw)
        test_x, _discarded, test_users = encoded["test"]
        encoded["test"] = (test_x, None, test_users)
        return encoded, dim


def load_dataset(
    data_dir: str | pathlib.Path,
    *,
    include_test: bool = False,
) -> DatasetBundle:
    data_dir = pathlib.Path(data_dir).expanduser().resolve()

    raw = module("data").load(
        str(data_dir),
        include_test=include_test,
        expose_test_labels=False,
    )

    test = None
    if include_test:
        test_rows = [tuple(row[:6]) + (None,) for row in raw["test"]]
        test = SplitView("test", test_rows, False)

    return DatasetBundle(
        train=SplitView("train", raw["train"], True),
        valid=SplitView("valid", raw["valid"], True),
        test=test,
        data_dir=data_dir,
    )

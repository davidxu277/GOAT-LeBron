"""Stable, label-safe dataset interface for teammate models."""

from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Iterator, Sequence

import numpy as np

from .official import module


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

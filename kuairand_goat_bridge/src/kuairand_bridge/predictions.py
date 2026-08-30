"""Normalize teammate prediction files into the exact official CSV contract."""

from __future__ import annotations

import csv
import pathlib
from typing import Sequence

import numpy as np

from .official import module


def _read_scores(path: pathlib.Path) -> tuple[np.ndarray, list[dict] | None]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.asarray(np.load(path), dtype=float).reshape(-1), None
    if suffix == ".npz":
        obj = np.load(path)
        keys = [k for k in ("score", "scores", "prediction", "predictions") if k in obj]
        if not keys:
            raise ValueError("NPZ 必须包含 score/scores/prediction/predictions 之一")
        return np.asarray(obj[keys[0]], dtype=float).reshape(-1), None
    if suffix != ".csv":
        raise ValueError("预测文件只支持 .csv、.npy 或 .npz")
    with path.open(newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError("CSV 没有表头")
        score_col = next((c for c in ("score", "scores", "prediction", "predictions")
                          if c in reader.fieldnames), None)
        if score_col is None:
            raise ValueError("CSV 必须包含 score 列（也兼容 scores/prediction/predictions）")
        records = list(reader)
    return np.asarray([float(r[score_col]) for r in records], dtype=float), records


def normalize_predictions(source: str | pathlib.Path, destination: str | pathlib.Path,
                          rows: Sequence[tuple]) -> pathlib.Path:
    source, destination = pathlib.Path(source), pathlib.Path(destination)
    scores, records = _read_scores(source)
    if len(scores) != len(rows):
        raise ValueError(f"预测有 {len(scores):,} 行，目标 split 有 {len(rows):,} 行")
    if not np.isfinite(scores).all():
        raise ValueError("预测中存在 NaN 或 Inf")

    if records and {"row_id", "user_id", "video_id"} <= set(records[0]):
        for i, (record, expected) in enumerate(zip(records, rows)):
            if int(record["row_id"]) != i:
                raise ValueError(f"第 {i} 条预测 row_id 错位")
            if str(record["user_id"]) != str(expected[1]) or str(record["video_id"]) != str(expected[2]):
                raise ValueError(f"第 {i} 条预测 user_id/video_id 错位")

    destination.parent.mkdir(parents=True, exist_ok=True)
    module("submit").write_submission(str(destination), rows, scores)
    module("submit").read_submission(str(destination), rows)
    return destination

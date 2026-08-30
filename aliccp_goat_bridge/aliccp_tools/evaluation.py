"""Memory-bounded AliCCP prediction validation and evaluation."""
from __future__ import annotations

import json
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


LABEL_COLUMNS = ["sample_id", "click", "conversion"]
PREDICTION_COLUMNS = ["sample_id", "ctr", "cvr", "ctcvr"]
DEFAULT_BATCH_SIZE = 262_144
MAX_IN_MEMORY_SCORES = 1_000_000


def _iter_batches(path: Path, columns: list[str], batch_size: int) -> Iterator[object]:
    """Yield selected columns without materialising the complete input."""
    import pyarrow.csv as pacsv
    import pyarrow.dataset as ds
    import pyarrow.json as pajson
    import pyarrow.parquet as pq

    if path.is_dir():
        dataset = ds.dataset(path, format="parquet")
        missing = sorted(set(columns) - set(dataset.schema.names))
        if missing:
            raise ValueError(f"missing columns in {path}: {missing}")
        yield from dataset.to_batches(columns=columns, batch_size=batch_size)
        return

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        missing = sorted(set(columns) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"missing columns in {path}: {missing}")
        yield from parquet.iter_batches(columns=columns, batch_size=batch_size)
        return
    if suffix == ".csv":
        reader = pacsv.open_csv(path)
    elif suffix in {".json", ".jsonl", ".ndjson"}:
        reader = pajson.open_json(path)
    else:
        raise ValueError(f"unsupported table format: {path}")
    missing = sorted(set(columns) - set(reader.schema.names))
    if missing:
        raise ValueError(f"missing columns in {path}: {missing}")
    for batch in reader:
        yield batch.select(columns)


def _aligned_slices(
    labels_path: Path, predictions_path: Path, batch_size: int,
) -> Iterator[tuple[object, object, int]]:
    """Yield equally-sized label/prediction slices and enforce row alignment."""
    import pyarrow.compute as pc

    labels = iter(_iter_batches(labels_path, LABEL_COLUMNS, batch_size))
    predictions = iter(_iter_batches(predictions_path, PREDICTION_COLUMNS, batch_size))
    label_batch = next(labels, None)
    prediction_batch = next(predictions, None)
    label_offset = prediction_offset = row_offset = 0
    while label_batch is not None and prediction_batch is not None:
        size = min(
            label_batch.num_rows - label_offset,
            prediction_batch.num_rows - prediction_offset,
        )
        label_slice = label_batch.slice(label_offset, size)
        prediction_slice = prediction_batch.slice(prediction_offset, size)
        equal = pc.equal(label_slice.column(0), prediction_slice.column(0))
        if equal.null_count or not pc.all(equal).as_py():
            mismatch = 0
            for mismatch, matches in enumerate(equal.to_pylist()):
                if matches is not True:
                    break
            label_id = label_slice.column(0)[mismatch].as_py()
            prediction_id = prediction_slice.column(0)[mismatch].as_py()
            raise ValueError(
                "sample_id order mismatch at row "
                f"{row_offset + mismatch}: labels={label_id!r}, predictions={prediction_id!r}. "
                "Full-data streaming evaluation requires predictions in the same row order as labels."
            )
        yield label_slice, prediction_slice, row_offset
        row_offset += size
        label_offset += size
        prediction_offset += size
        if label_offset == label_batch.num_rows:
            label_batch = next(labels, None)
            label_offset = 0
        if prediction_offset == prediction_batch.num_rows:
            prediction_batch = next(predictions, None)
            prediction_offset = 0
    if label_batch is not None or prediction_batch is not None:
        raise ValueError(
            f"prediction coverage mismatch after {row_offset} aligned rows: "
            "labels and predictions have different row counts"
        )


class _AucGroups:
    """Exact tie-aware ROC AUC with automatic spill to temporary disk."""

    def __init__(
        self,
        connection: sqlite3.Connection | None = None,
        table_name: str | None = None,
        max_in_memory_scores: int | None = MAX_IN_MEMORY_SCORES,
    ) -> None:
        self.groups: dict[float, list[int]] = defaultdict(lambda: [0, 0])
        self.connection = connection
        self.table_name = table_name
        self.max_in_memory_scores = max_in_memory_scores
        self.spilled = False
        if connection is not None and table_name is not None:
            connection.execute(
                f"CREATE TABLE {table_name} ("
                "score REAL PRIMARY KEY, negative INTEGER NOT NULL, positive INTEGER NOT NULL"
                ") WITHOUT ROWID"
            )

    def _write_disk(self, rows: list[tuple[float, int, int]]) -> None:
        if not rows:
            return
        assert self.connection is not None and self.table_name is not None
        self.connection.executemany(
            f"INSERT INTO {self.table_name}(score, negative, positive) VALUES (?, ?, ?) "
            "ON CONFLICT(score) DO UPDATE SET "
            "negative = negative + excluded.negative, "
            "positive = positive + excluded.positive",
            rows,
        )

    def _spill(self) -> None:
        if self.spilled:
            return
        if self.connection is None or self.table_name is None:
            return
        self._write_disk(
            [(score, counts[0], counts[1]) for score, counts in self.groups.items()]
        )
        self.groups.clear()
        self.spilled = True

    def add(self, scores: object, labels: object) -> None:
        import pyarrow as pa

        if len(scores) == 0:
            return
        grouped = pa.table({"score": scores, "label": labels}).group_by("score").aggregate(
            [("label", "sum"), ("label", "count")]
        )
        rows = [
            (float(score), int(total - positive), int(positive))
            for score, positive, total in zip(
            grouped["score"].to_pylist(),
            grouped["label_sum"].to_pylist(),
            grouped["label_count"].to_pylist(),
            )
        ]
        if self.spilled:
            self._write_disk(rows)
            return
        for score, negative, positive in rows:
            bucket = self.groups[score]
            bucket[0] += negative
            bucket[1] += positive
        if (
            self.max_in_memory_scores is not None
            and len(self.groups) > self.max_in_memory_scores
        ):
            self._spill()

    def auc(self) -> float:
        if self.spilled:
            assert self.connection is not None and self.table_name is not None
            negatives, positives = self.connection.execute(
                f"SELECT COALESCE(SUM(negative), 0), COALESCE(SUM(positive), 0) "
                f"FROM {self.table_name}"
            ).fetchone()
            ordered = self.connection.execute(
                f"SELECT negative, positive FROM {self.table_name} ORDER BY score"
            )
        else:
            positives = sum(group[1] for group in self.groups.values())
            negatives = sum(group[0] for group in self.groups.values())
            ordered = (self.groups[score] for score in sorted(self.groups))
        if positives == 0 or negatives == 0:
            raise ValueError("ROC AUC is undefined when only one label class is present")
        lower_negatives = 0
        concordant = 0.0
        for negative, positive in ordered:
            concordant += positive * (lower_negatives + 0.5 * negative)
            lower_negatives += negative
        return concordant / (positives * negatives)


def binary_auc(labels: list[int], scores: list[float]) -> float:
    """Compute exact tie-aware ROC AUC for the public small-array interface."""
    if len(labels) != len(scores):
        raise ValueError("labels and scores have different lengths")
    import pyarrow as pa

    groups = _AucGroups(max_in_memory_scores=None)
    groups.add(pa.array(scores, type=pa.float64()), pa.array(labels, type=pa.int8()))
    return groups.auc()


def _validate_labels(batch: object, row_offset: int) -> None:
    import pyarrow.compute as pc

    click = batch.column(1)
    conversion = batch.column(2)
    valid_click = pc.or_(pc.equal(click, 0), pc.equal(click, 1))
    valid_conversion = pc.or_(pc.equal(conversion, 0), pc.equal(conversion, 1))
    invalid_funnel = pc.and_(pc.equal(click, 0), pc.equal(conversion, 1))
    if (
        valid_click.null_count
        or valid_conversion.null_count
        or not pc.all(valid_click).as_py()
        or not pc.all(valid_conversion).as_py()
        or pc.any(invalid_funnel).as_py()
    ):
        raise ValueError(f"invalid label values in batch beginning at row {row_offset}")


def _validate_predictions(batch: object, row_offset: int) -> None:
    import pyarrow.compute as pc

    for index, column in enumerate(("ctr", "cvr", "ctcvr"), start=1):
        values = batch.column(index)
        valid = pc.and_(pc.is_finite(values), pc.and_(pc.greater_equal(values, 0), pc.less_equal(values, 1)))
        if valid.null_count or not pc.all(valid).as_py():
            raise ValueError(
                f"{column} contains null, non-finite or out-of-range values "
                f"in batch beginning at row {row_offset}"
            )
    product = pc.multiply(batch.column(1), batch.column(2))
    difference = pc.abs(pc.subtract(batch.column(3), product))
    tolerance = pc.add(1e-8, pc.multiply(1e-6, pc.abs(product)))
    if not pc.all(pc.less_equal(difference, tolerance)).as_py():
        raise ValueError(f"ctcvr must equal ctr * cvr near row {row_offset}")


def evaluate_predictions(
    labels_path: Path,
    predictions_path: Path,
    output_path: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Stream aligned labels/predictions and return exact CTR/CVR AUC metrics.

    To remain memory bounded on full AliCCP, prediction rows must retain the
    same sample_id order as the label split. Different Parquet batch and row
    group boundaries are supported.
    """
    import pyarrow.compute as pc

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    with tempfile.TemporaryDirectory(prefix="aliccp-evaluation-") as temp_directory:
        connection = sqlite3.connect(Path(temp_directory) / "scores.sqlite3")
        connection.execute("PRAGMA journal_mode=OFF")
        connection.execute("PRAGMA synchronous=OFF")
        connection.execute("PRAGMA temp_store=FILE")
        connection.execute("PRAGMA cache_size=-65536")
        ctr_groups = _AucGroups(connection, "ctr_scores")
        cvr_groups = _AucGroups(connection, "cvr_scores")
        cvr_all_groups = _AucGroups(connection, "cvr_all_scores")
        rows = clicked_rows = click_positives = conversion_positives = 0

        for labels, predictions, row_offset in _aligned_slices(
            Path(labels_path), Path(predictions_path), batch_size
        ):
            _validate_labels(labels, row_offset)
            _validate_predictions(predictions, row_offset)
            click = labels.column(1)
            conversion = labels.column(2)
            clicked = pc.equal(click, 1)
            ctr_groups.add(predictions.column(1), click)
            cvr_groups.add(pc.filter(predictions.column(2), clicked), pc.filter(conversion, clicked))
            cvr_all_groups.add(predictions.column(2), conversion)
            rows += labels.num_rows
            batch_clicks = int(pc.sum(click).as_py())
            clicked_rows += batch_clicks
            click_positives += batch_clicks
            conversion_positives += int(pc.sum(pc.filter(conversion, clicked)).as_py())

        auc_values: dict[str, float | None] = {}
        warnings: list[str] = []
        for key, name, groups in (
            ("ctr_auc", "CTR AUC", ctr_groups),
            ("cvr_auc", "CVR AUC", cvr_groups),
            ("cvr_auc_all", "CVR AUC (all)", cvr_all_groups),
        ):
            try:
                auc_values[key] = groups.auc()
            except ValueError as error:
                auc_values[key] = None
                warnings.append(f"{name} unavailable: {error}")
        score_storage = {
            "ctr": "temporary_disk" if ctr_groups.spilled else "memory",
            "cvr": "temporary_disk" if cvr_groups.spilled else "memory",
            "cvr_all": "temporary_disk" if cvr_all_groups.spilled else "memory",
        }
        connection.close()

    metrics: dict[str, Any] = {
        "status": "succeeded",
        "streaming": True,
        "sample_ids_aligned": True,
        "batch_size": batch_size,
        "max_in_memory_distinct_scores": MAX_IN_MEMORY_SCORES,
        "score_storage": score_storage,
        "rows": rows,
        "clicked_rows": clicked_rows,
        "click_positives": click_positives,
        "conversion_positives_in_clicked": conversion_positives,
        "ctr_auc": auc_values["ctr_auc"],
        "cvr_auc": auc_values["cvr_auc"],
        "cvr_auc_all": auc_values["cvr_auc_all"],
        "warnings": warnings,
        "labels_path": str(Path(labels_path).resolve()),
        "predictions_path": str(Path(predictions_path).resolve()),
    }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics

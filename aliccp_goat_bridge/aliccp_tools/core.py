from __future__ import annotations

import csv
import hashlib
import json
import math
import secrets
import shutil
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, TextIO

from .schema import feature_ids, load_schema


FEATURE_SEPARATOR = "\x01"
KEY_SEPARATOR = "\x02"
VALUE_SEPARATOR = "\x03"

# Official feature IDs used by AliCCP. Keeping a stable set gives every parquet
# part the same schema even if a particular batch does not contain one feature.
FEATURE_IDS = feature_ids()
MULTI_VALUE_FIELDS = frozenset(load_schema()["agent_guidance"]["multi_value_fields"] + ["508", "509", "702", "853"])


def stable_fraction(value: str, salt: str) -> float:
    digest = hashlib.blake2b(f"{salt}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def shard_number(value: str, shards: int) -> int:
    if shards <= 0:
        raise ValueError("shards must be positive")
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big") % shards


def parse_feature_string(raw: str) -> tuple[dict[str, str], dict[str, float], int]:
    """Parse AliCCP's SOH/STX/ETX encoded triples.

    Repeated categorical IDs retain the last value (matching common public
    loaders); their numeric weights are summed into a corresponding D* column.
    The returned integer counts malformed triples.
    """
    categorical: dict[str, str] = {}
    dense: dict[str, float] = {}
    malformed = 0
    if not raw:
        return categorical, dense, malformed
    for token in raw.split(FEATURE_SEPARATOR):
        parts = token.split(KEY_SEPARATOR, 1)
        if len(parts) != 2:
            malformed += 1
            continue
        feature_type, remainder = parts
        values = remainder.split(VALUE_SEPARATOR, 1)
        if len(values) != 2 or not feature_type or not values[0]:
            malformed += 1
            continue
        feature_value, weight_text = values
        categorical[feature_type] = feature_value
        try:
            weight = float(weight_text)
            if not math.isfinite(weight):
                raise ValueError
            dense[f"D{feature_type}"] = dense.get(f"D{feature_type}", 0.0) + weight
        except ValueError:
            malformed += 1
    return categorical, dense, malformed


def parse_feature_lists(raw: str) -> tuple[dict[str, list[str]], dict[str, list[float]], int]:
    """Parse every value/weight without discarding repeated field IDs."""
    values: dict[str, list[str]] = {}
    weights: dict[str, list[float]] = {}
    malformed = 0
    if not raw:
        return values, weights, malformed
    for token in raw.split(FEATURE_SEPARATOR):
        field_parts = token.split(KEY_SEPARATOR, 1)
        if len(field_parts) != 2:
            malformed += 1
            continue
        field_id, value_and_weight = field_parts
        value_parts = value_and_weight.split(VALUE_SEPARATOR, 1)
        if len(value_parts) != 2 or not field_id or not value_parts[0]:
            malformed += 1
            continue
        value, weight_text = value_parts
        try:
            weight = float(weight_text)
            if not math.isfinite(weight):
                raise ValueError
        except ValueError:
            malformed += 1
            continue
        values.setdefault(field_id, []).append(value)
        weights.setdefault(field_id, []).append(weight)
    return values, weights, malformed


def parse_skeleton(row: list[str]) -> dict[str, object]:
    if len(row) != 6:
        raise ValueError(f"skeleton row has {len(row)} columns, expected 6")
    sample_id, click, conversion, common_id, feature_count, raw_features = row
    click_value, conversion_value = int(click), int(conversion)
    if click_value not in (0, 1) or conversion_value not in (0, 1):
        raise ValueError("labels must be 0 or 1")
    categorical, dense, malformed = parse_feature_lists(raw_features)
    return {
        "sample_id": sample_id,
        "click": click_value,
        "conversion": conversion_value,
        "common_id": common_id,
        "declared_feature_count": int(feature_count),
        "features": categorical,
        "dense": dense,
        "malformed_features": malformed,
    }


def parse_common(row: list[str]) -> dict[str, object]:
    if len(row) != 3:
        raise ValueError(f"common row has {len(row)} columns, expected 3")
    common_id, feature_count, raw_features = row
    categorical, dense, malformed = parse_feature_lists(raw_features)
    return {
        "common_id": common_id,
        "declared_feature_count": int(feature_count),
        "features": categorical,
        "dense": dense,
        "malformed_features": malformed,
    }


def iter_csv(path: Path) -> Iterator[list[str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        yield from csv.reader(handle)


def paths_for(data_dir: Path, split: str) -> tuple[Path, Path]:
    directory = data_dir / f"sample_{split}"
    return directory / f"sample_skeleton_{split}.csv", directory / f"common_features_{split}.csv"


def preview(data_dir: Path, split: str, rows: int) -> dict[str, object]:
    skeleton_path, common_path = paths_for(data_dir, split)
    result: dict[str, object] = {"split": split, "skeleton": [], "common": []}
    for row in _take(iter_csv(skeleton_path), rows):
        try:
            parsed = parse_skeleton(row)
            parsed.pop("dense")
            result["skeleton"].append(parsed)  # type: ignore[union-attr]
        except Exception as error:
            result["skeleton"].append({"error": str(error), "raw_columns": row[:5]})  # type: ignore[union-attr]
    for row in _take(iter_csv(common_path), rows):
        try:
            parsed = parse_common(row)
            parsed.pop("dense")
            result["common"].append(parsed)  # type: ignore[union-attr]
        except Exception as error:
            result["common"].append({"error": str(error), "raw_columns": row[:2]})  # type: ignore[union-attr]
    return result


def _take(items: Iterable[list[str]], count: int) -> Iterator[list[str]]:
    for index, item in enumerate(items):
        if index >= count:
            break
        yield item


def inspect_file(path: Path, kind: str) -> dict[str, object]:
    report: dict[str, object] = {
        "path": str(path), "bytes": path.stat().st_size, "rows": 0, "malformed_rows": 0,
        "malformed_feature_tokens": 0,
    }
    labels: Counter[str] = Counter()
    feature_types: Counter[str] = Counter()
    for row in iter_csv(path):
        report["rows"] = int(report["rows"]) + 1
        try:
            parsed = parse_skeleton(row) if kind == "skeleton" else parse_common(row)
            report["malformed_feature_tokens"] = int(report["malformed_feature_tokens"]) + int(parsed["malformed_features"])
            feature_types.update(parsed["features"].keys())  # type: ignore[union-attr]
            if kind == "skeleton":
                labels[f"click={parsed['click']},conversion={parsed['conversion']}"] += 1
        except Exception:
            report["malformed_rows"] = int(report["malformed_rows"]) + 1
    report["labels"] = dict(labels)
    report["feature_types"] = dict(feature_types)
    return report


@dataclass
class Quality:
    skeleton_rows: int = 0
    common_rows: int = 0
    output_rows: int = 0
    malformed_skeleton_rows: int = 0
    malformed_common_rows: int = 0
    missing_common_ids: int = 0
    invalid_funnel_rows: int = 0
    malformed_feature_tokens: int = 0
    labels: Counter[str] = field(default_factory=Counter)

    def merge(self, other: "Quality") -> None:
        for name in (
            "skeleton_rows", "common_rows", "output_rows", "malformed_skeleton_rows",
            "malformed_common_rows", "missing_common_ids", "invalid_funnel_rows",
            "malformed_feature_tokens",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.labels.update(other.labels)

    def as_dict(self) -> dict[str, object]:
        result = {key: value for key, value in self.__dict__.items() if key != "labels"}
        result["labels"] = dict(self.labels)
        result["association_hit_rate"] = (
            (self.skeleton_rows - self.missing_common_ids) / self.skeleton_rows if self.skeleton_rows else None
        )
        return result


class ShardHandles:
    def __init__(self, directory: Path, prefix: str, shards: int, max_open: int = 48):
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory
        self.prefix = prefix
        self.shards = shards
        self.max_open = max_open
        self.handles: OrderedDict[int, tuple[TextIO, object]] = OrderedDict()
        for index in range(shards):
            (directory / f"{prefix}-{index:04d}.csv").touch()

    def writerow(self, index: int, row: list[str]) -> None:
        if index in self.handles:
            handle, writer = self.handles.pop(index)
            self.handles[index] = (handle, writer)
        else:
            if len(self.handles) >= self.max_open:
                _, (old_handle, _) = self.handles.popitem(last=False)
                old_handle.close()
            handle = (self.directory / f"{self.prefix}-{index:04d}.csv").open("a", encoding="utf-8", newline="")
            writer = csv.writer(handle)
            self.handles[index] = (handle, writer)
        writer.writerow(row)  # type: ignore[attr-defined]

    def close(self) -> None:
        for handle, _ in self.handles.values():
            handle.close()
        self.handles.clear()


def partition_inputs(data_dir: Path, split: str, staging: Path, shards: int) -> None:
    done = staging / split / ".partitioned"
    if done.exists():
        return
    skeleton_path, common_path = paths_for(data_dir, split)
    split_dir = staging / split
    if split_dir.exists():
        shutil.rmtree(split_dir)
    common_handles = ShardHandles(split_dir, "common", shards)
    skeleton_handles = ShardHandles(split_dir, "skeleton", shards)
    try:
        for row in iter_csv(common_path):
            key = row[0] if row else ""
            common_handles.writerow(shard_number(key, shards), row)
        for row in iter_csv(skeleton_path):
            key = row[3] if len(row) > 3 else ""
            skeleton_handles.writerow(shard_number(key, shards), row)
    finally:
        common_handles.close()
        skeleton_handles.close()
    done.touch()


def _flat_record(skeleton: dict[str, object], common: dict[str, object]) -> dict[str, object]:
    common_features = common["features"]  # type: ignore[assignment]
    skeleton_features = skeleton["features"]  # type: ignore[assignment]
    common_weights = common["dense"]  # type: ignore[assignment]
    skeleton_weights = skeleton["dense"]  # type: ignore[assignment]
    record: dict[str, object] = {
        "sample_id": str(skeleton["sample_id"]),
        "common_id": str(skeleton["common_id"]),
        "click": int(skeleton["click"]),
        "conversion": int(skeleton["conversion"]),
    }
    for feature_id in FEATURE_IDS:
        raw_values = list(common_features.get(feature_id, [])) + list(skeleton_features.get(feature_id, []))
        raw_weights = list(common_weights.get(feature_id, [])) + list(skeleton_weights.get(feature_id, []))
        values = [int(value) for value in raw_values if str(value).lstrip("-").isdigit()]
        if feature_id in MULTI_VALUE_FIELDS:
            record[feature_id] = values
            record[f"D{feature_id}"] = [float(value) for value in raw_weights]
        else:
            record[feature_id] = values[-1] if values else None
            record[f"D{feature_id}"] = float(raw_weights[-1]) if raw_weights else None
    return record


def _destinations(
    record: dict[str, object], split: str, validation_fraction: float, sampling_seed: str,
) -> list[tuple[str, str]]:
    user_key = str(record.get("101") or record["common_id"])
    output_split = "public_test" if split == "test" else split
    if split == "train":
        output_split = (
            "val"
            if stable_fraction(user_key, f"split:{sampling_seed}") < validation_fraction
            else "train"
        )
    fraction = stable_fraction(user_key, f"sample:{sampling_seed}")
    scales = (("small_0.5pct", 0.005), ("medium_5pct", 0.05), ("large_25pct", 0.25), ("full", 1.0))
    return [(scale, output_split) for scale, limit in scales if fraction < limit]


def process_shard(
    split: str, index: int, staging: Path, output_dir: Path, batch_rows: int,
    validation_fraction: float, sampling_seed: str,
) -> Quality:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("Parquet output requires pyarrow; install aliccp_tools/requirements.txt") from error

    completed = staging / split / f".completed-{index:04d}"
    quality_path = staging / split / f"quality-{index:04d}.json"
    if completed.exists() and quality_path.exists():
        saved = json.loads(quality_path.read_text())
        fields = Quality.__dataclass_fields__
        values = {key: value for key, value in saved.items() if key in fields and key != "labels"}
        return Quality(**values, labels=Counter(saved.get("labels", {})))

    common: dict[str, dict[str, object]] = {}
    quality = Quality()
    for row in iter_csv(staging / split / f"common-{index:04d}.csv"):
        quality.common_rows += 1
        try:
            parsed = parse_common(row)
            quality.malformed_feature_tokens += int(parsed["malformed_features"])
            common[str(parsed["common_id"])] = parsed
        except Exception:
            quality.malformed_common_rows += 1

    buffers: dict[tuple[str, str], list[dict[str, object]]] = {}
    part_numbers: Counter[tuple[str, str]] = Counter()

    def flush(destination: tuple[str, str]) -> None:
        records = buffers.get(destination, [])
        if not records:
            return
        scale, output_split = destination
        target = output_dir / scale / output_split
        target.mkdir(parents=True, exist_ok=True)
        part = part_numbers[destination]
        pq.write_table(pa.Table.from_pylist(records), target / f"part-{split}-{index:04d}-{part:04d}.parquet", compression="zstd")
        part_numbers[destination] += 1
        buffers[destination] = []

    for row in iter_csv(staging / split / f"skeleton-{index:04d}.csv"):
        quality.skeleton_rows += 1
        try:
            skeleton = parse_skeleton(row)
            quality.malformed_feature_tokens += int(skeleton["malformed_features"])
        except Exception:
            quality.malformed_skeleton_rows += 1
            continue
        quality.labels[f"click={skeleton['click']},conversion={skeleton['conversion']}"] += 1
        if skeleton["click"] == 0 and skeleton["conversion"] == 1:
            quality.invalid_funnel_rows += 1
            continue
        common_row = common.get(str(skeleton["common_id"]))
        if common_row is None:
            quality.missing_common_ids += 1
            continue
        record = _flat_record(skeleton, common_row)
        quality.output_rows += 1
        for destination in _destinations(record, split, validation_fraction, sampling_seed):
            bucket = buffers.setdefault(destination, [])
            bucket.append(record)
            if len(bucket) >= batch_rows:
                flush(destination)
    for destination in list(buffers):
        flush(destination)
    quality_path.write_text(json.dumps(quality.as_dict(), indent=2) + "\n", encoding="utf-8")
    completed.touch()
    # Only remove staging files created by this program, never source data.
    (staging / split / f"common-{index:04d}.csv").unlink(missing_ok=True)
    (staging / split / f"skeleton-{index:04d}.csv").unlink(missing_ok=True)
    return quality


def prepare(
    data_dir: Path, output_dir: Path, shards: int, batch_rows: int, validation_fraction: float,
    sampling_seed: str | int | None = None,
) -> dict[str, object]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    resolved_seed = str(sampling_seed if sampling_seed is not None else secrets.randbits(64))
    seed_path = output_dir / ".sampling_seed"
    previous_seed = seed_path.read_text(encoding="utf-8").strip() if seed_path.exists() else None
    if previous_seed != resolved_seed:
        for scale in ("small_0.5pct", "medium_5pct", "large_25pct", "full"):
            scale_dir = output_dir / scale
            if scale_dir.is_dir():
                for path in scale_dir.rglob("*.parquet"):
                    path.unlink()
        shutil.rmtree(output_dir / "_staging", ignore_errors=True)
    staging = output_dir / "_staging" / f"seed-{resolved_seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(resolved_seed + "\n", encoding="utf-8")
    combined: dict[str, object] = {}
    for split in ("train", "test"):
        partition_inputs(data_dir, split, staging, shards)
        total = Quality()
        for index in range(shards):
            total.merge(process_shard(
                split, index, staging, output_dir, batch_rows,
                validation_fraction, resolved_seed,
            ))
        combined[split] = total.as_dict()
    manifest = {
        "source": str(data_dir.resolve()),
        "scales": {"small_0.5pct": 0.005, "medium_5pct": 0.05, "large_25pct": 0.25, "full": 1.0},
        "validation_fraction": validation_fraction,
        "sampling_seed": resolved_seed,
        "sampling_unit": "user_id_101_with_common_id_fallback",
        "sampling_salt": f"sample:{resolved_seed}",
        "split_salt": f"split:{resolved_seed}",
        "feature_ids": list(FEATURE_IDS),
        "quality": combined,
    }
    (output_dir / "quality_report.json").write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    split_manifest = {
        "strategy": "stable_user_hash",
        "source_train": "data/sample_train",
        "source_public_test": "data/sample_test",
        "user_field": "101",
        "fallback_key": "common_id",
        "train_fraction": 1.0 - validation_fraction,
        "validation_fraction": validation_fraction,
        "hash_algorithm": "blake2b-64",
        "sampling_seed": resolved_seed,
        "sampling_salt": f"sample:{resolved_seed}",
        "split_salt": f"split:{resolved_seed}",
        "public_test_used_for_training": False,
    }
    (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

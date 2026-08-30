"""Create a small, real joined dataset for reviewing the split before a full run."""
from __future__ import annotations

import json
import secrets
from collections import Counter
from pathlib import Path
from typing import Any

from .core import _flat_record, iter_csv, parse_common, parse_skeleton, paths_for, stable_fraction


def _read_skeleton(path: Path, limit: int) -> tuple[list[dict[str, object]], Counter[str]]:
    records: list[dict[str, object]] = []
    errors: Counter[str] = Counter()
    for row in iter_csv(path):
        if len(records) >= limit:
            break
        try:
            parsed = parse_skeleton(row)
        except Exception:
            errors["malformed_skeleton_rows"] += 1
            continue
        if parsed["click"] == 0 and parsed["conversion"] == 1:
            errors["invalid_funnel_rows"] += 1
            continue
        records.append(parsed)
    return records, errors


def _find_common(path: Path, wanted: set[str]) -> tuple[dict[str, dict[str, object]], int, int]:
    found: dict[str, dict[str, object]] = {}
    scanned = 0
    malformed = 0
    for row in iter_csv(path):
        scanned += 1
        if not row or row[0] not in wanted:
            continue
        try:
            parsed = parse_common(row)
            found[str(parsed["common_id"])] = parsed
        except Exception:
            malformed += 1
        if found.keys() >= wanted:
            break
    return found, scanned, malformed


def _write_parquet(records: list[dict[str, object]], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        raise RuntimeError(f"refusing to write empty split: {path.parent.name}")
    pq.write_table(pa.Table.from_pylist(records), path, compression="zstd")


def _clear_old_parquet(output_dir: Path) -> list[str]:
    """Remove only generated Parquet parts from the three managed splits."""
    removed: list[str] = []
    for split in ("train", "val", "public_test"):
        split_dir = output_dir / split
        if not split_dir.is_dir():
            continue
        for path in sorted(split_dir.glob("*.parquet")):
            path.unlink()
            removed.append(str(path))
    return removed


def prepare_sample(
    data_dir: Path,
    output_dir: Path,
    train_rows: int = 5000,
    public_test_rows: int = 2000,
    validation_fraction: float = 0.1,
    sampling_seed: str | int | None = None,
) -> dict[str, Any]:
    if train_rows <= 0 or public_test_rows <= 0:
        raise ValueError("row limits must be positive")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    try:
        import pyarrow  # noqa: F401
    except ImportError as error:
        raise RuntimeError("pyarrow is required; activate .venv or install testing/requirements.txt") from error

    resolved_seed = str(sampling_seed if sampling_seed is not None else secrets.randbits(64))
    train_skeleton_path, train_common_path = paths_for(data_dir, "train")
    test_skeleton_path, test_common_path = paths_for(data_dir, "test")
    raw_train, train_errors = _read_skeleton(train_skeleton_path, train_rows)
    raw_test, test_errors = _read_skeleton(test_skeleton_path, public_test_rows)

    wanted_train = {str(row["common_id"]) for row in raw_train}
    wanted_test = {str(row["common_id"]) for row in raw_test}
    common_train, train_scanned, malformed_train_common = _find_common(train_common_path, wanted_train)
    common_test, test_scanned, malformed_test_common = _find_common(test_common_path, wanted_test)
    missing_train = sorted(wanted_train - common_train.keys())
    missing_test = sorted(wanted_test - common_test.keys())
    if missing_train or missing_test:
        raise RuntimeError(f"missing common IDs: train={missing_train[:5]}, public_test={missing_test[:5]}")

    splits: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "public_test": []}
    user_sets: dict[str, set[str]] = {key: set() for key in splits}
    labels: dict[str, Counter[str]] = {key: Counter() for key in splits}
    for skeleton in raw_train:
        record = _flat_record(skeleton, common_train[str(skeleton["common_id"])])
        user_key = str(record.get("101") or record["common_id"])
        split = (
            "val"
            if stable_fraction(user_key, f"split:{resolved_seed}") < validation_fraction
            else "train"
        )
        splits[split].append(record)
        user_sets[split].add(user_key)
        labels[split][f"click={record['click']},conversion={record['conversion']}"] += 1
    for skeleton in raw_test:
        record = _flat_record(skeleton, common_test[str(skeleton["common_id"])])
        splits["public_test"].append(record)
        user_sets["public_test"].add(str(record.get("101") or record["common_id"]))
        labels["public_test"][f"click={record['click']},conversion={record['conversion']}"] += 1

    overlap = user_sets["train"] & user_sets["val"]
    if overlap:
        raise RuntimeError(f"user leakage detected between train and val: {list(overlap)[:5]}")
    # All source parsing and joins succeeded. Remove previous generated parts
    # immediately before rewriting so stale shards cannot mix with this run.
    removed_parquet = _clear_old_parquet(output_dir)
    for split, records in splits.items():
        _write_parquet(records, output_dir / split / "part-00000.parquet")

    split_manifest = {
        "kind": "sample_run_for_review_not_full_dataset",
        "strategy": "stable_user_hash",
        "user_field": "101",
        "fallback_key": "common_id",
        "train_fraction": 1.0 - validation_fraction,
        "validation_fraction": validation_fraction,
        "hash_algorithm": "blake2b-64",
        "sampling_seed": resolved_seed,
        "split_salt": f"split:{resolved_seed}",
        "selection_note": "review mode reads the first requested valid exposures; seed changes only the user-level train/val assignment",
        "public_test_source": "sample_test",
        "public_test_used_for_training": False,
        "user_overlap_train_val": 0,
    }
    quality = {
        "rows": {key: len(value) for key, value in splits.items()},
        "unique_users": {key: len(value) for key, value in user_sets.items()},
        "labels": {key: dict(value) for key, value in labels.items()},
        "common_rows_scanned": {"train": train_scanned, "public_test": test_scanned},
        "malformed_common_rows": {"train": malformed_train_common, "public_test": malformed_test_common},
        "source_errors": {"train": dict(train_errors), "public_test": dict(test_errors)},
        "association_hit_rate": 1.0,
        "old_parquet_files_removed": len(removed_parquet),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "quality_report.json").write_text(json.dumps(quality, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"output_dir": str(output_dir.resolve()), "split_manifest": split_manifest, "quality": quality}

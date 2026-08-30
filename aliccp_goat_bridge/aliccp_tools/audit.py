"""Memory-bounded audit of processed AliCCP Parquet splits."""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_BATCH_SIZE = 262_144


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _open_disk_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-32768")
    connection.execute(
        "CREATE TABLE users (split TEXT NOT NULL, value TEXT NOT NULL, "
        "PRIMARY KEY (split, value)) WITHOUT ROWID"
    )
    return connection


def _existing_split(root: Path, split: str) -> tuple[str, Path]:
    directory = root / split
    if split == "val" and not directory.is_dir() and (root / "validation").is_dir():
        return "validation", root / "validation"
    return split, directory


def audit_processed(
    root: Path,
    output_path: Path | None = None,
    checksums: bool = False,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict[str, Any]:
    """Audit all splits while bounding RAM by ``batch_size``.

    Exact global sample-id uniqueness and train/validation user overlap are
    checked with a temporary SQLite index on disk. The index is deleted when
    the audit finishes, including when an exception interrupts the run.
    """
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    root = Path(root)
    report: dict[str, Any] = {
        "root": str(root.resolve()),
        "streaming": True,
        "batch_size": batch_size,
        "duplicate_check": "temporary_disk_index",
        "splits": {},
        "errors": [],
        "warnings": [],
    }
    with tempfile.TemporaryDirectory(prefix="aliccp-audit-") as temp_directory:
        connection = _open_disk_index(Path(temp_directory) / "audit.sqlite3")
        try:
            for requested_split in ("train", "val", "public_test"):
                split, directory = _existing_split(root, requested_split)
                files = sorted(directory.glob("*.parquet")) if directory.is_dir() else []
                if not files:
                    report["errors"].append(f"missing parquet files for split: {requested_split}")
                    continue
                dataset = ds.dataset(directory, format="parquet")
                required = {"sample_id", "click", "conversion", "101", "205"}
                missing = sorted(required - set(dataset.schema.names))
                if missing:
                    report["errors"].append(f"{split} missing columns: {missing}")
                    continue

                connection.execute("DROP TABLE IF EXISTS sample_ids")
                connection.execute(
                    "CREATE TABLE sample_ids (value TEXT PRIMARY KEY) WITHOUT ROWID"
                )
                rows = invalid_funnel = null_sample_ids = 0
                label_counts: Counter[str] = Counter()
                connection.execute("BEGIN")
                for batch in dataset.to_batches(
                    columns=["sample_id", "click", "conversion", "101"],
                    batch_size=batch_size,
                ):
                    sample_ids = batch.column(0).to_pylist()
                    users = batch.column(3).to_pylist()
                    null_sample_ids += sum(value is None for value in sample_ids)
                    connection.executemany(
                        "INSERT OR IGNORE INTO sample_ids(value) VALUES (?)",
                        ((str(value),) for value in sample_ids if value is not None),
                    )
                    connection.executemany(
                        "INSERT OR IGNORE INTO users(split, value) VALUES (?, ?)",
                        ((requested_split, str(value)) for value in users if value is not None),
                    )
                    click = batch.column(1)
                    conversion = batch.column(2)
                    for click_value in (0, 1):
                        for conversion_value in (0, 1):
                            mask = pc.and_(
                                pc.equal(click, click_value),
                                pc.equal(conversion, conversion_value),
                            )
                            count = int(pc.sum(pc.cast(mask, "int64")).as_py())
                            if count:
                                label_counts[
                                    f"click={click_value},conversion={conversion_value}"
                                ] += count
                    invalid_funnel += int(
                        pc.sum(
                            pc.cast(
                                pc.and_(pc.equal(click, 0), pc.equal(conversion, 1)),
                                "int64",
                            )
                        ).as_py()
                    )
                    rows += batch.num_rows
                connection.commit()
                unique_sample_ids = int(
                    connection.execute("SELECT COUNT(*) FROM sample_ids").fetchone()[0]
                )
                unique_users = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM users WHERE split = ?", (requested_split,)
                    ).fetchone()[0]
                )
                duplicate_sample_ids = rows - null_sample_ids - unique_sample_ids
                split_report: dict[str, Any] = {
                    "rows": rows,
                    "unique_sample_ids": unique_sample_ids,
                    "duplicate_sample_ids": duplicate_sample_ids,
                    "null_sample_ids": null_sample_ids,
                    "unique_users": unique_users,
                    "files": len(files),
                    "bytes": sum(path.stat().st_size for path in files),
                    "columns": len(dataset.schema.names),
                    "labels": dict(sorted(label_counts.items())),
                    "invalid_funnel_rows": invalid_funnel,
                }
                if checksums:
                    split_report["sha256"] = {path.name: _sha256(path) for path in files}
                report["splits"][requested_split] = split_report
                if duplicate_sample_ids:
                    report["errors"].append(
                        f"{split} contains {duplicate_sample_ids} duplicate sample_id rows"
                    )
                if null_sample_ids:
                    report["errors"].append(
                        f"{split} contains {null_sample_ids} null sample_id rows"
                    )
                if invalid_funnel:
                    report["errors"].append(
                        f"{split} contains {invalid_funnel} invalid funnel rows"
                    )
                if label_counts.get("click=1,conversion=1", 0) == 0:
                    report["warnings"].append(
                        f"{split} has no conversion-positive rows; CVR AUC may be unavailable"
                    )

            overlap = int(
                connection.execute(
                    "SELECT COUNT(*) FROM users AS train "
                    "INNER JOIN users AS val ON train.value = val.value "
                    "WHERE train.split = 'train' AND val.split = 'val'"
                ).fetchone()[0]
            )
            report["train_val_user_overlap"] = overlap
            if overlap:
                report["errors"].append(f"train/val user leakage: {overlap} users")
        finally:
            connection.close()

    report["status"] = "failed" if report["errors"] else "passed"
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return report

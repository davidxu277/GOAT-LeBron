from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import inspect_file, paths_for, prepare, preview
from .schema import feature, load_schema
from .sample_prepare import prepare_sample
from .audit import audit_processed
from .evaluation import evaluate_predictions
from .results import build_results_table, record_result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect and prepare raw AliCCP files without loading them into RAM")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preview_parser = subparsers.add_parser("preview", help="Print a few decoded rows")
    preview_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    preview_parser.add_argument("--split", choices=("train", "test"), default="train")
    preview_parser.add_argument("--rows", type=int, default=3)

    inspect_parser = subparsers.add_parser("inspect", help="Scan all four raw files")
    inspect_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    inspect_parser.add_argument("--output", type=Path, default=Path("data_processed/inspection.json"))

    prepare_parser = subparsers.add_parser("prepare", help="Join, split and write parquet datasets")
    prepare_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    prepare_parser.add_argument("--output-dir", type=Path, default=Path("data_processed"))
    prepare_parser.add_argument("--shards", type=int, default=256)
    prepare_parser.add_argument("--batch-rows", type=int, default=100000)
    prepare_parser.add_argument("--validation-fraction", type=float, default=0.1)
    prepare_parser.add_argument("--seed", help="Sampling seed; omit to generate a new one")

    sample_parser = subparsers.add_parser("prepare-sample", help="Write a small real train/val/public_test dataset")
    sample_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    sample_parser.add_argument("--output-dir", type=Path, default=Path("data_processed/sample_run"))
    sample_parser.add_argument("--train-rows", type=int, default=5000)
    sample_parser.add_argument("--public-test-rows", type=int, default=2000)
    sample_parser.add_argument("--validation-fraction", type=float, default=0.1)
    sample_parser.add_argument("--seed", help="Split seed; omit to generate a new one")

    schema_parser = subparsers.add_parser("schema", help="Print the full dictionary or one field")
    schema_parser.add_argument("--field", help="Optional field ID, for example 205")

    audit_parser = subparsers.add_parser("audit", help="Validate processed train/val/public_test parquet")
    audit_parser.add_argument("--root", type=Path, required=True)
    audit_parser.add_argument("--output", type=Path, default=Path("results/dataset_audit.json"))
    audit_parser.add_argument("--checksums", action="store_true")
    audit_parser.add_argument("--batch-size", type=int, default=262144)

    evaluate_parser = subparsers.add_parser("evaluate", help="Validate predictions and compute CTR/CVR AUC")
    evaluate_parser.add_argument("--labels", type=Path, required=True)
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--output", type=Path, default=Path("results/metrics.json"))
    evaluate_parser.add_argument("--batch-size", type=int, default=262144)

    record_parser = subparsers.add_parser("record-result", help="Register one experiment result")
    record_parser.add_argument("--metrics", type=Path, required=True)
    record_parser.add_argument("--experiment", required=True)
    record_parser.add_argument("--results-dir", type=Path, default=Path("results/experiments"))
    record_parser.add_argument("--gpu-hours", type=float, default=0.0)
    record_parser.add_argument("--llm-tokens", type=int, default=0)
    record_parser.add_argument("--human-interventions", type=int, default=0)
    record_parser.add_argument("--notes", default="")

    table_parser = subparsers.add_parser("results-table", help="Build CSV/Markdown/JSON final result tables")
    table_parser.add_argument("--results-dir", type=Path, default=Path("results/experiments"))
    table_parser.add_argument("--output-dir", type=Path, default=Path("results/final"))
    table_parser.add_argument("--baseline-ctr", type=float)
    table_parser.add_argument("--baseline-cvr", type=float)

    args = parser.parse_args()
    if args.command == "preview":
        result = preview(args.data_dir, args.split, args.rows)
    elif args.command == "inspect":
        result = {}
        for split in ("train", "test"):
            skeleton, common = paths_for(args.data_dir, split)
            result[split] = {
                "skeleton": inspect_file(skeleton, "skeleton"),
                "common": inspect_file(common, "common"),
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    elif args.command == "prepare":
        result = prepare(
            args.data_dir, args.output_dir, args.shards, args.batch_rows,
            args.validation_fraction, args.seed,
        )
    elif args.command == "prepare-sample":
        result = prepare_sample(
            args.data_dir, args.output_dir, args.train_rows,
            args.public_test_rows, args.validation_fraction, args.seed,
        )
    elif args.command == "schema":
        result = feature(args.field) if args.field else load_schema()
    elif args.command == "audit":
        result = audit_processed(args.root, args.output, args.checksums, args.batch_size)
    elif args.command == "evaluate":
        result = evaluate_predictions(
            args.labels, args.predictions, args.output, args.batch_size,
        )
    elif args.command == "record-result":
        result = record_result(
            args.metrics, args.experiment, args.results_dir, args.gpu_hours,
            args.llm_tokens, args.human_interventions, args.notes,
        )
    else:
        result = build_results_table(
            args.results_dir, args.output_dir, args.baseline_ctr, args.baseline_cvr,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

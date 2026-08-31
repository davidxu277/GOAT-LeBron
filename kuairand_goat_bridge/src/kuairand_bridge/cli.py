from __future__ import annotations

import argparse
import csv
import json
import pathlib

from .dataset import load_dataset
from .evaluator import evaluate_predictions
from .runner import run_trainer


def main(argv=None):
    p = argparse.ArgumentParser(description="KuaiRand-Pure ↔ GOAT-LeBron interface")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("preflight", "template"):
        q = sub.add_parser(name)
        q.add_argument("--data-dir", required=True)
    q = sub.add_parser("evaluate")
    q.add_argument("--data-dir", required=True); q.add_argument("--predictions", required=True)
    q.add_argument("--split", choices=("valid", "test"), default="valid")
    q.add_argument("--output-dir", default="output")
    q = sub.add_parser("run-trainer")
    q.add_argument("--data-dir", required=True); q.add_argument("--trainer", required=True)
    q.add_argument("--output-dir", default="output"); q.add_argument("--seed", type=int, default=0)
    q.add_argument("--make-test", action="store_true")
    q = sub.add_parser("goat-run")
    q.add_argument("--config", required=True)
    q.add_argument("--dry-run", action="store_true",
                   help="只检查路径和参数，不调用LLM或训练")
    a = p.parse_args(argv)

    if a.command == "run-trainer":
        print(json.dumps(run_trainer(a.data_dir, a.trainer, a.output_dir, a.seed, a.make_test),
                         ensure_ascii=False, indent=2)); return
    if a.command == "goat-run":
        from .goat_run import run
        print(json.dumps(run(a.config, a.dry_run), ensure_ascii=False, indent=2)); return
    dataset = load_dataset(a.data_dir)
    if a.command == "preflight":
        report = {"status": "ok", "data_dir": str(dataset.data_dir),
                  "rows": {x: len(dataset.split(x)) for x in ("train", "valid", "test")},
                  "test_labels_exposed": False}
        expected = {"train": 1141112, "valid": 124909, "test": 170588}
        report["official_row_counts_match"] = report["rows"] == expected
        print(json.dumps(report, ensure_ascii=False, indent=2)); return
    if a.command == "template":
        path = pathlib.Path("prediction_template_valid.csv")
        with path.open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["score"])
            w.writerows([[0.0] for _ in range(len(dataset.valid))])
        print(path.resolve()); return
    print(json.dumps(evaluate_predictions(dataset, a.predictions, a.split, a.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import multiprocessing as mp

    mp.freeze_support()
    main()

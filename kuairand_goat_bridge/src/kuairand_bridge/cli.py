from __future__ import annotations

import argparse
import csv
import json
import pathlib
import yaml

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
    q.add_argument("--trainer-config",
                   help="传给 Trainer.fit(config=...) 的 YAML 配置文件")
    q.add_argument("--make-test", action="store_true")
    q = sub.add_parser("noise", help="量噪声带：同配置同数据，只换训练种子跑 N 次")
    q.add_argument("--config", required=True, help="任务配置（跟 goat-run 用同一份）")
    q.add_argument("--fidelity", default="小份",
                   help="在哪一档量。必须跟 goat-run 起步那一档一致，否则那一场不会用它")
    q.add_argument("--seeds", type=int, default=5, help="换几个训练种子（至少 3）")
    q = sub.add_parser("goat-run")
    q.add_argument("--config", required=True)
    q.add_argument("--dry-run", action="store_true",
                   help="只检查路径和参数，不调用LLM或训练")
    a = p.parse_args(argv)

    if a.command == "run-trainer":
        trainer_config = {}
        if a.trainer_config:
            config_path = pathlib.Path(a.trainer_config).expanduser().resolve()
            trainer_config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if not isinstance(trainer_config, dict):
                raise ValueError("--trainer-config 必须指向顶层为对象的 YAML 文件")
        print(json.dumps(run_trainer(a.data_dir, a.trainer, a.output_dir, a.seed, a.make_test,
                                     trainer_config=trainer_config),
                         ensure_ascii=False, indent=2)); return
    if a.command == "noise":
        from .goat_run import validate_task
        from .noise import measure_noise, save_bands
        task = validate_task(a.config)
        output = pathlib.Path(task["output_dir"])
        bands = measure_noise(
            data_dir=task["data_dir"], trainer_path=task["trainer"],
            output_dir=output / "noise" / a.fidelity, fidelity=a.fidelity,
            train_seeds=list(range(1, a.seeds + 1)),
            # 数据子集跟 goat-run 那一场一模一样，只有训练种子在变
            sample_seed=task["seed"], trainer_config=task.get("trainer_config") or {})
        path = save_bands(output / "logs", bands)
        print(json.dumps({k: bands[k] for k in ("保真度", "分指标噪声带", "单指标噪声带", "统计")},
                         ensure_ascii=False, indent=2))
        print(f"已写入 {path}（goat-run 会自动读它）"); return
    if a.command == "goat-run":
        from .goat_run import run
        print(json.dumps(run(a.config, a.dry_run), ensure_ascii=False, indent=2)); return
    if a.command == "preflight":
        # preflight 要核对三份官方切分的行数，因此读取 test 的非标签字段；
        # load_dataset 会强制 expose_test_labels=False，标签仍然锁定。
        dataset = load_dataset(a.data_dir, include_test=True)
        report = {"status": "ok", "data_dir": str(dataset.data_dir),
                  "rows": {x: len(dataset.split(x)) for x in ("train", "valid", "test")},
                  "test_labels_exposed": False}
        expected = {"train": 1141112, "valid": 124909, "test": 170588}
        report["official_row_counts_match"] = report["rows"] == expected
        print(json.dumps(report, ensure_ascii=False, indent=2)); return
    dataset = load_dataset(a.data_dir)
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

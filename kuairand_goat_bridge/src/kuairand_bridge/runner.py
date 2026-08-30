"""Model plug-in runner used by GOAT-LeBron or a teammate's standalone trainer."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

from .dataset import load_dataset
from .evaluator import evaluate_predictions


def _load_trainer(path):
    path = pathlib.Path(path).resolve()
    spec = importlib.util.spec_from_file_location("teammate_trainer", path)
    if not spec or not spec.loader:
        raise ImportError(f"无法加载 trainer：{path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for fn in ("fit", "predict"):
        if not callable(getattr(mod, fn, None)):
            raise TypeError(f"trainer 必须实现 {fn}()")
    return mod


def run_trainer(data_dir, trainer_path, output_dir="output", seed=0, make_test=False):
    dataset = load_dataset(data_dir)
    trainer = _load_trainer(trainer_path)
    model = trainer.fit(dataset.train, dataset.valid, seed=seed)
    valid_scores = np.asarray(trainer.predict(model, dataset.valid), dtype=float).reshape(-1)
    work = pathlib.Path(output_dir)
    work.mkdir(parents=True, exist_ok=True)
    valid_raw = work / "valid_scores.npy"
    np.save(valid_raw, valid_scores)
    result = {"validation": evaluate_predictions(dataset, valid_raw, "valid", work)}
    if make_test:
        test_scores = np.asarray(trainer.predict(model, dataset.test), dtype=float).reshape(-1)
        test_raw = work / "test_scores.npy"
        np.save(test_raw, test_scores)
        result["test"] = evaluate_predictions(dataset, test_raw, "test", work)
    return result

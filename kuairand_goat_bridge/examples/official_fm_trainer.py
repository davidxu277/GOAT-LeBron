"""Example of the exact two-function contract a teammate model must implement."""

import pathlib
import sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "official_starter_kit"))
import baseline
import data
import evaluate


def fit(train, valid, seed=0):
    # Official encoder needs all split keys; test may be an empty placeholder here.
    enc, dim = data.encode({"train": train.rows, "valid": valid.rows, "test": []})
    xtr, ytr, _ = enc["train"]; xva, yva, uva = enc["valid"]
    model = baseline.FM(dim, k=16, lr=0.001, seed=seed)
    rng = np.random.default_rng(seed)
    best, state, bad = -1.0, None, 0
    for _ in range(40):
        order = rng.permutation(len(ytr))
        for start in range(0, len(order), 8192):
            idx = order[start:start + 8192]
            model.step(xtr[idx], ytr[idx])
        score = evaluate.evaluate(uva, yva, model.predict(xva))["primary"]
        if score > best + 1e-5:
            best, bad = score, 0
            state = (model.V.copy(), model.W.copy(), np.float32(model.b))
        else:
            bad += 1
            if bad >= 4:
                break
    model.V, model.W, model.b = state
    # Retain official encoding artifacts by encoding all real rows on prediction.
    return {"model": model, "train": train.rows, "valid": valid.rows}


def predict(bundle, split):
    # This example is intentionally valid-focused. For final test generation use
    # a teammate trainer that persists its train-only vocabulary/edges in fit().
    if split.name != "valid":
        raise ValueError("示例 FM 只演示 validation；正式模型应在 fit 中保存 encoder 后预测 test")
    enc, _ = data.encode({"train": bundle["train"], "valid": bundle["valid"], "test": []})
    return bundle["model"].predict(enc["valid"][0])

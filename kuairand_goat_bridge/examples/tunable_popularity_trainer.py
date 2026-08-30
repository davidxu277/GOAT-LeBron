"""CPU reference trainer that accepts cumulative GOAT config patches."""

from collections import Counter
import pathlib

import numpy as np
import yaml


CONFIG = {"model": {"prior": 20.0}}


def _merge(dst, src):
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value


def apply_agent_patch(patch, output_dir):
    global CONFIG
    CONFIG = {"model": {"prior": 20.0}}
    for item in patch.get("history") or [patch]:
        if item.get("new_files"):
            raise NotImplementedError(
                "CPU参考Trainer只支持配置实验；模型代码实验请换成同学的Trainer")
        raw = item.get("config_patch") or ""
        parsed = yaml.safe_load(raw) if isinstance(raw, str) else raw
        if parsed:
            if not isinstance(parsed, dict):
                raise ValueError("config_patch必须是YAML对象")
            unknown = set(parsed) - {"model", "features", "train"}
            if unknown:
                raise ValueError(f"不允许修改这些配置根节点：{sorted(unknown)}")
            _merge(CONFIG, parsed)
    prior = float(CONFIG.get("model", {}).get("prior", 20.0))
    if not 1.0 <= prior <= 200.0:
        raise ValueError("model.prior必须在1到200之间")
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "effective_config.yaml").write_text(
        yaml.safe_dump(CONFIG, allow_unicode=True, sort_keys=False), encoding="utf-8")


def fit(train, valid, seed=0):
    del valid, seed
    positives, impressions = Counter(), Counter()
    for row in train.rows:
        video_id, label = row[2], row[6]
        impressions[video_id] += 1
        positives[video_id] += label
    global_mean = sum(positives.values()) / max(1, sum(impressions.values()))
    return {"positives": positives, "impressions": impressions,
            "global_mean": global_mean,
            "prior": float(CONFIG.get("model", {}).get("prior", 20.0))}


def predict(model, split):
    def score(video_id):
        count = model["impressions"][video_id]
        if not count:
            return model["global_mean"]
        return ((model["positives"][video_id] + model["prior"] * model["global_mean"])
                / (count + model["prior"]))
    return np.asarray([score(row[2]) for row in split.rows], dtype=float)

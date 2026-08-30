"""Fast end-to-end example: train-only item popularity, valid and test capable."""

from collections import Counter
import numpy as np


def fit(train, valid, seed=0):
    del valid, seed
    positives, impressions = Counter(), Counter()
    for row in train.rows:
        video_id, label = row[2], row[6]
        impressions[video_id] += 1
        positives[video_id] += label
    global_mean = sum(positives.values()) / max(1, sum(impressions.values()))
    return {"positives": positives, "impressions": impressions,
            "global_mean": global_mean, "prior": 20.0}


def predict(model, split):
    def score(video_id):
        n = model["impressions"][video_id]
        if not n:
            return model["global_mean"]
        return ((model["positives"][video_id] + model["prior"] * model["global_mean"])
                / (n + model["prior"]))
    return np.asarray([score(row[2]) for row in split.rows], dtype=float)

"""GOAT Trainer —— 把 AliCCP 那套「Agent 自己写代码」的能力接到 KuaiRand 上。

## 为什么要有这个文件

`official_fm_trainer.py` 只接受 6 个超参数，Agent 能做的事到调参为止 ——
而赛题写着「奖励那些**超越简单基线调参**的 Agent」。

但"让 Agent 自己写特征和模型"这套能力**我们早就建好了**，在 `harness/` 里：
`_load_op_class_by` 的路径守卫、`load_feature_ops` 的 R2 纪律、
`train_deep` 的 embedding 表 + epoch 循环 + TrainOp 回调 + 最佳权重回滚。
它们只是没被这条新路径用到 —— 换个数据集就把整套框架扔掉，没有道理。

这份 trainer 做的就是一件事：**把 KuaiRand 的数据翻译成那套框架吃的形状**，
然后原样调用它们。翻译层很薄，能力全是复用的。

## 接回来了哪些

    FeatureOp   Agent 写 modules/features/*.py，加特征          ← load_feature_ops
    ModelOp     Agent 写 modules/models/*.py，定义网络结构      ← load_model_op
    TrainOp     Agent 写 modules/train/*.py，早停/SWA/调度      ← load_train_ops
    R5 守卫     只许写 modules/ 下，路径带 .. 一律拒绝
    R2 纪律     所有统计量只在 train 上 fit，valid 只做 transform
    护栏        超参数越界当场报错，不静默夹回

## 任务差异只有两处，用 train_deep 的两个钩子解决

    task_loss    单标签 BCE（KuaiRand 只有 long_view，没有点击/转化双塔）
    task_metric  用户内 GAUC + nDCG@5（官方口径，直接调 starter kit 的 evaluate）
"""

from __future__ import annotations

import copy
import atexit
import pathlib
import sys
from typing import Any

import numpy as np
import pandas as pd
import yaml

BRIDGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
GOAT_ROOT = BRIDGE_ROOT.parent
STARTER_KIT = BRIDGE_ROOT / "official_starter_kit"

for path in (str(GOAT_ROOT), str(STARTER_KIT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import data as official_data                                  # noqa: E402
import evaluate as official_evaluate                           # noqa: E402

from harness.deep import train_deep, predict_deep              # noqa: E402
from harness.executor import (                                 # noqa: E402
    apply_feature_ops,
    load_feature_ops,
)

# 行元组的下标，跟 official_starter_kit/data.py 保持一致
COLUMNS = ["date", "user_id", "video_id", "author_id", "tab", "duration_ms", "label"]

# 进模型的基础字段。Agent 写的零件会往后面追加新列，自动进特征表。
BASE_FIELDS = ["user_id", "video_id", "author_id", "tab", "duration_bucket"]

AGENT_CONFIG_ROOTS = {"features", "model", "train"}

_AGENT_OVERRIDES: dict[str, Any] = {}
_GENERATED_FILES: set[pathlib.Path] = set()


# ────────────────────────── 数据翻译层 ──────────────────────────


def rows_to_frame(rows, duration_edges: np.ndarray | None = None
                  ) -> tuple[pd.DataFrame, np.ndarray]:
    """行元组 → DataFrame。duration 分桶边界只从 train 拟合（R2）。"""
    df = pd.DataFrame(list(rows), columns=COLUMNS)
    if duration_edges is None:
        duration_edges = official_data._bucket_edges(
            [float(v) for v in df["duration_ms"]])
    df["duration_bucket"] = np.searchsorted(
        duration_edges, df["duration_ms"].astype(float)).astype("int64")
    return df, duration_edges


# ────────────────────────── 任务钩子 ──────────────────────────


def task_loss(out: dict, batch: pd.DataFrame, torch):
    """单标签 BCE —— KuaiRand 只有 long_view，没有点击/转化两个塔。

    模型仍按双塔接口输出 ctr/cvr；这里只监督 ctr 那一路，
    这样 AliCCP 写的模型零件不用改就能拿过来用。
    """
    label = torch.tensor(batch["label"].to_numpy(), dtype=torch.float32,
                         device=out["ctr"].device)
    pred = out["ctr"].clamp(1e-7, 1 - 1e-7)
    return torch.nn.functional.binary_cross_entropy(pred, label)


def task_metric(op, model, vocab, val: pd.DataFrame) -> dict[str, float]:
    """官方口径的用户内排序指标 —— 直接调 starter kit 的 evaluate，不自己实现。

    自己实现一遍等于给自己造一个"跟官方差一点"的分数，
    到最终提交才发现对不上。
    """
    ctr, _ = predict_deep(op, model, vocab, val)
    m = official_evaluate.evaluate(
        list(val["user_id"]), list(val["label"]), list(ctr))
    # 第一个键是主指标（train_deep 拿它挑最佳轮次）
    return {"primary": round(float(m["primary"]), 6),
            "GAUC": round(float(m["GAUC"]), 6),
            "nDCG@5": round(float(m["nDCG@5"]), 6)}


# ────────────────────────── Agent 补丁 ──────────────────────────


def _merge(dst: dict, src: dict) -> None:
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = value


def apply_agent_patch(patch, output_dir) -> None:
    """接收 Agent 的补丁 —— **包括它自己写的代码**。

    补丁累积重放：每轮从初始配置重叠一遍 history，任何一轮都能从日志复现，
    也不会因为中间某轮被回滚而留下脏状态。

    new_files 落进 modules/ 下（R5 由 executor 的路径守卫强制），
    然后由 load_feature_ops / load_model_op 按配置里的 impl 加载。
    """
    global _AGENT_OVERRIDES
    _AGENT_OVERRIDES = {}

    for item in patch.get("history") or [patch]:
        for f in item.get("new_files") or []:
            rel = str(f["path"]).replace("\\", "/")
            if not rel.startswith("modules/") or ".." in rel.split("/"):
                raise ValueError(f"非法写入路径：{rel}（只能写 modules/ 下，R5）")
            target = GOAT_ROOT / rel
            # R5 只允许新建零件。绝不覆盖仓库里已经存在的用户代码；同一轮
            # history 对刚由本次运行创建的文件做后续版本覆盖则是合法的。
            if target.exists() and target not in _GENERATED_FILES:
                raise FileExistsError(f"Agent 想覆盖已有文件：{rel}；只允许新建零件")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f["content"], encoding="utf-8")
            _GENERATED_FILES.add(target)

        raw = item.get("config_patch") or ""
        parsed = yaml.safe_load(raw) if isinstance(raw, str) else raw
        if not parsed:
            continue
        if not isinstance(parsed, dict):
            raise ValueError("config_patch 必须是 YAML 键值对")
        unknown = set(parsed) - AGENT_CONFIG_ROOTS
        if unknown:
            raise ValueError(
                f"config_patch 想改 {sorted(unknown)}，"
                f"只准动 {sorted(AGENT_CONFIG_ROOTS)} 这三棵子树。")
        _merge(_AGENT_OVERRIDES, parsed)

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "agent_overrides.yaml").write_text(
        yaml.safe_dump(_AGENT_OVERRIDES, allow_unicode=True, sort_keys=False),
        encoding="utf-8")


def cleanup_agent_patch() -> None:
    """删除本次 Trainer 进程新建的零件，避免实验文件被误 push。

    只处理 ``apply_agent_patch`` 确认由本进程创建的文件；已有仓库文件永不删除。
    每轮的 patch history 会在下一轮重新生成所需零件，因此清理不影响复现。
    """
    while _GENERATED_FILES:
        _GENERATED_FILES.pop().unlink(missing_ok=True)


atexit.register(cleanup_agent_patch)


# ────────────────────────── Bridge 契约 ──────────────────────────


def fit(train, valid, seed: int = 0, config: dict[str, Any] | None = None):
    """训练 —— 数据翻译一层，能力全部复用 harness/。"""
    cfg = copy.deepcopy(dict(config or {}))
    _merge(cfg, _AGENT_OVERRIDES)
    cfg.setdefault("model", {}).setdefault("name", "goat_mlp")
    cfg["model"].setdefault("impl", "modules/models/mlp.py")

    train_df, edges = rows_to_frame(train.rows)
    valid_df, _ = rows_to_frame(valid.rows, edges)      # 边界沿用 train 的（R2）

    # ① Agent 写的加特征零件 —— fit 只看训练集，新列自动进特征表
    ops = load_feature_ops(cfg)
    train_df, (valid_df,), 新列 = apply_feature_ops(ops, train_df, [valid_df])
    fields = BASE_FIELDS + [c for c in 新列 if c in valid_df.columns]

    # ② 深度训练循环 —— embedding 表 / epoch / TrainOp 回调 / 最佳权重回滚
    op, model, 记录 = train_deep(
        cfg, train_df, valid_df, fields, seed,
        task_loss=task_loss, task_metric=task_metric)

    return {"op": op, "model": model, "vocab": 记录.pop("_vocab"),
            "fields": fields, "ops": ops, "duration_edges": edges,
            "config": cfg, "训练记录": 记录,
            "装上的零件": [name for name, _ in ops]}


def predict(model_bundle, split) -> np.ndarray:
    """对任意 split 出预测 —— 走跟训练**同一条**加工路径。"""
    df, _ = rows_to_frame(split.rows, model_bundle["duration_edges"])
    for _, op in model_bundle["ops"]:
        df = op.transform(df)
    ctr, _ = predict_deep(model_bundle["op"], model_bundle["model"],
                          model_bundle["vocab"], df)
    scores = np.asarray(ctr, dtype=float).reshape(-1)
    if len(scores) != len(split):
        raise ValueError("预测行数与目标 split 不一致")
    if not np.isfinite(scores).all():
        raise ValueError("预测里出现 NaN 或 Inf")
    return scores

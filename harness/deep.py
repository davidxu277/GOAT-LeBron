"""深度模型训练路径 —— 让 ModelOp / TrainOp 那两个接口真的能用起来。

`modules/base.py` 里 ModelOp（改模型）和 TrainOp（改训练过程）的接口早就定义好了，
但一直没有任何代码去加载和运行它们 —— 26 张卡里 21 张卡在这上面：
模型类的卡本身就是神经网络；损失函数类的卡只有在梯度下降里才存在；
训练策略类的卡（早停、权重平均）是挂在"每轮结束"这个时点的回调，
而 LightGBM 那条路一次 `.fit()` 就完事，根本没有"轮"可挂。

这份文件补的就是那台机器。**界线划在这里**（CLAUDE.md R5）：

    人写（这里）      建 embedding 表、跑 epoch 循环、算损失、回调 TrainOp
                     —— 这是"考场"，是跑实验的机器
    Agent 写（modules/） 网络长什么样、两个塔怎么组合、损失怎么算
                     —— 这是"考生"，是一次具体的实验

所以这里刻意**不实现任何一个具体模型**。想训 ESMM，就得有人（Agent）
在 modules/models/ 下写出 ESMM —— 那是它的活，不是我们替它写。
"""

from __future__ import annotations

import copy
import pathlib
from typing import Any

import numpy as np
import pandas as pd

from agent.events import emit

# 训练超参数的白名单与护栏，跟 LightGBM 那边同一套思路：
# 区间是护栏不是建议 —— epochs: 500 能把一晚上的算力烧光，而 Agent 自己看不出来。
DEEP_PARAMS = {
    "epochs":        (1, 50, 5),
    "batch_size":    (64, 65536, 4096),
    "learning_rate": (1e-5, 0.1, 1e-3),
    "embed_dim":     (4, 128, 16),
    "weight_decay":  (0.0, 0.1, 1e-5),
    "predict_batch_size": (64, 262144, 16384),
}

OOV = 0          # 训练集里没见过的 ID 一律落到 0 号槽位


def deep_kwargs(cfg: dict[str, Any] | None) -> dict[str, Any]:
    """把配置里的训练超参数夹回合理区间。纯函数，方便离线测。"""
    cfg = cfg or {}
    out: dict[str, Any] = {}
    for key, (lo, hi, default) in DEEP_PARAMS.items():
        raw = cfg.get(key, default)
        try:
            value = type(default)(raw)
        except (TypeError, ValueError):
            value = default                      # 写成 "多跑几轮" 这种，退回默认
        out[key] = min(hi, max(lo, value))
    return out


def select_device(torch, requested: str) -> Any:
    """按配置选择训练设备；auto 优先 CUDA，不可用时安全退回 CPU。"""
    name = str(requested or "auto").strip().lower()
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError("model.deep.device 只能是 auto / cpu / cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("配置要求使用 CUDA，但 PyTorch 没检测到可用的 NVIDIA GPU")
    return torch.device("cuda" if name == "cuda" or (
        name == "auto" and torch.cuda.is_available()) else "cpu")


class Vocab:
    """把高基数 ID 映射成连续下标 —— embedding 表要靠它开多大。

    ⚠️ **只在训练集上建**（R2）。验证集里没见过的 ID 落到 OOV 槽，
    而不是给它一个新编号 —— 后者等于偷看了验证集里有哪些 ID。
    """

    def __init__(self) -> None:
        self.maps: dict[str, dict[Any, int]] = {}

    def fit(self, df: pd.DataFrame, fields: list[str]) -> "Vocab":
        for f in fields:
            # 0 号留给 OOV，所以从 1 开始编
            self.maps[f] = {v: i + 1 for i, v in enumerate(df[f].dropna().unique())}
        return self

    def sizes(self) -> dict[str, int]:
        return {f: len(m) + 1 for f, m in self.maps.items()}

    def encode(self, df: pd.DataFrame) -> np.ndarray:
        # executor 会把离散特征转成 pandas category。直接在 category 上 map 后
        # fillna(OOV) 会报“Cannot setitem on a Categorical with a new category (0)”；
        # 先转 object，映射结果就是普通数值 Series，缺失值/未见 ID 才能安全落 OOV。
        cols = [df[f].astype("object").map(self.maps[f]).fillna(OOV)
                .astype("int64").to_numpy() for f in self.maps]
        return np.stack(cols, axis=1) if cols else np.zeros((len(df), 0), dtype="int64")


def feature_spec(vocab: Vocab, embed_dim: int) -> dict[str, Any]:
    """交给 ModelOp.build 的特征规格：每个字段的基数 + embedding 维度。"""
    return {
        "fields": list(vocab.maps),
        "cardinality": vocab.sizes(),
        "embed_dim": embed_dim,
    }


def _default_loss(out: dict[str, Any], click, conv, torch) -> Any:
    """没有自定义损失时的默认口径 —— 跟 LightGBM 那条路语义一致，分数才可比。

        点击塔：全部曝光上的 BCE
        购买塔：**只在点击过的行上**算 BCE（P(购买|点击) 的定义）

    想改成全曝光空间建模（ESMM 那套）的零件，自己实现 `loss` 方法覆盖它 ——
    那正是「损失函数」类卡片存在的意义。
    """
    bce = torch.nn.functional.binary_cross_entropy
    eps = 1e-7
    ctr = out["ctr"].clamp(eps, 1 - eps)
    loss = bce(ctr, click)
    mask = click > 0.5
    if mask.any():
        cvr = out["cvr"].clamp(eps, 1 - eps)
        loss = loss + bce(cvr[mask], conv[mask])
    return loss


def load_model_op(config: dict[str, Any]) -> Any:
    """按 model.impl 指路加载模型零件。跟 FeatureOp 同一套规矩。"""
    from .executor import _load_op_class_by

    impl = (config.get("model") or {}).get("impl")
    if not impl:
        raise ValueError(
            "配置里 model.name 不是 lightgbm，却没写 model.impl —— "
            "不知道该加载哪个模型零件。config_patch 里补一行 "
            "impl: modules/models/xxx.py（零件要实现 modules/base.py 的 ModelOp）")
    return _load_op_class_by(impl, ("build", "predict"), "ModelOp")(config)


def load_train_ops(config: dict[str, Any]) -> list[tuple[str, Any]]:
    """按 train.<名字>.enabled + impl 加载训练过程零件（早停、SWA…）。"""
    from .executor import _load_op_class_by

    ops = []
    for name, block in (config.get("train") or {}).items():
        if not isinstance(block, dict) or not block.get("enabled"):
            continue
        impl = block.get("impl")
        if not impl:
            continue                 # 没指路的当没开 —— 训练策略不像模型那样非有不可
        ops.append((name, _load_op_class_by(impl, ("on_epoch_end",), "TrainOp")(config)))
    return ops


def train_deep(config: dict[str, Any], train: pd.DataFrame, val: pd.DataFrame,
               features: list[str], seed: int,
               task_loss: Any = None, task_metric: Any = None,
               ) -> tuple[Any, Any, dict[str, Any]]:
    """训一个深度模型，返回 (模型零件, 训好的模型, 训练过程记录)。

    整个循环归我们管，模型长什么样归零件管 —— 这就是考场和考生的分界。

    task_loss / task_metric 让这套循环脱离具体任务：
        task_loss(out, batch_df, torch)          -> 标量张量
        task_metric(op, model, vocab, val_df)    -> {"指标名": 值, ...}（第一个是主指标）
    不给就用默认的双塔口径，行为跟以前完全一致。
    换任务（比如 KuaiRand 的用户内排序）只需换这两个函数 ——
    ID 词表、embedding 表、epoch 循环、TrainOp 回调、最佳权重回滚，全部照用。
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    deep_cfg = (config.get("model") or {}).get("deep") or {}
    kw = deep_kwargs(deep_cfg)
    device = select_device(torch, deep_cfg.get("device", "auto"))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    op = load_model_op(config)
    train_ops = load_train_ops(config)

    vocab = Vocab().fit(train, features)         # 只在训练集上建（R2）
    spec = feature_spec(vocab, kw["embed_dim"])
    model = op.build(spec)
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"model.impl 的 build() 要返回一个 torch.nn.Module，"
                        f"拿到的是 {type(model).__name__}")

    model = model.to(device)
    # AliCCP 很大：完整训练张量留在 CPU，每个 batch 才送入显卡，避免 8GB 显存 OOM。
    x = torch.tensor(vocab.encode(train), dtype=torch.long)
    # 双塔标签只有默认损失才用得上。给了任务级损失就跳过 ——
    # 别的任务（KuaiRand 只有一个 long_view）根本没有这两列。
    if task_loss is None:
        click = torch.tensor(train["click"].to_numpy(), dtype=torch.float32)
        conv = torch.tensor(train["conversion"].to_numpy(), dtype=torch.float32)
    else:
        click = conv = torch.zeros(len(train), dtype=torch.float32)
    model._goat_predict_batch_size = kw["predict_batch_size"]
    opt = torch.optim.Adam(model.parameters(), lr=kw["learning_rate"],
                           weight_decay=kw["weight_decay"])
    loss_fn = getattr(op, "loss", None)

    device_name = (torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU")
    emit("phase", name="训练设备", detail=f"{device.type} · {device_name}")
    context = {"config": config, "seed": seed, "epochs": kw["epochs"],
               "features": features, "spec": spec, "device": str(device)}
    for _, top in train_ops:
        if hasattr(top, "on_train_begin"):
            top.on_train_begin(context)

    best = {"auc": float("-inf"), "epoch": 0, "weights": None}
    history: list[dict[str, float]] = []
    n = len(x)

    for epoch in range(1, kw["epochs"] + 1):
        model.train()
        order = torch.randperm(n)
        总损失 = 0.0
        for start in range(0, n, kw["batch_size"]):
            idx = order[start:start + kw["batch_size"]]
            batch_x = x[idx].to(device, non_blocking=True)
            batch_click = click[idx].to(device, non_blocking=True)
            batch_conv = conv[idx].to(device, non_blocking=True)
            out = model(batch_x)
            if task_loss is not None:                 # 调用方给的任务级损失
                loss = task_loss(out, train.iloc[idx.numpy()], torch)
            elif loss_fn is not None:                 # 零件自带的损失（ESMM 那类）
                loss = loss_fn(out, batch_click, batch_conv)
            else:
                loss = _default_loss(out, batch_click, batch_conv, torch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            总损失 += float(loss.detach().cpu()) * len(idx)

        metrics = (task_metric(op, model, vocab, val) if task_metric is not None
                   else _eval_epoch(op, model, vocab, val, torch,
                                    batch_size=kw["predict_batch_size"]))
        metrics["loss"] = round(总损失 / max(1, n), 6)
        # 评分函数返回的第一个指标就是主指标 —— 换任务只换评分函数，这里不用动
        主指标 = next(k for k in metrics if k != "loss")
        history.append({"轮": epoch, **metrics})
        emit("phase", name=f"第 {epoch} 轮",
             detail=f"loss {metrics['loss']:.4f} · {主指标} {metrics[主指标]:.4f}")

        # 验证集最好的那一版权重留着 —— 这是 epoch 层面的"最佳 checkpoint"
        if metrics[主指标] > best["auc"]:
            best = {"auc": metrics[主指标], "epoch": epoch,
                    "weights": copy.deepcopy(model.state_dict())}

        # TrainOp 的回调点：早停、学习率调度、权重平均都挂在这里
        if any(top.on_epoch_end(epoch, metrics, model) for _, top in train_ops):
            emit("phase", name="早停", detail=f"第 {epoch} 轮被训练策略叫停")
            break

    if best["weights"] is not None:
        model.load_state_dict(best["weights"])
    for _, top in train_ops:
        if hasattr(top, "on_train_end"):
            top.on_train_end(model)

    peak_mb = (round(torch.cuda.max_memory_allocated(device) / 1024 ** 2, 1)
               if device.type == "cuda" else 0.0)
    return op, model, {
        "训练轮数": len(history), "最佳轮次": best["epoch"],
        "每轮": history, "超参数": kw,
        "训练设备": device.type,
        "GPU名称": device_name if device.type == "cuda" else "",
        "GPU峰值显存_MB": peak_mb,
        "装上的训练零件": [name for name, _ in train_ops],
        "_vocab": vocab,
    }


def _eval_epoch(op: Any, model: Any, vocab: Vocab, val: pd.DataFrame, torch,
                batch_size: int) -> dict[str, float]:
    """每轮结束在验证集上算一次分 —— TrainOp 拿它决定要不要停（R3：绝不碰锁定集）。"""
    from sklearn.metrics import roc_auc_score

    ctr, cvr = predict_deep(op, model, vocab, val, torch, batch_size=batch_size)
    out = {"点击分": 0.5, "购买分": 0.5}
    if val["click"].nunique() > 1:
        out["点击分"] = float(roc_auc_score(val["click"], ctr))
    clicked = val["click"] == 1
    if clicked.any() and val.loc[clicked, "conversion"].nunique() > 1:
        out["购买分"] = float(roc_auc_score(val.loc[clicked, "conversion"], cvr[clicked.to_numpy()]))
    return {k: round(v, 6) for k, v in out.items()}


def predict_deep(op: Any, model: Any, vocab: Vocab, df: pd.DataFrame, torch=None,
                 batch_size: int | None = None):
    """让零件出预测，并把结果整成两个 numpy 数组。"""
    if torch is None:
        import torch
    model.eval()
    device = next(model.parameters()).device
    size = int(batch_size or getattr(model, "_goat_predict_batch_size", len(df) or 1))
    ctr_parts, cvr_parts = [], []
    with torch.no_grad():
        encoded = vocab.encode(df)
        for start in range(0, len(encoded), size):
            x = torch.tensor(encoded[start:start + size], dtype=torch.long,
                             device=device)
            got = op.predict(model, x)
            ctr_parts.append(got["ctr"].detach().cpu())
            cvr_parts.append(got["cvr"].detach().cpu())
    ctr = torch.cat(ctr_parts) if ctr_parts else torch.empty(0)
    cvr = torch.cat(cvr_parts) if cvr_parts else torch.empty(0)
    to_np = lambda t: (t.detach().cpu().numpy() if hasattr(t, "detach") else np.asarray(t))
    return to_np(ctr).reshape(-1), to_np(cvr).reshape(-1)

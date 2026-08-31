"""早停 —— 训练过程类零件的范文。

给工兵看的：这是"改训练过程"这一类零件该长什么样。
另外两类的范文在 modules/features/（加特征）和 modules/models/（改模型）。

对应药方卡：knowledge/cards/early_stopping.yaml

⚠️ 官方 baseline 在这件事上有个洞：它拿 test 做早停
   （见 docs/baseline笔记.md 洞 2）。我们绝不能照抄 ——
   早停只看开发集，锁定集全程不碰（R3、R9）。

这份代码刻意不依赖任何框架：只要模型对象有 state_dict() / load_state_dict()
就能用（torch 天然满足）。工兵写别的训练零件时可以照这个路子。
"""

from __future__ import annotations

import copy
from typing import Any


class EarlyStopping:
    """连续若干轮没有变好就停下，并把权重回滚到最好的那一轮。

    配置项（全部从 config 读，不许写死 —— R7）：
        train.early_stopping.monitor    盯哪个指标——只能是训练循环真正产出的
                                         「点击分」「购买分」或 "loss"，写别的名字
                                         （比如常见的 "mean_auc"/"cvr_auc"）
                                         会在 on_epoch_end 里直接 KeyError
        train.early_stopping.patience   连续多少轮没变好就停
        train.early_stopping.min_delta  涨多少才算"变好"（建议和 R11 门槛对齐）
        train.early_stopping.mode       "max"（AUC 越大越好）或 "min"（loss 越小越好）
    """

    def __init__(self, config: dict[str, Any]):
        cfg = config["train"]["early_stopping"]
        self.monitor: str = cfg["monitor"]
        self.patience: int = cfg["patience"]
        self.min_delta: float = cfg["min_delta"]
        self.mode: str = cfg["mode"]
        if self.mode not in ("max", "min"):
            raise ValueError(f"mode 只能是 max 或 min，收到 {self.mode!r}")

        self.best_score: float | None = None
        self.best_epoch: int = -1
        self.best_weights: Any = None
        self.rounds_without_improvement: int = 0
        self.stopped_epoch: int | None = None

    # ── TrainOp 接口 ──────────────────────────────────────────────

    def on_train_begin(self, context: dict[str, Any]) -> None:
        self.best_score = None
        self.best_epoch = -1
        self.best_weights = None
        self.rounds_without_improvement = 0
        self.stopped_epoch = None

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, float], model: Any
    ) -> bool:
        """metrics 是本轮在开发集上的指标。返回 True 表示建议停止。"""
        if self.monitor not in metrics:
            raise KeyError(
                f"早停盯的是 {self.monitor!r}，但本轮指标里只有 {sorted(metrics)}。"
                f"检查配置里的 monitor 名字。"
            )
        score = metrics[self.monitor]

        if self._is_better(score):
            self.best_score = score
            self.best_epoch = epoch
            # 存副本，否则后续训练会就地改掉这份权重
            self.best_weights = copy.deepcopy(model.state_dict())
            self.rounds_without_improvement = 0
            return False

        self.rounds_without_improvement += 1
        if self.rounds_without_improvement >= self.patience:
            self.stopped_epoch = epoch
            return True
        return False

    def on_train_end(self, model: Any) -> None:
        """⭐ 最容易漏的一步：把权重回滚到最好的那一轮。

        漏了这步等于白早停 —— 最终留下的是"已经开始变差"的那一版。
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)

    # ── 内部 ──────────────────────────────────────────────────────

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        return score < self.best_score - self.min_delta

    # ── 给日志用 ──────────────────────────────────────────────────

    def summary(self) -> dict[str, Any]:
        """训练循环把这个写进本轮日志（R10）。

        `best_epoch == 总轮数 - 1` 是"学得不够"这个病的判定依据之一
        （说明还没学完就到头了），所以这个字段必须记。
        """
        return {
            "monitor": self.monitor,
            "best_score": self.best_score,
            "best_epoch": self.best_epoch,
            "stopped_epoch": self.stopped_epoch,
            "patience": self.patience,
        }

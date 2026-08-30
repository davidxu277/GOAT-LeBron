from __future__ import annotations

import copy
from typing import Any


class SWA:
    """SWA权重平均 —— 训练过程类零件。

    配置项（全部从 config 读，不许写死 —— R7）：
        train.swa.enabled             是否启用（由 config_patch 控制）
        train.swa.start_epoch_ratio   从训练总轮数的什么比例开始平均，如 0.75

    实现思路：
        从 start_epoch 开始，每轮结束后累加模型权重做滑动平均，
        训练结束时用平均权重替换模型权重。
        当前模型是纯 MLP + embedding，没有 BatchNorm，
        所以不需要额外跑一遍前向更新 BN 统计量（见药方卡提醒）。
        若未来模型加了 BN，需在此处补上 BN 修正步骤。
    """

    def __init__(self, config: dict[str, Any]):
        cfg = config["train"]["swa"]
        self.enabled: bool = cfg["enabled"]
        self.start_epoch_ratio: float = cfg["start_epoch_ratio"]
        self._swa_weights: Any = None
        self._n: int = 0
        self._start_epoch: int = -1

    def on_train_begin(self, context: dict[str, Any]) -> None:
        if not self.enabled:
            return
        # 从 context 读总轮数，计算出起始轮（0-based）
        total_epochs = int(context["epochs"])
        self._start_epoch = int(total_epochs * self.start_epoch_ratio)
        self._swa_weights = None
        self._n = 0

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, float], model: Any
    ) -> bool:
        if not self.enabled:
            return False
        if epoch >= self._start_epoch:
            current_weights = model.state_dict()
            if self._swa_weights is None:
                # 深拷贝，避免后续训练改动影响快照
                self._swa_weights = copy.deepcopy(current_weights)
            else:
                # 滑动平均：swa = (swa * n + current) / (n + 1)
                for key in self._swa_weights:
                    self._swa_weights[key] = (
                        self._swa_weights[key] * self._n + current_weights[key]
                    ) / (self._n + 1)
            self._n += 1
        return False

    def on_train_end(self, model: Any) -> None:
        if not self.enabled:
            return
        if self._swa_weights is not None:
            model.load_state_dict(self._swa_weights)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "start_epoch_ratio": self.start_epoch_ratio,
            "swa_epochs_averaged": self._n,
            "start_epoch": self._start_epoch,
        }

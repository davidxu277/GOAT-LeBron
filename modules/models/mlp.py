"""共享底层双塔 MLP —— 改模型类零件的**范文**。

给写模型的那个角色看的：这是 ModelOp 该长什么样，以及深度路径给你的东西
（feature_spec 里每个字段的基数）怎么用。

它本身是最朴素的一版：所有字段各发一张 embedding 身份证，拼成一条长向量，
过一个共享底座，再分成两个塔各自输出一个概率。**没有任何巧思** ——
ESMM、MMoE、PLE 那些卡片要做的，正是在这个基础上改结构和损失。

⚠️ 所有维度、层数都从 config 读（R7），不许写死。
⚠️ 这里只定义"模型长什么样"。epoch 循环、优化器、早停回调都归 harness/deep.py，
   不要在零件里自己开训练循环。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SharedBottomMLP(nn.Module):
    """embedding → 共享底座 → 两个塔。"""

    def __init__(self, cardinality: dict[str, int], embed_dim: int,
                 hidden: list[int], tower: list[int], dropout: float):
        super().__init__()
        # 每个字段一张 embedding 表。0 号槽位留给"训练集里没见过的 ID"
        self.embeds = nn.ModuleList(
            [nn.Embedding(n, embed_dim) for n in cardinality.values()])
        宽 = embed_dim * len(self.embeds)
        self.bottom = self._stack(宽, hidden, dropout)
        底宽 = hidden[-1] if hidden else 宽
        self.ctr_tower = self._stack(底宽, tower, dropout)
        self.cvr_tower = self._stack(底宽, tower, dropout)
        塔宽 = tower[-1] if tower else 底宽
        self.ctr_head, self.cvr_head = nn.Linear(塔宽, 1), nn.Linear(塔宽, 1)

    @staticmethod
    def _stack(in_dim: int, dims: list[int], dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        for d in dims:
            layers += [nn.Linear(in_dim, d), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = d
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        vec = torch.cat([e(x[:, i]) for i, e in enumerate(self.embeds)], dim=1)
        shared = self.bottom(vec)
        return {
            "ctr": torch.sigmoid(self.ctr_head(self.ctr_tower(shared))).squeeze(-1),
            "cvr": torch.sigmoid(self.cvr_head(self.cvr_tower(shared))).squeeze(-1),
        }


class MLPModel:
    """ModelOp 实现 —— 深度路径按 modules/base.py 的契约调这两个方法。"""

    def __init__(self, config: dict[str, Any]):
        cfg = ((config.get("model") or {}).get("mlp")) or {}
        self.hidden = [int(x) for x in cfg.get("hidden", [128, 64])]
        self.tower = [int(x) for x in cfg.get("tower", [32])]
        self.dropout = float(cfg.get("dropout", 0.1))

    def build(self, feature_spec: dict[str, Any]) -> nn.Module:
        return SharedBottomMLP(
            cardinality=feature_spec["cardinality"],
            embed_dim=int(feature_spec["embed_dim"]),
            hidden=self.hidden, tower=self.tower, dropout=self.dropout)

    def predict(self, model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return model(x)

"""共享底层双塔 MLP，与深度训练路径的 ModelOp 接口配套。"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class SharedBottomMLP(nn.Module):
    """embedding → 共享底座 → 两个塔。"""

    def __init__(self, cardinality: dict[str, int], embed_dim: int,
                 hidden: list[int], tower: list[int], dropout: float):
        super().__init__()
        self.embeds = nn.ModuleList(
            [nn.Embedding(n, embed_dim) for n in cardinality.values()])
        width = embed_dim * len(self.embeds)
        self.bottom = self._stack(width, hidden, dropout)
        bottom_width = hidden[-1] if hidden else width
        self.ctr_tower = self._stack(bottom_width, tower, dropout)
        self.cvr_tower = self._stack(bottom_width, tower, dropout)
        tower_width = tower[-1] if tower else bottom_width
        self.ctr_head = nn.Linear(tower_width, 1)
        self.cvr_head = nn.Linear(tower_width, 1)

    @staticmethod
    def _stack(in_dim: int, dims: list[int], dropout: float) -> nn.Sequential:
        layers: list[nn.Module] = []
        for dim in dims:
            layers += [nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = dim
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        vec = torch.cat([embedding(x[:, index])
                         for index, embedding in enumerate(self.embeds)], dim=1)
        shared = self.bottom(vec)
        return {
            "ctr": torch.sigmoid(self.ctr_head(self.ctr_tower(shared))).squeeze(-1),
            "cvr": torch.sigmoid(self.cvr_head(self.cvr_tower(shared))).squeeze(-1),
        }


class MLPModel:
    """ModelOp 实现，与 modules/base.py 的契约一致。"""

    def __init__(self, config: dict[str, Any]):
        cfg = ((config.get("model") or {}).get("mlp")) or {}
        self.hidden = [int(value) for value in cfg.get("hidden", [128, 64])]
        self.tower = [int(value) for value in cfg.get("tower", [32])]
        self.dropout = float(cfg.get("dropout", 0.1))

    def build(self, feature_spec: dict[str, Any]) -> nn.Module:
        return SharedBottomMLP(
            cardinality=feature_spec["cardinality"],
            embed_dim=int(feature_spec["embed_dim"]),
            hidden=self.hidden,
            tower=self.tower,
            dropout=self.dropout,
        )

    def predict(self, model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return model(x)

"""AITM 风格的注意力迁移 MLP，实现项目的 ModelOp 接口。

本项目中 ``cvr`` 表示 P(购买 | 点击)，与 ``ctr`` 不在同一条件
空间，因此本零件只做点击向购买塔的注意力迁移，不施加
``cvr <= ctr`` 这种不成立的单调性约束。损失沿用深度训练路径的固定口径：
全曝光点击 BCE，以及仅点击样本上的购买 BCE。
"""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


ATTENTION_SOURCE_COUNT = 2
BINARY_OUTPUT_DIM = 1


def _positive_int_list(value: Any, name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} 必须是整数列表")
    result = [int(item) for item in value]
    if any(item <= 0 for item in result):
        raise ValueError(f"{name} 中的维度必须全部大于 0")
    return result


def _stack(in_dim: int, dims: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for dim in dims:
        layers.extend((nn.Linear(in_dim, dim), nn.ReLU(), nn.Dropout(dropout)))
        in_dim = dim
    return nn.Sequential(*layers)


class AITMMLP(nn.Module):
    """embedding → 共享底座 → 双塔 → 点击信息注意力迁移。"""

    def __init__(
        self,
        cardinality: dict[str, int],
        embed_dim: int,
        hidden: list[int],
        tower: list[int],
        dropout: float,
        attention_hidden: list[int],
    ):
        super().__init__()
        self.embeds = nn.ModuleList(
            nn.Embedding(size, embed_dim) for size in cardinality.values()
        )

        input_width = embed_dim * len(self.embeds)
        self.bottom = _stack(input_width, hidden, dropout)
        bottom_width = hidden[-1] if hidden else input_width

        self.ctr_tower = _stack(bottom_width, tower, dropout)
        self.cvr_tower = _stack(bottom_width, tower, dropout)
        representation_width = tower[-1] if tower else bottom_width

        attention_layers: list[nn.Module] = []
        attention_width = representation_width * ATTENTION_SOURCE_COUNT
        for dim in attention_hidden:
            attention_layers.extend((nn.Linear(attention_width, dim), nn.ReLU()))
            attention_width = dim
        attention_layers.append(nn.Linear(attention_width, ATTENTION_SOURCE_COUNT))
        self.transfer_attention = nn.Sequential(*attention_layers)

        self.ctr_head = nn.Linear(representation_width, BINARY_OUTPUT_DIM)
        self.cvr_head = nn.Linear(representation_width, BINARY_OUTPUT_DIM)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        embedded = torch.cat(
            [embedding(x[:, index])
             for index, embedding in enumerate(self.embeds)],
            dim=1,
        )
        shared = self.bottom(embedded)
        ctr_representation = self.ctr_tower(shared)
        cvr_candidate = self.cvr_tower(shared)

        attention = torch.softmax(
            self.transfer_attention(
                torch.cat((cvr_candidate, ctr_representation), dim=1)
            ),
            dim=1,
        )
        transferred = (
            attention[:, 0:1] * cvr_candidate
            + attention[:, 1:2] * ctr_representation
        )
        return {
            "ctr": torch.sigmoid(self.ctr_head(ctr_representation)).squeeze(-1),
            "cvr": torch.sigmoid(self.cvr_head(transferred)).squeeze(-1),
        }


class AITMModel:
    """AITM 的 ModelOp 实现；所有实验参数必须显式出现在配置中。"""

    def __init__(self, config: dict[str, Any]):
        try:
            mlp_cfg = config["model"]["mlp"]
            aitm_cfg = mlp_cfg["aitm"]
            hidden_value = mlp_cfg["hidden"]
            tower_value = mlp_cfg["tower"]
            dropout_value = mlp_cfg["dropout"]
            attention_value = aitm_cfg["attention_hidden"]
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "AITM 必须配置 model.mlp.hidden、tower、dropout "
                "和 model.mlp.aitm.attention_hidden"
            ) from exc

        self.hidden = _positive_int_list(hidden_value, "model.mlp.hidden")
        self.tower = _positive_int_list(tower_value, "model.mlp.tower")
        self.attention_hidden = _positive_int_list(
            attention_value, "model.mlp.aitm.attention_hidden"
        )
        self.dropout = float(dropout_value)
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.mlp.dropout 必须在 [0, 1) 内")

    def build(self, feature_spec: dict[str, Any]) -> nn.Module:
        cardinality = feature_spec.get("cardinality")
        embed_dim = feature_spec.get("embed_dim")
        if not isinstance(cardinality, dict) or not cardinality:
            raise ValueError("feature_spec.cardinality 必须是非空字典")
        if embed_dim is None or int(embed_dim) <= 0:
            raise ValueError("feature_spec.embed_dim 必须大于 0")
        return AITMMLP(
            cardinality={str(name): int(size)
                         for name, size in cardinality.items()},
            embed_dim=int(embed_dim),
            hidden=self.hidden,
            tower=self.tower,
            dropout=self.dropout,
            attention_hidden=self.attention_hidden,
        )

    def predict(
        self, model: nn.Module, x: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """只处理框架传入的当前批次；分批大小由框架配置。"""
        return model(x)

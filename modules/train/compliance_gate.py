"""训练前合规闸门。

这个零件不改变模型，只在训练真正开始前核对不能依赖人工记忆的红线。
所有规则和值都来自 ``train.compliance_gate`` 配置，零件本身不保存实验参数。
"""

from __future__ import annotations

from typing import Any


class ComplianceGate:
    """实现 TrainOp 接口的只读校验器。"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.rules = config["train"]["compliance_gate"]
        self.checked = False
        self.seed: int | None = None
        self.features: list[str] = []

    def on_train_begin(self, context: dict[str, Any]) -> None:
        features = [str(name) for name in context["features"]]
        forbidden = {str(name) for name in self.rules["forbidden_fields"]}
        leaked = sorted(forbidden.intersection(features))
        if leaked:
            raise ValueError(f"R1：禁用字段进入模型输入：{leaked}")

        configured_seed = self.config["train"].get("seed")
        runtime_seed = context.get("seed")
        if configured_seed is None or runtime_seed is None:
            raise ValueError("R8：配置或运行上下文没有随机种子")
        if (self.rules["require_config_seed_match"]
                and int(configured_seed) != int(runtime_seed)):
            raise ValueError(
                f"R8：配置种子 {configured_seed} 与实际运行种子 {runtime_seed} 不一致"
            )

        eval_config = self.config.get("eval") or {}
        required_eval = self.rules["required_eval"]
        mismatches = {
            key: {"要求": expected, "实际": eval_config.get(key)}
            for key, expected in required_eval.items()
            if eval_config.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"评估口径不合规：{mismatches}")

        self.seed = int(runtime_seed)
        self.features = features
        self.checked = True

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, float], model: Any
    ) -> bool:
        if not self.checked:
            raise RuntimeError("合规闸门没有在训练前执行")
        return False

    def on_train_end(self, model: Any) -> None:
        if not self.checked:
            raise RuntimeError("训练结束时仍未通过合规闸门")

    def summary(self) -> dict[str, Any]:
        return {
            "已通过": self.checked,
            "随机种子": self.seed,
            "模型输入字段": self.features,
            "评估口径": dict(self.rules["required_eval"]),
        }

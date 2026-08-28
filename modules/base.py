"""零件接口规范 —— 工兵写的所有零件都必须实现这里的某一个接口。

负责人：成员2。这份文件定义"接口长什么样"，具体零件放在
modules/features/、modules/models/、modules/train/ 下面。

三种零件对应三种改法：
    FeatureOp  加特征     —— 范文见 modules/features/
    ModelOp    改模型     —— 范文见 modules/models/
    TrainOp    改训练过程 —— 范文见 modules/train/early_stopping.py

写零件时的死规矩（详见 CLAUDE.md）：
    R1  五个禁用字段永远不许进入模型输入
    R2  任何统计量只能在训练集上算，绝不许读验证集
    R5  只能在 modules/ 下新建文件，不许改主程序
    R7  所有参数从 config 读，代码里不许出现写死的数字
"""

from __future__ import annotations

from typing import Any, Protocol


class FeatureOp(Protocol):
    """加特征类零件。

    生命周期：先在训练集上 fit 一次，再对每个数据集分别 transform。
    """

    def needs(self) -> list[str]:
        """【可选，但强烈建议实现】除了 base_fields，还要读哪些原始列。

        执行器靠它决定这次只把哪几列读进内存。这份数据一共 50 列，
        其中 18 列是多值数组 —— 整份读，953 万行要吃约 100 G；
        只读用得上的十几列是 3.8 G。

        不实现不会报错，但执行器只能退回**整份读**，大数据规模上会直接 OOM。
        """
        return []

    def fit(self, train_df: Any) -> None:
        """在训练集上统计需要的量（出现次数、均值、分桶边界、词表……）。

        ⚠️ 只能读训练集。读验证集来算统计量 = 作弊（R2）。
        """

    def transform(self, df: Any) -> Any:
        """把 fit 阶段学到的统计量套用到 df 上，返回加工后的 DataFrame。

        对训练集、开发集、锁定集都调用同一个 transform，行为必须一致。
        """


class ModelOp(Protocol):
    """改模型类零件。

    返回的对象要能被训练循环直接使用；具体框架（torch / lightgbm）由零件自己决定，
    但必须遵守下面的签名。
    """

    def build(self, feature_spec: Any) -> Any:
        """根据特征规格构造模型对象。

        feature_spec 描述每个字段的类型、基数、embedding 维度。
        模型的层数、维度、dropout 等全部从 config 读（R7）。
        """

    def predict(self, model: Any, df: Any) -> dict[str, Any]:
        """输出预测。

        必须返回 {"ctr": <每行的点击概率>, "cvr": <每行的购买概率>}。
        购买概率的定义是 P(购买 | 点击)；报分时在哪些记录上算
        由评估代码决定，零件不要自己筛（见 CLAUDE.md 第五节）。
        """


class TrainOp(Protocol):
    """改训练过程类零件 —— 早停、权重平均、学习率调度、梯度裁剪等。

    设计成回调式：训练循环在固定的几个时点回调它，零件通过返回值影响训练。
    一次训练可以挂多个 TrainOp，按配置里的顺序依次回调。
    """

    def on_train_begin(self, context: dict[str, Any]) -> None:
        """训练开始前调用一次。context 里有配置、随机种子、总轮数等。"""

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, float], model: Any
    ) -> bool:
        """每轮训练结束后调用。

        metrics 是本轮在**开发集**上的指标（绝不会包含锁定集，R3）。
        返回 True 表示"建议现在停止训练"，返回 False 表示继续。
        多个 TrainOp 中任何一个返回 True，训练循环就停。
        """

    def on_train_end(self, model: Any) -> None:
        """训练结束后调用一次。

        需要修改最终权重的零件（早停回滚、SWA 平均）在这里动手。
        """

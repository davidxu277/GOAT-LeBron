"""真实执行器 —— 替掉 FakeExecutor，让 Agent 循环真的跟数据打交道。

实现 agent.loop.Executor 协议：run(patch, fidelity) -> RunResult。
一次 run 干四件事：
  1. 落地工兵的产出（新文件写进 modules/、配置合并）
  2. 训练（点击模型用全部行，购买模型只用 click=1 的行）
  3. 对 val_features 出预测
  4. 评分并组装成绩单（health_report）——医生要的分桶指标在这里算

评分口径写死（CLAUDE.md 第五节）：
  点击 AUC = 全部验证行，正样本 click
  购买 AUC = 仅 click=1 子集，正样本 conversion
两种口径都算都记（Q1 未定，见 docs/baseline笔记.md）。
"""

from __future__ import annotations

import importlib.util
import pathlib
import time
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import log_loss, roc_auc_score

from agent.events import emit
from agent.loop import RunResult
from .data import guard_features, read_any

ROOT = pathlib.Path(__file__).resolve().parent.parent

# v1 只用单值稀疏字段。9 个多值/加权字段（109_14 等）留给「多值字段接回来」那张卡，
# 由工兵写零件加回来 —— 这正是 baseline 身上那个洞，是 Agent 的第一枪。
BASE_FEATURES = ["101", "121", "122", "124", "125", "126", "127", "128", "129",
                 "205", "206", "207", "216", "301"]

FIDELITY_FRAC = {"小份": 0.15, "中份": 0.4, "大份": 0.75, "全量": 1.0}

# 工兵能调的超参数：配置里的键名 → (sklearn 参数名, 下限, 上限, 默认值)。
#
# 区间是**护栏不是建议**：一个 n_estimators: 100000 就能让一轮跑到天亮，
# 把整场的算力预算烧光，而 Agent 自己看不出这是它干的。
# 配置里没写、写歪了、写成字符串 —— 一律退回默认值，绝不让训练因为
# 一个配置错字整轮报废。
LGBM_PARAMS = {
    "n_estimators":     ("n_estimators",      10,   2000,  120),
    "num_leaves":       ("num_leaves",         4,    512,   31),
    "learning_rate":    ("learning_rate",  0.001,    0.5, 0.05),
    "min_data_in_leaf": ("min_child_samples",  1,  10000,   20),
    "feature_fraction": ("colsample_bytree", 0.1,    1.0,  1.0),
}


def _load_op_class_by(rel_path: str, methods: tuple[str, ...],
                      接口名: str = "") -> Any:
    """从 modules/ 下的一个文件里取出实现了指定方法的类。

    三种零件（FeatureOp / ModelOp / TrainOp）共用这一套：路径守卫一处、
    找类的规则一处，免得三份实现各自长歪。

    只认 modules/ 下的路径（R5）—— 这是 Agent 唯一被允许写入的地方，
    放开一寸就等于让它 import 任意文件。
    """
    rel = str(rel_path).replace("\\", "/")
    if not rel.startswith("modules/") or ".." in rel.split("/"):
        raise ValueError(f"非法零件路径：{rel}（只能在 modules/ 下，R5）")
    path = ROOT / rel
    if not path.exists():
        raise FileNotFoundError(f"配置里指的零件文件不存在：{rel}")

    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for obj in vars(module).values():
        if (isinstance(obj, type) and obj.__module__ == module.__name__
                and all(callable(getattr(obj, m, None)) for m in methods)):
            return obj
    raise TypeError(f"{rel} 里没有实现 {接口名 or '零件'} 接口"
                    f"（{' + '.join(methods)}）的类")


def _load_op_class(rel_path: str) -> Any:
    """加特征零件（FeatureOp）。"""
    return _load_op_class_by(rel_path, ("fit", "transform"), "FeatureOp")


def load_feature_ops(config: dict[str, Any]) -> list[tuple[str, Any]]:
    """按配置实例化启用了的加特征零件，返回 [(名字, 实例)]。

    约定：配置块里的 `impl` 指向实现文件。零件类自己去 config 里挖自己那一块
    （见 modules/features/frequency_bucket.py 的范文），所以这里把整份 config 传给它。

    `enabled: true` 却没写 `impl` —— 直接报错。以前这种情况是**静默无效**：
    文件写进去了、配置也改了，但没有任何东西去加载它，训练结果纹丝不动，
    却被记成"这个方案没用"，工兵白挨一次负分。宁可当场炸，也不要假装跑过。
    """
    ops: list[tuple[str, Any]] = []
    for name, block in (config.get("features") or {}).items():
        if not isinstance(block, dict) or not block.get("enabled"):
            continue
        impl = block.get("impl")
        if not impl:
            raise ValueError(
                f"features.{name} 启用了但没写 impl —— 不知道该加载哪个文件。"
                f"配置块里加一行 impl: modules/features/xxx.py")
        ops.append((name, _load_op_class(impl)(config)))
    return ops


def apply_feature_ops(ops: list[tuple[str, Any]], train: pd.DataFrame,
                      others: list[pd.DataFrame]
                      ) -> tuple[pd.DataFrame, list[pd.DataFrame], list[str]]:
    """先在训练集上 fit，再对每份数据 transform。

    返回 (加工后的训练集, 加工后的其他数据集, 新长出来的列名)。

    ⚠️ fit **只看训练集**（R2）。读验证集算统计量 = 作弊，分数虚高，测试集必掉。
    新列会自动接进特征列表 —— 工兵不用再记得去改 base_fields，
    忘了改就等于零件白装，那正是这次要修掉的那类"静默无效"。
    """
    before = set(train.columns)
    for name, op in ops:
        emit("phase", name="装零件", detail=name)
        op.fit(train)
        train = op.transform(train)
        others = [op.transform(df) for df in others]
        if not isinstance(train, pd.DataFrame):
            raise TypeError(f"零件「{name}」的 transform 没有返回 DataFrame")
    return train, others, [c for c in train.columns if c not in before]


def _flatten_array(v):
    """数组转单值。空数组不是错误 —— 它的含义是「用户和这个商品没有交集」，
    本身就是有用的信号（509 有 77% 是空的），所以映射成一个专门的类别 -1。

    模块级函数，不是某个方法里的闭包：训练时（`_fit`）和每一批预测时
    （`_predict_chunk`）都要用同一份逻辑处理同一批字段，闭包各写一份
    容易悄悄改出两个不一致的版本。
    """
    if isinstance(v, str) or not hasattr(v, "__len__"):
        return v
    return v[0] if len(v) else -1


def transform_feature_ops(ops: list[tuple[str, Any]], df: pd.DataFrame) -> pd.DataFrame:
    """零件已经 fit 过了，这里只做 transform —— 不重新 fit。

    分批出预测时，每一批目标数据都要过一遍这个函数：统计量早就在训练集上
    算好定死了（R2），每一批只是去"套用"，绝不能借机再看一眼这批数据自己长什么样。
    """
    for name, op in ops:
        df = op.transform(df)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"零件「{name}」的 transform 没有返回 DataFrame")
    return df


def available_memory_bytes() -> int:
    """这台机器现在还剩多少可用内存。取不到就保守地当 2GB 处理——

    宁可把批分小一点、多跑几批，也不要因为高估了内存又撞一次 OOM
    （真实撞过：`large_25pct` 的 train+public_test 加起来两千多万行，
    一次性读进内存，在 16GB 的机器上直接被系统强制杀掉，退出码 137）。
    """
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        return 2 * 1024 ** 3


def _list_shards(path: pathlib.Path) -> list[pathlib.Path]:
    """跟 harness.data.read_any 用同一套"怎么找分片"的规则。"""
    if path.is_dir():
        shards = sorted(path.glob("*.parquet")) or sorted(path.glob("*.csv"))
        if not shards:
            raise FileNotFoundError(f"目录里没有 parquet/csv 分片：{path}")
        return shards
    return [path]


def read_in_batches(path: pathlib.Path, budget_bytes: int):
    """按内存预算把一份分片数据分批读出来，每批尽量多装但不超预算。

    分片粒度：一批 = 若干个完整分片拼起来，不做分片内部的行级切分——
    这份数据集的分片本身就不大（几万到十来万行一个），够用了。
    单个文件（不是分片目录）会整份当一批，绕不开这个限制。

    第一个分片读出来后，量它的**真实内存占用**（`memory_usage(deep=True)`，
    把多值字段那种 object/数组列也算上，浅算会大幅低估），
    用这个数字换算"一批能装几个分片"，后面所有批次沿用这个估计
    ——不是精确值，但比瞎猜靠谱，而且只多读一个分片的成本，可以接受。
    """
    files = _list_shards(path)
    per_shard_bytes: int | None = None
    batch: list[pd.DataFrame] = []
    batch_bytes = 0
    for f in files:
        df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
        if per_shard_bytes is None:
            per_shard_bytes = max(1, int(df.memory_usage(deep=True).sum()))
        if batch and batch_bytes + per_shard_bytes > budget_bytes:
            yield pd.concat(batch, ignore_index=True)
            batch, batch_bytes = [], 0
        batch.append(df)
        batch_bytes += per_shard_bytes
    if batch:
        yield pd.concat(batch, ignore_index=True)


def lgbm_kwargs(base: dict[str, Any] | None,
                overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """把配置里的超参数翻译成 LGBMClassifier 的入参，越界的夹回区间内。

    纯函数，不碰 lightgbm，方便离线测。
    """
    merged = {**(base or {}), **(overrides or {})}
    out: dict[str, Any] = {}
    for key, (sk_name, lo, hi, default) in LGBM_PARAMS.items():
        raw = merged.get(key, default)
        try:
            value = type(default)(raw)
        except (TypeError, ValueError):
            value = default                      # 写成 "很多" 这种，退回默认
        out[sk_name] = min(hi, max(lo, value))
    return out


# 这条执行器能兑现什么、不能兑现什么。
#
# 26 张卡里只有 3 张是纯特征卡，其余 23 张落在 模型 / 损失函数 / 训练策略 ——
# 而这里走的是 LightGBM：ModelOp / TrainOp 两类零件没有加载机制，
# TrainOp 的接口（按 epoch 回调 + state_dict）跟 LightGBM 结构上也对不上
# （它没有 epoch 循环可以挂回调）。
#
# 最贵的错误不是"做不了"，是"接受了却没做"：工兵改了配置、训练跑完、
# 分数纹丝不动，复盘官据此判「猜错了」，一张好卡被拉黑、信任分被扣，
# 错误结论还会顺着黑名单和升档决策传染下去。宁可当场炸。
SUPPORTED_MODELS = {"lightgbm"}
EPOCH_ONLY_BLOCKS = {
    "swa": "SWA 要按 epoch 累积权重平均",
    "early_stopping_by_epoch": "按 epoch 早停",
}


class UnsupportedByExecutor(ValueError):
    """配置要的东西这条执行器兑现不了 —— 跟「代码写错了」「训练崩了」不是一回事。

    分开是为了别扣错账：ESMM、DeepFM 这些卡跑不了，是我们的流水线还没有
    深度模型训练路径，不是方法本身不靠谱。要是混为一谈，一场跑下来会把
    真正的好方法全扣成低信任分，下一场军师就再也不提它们了 ——
    一个纯属自己造成的错误结论被固化进账本。

    仍然继承 ValueError：老代码里 except ValueError 的地方不会漏接。
    """


def check_supported(config: dict[str, Any]) -> None:
    """配置里有执行器兑现不了的东西就当场炸，绝不静默无视。

    报错信息要让人（和下一轮的军师）看得出这是「跑不了」不是「方法不行」——
    两者对卡片的处置完全不同。
    """
    model_cfg = config.get("model") or {}
    name = str(model_cfg.get("name", "lightgbm")).lower()
    深度 = name not in SUPPORTED_MODELS
    if 深度:
        # 不是 LightGBM 就走深度路径（harness/deep.py）。它需要两样东西：
        # 装了 torch，以及配置里指明模型零件在哪。
        try:
            import torch                                    # noqa: F401,PLC0415
        except Exception as exc:                            # noqa: BLE001
            raise UnsupportedByExecutor(
                f"配置要的模型是「{name}」，走深度路径，但这台机器上 torch 用不了："
                f"{type(exc).__name__}: {exc}。装一下：pip install torch") from exc
        if not model_cfg.get("impl"):
            raise UnsupportedByExecutor(
                f"配置要的模型是「{name}」，但没写 model.impl —— 不知道该加载哪个零件。"
                f"写这个模型的人要同时在 modules/models/ 下建文件，并在 config_patch 里"
                f"补一行 impl: modules/models/xxx.py（零件实现 modules/base.py 的 ModelOp）。")

    train_cfg = config.get("train") or {}
    for key, why in EPOCH_ONLY_BLOCKS.items():
        block = train_cfg.get(key)
        if isinstance(block, dict) and block.get("enabled") and not 深度:
            raise UnsupportedByExecutor(
                f"train.{key} 开着，但{why}，而 LightGBM 这条路没有 epoch 循环"
                f"可以挂回调。深度路径上可以用 —— 把 model.name 换成深度模型，"
                f"并给这个训练零件写上 impl。")

    strategy = str((train_cfg.get("loss_weight") or {}).get("strategy", "fixed")).lower()
    if strategy != "fixed" and not 深度:
        raise UnsupportedByExecutor(
            f"train.loss_weight.strategy 是「{strategy}」，这类动态权重是给"
            f"「一个模型同时学点击和购买」用的；这里点击塔和购买塔是两个独立的"
            f"LightGBM，权重无处可施。要用它得先有多任务模型（ESMM/MMOE 那条路）。")


def recalibrate(probs, keep_ratio: float):
    """负采样之后把概率还原回真实尺度。

    负样本按 w 的比例抽样后，模型看到的正样本占比虚高，预测概率整体偏大。
    按几率（odds）换算回去：odds_真 = w · odds_采样，即

        p_真 = w·p / (1 − p + w·p)

    ⚠️ 这是单调变换，**不会改变 AUC**（AUC 只看排序）。它修的是 logloss
    和「预测均值对不对得上真实点击率」—— 负采样真正的收益在于训练更快、
    购买塔的正负比更平衡，不在于这一步。
    """
    w = float(keep_ratio)
    if w >= 1.0 or w <= 0.0:
        return [float(p) for p in probs]
    return [float(w * p / (1.0 - p + w * p)) if (1.0 - p + w * p) else 0.0
            for p in probs]


def _load_pipeline_config() -> dict[str, Any]:
    """读 config/pipeline.yaml。文件不在就返回空配置，用代码里的默认值兜底。"""
    path = ROOT / "config" / "pipeline.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _deep_set(cfg: dict, path: list[str], value) -> None:
    """按路径深度写入。嵌套字典做递归合并，不覆盖同级的其他键。"""
    node = cfg
    for part in path[:-1]:
        nxt = node.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            node[part] = nxt
        node = nxt
    last = path[-1]
    if isinstance(value, dict) and isinstance(node.get(last), dict):
        for k, v in value.items():
            _deep_set(node[last], str(k).split("."), v)
    else:
        node[last] = value


def _auc(y_true, y_score) -> float | None:
    """算不出来就返回 None，绝不返回 0.5 蒙混过关。"""
    try:
        if len(set(y_true)) < 2:
            return None
        return float(roc_auc_score(y_true, y_score))
    except (ValueError, IndexError):
        return None


def _bucket_metrics(df: pd.DataFrame, ctr_pred, cvr_pred, by: pd.Series,
                    edges: list, names: list[str]) -> list[dict]:
    """按某个维度分桶算指标。样本量太小的桶会被标记，医生据此判断证据可不可信。"""
    out = []
    idx = np.digitize(by.values, edges)
    for i, name in enumerate(names):
        mask = idx == i
        n = int(mask.sum())
        if n == 0:
            continue
        sub = df[mask]
        clicked = sub["click"] == 1
        out.append({
            "区间": name,
            "样本数": n,
            "样本占比": round(n / len(df), 4),
            "点击正样本数": int(sub["click"].sum()),
            "转化正样本数": int(sub.loc[clicked, "conversion"].sum()) if clicked.any() else 0,
            "点击分": _auc(sub["click"], ctr_pred[mask]),
            "购买分": _auc(sub.loc[clicked, "conversion"], cvr_pred[mask][clicked.values])
                       if clicked.any() else None,
        })
    return out


class RealExecutor:
    """真实执行器。数据路径在构造时给定，run() 时按 fidelity 决定用多少数据。"""

    def __init__(self, train_path: str, val_features_path: str,
                 val_labels_path: str | None = None,
                 seed: int = 20260827, config: dict[str, Any] | None = None,
                 holdout_path: str | None = None):
        self.train_path = pathlib.Path(train_path)
        self.val_features_path = pathlib.Path(val_features_path)
        # 验证集自带标签时可以不给这个 —— 见 _train_and_score
        self.val_labels_path = pathlib.Path(val_labels_path) if val_labels_path else None
        # 锁定集（CLAUDE.md R3）：全程锁死，只在选定最终版本时读一次。
        # 开发集被反复看几十轮会挑出「恰好迎合它」的改动，
        # 这份没被任何决策看过的数据是唯一能说清"涨的是不是真本事"的裁判。
        self.holdout_path = pathlib.Path(holdout_path) if holdout_path else None
        self.holdout_reads = 0          # 读过几次 —— 超过 1 次就是违反 R3
        self.seed = seed
        # 不给 config 就读 config/pipeline.yaml —— 工兵改的就是这份，
        # 执行器不读它的话，改配置类的方案永远等于没改。
        self.config = config if config is not None else _load_pipeline_config()
        self._cache: dict[str, pd.DataFrame] = {}

    # ── 数据 ──

    def _read(self, path: pathlib.Path) -> pd.DataFrame:
        """读单个文件或整个分片目录（见 harness.data.read_any）。"""
        key = str(path)
        if key not in self._cache:
            self._cache[key] = read_any(path)
        return self._cache[key]

    # ── Executor 协议 ──

    def run(self, patch: dict[str, Any], fidelity: str) -> RunResult:
        t0 = time.time()
        try:
            self._apply_patch(patch)
            # 配置里有兑现不了的东西就当场炸 —— 静默无视会让工兵白改一轮，
            # 分数纹丝不动，复盘官却据此判「猜错了」，把好卡片拉黑。
            check_supported(self.config)
            report = self._train_and_score(fidelity)
            return RunResult(ok=True, health_report=report,
                             seconds=time.time() - t0, fidelity=fidelity)
        except Exception as exc:
            emit("recovery", text=f"执行失败：{type(exc).__name__}: {exc}")
            return RunResult(ok=False, error=f"{type(exc).__name__}: {exc}",
                             seconds=time.time() - t0, fidelity=fidelity,
                             unsupported=isinstance(exc, UnsupportedByExecutor))

    def final_judge(self, fidelity: str = "全量") -> RunResult:
        """在锁定集上评一次 —— 整场只许调用一次（CLAUDE.md R3）。

        用当前配置重训一遍，在这份从没被任何决策看过的数据上评分。
        它回答的问题是：开发集上涨的那些分，有多少是真本事、
        有多少只是反复筛选筛出来的迎合。

        没配锁定集就返回 ok=False 的空结果，不算错 —— 只是没有裁判。
        """
        if self.holdout_path is None:
            return RunResult(ok=False, error="没有配锁定集", fidelity=fidelity)
        if self.holdout_reads:
            # 读第二次就失去意义了：一旦拿它的分数做过任何决策，
            # 它就跟开发集一样被污染了。这里硬拦，不靠自觉。
            raise RuntimeError(
                f"锁定集已经读过 {self.holdout_reads} 次。R3：全程只许读一次，"
                f"读第二次它就不再是干净的裁判了。"
            )
        t0 = time.time()
        self.holdout_reads += 1
        emit("phase", name="锁定集裁决", detail=f"{self.holdout_path.name} · 整场唯一一次")
        try:
            report = self._train_and_score(fidelity, eval_path=self.holdout_path)
            return RunResult(ok=True, health_report=report,
                             seconds=time.time() - t0, fidelity=fidelity)
        except Exception as exc:
            emit("recovery", text=f"锁定集裁决失败：{type(exc).__name__}: {exc}")
            return RunResult(ok=False, error=f"{type(exc).__name__}: {exc}",
                             seconds=time.time() - t0, fidelity=fidelity)

    def _apply_patch(self, patch: dict[str, Any]) -> None:
        """把工兵的产出落地。只允许写 modules/ 下的文件（R5）。"""
        for f in patch.get("new_files", []):
            rel = f["path"].replace("\\", "/")
            if not rel.startswith("modules/") or ".." in rel:
                raise ValueError(f"非法写入路径：{rel}（只能写 modules/ 下，R5）")
            target = ROOT / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(f["content"], encoding="utf-8")
            emit("phase", name="落地代码", detail=rel)
        # 工兵产出的 config_patch 是 YAML **文本**（见 agent/schemas.py），
        # 不是 dict —— 直接 .items() 会当场 AttributeError。
        raw = patch.get("config_patch") or {}
        parsed = yaml.safe_load(raw) if isinstance(raw, str) else raw
        if parsed and not isinstance(parsed, dict):
            raise ValueError(f"config_patch 解析出来是 {type(parsed).__name__}，必须是键值对")
        for key, value in (parsed or {}).items():
            # 支持点号路径（features.类目兜底.K: 20）和嵌套字典两种写法，
            # 都做深度合并 —— 浅层赋值会把整棵子树冲掉，
            # 工兵只想改一个 K，结果把其他零件的配置全抹了。
            _deep_set(self.config, str(key).split("."), value)
            emit("phase", name="改配置", detail=str(key))

    def _train_and_score(self, fidelity: str,
                         eval_path: pathlib.Path | None = None) -> dict[str, Any]:
        """训练并评分。eval_path 不给就用开发集；给了就在那份数据上评
        （锁定集用这条路 —— 训练逻辑完全一样，只换被评的数据）。"""
        bundle = self._fit(fidelity)
        features = bundle["features"]

        val_x = self._read(eval_path or self.val_features_path)
        # 标签来源二选一：单独的私藏文件，或验证集自带（分片数据集常见）
        有标签 = {"click", "conversion"} <= set(val_x.columns)
        if eval_path is not None:
            # 锁定集自带标签（跟开发集同源同格式），不走那份私藏文件
            if not 有标签:
                raise ValueError(f"锁定集 {eval_path} 里没有标签，无法当裁判")
            val_y = val_x[["sample_id", "click", "conversion"]].copy()
        elif self.val_labels_path is not None:
            val_y = self._read(self.val_labels_path)
        elif 有标签:
            val_y = val_x[["sample_id", "click", "conversion"]].copy()
        else:
            raise ValueError("验证集不含标签，且没有提供 val_labels 文件，无法评分")

        val_x, ctr_pred, cvr_pred = self._predict_chunk(bundle, val_x)

        # 预测自检：宁可崩掉，也不产出格式对但内容有问题的结果
        assert len(ctr_pred) == len(val_x), "预测行数与目标数据对不上"
        assert not np.isnan(ctr_pred).any() and not np.isnan(cvr_pred).any(), "预测里有 NaN"

        merged = val_x[["sample_id"] + features].merge(val_y, on="sample_id", how="inner")
        assert len(merged) == len(val_x), "按 sample_id 关联后丢行了"

        return self._build_report(merged, ctr_pred, cvr_pred,
                                  bundle["train"], features,
                                  bundle["ctr_model"], bundle["cvr_model"], fidelity,
                                  ctr_kw=bundle["ctr_kw"], cvr_kw=bundle["cvr_kw"],
                                  op_names=[name for name, _ in bundle["ops"]],
                                  深度训练=bundle.get("深度训练"),
                                  训练集预测=bundle.get("训练集预测"))

    def _fit(self, fidelity: str) -> dict[str, Any]:
        """只训练，不碰任何目标数据。返回一份"拟合好的家当"。

        评分（`_train_and_score`）和导出预测（`predict_frame`）**都从这里拿家当**，
        区别只在于拿着它去预测哪份数据、要不要分批喂——不能各自训一遍，
        那样连"两边是不是同一个模型"都保证不了，这正是我们这两天一直在抓的那类 bug。
        """
        # 评分那条路在 run() 里已经查过一遍，但导出预测是从这里直接进来的 ——
        # 不查的话，一份写着 deepfm 的配置会**悄悄训一个普通 LightGBM**，
        # 给你一份看起来完全正常的预测文件，没有任何迹象说明它没按配置跑。
        # 交付物 #4 是最终提交物，这种"看起来正常但其实不对"最要命。
        check_supported(self.config)

        train = self._read(self.train_path)
        frac = FIDELITY_FRAC.get(fidelity, 1.0)
        if frac < 1.0:
            # 分层抽样：正样本全留，负样本按比例抽，保证小份数据也有正样本可学
            pos = train[train["click"] == 1]
            neg = train[train["click"] == 0].sample(frac=frac, random_state=self.seed)
            train = pd.concat([pos, neg]).sample(frac=1.0, random_state=self.seed)

        # 特征清单从配置读（R7）—— 工兵改 features.base_fields 才真的生效。
        # 写死在代码里的话，医生诊断出「特征没用上」也没人能修。
        wanted = (self.config.get("features") or {}).get("base_fields") or BASE_FEATURES
        # 装上配置里启用的加特征零件 —— 以前工兵写的零件文件躺在 modules/ 下
        # 从来没有被 import 过，训练结果纹丝不动却被记成"这个方案没用"
        ops = load_feature_ops(self.config)
        train, _, 新列 = apply_feature_ops(ops, train, [])

        features = [str(c) for c in wanted if str(c) in train.columns]
        missing = [str(c) for c in wanted if str(c) not in train.columns]
        if missing:
            emit("phase", name="特征缺失",
                 detail=f"配置里有 {len(missing)} 个字段数据里没有：{missing[:5]}")
        # 零件新长出来的列自动进特征表，工兵不用再记得改 base_fields
        features += [c for c in 新列 if c in train.columns and c not in features]
        if not features:
            raise ValueError(f"配置里的特征一个都不在数据里：{wanted[:8]}")
        guard_features(features)          # R1 运行时防线（新列也走这一关）
        emit("phase", name="训练", detail=f"{fidelity} · {len(train):,} 行 · {len(features)} 个特征")

        # 数据里有些字段是数组（4 个交叉字段大多单值，853 和历史行为字段是真多值）。
        # 数组不能直接当类别特征 —— LightGBM 要求可哈希，会当场抛
        # 「unhashable type: numpy.ndarray」。单值的拆出来用，
        # 真多值的交给专门的编码零件（见「多值字段接回来」那张卡），这里先跳过。
        #
        # 这里定下的"拍平哪些列、跳过哪些列"要原样喂给 _predict_chunk，
        # 每一批目标数据都照这份决定处理——不能每批自己重新判断一次，
        # 万一某一批恰好该字段全是单值，会跟训练时的决定对不上。
        multivalue: list[str] = []
        flatten_cols: list[str] = []
        for col in list(features):
            if not train[col].map(lambda v: hasattr(v, "__len__")
                                  and not isinstance(v, str)).any():
                continue
            sizes = train[col].map(lambda v: len(v) if hasattr(v, "__len__") else 1)
            if sizes.max() > 1:
                # 真多值（853、历史行为字段）交给专门的编码零件，
                # 见「多值字段接回来」那张卡。硬转单值会丢信息。
                multivalue.append(col)
                features.remove(col)
                continue
            train[col] = train[col].map(_flatten_array)
            flatten_cols.append(col)
        if multivalue:
            emit("phase", name="跳过多值字段",
                 detail=f"{multivalue} 是真多值，需要编码零件才能用")

        for col in features:
            train[col] = train[col].astype("category")

        # 负采样：负样本按比例抽，正样本全留（R7，比例从配置读）。
        # 收益是训练更快、购买塔正负比更平衡；预测出来的概率整体偏大，
        # 后面用 recalibrate 还原回真实尺度。
        train_cfg = self.config.get("train") or {}
        ns = train_cfg.get("negative_sampling") or {}
        keep_ratio = 1.0
        if ns.get("enabled"):
            keep_ratio = float(ns.get("keep_ratio", 1.0))
            if not 0.0 < keep_ratio < 1.0:
                raise ValueError(
                    f"train.negative_sampling.keep_ratio 得在 0~1 之间，收到 {keep_ratio}")
            pos_rows = train[train["click"] == 1]
            neg_rows = train[train["click"] == 0].sample(
                frac=keep_ratio, random_state=self.seed)
            train = pd.concat([pos_rows, neg_rows]).sample(
                frac=1.0, random_state=self.seed)
            emit("phase", name="负采样",
                 detail=f"负样本保留 {keep_ratio:.0%} · 剩 {len(train):,} 行")
            if not ns.get("recalibrate", True):
                keep_ratio = 1.0        # 明确不还原，那就别在预测上动手

        model_cfg = self.config.get("model") or {}

        # ── 深度路径：model.name 不是 lightgbm 就走这边（harness/deep.py）──
        # 循环归我们管，模型长什么样归零件管 —— 考场和考生的分界（CLAUDE.md R5）
        #
        # 分叉放在 LightGBM 那套早停设置**之前**：那套是内切一块训练集当裁判 +
        # 挂 lgb 回调，深度路径一样都用不上（它在 epoch 之间用开发集早停，
        # 回调走 TrainOp）。放在后面会白切一块数据，还会 import 用不着的 lightgbm。
        if str(model_cfg.get("name", "lightgbm")).lower() not in SUPPORTED_MODELS:
            from .deep import predict_deep, train_deep

            # 从**训练集内部**切一小块当每轮的裁判（R2/R3）——
            # 跟 LightGBM 那条路同一个规矩：早停绝不能盯着"当前在评的那份数据"，
            # 否则 final_judge 时会变成拿锁定集决定停在哪一轮。
            frac = float((train_cfg.get("early_stopping") or {}).get(
                "inner_holdout_frac", 0.1))
            裁判 = train.sample(frac=frac, random_state=self.seed)
            train = train.drop(裁判.index)
            emit("phase", name="深度早停裁判",
                 detail=f"训练集内切 {frac:.0%}（{len(裁判):,} 行）")

            # 只训，不预测 —— 预测统一交给 _predict_chunk，深度模型也能分批喂
            op, model, 训练记录 = train_deep(self.config, train, 裁判, features, self.seed)
            vocab = 训练记录.pop("_vocab")
            return {
                "features": features, "flatten_cols": flatten_cols, "ops": ops,
                "keep_ratio": keep_ratio, "train": train,
                "ctr_model": None, "cvr_model": None, "cvr_fallback": 0.0,
                "ctr_kw": 训练记录["超参数"], "cvr_kw": 训练记录["超参数"],
                "深度": {"op": op, "model": model, "vocab": vocab},
                "深度训练": 训练记录,
                # 训练集上的预测：医生判「在背题」要拿它跟验证分比
                "训练集预测": dict(zip(("ctr", "cvr"),
                                   predict_deep(op, model, vocab, train))),
            }

        # 早停：从**训练集内部**再切一小块当早停的裁判（R2/R3）。
        # 为什么不用开发集：final_judge 走的是同一条训练路径，只换被评的数据；
        # 早停若盯着"当前在评的那份数据"，锁定集大考时就会拿锁定集来决定停在哪 ——
        # 那既违反 R3，也让大考评的模型跟当轮选中的不是同一个。
        es = train_cfg.get("early_stopping") or {}
        fit_extra: dict[str, Any] = {}
        inner = None
        if es.get("enabled"):
            hold_frac = float(es.get("inner_holdout_frac", 0.1))
            inner = train.sample(frac=hold_frac, random_state=self.seed)
            train = train.drop(inner.index)
            patience = int(es.get("patience", 3))
            emit("phase", name="早停",
                 detail=f"训练集内切 {hold_frac:.0%} 当裁判 · patience={patience}")
            import lightgbm as lgb
            fit_extra["callbacks"] = [lgb.early_stopping(patience, verbose=False)]

        # 点击模型：全部行
        # import 放在分叉之后：深度路径根本用不到 LightGBM，
        # 也不该因为这台机器装不上它（缺 libomp 之类）就跑不了
        from lightgbm import LGBMClassifier

        # 超参数从配置读（R7）—— 这以前是写死的，工兵改了 model.lightgbm.* 也纹丝不动
        ctr_kw = lgbm_kwargs(model_cfg.get("lightgbm"))
        cvr_kw = lgbm_kwargs(model_cfg.get("lightgbm"), model_cfg.get("cvr_overrides"))
        emit("phase", name="超参数", detail=f"点击塔 {ctr_kw}")

        ctr_model = LGBMClassifier(random_state=self.seed, verbosity=-1, **ctr_kw)
        ctr_model.fit(train[features], train["click"], categorical_feature=features,
                      **({**fit_extra, "eval_X": inner[features],
                          "eval_y": inner["click"]}
                         if inner is not None else {}))

        # 购买模型：只用 click=1 的行
        clicked = train[train["click"] == 1]
        cvr_model = None
        if clicked["conversion"].nunique() >= 2:
            cvr_model = LGBMClassifier(random_state=self.seed, verbosity=-1, **cvr_kw)
            # 早停的裁判也只取点击过的行；不够两类就退回不早停，
            # 别为了早停把购买塔训崩（转化正样本本来就少）
            inner_clicked = (inner[inner["click"] == 1] if inner is not None
                             else None)
            cvr_extra = ({**fit_extra,
                          "eval_X": inner_clicked[features],
                          "eval_y": inner_clicked["conversion"]}
                         if inner_clicked is not None
                         and inner_clicked["conversion"].nunique() >= 2 else {})
            cvr_model.fit(clicked[features], clicked["conversion"],
                          categorical_feature=features, **cvr_extra)

        return {
            "features": features, "flatten_cols": flatten_cols, "ops": ops,
            "keep_ratio": keep_ratio, "train": train,
            "ctr_model": ctr_model, "cvr_model": cvr_model,
            "cvr_fallback": float(clicked["conversion"].mean() or 0.005),
            "ctr_kw": ctr_kw, "cvr_kw": cvr_kw,
        }

    def _predict_chunk(self, bundle: dict[str, Any], target: pd.DataFrame
                       ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """拿 `_fit` 拟合好的家当，对**一批**目标数据出预测。

        不重新 fit 任何东西——零件的统计量、多值字段的拍平/跳过决定、
        特征列表，全部沿用训练时定的那份，保证不管分几批喂、每一批用的
        都是同一个模型、同一套处理方式。

        返回处理后的 target（带上零件新长出的列，供调用方拼成绩单用）
        和两个预测数组。
        """
        features = bundle["features"]
        target = transform_feature_ops(bundle["ops"], target)
        for col in bundle["flatten_cols"]:
            if col in target.columns:
                target[col] = target[col].map(_flatten_array)
        for col in features:
            target[col] = target[col].astype("category")

        深度 = bundle.get("深度")
        if 深度 is not None:
            # 深度模型不吃 category 类型，走自己的 ID 词表（训练集上建的，R2）
            from .deep import predict_deep

            ctr_pred, cvr_pred = predict_deep(
                深度["op"], 深度["model"], 深度["vocab"], target)
        else:
            ctr_pred = bundle["ctr_model"].predict_proba(target[features])[:, 1]
            cvr_pred = (bundle["cvr_model"].predict_proba(target[features])[:, 1]
                       if bundle["cvr_model"] is not None
                       else np.full(len(target), bundle["cvr_fallback"]))
        if bundle["keep_ratio"] < 1.0:
            # 只有点击塔的负样本被抽掉了；购买塔用的是 click=1 子集，不受影响
            ctr_pred = np.asarray(recalibrate(ctr_pred, bundle["keep_ratio"]))
        return target, ctr_pred, cvr_pred

    def predict_frame(self, test_path: str | pathlib.Path,
                      fidelity: str = "全量") -> pd.DataFrame:
        """对一份数据出预测，返回 DataFrame（交付物 #4 的原料）。

        跟评分**同一条训练路径**（先 `_fit` 训一次，再用 `_predict_chunk` 出预测），
        只是目标数据换成测试集、不要求它带标签，而且**按内存预算分批读**——
        真撞过 OOM：`large_25pct` 的 train+public_test 加起来两千多万行，
        一次性读进 16GB 的机器直接被系统杀掉。模型只训一次，分批的只是
        "喂给它做预测的数据"，所以不管分几批，出来的都是同一个模型的结果。

        列的含义：
            sample_id  原样带出来，用来跟官方的行对齐
            ctr        P(点击)
            cvr        P(购买 | 点击) —— 条件概率，不是 P(点击且购买)
            ctcvr      P(点击且购买) = ctr × cvr，顺手给出，
                       因为不同评测脚本要的口径不一样（见 docs/baseline笔记.md Q1）
        """
        bundle = self._fit(fidelity)
        budget = min(2 * 1024 ** 3, max(256 * 1024 ** 2, available_memory_bytes() // 4))
        emit("phase", name="导出预测", detail=f"分批读取 · 每批预算约 {budget // 1024**2}MB")

        chunks: list[pd.DataFrame] = []
        total = 0
        for raw in read_in_batches(pathlib.Path(test_path), budget):
            if "sample_id" not in raw.columns:
                raise ValueError(f"{test_path} 里没有 sample_id，预测没法跟官方的行对齐")
            target, ctr, cvr = self._predict_chunk(bundle, raw)
            chunks.append(pd.DataFrame({
                "sample_id": target["sample_id"].to_numpy(),
                "ctr": ctr, "cvr": cvr, "ctcvr": ctr * cvr,
            }))
            total += len(target)
            emit("phase", name="导出预测·进度", detail=f"已出 {total:,} 行")

        out = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(
            columns=["sample_id", "ctr", "cvr", "ctcvr"])
        emit("phase", name="导出预测完成", detail=f"{len(out):,} 行 · {fidelity}训练")
        return out

    def _build_report(self, val, ctr_pred, cvr_pred, train, features,
                      ctr_model, cvr_model, fidelity,
                      ctr_kw: dict[str, Any] | None = None,
                      cvr_kw: dict[str, Any] | None = None,
                      深度训练: dict[str, Any] | None = None,
                      训练集预测: dict[str, Any] | None = None,
                      op_names: list[str] | None = None) -> dict[str, Any]:
        """组装成绩单 —— 医生诊断需要的全部字段都在这里。"""
        clicked_mask = (val["click"] == 1).values
        ctr_auc = _auc(val["click"], ctr_pred)
        cvr_auc = _auc(val.loc[clicked_mask, "conversion"], cvr_pred[clicked_mask])
        # Q1 未定：全曝光口径的购买 AUC 也算也记（见 docs/baseline笔记.md）
        cvr_auc_all = _auc(val["conversion"], cvr_pred)

        # 训练集自评，用来判断「在背题」
        # 训练集自评 —— 医生判「在背题」靠训练分和验证分的差。
        # 深度路径没有 sklearn 那样的 predict_proba，预测由调用方算好传进来。
        if ctr_model is None:
            tr_ctr = _auc(train["click"], (训练集预测 or {}).get("ctr")) \
                if (训练集预测 or {}).get("ctr") is not None else None
        else:
            tr_ctr = _auc(train["click"], ctr_model.predict_proba(train[features])[:, 1])
        tr_clicked = train[train["click"] == 1]
        tr_cvr = (_auc(tr_clicked["conversion"],
                       cvr_model.predict_proba(tr_clicked[features])[:, 1])
                  if cvr_model is not None else None)

        # 商品出现次数分桶（「冷门商品学不动」的判定依据）
        item_freq = train["205"].value_counts() if "205" in train.columns else pd.Series(dtype=int)
        val_item_freq = val["205"].map(item_freq).fillna(0) if "205" in val.columns else None

        report: dict[str, Any] = {
            "保真度": fidelity,
            "随机种子": self.seed,
            "验证集": {
                "总行数": len(val),
                "点击数": int(val["click"].sum()),
                "转化数": int(val["conversion"].sum()),
                "点击分": ctr_auc,
                "购买分": cvr_auc,
                "购买分_全曝光口径": cvr_auc_all,
            },
            "训练集": {
                "总行数": len(train),
                "点击分": tr_ctr,
                "购买分": tr_cvr,
            },
            "当前特征": features,
            "装上的零件": op_names or [],
            # 深度路径的每轮曲线 —— 医生判「在背题」「学得不够」要看它
            **({"深度训练": 深度训练} if 深度训练 else {}),
            "实际超参数": {"点击塔": ctr_kw or {}, "购买塔": cvr_kw or {}},
            "未使用的字段": [c for c in ("109_14", "110_14", "127_14", "150_14",
                                      "508", "509", "702", "853")
                          if c in train.columns and c not in features],
        }

        if cvr_auc is not None and val.loc[clicked_mask, "conversion"].sum() < 50:
            report["验证集"]["购买分可信度警告"] = (
                f"点击子集里只有 {int(val.loc[clicked_mask, 'conversion'].sum())} 条转化，"
                f"这个购买分波动极大，不足以支撑「涨了还是没涨」的判断")

        if val_item_freq is not None:
            report["按商品出现次数分组"] = _bucket_metrics(
                val, ctr_pred, cvr_pred, val_item_freq,
                edges=[10, 100, 1000],
                names=["<10次", "10-100次", "100-1000次", ">1000次"])

        if "101" in train.columns and "101" in val.columns:
            seen = set(train["101"].unique())
            is_seen = val["101"].isin(seen)
            report["按用户是否见过分组"] = []
            for label, mask in (("训练集里见过的", is_seen.values),
                                ("训练集里没见过的", (~is_seen).values)):
                if mask.sum() == 0:
                    continue
                sub = val[mask]
                sub_clicked = (sub["click"] == 1).values
                report["按用户是否见过分组"].append({
                    "区间": label,
                    "样本数": int(mask.sum()),
                    "样本占比": round(float(mask.sum()) / len(val), 4),
                    "转化正样本数": int(sub.loc[sub_clicked, "conversion"].sum())
                                   if sub_clicked.any() else 0,
                    "点击分": _auc(sub["click"], ctr_pred[mask]),
                    "购买分": _auc(sub.loc[sub_clicked, "conversion"],
                                  cvr_pred[mask][sub_clicked]) if sub_clicked.any() else None,
                })

        try:
            report["验证集"]["点击LogLoss"] = float(log_loss(val["click"], ctr_pred))
        except ValueError:
            pass

        emit("metrics", ctr_auc=ctr_auc, cvr_auc=cvr_auc)
        return report

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


def _load_op_class(rel_path: str) -> Any:
    """从 modules/ 下的一个文件里取出零件类。

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
                and callable(getattr(obj, "fit", None))
                and callable(getattr(obj, "transform", None))):
            return obj
    raise TypeError(f"{rel} 里没有实现 FeatureOp 接口（fit + transform）的类")


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
            report = self._train_and_score(fidelity)
            return RunResult(ok=True, health_report=report,
                             seconds=time.time() - t0, fidelity=fidelity)
        except Exception as exc:
            emit("recovery", text=f"执行失败：{type(exc).__name__}: {exc}")
            return RunResult(ok=False, error=f"{type(exc).__name__}: {exc}",
                             seconds=time.time() - t0, fidelity=fidelity)

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
        from lightgbm import LGBMClassifier

        train = self._read(self.train_path)
        frac = FIDELITY_FRAC.get(fidelity, 1.0)
        if frac < 1.0:
            # 分层抽样：正样本全留，负样本按比例抽，保证小份数据也有正样本可学
            pos = train[train["click"] == 1]
            neg = train[train["click"] == 0].sample(frac=frac, random_state=self.seed)
            train = pd.concat([pos, neg]).sample(frac=1.0, random_state=self.seed)

        val_x = self._read(eval_path or self.val_features_path)
        # 标签来源二选一：单独的私藏文件，或验证集自带（分片数据集常见）
        if eval_path is not None:
            # 锁定集自带标签（跟开发集同源同格式），不走那份私藏文件
            if not {"click", "conversion"} <= set(val_x.columns):
                raise ValueError(f"锁定集 {eval_path} 里没有标签，无法当裁判")
            val_y = val_x[["sample_id", "click", "conversion"]].copy()
        elif self.val_labels_path is not None:
            val_y = self._read(self.val_labels_path)
        elif {"click", "conversion"} <= set(val_x.columns):
            val_y = val_x[["sample_id", "click", "conversion"]].copy()
        else:
            raise ValueError("验证集不含标签，且没有提供 val_labels 文件，无法评分")

        # 特征清单从配置读（R7）—— 工兵改 features.base_fields 才真的生效。
        # 写死在代码里的话，医生诊断出「特征没用上」也没人能修。
        wanted = (self.config.get("features") or {}).get("base_fields") or BASE_FEATURES
        # 装上配置里启用的加特征零件 —— 以前工兵写的零件文件躺在 modules/ 下
        # 从来没有被 import 过，训练结果纹丝不动却被记成"这个方案没用"
        ops = load_feature_ops(self.config)
        train, (val_x,), 新列 = apply_feature_ops(ops, train, [val_x])

        features = [str(c) for c in wanted if str(c) in train.columns]
        missing = [str(c) for c in wanted if str(c) not in train.columns]
        if missing:
            emit("phase", name="特征缺失",
                 detail=f"配置里有 {len(missing)} 个字段数据里没有：{missing[:5]}")
        # 零件新长出来的列自动进特征表，工兵不用再记得改 base_fields
        features += [c for c in 新列 if c in val_x.columns and c not in features]
        if not features:
            raise ValueError(f"配置里的特征一个都不在数据里：{wanted[:8]}")
        guard_features(features)          # R1 运行时防线（新列也走这一关）
        emit("phase", name="训练", detail=f"{fidelity} · {len(train):,} 行 · {len(features)} 个特征")

        # 数据里有些字段是数组（4 个交叉字段大多单值，853 和历史行为字段是真多值）。
        # 数组不能直接当类别特征 —— LightGBM 要求可哈希，会当场抛
        # 「unhashable type: numpy.ndarray」。单值的拆出来用，
        # 真多值的交给专门的编码零件（见「多值字段接回来」那张卡），这里先跳过。
        def _flatten(v):
            """数组转单值。空数组不是错误 —— 它的含义是「用户和这个商品没有交集」，
            本身就是有用的信号（509 有 77% 是空的），所以映射成一个专门的类别 -1。"""
            if isinstance(v, str) or not hasattr(v, "__len__"):
                return v
            return v[0] if len(v) else -1

        multivalue: list[str] = []
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
            train[col] = train[col].map(_flatten)
            val_x[col] = val_x[col].map(_flatten)
        if multivalue:
            emit("phase", name="跳过多值字段",
                 detail=f"{multivalue} 是真多值，需要编码零件才能用")

        for col in features:
            train[col] = train[col].astype("category")
            val_x[col] = val_x[col].astype("category")

        # 点击模型：全部行
        # 超参数从配置读（R7）—— 这以前是写死的，工兵改了 model.lightgbm.* 也纹丝不动
        model_cfg = self.config.get("model") or {}
        ctr_kw = lgbm_kwargs(model_cfg.get("lightgbm"))
        cvr_kw = lgbm_kwargs(model_cfg.get("lightgbm"), model_cfg.get("cvr_overrides"))
        emit("phase", name="超参数", detail=f"点击塔 {ctr_kw}")

        ctr_model = LGBMClassifier(random_state=self.seed, verbosity=-1, **ctr_kw)
        ctr_model.fit(train[features], train["click"], categorical_feature=features)

        # 购买模型：只用 click=1 的行
        clicked = train[train["click"] == 1]
        cvr_model = None
        if clicked["conversion"].nunique() >= 2:
            cvr_model = LGBMClassifier(random_state=self.seed, verbosity=-1, **cvr_kw)
            cvr_model.fit(clicked[features], clicked["conversion"],
                          categorical_feature=features)

        ctr_pred = ctr_model.predict_proba(val_x[features])[:, 1]
        cvr_pred = (cvr_model.predict_proba(val_x[features])[:, 1] if cvr_model is not None
                    else np.full(len(val_x), float(clicked["conversion"].mean() or 0.005)))

        # 预测自检：宁可崩掉，也不产出格式对但内容有问题的结果
        assert len(ctr_pred) == len(val_x), "预测行数与验证集对不上"
        assert not np.isnan(ctr_pred).any() and not np.isnan(cvr_pred).any(), "预测里有 NaN"

        merged = val_x[["sample_id"] + features].merge(val_y, on="sample_id", how="inner")
        assert len(merged) == len(val_x), "按 sample_id 关联后丢行了"

        return self._build_report(merged, ctr_pred, cvr_pred, train, features,
                                  ctr_model, cvr_model, fidelity,
                                  op_names=[name for name, _ in ops])

    def _build_report(self, val, ctr_pred, cvr_pred, train, features,
                      ctr_model, cvr_model, fidelity,
                      op_names: list[str] | None = None) -> dict[str, Any]:
        """组装成绩单 —— 医生诊断需要的全部字段都在这里。"""
        clicked_mask = (val["click"] == 1).values
        ctr_auc = _auc(val["click"], ctr_pred)
        cvr_auc = _auc(val.loc[clicked_mask, "conversion"], cvr_pred[clicked_mask])
        # Q1 未定：全曝光口径的购买 AUC 也算也记（见 docs/baseline笔记.md）
        cvr_auc_all = _auc(val["conversion"], cvr_pred)

        # 训练集自评，用来判断「在背题」
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
            "实际超参数": {"点击塔": ctr_kw, "购买塔": cvr_kw},
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

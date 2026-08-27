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

import pathlib
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

from agent.events import emit
from agent.loop import RunResult
from .data import guard_features

ROOT = pathlib.Path(__file__).resolve().parent.parent

# v1 只用单值稀疏字段。9 个多值/加权字段（109_14 等）留给「多值字段接回来」那张卡，
# 由工兵写零件加回来 —— 这正是 baseline 身上那个洞，是 Agent 的第一枪。
BASE_FEATURES = ["101", "121", "122", "124", "125", "126", "127", "128", "129",
                 "205", "206", "207", "216", "301"]

FIDELITY_FRAC = {"小份": 0.15, "中份": 0.4, "大份": 0.75, "全量": 1.0}


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

    def __init__(self, train_path: str, val_features_path: str, val_labels_path: str,
                 seed: int = 20260827, config: dict[str, Any] | None = None):
        self.train_path = pathlib.Path(train_path)
        self.val_features_path = pathlib.Path(val_features_path)
        self.val_labels_path = pathlib.Path(val_labels_path)
        self.seed = seed
        self.config = config or {}
        self._cache: dict[str, pd.DataFrame] = {}

    # ── 数据 ──

    def _read(self, path: pathlib.Path) -> pd.DataFrame:
        key = str(path)
        if key not in self._cache:
            self._cache[key] = (pd.read_parquet(path) if path.suffix == ".parquet"
                                else pd.read_csv(path))
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
        for key, value in (patch.get("config_patch") or {}).items():
            self.config[key] = value

    def _train_and_score(self, fidelity: str) -> dict[str, Any]:
        from lightgbm import LGBMClassifier

        train = self._read(self.train_path)
        frac = FIDELITY_FRAC.get(fidelity, 1.0)
        if frac < 1.0:
            # 分层抽样：正样本全留，负样本按比例抽，保证小份数据也有正样本可学
            pos = train[train["click"] == 1]
            neg = train[train["click"] == 0].sample(frac=frac, random_state=self.seed)
            train = pd.concat([pos, neg]).sample(frac=1.0, random_state=self.seed)

        val_x = self._read(self.val_features_path)
        val_y = self._read(self.val_labels_path)

        features = [c for c in BASE_FEATURES if c in train.columns]
        guard_features(features)          # R1 运行时防线
        emit("phase", name="训练", detail=f"{fidelity} · {len(train):,} 行 · {len(features)} 个特征")

        for col in features:
            train[col] = train[col].astype("category")
            val_x[col] = val_x[col].astype("category")

        # 点击模型：全部行
        ctr_model = LGBMClassifier(n_estimators=120, num_leaves=31, learning_rate=0.05,
                                   random_state=self.seed, verbosity=-1)
        ctr_model.fit(train[features], train["click"], categorical_feature=features)

        # 购买模型：只用 click=1 的行
        clicked = train[train["click"] == 1]
        cvr_model = None
        if clicked["conversion"].nunique() >= 2:
            cvr_model = LGBMClassifier(n_estimators=60, num_leaves=15, learning_rate=0.05,
                                       random_state=self.seed, verbosity=-1)
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
                                  ctr_model, cvr_model, fidelity)

    def _build_report(self, val, ctr_pred, cvr_pred, train, features,
                      ctr_model, cvr_model, fidelity) -> dict[str, Any]:
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

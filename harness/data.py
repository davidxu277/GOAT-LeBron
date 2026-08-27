"""数据加载与预检 —— 真实执行器的数据层。

两个职责：
  1. preflight()：读一遍数据，报告规模与质量，按硬标准亮红绿灯。
     这把「数据够不够大」从口头约定变成可执行的检查。
  2. load_split()：把 train / val_features / val_labels 读进内存，
     顺带做 R1 禁用字段检查（架构级堵死，不靠自觉）。

判定标准的来源（见对话记录与 docs/方法库进度.md）：
AliCCP 真实比例约为「每 1 万次曝光 2 次转化」。购买 AUC 在点击子集上算，
验证集转化正样本少于 50 条时这个指标就是噪声，少于 2 条直接算不出来。
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# CLAUDE.md R1：这五个字段永远不许进入模型输入
FORBIDDEN_FIELDS = ("sample_id", "common_id", "click", "conversion", "ctcvr")

# 预检门槛
MIN_VAL_ROWS = 200_000          # 验证集行数下限（约对应 40 条转化）
COMFORTABLE_VAL_ROWS = 800_000  # 舒适区（约 160 条转化）
MIN_CONVERSIONS = 50            # 转化正样本下限，低于此购买 AUC 不可信
MIN_CLICKED_FOR_CVR = 2         # 低于此 roc_auc_score 直接抛异常


@dataclass
class Check:
    level: str      # ok / warn / bad
    text: str


@dataclass
class Preflight:
    """预检报告。level 取最严重的一项。"""
    rows: dict[str, int] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def level(self) -> str:
        levels = [c.level for c in self.checks]
        return "bad" if "bad" in levels else ("warn" if "warn" in levels else "ok")

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "rows": self.rows,
            "stats": self.stats,
            "checks": [{"level": c.level, "text": c.text} for c in self.checks],
        }


def _peek(path: pathlib.Path, nrows: int | None = None) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, nrows=nrows)


def preflight(train_path: str, val_features_path: str,
              val_labels_path: str | None = None) -> Preflight:
    """读一遍数据，报告规模与质量。不训练，纯检查。"""
    report = Preflight()
    tp, vp = pathlib.Path(train_path), pathlib.Path(val_features_path)

    for name, p in (("train", tp), ("val_features", vp)):
        if not p.exists():
            report.checks.append(Check("bad", f"{name} 文件不存在：{p}"))
            return report

    train = _peek(tp)
    val_features = _peek(vp)
    report.rows = {"train": len(train), "val_features": len(val_features)}

    # ── R1：验证集特征文件里绝不能带答案列 ──
    leaked = [c for c in ("click", "conversion", "ctcvr") if c in val_features.columns]
    if leaked:
        report.checks.append(Check(
            "bad", f"val_features 里出现了答案列 {leaked}——这等于把答案交给模型（R1）"))
    else:
        report.checks.append(Check("ok", "验证集特征文件不含答案列，R1 满足"))

    # ── 训练集标签 ──
    if "click" in train.columns and "conversion" in train.columns:
        n_click = int(train["click"].sum())
        n_conv = int(train["conversion"].sum())
        report.stats["train_clicks"] = n_click
        report.stats["train_conversions"] = n_conv
        bad_rows = int(((train["click"] == 0) & (train["conversion"] == 1)).sum())
        if bad_rows:
            report.checks.append(Check(
                "warn", f"训练集有 {bad_rows} 行 click=0 但 conversion=1（违反漏斗，应清除）"))
        if n_conv < 10:
            report.checks.append(Check(
                "bad", f"训练集只有 {n_conv} 条转化，购买模型学不出任何东西"))
        else:
            report.checks.append(Check(
                "ok", f"训练集 {len(train):,} 行 · 点击 {n_click:,} · 转化 {n_conv:,}"))
    else:
        report.checks.append(Check("bad", "训练集缺少 click / conversion 标签列"))

    # ── 验证集规模：决定购买 AUC 可不可信 ──
    n_val = len(val_features)
    if n_val < MIN_VAL_ROWS:
        report.checks.append(Check(
            "warn", f"验证集只有 {n_val:,} 行，低于 {MIN_VAL_ROWS:,} 的下限——"
                    f"按真实比例推算转化样本不足 40 条，购买 AUC 会是噪声"))
    elif n_val < COMFORTABLE_VAL_ROWS:
        report.checks.append(Check(
            "ok", f"验证集 {n_val:,} 行，够用（舒适区是 {COMFORTABLE_VAL_ROWS:,} 行以上）"))
    else:
        report.checks.append(Check("ok", f"验证集 {n_val:,} 行，规模充足"))

    # ── 用户切分：同一用户不该同时出现在两边 ──
    if "101" in train.columns and "101" in val_features.columns:
        train_users = set(train["101"].unique())
        val_users = set(val_features["101"].unique())
        overlap = len(train_users & val_users)
        report.stats["train_users"] = len(train_users)
        report.stats["val_users"] = len(val_users)
        report.stats["user_overlap"] = overlap
        if overlap:
            ratio = overlap / max(1, len(val_users))
            report.checks.append(Check(
                "warn" if ratio < 0.5 else "bad",
                f"{overlap:,} 个用户同时出现在训练集和验证集（占验证集用户 {ratio:.0%}）——"
                f"应该按用户整体切分，否则「新用户不会做」这个病永远测不出来"))
        else:
            report.checks.append(Check("ok", "按用户切分正确，两边用户无重叠"))

    # ── 私藏标签（若提供）──
    if val_labels_path:
        lp = pathlib.Path(val_labels_path)
        if not lp.exists():
            report.checks.append(Check("warn", f"val_labels 文件不存在：{lp}，无法评分"))
        else:
            labels = _peek(lp)
            report.rows["val_labels"] = len(labels)
            if len(labels) != n_val:
                report.checks.append(Check(
                    "bad", f"val_labels {len(labels):,} 行与 val_features {n_val:,} 行对不上"))
            n_val_conv = int(labels["conversion"].sum()) if "conversion" in labels else 0
            n_val_click = int(labels["click"].sum()) if "click" in labels else 0
            report.stats["val_clicks"] = n_val_click
            report.stats["val_conversions"] = n_val_conv
            if n_val_conv < MIN_CLICKED_FOR_CVR:
                report.checks.append(Check(
                    "bad", f"验证集只有 {n_val_conv} 条转化，购买 AUC 根本算不出来"))
            elif n_val_conv < MIN_CONVERSIONS:
                report.checks.append(Check(
                    "warn", f"验证集只有 {n_val_conv} 条转化（建议 ≥{MIN_CONVERSIONS}），"
                            f"购买 AUC 波动会非常大，不足以支撑「涨了还是没涨」的判断"))
            else:
                report.checks.append(Check(
                    "ok", f"验证集点击 {n_val_click:,} · 转化 {n_val_conv:,}，购买 AUC 可信"))

    return report


def guard_features(feature_cols: list[str]) -> None:
    """R1 运行时防线：特征清单与五个禁用字段求交集，非空立即终止。

    CLAUDE.md R1 原文要求的就是这个检查——静态正则只是纱窗，这里才是防盗门。
    """
    bad = set(FORBIDDEN_FIELDS) & set(map(str, feature_cols))
    if bad:
        raise ValueError(
            f"禁用字段混进特征列表：{sorted(bad)}（CLAUDE.md R1）。"
            f"这五个字段等于答案，进模型即作弊。"
        )

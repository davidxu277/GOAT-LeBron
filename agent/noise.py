"""噪声带 —— 一个知道自己的测量误差有多大的 Agent。

问题：`symptoms.yaml` 里的 0.03、0.04、0.002 全是拍出来的。
同一个 0.03，在小份数据的冷门桶上会把纯噪声当病报（那个桶只有几十条正样本，
AUC 本身就抖 ±0.08）；在全量数据上又会漏掉真实差距（那里抖动只有 ±0.003）。

办法跟量体温一样：先知道体温计的误差，再决定 37.2°C 算不算发烧。

    同一份配置，只换随机种子，跑 N 次 → 看每个数字自己抖多少 → 噪声带 = 2×标准差

再用 Hanley-McNeil 公式从正负样本数算一个理论值做交叉验证：
两个数量级对得上，说明这次测量可信；对不上，说明种子数不够或数据有问题。

    python -m agent.cli noise --seeds 3 --train ... --val-features ...

⚠️ 别拿一次运行的数字去"调"阈值 —— 那等于拿一次考试成绩定及格线。必须重复多次看抖动。
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
from typing import Any

# 噪声带 = 这么多倍标准差。2 倍 ≈ 95% 的抖动都落在带内。
BAND_SIGMAS = 2.0


def hanley_mcneil_se(auc: float, n_pos: int, n_neg: int) -> float | None:
    """AUC 的理论标准误。只需要正负样本数，不需要重复跑。

    Hanley & McNeil (1982)。正样本越少，这个数越大 —— 这正是
    "47 条正样本算出来的 0.702 不可信" 的定量说法。
    """
    if n_pos < 1 or n_neg < 1 or not (0.0 < auc < 1.0):
        return None
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1 - auc)
           + (n_pos - 1) * (q1 - auc * auc)
           + (n_neg - 1) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(var) if var > 0 else None


def _band(values: list[float]) -> dict[str, float]:
    clean = [v for v in values if v is not None]
    if len(clean) < 2:
        return {"均值": clean[0] if clean else 0.0, "标准差": 0.0, "噪声带": 0.0, "次数": len(clean)}
    sd = statistics.stdev(clean)
    return {
        "均值": round(statistics.fmean(clean), 5),
        "标准差": round(sd, 5),
        "噪声带": round(BAND_SIGMAS * sd, 5),
        "次数": len(clean),
    }


def _collect_buckets(reports: list[dict[str, Any]], group_key: str) -> dict[str, Any]:
    """把每个分桶在各个种子下的分数收集起来，逐桶算噪声带。"""
    out: dict[str, Any] = {}
    for report in reports:
        for row in report.get(group_key, []) or []:
            slot = out.setdefault(row["区间"], {"点击分": [], "购买分": [], "转化正样本数": []})
            slot["点击分"].append(row.get("点击分"))
            slot["购买分"].append(row.get("购买分"))
            slot["转化正样本数"].append(row.get("转化正样本数") or 0)
    result: dict[str, Any] = {}
    for name, slot in out.items():
        n_pos = int(statistics.fmean(slot["转化正样本数"])) if slot["转化正样本数"] else 0
        cvr_band = _band(slot["购买分"])
        entry = {"点击分": _band(slot["点击分"]), "购买分": cvr_band, "转化正样本数": n_pos}
        se = hanley_mcneil_se(cvr_band["均值"], n_pos, max(1, n_pos * 20))
        if se is not None:
            entry["购买分_理论噪声带"] = round(BAND_SIGMAS * se, 5)
        result[name] = entry
    return result


def summarize(reports: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    """把 N 次运行汇总成一份噪声带报告。纯计算，方便离线测试。"""
    ctr = _band([r.get("验证集", {}).get("点击分") for r in reports])
    cvr = _band([r.get("验证集", {}).get("购买分") for r in reports])
    buckets = {
        key: _collect_buckets(reports, key)
        for key in ("按商品出现次数分组", "按用户是否见过分组")
        if any(r.get(key) for r in reports)
    }

    bands = {
        "种子": seeds,
        "保真度": reports[0].get("保真度", "") if reports else "",
        "点击分": ctr,
        "购买分": cvr,
        # 复盘官拿它当"这次的提升是不是真的"的门槛：单项变化小于它就是噪声
        "单指标噪声带": round(max(ctr["噪声带"], cvr["噪声带"]), 5),
        "分组": buckets,
        "说明": ("同一配置只换随机种子跑出来的抖动。任何小于噪声带的差距都是噪声，"
                "不能当成病，也不能当成提升。"),
    }
    bands["表格"] = render(bands)
    return bands


def render(bands: dict[str, Any]) -> str:
    lines = [
        f"══════ 噪声带（{len(bands['种子'])} 个种子 · {bands['保真度'] or '?'}数据）══════",
        f"{'指标':<22}{'均值':>9}{'标准差':>10}{'噪声带':>10}",
        f"{'总体 点击分':<20}{bands['点击分']['均值']:>10.4f}"
        f"{bands['点击分']['标准差']:>10.4f}{bands['点击分']['噪声带']:>10.4f}",
        f"{'总体 购买分':<20}{bands['购买分']['均值']:>10.4f}"
        f"{bands['购买分']['标准差']:>10.4f}{bands['购买分']['噪声带']:>10.4f}",
    ]
    for group, rows in bands.get("分组", {}).items():
        lines.append(f"── {group} ──")
        for name, entry in rows.items():
            theory = entry.get("购买分_理论噪声带")
            lines.append(
                f"{name:<20}{entry['购买分']['均值']:>10.4f}"
                f"{entry['购买分']['标准差']:>10.4f}{entry['购买分']['噪声带']:>10.4f}"
                f"   正样本 {entry['转化正样本数']}"
                + (f" · 理论 {theory:.4f}" if theory else "")
            )
    lines += [
        "",
        f"→ 单指标噪声带 = {bands['单指标噪声带']:.4f}："
        f"任何小于它的「提升」一律记「说不清」",
        "→ 分桶差距的判定阈值应该逐桶用上面这一列，而不是 symptoms.yaml 里那个统一的 0.03",
    ]
    return "\n".join(lines)


def measure(*, train: str, val_features: str, val_labels: str | None,
            seeds: list[int], fidelity: str,
            out_path: pathlib.Path | None = None) -> dict[str, Any]:
    """真跑 N 次，量出噪声带。除了种子，什么都不改。"""
    from harness.executor import RealExecutor

    reports = []
    for seed in seeds:
        print(f"  种子 {seed} 跑中……", flush=True)
        ex = RealExecutor(train, val_features, val_labels, seed=seed)
        result = ex.run({"new_files": [], "config_patch": ""}, fidelity)
        if not result.ok:
            print(f"  ⚠️ 种子 {seed} 失败：{result.error}")
            continue
        reports.append(result.health_report)

    if len(reports) < 2:
        raise SystemExit("至少要有 2 次成功运行才能量抖动")

    bands = summarize(reports, seeds[:len(reports)])
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(bands, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n已写入 {out_path} —— 复盘官和医生下次跑会自动读它")
    return bands

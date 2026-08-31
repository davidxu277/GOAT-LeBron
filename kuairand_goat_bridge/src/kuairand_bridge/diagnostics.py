"""成绩单里的分组证据 —— 医生判病靠的就是这些块。

## 为什么要有这个文件

12 个病名里有 6 个的判定依据不是"验证集总分"，而是**分组之后的分数**：

    冷门视频排不上去   要按训练集曝光次数分 4 桶，比最低桶和最高桶
    新用户不会做       要把验证集用户分成"训练集里见过的"和"没见过的"
    在背题             要训练集和验证集各一份分数
    退化用户占比高     要 GAUC 参与用户数 ÷ 总用户数
    时间漂移           要验证集按日期分段的分数
    数据对不上         要预测行数、NaN、分数分布

这些数字不在成绩单里，医生就只能一轮一轮回「看不出来」—— 它没做错，
是我们没给证据。08-31 那场真跑的结果就是全程 no_finding、军师和工兵
一次都没被叫起来。

AliCCP 那条路（`harness/executor.py` 的 `_build_report`）本来就产出这些块，
换数据集时病名表迁过来了，成绩单没跟着迁。这个模块把它们补齐。

## 纪律

R2：所有**分组依据**（视频曝光次数、用户见没见过）只从 train 统计，
验证集只做查表。否则分桶本身就带了验证集的信息。

分组指标一律用官方 `evaluate` 算，不自己实现一遍 —— 自己写等于给自己
造一个"跟官方差一点"的分数，到最终提交才发现对不上。
"""

from __future__ import annotations

import collections
from typing import Any, Iterable, Sequence

import numpy as np

from .official import module as official_module


ROW_DATE = 0
ROW_USER = 1
ROW_VIDEO = 2
ROW_LABEL = 6

# 跟 knowledge/symptoms.yaml「冷门视频排不上去」里写的桶边界保持一致。
# 改这里就要同步改那边，否则医生拿着一套边界去读另一套数字。
EXPOSURE_EDGES = (10, 100, 1000)
EXPOSURE_NAMES = (
    "曝光<10次",
    "曝光10-100次",
    "曝光100-1000次",
    "曝光>1000次",
)

# 比这更小的分组不出分。几十行算出来的 GAUC 是纯噪声，
# 而医生被要求"证据必须带数字"—— 给它一个噪声数字，它就会拿噪声当病。
MIN_GROUP_ROWS = 200

# 训练集自评的行数上限。它只是"在背题"的判据，不需要全量精度，
# 而全量 114 万行多跑一遍前向在小份轮次上就成了最贵的一步。
DEFAULT_TRAIN_EVAL_ROWS = 200_000


def _evaluate(users: Sequence, labels: Sequence, scores: Sequence) -> dict[str, float]:
    """官方口径的用户内排序指标。"""
    return official_module("evaluate").evaluate(
        list(users), list(labels), list(scores)
    )


def _round(value: Any, digits: int = 4) -> Any:
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def _group_block(
    name: str,
    users: Sequence,
    labels: Sequence,
    scores: Sequence,
    total_rows: int,
) -> dict[str, Any]:
    """一个分组的成绩块。样本量不够就**明说不够**，不给数字。"""
    rows = len(labels)
    positives = int(sum(int(y) for y in labels))
    block: dict[str, Any] = {
        "分组": name,
        "行数": rows,
        "占比": _round(rows / total_rows if total_rows else 0.0),
        "正样本数": positives,
    }

    if rows < MIN_GROUP_ROWS or positives == 0 or positives == rows:
        block["说明"] = (
            f"只有 {rows} 行、{positives} 个正样本，"
            f"低于 {MIN_GROUP_ROWS} 行门槛，这个分组不出分"
        )
        return block

    metrics = _evaluate(users, labels, scores)
    block.update({
        "GAUC": _round(metrics["GAUC"]),
        "nDCG@5": _round(metrics["nDCG@5"]),
        "主分": _round(metrics["primary"]),
        "用户数": int(metrics["users"]),
    })
    return block


def _user_composition(users: Sequence, labels: Sequence) -> dict[str, Any]:
    """用户构成 ——「退化用户占比高」的判据。

    官方 GAUC **只统计 0 < 正例数 < 曝光数 的用户**，而 nDCG 对所有用户都算、
    零正例记 0。这两类用户占比一高，两个指标就会朝相反方向动，
    而只看总分完全看不出来。
    """
    by_user: dict[Any, list[int]] = collections.defaultdict(list)
    for user, label in zip(users, labels):
        by_user[user].append(int(label))

    total = len(by_user)
    participating = zero_positive = all_positive = 0

    for group in by_user.values():
        positives = sum(group)
        if positives == 0:
            zero_positive += 1
        elif positives == len(group):
            all_positive += 1
        else:
            participating += 1

    return {
        "总用户数": total,
        "GAUC参与用户数": participating,
        "GAUC参与用户占比": _round(participating / total if total else 0.0),
        "零正例用户占比": _round(zero_positive / total if total else 0.0),
        "全正例用户占比": _round(all_positive / total if total else 0.0),
        "口径": (
            "GAUC 只统计 0 < 正例数 < 曝光数 的用户；"
            "nDCG@5 对所有用户都算，零正例记 0"
        ),
    }


def _prediction_health(
    scores: np.ndarray,
    target_rows: int,
) -> dict[str, Any]:
    """预测本身健不健康 ——「数据对不上」的判据。"""
    values = np.asarray(scores, dtype=float).reshape(-1)
    finite = np.isfinite(values)
    clean = values[finite]

    return {
        "预测行数": int(values.size),
        "目标行数": int(target_rows),
        "行数一致": bool(values.size == target_rows),
        "NaN或Inf数": int((~finite).sum()),
        "最小值": _round(clean.min(), 6) if clean.size else None,
        "最大值": _round(clean.max(), 6) if clean.size else None,
        "去重后取值数": int(np.unique(clean).size) if clean.size else 0,
    }


def _exposure_buckets(train_rows: Iterable[Sequence]) -> dict[Any, int]:
    """视频在**训练集**里的曝光次数（R2：只数 train）。"""
    counter: collections.Counter = collections.Counter()
    for row in train_rows:
        counter[row[ROW_VIDEO]] += 1
    return counter


def _bucket_index(count: int) -> int:
    for index, edge in enumerate(EXPOSURE_EDGES):
        if count < edge:
            return index
    return len(EXPOSURE_EDGES)


def sample_rows_by_user(
    rows: Sequence[Sequence],
    max_rows: int,
    seed: int,
) -> list[Sequence]:
    """按**整个用户**抽样，不打散用户内部。

    GAUC 和 nDCG 都是用户内指标，随机抽行会把用户的曝光列表撕碎，
    算出来的分数和全量不可比 —— 那就不再是「训练集分数」了。
    """
    if len(rows) <= max_rows:
        return list(rows)

    by_user: dict[Any, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(rows):
        by_user[row[ROW_USER]].append(index)

    users = sorted(by_user)
    rng = np.random.default_rng(int(seed))
    rng.shuffle(users)

    picked: list[int] = []
    for user in users:
        if len(picked) >= max_rows:
            break
        picked.extend(by_user[user])

    picked.sort()
    return [rows[index] for index in picked]


def build(
    *,
    train_rows: Sequence[Sequence],
    valid_rows: Sequence[Sequence],
    valid_scores: Sequence[float],
    train_eval: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """组装医生要看的全部分组证据。

    参数
    ----
    train_rows / valid_rows:
        行元组，结构见 official_starter_kit/data.py。

    valid_scores:
        验证集预测分，顺序与 valid_rows 对齐。

    train_eval:
        训练集自评结果，由调用方跑一次 predict 得到（见 runner）。
        拿不到就整块不出现 —— 缺证据要**明说缺**，不要填 0 冒充。
    """
    scores = np.asarray(valid_scores, dtype=float).reshape(-1)

    if len(scores) != len(valid_rows):
        raise ValueError(
            f"验证集预测行数 {len(scores)} 与数据行数 {len(valid_rows)} 不一致"
        )

    users = [row[ROW_USER] for row in valid_rows]
    labels = [int(row[ROW_LABEL]) for row in valid_rows]
    total = len(valid_rows)

    report: dict[str, Any] = {
        # ⚠️ 这一条必须跟着分组一起给医生看。
        # 实测：小份上验证集总主分 0.5800，而按日期切开之后每天的主分只有
        # 0.50~0.53 —— 看着像"分数暴跌"，其实一天都没跌。nDCG@5 是**用户内**
        # 指标，把数据切小之后每个用户的曝光条数变少，nDCG 天然就低，
        # 跟模型好坏无关。不写明这一点，医生会拿分组分数去比总分，
        # 然后每一轮都报一个不存在的「时间漂移」。
        "分组口径提醒": (
            "分组分数只能跟**同一类**的其他分组比（这一天 vs 那一天、"
            "这一桶 vs 那一桶），绝不能跟验证集总分比。"
            "分组之后每个用户的曝光条数变少，nDCG@5 会系统性偏低，"
            "这是指标的性质，不是模型退步。"
        ),
        "用户构成": _user_composition(users, labels),
        "预测健康": _prediction_health(scores, total),
    }

    # ── 按视频曝光次数分桶（冷门视频排不上去）──
    exposure = _exposure_buckets(train_rows)
    by_bucket: dict[int, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(valid_rows):
        by_bucket[_bucket_index(exposure.get(row[ROW_VIDEO], 0))].append(index)

    report["按视频曝光次数分组"] = [
        _group_block(
            EXPOSURE_NAMES[bucket],
            [users[i] for i in by_bucket.get(bucket, ())],
            [labels[i] for i in by_bucket.get(bucket, ())],
            [scores[i] for i in by_bucket.get(bucket, ())],
            total,
        )
        for bucket in range(len(EXPOSURE_NAMES))
    ]

    # ── 按用户见没见过分堆（新用户不会做）──
    seen_users = {row[ROW_USER] for row in train_rows}
    groups: dict[str, list[int]] = {"训练集里见过的": [], "训练集里没见过的": []}
    for index, user in enumerate(users):
        key = "训练集里见过的" if user in seen_users else "训练集里没见过的"
        groups[key].append(index)

    report["按用户是否见过分组"] = [
        _group_block(
            name,
            [users[i] for i in idx],
            [labels[i] for i in idx],
            [scores[i] for i in idx],
            total,
        )
        for name, idx in groups.items()
    ]

    # ── 验证集按日期分段（时间漂移）──
    by_date: dict[Any, list[int]] = collections.defaultdict(list)
    for index, row in enumerate(valid_rows):
        by_date[row[ROW_DATE]].append(index)

    report["验证集按日期分组"] = [
        _group_block(
            str(date),
            [users[i] for i in by_date[date]],
            [labels[i] for i in by_date[date]],
            [scores[i] for i in by_date[date]],
            total,
        )
        for date in sorted(by_date)
    ]

    if train_eval:
        report["训练集"] = dict(train_eval)

    return report

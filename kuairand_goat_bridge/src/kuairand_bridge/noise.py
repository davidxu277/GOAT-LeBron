"""噪声带 —— 先知道体温计的误差，再决定 37.2°C 算不算发烧。

代码里到处在用「噪声带」判「这个差距算不算真的」（医生判分组差距算不算病、
复盘官判改动算不算有效、靠谱度账本判该不该加分），但在 KuaiRand 上从没量过，
一直用的是 R11 那个拍出来的 0.0005。

怎么量：同一份配置、**同一份数据子集**，只换训练种子跑 N 次。
数据子集必须钉死（sample_seed）—— 一场里轮与轮之间数据从头到尾不变，
抖动只来自训练随机性；换种子连数据一起换，会把抽样的抖动也量进去，门槛虚高，
真提升被当成噪声抹掉。

门槛怎么定：比的是**两次运行之间的差**。每次各带 σ 的抖动，差的标准差是 √2·σ，
取两倍 → 噪声带 = 2·√2·σ：两次同配置跑，差距约 95% 落在带内。
（旧 AliCCP 工具用 2σ —— 那是单次数值的带子，拿来卡差值偏松。）

⚠️ 带子只对量它的那个档位有效。正样本一多抖动就小，拿小份量的带子卡中份的结果
是一把过松的尺子。`run_session` 会自己核对档位，对不上就不用并写进结果表。

    python -m kuairand_bridge noise --config configs/kuairand_task.yaml --fidelity 小份 --seeds 5
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import time
from typing import Any, Callable

from .goat_executor import KuaiRandGoatExecutor
from .runner import run_trainer

BAND_SIGMAS = 2.0
MIN_SEEDS = 3               # 两个点算不出像样的标准差
主分 = "主分"
FILENAME = "noise_bands.json"


def _统计(values: list[float]) -> dict[str, float]:
    sd = statistics.stdev(values)
    return {
        "均值": round(statistics.fmean(values), 5),
        "标准差": round(sd, 5),
        "极差": round(max(values) - min(values), 5),
        "噪声带": round(BAND_SIGMAS * math.sqrt(2) * sd, 5),
        "次数": len(values),
    }


def bands_from_scores(runs: list[dict[str, float]]) -> dict[str, Any]:
    """一组同配置不同训练种子的分数 → 分指标噪声带。

    主分单独放进「单指标噪声带」：它是 GAUC 和 nDCG@5 的均值，判「这次改动算不算
    真提升」时要各比各的（两个指标的抖动可能差一个数量级，合成一个数会被抖得
    最凶的那个主导）。
    """
    if len(runs) < MIN_SEEDS:
        raise ValueError(f"至少要 {MIN_SEEDS} 个种子才能算噪声带，收到 {len(runs)} 个")
    指标 = [k for k in runs[0] if k != 主分]
    统计 = {k: _统计([float(r[k]) for r in runs]) for k in [*指标, 主分] if k in runs[0]}
    return {
        "分指标噪声带": {k: 统计[k]["噪声带"] for k in 指标},
        "单指标噪声带": 统计[主分]["噪声带"] if 主分 in 统计 else None,
        "统计": 统计,
    }


def _分数(report: dict[str, Any]) -> dict[str, float]:
    验证集 = report.get("验证集") or {}
    return {k: float(验证集[k]) for k in ("GAUC", "nDCG@5", 主分)}


def measure_noise(
    *,
    data_dir: str,
    trainer_path: str,
    output_dir: pathlib.Path,
    fidelity: str,
    train_seeds: list[int],
    sample_seed: int,
    trainer_config: dict[str, Any],
    runner: Callable[..., dict[str, Any]] = run_trainer,
    # 跟那一场的任务配置一致：开了锁定集，每一轮在开发集上打分，带子也得在开发集上量
    holdout_frac: float = 0.0,
) -> dict[str, Any]:
    """每个训练种子各跑一次**原样配置**（不带任何改动），算带子。

    有一次跑挂就整体报错：少一个点的标准差会偏小，给出一把过紧的尺子，
    纯抖动会被奖励成「猜对了」。宁可没有，不要歪的。
    """
    if len(train_seeds) < MIN_SEEDS:
        raise ValueError(f"至少要 {MIN_SEEDS} 个种子，收到 {len(train_seeds)} 个")
    output_dir = pathlib.Path(output_dir)
    逐次: list[dict[str, Any]] = []
    t0 = time.time()
    for seed in train_seeds:
        ex = KuaiRandGoatExecutor(
            data_dir=data_dir, trainer_path=trainer_path,
            output_dir=str(output_dir / f"seed_{seed}"),
            seed=seed, sample_seed=sample_seed, holdout_frac=holdout_frac,
            trainer_config=dict(trainer_config), runner=runner)
        r = ex.run({"new_files": [], "config_patch": ""}, fidelity)
        if not r.ok:
            raise RuntimeError(f"种子 {seed} 跑挂了，不拿残缺数据算带子：{r.error}")
        逐次.append({"训练种子": seed, "秒": round(r.seconds, 1), **_分数(r.health_report)})

    bands = bands_from_scores([{k: v for k, v in x.items() if k not in ("训练种子", "秒")}
                               for x in 逐次])
    return {
        "保真度": fidelity,
        "数据抽样种子": int(sample_seed),
        "锁定集比例": float(holdout_frac),
        **bands,
        "逐次分数": 逐次,
        "怎么算的": ("同一份配置、同一份数据子集，只换训练种子各跑一次；"
                  "噪声带 = 2×√2×标准差 —— 两次同配置跑，差距约 95% 落在带内"),
        "量于": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "总耗时秒": round(time.time() - t0, 1),
    }


def save_bands(logs_dir: pathlib.Path, bands: dict[str, Any]) -> pathlib.Path:
    path = pathlib.Path(logs_dir) / FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bands, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def load_bands(logs_dir: pathlib.Path) -> dict[str, Any] | None:
    """没量过就是 None —— run_session 会退回兜底门槛并在结果表里写明。"""
    path = pathlib.Path(logs_dir) / FILENAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

"""命令行入口。

  python -m agent.cli check                 只校验知识库，不调用模型，不花钱
  python -m agent.cli doctor 一切正常        用一份假成绩单跑医生
  python -m agent.cli doctor --all          跑全部 5 份，对照标准答案
  python -m agent.cli round 正常起步         用假执行器跑完整一轮
  python -m agent.cli run --offline         不花钱的整场演习（假模型+假执行器）
  python -m agent.cli run --rounds 10 --train ... --val-features ...
                                            真数据自主迭代，中途不需要人碰键盘
  python -m agent.cli noise --seeds 3 --train ...
                                            量噪声带：同配置换种子，看分数自己抖多少
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import yaml

from .knowledge import CardLibrary, SymptomVocab
from .llm import LLM, SchemaViolation
from .loop import (
    InterventionLog,
    DEFAULT_EPSILON,
    DEFAULT_PATIENCE,
    DEFAULT_ROUNDS,
    DEFAULT_TOKEN_BUDGET,
    CostAwareScheduler,
    FakeExecutor,
    PriorLedger,
    TimeLedger,
    read_scores,
    run_round,
    run_session,
)
from . import roles

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "agent" / "fixtures" / "health_reports.yaml"

# 工兵看到的三份材料，全部读自真实文件 —— 它照着真接口、真范文、真配置写代码。
# 按方案所属环节选范文：改训练过程的看训练类范文，加特征的看特征类范文。
INTERFACE_SPEC = (ROOT / "modules" / "base.py").read_text(encoding="utf-8")
PIPELINE_CONFIG = (ROOT / "config" / "pipeline.yaml").read_text(encoding="utf-8")

_EXAMPLES = {
    "训练": ROOT / "modules" / "train" / "early_stopping.py",
    "特征": ROOT / "modules" / "features" / "frequency_bucket.py",
}


def example_for(stage: str) -> str:
    """按环节挑范文。找不到对应的就用训练类那份（注释最全）。"""
    for key, path in _EXAMPLES.items():
        if key in (stage or "") and path.exists():
            return path.read_text(encoding="utf-8")
    return _EXAMPLES["训练"].read_text(encoding="utf-8")


LOGS = ROOT / "logs"


def make_llm(*args, **kwargs):
    """延迟导入真模型入口 —— check 和 run --offline 不该因为缺 SDK 就跑不起来。"""
    from .llm_deepseek import make_llm as _make
    return _make(*args, **kwargs)


def _load_fixtures() -> dict:
    return yaml.safe_load(FIXTURES.read_text(encoding="utf-8"))


def _add_data_args(p) -> None:
    """三条数据路径 + 锁定集。给了 --train 就用真执行器，不给就用假的。"""
    p.add_argument("--train", help="训练集：单个文件或分片目录")
    p.add_argument("--val-features", help="验证集特征")
    p.add_argument("--val-labels", help="验证集标签（验证集自带标签时可省）")
    p.add_argument("--holdout", help="锁定集（R3）：全程只读一次，用于收敛后的最终裁决，"
                                     "不给就跳过大考，不算错")
    p.add_argument("--seed", type=int, default=20260827, help="随机种子（CLAUDE.md R8）")


def _make_executor(args, fallback_report: dict):
    """有数据路径 → 真执行器；没有 → 假执行器。返回 (执行器, 是不是真的)。"""
    if not args.train:
        return FakeExecutor(next_report=fallback_report), False
    if not args.val_features:
        raise SystemExit("给了 --train 就必须给 --val-features")
    from harness.executor import RealExecutor
    return RealExecutor(args.train, args.val_features, args.val_labels,
                        seed=args.seed,
                        holdout_path=getattr(args, "holdout", None)), True


def _initial_report(executor, real: bool, fallback: dict,
                    fidelity: str = "小份") -> tuple[dict, float]:
    """第 0 轮体检：真执行器就原样跑一次拿真成绩单，假的就用假成绩单。

    医生必须看着真数字做诊断 —— 拿假成绩单配真训练是最容易骗到自己的组合。
    体检必须跑在起步档位上，否则第 1 轮的分数没法跟它比。
    """
    if not real:
        return fallback, 0.0
    print(f"体检中（在{fidelity}数据上原样跑一次，拿第 0 轮成绩单）……")
    first = executor.run({"new_files": [], "config_patch": ""}, fidelity)
    if not first.ok:
        raise SystemExit(f"❌ 第 0 轮就没跑起来：{first.error}")
    print(f"第 0 轮：{read_scores(first.health_report)}  用时 {first.seconds:.0f}s\n")
    return first.health_report, first.seconds


def _noise_bands() -> dict:
    """读实测噪声带。没测过就返回空 dict（见 agent/noise.py）。"""
    path = LOGS / "noise_bands.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _noise_floor(bands: dict | None = None) -> float:
    """判断"这次提升是不是真的"的门槛。没测过就退回 R11 的 0.0005。"""
    bands = _noise_bands() if bands is None else bands
    return float(bands.get("单指标噪声带") or roles.MIN_REAL_GAIN)


def _show(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False, indent=2))


NO_CREDS = (
    "找不到 Anthropic 凭据。设置一下再跑：\n"
    "    export ANTHROPIC_API_KEY=sk-ant-...\n\n"
    "只想检查知识库是否自洽的话，用 `python -m agent.cli check`，它不调用模型。"
)


def cmd_check(args) -> int:
    """只校验知识库是否自洽。不调用模型，零成本。"""
    vocab = SymptomVocab.load()
    print(f"病名词表：{len(vocab.ids)} 个")
    for sid in vocab.ids:
        mark = "★" if vocab[sid].core else " "
        print(f"  {mark} {sid}")
    cards = CardLibrary.load(vocab)
    print(f"\n药方卡：{len(cards)} 张（全部标签合法）")
    for c in cards.cards:
        print(f"    {c.id:<12} 治 {', '.join(c.treats)}")

    fixtures = _load_fixtures()
    print(f"\n假成绩单：{len(fixtures)} 份 —— {', '.join(fixtures)}")

    uncovered = [
        sid for sid in vocab.ids
        if not any(sid in c.treats for c in cards.cards)
    ]
    if uncovered:
        print(f"\n⚠️  还没有卡片对症的病（{len(uncovered)} 个）：")
        print("   " + "、".join(uncovered))
    return 0


def cmd_doctor(args) -> int:
    vocab = SymptomVocab.load()
    fixtures = _load_fixtures()
    names = list(fixtures) if args.all else [args.name]
    llm = make_llm()

    for name in names:
        if name not in fixtures:
            print(f"没有这份假成绩单：{name}（可选：{', '.join(fixtures)}）")
            return 2
        print(f"\n{'=' * 66}\n【{name}】\n{'=' * 66}")
        try:
            out = roles.diagnose(llm, vocab, fixtures[name]["report"])
        except SchemaViolation as exc:
            print(f"❌ 医生输出不合格：{exc}")
            continue
        if out["no_finding"]:
            print(f"没查出明显问题 —— {out['reason_if_none']}")
        for f in out["findings"]:
            print(
                f"  · {f['symptom']}（严重 {f['severity']:.2f}，确定 {f['confidence']}）\n"
                f"    {f['evidence']}"
            )
        print(f"\n  标准答案：{fixtures[name]['expect'].strip()}")

    print(f"\n{'=' * 66}\n{llm.ledger.report()}")
    return 0


def cmd_round(args) -> int:
    vocab = SymptomVocab.load()
    cards = CardLibrary.load(vocab)
    fixtures = _load_fixtures()
    if args.name not in fixtures:
        print(f"没有这份假成绩单：{args.name}")
        return 2

    report = fixtures[args.name]["report"]
    llm = make_llm()
    # 两个账本：耗时（实测倍数覆盖卡上拍的数）、靠谱度（复盘结论累积）
    time_ledger = TimeLedger.load(LOGS / "time_ledger.json")
    prior_ledger = PriorLedger.load(LOGS / "prior_ledger.json")
    prior_ledger.apply_to(cards)
    executor, real = _make_executor(args, report)
    if real:
        print(f"真执行器：{args.train}")
    report, _ = _initial_report(executor, real, report)
    log = run_round(
        round_id=1,
        llm=llm,
        vocab=vocab,
        cards=cards,
        health_report=report,
        parent_result=report,
        executor=executor,
        scheduler=CostAwareScheduler(time_ledger=time_ledger),
        module_interface=INTERFACE_SPEC,
        example_module=example_for,      # 按方案环节选范文
        current_config=PIPELINE_CONFIG,
        time_ledger=time_ledger,
        prior_ledger=prior_ledger,
        noise_floor=_noise_floor(),
    )
    time_ledger.dump(LOGS / "time_ledger.json")
    prior_ledger.dump(LOGS / "prior_ledger.json")

    print("\n【诊断】")
    _show(log.diagnosis)
    if log.proposals:
        print("\n【提案】")
        _show(log.proposals)
    if log.chosen:
        print(f"\n【调度器选中】{log.chosen['card_id'] or '自创方案'}  →  {log.fidelity}数据")
    if log.patch_summary:
        print("\n【工兵产出】")
        _show(log.patch_summary)
    if log.reflection:
        print("\n【复盘】")
        _show(log.reflection)
    if log.recoveries:
        print("\n【出错与恢复】")
        for r in log.recoveries:
            print(f"  · {r}")

    log.dump(LOGS / "rounds.jsonl")
    print(f"\n{llm.ledger.report()}")
    print(f"本轮耗时 {log.seconds:.1f}s，日志已追加到 logs/rounds.jsonl")
    return 0


def cmd_run(args) -> int:
    """自主迭代：一轮接一轮，中途不需要人碰键盘。

    这条命令就是赛题里「自主性」那一项的实现。跑完会打印结果表。
    """
    vocab = SymptomVocab.load()
    cards = CardLibrary.load(vocab)

    logs_dir = LOGS
    if getattr(args, "fresh", False):
        # 之前那些"改动其实没生效却被判无效"的轮次，已经把靠谱度和黑名单污染了。
        # 正式跑之前清一次，别让 Agent 带着一堆错误的偏见开跑。
        清掉 = []
        for name in ("prior_ledger.json", "time_ledger.json", "shelf.json"):
            f = LOGS / name
            if f.exists():
                f.unlink()
                清掉.append(name)
        print(f"已清空历史记忆：{', '.join(清掉) or '（本来就是空的）'}\n")
    if args.offline:
        from .offline import DriftingExecutor, ScriptedLLM
        faults = {"医生": [args.fail_role_call]} if args.fail_role_call else {}
        llm = ScriptedLLM(faults=faults)
        executor = DriftingExecutor(fail_rounds=tuple(args.fail_round or ()))
        initial, first_seconds = executor.report("小份"), 0.0
        # 演习产出的假日志绝不能混进交付物 —— 单独一个目录
        logs_dir = LOGS / "offline"
        print(f"离线演习：假模型 + 假执行器，不联网不花钱（日志写 {logs_dir}）\n")
    else:
        llm = make_llm()
        fallback = _load_fixtures()[args.name]["report"]
        executor, real = _make_executor(args, fallback)
        if real:
            print(f"真执行器：{args.train}")
        else:
            print(f"假执行器 + 假成绩单「{args.name}」——"
                  f"要跑真数据请给 --train / --val-features\n")
        initial, first_seconds = _initial_report(
            executor, real, fallback, fidelity=args.start_fidelity)

    baseline = {}
    if args.baseline_ctr is not None:
        baseline["点击AUC"] = args.baseline_ctr
    if args.baseline_cvr is not None:
        baseline["购买AUC"] = args.baseline_cvr

    def on_round(log, summary) -> None:
        ref = log.reflection or {}
        scores = read_scores(log.metrics or {})
        line = (f"第 {log.round_id:>2} 轮 · {log.fidelity or '—'} · "
                f"{(log.chosen or {}).get('card_id') or '（自创/未选）'} · "
                f"{ref.get('verdict', '本轮作废')}")
        if scores:
            line += f" · 点击 {scores['点击AUC']:.4f} 购买 {scores['购买AUC']:.4f}"
        print(line)
        for r in log.recoveries:
            print(f"        ↳ 恢复：{r}")

    bands = _noise_bands()
    if bands:
        print(f"读到实测噪声带：单指标 {bands['单指标噪声带']:.4f}\n")
    summary = run_session(
        llm=llm, vocab=vocab, cards=cards, executor=executor,
        initial_report=initial,
        initial_train_seconds=first_seconds,
        module_interface=INTERFACE_SPEC,
        example_module=example_for,      # 按方案环节选范文
        current_config=PIPELINE_CONFIG,
        rounds=args.rounds,
        start_fidelity=args.start_fidelity,
        token_budget=args.token_budget,
        epsilon=args.epsilon,
        patience=args.patience,
        noise_floor=_noise_floor(bands),
        noise_bands=bands,
        baseline=baseline,
        logs_dir=logs_dir,
        on_round=on_round,
    )
    print(f"\n{summary.as_table()}")
    print(f"\n{llm.ledger.report()}")
    rel = logs_dir.relative_to(ROOT)
    print(f"\n我交这一版：第 {summary.best_round} 轮")
    print(f"  分数记录 {rel}/best_report.json · 结果表 {rel}/session_summary.json")
    print(f"  交付物 #4（模型/预测结果）还要照着这一版的配方重跑一次："
          f"agent.cli finalize 会把配方整理到 deliverables/best_pipeline/")
    return 0


def cmd_intervene(args) -> int:
    """记一次人工干预。

    报出来的「干预 0 次」要有分量，前提是"非零"随手可得 ——
    一个只能是 0 的数字，评委翻一眼代码就知道不算数。
    """
    InterventionLog.record(LOGS / "interventions.jsonl", args.reason, args.round)
    print(f"已记一次人工干预：{args.reason}")
    print(f"（写入 logs/interventions.jsonl，正在跑的那一场下一轮就会把它记进日志）")
    return 0


def cmd_restore(args) -> int:
    """把某一轮的流水线原样还原出来 —— 交付物 #4 靠它。

    工兵的改动是叠加在同一份配置和同一个 modules/ 上的，跑完 20 轮，
    磁盘上只剩最后那个叠加态。要交"验证集最佳的那一版"，就得从日志里还原。
    """
    logs = pathlib.Path(args.logs)
    name = f"round_{args.round:03d}.json"
    run = getattr(args, "run", None)
    # 快照按场分目录。没指定哪一场就挑最新的那一场
    候选 = ([logs / "snapshots" / run / name] if run
            else sorted((logs / "snapshots").glob(f"*/{name}")))
    候选 = [p_ for p_ in 候选 if p_.exists()]
    if not 候选:
        raise SystemExit(f"没有第 {args.round} 轮的快照（找过 {logs / 'snapshots'}）")
    snap_path = 候选[-1]
    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    这一场 = snap_path.parent.name

    rows = [json.loads(l) for l in
            (logs / "rounds.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    # 只认同一场的记录 —— 轮次编号每场都从 1 重数，跨场取会拿到别人那一轮的代码
    by_round = {r["round_id"]: r for r in rows if (r.get("run_id") or "") == 这一场}
    if not by_round:
        by_round = {r["round_id"]: r for r in rows}      # 旧日志没有 run_id，退回全量

    out = pathlib.Path(args.out)
    (out / "config").mkdir(parents=True, exist_ok=True)
    (out / "config" / "pipeline.yaml").write_text(snap["配置"], encoding="utf-8")

    missing = []
    for rel, owner in (snap["零件"] or {}).items():
        content = (by_round.get(owner, {}).get("patch_files") or {}).get(rel)
        if content is None:
            missing.append(f"{rel}（第 {owner} 轮）")
            continue
        target = out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    print(f"第「{这一场}」场第 {args.round} 轮已还原到 {out}")
    print(f"  配置 1 份 · 零件 {len(snap['零件'] or {}) - len(missing)} 个 · 分数 {snap['分数']}")
    if missing:
        print(f"  ⚠️ 这些零件在日志里找不到内容：{', '.join(missing)}")
    return 1 if missing else 0


def cmd_finalize(args) -> int:
    """把一场跑的产物整理成可提交的一包。

    最后一天不该再想"该交什么、在哪" —— 一条命令出齐，照着清单核对就行。

    只取**一场**（run_id）的记录：日志是追加的、轮次每场都从 1 重数，
    几个人各跑几次混在一个文件里，评委读到的会是 [1,2,3,1,2,3,4] 这么一串。
    """
    logs = pathlib.Path(args.logs)
    out = pathlib.Path(args.out)
    rounds_path = logs / "rounds.jsonl"
    if not rounds_path.exists():
        raise SystemExit(f"没有逐轮日志：{rounds_path}")

    rows = [json.loads(l) for l in
            rounds_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    run_id = args.run or (rows[-1].get("run_id") or "")
    mine = [r for r in rows if (r.get("run_id") or "") == run_id]
    if not mine:
        有哪些 = sorted({r.get("run_id") or "(无编号)" for r in rows})
        raise SystemExit(f"日志里没有第「{run_id}」场。现有：{', '.join(有哪些)}")

    out.mkdir(parents=True, exist_ok=True)
    # ① 逐轮日志：只留这一场，轮次编号才是连续可读的
    (out / "rounds.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in mine) + "\n", encoding="utf-8")

    # ② 叙事 / ③ 结果表 / ④ 最佳版本的成绩单
    带过去 = []
    for name in ("narrative.md", "session_summary.json", "best_report.json",
                 "noise_bands.json", "interventions.jsonl"):
        src = logs / name
        if src.exists():
            (out / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
            带过去.append(name)

    # ⑤ 最佳那一轮的流水线 —— 交付物 #4 要照着它重跑出预测结果
    summary = {}
    sp = logs / "session_summary.json"
    if sp.exists():
        summary = json.loads(sp.read_text(encoding="utf-8"))
    best = summary.get("best_round") or 0
    restored = "（没有 session_summary.json，跳过）"
    if best:
        rc = cmd_restore(argparse.Namespace(
            round=best, out=str(out / "best_pipeline"), logs=str(logs), run=run_id))
        restored = f"第 {best} 轮 → {out / 'best_pipeline'}" + ("（有缺件，见上）" if rc else "")

    # ⑥ 评委可读的看板
    try:
        sys.path.insert(0, str(ROOT / "web"))
        import build_report                                   # noqa: PLC0415
        (out / "dashboard.html").write_text(build_report.build(mine), encoding="utf-8")
        带过去.append("dashboard.html")
    except Exception as exc:                                  # noqa: BLE001
        print(f"⚠️ 看板没生成：{exc}（可以手动跑 web/build_report.py）")

    print(f"\n══════════ 提交包已备好 · 第「{run_id}」场 ══════════")
    print(f"目录：{out}")
    print(f"  逐轮日志      {len(mine)} 轮 —— 交付物 #3")
    print(f"  结果表        session_summary.json —— 交付物 #5")
    print(f"  一起带过去    {', '.join(带过去) or '—'}（佐证材料，不是提交物）")
    print(f"  最佳版本配方  {restored}")
    print(f"\n⚠️ 交付物 #4（模型本身）这一步产不出来 —— 上面给的是**配方**。")
    print(f"   照着 best_pipeline/ 重跑一次，导出预测结果或 checkpoint，那个才是要交的。")
    print(f"\n还要人做的：")
    print(f"  · 把 best_pipeline/ 重跑一次并导出提交文件（等 Starter Kit 的输出 Schema）")
    print(f"  · README 的「局限性与改进方向」（交付物 #2 明确要求）")
    return 0


def cmd_noise(args) -> int:
    """量噪声带：同一份配置换种子跑几次，看分数自己抖多少。"""
    from .noise import measure

    if not args.train:
        raise SystemExit("量噪声带必须给真数据：--train / --val-features")
    bands = measure(
        train=args.train, val_features=args.val_features, val_labels=args.val_labels,
        seeds=[args.seed + i for i in range(args.seeds)],
        fidelity=args.fidelity,
        out_path=LOGS / "noise_bands.json",
    )
    print(bands["表格"])
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="校验知识库，不调用模型").set_defaults(func=cmd_check)

    p = sub.add_parser("doctor", help="用假成绩单跑医生")
    p.add_argument("name", nargs="?", default="正常起步")
    p.add_argument("--all", action="store_true", help="跑全部 5 份")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("round", help="跑完整一轮")
    p.add_argument("name", nargs="?", default="正常起步", help="没给真数据时用哪份假成绩单")
    _add_data_args(p)
    p.set_defaults(func=cmd_round)

    p = sub.add_parser("run", help="自主迭代 N 轮，中途不碰键盘")
    p.add_argument("name", nargs="?", default="正常起步", help="没给真数据时用哪份假成绩单")
    _add_data_args(p)
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--start-fidelity", default="小份",
                   help="从哪一档数据起步：小份 / 中份 / 大份 / 全量")
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON,
                   help="提升小于它不算提升。Starter Kit 给了官方 ε 就换成官方的")
    p.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                   help="连续几轮没有真提升算收敛")
    p.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    p.add_argument("--baseline-ctr", type=float, help="官方基线的点击 AUC，用来算 delta")
    p.add_argument("--baseline-cvr", type=float, help="官方基线的购买 AUC，用来算 delta")
    p.add_argument("--fresh", action="store_true",
                   help="清空历史账本再跑 —— 正式那一场应该带上它，"
                        "免得带着开发期间攒下的错误偏见开跑")
    p.add_argument("--offline", action="store_true",
                   help="演习模式：假模型 + 假执行器，不联网不花钱")
    p.add_argument("--fail-round", type=int, action="append",
                   help="演习：让第几轮训练失败（可重复）")
    p.add_argument("--fail-role-call", type=int,
                   help="演习：让医生的第几次调用抛异常")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("finalize", help="把一场跑的产物整理成可提交的一包")
    p.add_argument("--run", help="哪一场（run_id）。默认最后那一场")
    p.add_argument("--out", default="deliverables", help="整理到哪个目录")
    p.add_argument("--logs", default=str(LOGS), help="从哪份日志整理")
    p.set_defaults(func=cmd_finalize)

    p = sub.add_parser("intervene", help="记一次人工干预（跑的过程中插了手就敲一条）")
    p.add_argument("reason", help="干了什么、为什么")
    p.add_argument("--round", type=int, help="当时第几轮（可选）")
    p.set_defaults(func=cmd_intervene)

    p = sub.add_parser("restore", help="把某一轮的流水线还原出来（交付物 #4）")
    p.add_argument("round", type=int, help="要还原第几轮")
    p.add_argument("--out", default="restored", help="还原到哪个目录")
    p.add_argument("--logs", default=str(LOGS), help="从哪份日志还原")
    p.add_argument("--run", help="哪一场（run_id）。默认最新那一场")
    p.set_defaults(func=cmd_restore)

    p = sub.add_parser("noise", help="量噪声带：同配置换种子看分数抖多少")
    _add_data_args(p)
    p.add_argument("--seeds", type=int, default=3, help="跑几个种子")
    p.add_argument("--fidelity", default="小份", help="在哪个数据档位上量")
    p.set_defaults(func=cmd_noise)

    args = parser.parse_args()
    try:
        return args.func(args)
    except TypeError as exc:
        # SDK 在第一次发请求时才抛缺凭据的错，给一句人话而不是一坨堆栈
        if "Could not resolve authentication" in str(exc):
            print(NO_CREDS)
            return 1
        raise


if __name__ == "__main__":
    sys.exit(main())

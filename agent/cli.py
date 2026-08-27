"""命令行入口。

  python -m agent.cli check                 只校验知识库，不调用模型，不花钱
  python -m agent.cli doctor 一切正常        用一份假成绩单跑医生
  python -m agent.cli doctor --all          跑全部 5 份，对照标准答案
  python -m agent.cli round 正常起步         用假执行器跑完整一轮
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import yaml

from .knowledge import CardLibrary, SymptomVocab
from .llm import LLM, SchemaViolation
from .llm_deepseek import make_llm
from .loop import CostAwareScheduler, FakeExecutor, TimeLedger, run_round
from . import roles

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "agent" / "fixtures" / "health_reports.yaml"

# 工兵的范文与接口。成员2 接手后换成真的。
STUB_INTERFACE = """\
class FeatureOp:
    def fit(self, train_df) -> None:
        \"\"\"只在训练集上统计。绝不许读验证集。\"\"\"
    def transform(self, df):
        \"\"\"返回加工后的 DataFrame。\"\"\"
"""
STUB_EXAMPLE = """\
from modules.base import FeatureOp


class FrequencyBucket(FeatureOp):
    \"\"\"把商品出现次数分桶，作为一个新的类别特征。\"\"\"

    def __init__(self, config):
        self.field = config["field"]
        self.edges = config["edges"]
        self.counts = None

    def fit(self, train_df):
        self.counts = train_df[self.field].value_counts()

    def transform(self, df):
        freq = df[self.field].map(self.counts).fillna(0)
        df[f"{self.field}_freq_bucket"] = freq.searchsorted(self.edges)
        return df
"""
STUB_CONFIG = "features:\n  frequency_bucket:\n    enabled: false\nmodel:\n  name: mlp\n"


def _load_fixtures() -> dict:
    return yaml.safe_load(FIXTURES.read_text(encoding="utf-8"))


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
    # 耗时账本：实测倍数覆盖卡上拍的「训练时间倍数」（假执行器耗时为 0，不会记账）
    ledger_path = ROOT / "logs" / "time_ledger.json"
    time_ledger = TimeLedger.load(ledger_path)
    log = run_round(
        round_id=1,
        llm=llm,
        vocab=vocab,
        cards=cards,
        health_report=report,
        parent_result=report,
        # 假执行器直接回放同一份成绩单：链路能跑通即可，分数无意义
        executor=FakeExecutor(next_report=report),
        scheduler=CostAwareScheduler(time_ledger=time_ledger),
        module_interface=STUB_INTERFACE,
        example_module=STUB_EXAMPLE,
        current_config=STUB_CONFIG,
        time_ledger=time_ledger,
    )
    time_ledger.dump(ledger_path)

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

    log.dump(ROOT / "logs" / "rounds.jsonl")
    print(f"\n{llm.ledger.report()}")
    print(f"本轮耗时 {log.seconds:.1f}s，日志已追加到 logs/rounds.jsonl")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check", help="校验知识库，不调用模型").set_defaults(func=cmd_check)

    p = sub.add_parser("doctor", help="用假成绩单跑医生")
    p.add_argument("name", nargs="?", default="正常起步")
    p.add_argument("--all", action="store_true", help="跑全部 5 份")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("round", help="用假执行器跑完整一轮")
    p.add_argument("name", nargs="?", default="正常起步")
    p.set_defaults(func=cmd_round)

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

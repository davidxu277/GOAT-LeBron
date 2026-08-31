"""把一场跑的产物整理成交付物 #3 —— 每轮的假设 / 代码 diff / 指标 / 错误与恢复。

四份原料，各补各的缺口：

  logs/transcript-*.jsonl   四个角色的完整往返（提示词 + 回答），本工具链自己存的
  logs/live_events.jsonl    模型原文流，含思维链（kind=reasoning）
  <output_dir>/logs/rounds.jsonl  每轮结构化产物：诊断/提案/补丁全文/成绩单/复盘
  <output_dir>/logs/session_summary.json  结果表

产出 deliverables/run_logs/ 下：
  每轮一个 markdown（人读）+ 一份合并 jsonl（机读）+ 结果表

用法：
    python3 tools/collect_logs.py [output_dir]      默认 kuairand_goat_bridge/output/agent_run
"""

from __future__ import annotations

import collections
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def _读jsonl(path: pathlib.Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass          # 跑到一半被 Ctrl+C，最后一行可能是半条
    return out


def _思维链(events: list[dict]) -> dict[str, str]:
    """把散成 delta 的模型原文按角色拼回去。"""
    buf: dict[str, list[str]] = collections.defaultdict(list)
    for e in events:
        if e.get("type") == "llm_delta" and e.get("kind") == "reasoning":
            buf[e.get("role", "?")].append(e.get("text", ""))
    return {k: "".join(v) for k, v in buf.items()}


def main() -> int:
    out_dir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "kuairand_goat_bridge" / "output" / "agent_run")
    # 两种布局都认：bridge 那一场写 <out>/logs/，离线演习直接写 logs/offline/
    logs = out_dir / "logs"
    if not (logs / "rounds.jsonl").exists() and (out_dir / "rounds.jsonl").exists():
        logs = out_dir
    rounds = _读jsonl(logs / "rounds.jsonl")
    if not rounds:
        print(f"❌ 在 {out_dir} 下找不到 rounds.jsonl（试过 ./logs/ 和 ./）"
              f" —— 这一场还没跑，或路径给错了")
        return 1

    transcripts: list[dict] = []
    for p in sorted((ROOT / "logs").glob("transcript-*.jsonl")):
        transcripts += _读jsonl(p)
    按角色 = collections.defaultdict(list)
    for t in transcripts:
        按角色[t["角色"]].append(t)

    dest = ROOT / "deliverables" / "run_logs"
    dest.mkdir(parents=True, exist_ok=True)

    for r in rounds:
        rid = r.get("round_id", 0)
        lines = [f"# 第 {rid} 轮\n",
                 f"- 数据档位：{r.get('fidelity') or '—'}",
                 f"- 跑成功：{r.get('run_ok')}",
                 f"- 耗时：{r.get('seconds', 0):.1f}s（其中训练 {r.get('train_seconds', 0):.1f}s）",
                 f"- token：{r.get('tokens', 0):,}",
                 f"- 人工干预：{r.get('interventions', 0)} 次\n"]

        if r.get("diagnosis"):
            lines.append("## 医生 —— 这一轮的病\n")
            lines.append("```json\n" + json.dumps(r["diagnosis"], ensure_ascii=False, indent=1) + "\n```\n")
        if r.get("proposals"):
            lines.append("## 军师 —— 提案（交付物要的「假设」）\n")
            lines.append("```json\n" + json.dumps(r["proposals"], ensure_ascii=False, indent=1) + "\n```\n")
        if r.get("chosen"):
            lines.append("## 调度器 —— 选中的方案\n")
            lines.append("```json\n" + json.dumps(r["chosen"], ensure_ascii=False, indent=1) + "\n```\n")
        if r.get("patch_files"):
            lines.append("## 工兵 —— 代码 diff（新建的零件全文）\n")
            for path, code in r["patch_files"].items():
                lines.append(f"### `{path}`\n\n```python\n{code}\n```\n")
        if r.get("metrics"):
            lines.append("## 指标 —— 本轮成绩单\n")
            lines.append("```json\n" + json.dumps(r["metrics"], ensure_ascii=False, indent=1) + "\n```\n")
        if r.get("reflection"):
            lines.append("## 复盘官 —— 这一改到底有没有效\n")
            lines.append("```json\n" + json.dumps(r["reflection"], ensure_ascii=False, indent=1) + "\n```\n")
        if r.get("recoveries"):
            lines.append("## 错误与恢复\n")
            lines += [f"- {x}" for x in r["recoveries"]] + [""]
        if r.get("intervention_notes"):
            lines.append("## 人工干预记录\n")
            lines += [f"- {x}" for x in r["intervention_notes"]] + [""]

        (dest / f"round_{rid:03d}.md").write_text("\n".join(lines), encoding="utf-8")

    with (dest / "all_rounds.jsonl").open("w", encoding="utf-8") as fh:
        for r in rounds:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    if transcripts:
        with (dest / "role_transcripts.jsonl").open("w", encoding="utf-8") as fh:
            for t in transcripts:
                fh.write(json.dumps(t, ensure_ascii=False) + "\n")

    summary = logs / "session_summary.json"
    if summary.exists():
        (dest / "session_summary.json").write_text(
            summary.read_text(encoding="utf-8"), encoding="utf-8")

    reasoning = _思维链(_读jsonl(ROOT / "logs" / "live_events.jsonl"))
    if reasoning:
        (dest / "reasoning_by_role.json").write_text(
            json.dumps(reasoning, ensure_ascii=False, indent=1), encoding="utf-8")

    干预 = sum(r.get("interventions", 0) for r in rounds)
    print(f"✅ 整理完毕 → {dest.relative_to(ROOT)}")
    print(f"   轮次 {len(rounds)} · 角色调用留痕 {len(transcripts)} 条"
          f"（{dict((k, len(v)) for k, v in 按角色.items())}）")
    print(f"   人工干预合计 {干预} 次 —— 交付物 #3 要单独报这个数（评自主性）")
    print(f"   token 合计 {sum(r.get('tokens', 0) for r in rounds):,}"
          f" · 训练总时长 {sum(r.get('train_seconds', 0) for r in rounds)/3600:.3f} 小时")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""把 logs/rounds.jsonl 渲染成一个自包含的单页 HTML（看板 V0 · 静态回放）。

用法：
    python web/build_report.py                 # 默认读 logs/rounds.jsonl，写 logs/report.html
    python web/build_report.py 日志路径 输出路径

零依赖、零构建：产出的 HTML 双击即可打开，不需要服务器。
它同时就是交付物 #3 逐轮日志的「评委可读版」——评委不用啃 jsonl。

布局：横版一屏铺满（100vh），左侧轮次列表 + 右侧四宫格便当盒，
页面本身不滚动，只有单个格子内容过长时格子内部滚动。
"""

from __future__ import annotations

import html
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_IN = ROOT / "logs" / "rounds.jsonl"
DEFAULT_OUT = ROOT / "logs" / "report.html"

VERDICT_TONE = {"猜对了": "good", "猜错了": "bad", "说不清": "muted", "没跑起来": "bad"}

# 每 100 万 token 的价格（美元），与 agent/llm.py 的 PRICING 保持一致。
# 日志里只记了 token 总数没记模型，按默认模型估算，是保守上限。
PRICE_PER_MTOK = float(__import__("os").getenv("REPORT_PRICE_PER_MTOK", "0.44"))


def estimate_cost(rows: list[dict]) -> float:
    """按 token 总量估算花费。输入输出价不同，这里取中间值做粗估。"""
    return sum(r.get("tokens", 0) for r in rows) / 1e6 * PRICE_PER_MTOK


def esc(x) -> str:
    return html.escape(str(x))


def fmt_delta(v) -> str:
    """把数值格式化成带正负号的样子；非数值原样返回。"""
    if isinstance(v, (int, float)):
        return f"{v:+.4f}" if abs(v) < 1 else f"{v:.4f}"
    return esc(v)


def kv_chips(d: dict) -> str:
    if not isinstance(d, dict) or not d:
        return ""
    return "".join(
        f'<span class="chip"><b>{esc(k)}</b>{fmt_delta(v)}</span>' for k, v in d.items()
    )


def render_panel(r: dict, idx: int) -> str:
    """一轮的四宫格便当盒面板。"""
    diag = r.get("diagnosis") or {}
    props = (r.get("proposals") or {}).get("proposals") or []
    chosen = r.get("chosen") or {}
    patch = r.get("patch_summary") or {}
    refl = r.get("reflection") or {}
    recoveries = r.get("recoveries") or []

    # ── ① 医生 ──
    findings = diag.get("findings") or []
    if diag.get("no_finding"):
        doctor_body = (
            f'<p class="empty">没查出明显问题</p>'
            f'<p class="note">{esc(diag.get("reason_if_none", ""))}</p>'
        )
    else:
        doctor_body = "".join(
            f'<div class="item">'
            f'<div class="item-head"><span class="tag">{esc(f.get("symptom"))}</span>'
            f'<span class="meta">严重 {esc(f.get("severity"))} · {esc(f.get("confidence"))}</span></div>'
            f'<p class="body-text">{esc(f.get("evidence"))}</p></div>'
            for f in findings
        )

    # ── ② 军师 ──
    chosen_id = chosen.get("card_id")
    strat_body = "".join(
        f'<div class="item{" picked" if p.get("card_id") == chosen_id else ""}">'
        f'<div class="item-head">'
        f'<span class="rank">#{esc(p.get("rank"))}</span>'
        f'<span class="tag">{esc(p.get("card_id") or "自创方案")}</span>'
        f'{"<span class=pick>选中</span>" if p.get("card_id") == chosen_id else ""}'
        f'</div>'
        f'<p class="body-text">{esc(p.get("rationale"))}</p>'
        f'<div class="chips">{kv_chips(p.get("expected") or {})}'
        f'{kv_chips(p.get("cost") or {})}</div>'
        f'</div>'
        for p in props
    ) or '<p class="empty">本轮没有提案</p>'

    # ── ③ 工兵 ──
    if patch:
        files = patch.get("new_files") or []
        checks = patch.get("self_check") or []
        impl_body = (
            f'<p class="note">改动类型：{esc(patch.get("change_type", "—"))}</p>'
            + "".join(f'<div class="file">{esc(f)}</div>' for f in files)
            + '<ul class="checks">'
            + "".join(f"<li>{esc(c)}</li>" for c in checks)
            + "</ul>"
        )
    else:
        impl_body = '<p class="empty">本轮没有产出代码</p>'

    # ── ④ 复盘官 ──
    if refl:
        verdict = refl.get("verdict", "—")
        sym = refl.get("symptom_resolved") or {}
        card_up = refl.get("card_update") or {}
        refl_body = (
            f'<div class="verdict {VERDICT_TONE.get(verdict, "muted")}">{esc(verdict)}</div>'
            f'<div class="chips">{kv_chips(refl.get("actual") or {})}</div>'
            f'<p class="body-text">{esc(refl.get("vs_expected", ""))}</p>'
            + (
                f'<p class="note">目标毛病「{esc(sym.get("symptom"))}」好转：'
                f'<b>{esc(sym.get("resolved"))}</b>'
                f'（{fmt_delta(sym.get("before"))} → {fmt_delta(sym.get("after"))}）</p>'
                if sym else ""
            )
            + (
                f'<p class="note">卡片信任分 {fmt_delta(card_up.get("prior_delta"))}'
                f' · {esc(card_up.get("note", ""))}</p>' if card_up else ""
            )
            + (
                f'<p class="note hint">下一步：{esc(refl.get("next_hint"))}</p>'
                if refl.get("next_hint") else ""
            )
        )
    else:
        refl_body = '<p class="empty">本轮没有复盘（多半中途失败了）</p>'

    # ── 底部状态栏：四阶段进度 + 完整报错（点击展开）──
    stages = [
        ("医生", bool(findings) or diag.get("no_finding")),
        ("军师", bool(props)),
        ("工兵", bool(patch)),
        ("复盘", bool(refl)),
    ]
    # 找出第一个没完成的阶段，它之前的算已完成、它本身算中断点
    first_fail = next((i for i, (_, ok) in enumerate(stages) if not ok), None)
    steps = "".join(
        f'<span class="step {"done" if ok else ("stop" if i == first_fail else "skip")}">'
        f'{esc(name)}</span>'
        for i, (name, ok) in enumerate(stages)
    )

    if recoveries:
        status_tone = "warn"
        status_text = f"{len(recoveries)} 条恢复事件"
    elif refl.get("verdict") == "猜对了":
        status_tone = "ok"
        status_text = "本轮顺利完成"
    else:
        status_tone = "idle"
        status_text = "本轮完成，无异常事件"

    rec_detail = "".join(
        f'<li>{esc(x)}</li>' for x in recoveries
    )
    rec_panel = (
        f'<ul class="rec-list">{rec_detail}</ul>' if recoveries else ""
    )

    mins = r.get("seconds", 0) / 60
    return f"""
<div class="panel{' active' if idx == 0 else ''}" data-panel="{idx}">
  <div class="bento">
    <section class="cell"><h3><i>①</i>医生 · 诊断</h3><div class="scroll">{doctor_body}</div></section>
    <section class="cell"><h3><i>②</i>军师 · 提案</h3><div class="scroll">{strat_body}</div></section>
    <section class="cell"><h3><i>③</i>工兵 · 实现</h3><div class="scroll">{impl_body}</div></section>
    <section class="cell"><h3><i>④</i>复盘官 · 判定</h3><div class="scroll">{refl_body}</div></section>
  </div>
  <div class="statusbar {status_tone}">
    <div class="status-line">
      <div class="steps">{steps}</div>
      <div class="status-msg">
        <span class="dot"></span>{esc(status_text)}
        {'<button class="expand" type="button">展开详情</button>' if recoveries else ''}
      </div>
      <div class="status-meta">
        {esc(r.get('fidelity', '—'))}数据 · {r.get('tokens', 0):,} token ·
        {mins:.1f} 分钟 · 人工干预 {esc(r.get('interventions', 0))}
      </div>
    </div>
    {rec_panel}
  </div>
</div>"""


CSS = """
:root{
  --bg:#F7F6F3; --card:#FFFFFF; --ink:#2E2C28; --soft:#6F6B63; --faint:#9C978D;
  --line:#E8E4DC; --accent:#7C9885; --accent-soft:#EDF2EE;
  --warm:#C7A17A; --warm-soft:#F7F0E7; --bad:#B5766B; --bad-soft:#F7EBE8;
}
*{box-sizing:border-box}
html,body{height:100%; overflow:hidden}
body{
  margin:0; background:var(--bg); color:var(--ink);
  font-family:"Noto Sans SC",-apple-system,"PingFang SC","Microsoft YaHei",sans-serif;
  font-size:14px; line-height:1.7; letter-spacing:.01em;
}
.app{height:100vh; display:flex; flex-direction:column; padding:1.1rem 1.4rem 1.2rem; gap:.9rem}

/* ── 顶栏：标题 + 统计一行 ── */
.top{display:flex; align-items:baseline; justify-content:space-between; gap:1.5rem; flex:none}
.brand h1{font-size:1.15rem; font-weight:700; margin:0; letter-spacing:.02em}
.brand p{margin:0; font-size:.78rem; color:var(--faint)}
.stats{display:flex; gap:1.5rem}
.stat{text-align:right}
.stat b{display:block; font-size:1.05rem; font-weight:600; font-variant-numeric:tabular-nums; line-height:1.3}
.stat span{font-size:.7rem; color:var(--faint); letter-spacing:.04em}

/* ── 主体：左列表 + 右便当盒 ── */
.main{flex:1; display:grid; grid-template-columns:14rem 1fr; gap:.9rem; min-height:0}

.rounds{
  background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:.6rem; overflow-y:auto; display:flex; flex-direction:column; gap:.3rem;
}
.rounds-title{
  font-size:.76rem; font-weight:600; color:var(--soft); letter-spacing:.04em;
  padding:.2rem .7rem .5rem; border-bottom:1px solid var(--line);
  margin-bottom:.4rem; display:flex; flex-direction:column;
}
.rounds-title span{font-size:.68rem; font-weight:400; color:var(--faint); letter-spacing:0}
.round-btn{
  text-align:left; background:none; border:1px solid transparent; border-radius:9px;
  padding:.55rem .7rem; cursor:pointer; font:inherit; color:var(--ink);
  display:flex; flex-direction:column; gap:.15rem; transition:background .15s;
}
.round-btn:hover{background:#FAF9F6}
.round-btn.active{background:var(--accent-soft); border-color:#DCE6DF}
.round-btn .no{font-weight:600; font-size:.9rem}
.round-btn .card-id{font-size:.75rem; color:var(--soft); overflow:hidden;
  text-overflow:ellipsis; white-space:nowrap}
.round-btn .v{font-size:.7rem; color:var(--faint)}
.round-btn.active .v{color:var(--accent)}

.stage{position:relative; min-height:0}
.panel{position:absolute; inset:0; display:none; flex-direction:column; gap:.7rem}
.panel.active{display:flex}

/* ── 底部状态栏：阶段进度 + 状态消息 + 可展开的完整报错 ── */
.statusbar{
  flex:none; background:var(--card); border:1px solid var(--line);
  border-radius:12px; padding:.6rem .9rem;
}
.statusbar.warn{background:var(--warm-soft); border-color:#EADCC8}
.statusbar.ok{background:var(--accent-soft); border-color:#DCE6DF}
.status-line{display:flex; align-items:center; gap:1rem; flex-wrap:wrap}
.steps{display:flex; align-items:center; gap:.3rem}
.step{
  font-size:.74rem; padding:.1rem .55rem; border-radius:99px;
  background:#F0EEE9; color:var(--faint); position:relative;
}
.step + .step::before{
  content:""; position:absolute; left:-.3rem; top:50%; width:.3rem; height:1px;
  background:var(--line);
}
.step.done{background:var(--accent-soft); color:var(--accent)}
.step.stop{background:var(--bad-soft); color:var(--bad); font-weight:500}
.step.skip{opacity:.45}
.status-msg{display:flex; align-items:center; gap:.5rem; font-size:.78rem; color:var(--soft)}
.status-msg .dot{width:.45rem; height:.45rem; border-radius:50%; background:var(--faint)}
.statusbar.warn .dot{background:var(--warm)}
.statusbar.ok .dot{background:var(--accent)}
.expand{
  font:inherit; font-size:.73rem; color:var(--soft); cursor:pointer;
  background:rgba(255,255,255,.7); border:1px solid var(--line);
  border-radius:5px; padding:.05rem .5rem;
}
.expand:hover{color:var(--ink)}
.status-meta{
  margin-left:auto; font-size:.74rem; color:var(--faint);
  font-variant-numeric:tabular-nums;
}
.rec-list{
  margin:.6rem 0 .1rem; padding:0 0 0 1rem; display:none;
  font-size:.78rem; color:#7A5C3C; line-height:1.6;
}
.rec-list.show{display:block}
.rec-list li{margin:.25rem 0}

/* ── 便当盒：2×2 等分，格子内部各自滚动 ── */
.bento{flex:1; display:grid; grid-template-columns:1fr 1fr; grid-template-rows:1fr 1fr;
  gap:.9rem; min-height:0}
.cell{
  background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:.9rem 1rem .6rem; display:flex; flex-direction:column; min-height:0; min-width:0;
}
.cell h3{
  font-size:.82rem; font-weight:600; margin:0 0 .6rem; color:var(--soft);
  display:flex; align-items:center; gap:.45rem; letter-spacing:.03em; flex:none;
}
.cell h3 i{
  font-style:normal; width:1.25rem; height:1.25rem; border-radius:50%;
  background:var(--accent-soft); color:var(--accent);
  display:grid; place-items:center; font-size:.7rem;
}
.scroll{overflow-y:auto; min-height:0; padding-right:.3rem; flex:1}
.scroll::-webkit-scrollbar{width:5px}
.scroll::-webkit-scrollbar-thumb{background:#E0DCD3; border-radius:99px}

.item{padding:.5rem 0; border-top:1px solid var(--line)}
.item:first-child{border-top:none; padding-top:0}
.item-head{display:flex; align-items:center; gap:.4rem; flex-wrap:wrap; margin-bottom:.25rem}
.tag{background:var(--warm-soft); color:#8A6A47; font-size:.75rem;
  padding:.05rem .5rem; border-radius:5px; font-weight:500}
.rank,.meta{color:var(--faint); font-size:.72rem; font-variant-numeric:tabular-nums}
.pick{font-size:.68rem; color:var(--accent); background:var(--accent-soft);
  padding:.02rem .45rem; border-radius:99px}
.item.picked{background:linear-gradient(90deg,var(--accent-soft),transparent 65%);
  margin:0 -.5rem; padding:.5rem; border-radius:8px}
.body-text{margin:0; color:var(--soft); font-size:.82rem}
.chips{display:flex; flex-wrap:wrap; gap:.3rem; margin-top:.4rem}
.chip{font-size:.72rem; background:#F5F3EF; border:1px solid var(--line);
  padding:.05rem .45rem; border-radius:5px; color:var(--soft);
  font-variant-numeric:tabular-nums}
.chip b{font-weight:500; color:var(--faint); margin-right:.25rem}
.file{font-family:"SF Mono",ui-monospace,Menlo,monospace; font-size:.75rem;
  background:#F5F3EF; border:1px solid var(--line); border-radius:5px;
  padding:.2rem .5rem; margin:.3rem 0; word-break:break-all}
.checks{margin:.5rem 0 0; padding-left:1rem; color:var(--soft); font-size:.78rem}
.checks li{margin:.2rem 0}
.checks li::marker{color:var(--accent)}
.verdict{display:inline-block; font-size:.98rem; font-weight:600;
  color:var(--accent); margin-bottom:.3rem}
.verdict.bad{color:var(--bad)}
.verdict.muted{color:var(--soft)}
.note{margin:.35rem 0 0; color:var(--soft); font-size:.78rem}
.note.hint{color:#A8834F}
.empty{color:var(--faint); font-size:.82rem; margin:0}
"""

JS = """
document.querySelectorAll('.round-btn').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const i = btn.dataset.round;
    document.querySelectorAll('.round-btn').forEach(b=>b.classList.toggle('active', b===btn));
    document.querySelectorAll('.panel').forEach(p=>
      p.classList.toggle('active', p.dataset.panel===i));
  });
});
// 状态栏「展开详情」：显示完整的恢复事件文本，不再被截断
document.querySelectorAll('.expand').forEach(btn=>{
  btn.addEventListener('click',()=>{
    const list = btn.closest('.statusbar').querySelector('.rec-list');
    const shown = list.classList.toggle('show');
    btn.textContent = shown ? '收起' : '展开详情';
  });
});
"""


def build(rows: list[dict]) -> str:
    total_tokens = sum(r.get("tokens", 0) for r in rows)
    total_min = sum(r.get("seconds", 0) for r in rows) / 60
    interventions = sum(r.get("interventions", 0) for r in rows)
    verdicts = [(r.get("reflection") or {}).get("verdict") for r in rows]
    hits = sum(1 for v in verdicts if v == "猜对了")

    stats = [
        (f"{len(rows)}", "轮次"),
        (f"{hits}", "猜对了"),
        (f"{total_tokens:,}", f"token · ${estimate_cost(rows):.3f}"),
        (f"{total_min:.0f}", "分钟"),
        (f"{interventions}", "人工干预"),
    ]
    stat_html = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in stats)

    btns = "".join(
        f'<button class="round-btn{" active" if i == 0 else ""}" data-round="{i}">'
        f'<span class="no">第 {esc(r.get("round_id", i + 1))} 轮</span>'
        f'<span class="card-id">{esc((r.get("chosen") or {}).get("card_id") or "—")}</span>'
        f'<span class="v">{esc((r.get("reflection") or {}).get("verdict", "未完成"))}</span>'
        f'</button>'
        for i, r in enumerate(rows)
    ) or '<p class="empty">暂无轮次</p>'

    panels = "".join(render_panel(r, i) for i, r in enumerate(rows))

    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GOAT-LeBron 迭代日志</title>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@400;500;600;700&display=swap">
<style>{CSS}</style></head>
<body><div class="app">
  <header class="top">
    <div class="brand">
      <h1>GOAT-LeBron 迭代日志</h1>
      <p>自主 ML 研究智能体 · AliCCP · 医生 → 军师 → 工兵 → 复盘官</p>
    </div>
    <div class="stats">{stat_html}</div>
  </header>
  <div class="main">
    <nav class="rounds">
      <div class="rounds-title">迭代轮次<span>点击查看该轮</span></div>
      {btns}
    </nav>
    <div class="stage">{panels}</div>
  </div>
</div>
<script>{JS}</script>
</body></html>"""


def main() -> int:
    src = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_IN
    out = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
    if not src.exists():
        print(f"找不到日志：{src}")
        return 1
    rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(rows), encoding="utf-8")
    print(f"已生成 {out}（{len(rows)} 轮）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

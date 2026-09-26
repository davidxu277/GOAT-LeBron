"""把一轮的经过写成人话 —— 终端一行行打，每轮另存一份全文。

为什么要有这个文件：跑起来的时候人什么都看不见。真跑那条路整场不打一行，
离线演习每轮只打一行摘要，四个角色的真实输出只进了 `rounds.jsonl` ——
一行几万字的 JSON，人读不了。于是「工兵第一次写错、拿到报错后改对」这种
最该被看见的经过，跑完就没了。

排版只许有这一份：终端那几行和 markdown 全文共用下面这批摘要函数。

⚠️ 这里的任何异常都不许往外冒。记录是旁观者，绝不反过来弄崩正在跑的训练 ——
跟 `agent/events.py` 一个待遇。`Tracer.step` 整个包在 try 里，
写文件那一步由调用方（`run_session` 的落盘循环）兜着。
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Callable, Iterable

# 终端左边那一栏的宽度。中文按两格算，`医生` 和 `复盘官` 要对齐
_栏宽 = 8
_续行缩进 = " " * (2 + _栏宽)


# ────────────────────────────── 小工具 ──────────────────────────────


def _数(x: Any, 位: int = 4) -> str:
    """数字统一格式。取不到就是 `—`，不要打出 None 让人以为是 0。"""
    try:
        return f"{float(x):.{位}f}"
    except (TypeError, ValueError):
        return "—"


def _带符号(x: Any, 位: int = 4) -> str:
    try:
        return f"{float(x):+.{位}f}"
    except (TypeError, ValueError):
        return "—"


def _时长(秒: Any) -> str:
    try:
        s = float(秒)
    except (TypeError, ValueError):
        return "—"
    if s < 60:
        return f"{s:.1f} 秒"
    return f"{int(s // 60)} 分 {int(s % 60)} 秒"


def _token(n: Any) -> str:
    try:
        v = int(n)
    except (TypeError, ValueError):
        return "—"
    return f"{v / 10000:.1f} 万 token" if v >= 10000 else f"{v} token"


def _首行(文本: Any) -> str:
    行 = str(文本 or "").strip().splitlines()
    return 行[0][:160] if 行 else ""


def _一行(文本: Any, 上限: int = 110) -> str:
    """多行证据压成一行，塞进终端那一栏。"""
    s = " ".join(str(文本 or "").split())
    return s[:上限] + ("…" if len(s) > 上限 else "")


def _分数串(分数: dict[str, Any] | None, 带符号: bool = False) -> str:
    if not 分数:
        return "—"
    f = _带符号 if 带符号 else _数
    return " · ".join(f"{k} {f(v)}" for k, v in 分数.items())


def _宽(文本: str) -> int:
    """中文按两格算，好让终端那一栏真的对齐。"""
    return sum(2 if ord(c) > 0x2E80 else 1 for c in 文本)


# ────────────────────────── 每个步骤的摘要 ──────────────────────────
#
# 每个函数吃这一步的输出，吐 1~2 行人话。终端和 markdown 都从这里取。


def _医生(d: dict[str, Any]) -> list[str]:
    if d.get("no_finding"):
        return [f"没查出明显问题：{_一行(d.get('reason_if_none'))}"]
    findings = list(d.get("findings") or [])
    病 = "、".join(
        f"{f.get('symptom')}（严重 {_数(f.get('severity'), 1)}，把握{f.get('confidence', '')}）"
        for f in findings[:3])
    行 = [f"发现 {len(findings)} 个病：{病}" + ("…" if len(findings) > 3 else "")]
    证据 = _一行(findings[0].get("evidence")) if findings else ""
    if 证据:
        行.append(f"证据：{证据}")
    return 行


def _筛卡(candidates: Iterable[dict[str, Any]]) -> list[str]:
    cands = list(candidates or [])
    if not cands:
        return ["一张对症的卡都没有 —— 军师只能自创"]
    名 = " ".join(f"{c.get('card_id')}({_数(c.get('信任分'), 2)})" for c in cands[:4])
    return [f"对症的卡 {len(cands)} 张：{名}" + ("…" if len(cands) > 4 else "")]


def _军师(p: dict[str, Any]) -> list[str]:
    方案 = list(p.get("proposals") or [])
    if not 方案:
        return ["一个方案都没提"]
    头 = 方案[0]
    行 = [f"提了 {len(方案)} 个方案，第一个「{头.get('card_id') or '自创'}」，"
          f"预计 {_分数串(头.get('expected'), 带符号=True)}"]
    理由 = _一行(头.get("rationale"))
    if 理由:
        行.append(f"理由：{理由}")
    return 行


def _调度(内容: dict[str, Any]) -> list[str]:
    备胎 = 内容.get("备胎") or []
    return [f"选「{内容.get('选了') or '自创'}」，{内容.get('数据') or '—'}数据，"
            f"备胎：{'、'.join(备胎) if 备胎 else '没有'}"]


def _工兵(a: dict[str, Any]) -> list[str]:
    文件 = list(a.get("新文件") or [])
    改了 = f"新写 {'、'.join(文件)}" if 文件 else "只改配置"
    配置行数 = len([x for x in str(a.get("config_patch") or "").splitlines() if x.strip()])
    if 配置行数:
        改了 += f"，配置改 {配置行数} 行"
    return [f"第 {a.get('第几次')} 次（{a.get('候选') or '自创'}）：{改了}"]


def _试跑(内容: dict[str, Any]) -> list[str]:
    状态 = 内容.get("试跑")
    if 状态 == "过了":
        秒 = 内容.get("秒")
        return [f"过了（{_时长(秒)}）" if 秒 is not None else "过了"]
    if 状态 == "挂了":
        return [f"挂了 —— {_首行(内容.get('报错'))}"]
    return []                       # 没试跑 = 执行器没这功能，不必占一行


def _训练(内容: dict[str, Any]) -> list[str]:
    if not 内容.get("ok"):
        return [f"跑挂了 —— {_首行(内容.get('报错'))}"]
    return [f"{_分数串(内容.get('分数'))}（{_时长(内容.get('秒'))}）"]


def _复盘官(r: dict[str, Any]) -> list[str]:
    行 = [f"{r.get('verdict')}：{_分数串(r.get('actual'), 带符号=True)}"]
    好转 = [f"{x.get('symptom')} {x.get('resolved')}" for x in (r.get("symptom_resolved") or [])]
    if 好转:
        行[0] += f"；{'、'.join(好转[:3])}"
    下一步 = _一行(r.get("next_hint"))
    if 下一步:
        行.append(f"下一步：{下一步}")
    return 行


def _本轮(内容: dict[str, Any]) -> list[str]:
    行 = f"{_时长(内容.get('秒'))} · {_token(内容.get('token'))}"
    恢复 = 内容.get("恢复") or 0
    if 恢复:
        行 += f" · 出错并恢复 {恢复} 次"
    return [行]


_摘要 = {"医生": _医生, "筛卡": _筛卡, "军师": _军师, "调度": _调度, "工兵": _工兵,
        "试跑": _试跑, "训练": _训练, "复盘官": _复盘官, "本轮": _本轮}


def 步骤摘要(步骤: str, 内容: Any) -> list[str]:
    """一步的输出压成 1~2 行人话。不认识的步骤原样截断，不抛异常。"""
    fn = _摘要.get(步骤)
    if fn is None or 内容 is None:
        return [str(内容)[:200]] if 内容 is not None else []
    return [x for x in fn(内容) if x]


# ────────────────────────────── 终端 ──────────────────────────────


class Tracer:
    """边跑边打。一步做完打一次 —— 一轮真跑几分钟到几十分钟，攒到最后等于没打。

    printer 可注入（测试传 list.append）。enabled=False 就彻底闭嘴，
    现有测试和 `run_round(tracer=None)` 走这条路。
    """

    def __init__(self, printer: Callable[[str], None] = print, enabled: bool = True):
        self._print, self.enabled = printer, enabled

    def round_header(self, round_id: int, fidelity: str = "") -> None:
        if not self.enabled:
            return
        try:
            self._print(f"第 {round_id} 轮 · {fidelity or '—'}数据")
        except Exception:                        # noqa: BLE001 —— 见模块开头
            pass

    def step(self, 步骤: str, 内容: Any) -> None:
        if not self.enabled:
            return
        try:
            行 = 步骤摘要(步骤, 内容)
            if not 行:
                return
            垫 = " " * max(1, _栏宽 - _宽(步骤))
            self._print(f"  {步骤}{垫}{行[0]}")
            for 续 in 行[1:]:
                self._print(f"{_续行缩进}{续}")
        except Exception:                        # noqa: BLE001 —— 排版有 bug 也不许弄崩这一轮
            pass


# ─────────────────────────── 每轮的全文 ───────────────────────────


def _段(标题: str, 内容: list[str] | None) -> list[str]:
    return [f"## {标题}", "", *(内容 or ["（没走到这一步）"]), ""]


def _围栏(文本: str, 语言: str = "") -> list[str]:
    return [f"```{语言}", *str(文本).rstrip().splitlines(), "```"]


def _医生全文(d: dict[str, Any] | None) -> list[str] | None:
    if not d:
        return None
    if d.get("no_finding"):
        return [f"没查出明显问题：{d.get('reason_if_none') or '（没写原因）'}"]
    out: list[str] = []
    for f in d.get("findings") or []:
        out.append(f"- **{f.get('symptom')}** · 严重 {_数(f.get('severity'), 2)} · "
                   f"把握{f.get('confidence', '')} · 影响 {'、'.join(f.get('affects') or [])}")
        out.append(f"  - 证据：{f.get('evidence')}")
    return out or None


def _筛卡全文(cands: list[dict[str, Any]]) -> list[str] | None:
    if not cands:
        return None
    out = ["| 编号 | 名字 | 治哪些病 | 信任分 |", "|---|---|---|---|"]
    out += [f"| {c.get('card_id')} | {c.get('名字', '')} | "
            f"{'、'.join(c.get('治哪些病') or [])} | {_数(c.get('信任分'), 2)} |"
            for c in cands]
    return out


def _军师全文(p: dict[str, Any] | None) -> list[str] | None:
    if not p:
        return None
    out: list[str] = []
    for 方案 in p.get("proposals") or []:
        out.append(f"### 方案 {方案.get('rank')} · {方案.get('card_id') or '自创'}")
        out.append("")
        out.append(f"- 治：{'、'.join(方案.get('targets') or [])}")
        out.append(f"- 理由：{方案.get('rationale')}")
        out.append(f"- 预计：{_分数串(方案.get('expected'), 带符号=True)}")
        out.append(f"- 代价：{json.dumps(方案.get('cost') or {}, ensure_ascii=False)}")
        out.append(f"- 风险：{方案.get('risk')}")
        if 方案.get("how_to"):
            out.append(f"- 怎么做：{方案['how_to']}")
        out.append("")
    return out or None


def _调度全文(log: Any) -> list[str] | None:
    if not log.chosen:
        return None
    return [f"- 选中：**{log.chosen.get('card_id') or '自创'}**（军师排第 "
            f"{log.chosen.get('rank')}）",
            f"- 用多少数据：{log.fidelity or '—'}",
            f"- 备胎：{'、'.join(log.backups) if log.backups else '没有'}"]


def _工兵全文(log: Any) -> list[str] | None:
    if not log.attempts:
        return None
    out: list[str] = []
    for a in log.attempts:
        out.append(f"### 第 {a.get('第几次')} 次 · {a.get('候选') or '自创'} · 试跑{a.get('试跑')}")
        out.append("")
        out.append(f"- 新文件：{'、'.join(a.get('新文件') or []) or '（没有，只改配置）'}")
        if a.get("config_patch"):
            out.append("- 配置改动：")
            out += _围栏(a["config_patch"], "yaml")
        if a.get("报错"):
            out.append("- 试跑报错：")
            out += _围栏(a["报错"])
        for 路径, 代码 in (a.get("失败代码") or {}).items():
            out.append(f"- 这一版的 `{路径}`（没跑通的那版）：")
            out += _围栏(代码, "python")
        out.append("")
    if log.patch_files:
        out.append(f"最后进了训练的那一版：{'、'.join(log.patch_files)} —— "
                   "完整代码在 `rounds.jsonl` 的 `patch_files` 里，这里不重抄一遍。")
        out.append("")
    return out


def _训练全文(log: Any) -> list[str] | None:
    if not log.metrics:
        return None
    return ["成绩单原文：", *_围栏(json.dumps(log.metrics, ensure_ascii=False, indent=1), "json")]


def _复盘全文(r: dict[str, Any] | None) -> list[str] | None:
    if not r:
        return None
    out = [f"- 结论：**{r.get('verdict')}**",
           f"- 实际变化：{_分数串(r.get('actual'), 带符号=True)}",
           f"- 跟预计比：{r.get('vs_expected')}"]
    for x in r.get("symptom_resolved") or []:
        out.append(f"- {x.get('symptom')}：{_数(x.get('before'), 3)} → "
                   f"{_数(x.get('after'), 3)}（{x.get('resolved')}）")
    if r.get("next_hint"):
        out.append(f"- 下一步：{r['next_hint']}")
    return out


def _本轮全文(log: Any) -> list[str]:
    out = [f"- 整轮耗时：{_时长(log.seconds)}（其中训练 {_时长(log.train_seconds)}）",
           f"- 花掉：{_token(log.tokens)}"]
    if log.steps:
        out.append("")
        out.append("| 角色 | 耗时 | token |")
        out.append("|---|---|---|")
        out += [f"| {s.get('角色')} | {_时长(s.get('秒'))} | {_token(s.get('token'))} |"
                for s in log.steps]
    if log.recoveries:
        out.append("")
        out.append("出错并恢复：")
        out += [f"- {r}" for r in log.recoveries]
    if log.interventions:
        out.append("")
        out.append(f"人工干预 {log.interventions} 次：{'；'.join(log.intervention_notes)}")
    return out


def render_round(log: Any) -> str:
    """一轮的全文 markdown。半路死掉的一轮也要能渲染 —— 缺的段落写明没走到。"""
    ref = log.reflection or {}
    行: list[str] = [
        f"# 第 {log.round_id} 轮 · {log.fidelity or '—'}数据",
        "",
        f"> {log.started_at} · 第 `{log.run_id or '—'}` 场 · "
        f"结论 **{ref.get('verdict') or '本轮作废'}**",
        "",
    ]
    行 += _段("医生", _医生全文(log.diagnosis))
    行 += _段("筛卡", _筛卡全文(log.candidates))
    行 += _段("军师", _军师全文(log.proposals))
    行 += _段("调度", _调度全文(log))
    行 += _段("工兵", _工兵全文(log))
    行 += _段("训练", _训练全文(log))
    行 += _段("复盘官", _复盘全文(ref or None))
    行 += _段("本轮", _本轮全文(log))
    return "\n".join(行)


def write_round_md(logs_dir: pathlib.Path, log: Any) -> pathlib.Path:
    """写到 `<logs_dir>/rounds/<场次>/round_NNN.md` —— 跟 snapshots/ 同一个命名法。"""
    path = (pathlib.Path(logs_dir) / "rounds" / (log.run_id or "unknown")
            / f"round_{log.round_id:03d}.md")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_round(log) + "\n", encoding="utf-8")
    return path

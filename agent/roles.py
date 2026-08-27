"""四个角色。

每个角色 = 一段提示词 + 一个输出结构 + 一层额外校验。
额外校验做的是 JSON Schema 表达不了的事，比如"证据里必须出现数字"。
"""

from __future__ import annotations

import json
import pathlib
import re
from typing import Any

import yaml

from .knowledge import Card, CardLibrary, SymptomVocab
from .llm import LLM, SchemaViolation
from . import schemas

PROMPTS = pathlib.Path(__file__).resolve().parent / "prompts"

# CLAUDE.md R1：这五个字段永远不许进入模型输入
FORBIDDEN_FIELDS = ("sample_id", "common_id", "click", "conversion", "ctcvr")
_HAS_DIGIT = re.compile(r"\d")
_HEDGE_WORDS = ("试试看", "可能有帮助", "值得一试", "一般来说效果不错", "应该有帮助")

# 工兵只准动配置里的这三棵子树。别的键（数据路径、评估口径、预算）
# 一旦被改，跑出来的分数就没法跟前几轮比了 —— 那等于偷偷换了考卷。
ALLOWED_CONFIG_ROOTS = ("features", "model", "train")

# CLAUDE.md R11：提升小于这个数一律记「说不清」。
# 测过噪声带之后（agent/noise.py）用实测值顶掉它，取两者较大的那个。
MIN_REAL_GAIN = 0.0005


def _prompt(name: str, **subs: str) -> str:
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    for key, value in subs.items():
        text = text.replace("{{" + key + "}}", value)
    return text


def _dump(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def _check_config_patch(text: str) -> None:
    """配置补丁必须是能解析的 YAML，且只碰 ALLOWED_CONFIG_ROOTS 那几棵子树。

    执行器会把这段文本合并进当前配置。语法炸了或者键写歪了，
    要么当场崩、要么悄悄改掉评估口径 —— 两种都比打回重写贵得多。
    """
    if not text.strip():
        return
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SchemaViolation(f"config_patch 不是合法 YAML：{exc}")
    if parsed is None:
        return
    if not isinstance(parsed, dict):
        raise SchemaViolation(
            f"config_patch 解析出来是 {type(parsed).__name__}，必须是键值对"
        )
    for key in parsed:
        root = str(key).split(".", 1)[0]
        if root not in ALLOWED_CONFIG_ROOTS:
            raise SchemaViolation(
                f"config_patch 想改「{key}」。只准动 "
                f"{' / '.join(ALLOWED_CONFIG_ROOTS)} 这三棵子树，别的键由人来管。"
            )


# ────────────────────────────── ① 医生 ──────────────────────────────


def diagnose(
    llm: LLM,
    vocab: SymptomVocab,
    health_report: dict[str, Any],
    history_brief: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    def validate(data: dict[str, Any]) -> None:
        for f in data["findings"]:
            if not _HAS_DIGIT.search(f["evidence"]):
                raise SchemaViolation(
                    f"病名「{f['symptom']}」的证据里没有任何数字。"
                    f"证据必须直接引用成绩单里的数值。"
                )
        if data["no_finding"] and data["findings"]:
            raise SchemaViolation("no_finding 为 true 时 findings 必须为空")
        if not data["no_finding"] and not data["findings"]:
            raise SchemaViolation("findings 为空时必须把 no_finding 置为 true 并说明原因")

    user = (
        f"## 本轮成绩单\n\n{_dump(health_report)}\n\n"
        f"## 最近几轮的简要记录\n\n{_dump(history_brief or [])}"
    )
    return llm.call(
        role="医生",
        system=_prompt("doctor", 病名清单=vocab.as_prompt_block()),
        user=user,
        schema=schemas.doctor_schema(vocab),
        big=True,
        validate=validate,
    )


# ────────────────────────────── ② 军师 ──────────────────────────────


def propose(
    llm: LLM,
    vocab: SymptomVocab,
    findings: list[dict[str, Any]],
    candidates: list[Card],
    tried_before: list[dict[str, Any]] | None = None,
    shelved: list[dict[str, Any]] | None = None,
    budget_left: str = "一般",
    pipeline_state: str = "",
) -> dict[str, Any]:
    card_ids = [c.id for c in candidates]

    def validate(data: dict[str, Any]) -> None:
        for p in data["proposals"]:
            for word in _HEDGE_WORDS:
                if word in p["rationale"]:
                    raise SchemaViolation(
                        f"方案 {p['rank']} 的理由里出现了禁用词「{word}」。"
                        f"要给出基于数字的因果推理，不要用模糊的推测语气。"
                    )
            if not _HAS_DIGIT.search(p["rationale"]):
                raise SchemaViolation(
                    f"方案 {p['rank']} 的理由里没有引用任何数字。"
                    f"必须从成绩单的具体数值推出病根。"
                )
            if p["novel"] and not p["how_to"].strip():
                raise SchemaViolation(
                    f"方案 {p['rank']} 标为自创，但没有写实现草图。写不清楚就不要提。"
                )
            if not p["novel"] and p["card_id"] not in card_ids:
                raise SchemaViolation(
                    f"方案 {p['rank']} 不是自创，card_id 却不在候选卡片里"
                )
        if findings:
            top = findings[0]["symptom"]
            if not any(top in p["targets"] for p in data["proposals"]):
                raise SchemaViolation(f"至少要有一个方案针对最严重的毛病「{top}」")

    cards_block = (
        "\n\n".join(c.as_prompt_block() for c in candidates)
        if candidates
        else "（没有对症的卡片。你需要自己想一个方案。）"
    )
    shelf_block = (
        f"## 你以前提过、但还没轮到的方案\n\n{_dump(shelved)}\n\n"
        f"这些是前几轮你自己提的，当时因为性价比排在后面没被执行，现在还对症。\n"
        f"**仍然合适就直接复用**（理由可以写得短，不必把当时的推理再走一遍）；\n"
        f"条件已经变了就别提，也不用解释为什么放弃。\n\n"
        if shelved else ""
    )
    user = (
        f"## 医生诊断\n\n{_dump(findings)}\n\n"
        f"## 对症的药方卡（已按病名筛选过）\n\n{cards_block}\n\n"
        f"## 本轮已经试过的\n\n{_dump(tried_before or [])}\n\n"
        f"{shelf_block}"
        f"## 当前流水线\n\n{pipeline_state or '（初始配置）'}\n\n"
        f"## 剩余预算\n\n{budget_left}"
    )
    return llm.call(
        role="军师",
        system=_prompt("strategist"),
        user=user,
        schema=schemas.strategist_schema(vocab, card_ids),
        big=True,
        validate=validate,
    )


# ────────────────────────────── ③ 工兵 ──────────────────────────────


def implement(
    llm: LLM,
    proposal: dict[str, Any],
    card: Card | None,
    module_interface: str,
    example_module: str,
    current_config: str,
    last_error: str = "",
) -> dict[str, Any]:
    def validate(data: dict[str, Any]) -> None:
        for f in data["new_files"]:
            path = f["path"].replace("\\", "/")
            if not path.startswith("modules/"):
                raise SchemaViolation(
                    f"{path} 不在 modules/ 下。只能在 modules/ 里新建文件（CLAUDE.md R5）。"
                )
            if ".." in path.split("/"):
                raise SchemaViolation(
                    f"{path} 里有 `..`。用它可以从 modules/ 爬出去改主程序，一律打回。"
                )
            for bad in FORBIDDEN_FIELDS:
                if re.search(rf"""["']{bad}["']""", f["content"]):
                    raise SchemaViolation(
                        f"{path} 里出现了禁用字段 {bad}（CLAUDE.md R1）。"
                        f"这五个字段永远不许进入模型输入。"
                    )
        _check_config_patch(data.get("config_patch") or "")
        blob = " ".join(data["self_check"])
        if not any(x in blob for x in FORBIDDEN_FIELDS) and "禁用" not in blob:
            raise SchemaViolation("self_check 里必须有一条确认没有使用禁用字段")
        if "训练集" not in blob:
            raise SchemaViolation("self_check 里必须有一条确认统计量只用了训练集")

    how_to = proposal.get("how_to") or (card.how_to if card else "")
    user = (
        f"## 要实现的方案\n\n{_dump(proposal)}\n\n"
        f"## 实现草图\n\n{how_to or '（无，按方案描述自行设计）'}\n\n"
        f"## 零件接口（必须严格实现）\n\n{module_interface}\n\n"
        f"## 范文：一个现成的零件\n\n```python\n{example_module}\n```\n\n"
        f"## 当前配置\n\n```yaml\n{current_config}\n```"
    )
    if last_error:
        user += f"\n\n## 上次失败的完整报错\n\n```\n{last_error}\n```\n请针对这个报错修正。"

    return llm.call(
        role="工兵",
        system=_prompt("implementer"),
        user=user,
        schema=schemas.implementer_schema(),
        big=False,          # 照着范文写代码，小模型足够
        # 工兵要在 JSON 字符串里塞下一整个代码文件，而带思维链的模型
        # 推理过程也算在输出预算里 —— 给窄了 JSON 会被截断，
        # 表现为「Unterminated string」这种解析失败，白白烧掉整次调用。
        # 留这个上限不是为了省钱，是防失控：模型偶尔会陷入重复生成，
        # 挂机时没有保险丝会静默烧掉大量预算和时间。撞上限会明确报错。
        max_tokens=96000,
        validate=validate,
    )


# ────────────────────────────── ④ 复盘官 ──────────────────────────────


def reflect(
    llm: LLM,
    vocab: SymptomVocab,
    hypothesis: dict[str, Any],
    result: dict[str, Any],
    parent_result: dict[str, Any],
    card: Card | None,
    noise_floor: float = MIN_REAL_GAIN,
) -> dict[str, Any]:
    """复盘一轮。

    noise_floor：同配置换种子的实测抖动（agent/noise.py 测出来的）。
    小于它的"提升"是噪声，不许当成假设成立。没测过就退回 R11 的 0.0005。
    """
    floor = max(MIN_REAL_GAIN, float(noise_floor))

    targets = list(hypothesis.get("targets") or [])

    def validate(data: dict[str, Any]) -> None:
        verdict = data["verdict"]
        gains = data["actual"]
        best = max(abs(v) for v in gains.values()) if gains else 0.0
        # CLAUDE.md R11：提升小于门槛（或小于实测噪声带）一律判「说不清」
        if best < floor and verdict == "猜对了":
            raise SchemaViolation(
                f"最大变化只有 {best:.6f}，低于 {floor:.6f} 的门槛，不能判「猜对了」"
            )

        items = data["symptom_resolved"]
        reported = {item["symptom"] for item in items}
        # 方案声称要治哪几个病，就得逐个交代 —— 少报一个，那个病就没人验证了。
        # 这正是"一个方案打三个病、复盘只报一个"那个洞。
        missing = [t for t in targets if t not in reported]
        if missing:
            raise SchemaViolation(
                f"方案声称要治 {targets}，但 symptom_resolved 里没有交代 {missing}。"
                f"每一个目标毛病都要给出 before / after / resolved。"
            )

        for item in items:
            # 防一种隐蔽的自欺：before / after 一模一样，却自己填 resolved=是。
            # resolved 是模型自己报的，必须拿它自己给的数字对一遍。
            if item["resolved"] in ("是", "部分") and abs(item["after"] - item["before"]) < 1e-9:
                raise SchemaViolation(
                    f"「{item['symptom']}」的 before={item['before']} 与 "
                    f"after={item['after']} 完全没变，却填了 resolved=「{item['resolved']}」。"
                    f"自我申报必须跟数字一致。"
                )

        # 最重要的一条：分数涨了但毛病没治好 → 必须判「说不清」。
        # 多个目标时只要有一个真的好转就算数，全是「否」才拦。
        if verdict == "猜对了" and all(item["resolved"] == "否" for item in items):
            raise SchemaViolation(
                "所有目标毛病都没有改善却判「猜对了」。"
                "分数上涨另有原因时必须判「说不清」。"
            )
        # 两个指标一个都没涨，就不存在"猜对了"这回事
        if verdict == "猜对了" and all(v <= 0 for v in gains.values()):
            raise SchemaViolation(
                f"两个指标都没有上涨（{gains}），不能判「猜对了」。"
            )
        # 判错了还给卡片加分 = 记忆被污染，下次还会挑中这张卡
        if verdict == "猜错了" and data["card_update"]["prior_delta"] > 0:
            raise SchemaViolation("判「猜错了」却调高卡片可信度，方向反了")
        if verdict == "说不清" and abs(data["card_update"]["prior_delta"]) > 0.05:
            raise SchemaViolation("判「说不清」时不应大幅调整卡片可信度")
        # 升到更大数据要烧算力，只有确凿的结论才配
        if data["promote"] and verdict != "猜对了":
            raise SchemaViolation(
                f"verdict 是「{verdict}」却建议升到更大数据。只有「猜对了」才准 promote。"
            )

    user = (
        f"## 当初的假设\n\n{_dump(hypothesis)}\n\n"
        f"## 这次跑出来的结果\n\n{_dump(result)}\n\n"
        f"## 改动之前那一版\n\n{_dump(parent_result)}\n\n"
        f"## 用到的药方卡\n\n{card.as_prompt_block() if card else '（自创方案，无卡片）'}"
    )
    if card and card.failure_signals:
        user += (
            f"\n\n## 这一招失败时通常长什么样\n\n{card.failure_signals}\n\n"
            f"对照上面这些信号看结果：对得上就说明是这招本身没起作用，"
            f"而不是方向选错了 —— 写进 card_update.note 里。"
        )
    return llm.call(
        role="复盘官",
        system=_prompt("reflector"),
        user=user,
        schema=schemas.reflector_schema(vocab),
        big=True,
        validate=validate,
    )

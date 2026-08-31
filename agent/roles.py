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

# CLAUDE.md R1：这六个字段永远不许进入模型输入
FORBIDDEN_FIELDS = (
    "sample_id",
    "common_id",
    "click",
    "conversion",
    "ctcvr",
    "long_view",
)
_HAS_DIGIT = re.compile(r"\d")
_HEDGE_WORDS = ("试试看", "可能有帮助", "值得一试", "一般来说效果不错", "应该有帮助")

# 工兵只准动配置里的这三棵子树。别的键（数据路径、评估口径、预算）
# 一旦被改，跑出来的分数就没法跟前几轮比了 —— 那等于偷偷换了考卷。
ALLOWED_CONFIG_ROOTS = ("features", "model", "train")

# CLAUDE.md R11：提升小于这个数一律记「说不清」。
# 测过噪声带之后（agent/noise.py）用实测值顶掉它，取两者较大的那个。
MIN_REAL_GAIN = 0.0005

# CLAUDE.md 危险信号：训练集与验证集 AUC 差超过这个数 = 严重过拟合或数据切分错了。
# 撞上它时不许让医生用「没有 baseline」「可能是分布差异」把它说成正常 ——
# 红线用代码强制，不靠提示词自觉。
DANGEROUS_TRAIN_VAL_AUC_GAP = 0.15


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

    serialized = yaml.safe_dump(parsed, allow_unicode=True)
    for forbidden in FORBIDDEN_FIELDS:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(forbidden)}(?![A-Za-z0-9_])",
                    serialized):
            raise SchemaViolation(
                f"config_patch 中出现禁用字段 {forbidden}，标签不得进入模型输入"
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
    train_auc = (health_report.get("训练集") or {}).get("点击分")
    val_auc = (health_report.get("验证集") or {}).get("点击分")
    dangerous_gap: float | None = None
    if train_auc is not None and val_auc is not None:
        try:
            dangerous_gap = abs(float(train_auc) - float(val_auc))
        except (TypeError, ValueError):
            dangerous_gap = None

    def validate(data: dict[str, Any]) -> None:
        # 数量上限和数值范围以前写在 schema 里，但结构化输出接口不支持
        # maxItems / minimum / maximum（写了整个请求 400），所以搬到这里。
        # 详见 agent/schemas.py 顶部那段说明。
        if len(data["findings"]) > schemas.MAX_FINDINGS:
            raise SchemaViolation(
                f"报了 {len(data['findings'])} 条毛病，最多 {schemas.MAX_FINDINGS} 条。"
                f"只留最严重的几条。")
        低, 高 = schemas.SEVERITY_RANGE
        for f in data["findings"]:
            if not 低 <= f["severity"] <= 高:
                raise SchemaViolation(
                    f"病名「{f['symptom']}」的 severity={f['severity']} 超出 {低}~{高}。")
            if not _HAS_DIGIT.search(f["evidence"]):
                raise SchemaViolation(
                    f"病名「{f['symptom']}」的证据里没有任何数字。"
                    f"证据必须直接引用成绩单里的数值。"
                )
        if data["no_finding"] and data["findings"]:
            raise SchemaViolation("no_finding 为 true 时 findings 必须为空")
        if not data["no_finding"] and not data["findings"]:
            raise SchemaViolation("findings 为空时必须把 no_finding 置为 true 并说明原因")

        if (dangerous_gap is not None
                and dangerous_gap > DANGEROUS_TRAIN_VAL_AUC_GAP):
            if data["no_finding"]:
                raise SchemaViolation(
                    f"训练与验证点击AUC差值 {dangerous_gap:.4f} 超过 "
                    f"{DANGEROUS_TRAIN_VAL_AUC_GAP:.2f}，触发危险信号，"
                    "不能返回 no_finding")
            danger = [f for f in data["findings"]
                      if f["symptom"] in ("在背题", "数据对不上")]
            if not danger:
                raise SchemaViolation(
                    f"训练与验证点击AUC差值 {dangerous_gap:.4f} 超过 "
                    f"{DANGEROUS_TRAIN_VAL_AUC_GAP:.2f}，必须报告「在背题」或「数据对不上」")
            if max(float(f["severity"]) for f in danger) < 0.7:
                raise SchemaViolation(
                    f"训练与验证点击AUC差值 {dangerous_gap:.4f} 超过 "
                    f"{DANGEROUS_TRAIN_VAL_AUC_GAP:.2f}，危险信号 severity 不得低于 0.7")

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
    # 最近几轮发生了什么。以前只有医生拿得到，军师完全看不见 ——
    # 于是「上一版为什么没跑起来」只能顺着医生的证据文字间接漏一点过来。
    # 开药的人看不到上一副药为什么没煎成，只能靠猜。
    history_brief: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    card_ids = [c.id for c in candidates]

    def validate(data: dict[str, Any]) -> None:
        # 以下三条以前靠 schema 的 maxItems / minimum / maximum，
        # 但结构化输出接口不支持数值约束 —— 也就是说 expected 那个 ±0.05 的
        # 闸门**从来没生效过**。搬到这里才真的拦得住。
        if len(data["proposals"]) > schemas.MAX_PROPOSALS:
            raise SchemaViolation(
                f"提了 {len(data['proposals'])} 个方案，最多 {schemas.MAX_PROPOSALS} 个。")
        for p in data["proposals"]:
            低, 高 = schemas.RANK_RANGE
            if not 低 <= p["rank"] <= 高:
                raise SchemaViolation(f"rank={p['rank']} 超出 {低}~{高}。")
            for m, v in p["expected"].items():
                if abs(v) > schemas.EXPECTED_CAP:
                    raise SchemaViolation(
                        f"方案 {p['rank']} 预计 {m} 变动 {v}，超过 ±{schemas.EXPECTED_CAP}。"
                        f"AUC 上一次改动能挪的量级就在千分位到百分位之间，"
                        f"报大了会把调度器的性价比算歪。")
            if p["cost"]["训练时间倍数"] < schemas.MIN_TIME_MULTIPLIER:
                raise SchemaViolation(
                    f"方案 {p['rank']} 的训练时间倍数 {p['cost']['训练时间倍数']} "
                    f"小于 {schemas.MIN_TIME_MULTIPLIER}。")
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
            if not p["how_to"].strip():
                raise SchemaViolation(
                    f"方案 {p['rank']} 没写 how_to。用现成卡片时也要写 —— "
                    f"写这张卡怎么落到**当前**这条流水线上（动哪几个配置键、"
                    f"用哪些字段、新零件叫什么），不是重述卡片。写不清楚就不要提。"
                )
            # 照抄卡片等于没写：卡片工兵本来就看得到，重复一遍只是烧 token
            抄了 = next((c for c in candidates
                        if c.id == p["card_id"] and c.how_to
                        and p["how_to"].strip()[:40] in c.how_to), None)
            if 抄了 is not None:
                raise SchemaViolation(
                    f"方案 {p['rank']} 的 how_to 是照抄卡片 {抄了.id} 的原文。"
                    f"工兵本来就会看到那张卡 —— 这里要写的是它在当前流水线上怎么落地。"
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
    # 「出了什么错」那一栏是执行器和工兵的原话。军师必须看到它 ——
    # 否则同一堵墙会被撞第二次、第三次，每撞一次就是一整轮白跑。
    历史块 = (
        f"## 最近几轮发生了什么\n\n{_dump(history_brief)}\n\n"
        f"**「出了什么错」那一栏是执行器或工兵的原话，不是套话。**"
        f"里面写明的限制是真实存在的：同一个限制撞第二次，这一轮就白跑了。"
        f"如果上一版是因为某个限制没跑起来，要么绕开它，要么明确说明这次为什么不会再撞上。\n\n"
        if history_brief else ""
    )
    user = (
        f"## 医生诊断\n\n{_dump(findings)}\n\n"
        f"{历史块}"
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
        patch_text = data.get("config_patch") or ""
        _check_config_patch(patch_text)
        parsed_patch = yaml.safe_load(patch_text) or {}

        # 深度训练每轮真正产出的指标名只有这三个（点击分/购买分/loss，
        # 见 harness/deep.py 的 metrics 字典）。工兵曾反复写
        # ctr_auc / cvr_auc / mean_auc 这类英文名，直到烧完一次真训练才 KeyError——
        # 这条检查本来在这里，一次合并冲突手动解决时被整段删掉，没人发现，
        # 直到范文（modules/train/early_stopping.py）还在教错误名字才被揪出来。
        monitor = parsed_patch.get("train.early_stopping.monitor")
        train_patch = parsed_patch.get("train")
        if monitor is None and isinstance(train_patch, dict):
            early_patch = train_patch.get("early_stopping")
            if isinstance(early_patch, dict):
                monitor = early_patch.get("monitor")
        allowed_monitors = {"点击分", "购买分", "loss"}
        if monitor is not None and str(monitor) not in allowed_monitors:
            raise SchemaViolation(
                f"train.early_stopping.monitor 写成了 {monitor!r}，"
                f"但训练循环只产出 {sorted(allowed_monitors)}。"
                "请改用「点击分」、「购买分」或 loss。"
            )

        mlp_patch = ((parsed_patch.get("model") or {}).get("mlp")
                     if isinstance(parsed_patch.get("model"), dict) else None)
        if ("model.mlp.epochs" in parsed_patch
                or isinstance(mlp_patch, dict) and "epochs" in mlp_patch):
            raise SchemaViolation(
                "epochs 写在 model.mlp 下不会生效；"
                "深度训练轮数必须写在 model.deep.epochs。"
            )

        # 接口的 minItems 只认 0 和 1，所以"至少 3 条"只能在这里卡
        if len(data["self_check"]) < schemas.MIN_SELF_CHECKS:
            raise SchemaViolation(
                f"self_check 只写了 {len(data['self_check'])} 条，"
                f"至少要 {schemas.MIN_SELF_CHECKS} 条。")
        blob = " ".join(data["self_check"])
        if not any(x in blob for x in FORBIDDEN_FIELDS) and "禁用" not in blob:
            raise SchemaViolation("self_check 里必须有一条确认没有使用禁用字段")
        if "训练集" not in blob:
            raise SchemaViolation("self_check 里必须有一条确认统计量只用了训练集")

    # 两份草图都给，各有各的用处，别互相顶掉：
    #   卡片的  —— 从论文来的通用做法，讲的是"这招本来该怎么做"
    #   军师的  —— 针对当前配置和当前诊断的适配，讲的是"在我们这儿怎么落"
    # 以前是 `军师的 or 卡片的`，等于二选一，无论丢哪份都少了一半信息。
    通用 = card.how_to if card else ""
    落地 = (proposal.get("how_to") or "").strip()
    草图 = "\n\n".join(part for part in (
        f"### 这招通用的做法（来自药方卡 {card.id}）\n\n{通用}" if 通用 else "",
        f"### 军师要求在当前流水线上这样落地\n\n{落地}" if 落地 else "",
    ) if part) or "（无，按方案描述自行设计）"

    user = (
        f"## 要实现的方案\n\n{_dump(proposal)}\n\n"
        f"## 实现草图\n\n{草图}\n\n"
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
    noise_bands_by_metric: dict[str, float] | None = None,
) -> dict[str, Any]:
    """复盘一轮。

    noise_floor：同配置换种子的实测抖动（agent/noise.py 测出来的）。
    小于它的"提升"是噪声，不许当成假设成立。没测过就退回 R11 的 0.0005。

    noise_bands_by_metric：每个指标各自的门槛。必须分指标 —— 点击分和购买分的
    抖动差一个数量级（实测验证集里点击正样本 8,950 个、转化正样本只有 38 个）。
    用一个标量管两个，真实的点击提升会被购买分的抖动淹掉判成「说不清」，
    购买分自己抖一下又能越过门槛白拿 +0.15 信任分。
    """
    floor = max(MIN_REAL_GAIN, float(noise_floor))

    def 够不够(gains: dict[str, float]) -> bool:
        """有没有哪一项越过了**它自己**的噪声带。"""
        if not gains:
            return False
        if not noise_bands_by_metric:
            return max(abs(v) for v in gains.values()) >= floor
        return any(
            abs(v) >= max(MIN_REAL_GAIN,
                          float(noise_bands_by_metric.get(k, 0.0) or 0.0))
            for k, v in gains.items()
        )

    targets = list(hypothesis.get("targets") or [])

    def validate(data: dict[str, Any]) -> None:
        # 数量上限与限幅：接口不支持 maxItems / minimum / maximum，只能在这里卡
        if len(data["symptom_resolved"]) > schemas.MAX_RESOLVED:
            raise SchemaViolation(
                f"交代了 {len(data['symptom_resolved'])} 个病，"
                f"最多 {schemas.MAX_RESOLVED} 个。")
        if abs(data["card_update"]["prior_delta"]) > schemas.PRIOR_DELTA_CAP:
            raise SchemaViolation(
                f"prior_delta={data['card_update']['prior_delta']}，"
                f"超过 ±{schemas.PRIOR_DELTA_CAP}。一次实验不足以把一张卡捧上天或打死。")
        verdict = data["verdict"]
        gains = data["actual"]
        best = max(abs(v) for v in gains.values()) if gains else 0.0
        # CLAUDE.md R11：提升小于门槛（或小于该指标自己的实测噪声带）判「说不清」
        if verdict == "猜对了" and not 够不够(gains):
            带 = noise_bands_by_metric or {"（统一门槛）": floor}
            raise SchemaViolation(
                f"没有任何一项越过它自己的噪声带（最大变化 {best:.6f}，"
                f"各指标门槛 {带}），不能判「猜对了」"
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

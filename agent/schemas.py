"""四个角色的输出结构。

用 Claude 的结构化输出（output_config.format）约束，模型只能按这个格式吐 JSON。

关键：医生的 symptom 字段是一个 **enum**，取值直接从 symptoms.yaml 生成。
也就是说模型在物理上说不出词表以外的病名 —— 这是"对暗号"最硬的实现方式。

⚠️ 这个接口只吃 JSON Schema 的一个**子集**，超出的关键字不是被忽略，是整个请求
   **400 打回**（连模型都没调到，那一轮直接作废）。踩过的坑：

     maxItems              完全不支持
     minItems              只允许 0 或 1
     minimum / maximum     数值约束完全不支持
     multipleOf            不支持
     minLength / maxLength 不支持
     oneOf                 不支持（anyOf / allOf 可以）
     additionalProperties  必须是 false

   所以「数量上限」和「数值范围」这两类约束**没法写在 schema 里**，全部下沉到
   roles.py 的 validate（那本来就是这个项目更硬的那道墙）。下面这些常量是
   两边共用的唯一来源，改这里就够了。

   `assert_api_compatible()` 会走一遍 schema 把越界的关键字揪出来 ——
   有测试盯着，免得以后有人顺手加一个 maxItems，等到真跑那天才发现。
"""

from __future__ import annotations

from typing import Any

from .knowledge import SymptomVocab

CONFIDENCE = ["高", "中", "低"]
VERDICT = ["猜对了", "猜错了", "说不清", "没跑起来"]
RESOLVED = ["是", "部分", "否"]
METRICS = ["点击AUC", "购买AUC"]

# ── 以前写在 schema 里、现在由 validate 强制的约束 ──────────────────
# 军师报的单项预计提升上限（绝对值）。AUC 上一次改动能挪的量级就在千分位到
# 百分位之间；报出 +0.3 这种数字会把调度器的性价比公式整个带偏。
EXPECTED_CAP = 0.05
MIN_TIME_MULTIPLIER = 0.1      # 训练时间倍数的下限
SEVERITY_RANGE = (0.0, 1.0)    # 医生给的严重度
RANK_RANGE = (1, 3)            # 军师方案的排名
PRIOR_DELTA_CAP = 0.2          # 复盘官一次能挪动的卡片可信度
MAX_FINDINGS = 3               # 医生最多报几条
MAX_PROPOSALS = 3              # 军师最多提几个方案
MAX_RESOLVED = 3               # 复盘官最多交代几个病
MIN_SELF_CHECKS = 3            # 工兵至少写几条自检

# 这个接口不认的关键字。写进 schema 会让整个请求 400，不是被忽略。
UNSUPPORTED_KEYWORDS = (
    "maxItems", "minLength", "maxLength", "minimum", "maximum",
    "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "oneOf",
    "patternProperties", "uniqueItems",
)


def assert_api_compatible(schema: Any, path: str = "$") -> None:
    """走一遍 schema，确认没用上接口不认的关键字。不合规就抛 ValueError。

    存在的理由：这类错误在本地是零成本的，在线上是一次 400 —— 那一轮
    连模型都没调到就作废了，而且报错信息（"For 'array' type, property
    'maxItems' is not supported"）只会点名**第一个**违规项，得一个一个撞。
    """
    if isinstance(schema, dict):
        for bad in UNSUPPORTED_KEYWORDS:
            if bad in schema:
                raise ValueError(
                    f"{path} 用了 `{bad}`，结构化输出接口不支持，整个请求会 400。"
                    f"这类约束请写进 roles.py 的 validate。")
        if schema.get("minItems", 0) not in (0, 1):
            raise ValueError(
                f"{path} 的 minItems={schema['minItems']}，接口只允许 0 或 1。")
        if schema.get("type") == "object" and schema.get("additionalProperties") is not False:
            raise ValueError(f"{path} 的 additionalProperties 必须是 False。")
        for key, value in schema.items():
            assert_api_compatible(value, f"{path}.{key}")
    elif isinstance(schema, list):
        for i, value in enumerate(schema):
            assert_api_compatible(value, f"{path}[{i}]")


def _obj(props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": props,
        "required": required,
        "additionalProperties": False,
    }


def doctor_schema(vocab: SymptomVocab) -> dict[str, Any]:
    finding = _obj(
        {
            "symptom": {"type": "string", "enum": vocab.ids},
            # 范围 0~1 由 roles.diagnose 的 validate 卡（schema 不支持数值约束）
            "severity": {"type": "number", "description": "0~1，越大越严重"},
            "confidence": {"type": "string", "enum": CONFIDENCE},
            "evidence": {
                "type": "string",
                "description": "必须引用成绩单里的具体数字，禁止『明显偏低』这类描述",
            },
            "affects": {
                "type": "array",
                "items": {"type": "string", "enum": METRICS},
            },
        },
        ["symptom", "severity", "confidence", "evidence", "affects"],
    )
    return _obj(
        {
            # 上限 MAX_FINDINGS 由 validate 卡（schema 不支持 maxItems）
            "findings": {"type": "array", "items": finding},
            "no_finding": {
                "type": "boolean",
                "description": "确实没查出明显问题时置 true。不许硬凑病。",
            },
            "reason_if_none": {"type": "string"},
        },
        ["findings", "no_finding", "reason_if_none"],
    )


def strategist_schema(vocab: SymptomVocab, card_ids: list[str]) -> dict[str, Any]:
    # card_id 允许为空字符串，表示这是自创方案（库里没有的招）
    proposal = _obj(
        {
            "rank": {"type": "integer", "description": "1~3，1 最推荐"},
            "card_id": {"type": "string", "enum": [*card_ids, ""]},
            "targets": {
                "type": "array",
                "items": {"type": "string", "enum": vocab.ids},
                "minItems": 1,
            },
            "rationale": {
                "type": "string",
                "description": "必须走完两段因果：证据→病根，病根→这招为什么对症",
            },
            # 预计提升限幅 ±EXPECTED_CAP：报出 +0.3 这种数字会把调度器的
            # 性价比公式整个带偏。⚠️ 这里以前写的是 minimum/maximum，
            # 而接口根本不支持数值约束 —— 也就是说这道闸门从来没生效过。
            # 现在由 roles.propose 的 validate 卡。
            "expected": _obj(
                {m: {"type": "number", "description": f"预计提升，绝对值不超过 {EXPECTED_CAP}"}
                 for m in METRICS},
                METRICS,
            ),
            "cost": _obj(
                {
                    "代码难度": {"type": "string", "enum": ["改配置", "简单", "中等", "难"]},
                    "训练时间倍数": {"type": "number",
                                "description": f"相对当前配置的倍数，不小于 {MIN_TIME_MULTIPLIER}"},
                },
                ["代码难度", "训练时间倍数"],
            ),
            "risk": {"type": "string"},
            "novel": {
                "type": "boolean",
                "description": "true = 卡片库里没有，是自己想的招。此时 card_id 留空字符串。",
            },
            # 一律要写，但两种情况写的东西不一样：
            #   自创方案 —— 从零写实现草图
            #   用现成卡片 —— **不要重述卡片**，写这张卡怎么落到当前这条流水线上：
            #                 动哪几个配置键、用哪些字段、新零件叫什么名字
            # 卡片上的「怎么实现」是从论文来的通用知识（人整理的），
            # 这里要的是 Agent 自己针对当前配置和当前诊断做的适配 ——
            # 那正是评分标准里「识别出什么值得尝试的、以及为什么」要看的东西。
            "how_to": {
                "type": "string",
                "description": ("怎么落地。自创方案写完整草图；用现成卡片时写"
                                "「这张卡怎么套到当前流水线上」，别重述卡片内容"),
            },
        },
        ["rank", "card_id", "targets", "rationale", "expected", "cost", "risk", "novel", "how_to"],
    )
    return _obj(
        # 上限 MAX_PROPOSALS 由 validate 卡（minItems 1 是接口允许的少数几个之一）
        {"proposals": {"type": "array", "items": proposal, "minItems": 1}},
        ["proposals"],
    )


def implementer_schema() -> dict[str, Any]:
    new_file = _obj(
        {
            "path": {
                "type": "string",
                "description": "只能在 modules/ 下新建。不许改主程序。",
            },
            "content": {
                "type": "string",
                "description": "完整文件内容，不是 diff 片段",
            },
        },
        ["path", "content"],
    )
    return _obj(
        {
            "change_type": {
                "type": "string",
                "enum": ["只改配置", "加新零件", "两者都有"],
            },
            "config_patch": {
                "type": "string",
                "description": "YAML 文本，会被合并进当前配置",
            },
            "new_files": {"type": "array", "items": new_file},
            # 至少 MIN_SELF_CHECKS 条由 validate 卡 —— 接口的 minItems 只认 0 和 1
            "self_check": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": (f"至少 {MIN_SELF_CHECKS} 条，必须包含对 CLAUDE.md R1"
                                f"（禁用字段）和 R2（统计量只用训练集）的确认"),
            },
        },
        ["change_type", "config_patch", "new_files", "self_check"],
    )


def reflector_schema(vocab: SymptomVocab) -> dict[str, Any]:
    return _obj(
        {
            "verdict": {"type": "string", "enum": VERDICT},
            "actual": _obj({m: {"type": "number"} for m in METRICS}, METRICS),
            "vs_expected": {"type": "string"},
            # 数组，不是单个 —— 一个方案可以同时打好几个病（26 张卡里 11 张是多病卡）。
            # 方案声称要治的每一个病，都必须在这里给出 before/after，
            # 少报一个就等于那个病没人验证。校验见 roles.reflect。
            # 上限 MAX_RESOLVED 由 validate 卡（schema 不支持 maxItems）
            "symptom_resolved": {
                "type": "array",
                "minItems": 1,
                "items": _obj(
                    {
                        "symptom": {"type": "string", "enum": vocab.ids},
                        "before": {"type": "number"},
                        "after": {"type": "number"},
                        "resolved": {"type": "string", "enum": RESOLVED},
                    },
                    ["symptom", "before", "after", "resolved"],
                ),
            },
            "card_update": _obj(
                {
                    "card_id": {"type": "string"},
                    # 限幅 ±PRIOR_DELTA_CAP 由 validate 卡（同上，接口不支持数值约束）
                    "prior_delta": {"type": "number",
                                    "description": f"绝对值不超过 {PRIOR_DELTA_CAP}"},
                    "note": {"type": "string"},
                },
                ["card_id", "prior_delta", "note"],
            ),
            "next_hint": {"type": "string"},
            "promote": {"type": "boolean"},
        },
        ["verdict", "actual", "vs_expected", "symptom_resolved", "card_update", "next_hint", "promote"],
    )

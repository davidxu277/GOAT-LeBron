"""四个角色的输出结构。

用 Claude 的结构化输出（output_config.format）约束，模型只能按这个格式吐 JSON。

关键：医生的 symptom 字段是一个 **enum**，取值直接从 symptoms.yaml 生成。
也就是说模型在物理上说不出词表以外的病名 —— 这是"对暗号"最硬的实现方式。
"""

from __future__ import annotations

from typing import Any

from .knowledge import SymptomVocab

CONFIDENCE = ["高", "中", "低"]
VERDICT = ["猜对了", "猜错了", "说不清", "没跑起来"]
RESOLVED = ["是", "部分", "否"]
METRICS = ["点击AUC", "购买AUC"]

# 军师报的单项预计提升上限（绝对值）。见 strategist_schema 里的说明。
EXPECTED_CAP = 0.05


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
            "severity": {"type": "number", "minimum": 0, "maximum": 1},
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
            "findings": {"type": "array", "items": finding, "maxItems": 3},
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
            "rank": {"type": "integer", "minimum": 1, "maximum": 3},
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
            # 预计提升限幅：AUC 上一次改动能挪的量级就在千分位到百分位之间。
            # 报出 +0.3 这种数字的方案会把调度器的性价比公式整个带偏，
            # 所以在 schema 层就封死，不靠提示词自觉。
            "expected": _obj(
                {m: {"type": "number", "minimum": -EXPECTED_CAP, "maximum": EXPECTED_CAP}
                 for m in METRICS},
                METRICS,
            ),
            "cost": _obj(
                {
                    "代码难度": {"type": "string", "enum": ["改配置", "简单", "中等", "难"]},
                    "训练时间倍数": {"type": "number", "minimum": 0.1},
                },
                ["代码难度", "训练时间倍数"],
            ),
            "risk": {"type": "string"},
            "novel": {
                "type": "boolean",
                "description": "true = 卡片库里没有，是自己想的招。此时 card_id 留空字符串。",
            },
            "how_to": {
                "type": "string",
                "description": "自创方案必须自己写实现草图；用现成卡片时可留空",
            },
        },
        ["rank", "card_id", "targets", "rationale", "expected", "cost", "risk", "novel", "how_to"],
    )
    return _obj(
        {"proposals": {"type": "array", "items": proposal, "minItems": 1, "maxItems": 3}},
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
            "self_check": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 3,
                "description": "必须包含对 CLAUDE.md R1（禁用字段）和 R2（统计量只用训练集）的确认",
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
            "symptom_resolved": _obj(
                {
                    "symptom": {"type": "string", "enum": vocab.ids},
                    "before": {"type": "number"},
                    "after": {"type": "number"},
                    "resolved": {"type": "string", "enum": RESOLVED},
                },
                ["symptom", "before", "after", "resolved"],
            ),
            "card_update": _obj(
                {
                    "card_id": {"type": "string"},
                    "prior_delta": {"type": "number", "minimum": -0.2, "maximum": 0.2},
                    "note": {"type": "string"},
                },
                ["card_id", "prior_delta", "note"],
            ),
            "next_hint": {"type": "string"},
            "promote": {"type": "boolean"},
        },
        ["verdict", "actual", "vs_expected", "symptom_resolved", "card_update", "next_hint", "promote"],
    )

"""知识层：病名词表 + 药方卡。

这一层完全不调用 LLM。
"筛卡片"就发生在这里 —— 医生喊出病名，这里对暗号找出对症的卡。
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml

ROOT = pathlib.Path(__file__).resolve().parent.parent
SYMPTOMS_PATH = ROOT / "knowledge" / "symptoms.yaml"
CARDS_DIR = ROOT / "knowledge" / "cards"


# ────────────────────────────── 病名词表 ──────────────────────────────


@dataclass(frozen=True)
class Symptom:
    id: str
    detect: str
    needs: str
    core: bool


class SymptomVocab:
    """医生和药方卡之间唯一的共同语言。

    医生只能说出这里面的词，卡片只能贴这里面的标签。
    两边都由代码强制校验，不靠自觉（见 CLAUDE.md）。
    """

    def __init__(self, symptoms: list[Symptom]):
        self._by_id = {s.id: s for s in symptoms}

    @classmethod
    def load(cls, path: pathlib.Path = SYMPTOMS_PATH) -> "SymptomVocab":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls([
            Symptom(
                id=s["id"],
                detect=s.get("detect", "").strip(),
                needs=s.get("needs", "").strip(),
                core=bool(s.get("core", False)),
            )
            for s in raw["symptoms"]
        ])

    @property
    def ids(self) -> list[str]:
        return list(self._by_id)

    def __contains__(self, symptom_id: str) -> bool:
        return symptom_id in self._by_id

    def __getitem__(self, symptom_id: str) -> Symptom:
        return self._by_id[symptom_id]

    def validate(self, symptom_ids: list[str]) -> list[str]:
        """返回不合法的病名。空列表 = 全部合法。"""
        return [s for s in symptom_ids if s not in self._by_id]

    def as_prompt_block(self) -> str:
        """喂给医生的病名清单。只给 id 和判定规则，不给多余的东西。"""
        lines = []
        for s in self._by_id.values():
            lines.append(f"- {s.id}\n  判定：{s.detect}")
        return "\n".join(lines)


# ────────────────────────────── 药方卡 ──────────────────────────────


@dataclass
class Card:
    id: str
    name: str
    treats: list[str]
    stage: str
    mechanism: str            # 为什么管用 —— 军师讲道理的原料，最重要的一栏
    expected: dict[str, float]
    cost: dict[str, Any]
    preconditions: list[str] = field(default_factory=list)
    prior: float = 0.5        # 靠不靠谱，会被复盘官更新
    how_to: str = ""          # 实现草图（不是代码，代码由工兵自己写）
    source: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Card":
        return cls(
            id=d["编号"],
            name=d.get("名字", d["编号"]),
            treats=list(d.get("治哪些毛病", [])),
            stage=d.get("属于哪个环节", ""),
            mechanism=(d.get("为什么管用") or "").strip(),
            expected=dict(d.get("预计能提多少", {})),
            cost=dict(d.get("要花多少力气", {})),
            preconditions=list(d.get("前提条件", [])),
            prior=float(d.get("靠不靠谱", 0.5)),
            how_to=(d.get("怎么实现") or "").strip(),
            source=d.get("出处", ""),
        )

    def as_prompt_block(self) -> str:
        """喂给军师的卡片摘要。刻意不含 how_to —— 那是工兵才需要的。"""
        return (
            f"【{self.id}】{self.name}\n"
            f"  治：{', '.join(self.treats)}\n"
            f"  环节：{self.stage}\n"
            f"  为什么管用：{self.mechanism}\n"
            f"  预计：{self.expected}   力气：{self.cost}\n"
            f"  前提：{self.preconditions or '无'}   历史靠谱度：{self.prior:.2f}"
        )


class CardLibrary:
    def __init__(self, cards: list[Card]):
        self.cards = cards
        self._by_id = {c.id: c for c in cards}

    @classmethod
    def load(cls, vocab: SymptomVocab, cards_dir: pathlib.Path = CARDS_DIR) -> "CardLibrary":
        """读取全部卡片，并强制校验标签合法性。

        任何一张卡贴了词表外的标签 → 直接报错，不许运行。
        贴错了当场就炸，五分钟修好；而不是跑到第四天才发现实验全白做。
        """
        cards: list[Card] = []
        for path in sorted(cards_dir.glob("*.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
            if not raw:
                continue
            card = Card.from_dict(raw)
            bad = vocab.validate(card.treats)
            if bad:
                raise ValueError(
                    f"卡片 {path.name} 贴了词表里没有的病名：{bad}\n"
                    f"合法的病名见 knowledge/symptoms.yaml"
                )
            if not card.mechanism:
                raise ValueError(f"卡片 {path.name} 缺少「为什么管用」—— 这一栏不能省")
            cards.append(card)
        return cls(cards)

    def __len__(self) -> int:
        return len(self.cards)

    def get(self, card_id: str) -> Card | None:
        return self._by_id.get(card_id)

    def match(
        self,
        symptom_ids: list[str],
        exclude_ids: set[str] | None = None,
        limit: int = 5,
    ) -> list[Card]:
        """按病名对暗号，找出对症的卡。

        这是整条链路上最省钱的一步：一次集合求交，不调用任何模型。
        排序按 (命中几个病, 历史靠谱度) 降序。
        """
        exclude_ids = exclude_ids or set()
        wanted = set(symptom_ids)
        hits = []
        for card in self.cards:
            if card.id in exclude_ids:
                continue
            overlap = len(wanted & set(card.treats))
            if overlap:
                hits.append((overlap, card.prior, card))
        hits.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [c for _, _, c in hits[:limit]]

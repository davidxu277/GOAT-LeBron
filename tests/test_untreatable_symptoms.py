"""有的病治不了 —— 自检不该天天为它报警，把真正缺卡的病淹掉。

「退化用户占比高」说的是评测口径：GAUC 只统计 0 < 正例数 < 曝光数 的用户，
全正或全负的用户 GAUC 不算、nDCG 记 0。这是数据本身的构成，换什么模型都改不了。
它存在的意义是给医生当背景（解释为什么两个指标朝相反方向动），不是等着被治。

以前自检每次都报「还没有卡片对症的病：退化用户占比高」，看久了人就不看了 ——
哪天真有一个新病没卡可用，混在这条假警报里一起被忽略。
"""

from agent.cli import uncovered_symptoms
from agent.knowledge import CardLibrary, Symptom, SymptomVocab


def test_退化用户占比高标了治不了_还写了为什么():
    s = SymptomVocab.load()["退化用户占比高"]
    assert s.treatable is False
    assert "口径" in s.untreatable_because


def test_其他病默认能治():
    vocab = SymptomVocab.load()
    assert all(vocab[sid].treatable for sid in vocab.ids if sid != "退化用户占比高")


def test_自检不为治不了的病报警():
    vocab = SymptomVocab.load()
    assert "退化用户占比高" not in uncovered_symptoms(vocab, CardLibrary.load(vocab))


def test_真没卡可用的病照样报():
    vocab = SymptomVocab.load()
    cards = CardLibrary.load(vocab)
    多一个 = SymptomVocab(vocab.all() + [Symptom(id="新病", detect="x", needs="y", core=False)])
    assert uncovered_symptoms(多一个, cards) == ["新病"]

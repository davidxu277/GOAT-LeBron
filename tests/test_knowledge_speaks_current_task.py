"""知识库必须说「当前任务」的话。

08-31 任务从 AliCCP 换成 KuaiRand-Pure，病名词表迁了，卡片和提示词只迁了一半：
14 张卡里 11 张的「预计能提多少」还写着 点击AUC / 购买AUC，军师提示词的正面例子
还是「低频桶购买AUC 0.611」，好几张卡还在引用 mlp_model.py 和 109_14 这类字段。

后果是实打实的（2026-09-01 真跑）：军师在方案理由里引用了「mlp_model.py:69」
和「109_14」当证据 —— 两个都来自卡片原文，**一个不在本仓库，一个不在本数据集**。
LLM 没有编造，它忠实地转述了我们喂给它的过期事实。

根子跟之前修过的几处一样：任务身份是**手写**进每张卡、每份提示词的，换任务时
得有人记得挨个改。这里不再靠记得 —— 每一类事实都跟一个派生出来的真值对：

    指标名    ← schemas.METRICS（跟成绩单、医生、复盘官同源）
    源码文件  ← 仓库里真的有没有这个文件
    数据字段  ← 退役任务独有的字段不许再出现

指标名那一条在 CardLibrary.load 里强制（跟"贴了词表外的病名"同一个级别，
启动就炸）；其余两条是自由文本，放在测试里盯。
"""

import pathlib
import re

import pytest
import yaml

from agent import schemas
from agent.knowledge import CARDS_DIR, CardLibrary, SymptomVocab

ROOT = pathlib.Path(__file__).resolve().parent.parent
PROMPTS_DIR = ROOT / "agent" / "prompts"
CARD_FORMAT_DOC = ROOT / "knowledge" / "卡片格式.md"

# 退役任务独有的字段：AliCCP 的标签和连接键。KuaiRand 的原始表里一个都没有。
RETIRED_FIELDS = ("click", "conversion", "ctcvr", "common_id", "sample_id")
# AliCCP 的特征是数字编号（206 类目、109_14 用户历史类目……）。
# KuaiRand 没有任何数字命名的列，所以"点名一个数字字段"本身就是过期事实。
_ALICCP_FIELD_CODE = re.compile(r"(?<![\w.])\d{3}_\d{2}(?![\w.])"
                                r"|(?:字段|_field)\s*(?:用|=|:)?\s*\d{3}(?![\w.])")
_PY_REF = re.compile(r"[\w./-]+\.py\b")
_SEARCH_DIRS = ("agent", "harness", "modules", "kuairand_goat_bridge", "config", "tests")


@pytest.fixture(scope="module")
def vocab():
    return SymptomVocab.load()


def _card_yaml(预计):
    return yaml.safe_dump({
        "编号": "测试卡", "名字": "测试卡", "治哪些毛病": ["在背题"],
        "为什么管用": "为了测试。", "预计能提多少": 预计,
        "要花多少力气": {"代码难度": "简单", "训练时间倍数": 1.0},
    }, allow_unicode=True)


# ───────────────────── 指标名：启动就炸 ─────────────────────


def test_卡片预计写了本任务没有的指标名_启动就报错(tmp_path, vocab):
    (tmp_path / "x.yaml").write_text(_card_yaml({"点击AUC": 0.003}), encoding="utf-8")
    with pytest.raises(ValueError, match="点击AUC"):
        CardLibrary.load(vocab, cards_dir=tmp_path)


def test_卡片预计用本任务的指标名照常加载(tmp_path, vocab):
    (tmp_path / "x.yaml").write_text(
        _card_yaml({m: 0.001 for m in schemas.METRICS}), encoding="utf-8")
    assert len(CardLibrary.load(vocab, cards_dir=tmp_path)) == 1


def test_出厂卡片全部能加载(vocab):
    """真卡片库过得了上面那道闸 —— 否则 agent 根本启动不了。"""
    assert len(CardLibrary.load(vocab)) > 0


# ───────────────────── 自由文本：测试里盯 ─────────────────────


def _llm_visible_texts():
    """LLM 真正读得到的文字：卡片的字段值（注释不进提示词）、全部提示词、
    病名的判据。外加给写卡的人看的格式说明 —— 样例写错，新卡照着抄就错。"""
    for path in sorted(CARDS_DIR.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        yield f"cards/{path.name}", yaml.safe_dump(raw, allow_unicode=True)
    for path in sorted(PROMPTS_DIR.glob("*.md")):
        yield f"prompts/{path.name}", path.read_text(encoding="utf-8")
    for s in SymptomVocab.load().all():
        yield f"symptoms/{s.id}", f"{s.detect}\n{s.needs}"
    yield CARD_FORMAT_DOC.name, CARD_FORMAT_DOC.read_text(encoding="utf-8")


# AliCCP 的指标名。以前从 METRIC_PAIRS 里旧任务那一套派生；那一套随旧流水线
# 拆了（留着它就是一条旧名字漏进 agent 的路），这里改成显式列出。
RETIRED_METRICS = ("点击分", "购买分", "点击AUC", "购买AUC")


def test_退役任务的指标名不再出现():
    retired = [m for m in RETIRED_METRICS if m not in schemas.METRICS]
    assert retired == list(RETIRED_METRICS), "当前任务的指标名跟退役的撞了"
    hits = [
        (where, name)
        for where, text in _llm_visible_texts()
        for name in retired
        if name in re.sub(r"\s+", "", text)      # 「点击 AUC」带空格也算
    ]
    assert not hits, f"还在说旧任务的指标名：{hits}"


def _exists_in_repo(ref: str) -> bool:
    if not ref.isascii():       # 「modules/features/你写的那个文件.py」这类是占位符
        return True
    if (ROOT / ref).exists():
        return True
    name = pathlib.PurePath(ref).name
    return any(any((ROOT / d).rglob(name)) for d in _SEARCH_DIRS)


def test_点名的源码文件都真实存在():
    """军师照着卡片引用「mlp_model.py:69」当证据 —— 那是别的仓库的文件。"""
    missing = [
        (where, ref)
        for where, text in _llm_visible_texts()
        for ref in set(_PY_REF.findall(text))
        if not _exists_in_repo(ref)
    ]
    assert not missing, f"引用了仓库里不存在的文件：{missing}"


_YAML_BLOCK = re.compile(r"```yaml\n(.*?)```", re.S)


def test_提示词里的配置样例真的能把现成零件装上():
    """工兵照抄样例是常态。样例的键名跟零件读的键名对不上，照抄就是 KeyError。

    实例：implementer.md 教工兵复用 SWA 时写的是 `start_epoch: 8`，
    而 modules/train/swa.py 读的是 `start_epoch_ratio` —— 照抄必崩。
    这里不比对键名清单，直接拿样例去**真的加载**那个零件。
    """
    from harness.deep import load_train_ops
    from harness.ops import load_feature_ops

    loaders = {"train": load_train_ops, "features": load_feature_ops}
    sources = [(p.name, p.read_text(encoding="utf-8"))
               for p in [*sorted(PROMPTS_DIR.glob("*.md")), CARD_FORMAT_DOC]]
    sources += [(c.id, c.how_to) for c in CardLibrary.load(SymptomVocab.load()).cards]
    checked, broken = 0, []
    for where, text in sources:
        for block in _YAML_BLOCK.findall(text):
            try:
                cfg = yaml.safe_load(block)
            except yaml.YAMLError:
                continue
            if not isinstance(cfg, dict):
                continue
            for section, load in loaders.items():
                for name, part in (cfg.get(section) or {}).items():
                    impl = isinstance(part, dict) and part.get("impl")
                    if not impl or not (ROOT / impl).exists():
                        continue          # 新文件的样例没法加载，只查复用现成零件的
                    checked += 1
                    try:
                        load({section: {name: part}})
                    except Exception as e:          # noqa: BLE001
                        broken.append((where, f"{section}.{name}", repr(e)))
    assert checked, "一个复用现成零件的样例都没找到 —— 正则是不是失效了"
    assert not broken, f"照抄这些样例会装不上零件：{broken}"


def test_不再点名退役任务的数据字段():
    hits = []
    for where, text in _llm_visible_texts():
        hits += [(where, f) for f in RETIRED_FIELDS
                 if re.search(rf"(?<![\w]){f}(?![\w])", text)]
        hits += [(where, m.group(0)) for m in _ALICCP_FIELD_CODE.finditer(text)]
    assert not hits, f"还在点名 AliCCP 的字段：{hits}"

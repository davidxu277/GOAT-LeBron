"""离线演习和假成绩单要说当前任务的话。

演习是整个系统的安全网 —— 不花钱就把四个角色 + 外层循环从头跑到尾。可它的
假模型、假执行器、5 份假成绩单一直是 AliCCP 格式（点击分 / 购买分 / 购买AUC），
新任务的指标名、成绩单块名在演习里一次都没出现过。网眼跟问题一样大：
KuaiRand 上才会出的错，演习永远跑不出来。

现在：假执行器出的是 KuaiRand 成绩单（验证集 GAUC / nDCG@5 / 主分，分组块沿用
真成绩单的块名），假模型从 schema 里读指标名而不是写死，5 份假成绩单全部重写。
"""

import re

import yaml

from agent import schemas
from agent.cli import FIXTURES
from agent.knowledge import CardLibrary, SymptomVocab
from agent.loop import run_session
from agent.offline import DriftingExecutor, ScriptedLLM

RETIRED = re.compile(r"点击分|购买分|点击AUC|购买AUC|\bclick\b|conversion|ctcvr|\b\d{3}_\d{2}\b")


def test_假执行器出的是KuaiRand成绩单():
    r = DriftingExecutor().report()
    assert {"GAUC", "nDCG@5", "主分"} <= set(r["验证集"])
    assert schemas.metric_names(r) == list(schemas.METRICS)
    assert not RETIRED.search(yaml.safe_dump(r, allow_unicode=True))


def test_假成绩单全是KuaiRand格式():
    text = FIXTURES.read_text(encoding="utf-8")
    for name, fx in yaml.safe_load(text).items():
        assert {"GAUC", "nDCG@5", "主分"} <= set(fx["report"]["验证集"]), name
    assert not RETIRED.search(text)


def test_假成绩单的标准答案只用现行病名():
    vocab = SymptomVocab.load()
    text = FIXTURES.read_text(encoding="utf-8")
    for name, fx in yaml.safe_load(text).items():
        for sid in re.findall(r"「([^」]+)」", str(fx["expect"])):
            assert sid in vocab, f"{name} 的标准答案里有不存在的病名「{sid}」"


def test_演习整场说的都是当前任务的指标名(tmp_path):
    v = SymptomVocab.load()
    ex = DriftingExecutor(fail_rounds=(3,))
    run_session(
        llm=ScriptedLLM(), vocab=v, cards=CardLibrary.load(v), executor=ex,
        initial_report=ex.report(), module_interface="", example_module="",
        current_config="", rounds=4, logs_dir=tmp_path)
    logs = [p for p in tmp_path.rglob("*") if p.suffix in (".json", ".jsonl")]
    assert logs, "演习一条日志都没写"
    text = "\n".join(p.read_text(encoding="utf-8") for p in logs)
    assert not RETIRED.search(text)

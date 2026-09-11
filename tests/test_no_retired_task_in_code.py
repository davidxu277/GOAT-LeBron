"""代码里不许再有退役任务（AliCCP）的字段名和指标名。

换任务之后，agent 核心里还留着好几处"如果是旧任务就……"的兼容分支：
schemas 里 AliCCP 那一套指标对照、roles 里认「点击分」的过拟合闸门、
禁用字段里 AliCCP 的五个列名、loop 里"成绩单没有主分就退回双指标求和"……
它们平时不出声，但每一处都是一条旧东西漏进 agent 的路 —— 过期卡片、
过期范文、过期的指标名单，都是这么来的。

只查**代码里的字符串字面量**（跳过注释和 docstring）：解释历史的文字可以留，
能被程序读到、比较、写进提示词的名字不行。
"""

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
RETIRED = re.compile(
    r"^(点击分|购买分|点击AUC|购买AUC|click|conversion|ctcvr|sample_id|common_id|\d{3}(_\d{2})?)$")
SCOPE = [*ROOT.glob("agent/*.py"), ROOT / "harness" / "deep.py", ROOT / "harness" / "ops.py",
         *ROOT.glob("modules/**/*.py")]


def _string_literals(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = {
        id(node.body[0].value)
        for node in ast.walk(tree)
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
        and node.body and isinstance(node.body[0], ast.Expr)
        and isinstance(node.body[0].value, ast.Constant)
    }
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            yield node.value


def test_代码里不再有退役任务的字段和指标名():
    hits = [
        (p.relative_to(ROOT).as_posix(), s)
        for p in SCOPE if p.exists() and "__pycache__" not in p.parts
        for s in _string_literals(p) if RETIRED.match(s)
    ]
    assert not hits, f"还在用退役任务的名字：{hits}"


# 写给人和 Agent 看的规矩与说明 —— CLAUDE.md 是硬约束，工兵写代码前要读它。
# 它要是还在讲 click / conversion，等于让 Agent 照着别的任务守规矩。
# 「AliCCP」这个名字本身可以出现（讲历史），它的字段和指标名不行。
# AGENTS.md 是 CLAUDE.md 的同内容副本（给别的编码助手读），两份一起盯。
DOCS = [ROOT / "CLAUDE.md", ROOT / "AGENTS.md", ROOT / "README.md", ROOT / "agent" / "README.md"]
RETIRED_IN_TEXT = re.compile(
    r"(?<![\w])(click|conversion|ctcvr|sample_id|common_id)(?![\w])"
    r"|点击分|购买分|点击\s*AUC|购买\s*AUC|(?<![\w.])\d{3}_\d{2}(?![\w.])")


def test_规矩和说明文档不再讲退役任务的字段和指标名():
    hits = [
        (p.relative_to(ROOT).as_posix(), m.group(0))
        for p in DOCS
        for m in RETIRED_IN_TEXT.finditer(p.read_text(encoding="utf-8"))
    ]
    assert not hits, f"文档还在讲退役任务：{hits}"

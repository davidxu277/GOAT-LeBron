"""噪声带：先知道体温计的误差，再决定 37.2°C 算不算发烧。

同一份配置、同一份数据子集，只换训练种子跑 N 次，看每个指标自己抖多少。
比的是「两次运行之间的差」：每次各带 σ 的抖动，差的标准差是 √2·σ，
取两倍 → 噪声带 = 2·√2·σ，两次同配置跑，差距约 95% 落在带内。

代码里到处在用「噪声带」判「这个差距算不算真的」，但在 KuaiRand 上从没量过，
一直用的是拍出来的 0.0005。
"""

import json
import math
import pathlib
import statistics
import sys
import tempfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from kuairand_bridge import noise  # noqa: E402

TRAINER = str(ROOT / "examples" / "goat_trainer.py")


def test_噪声带是两倍根号二倍标准差():
    runs = [{"GAUC": g, "nDCG@5": n, "主分": (g + n) / 2}
            for g, n in [(0.660, 0.530), (0.662, 0.534), (0.661, 0.529), (0.659, 0.531)]]
    b = noise.bands_from_scores(runs)
    sd_g = statistics.stdev([0.660, 0.662, 0.661, 0.659])
    assert b["分指标噪声带"]["GAUC"] == pytest.approx(2 * math.sqrt(2) * sd_g, abs=1e-5)
    assert set(b["分指标噪声带"]) == {"GAUC", "nDCG@5"}      # 主分不算分指标
    assert b["单指标噪声带"] > 0                               # 主分的带子单独放
    assert b["统计"]["GAUC"]["次数"] == 4


def test_少于三次拒绝算():
    """两个点算出来的标准差没有意义，宁可不给也别给一把歪尺子。"""
    with pytest.raises(ValueError, match="至少"):
        noise.bands_from_scores([{"GAUC": 0.66, "nDCG@5": 0.53, "主分": 0.595}] * 2)


# ── 量：用假 runner，看流程对不对 ──

def 按种子出分(data_dir, trainer_path, output_dir, seed, make_test, **kw):
    """模块顶层（子进程要能 pickle）。分数随训练种子变，并记下抽样种子。"""
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample_seed.txt").write_text(str(kw.get("sample_seed")), encoding="utf-8")
    g, n = 0.66 + seed * 0.001, 0.53 - seed * 0.0005
    return {"validation": {"metrics": {"GAUC": g, "nDCG@5": n, "primary": (g + n) / 2,
                                       "rows": 10, "users": 2}}}


def 第二个种子挂(data_dir, trainer_path, output_dir, seed, make_test, **kw):
    if seed == 2:
        raise RuntimeError("显存炸了")
    return 按种子出分(data_dir, trainer_path, output_dir, seed, make_test, **kw)


def _量(tmp, runner, seeds=(1, 2, 3)):
    return noise.measure_noise(
        data_dir=tmp, trainer_path=TRAINER, output_dir=pathlib.Path(tmp) / "noise",
        fidelity="小份", train_seeds=list(seeds), sample_seed=0,
        trainer_config={}, runner=runner)


def test_每个训练种子各跑一次_数据子集钉死():
    with tempfile.TemporaryDirectory() as tmp:
        b = _量(tmp, 按种子出分)
        assert [r["训练种子"] for r in b["逐次分数"]] == [1, 2, 3]
        assert b["保真度"] == "小份" and b["数据抽样种子"] == 0
        抽样 = {p.read_text(encoding="utf-8")
                for p in pathlib.Path(tmp, "noise").rglob("sample_seed.txt")}
        assert 抽样 == {"0"}
        assert b["分指标噪声带"]["GAUC"] > 0


def test_有一次跑挂就整个报错_不拿残缺数据算():
    with tempfile.TemporaryDirectory() as tmp, pytest.raises(RuntimeError, match="种子 2"):
        _量(tmp, 第二个种子挂)


def test_落盘的文件run_session直接能用(tmp_path):
    b = {"保真度": "小份", "分指标噪声带": {"GAUC": 0.001, "nDCG@5": 0.002},
         "单指标噪声带": 0.0015}
    p = noise.save_bands(tmp_path, b)
    assert p == tmp_path / "noise_bands.json"
    assert noise.load_bands(tmp_path) == b
    assert noise.load_bands(tmp_path / "没有这个目录") is None


# ── 真跑时读进来 ──

def test_真跑时找到带子就用(tmp_path):
    from kuairand_bridge import goat_run
    noise.save_bands(tmp_path, {"保真度": "小份", "数据抽样种子": 0,
                                "分指标噪声带": {"GAUC": 0.001, "nDCG@5": 0.002}})
    b = goat_run._noise_bands_for(tmp_path, seed=0)
    assert b["分指标噪声带"]["GAUC"] == 0.001


def test_没量过就是None_run_session自己退回兜底(tmp_path):
    from kuairand_bridge import goat_run
    assert goat_run._noise_bands_for(tmp_path, seed=0) is None


def test_抽样种子对不上要喊出来_但照用(tmp_path, capsys):
    """带子量的是训练随机性，换一份同规模的数据子集，幅度差不多 —— 照用，但要说清楚。"""
    from kuairand_bridge import goat_run
    noise.save_bands(tmp_path, {"保真度": "小份", "数据抽样种子": 7,
                                "分指标噪声带": {"GAUC": 0.001, "nDCG@5": 0.002}})
    assert goat_run._noise_bands_for(tmp_path, seed=0) is not None
    assert "抽样种子" in capsys.readouterr().out


def test_命令行有noise子命令():
    from kuairand_bridge import cli
    with pytest.raises(SystemExit) as 停:
        cli.main(["noise", "--help"])
    assert 停.value.code == 0          # 不认识的子命令也会 SystemExit，只是返回码是 2


def 记验证集用哪块(data_dir, trainer_path, output_dir, seed, make_test, **kw):
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "valid_part.txt").write_text(str(kw.get("valid_part")), encoding="utf-8")
    return 按种子出分(data_dir, trainer_path, output_dir, seed, make_test, **kw)


def test_开了锁定集时噪声也在开发集上量():
    """一场里每一轮在开发集上打分，带子就得在开发集上量 —— 人少了抖动会大，尺子要对得上。"""
    with tempfile.TemporaryDirectory() as tmp:
        b = noise.measure_noise(
            data_dir=tmp, trainer_path=TRAINER, output_dir=pathlib.Path(tmp) / "noise",
            fidelity="小份", train_seeds=[1, 2, 3], sample_seed=0,
            trainer_config={}, runner=记验证集用哪块, holdout_frac=0.2)
        assert {p.read_text(encoding="utf-8")
                for p in pathlib.Path(tmp, "noise").rglob("valid_part.txt")} == {"开发"}
        assert b["锁定集比例"] == 0.2

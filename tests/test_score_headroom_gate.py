"""上端饱和门限 + 技能指数留白规则（score_review_proposal §2 P1/P3，2026-10-18 修）。

两条被锁死的纪律：

1. **上端饱和门限**（标定脚本 §1.5，与下端门限对称）：以往体检只看 [0,5]，
   "过半数格子接近满分"没人管；现在 [98,100] 占比超阈值同样告警/失败。带沿取
   98 而不是 95 是**测量结果不是口味**：这个窗口上 slope/amt_bias 的 [95,100]
   占比 47%/52%，那是 log/dev 族 v≈1 点质量的数据本性（任何 λ 都 cure 不掉），
   门限若定在 95 就会天天亮红灯、被无视；定在 98 仍能抓住真饱和（>30% 格子
   ≥98 分）。
2. **维度分留白规则**（标定脚本 §1.6）：技能指数增益不再按"子分 sd=8"反解
   （那会把 >68.8 的维度分全钉在 100——P1），而是取标定窗口全部维度分的 P99.5
   映到软上限 98：gain=(98−50)/(P99.5−50)。撞顶允许按构造存在，但占比上限
   0.5% 且**不允许同 (轨×维×桶) 并列**（同维并列 = 系统性截断 = P1 本象）。

测试用合成分布锁算术与判定边界；真实标定窗口上的实测值（gain=1.851、
撞顶 2/1541）在 docs/score_slopes.md §九 留档，并由 test_evaluate 的冻结值断言。
"""
from __future__ import annotations

import functools
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "calibrate_score_slopes.py"


@functools.lru_cache(maxsize=1)
def _cal():
    """加载标定脚本模块（scripts/ 不是包，按路径装载；主入口有 __main__ 保护）。"""
    spec = importlib.util.spec_from_file_location("calibrate_score_slopes", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------- 门限常量与判定
def test_upper_saturation_gate_constants():
    cal = _cal()
    # 上带沿必须高于诊断列 95（否则数据本性项会把告警训练成噪声，见模块注释）
    assert cal.SATURATION_HI_BAND == 98.0
    assert cal.SATURATION_HI_BAND > 95.0
    # 与下端门限同阈值（对称设计）
    assert cal.SATURATION_FAIL_SHARE_HI == cal.SATURATION_FAIL_SHARE == 0.30


def test_saturation_bands_math_and_boundaries():
    cal = _cal()
    # 均匀铺满 10~90：两端带都不触发
    mid = np.linspace(10, 90, 1001)
    hi95, hi, lo = cal.saturation_bands(mid)
    assert hi95 == 0.0 and hi == 0.0 and lo == 0.0
    # 40% 贴 99：上带触发（>30%），95 诊断列同时亮
    a = np.concatenate([np.full(400, 99.0), np.linspace(10, 90, 600)])
    hi95, hi, lo = cal.saturation_bands(a)
    assert hi == pytest.approx(0.4) and hi95 == pytest.approx(0.4) and lo == 0.0
    # 60% 贴 0：下带触发
    b = np.concatenate([np.zeros(600), np.linspace(10, 90, 400)])
    _hi95, hi, lo = cal.saturation_bands(b)
    assert lo == pytest.approx(0.6) and hi == 0.0
    # 截断纪律：越界值按量表边界归类（-5 → ≤5；105 → ≥98），宽口径不松口
    _h95, hi, lo = cal.saturation_bands(np.array([-5.0, 105.0] * 10))
    assert hi == pytest.approx(0.5) and lo == pytest.approx(0.5)
    # 30% 恰好不触发（判定是严格大于，边界不放行也不误杀）
    c = np.concatenate([np.full(300, 99.0), np.linspace(10, 90, 700)])
    _h95, hi, _lo = cal.saturation_bands(c)
    assert hi == pytest.approx(0.3)


# ------------------------------------------------------------ 留白增益反解（P1）
def test_derive_headroom_gain_rule_and_anchor():
    cal = _cal()
    assert cal.HEADROOM_SOFT_CAP == 98.0 and cal.HEADROOM_QUANTILE == 99.5
    # 均匀 0~100（2001 点）：P99.5 = 99.5（线性插值），gain = 48/(99.5−50)
    vals = np.linspace(0.0, 100.0, 2001)
    gain, pq = cal.derive_headroom_gain(vals)
    assert pq == pytest.approx(99.5)
    assert gain == pytest.approx(48.0 / (99.5 - 50.0))
    # 规则语义：P99.5 映到软上限（50 + gain·(Pq−50) == 98）
    assert 50.0 + gain * (pq - 50.0) == pytest.approx(98.0)
    # 中心是仿射不动点
    assert 50.0 + gain * (50.0 - 50.0) == pytest.approx(50.0)
    # 数据不足（Pq ≤ 50，全榜挤在中点以下）→ 明确抛错而不是产出假增益
    with pytest.raises(ValueError):
        cal.derive_headroom_gain(np.full(100, 30.0))


def test_clip_structure_counts_ties_and_share():
    cal = _cal()
    # 合成 1000 格：gain=2.0 时阈值 = 75（50+2·(75−50) == 100，恰好不撞；
    # >75 才撞）。造 2 个孤立撞顶 → 占比 0.2% 通过、无并列。
    cells = [("hourly", "temp", f"m{i}", "1d", 60.0) for i in range(998)]
    cells += [("hourly", "temp", "mA", "1d", 80.0),
              ("daily", "precip", "mB", "2d", 90.0)]
    st = cal.clip_structure(cells, 2.0)
    assert st["n_clip"] == 2 and st["max_in_bucket"] == 1 and st["ok"]
    # 边界：恰好 100 不算撞顶（截断后与 100 相同、线性外推未越限）
    edge = cells + [("hourly", "temp", "mE", "1d", 75.0)]
    assert cal.clip_structure(edge, 2.0)["n_clip"] == 2
    # 同 (轨×维×桶) 两格并列撞顶 = 系统性截断（P1 本象）→ 判定失败
    tie = [("hourly", "temp", f"m{i}", "1d", 60.0) for i in range(998)]
    tie += [("hourly", "temp", "mA", "1d", 80.0),
            ("hourly", "temp", "mB", "1d", 82.0)]
    st2 = cal.clip_structure(tie, 2.0)
    assert st2["n_clip"] == 2 and st2["max_in_bucket"] == 2 and not st2["ok"]
    # 并列跨桶不算并列（每个 桶×维 仍是单格触顶）
    st3 = cal.clip_structure(cells, 2.0)
    assert st3["by_bucket"][("hourly", "temp", "1d")] == 1
    # 占比超上限（0.5%）→ 失败，即使无并列
    many = [("hourly", "temp", f"m{i}", "1d", 60.0) for i in range(990)]
    many += [("hourly", "temp", f"top{i}", "1d", 90.0) for i in range(10)]
    st4 = cal.clip_structure(many, 2.0)
    assert st4["n_clip"] == 10 and st4["max_in_bucket"] == 10 and not st4["ok"]


def test_frozen_gain_matches_rule_and_config_sources():
    """冻结的 gain 必须与规则、config 缺省、stations.yaml 三方一致。"""
    from weather_eval.config import DEFAULT_EVAL, load_config

    g = float(DEFAULT_EVAL["score_index_gain"])
    # 1.851 = (98−50)/(75.93−50)（标定窗口 P99.5，留档 docs/score_slopes.md §九）
    assert abs(g - 48.0 / (75.93 - 50.0)) < 5e-4
    assert abs(g - 1.851) < 1e-12
    # 规则描述必须留在注释里（"声明规则、不是只声明数字"）
    import inspect
    from weather_eval import config as cfg_mod
    src = inspect.getsource(cfg_mod)
    assert "P99.5" in src and "1.851" in src
    # stations.yaml 的运维值与代码缺省一致（P8 门槛同一测试模式）
    cfg = load_config(None)
    assert abs(float(cfg.eval["score_index_gain"]) - g) < 1e-12

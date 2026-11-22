"""skill-v8（2026-11 绝对技能综合分轮）机制断言。

v8 继承 v6/v7 的全部单格评分机制，只改**综合分的构念**：
  1. 综合分是 `50 + α(T−50) + (1−α)(P−50)` 的绝对技能分，不是桶内 z 分；
  2. 50 = 气候参照，不是"本档同台平均"；1 分 = 技能尺 1 个百分点；
  3. 可靠性由聚合层分块 EB 收缩 + 目标覆盖份额 q 承担，不做逐格 √ρ 加权；
  4. 单维证据不产生综合分排名（只在该维可见）；
  5. 同台家数不再是绝对分的前提：1 家也能算出自己的绝对分。

这些断言锁行为，不锁某个快照数字。把实现改回 v7 的 z 分/证据乘子，
对应断言会变红。
"""
from __future__ import annotations

import numpy as np
import pytest

from weather_eval import evaluate as ev
from weather_eval import stats


def setup_function():
    ev.apply_score_scheme("skill-v8")
    ev.apply_score_composite_params(
        alpha=0.5, sd_temp=8.77, sd_precip=3.73, rank_scale=10.0,
        tau=1.0, display_gain=1.0,
        lead_weights={"1-3": 1.0, "4-7": 0.7, "8-14": 0.4, "15-16": 0.2})
    ev.apply_score_reliability(True, 0.20, 3.5, 0.20, 0.10)


def teardown_function():
    ev.apply_score_scheme("slope-v2")


# ---------------------------------------------------------------- V8-1 绝对锚点
def test_v8_composite_is_absolute_skill_not_pool_z():
    """50/100 是气候参照/完美锚点；两维各 50 的源必须得 50，与同台对手无关。"""
    got = stats.composite_from_skill(
        np.array([50.0, 100.0, 70.0, 30.0]),
        np.array([50.0, 100.0, 50.0, 30.0]),
        alpha=0.5)
    assert np.allclose(got["rank_score"], [50.0, 100.0, 60.0, 30.0])
    # 贡献列 50 居中且严格可加
    assert np.allclose(
        got["rank_score"],
        got["temp_contrib"] + got["precip_contrib"] - 50.0)
    # 同样的绝对分，换一批更强的对手也不改变它——z 分会
    solo = stats.composite_from_skill(
        np.array([70.0]), np.array([60.0]), alpha=0.5)
    assert float(solo["rank_score"][0]) == 65.0


def test_v8_composite_is_monotone_in_both_dimensions():
    t = np.array([40.0, 50.0, 60.0, 70.0])
    p = np.full(4, 50.0)
    got = stats.composite_from_skill(t, p, alpha=0.5)
    assert np.all(np.diff(got["rank_score"]) > 0)
    p2 = np.array([40.0, 50.0, 60.0, 70.0])
    got2 = stats.composite_from_skill(np.full(4, 50.0), p2, alpha=0.5)
    assert np.all(np.diff(got2["rank_score"]) > 0)


def test_v8_alpha_is_value_weight_in_skill_points_not_variance():
    """α=0.5 时两维各贡献 5 分（T=60/P=50 → 55），不是 z 分的方差份额。"""
    got = stats.composite_from_skill(
        np.array([60.0]), np.array([50.0]), alpha=0.5)
    assert float(got["temp_contrib"][0]) == 55.0
    assert float(got["precip_contrib"][0]) == 50.0
    assert float(got["rank_score"][0]) == 55.0
    got = stats.composite_from_skill(
        np.array([60.0]), np.array([70.0]), alpha=0.3)
    assert float(got["temp_contrib"][0]) == pytest.approx(53.0)
    assert float(got["precip_contrib"][0]) == pytest.approx(64.0)


# ---------------------------------------------------------------- V8-2 单维/缺证
def test_v8_single_dimension_is_visible_but_not_a_composite():
    got = stats.composite_from_skill(
        np.array([70.0]), np.array([np.nan]), alpha=0.5)
    assert float(got["rank_score"][0]) == 60.0
    assert float(got["precip_contrib"][0]) == 50.0
    assert bool(got["single_dim"][0])
    assert str(got["basis"][0]) == "temp_only"
    # require_both：单维不产生综合分（总榜/分榜的排名口径）
    got2 = stats.composite_from_skill(
        np.array([70.0]), np.array([np.nan]), alpha=0.5, require_both=True)
    assert not np.isfinite(got2["rank_score"][0])
    board = ev._bucket_absolute_rows([("a", 70.0, None)])
    assert board["a"][0] is None and board["a"][3] == "temp_only"
    assert board["a"][1] == 60.0 and board["a"][2] == 50.0


def test_v8_per_lead_board_does_not_require_three_models():
    """绝对分对 1~2 家同样有定义；旧 z 分在 <3 家时直接作废。"""
    got = ev._bucket_absolute_rows([("a", 70.0, 60.0), ("b", 55.0, 50.0)])
    assert got["a"][0] == 65.0 and got["a"][3] == "both"
    assert got["b"][0] == 52.5 and got["b"][3] == "both"


def test_v8_no_skill_bucket_neutralizes_precip_but_keeps_temp():
    rng = np.random.default_rng(7)
    M, K = 8, 13
    T = rng.normal(55.0, 8.0, (M, K))
    P = rng.normal(55.0, 5.0, (M, K))
    mats = {"temp_score": T, "precip_score": P, "ets": np.zeros_like(P)}
    out = ev._attach_rank_matrices(mats, n_blocks=K, days=1, track="hourly")
    assert np.allclose(out["precip_contrib"], 50.0)
    assert np.allclose(out["rank_score"], out["temp_contrib"])
    assert not out["_gate_precip"].any()
    assert out["_gate_temp"].all()
    assert out["_no_skill_bucket"][0]


# ---------------------------------------------------------------- V8-3 覆盖收缩
def test_v8_coverage_share_is_horizon_coverage_not_sample_volume():
    valid = np.array([[True, True, False, False],
                      [True, True, True, True],
                      [False, False, False, False]])
    target = np.array([1.0, 1.0, 1.0, 1.0])
    q = stats.coverage_share(valid, target)
    assert np.allclose(q, [0.5, 1.0, 0.0])
    # 目标权重只由 lead 决定：同样的格子，长时效权重低则 q 不同
    q2 = stats.coverage_share(valid, np.array([1.0, 1.0, 0.2, 0.2]))
    assert q2[0] == pytest.approx(2.0 / 2.4)
    assert q2[1] == pytest.approx(1.0)


def test_v8_eb_coverage_shrinks_short_coverage_toward_panel_mean():
    """覆盖越短，行效应越向面板均值收缩；满覆盖不收缩。"""
    alpha = np.array([10.0, -5.0, 0.0])
    resid = np.repeat(np.array([5.0, -2.0, 0.0])[:, None], 4, axis=1)
    w = np.ones_like(resid)
    present = np.ones_like(resid, dtype=bool)
    full, _ = stats.eb_shrink(alpha, resid, w, present,
                              n_eff=np.array([1.0, 1.0, 1.0]))
    half, _ = stats.eb_shrink(alpha, resid, w, present,
                              n_eff=np.array([0.5, 0.5, 0.5]))
    # 面板均值 ≈ 1.67；半覆盖的偏离必须更小
    mu = 1.6666666667
    assert abs(half[0] - mu) < abs(full[0] - mu)
    assert abs(half[1] - mu) < abs(full[1] - mu)
    # 覆盖为 0 = 没有目标人口证据 ⇒ 收缩系数 0（回到面板均值）
    zero, info = stats.eb_shrink(alpha, resid, w, present,
                                 n_eff=np.zeros(3))
    assert np.allclose(zero, info["mu"])


# ---------------------------------------------------------------- V8-4 家族/披露
def test_v8_family_inherits_v6_mechanisms():
    assert "skill-v8" in ev.SKILL_SCHEMES
    assert ev.is_v8_family("skill-v8")
    assert ev.is_v6_family("skill-v8")
    assert ev.is_v5_family("skill-v8")
    assert not ev.is_v7_family("skill-v8")
    with pytest.raises(ValueError):
        ev.apply_score_scheme("skill-v9")


def test_v8_rank_diagnostics_uses_column_level_availability():
    """v8 的逐格门不得把"第一个源没数据"误报成整列退出。"""
    M, K = 5, 3
    T = np.full((M, K), 55.0)
    P = np.full((M, K), 52.0)
    T[0, :] = np.nan
    mats = {"temp_score": T, "precip_score": P,
            "ets": np.full((M, K), 0.2)}
    out = ev._attach_rank_matrices(mats, n_blocks=1, days=K, track="hourly")
    diag = ev._rank_diagnostics(out, 1, K, "hourly")
    assert diag["gate_counts"]["temp_exit"] == 0
    assert diag["gate_counts"]["precip_exit"] == 0
    assert diag["gate_counts"]["gate_basis"] == "data_availability_absolute"
    assert diag["gate_counts"]["weighted"] is False


def test_v8_old_schemes_are_not_switched():
    """v7 归档口径必须仍叫 v7，不能被 v8 家族判定吞掉。"""
    ev.apply_score_scheme("skill-v7")
    assert ev.is_v7_family() and not ev.is_v8_family()
    ev.apply_score_scheme("skill-v8")
    assert ev.is_v8_family() and not ev.is_v7_family()

"""综合分口径优化（2026-11 A/B 批）的机制断言。

每条都对应 `SCORING.md §9` 的一行：锁的是**行为**而不是某个数字。
"""
from __future__ import annotations

import numpy as np
import pytest

from weather_eval import evaluate as ev
from weather_eval import stats


@pytest.fixture(autouse=True)
def _v5():
    ev.apply_score_scheme("skill-v5")
    ev.apply_score_composite_params(
        alpha=0.5, sd_temp=8.77, sd_precip=3.73, rank_scale=10.0,
        tau=1.0, display_gain=1.0,
        lead_weights={"1-3": 1.0, "4-7": 0.7, "8-14": 0.4, "15-16": 0.2})
    yield
    ev.apply_score_scheme("slope-v2")


# ---------------------------------------------------------------- A5 / T01
def test_grade_bounds_are_open_ended():
    inf = float("inf")
    assert stats.grade_bounds("1h") == ((2.0, inf), (5.0, inf))
    assert stats.grade_bounds("24h") == ((10.0, inf), (25.0, inf))
    # obs=5.0 / fcst=4.9 不得同时产生 2 档空报与 3 档漏报
    (h1, fa1, mi1), (h2, fa2, mi2) = stats.grade_counts(
        np.array([5.0]), np.array([4.9]), stats.grade_bounds("1h"))
    assert not (fa1 and mi2)
    # 报 4.9 的罚分不得重于报 0（单调性）
    _, fa_zero, mi_zero = stats.grade_counts(
        np.array([5.0]), np.array([0.0]), stats.grade_bounds("1h"))[1]
    _, fa_near, mi_near = stats.grade_counts(
        np.array([5.0]), np.array([4.9]), stats.grade_bounds("1h"))[1]
    assert fa_near <= fa_zero and mi_near <= mi_zero


# ---------------------------------------------------------------- A1 / A3
def test_rank_core_is_additive_and_alpha_is_value_weight():
    rng = np.random.default_rng(20261108)
    n = 4000
    T = rng.normal(70.0, 8.0, (n, 1))
    P = rng.normal(45.0, 4.0, (n, 1))
    for a in (0.5, 0.3, 0.7):
        got = stats.rank_from_scores(
            T, P, alpha=a, rho_t=0.8, rho_p=0.4, winsor_k=0)
        r, tc, pc = got["rank_score"][:, 0], got["temp_contrib"][:, 0], \
            got["precip_contrib"][:, 0]
        assert np.allclose(r, tc + pc - 50.0, atol=1e-9)
        ratio = float(np.var(tc) / np.var(pc))
        expect = (a / (1 - a)) ** 2
        assert abs(ratio - expect) / expect < 0.05


def test_reliability_only_changes_amplitude_not_shares():
    rng = np.random.default_rng(7)
    T = rng.normal(70.0, 8.0, (2000, 1))
    P = rng.normal(45.0, 4.0, (2000, 1))
    off = stats.rank_from_scores(T, P, alpha=0.5, rho_t=1.0, rho_p=1.0,
                                 reliability=False)
    on = stats.rank_from_scores(T, P, alpha=0.5, rho_t=0.2, rho_p=0.9)
    share_off = float(np.var(off["temp_contrib"]) /
                      np.var(off["precip_contrib"]))
    share_on = float(np.var(on["temp_contrib"]) /
                     np.var(on["precip_contrib"]))
    assert abs(share_on - share_off) / share_off < 0.02
    assert float(np.std(on["rank_score"])) < float(np.std(off["rank_score"]))


# ---------------------------------------------------------------- A1 three chains
def test_attach_rank_matrices_delegates_to_the_single_core():
    rng = np.random.default_rng(11)
    M, K, D = 8, 3, 2
    T = rng.normal(70.0, 8.0, (M, K * D))
    P = rng.normal(45.0, 4.0, (M, K * D))
    mats = {"temp_score": T, "precip_score": P,
            "ets": rng.normal(0.1, 0.1, (M, K * D))}
    got = ev._attach_rank_matrices(mats, n_blocks=K, days=D)
    direct = stats.rank_from_scores(
        T, P, alpha=ev.SCORE_COMPOSITE_ALPHA,
        rho_t=got["_rel_temp"], rho_p=got["_rel_precip"],
        gate_t=got["_gate_temp"][0], gate_p=got["_gate_precip"][0],
        min_models=ev.SCORE_BUCKET_MIN_MODELS,
        prior_sd=(ev.SCORE_COMPOSITE_SD_TEMP,
                  ev.SCORE_COMPOSITE_SD_PRECIP),
        prior_n=ev.SCORE_BUCKET_SD_PRIOR_N,
        floor_frac=ev.SCORE_BUCKET_SD_FLOOR_FRAC,
        winsor_k=ev.SCORE_WINSOR_K, scale=ev.SCORE_RANK_SCALE)
    assert np.allclose(got["rank_score"], direct["rank_score"],
                       atol=1e-9, equal_nan=True)
    assert np.allclose(got["temp_contrib"], direct["temp_contrib"],
                       atol=1e-9, equal_nan=True)
    assert np.allclose(got["precip_contrib"], direct["precip_contrib"],
                       atol=1e-9, equal_nan=True)


# ---------------------------------------------------------------- A4 / T06/T07
def test_single_dimension_is_reachable_and_has_same_scale():
    T = np.array([[65.0, 62.0], [60.0, 57.0], [55.0, 52.0],
                  [50.0, 47.0], [45.0, 42.0], [40.0, 37.0]])
    P = np.array([[57.5, np.nan], [55.5, np.nan], [53.5, np.nan],
                  [51.5, np.nan], [49.5, np.nan], [47.5, np.nan]])
    got = stats.rank_from_scores(
        T, P, alpha=0.5, rho_t=0.6, rho_p=0.6, min_models=3, winsor_k=0)
    r, tc, pc = got["rank_score"][:, 1], got["temp_contrib"][:, 1], \
        got["precip_contrib"][:, 1]
    assert np.all(np.isfinite(r))
    assert got["single_dim"][:, 1].all()
    assert np.allclose(pc, 50.0)
    assert np.allclose(r, tc)
    # 两维齐备列（φ=1）与单维列的 R 条件 sd 同量级（旧 1/α 放大已删除）
    ratio = float(np.std(got["rank_score"][:, 0]) /
                  np.std(got["rank_score"][:, 1]))
    assert 0.9 <= ratio <= 1.1


def test_bucket_level_gate_exits_precip_but_keeps_temp():
    M, K, D = 6, 3, 1
    T = np.array([[60.0] * K, [58.0] * K, [56.0] * K,
                  [54.0] * K, [52.0] * K, [50.0] * K]).reshape(M, K)
    P = np.full((M, K), 40.0)          # 降水无区分度
    ets = np.zeros((M, K))             # 全 0 ETS → 无技巧桶
    mats = {"temp_score": T, "precip_score": P, "ets": ets}
    got = ev._attach_rank_matrices(mats, n_blocks=K, days=D)
    assert not got["_gate_precip"][0, 0]
    assert got["_gate_temp"][0, 0]
    assert got["_no_skill_bucket"][0]


# ---------------------------------------------------------------- A2 / T09
def test_icc1_recovers_known_reliability():
    rng = np.random.default_rng(17)
    M, K = 30, 10
    a = rng.normal(0, 1.0, M)
    beta = rng.normal(0, 2.0, K)
    e = rng.normal(0, 0.5, (M, K))
    X = a[:, None] + beta[None, :] + e
    rho = stats.icc1_reliability(X)
    expect = 1.0 / (1.0 + 0.25)
    assert abs(rho - expect) < 0.1
    # 纯块效应 → ρ≈0
    X_block = np.repeat(beta[None, :], M, axis=0)
    assert stats.icc1_reliability(X_block) == pytest.approx(0.0, abs=0.05)
    # 纯模型效应（无噪声）→ ρ=1
    X_model = np.repeat(a[:, None], K, axis=1)
    assert stats.icc1_reliability(X_model) == pytest.approx(1.0, abs=1e-9)


# ---------------------------------------------------------------- B1 / T12
def test_amt_skill_is_full_sample_relative_mae():
    def s(obs, fc):
        return ev.precip_amount_metrics(
            np.asarray(obs), np.asarray(fc), 5, amt_floor=0.05,
            wet_threshold=1.0)["amt_skill"]

    obs = np.array([0.0, 2.0, 3.0, 1.0, 0.0, 2.0])
    base = obs.copy()
    near_up = s(obs, np.array([1.01, 2.0, 3.0, 1.0, 0.0, 2.0]))
    near_down = s(obs, np.array([0.99, 2.0, 3.0, 1.0, 0.0, 2.0]))
    assert abs(near_up - near_down) < 0.01          # 阈值跳变消失
    assert s(obs, np.zeros_like(obs)) == 0.0         # 恒零雨 = 参照
    assert s(obs, base) == 1.0                       # 完美
    assert ev.precip_amount_metrics(
        np.zeros(5), np.ones(5), 5, amt_floor=0.05,
        wet_threshold=1.0)["amt_skill"] is None
    # 新增一个真实更差的预报不得让 s 上升
    obs2 = np.array([0.9, 20.0, 30.0, 1.0, 2.0, 3.0])
    worse = s(obs2, np.array([5.0, 60.0, 60.0, 1.0, 2.0, 3.0]))
    better = s(obs2, np.array([0.0, 60.0, 60.0, 1.0, 2.0, 3.0]))
    assert worse < better


# ---------------------------------------------------------------- B4 / T15
def test_structural_absence_is_source_level():
    amp = {"rmse": 1.5, "mae": 1.0, "acc2": 70.0,
           "ref": {"rmse_ref": 2.0, "mae_ref": 1.5, "acc2_ref": 70.0}}
    assert ev.temp_score(amp, absent_families={"形态与相位"}) is not None
    assert ev.temp_score(amp) is None
    # 源在别处有 r_anom、本格全缺 → 半截证据
    t_partial = {"rmse": 1.5, "ref": {"rmse_ref": 2.0}}
    assert ev.temp_score(t_partial, absent_families={"形态与相位"}) is None
    # 降水：判别族结构性缺席时雨量族可单独撑起
    p_amt = {"amt_skill": 0.5}
    assert ev.precip_score(
        p_amt, absent_families={"晴雨判别/雨强分档"}) is not None
    assert ev.precip_score(p_amt) is None


# ---------------------------------------------------------------- B5 / T16
def test_trailing_reference_ignores_future_days():
    obs = {f"2026-09-{d:02d}T{h:02d}:00": {"temp": 20.0 + d + 0.1 * h,
                                          "rain": 0.0}
           for d in range(1, 31) for h in range(24)}
    idx = ev._build_reference_index(["s1"], {"s1": obs},
                                    mode="fixed_trailing", min_days=3)
    target = "2026-09-15T06:00"
    a = idx.hourly_ref("s1", target, "2026-09-10T06:00")
    dirty = dict(obs)
    for key in dirty:
        if key[:10] > "2026-09-15":
            dirty[key] = {"temp": -100.0, "rain": 0.0}
    idx2 = ev._build_reference_index(["s1"], {"s1": dirty},
                                     mode="fixed_trailing", min_days=3)
    assert idx2.hourly_ref("s1", target, "2026-09-10T06:00") == a


# ---------------------------------------------------------------- B2
def test_frequency_damping_covers_the_whole_discrimination_family():
    cal = {"ets": 0.18, "ord_skill": 0.18, "grade_ets": 0.18,
           "amt_skill": 0.2, "bias": 1.0}
    over = dict(cal, bias=2.2)
    for key in ("ets", "ord_skill", "grade_ets"):
        s_cal = ev._skill_value(key, cal)
        s_over = ev._skill_value(key, over)
        assert s_cal is not None and s_over is not None and s_over < s_cal
    # 雨量项不乘阻尼（已是 proper 的相对 MAE）
    assert ev._skill_value("amt_skill", cal) == ev._skill_value(
        "amt_skill", over)


# ---------------------------------------------------------------- B6/B8
def test_lead_weights_are_relative_preferences_and_ci_is_relative():
    S = np.ones((4, 16 * 2))
    W = np.full_like(S, np.sqrt(10.0))
    out = ev._apply_lead_weights(W, S, "hourly", 2, 16, 16)
    assert out[0, 0] / out[0, 7] == pytest.approx(1.0 / 0.4)
    rows = [{"score": 60, "qualified": True, "ci90": [58, 62],
             "bootstrap_center_delta": 0.1},
            {"score": 50, "qualified": True, "ci90": [48, 56],
             "bootstrap_center_delta": 0.1}]
    ev._mark_ci_quality(rows, wide_threshold=4.0)
    assert rows[0]["ci_wide"] is False
    assert rows[1]["ci_wide"] is True
    assert rows[1]["ci_wide_threshold"] == 4.0

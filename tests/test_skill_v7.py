"""skill-v7（2026-11 证据加权轮）机制断言。

v7 不改单格评分函数（继承 v6 的湿域对数比/阻尼/族门），只改**聚合层**：
  1. 证据门用聚合信度 ICC(K)（Spearman-Brown），不是单块 ICC(1)；
  2. 二值清零换成 √ρ_K 连续加权（no-skill 桶才为 0）；
  3. 综合分逐维 α×√ρ_K 加权，单维格不再被放大 1/α；
  4. 发布层证据份额不足时输出 null，不再把中性占位 50 当"平均"。

这里的断言锁行为，不锁某个快照数字。把实现改回 v6 的硬门/公共收缩，
对应断言会变红。
"""
from __future__ import annotations

import numpy as np
import pytest

from weather_eval import evaluate as ev
from weather_eval import stats


def setup_function():
    ev.apply_score_scheme("skill-v7")
    ev.apply_score_composite_params(
        alpha=0.5, sd_temp=8.77, sd_precip=3.73, rank_scale=10.0,
        tau=1.0, display_gain=1.0,
        lead_weights={"1-3": 1.0, "4-7": 0.7, "8-14": 0.4, "15-16": 0.2})
    ev.apply_score_reliability(True, 0.20, 3.5, 0.20, 0.10)


def teardown_function():
    ev.apply_score_scheme("slope-v2")


# ---------------------------------------------------------------- V7-1 聚合信度
def test_v7_spearman_brown_matches_textbook_values():
    """ρ(1)=0.10、K=13 ⇒ ICC(K)≈0.591；ρ(1)=0.08 ⇒ ≈0.531。"""
    assert stats.aggregate_reliability(0.10, 13) == pytest.approx(0.5909, abs=1e-4)
    assert stats.aggregate_reliability(0.08, 13) == pytest.approx(0.5306, abs=1e-4)
    # 单调性：ρ 或 K 上升，聚合信度不下降
    assert stats.aggregate_reliability(0.2, 13) > stats.aggregate_reliability(0.1, 13)
    assert stats.aggregate_reliability(0.1, 26) > stats.aggregate_reliability(0.1, 13)
    # 边界
    assert stats.aggregate_reliability(0.0, 13) == pytest.approx(0.0)
    assert stats.aggregate_reliability(1.0, 13) == pytest.approx(1.0)
    assert np.isnan(stats.aggregate_reliability(float("nan"), 13))


def test_v7_bucket_gate_uses_aggregate_not_single_block():
    """单块 ρ≈0.1 的桶在 v6 门（ICC(1)<0.2）退出、在 v7 必须存活并得到证据权重。

    判决性理由：排行榜的名次来自 13 个块的聚合；用单块信度判死等于用
    "一次测量"的精度回答"13 次平均"的问题。
    """
    M, K = 10, 13
    rng = np.random.default_rng(20261122)
    model_t = rng.normal(0.0, 6.0, M)
    model_p = rng.normal(0.0, 2.5, M)
    block = rng.normal(0.0, 2.0, K)
    T = np.array([[50.0 + model_t[i] + block[k] + rng.normal(0.0, 9.0)
                   for k in range(K)] for i in range(M)], dtype=float)
    P = np.array([[50.0 + model_p[i] + 0.5 * block[k] + rng.normal(0.0, 6.0)
                   for k in range(K)] for i in range(M)], dtype=float)
    mats = {"temp_score": T, "precip_score": P, "ets": (P - 50.0) / 50.0}
    out = ev._attach_rank_matrices(mats, n_blocks=K, days=1, track="hourly")
    rho1 = float(out["_rel_precip_raw"][0])
    rho_k = float(out["_rel_precip_agg"][0])
    assert rho1 < float(ev.SCORE_RELIABILITY_EXIT) < rho_k
    # v7 连续权重：sqrt(ρ_K) > 0，且该桶仍在打分
    assert float(out["_evidence_w_precip"][0]) == pytest.approx(
        np.sqrt(rho_k), abs=1e-9)
    assert float(out["_evidence_w_precip"][0]) > 0.5
    assert np.isfinite(out["precip_contrib"]).all()


def test_v7_no_skill_bucket_gets_zero_weight_and_neutral_contribution():
    """物理上未检出降水技巧的桶必须 w=0、降水贡献恰为 50（不是被门改成 50 的 47）。"""
    M, K = 10, 13
    rng = np.random.default_rng(7)
    T = rng.normal(50.0, 8.0, (M, K))
    P = rng.normal(50.0, 5.0, (M, K))
    mats = {"temp_score": T, "precip_score": P,
            "ets": np.zeros_like(P)}
    out = ev._attach_rank_matrices(mats, n_blocks=K, days=1, track="hourly")
    assert float(out["_evidence_w_precip"][0]) == 0.0
    assert np.allclose(out["precip_contrib"], 50.0)
    # 温度维仍在打分（单维存活），R = 温度贡献（保持 α 语义，不再 ×2）
    assert np.isfinite(out["rank_score"]).all()
    assert np.allclose(out["rank_score"], out["temp_contrib"])


def test_v7_no_skill_decision_unit_is_aggregate_not_cell():
    """no-skill 判据必须按"源跨块聚合 ETS"判定，不能按单格 ETS 占比。

    构造：每源 12 块里 8 块 ETS=0、4 块 ETS=0.2 ⇒ 单格 |ETS|≤0.05 占比
    66.7%（旧判据会清零），但每源聚合 ETS≈0.067 >0.05（聚合判据应保留）。
    这与 ICC(1) vs ICC(K) 是同一类决策单位错误。
    """
    M, K = 10, 12
    rng = np.random.default_rng(12)
    T = rng.normal(50.0, 8.0, (M, K))
    # P 要有稳定的源间差异，否则聚合信度=0、证据权重也=0，测不到 no-skill 口径
    P = (50.0 + np.linspace(-5.0, 5.0, M)[:, None]
         + rng.normal(0.0, 1.0, (M, K)))
    ets = np.zeros((M, K))
    ets[:, :4] = 0.2
    mats = {"temp_score": T, "precip_score": P, "ets": ets}
    ev.apply_score_scheme("skill-v7")
    out_v7 = ev._attach_rank_matrices(mats, n_blocks=K, days=1,
                                      track="hourly")
    assert not bool(out_v7["_no_skill_bucket"][0])
    assert float(out_v7["_evidence_w_precip"][0]) > 0.0
    # v6 归档口径仍用单格判据：同一矩阵应被判 no-skill（逐位回放）
    ev.apply_score_scheme("skill-v6")
    out_v6 = ev._attach_rank_matrices(mats, n_blocks=K, days=1,
                                      track="hourly")
    assert bool(out_v6["_no_skill_bucket"][0])
    ev.apply_score_scheme("skill-v7")


# ---------------------------------------------------------------- V7-2 合成式
def test_v7_composite_is_additive_and_value_weight_preserved():
    """R = t_c + p_c − 50 严格成立；两维贡献各按 α 与 (1−α) 缩放。"""
    z_t = np.array([1.0, -1.0, 0.0])
    z_p = np.array([1.0, 0.0, -1.0])
    got = stats.composite_from_z(
        z_t, z_p, alpha=0.5, rho_t=np.full(3, 0.8), rho_p=np.full(3, 0.5),
        weight_t=np.full(3, np.sqrt(0.8)), weight_p=np.full(3, np.sqrt(0.5)),
        mode="evidence")
    assert np.allclose(
        got["rank_score"],
        got["temp_contrib"] + got["precip_contrib"] - 50.0)
    # 温度贡献的斜率 = scale·α·√ρt；降水 = scale·(1−α)·√ρp
    assert float(got["temp_contrib"][0] - 50.0) == pytest.approx(
        10.0 * 0.5 * np.sqrt(0.8), abs=1e-9)
    assert float(got["precip_contrib"][0] - 50.0) == pytest.approx(
        10.0 * 0.5 * np.sqrt(0.5), abs=1e-9)


def test_v7_single_dim_no_alpha_doubling():
    """单维格必须是 α 语义下的那一半，不能像 v6 那样放大成 10·√ρ·z（≈2 倍）。"""
    z_t = np.array([1.0])
    z_p = np.array([np.nan])
    got = stats.composite_from_z(
        z_t, z_p, alpha=0.5, rho_t=np.array([0.8]), rho_p=np.array([0.5]),
        weight_t=np.array([np.sqrt(0.8)]), weight_p=np.array([0.0]),
        mode="evidence")
    assert float(got["rank_score"][0]) == pytest.approx(
        50.0 + 10.0 * 0.5 * np.sqrt(0.8), abs=1e-9)
    assert float(got["precip_contrib"][0]) == pytest.approx(50.0)
    assert bool(got["single_dim"][0])


def test_v7_evidence_weight_is_monotone_in_reliability():
    """同一 z 下，ρ_K 越高的桶贡献越接近真实偏差（不会出现高 ρ 反而更小）。"""
    z = np.array([1.0, 1.0, 1.0])
    got = stats.composite_from_z(
        z, np.zeros(3), alpha=0.5,
        rho_t=np.array([0.2, 0.5, 0.9]), rho_p=np.zeros(3),
        weight_t=np.sqrt(np.array([0.2, 0.5, 0.9])),
        weight_p=np.zeros(3), mode="evidence")
    dev = got["temp_contrib"] - 50.0
    assert dev[0] < dev[1] < dev[2]


# ---------------------------------------------------------------- V7-3 口径家族
def test_v7_family_inherits_v6_mechanisms():
    assert "skill-v7" in ev.SKILL_SCHEMES
    assert ev.is_v7_family("skill-v7")
    assert ev.is_v6_family("skill-v7")   # v6 的单格机制必须继续生效
    assert ev.is_v5_family("skill-v7")
    ev.apply_score_scheme("skill-v6")
    assert not ev.is_v7_family()
    with pytest.raises(ValueError):
        ev.apply_score_scheme("skill-v9")


def test_v7_v6_path_is_bitwise_preserved():
    """v6 归档口径不得被 v7 改动：同一输入下 common_shrink 结果与旧式一致。"""
    rng = np.random.default_rng(3)
    T = rng.normal(50.0, 8.0, (8, 12))
    P = rng.normal(45.0, 4.0, (8, 12))
    a = stats.rank_from_scores(T, P, alpha=0.5, rho_t=0.4, rho_p=0.25,
                               gate_t=True, gate_p=True)
    b = stats.rank_from_scores(T, P, alpha=0.5, rho_t=0.4, rho_p=0.25,
                               gate_t=True, gate_p=True, mode="common_shrink")
    assert np.allclose(a["rank_score"], b["rank_score"], equal_nan=True)
    # v6 的单维分支仍是 10·√ρ·z（v7 才改）
    one = stats.composite_from_z(
        np.array([1.0]), np.array([np.nan]), alpha=0.5,
        rho_t=np.array([0.25]), rho_p=np.array([0.25]))
    assert float(one["rank_score"][0]) == pytest.approx(50.0 + 10.0 * 0.5)


# ---------------------------------------------------------------- V7-4 披露
def test_v7_rank_diagnostics_publishes_both_reliability_scales():
    """页面/脚本必须能同时读到 ICC(1) 与 ICC(K)，不能只给一个数让读者猜。"""
    M, K = 8, 13
    rng = np.random.default_rng(11)
    T = rng.normal(50.0, 8.0, (M, K))
    P = rng.normal(50.0, 5.0, (M, K))
    mats = {"temp_score": T, "precip_score": P,
            "ets": (P - 50.0) / 50.0}
    out = ev._attach_rank_matrices(mats, n_blocks=K, days=1, track="hourly")
    diag = ev._rank_diagnostics(out, K, 1, "hourly")
    b = diag["buckets"][0]
    for key in ("rho_temp", "rho_precip", "rho_precip_icc1",
                "rho_precip_aggregate", "n_blocks_precip",
                "evidence_weight_precip", "gate_precip", "no_skill_precip"):
        assert key in b
    assert b["n_blocks_precip"] == K
    assert b["rho_precip_aggregate"] >= b["rho_precip_icc1"]
    assert diag["gate_counts"]["gate_basis"] == "aggregate_icc_k"

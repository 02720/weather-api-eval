"""第二轮全指标入分（2026-10-06）的新增维度：口径与可加性守卫。

两块新信息入分（见 docs/metric_inclusion_v2.md）：
  * 降水「雨强分辨力」`grade_ets` —— 中雨/大雨档 ETS 按事件数加权；
  * 温度「站间一致性」`mbe_bdisp` —— 各站系统偏差的样本量加权离散度。

每个维度各有一组**会咬人**的口径坑，这里逐个钉上钉子：
  1. 分档二值化必须与权威路径 `graded.graded_counts` 同一把尺子（含 np.round
     的 1 位舍入——Python 内置 round 在二进制边界上与它不一致）；
  2. 点估计与重采样必须逐位同值（bootstrap 换尺子是本项目反复堵的洞）；
  3. 站间离散度必须抓得住"池化偏差为零但各站方向相反"的抵消陷阱，且在
     bootstrap 的"先逐日聚合、再重采样加权"路径下与直接全量计算同值。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from weather_eval import stats
from weather_eval.evaluate import (PRECIP_SCORE_PARTS, TEMP_SCORE_PARTS,
                                   precip_score, temp_score)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- 分级计数口径
def test_grade_counts_match_cyeva():
    """评分轨的分档计数与权威路径 graded.graded_counts 逐位一致（整数零容差）。

    覆盖两类陷阱样本：.05 类二进制舍入边界（np.round 与 Python round 在这里
    分家）、以及正好落在档位端点上的值（2.0 / 4.9 / 5.0 / 10.0 / 25.0）。
    """
    from weather_eval.graded import graded_counts
    obs = np.array([0.0, 0.04, 0.06, 0.14, 1.9, 1.95, 2.0, 4.9, 4.95, 5.0,
                    9.9, 9.95, 10.0, 24.9, 25.0, 60.0])
    fcst = np.array([2.0, 0.14, 0.04, 5.0, 2.0, 4.9, 1.9, 4.95, 2.1, 9.9,
                     5.0, 10.0, 9.9, 25.0, 24.95, 0.0])
    for kind, levs in (("1h", ("2", "3")), ("24h", ("+2", "+3"))):
        want = graded_counts(obs, fcst, kind, levs)
        bounds = stats.grade_bounds(kind)
        got = stats.grade_counts(obs, fcst, bounds)
        for j, lev in enumerate(levs):
            h_cy, mi_cy, fa_cy, _c, total_cy = want[lev]
            h, fa, mi = got[j]
            assert (h, fa, mi) == (h_cy, fa_cy, mi_cy), (
                f"{kind}/{lev}: 权威 (h,fa,mi)=({h_cy},{fa_cy},{mi_cy}) "
                f"评分轨=({h},{fa},{mi})")
            assert h + fa + mi <= total_cy


def test_grade_counts_ignore_nan_pairs():
    """缺测对不进计数（与晴雨维同一个 NaN 掩膜），且逐条与手算一致。"""
    obs = np.array([3.0, np.nan, 6.0, 0.0])
    fcst = np.array([3.0, 3.0, np.nan, 6.0])
    (h1, fa1, mi1), (h2, fa2, mi2) = stats.grade_counts(
        obs, fcst, stats.grade_bounds("1h"))
    # 缺测对不产生任何计数：有效对只剩 (3,3) 与 (0,6)（(6,nan) 整对剔除）。
    # 2 档 = [2.0,4.9]：仅 (3,3) 命中；3 档 = [5.0,9.9]：仅 (0,6) 空报。
    assert (h1, fa1, mi1) == (1, 0, 0)
    assert (h2, fa2, mi2) == (0, 1, 0)


# --------------------------------------------------- 点估计 ↔ 重采样同尺子
def test_grade_ets_point_matches_aggregate_path():
    """点估计 grade_ets 与 bootstrap 聚合路径必须**逐位**同值。

    构造一份样本，分别走：
      a) `precip_binary_metrics`（点估计，逐记录计数）；
      b) `build_day_stat_tables` 聚合 → `_rain_scores_from_aggregate`（重采样口径）。
    两者共享同一个 `grade_ets_from_counts`，但计数入口不同——任何一条入口自己
    二值化（而不是复用同一个函数）都会在这里红。
    """
    rng = np.random.default_rng(20261006)
    obs = np.round(rng.gamma(0.6, 4.0, 400), 2)
    fcst = np.round(np.clip(obs + rng.normal(0, 3.0, 400), 0, None), 2)
    gbounds = stats.grade_bounds("24h")

    n = int((~np.isnan(obs) & ~np.isnan(fcst)).sum())
    counts = stats.grade_counts(obs, fcst, gbounds)
    want = stats.grade_ets_point(counts, n, 5)

    days = [f"2026-10-{d:02d}" for d in range(1, 21)]
    daily = [{"model": "m", "station": "s", "valid_day": days[i % 20],
              "offset": (i % 20) + 1, "rain_obs": float(obs[i]),
              "rain_fcst": float(fcst[i]), "temp_max_obs": None,
              "temp_max_fcst": None, "temp_min_obs": None,
              "temp_min_fcst": None} for i in range(400)]
    hourly = []
    _days, tables = stats.build_day_stat_tables(hourly, daily, ["m"], 1, 20,
                                                1.0, 1.0)
    A = tables["rain_daily"]
    # 整表（不重采样）过一次生产评分路径。它内部对整表计数做向量化
    # grade_ets_from_counts，与下面逐桶切片同值，故不在这里重复调用；能从复合分
    # 里反推的只有分数本身——"表能被评分路径消费出有限分"必须钉住，否则下面的
    # 计数层对拍可能是在一张空表上空跑。
    got = stats._rain_scores_from_aggregate(A, PRECIP_SCORE_PARTS, 5)
    assert np.isfinite(got).all(), "聚合表必须能被评分路径消费出有限分"
    tot = A.sum(axis=(0, 2, 3))          # (bucket, 13)：整表计数汇总
    # 点估计的计数按 offset 分桶后应与聚合表逐位相同
    want_by_bucket = {}
    for i in range(400):
        b = i % 20
        want_by_bucket.setdefault(b, []).append((obs[i], fcst[i]))
    for b in range(20):
        oo = np.array([p[0] for p in want_by_bucket[b]])
        ff = np.array([p[1] for p in want_by_bucket[b]])
        cnt = stats.grade_counts(oo, ff, gbounds)
        nn = len(oo)
        single = stats.grade_ets_point(cnt, nn, 5)
        from_agg = stats.grade_ets_from_counts(
            tot[b:b + 1], np.float64(tot[b, 0] + tot[b, 1] + tot[b, 2] + tot[b, 3]), 5)
        got_scalar = float(np.asarray(from_agg).reshape(()))
        if single is None:
            assert not math.isfinite(got_scalar)
        else:
            assert abs(single - got_scalar) < 1e-12, (
                f"桶 {b}: 点估计 {single} vs 聚合 {got_scalar}")
    assert want is not None  # 这批样本必有至少一档达标


def test_grade_ets_weighting_and_missing_discipline():
    """加权与缺项纪律：稀疏档少说话；两档全缺 → NaN（缺项归一）而非 0 分。"""
    # 两档都"满员"：ETS 各 0.5，事件数 300 vs 30 → 加权均值应偏向大档
    tot = np.zeros(13)
    tot[7], tot[8], tot[9] = 150, 0, 150          # 档1: h=150, fa=0, mi=150 → ev=300
    tot[10], tot[11], tot[12] = 3, 0, 3           # 档2: h=3, fa=0, mi=3 → ev=6
    n = 2000.0
    v = float(stats.grade_ets_from_counts(tot, np.float64(n), 5))
    # 档1 ETS = (150 − href)/(300 − href)，href = (h+mi)(h+fa)/n = 300×150/2000
    # = 22.5 → 127.5/277.5 ≈ 0.4595；档2 href = 6×3/2000 = 0.009 → ≈ 0.4993。
    # 加权 (300×0.4595 + 6×0.4993)/306 ≈ 0.4602 —— 显著偏向档1（等权会是 0.479）
    assert 0.45 < v < 0.47
    # 档2 事件数低于门槛（min_sample=5 → 6 仍达标；改 min_sample=10 则缺项）。
    # 档2 剔除后只剩档1：grade_ets 应严格等于档1 自己的 ETS（权重自然归一），
    # 而不是"两档加权后再减档2"的任何混合值。
    v10 = float(stats.grade_ets_from_counts(tot, np.float64(n), 10))
    assert v10 == pytest.approx(127.5 / 277.5, rel=1e-12)
    assert v10 < v  # 稀疏档权重被移除后，加权均值回到大档自己的水平
    tot2 = np.zeros(13)
    tot2[7], tot2[8], tot2[9] = 0, 0, 0
    tot2[10], tot2[11], tot2[12] = 3, 0, 3        # 仅档2 且事件数 6 < 门槛 10
    v_nan = stats.grade_ets_from_counts(tot2, np.float64(n), 10)
    assert not math.isfinite(float(v_nan)), "两档全缺必须 NaN（缺项归一），不得记 0 分"


# ------------------------------------------------------------ 站间一致性
def test_between_station_mbe_sd_catches_cancellation():
    """抵消陷阱：池化 MBE = 0 但各站方向相反 → 离散度必须显著大于 0。

    这是 |mbe| 完全看不见、而 mbe_bdisp 专门要抓的情形（实测两者跨源相关 0.115）。
    """
    n = np.full(4, 100.0)
    se_cancel = np.array([200.0, 200.0, -200.0, -200.0])   # 各站 MBE ±2，池化 = 0
    se_same = np.full(4, 200.0)                            # 各站一致 +2，池化 = +2
    d_cancel = float(stats.between_station_mbe_sd(n, se_cancel))
    d_same = float(stats.between_station_mbe_sd(n, se_same))
    assert d_cancel == pytest.approx(2.0)
    assert d_same == pytest.approx(0.0)
    assert d_cancel > d_same


def test_between_station_mbe_sd_matches_full_sample_computation():
    """可加性等价：先逐日聚合再归约 == 对全量样本直接算（bootstrap 同尺子）。"""
    rng = np.random.default_rng(7)
    n_days, n_st = 30, 4
    obs = rng.normal(20, 3, (n_days, n_st))
    bias = np.array([1.5, -1.2, 0.3, 0.8])            # 各站系统偏差各不相同
    fcst = obs + bias[None, :] + rng.normal(0, 0.5, (n_days, n_st))

    # 直接全量：每站的 (n, Σe)
    n_s = np.full(n_st, float(n_days))
    se_s = (fcst - obs).sum(axis=0)
    direct = float(stats.between_station_mbe_sd(n_s, se_s))

    # bootstrap 口径：逐日行进表 → 按天权重求和 → 归约
    def row(o, f):
        return [1.0, (f - o) ** 2, abs(f - o), f - o] + [0.0] * 7
    A = np.zeros((1, 1, n_st, n_days, len(stats._TEMP_STATS)))
    for d in range(n_days):
        for s in range(n_st):
            A[0, 0, s, d] = row(obs[d, s], fcst[d, s])
    W = np.ones((1, n_days))
    agg = stats.aggregate_day_stats(W, A)             # (1, 1, 1, n_st, 11)
    via_boot = float(stats.between_station_mbe_sd(agg[0, 0, 0, :, 0],
                                                  agg[0, 0, 0, :, 3]))
    assert via_boot == pytest.approx(direct, rel=1e-12)


def test_between_station_mbe_sd_requires_three_qualified_stations():
    """站数门槛：达标站 <3 → NaN（缺项归一），两站距离撑不起一致性命题。"""
    n = np.array([[40.0, 40.0, 40.0, 40.0],
                  [40.0, 40.0, 0.0, 0.0]])
    se = np.array([[100.0, -100.0, 50.0, -50.0],
                   [100.0, -100.0, 0.0, 0.0]])
    out = stats.between_station_mbe_sd(n, se)
    assert math.isfinite(float(out[0]))
    assert not math.isfinite(float(out[1]))


# ------------------------------------------------------------ 评分表契约
def test_new_dims_move_scores_in_the_right_direction():
    """新维度在不在会改变分数，且方向正确；缺项（None）按剩余权重归一。"""
    t = {"acc2": 80, "acc1": 55, "rmse": 2.0, "mae": 1.6, "r": 0.9,
         "mbe": 0.5, "slope": 1.1}
    assert temp_score(dict(t, mbe_bdisp=0.9)) < temp_score(dict(t, mbe_bdisp=0.1))
    assert temp_score(dict(t, mbe_bdisp=None)) == temp_score(t)

    p = {"ets": 0.2, "pod": 50.0, "far": 60.0, "bias": 1.5,
         "amt_mae": 1.0, "amt_bias": 1.0}
    assert precip_score(dict(p, grade_ets=0.4)) > precip_score(dict(p, grade_ets=0.0))
    assert precip_score(dict(p, grade_ets=None)) == precip_score(p)


def test_weights_still_sum_to_one_and_descend():
    """权重和恒为 1；PRECIP 表保持权重降序（ETS 首位的设计意图）。"""
    for parts in (TEMP_SCORE_PARTS, PRECIP_SCORE_PARTS):
        assert abs(sum(w for _k, w, *_ in parts) - 1.0) < 1e-9
    ws = [w for _k, w, *_ in PRECIP_SCORE_PARTS]
    assert ws == sorted(ws, reverse=True)

"""skill-v3 性质测试：50=无技巧、100=完美、下半段不硬触底、ETS 必须是降水锚点。"""
from __future__ import annotations

import math

import numpy as np

from weather_eval import evaluate as ev
from weather_eval import stats as st


def setup_function():
    ev.apply_score_scheme("skill-v3")
    ev.apply_score_skill_k(1.535)


def teardown_function():
    ev.apply_score_scheme("slope-v2")


def test_temp_no_skill_is_50_and_perfect_is_100():
    no_skill = {
        "rmse": 2.0, "mae": 1.5, "acc1": 40.0, "acc2": 70.0, "r_anom": 0.0,
        "ref": {"available": True, "rmse_ref": 2.0, "mae_ref": 1.5,
                "acc1_ref": 40.0, "acc2_ref": 70.0},
    }
    assert ev.temp_score(no_skill) == 50.0
    perfect = dict(no_skill, rmse=0.0, mae=0.0, acc1=100.0, acc2=100.0,
                   r_anom=1.0)
    assert ev.temp_score(perfect) == 100.0


def test_temp_below_reference_is_compressed_not_fake_zero():
    bad = {
        "rmse": 4.0, "mae": 3.0, "acc1": 0.0, "acc2": 0.0, "r_anom": -1.0,
        "ref": {"available": True, "rmse_ref": 2.0, "mae_ref": 1.5,
                "acc1_ref": 40.0, "acc2_ref": 70.0},
    }
    raw = ev.temp_score(bad)
    shown = ev.display_dim_score(raw)
    assert raw is not None and raw <= 0
    # 2 倍参照误差 + 全错 + 反相：原始线性指数触底，但展示分仍 >0
    assert shown is not None and 0 < shown < 50


def test_precip_no_skill_is_50_and_ets_is_required():
    assert ev.precip_score({"ets": 0.0, "amt_skill": 0.0, "grade_ets": 0.0}) == 50.0
    # ETS 缺失（无雨事件）→ 整维缺测，不允许 amount-only 出分
    assert ev.precip_score({"amt_skill": 1.0, "grade_ets": 1.0}) is None


def test_precip_perfect_is_100():
    assert ev.precip_score({"ets": 1.0, "amt_skill": 1.0, "grade_ets": 1.0}) == 100.0


def test_amt_skill_is_log_bounded_and_monotone():
    # MAE=参照 → 0；MAE=0 → 1；更差 → 负值，但不会和无界比值一样爆炸
    base = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
    fc = np.array([1.0, 2.0, 3.0, 0.0, 0.0])
    out = ev.precip_amount_metrics(base, fc, 5)
    assert out["amt_skill"] == 1.0
    fc_bad = np.array([10.0, 20.0, 30.0, 100.0, 200.0])
    out_bad = ev.precip_amount_metrics(base, fc_bad, 5)
    assert out_bad["amt_skill"] < 0
    assert math.isfinite(out_bad["amt_skill"])


def test_reference_index_is_past_only_and_persistence_is_separate():
    obs = {}
    for d in range(1, 10):
        for h in (0, 12):
            iso = f"2026-09-{d:02d}T{h:02d}:00"
            obs[iso] = {"temp": 20.0 + d, "rain": 0.0}
    idx = ev._build_reference_index(["s1"], {"s1": obs},
                                    window_days=5, min_days=3)
    assert idx.hourly_ref("s1", "2026-09-01T00:00") is None
    # 第 4 天：前 3 天可用 → 气候参照 = 21,22,23 的均值 22
    assert abs(idx.hourly_ref("s1", "2026-09-04T00:00") - 22.0) < 1e-9
    # 持续性 = 前一天同小时
    assert abs(idx.hourly_persist("s1", "2026-09-04T00:00") - 23.0) < 1e-9
    # 参照不得偷看目标日之后：第 4 天的参照不得等于 24
    assert idx.hourly_ref("s1", "2026-09-04T00:00") != 24.0


def test_ref_stats_falls_back_to_persistence_never_bucket_mean():
    obs = np.array([20.0])
    ref = np.array([np.nan])
    persist = np.array([20.0])
    st = ev._ref_stats(obs, ref, persist, [1, 2], min_sample=1)
    assert st["available"] is True
    assert st["ref_type"] == "persistence"
    assert st["rmse_ref"] == 0.0


def test_display_mapping_is_monotone_and_never_fake_zero():
    raw = np.array([-100.0, -50.0, 0.0, 40.0, 49.9, 50.0, 60.0, 100.0])
    shown = ev.skill_display(raw)
    assert np.all(np.diff(shown) >= 0)
    # 指数压缩是渐近的：极端负值趋近 0，但不硬触底
    assert 0.0 <= shown[0] < 1.0
    assert shown[-1] == 100.0
    assert all(shown[i] > 0 for i in range(1, len(raw)))


# ------------------------------------------------------------------
# skill-v4：有界化 + 锚点安全的雨量公式 + 校准扣分项
# ------------------------------------------------------------------

def _v4_temp(**kw):
    base = {
        "rmse": 2.0, "mae": 1.5, "acc1": 40.0, "acc2": 70.0, "r_anom": 0.0,
        "mbe": 0.3, "slope": 0.8,
        "ref": {"available": True, "rmse_ref": 2.0, "mae_ref": 1.5,
                "acc1_ref": 40.0, "acc2_ref": 70.0,
                "mbe_ref": 0.3, "slope_ref": 0.8},
    }
    base.update(kw)
    return base


def test_v4_temp_anchors_and_penalty_cap():
    ev.apply_score_scheme("skill-v4")
    assert ev.temp_score(_v4_temp()) == 50.0
    perfect = _v4_temp(rmse=0.0, mae=0.0, acc1=100.0, acc2=100.0,
                       r_anom=1.0, mbe=0.0, slope=1.0)
    assert ev.temp_score(perfect) == 100.0
    bad = _v4_temp(rmse=6.0, mae=4.0, acc1=0.0, acc2=0.0, r_anom=-1.0,
                   mbe=3.0, slope=0.1)
    raw = ev.temp_score(bad)
    assert raw is not None
    # 正向技能全 −1 + 扣分项全 1：raw = 50 + 50·(−1 − 0.15) = −7.5
    assert raw >= -7.5 - 1e-9


def test_v4_precip_zero_rain_is_below_50_but_random_is_50():
    ev.apply_score_scheme("skill-v4")
    assert ev.precip_score({"ets": 0.0, "amt_skill": 0.0,
                            "grade_ets": 0.0, "bias": 1.0}) == 50.0
    assert ev.precip_score({"ets": 0.0, "amt_skill": 0.0,
                            "grade_ets": 0.0, "bias": 0.0}) == 45.0
    assert ev.precip_score({"ets": 1.0, "amt_skill": 1.0,
                            "grade_ets": 1.0, "bias": 1.0}) == 100.0


def test_v4_amt_skill_anchor_and_dry_gradient():
    # 干格：完美上限 <1，但小绝对误差不再一刀切 −1
    obs = np.array([0.001] * 100)
    fc_perfect = obs.copy()
    fc_tiny = np.array([0.01] * 100)
    fc_huge = np.array([1.0] * 100)
    p = ev.precip_amount_metrics(obs, fc_perfect, 5, amt_floor=0.05)["amt_skill"]
    tiny = ev.precip_amount_metrics(obs, fc_tiny, 5, amt_floor=0.05)["amt_skill"]
    huge = ev.precip_amount_metrics(obs, fc_huge, 5, amt_floor=0.05)["amt_skill"]
    ref = ev.precip_amount_metrics(obs, np.zeros_like(obs), 5,
                                   amt_floor=0.05)["amt_skill"]
    assert 0.0 < p < 1.0          # 干格没有足够可改进空间
    assert -1.0 < tiny < 0.0      # 小绝对误差不是最大惩罚
    assert huge == -1.0           # 大误差封底
    assert ref == 0.0             # “永远报零雨”仍是 0 锚点


def test_v4_rmse_mae_skill_is_bounded():
    s = ev._skill_value("rmse", {"rmse": 100.0, "ref": {"rmse_ref": 0.1}})
    assert s == -1.0
    s = ev._skill_value("amt_skill", {"amt_skill": -999.0})
    assert s == -1.0


def test_v4_temp_single_axis_is_missing():
    ev.apply_score_scheme("skill-v4")
    only_r = {"r_anom": 0.5, "ref": {"r_anom": 0.0}}
    assert ev.temp_score(only_r) is None


def test_ref_stats_mask_and_components():
    obs = np.array([1.0, 2.0, 3.0])
    ref = np.array([1.0, 2.0, 3.0])
    persist = np.array([1.0, 2.0, 3.0])
    rs = ev._ref_stats(obs, ref, persist, [1, 2], min_sample=1,
                       mask=np.array([True, True, False]))
    assert rs["n"] == 2
    assert rs["rmse_ref"] == 0.0
    assert rs["mbe_ref"] == 0.0
    assert rs["slope_ref"] == 1.0


def test_ordinal_skill_uses_all_thresholds_and_has_anchor():
    obs = np.array([0, 0, 1, 2, 5, 10, 25, 50.0])
    assert st.ordinal_precip_skill(obs, obs.copy(), "24h", 5) == 1.0
    shifted = np.array([0, 0, 0, 1, 2, 5, 10, 25.0])
    s = st.ordinal_precip_skill(obs, shifted, "24h", 5)
    assert s is not None and 0.0 < s < 1.0
    # 整格没有可评降水 → 无定义，不是 0 分
    assert st.ordinal_precip_skill(np.zeros(10), np.zeros(10), "1h", 5) is None

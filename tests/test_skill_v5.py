"""skill-v5 性质测试：起报锚定参照、湿样本雨量、对称命中、软压缩、双综合分。

纪律同 test_first_principles_fixes：每条都写成"把实现改回去就变红"，
盯的是 2026-11 方案里被独立复算证实的失败模式，而不是功能表面。
"""
from __future__ import annotations

import math

import numpy as np

from weather_eval import evaluate as ev


def setup_function():
    ev.apply_score_scheme("skill-v5")
    ev.apply_score_composite_params(
        alpha=0.5, sd_temp=8.77, sd_precip=3.73, rank_scale=10.0,
        tau=0.9, display_gain=1.0,
        lead_weights={"1-3": 1.0, "4-7": 0.7, "8-14": 0.4, "15-16": 0.2})


def teardown_function():
    ev.apply_score_scheme("slope-v2")


def _obs_trend(days=12, hours=24, start_day=1):
    """逐日 +1°C 的线性升温观测；每天 24 小时覆盖，供日极值使用。"""
    out = {}
    for d in range(start_day, start_day + days):
        for h in range(hours):
            out[f"2026-09-{d:02d}T{h:02d}:00"] = {
                "temp": 20.0 + d + 0.1 * h, "rain": 0.0}
    return out


# --------------------------------------------------------------- P1-1 参照集

def test_reference_is_anchored_at_issue_and_ignores_future():
    """起报日之后的观测不得进入参照窗口；窗口内线性趋势外推到目标时刻。

    旧实现锚在目标日（target−1…target−W），长时效参照因此"看到了未来"。
    这里把起报后的观测改成 −100，参照必须一字不变。
    """
    obs = _obs_trend(days=12)
    idx = ev._build_reference_index(["s1"], {"s1": obs}, window_days=5,
                                    min_days=3, run_anchored=True)
    issue = "2026-09-10T06:00"
    target = "2026-09-12T06:00"
    ref = idx.hourly_ref("s1", target, issue)
    assert ref is not None
    # 窗口是 9/5~9/9 的同一小时，线性外推应继续升温而不是回落到目标日均值
    assert ref > 29.0
    # 把 9/10 之后（含目标日）的观测全部改成 −100：参照不得改变
    obs2 = dict(obs)
    for key in obs2:
        if key[:10] >= "2026-09-10":
            obs2[key] = {"temp": -100.0, "rain": 0.0}
    idx2 = ev._build_reference_index(["s1"], {"s1": obs2}, window_days=5,
                                     min_days=3, run_anchored=True)
    assert idx2.hourly_ref("s1", target, issue) == ref
    # 持续性对照同样锚在起报前：取 9/9 的同一小时
    persist = idx.hourly_persist("s1", target, issue)
    assert persist is not None and abs(persist - (29.0 + 0.6)) < 1e-9


def test_reference_daily_is_also_issue_anchored():
    obs = _obs_trend(days=12)
    idx = ev._build_reference_index(["s1"], {"s1": obs}, window_days=5,
                                    min_days=3, run_anchored=True)
    issue = "2026-09-10T06:00"
    got = idx.daily_ref("s1", "2026-09-12", "max", issue)
    assert got is not None and got > 30.0
    obs2 = dict(obs)
    for key in obs2:
        if key[:10] >= "2026-09-10":
            obs2[key] = {"temp": -100.0, "rain": 0.0}
    idx2 = ev._build_reference_index(["s1"], {"s1": obs2}, window_days=5,
                                     min_days=3, run_anchored=True)
    assert idx2.daily_ref("s1", "2026-09-12", "max", issue) == got


def test_reference_v4_keeps_target_anchored_replay():
    """v4 归档回放必须保持旧语义（target 前 W 天），不能被 v5 静默改写。"""
    obs = _obs_trend(days=12)
    idx = ev._build_reference_index(["s1"], {"s1": obs}, window_days=5,
                                    min_days=3, run_anchored=False)
    # 旧口径：9/12 的参照 = 9/7~9/11 同小时均值 = 29.6
    assert abs(idx.hourly_ref("s1", "2026-09-12T06:00") - (29.0 + 0.6)) < 1e-9


# --------------------------------------------------------------- P1-2 雨量

def test_wet_amount_skill_is_scale_invariant_and_zero_rain_anchor():
    """同一相对误差在不同量级/干湿桶给同一分；永远报零雨仍是 0。"""
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    fc_bad = obs * 1.5
    o1 = ev.precip_amount_metrics(obs, fc_bad, 5, amt_floor=0.05,
                                  wet_threshold=1.0)
    o2 = ev.precip_amount_metrics(obs * 20.0, fc_bad * 20.0, 5,
                                  amt_floor=0.05, wet_threshold=1.0)
    assert o1["amt_skill"] is not None and o2["amt_skill"] is not None
    assert abs(o1["amt_skill"] - 0.5) < 1e-9       # (3−1.5)/3
    assert abs(o1["amt_skill"] - o2["amt_skill"]) < 1e-9
    # 参照锚：永远报零雨 = 0；完美 = 1
    zero = ev.precip_amount_metrics(obs, np.zeros_like(obs), 5,
                                    amt_floor=0.05, wet_threshold=1.0)
    perfect = ev.precip_amount_metrics(obs, obs.copy(), 5,
                                       amt_floor=0.05, wet_threshold=1.0)
    assert zero["amt_skill"] == 0.0
    assert perfect["amt_skill"] == 1.0


def test_wet_subset_excludes_dry_samples():
    """判别与量级不交叉污染：干样本再多也不该改变"有雨时下了多少"的分。"""
    obs = np.array([0.0] * 50 + [2.0, 4.0, 6.0, 8.0, 10.0])
    fc = np.array([0.0] * 50 + [3.0, 6.0, 9.0, 12.0, 15.0])
    out = ev.precip_amount_metrics(obs, fc, 5, amt_floor=0.05,
                                   wet_threshold=1.0)
    assert out["amt_n_wet"] == 5
    assert abs(out["amt_skill"] - 0.5) < 1e-9


# --------------------------------------------------------------- P1-3/4/5

def test_acc_skill_is_symmetric_around_reference():
    base = {"acc2": 83.0, "ref": {"acc2_ref": 83.0}}
    # 先关掉软压缩，单测"换算本身"的对称性：旧口径两个分母
    # 1−a_ref=0.17 vs a_ref=0.83（4.9 倍）；新口径同分母 ⇒ 两侧等幅。
    ev.apply_score_composite_params(tau=0.0)
    pos = ev._skill_value("acc2", dict(base, acc2=90.0))
    neg = ev._skill_value("acc2", dict(base, acc2=76.0))
    assert pos is not None and neg is not None
    # ar=0.83、偏差 ±7 个百分点：旧口径比值 4.9；Möbius 把两侧拉到同一量级
    # （0.72，有限偏差仍受分母的二阶项影响，但不再是未声明的价值判断）。
    assert 0.5 < (-neg) / pos < 1.0
    # 还原软压缩后仍保留这一性质，且下半段被有意再压缩
    ev.apply_score_composite_params(tau=0.9)
    pos = ev._skill_value("acc2", dict(base, acc2=90.0))
    neg = ev._skill_value("acc2", dict(base, acc2=76.0))
    assert 0.5 < (-neg) / pos < 1.0


def test_bias_penalty_is_removed_from_precip_score():
    """BIAS 是列联表恒等式，v5 不再重复计罚。"""
    p = {"ets": 0.0, "amt_skill": 0.0, "ord_skill": 0.0, "grade_ets": 0.0,
         "bias": 0.0}
    assert ev.precip_score(p) == 50.0
    assert ev.precip_score(dict(p, bias=10.0)) == 50.0


def test_soft_skill_keeps_anchors_and_softens_floor():
    s0 = ev._soft_skill(0.0)
    s1 = ev._soft_skill(1.0)
    sm1 = ev._soft_skill(-1.0)
    deep = ev._soft_skill(-10.0)
    assert s0 == 0.0 and s1 == 1.0
    assert -1.0 < sm1 < -0.5          # 不再在 −1 处硬触底
    assert -1.0 < deep < sm1          # 更深误差仍可分
    vals = [ev._soft_skill(x) for x in (-10, -2, -1, -0.5, 0, 0.5, 1)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_perfect_and_no_skill_anchors_survive_soft_compression():
    no_skill = {
        "rmse": 2.0, "mae": 1.5, "acc1": 40.0, "acc2": 70.0, "r_anom": 0.0,
        "mbe": 0.3, "slope": 0.8,
        "ref": {"available": True, "rmse_ref": 2.0, "mae_ref": 1.5,
                "acc1_ref": 40.0, "acc2_ref": 70.0,
                "mbe_ref": 0.3, "slope_ref": 0.8},
    }
    assert ev.temp_score(no_skill) == 50.0
    perfect = dict(no_skill, rmse=0.0, mae=0.0, acc1=100.0, acc2=100.0,
                   r_anom=1.0, mbe=0.0, slope=1.0,
                   ref=dict(no_skill["ref"], acc1_ref=40.0, acc2_ref=70.0))
    assert ev.temp_score(perfect) == 100.0


# --------------------------------------------------------------- P2 综合分

def test_skill_composite_uses_declared_alpha_and_missing_is_missing():
    assert ev._composite_score(80.0, 40.0) == 60.0
    assert ev._composite_score(None, 40.0) is None
    assert ev._composite_score(80.0, None) is None
    ev.apply_score_composite_params(alpha=0.3)
    assert abs(ev._composite_score(80.0, 40.0) - 52.0) < 1e-9
    ev.apply_score_composite_params(alpha=0.5)


def test_rank_composite_realizes_alpha_in_variance_share_and_is_additive():
    rng = np.random.default_rng(20261107)
    n = 2000
    t = rng.normal(70.0, 8.77, n)
    p = rng.normal(45.0, 3.73, n)
    rank = ev._bucket_rank_rows([(i, float(t[i]), float(p[i]))
                                 for i in range(n)])
    r = np.array([rank[i][0] for i in range(n)])
    tc = np.array([rank[i][1] for i in range(n)])
    pc = np.array([rank[i][2] for i in range(n)])
    # 可加恒等式：R = t_contrib + p_contrib − 50
    assert np.allclose(r, tc + pc - 50.0, atol=0.02)
    # α=0.5 时两维对名次方差的贡献接近 1:1（旧线性平均是 0.75~0.85:0.12~0.16）
    ratio = float(np.var(tc) / np.var(pc))
    assert 0.9 < ratio < 1.1


def test_rank_and_skill_are_two_different_objects():
    """同一对 T/P：S 是绝对技能，R 是桶内名次；缺一维时两者都判缺。"""
    a = ev._rank_contributions(80.0, 40.0, 70.0, 45.0)
    b = ev._rank_contributions(80.0, 40.0, 70.0, 45.0)
    assert a == b
    assert a[0] is not None
    assert ev._rank_contributions(None, 40.0, 70.0, 45.0) == (None, None, None)
    assert ev._composite_score(80.0, 40.0) == 60.0
    # R 以本档均值为 50：把两维都抬到均值之上，R 必须高于 50
    above = ev._rank_contributions(75.0, 48.0, 70.0, 45.0)[0]
    assert above is not None and above > 50.0


def test_rank_needs_at_least_two_models():
    got = ev._bucket_rank_rows([("only", 80.0, 40.0)])
    assert got["only"] == (None, None, None)
    got = ev._bucket_rank_rows([("a", 80.0, 40.0), ("b", None, 40.0)])
    assert got["a"] == (None, None, None)


def test_display_is_frozen_affine_and_additive():
    for t, p in ((80.0, 40.0), (55.0, 48.0), (30.0, 70.0)):
        c = ev._composite_score(t, p)
        assert abs(ev.skill_display(c)
                   - (0.5 * ev.skill_display(t)
                      + 0.5 * ev.skill_display(p))) < 0.01
    # 50 分处连续、单调
    vals = [ev.skill_display(x) for x in (40.0, 49.0, 50.0, 51.0, 60.0)]
    assert vals == sorted(vals)


def test_lead_weights_are_explicit_and_configurable():
    cols = ev._lead_weight_columns("all", K=1, H=16, D=16)
    assert cols[0] == 1.0          # 1d
    assert cols[2] == 1.0          # 3d
    assert cols[3] == 0.7          # 4d
    assert cols[6] == 0.7          # 7d
    assert cols[7] == 0.4          # 8d
    assert cols[10] == 0.4         # 11d
    assert cols[13] == 0.4         # 14d
    assert cols[14] == 0.2         # 15d
    assert cols[15] == 0.2         # 16d

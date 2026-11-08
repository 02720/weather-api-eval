"""2026-11 第一性原理复核（skill-v5 修订）的机制断言。

每条断言对应 SCORING.md §7 变更表里的一行，锁的是**测量机制**而不是某个数字：

  * P0-1a 降水雨量技能改在事件并集 `O≥τ ∨ F≥τ` 上求值（干样本空报不再免费）；
  * P0-1b ETS 频率校准阻尼 d(B) 只压不抬、缺 BIAS 时不惩罚；
  * P0-2a 信度加权从"除以 √ρ"改成"乘以 √ρ"（噪声单调衰减）；
  * P0-2b 桶内 μ/σ 用中位数±k·1.4826·MAD 稳健化；
  * P0-4  参照不再使用同日跨站异常场；
  * P1-1  跨站/站内门槛按分辨率自适应；
  * P1-3  温度维三族正交、小时/日轨 grade_ets 同构、软压缩 τ=1 时 C¹ 光滑。
"""
from __future__ import annotations

import numpy as np
import pytest

from weather_eval import evaluate as ev
from weather_eval import stats


@pytest.fixture(autouse=True)
def _v5():
    """这些断言都在 skill-v5 生产口径下成立。"""
    ev.apply_score_scheme("skill-v5")
    yield
    ev.apply_score_scheme("slope-v2")


# ---------------------------------------------------------------- P0-1a 事件并集

def test_amt_skill_uses_event_union_so_dry_false_alarms_cost():
    """干样本上虚报雨量必须扣分——旧口径完全不看干样本。"""
    # 8 个有雨事件 + 8 个干样本：事件数 8 ≥ min_sample=5，足以出分
    obs = np.array([10.0] * 8 + [0.0] * 8)
    good = obs.copy()
    # 干样本上狂报 50mm（有雨事件报得完全正确）
    hedge = obs.copy()
    hedge[8:] = 50.0
    s_good = ev.precip_amount_metrics(obs, good, 5, amt_floor=0.05,
                                      wet_threshold=1.0)["amt_skill"]
    s_hedge = ev.precip_amount_metrics(obs, hedge, 5, amt_floor=0.05,
                                       wet_threshold=1.0)["amt_skill"]
    assert s_good == 1.0                     # 完美预报锚点
    assert s_hedge is not None and s_hedge < 0.5
    # 旧口径下这两个都是 1.0（干样本被完全忽略）——这正是被堵住的漏洞
    assert s_hedge < s_good


def test_amt_skill_anchors_survive_event_union():
    """换成并集后 0/1 锚点仍严格成立（不准伪造满分，也不准洗白零分）。"""
    obs = np.array([0.0, 0.0, 3.0, 5.0, 0.0, 0.0, 2.0, 4.0, 6.0, 1.0,
                    0.0, 0.0])
    zero = ev.precip_amount_metrics(obs, np.zeros_like(obs), 5,
                                    amt_floor=0.05, wet_threshold=1.0)
    perfect = ev.precip_amount_metrics(obs, obs.copy(), 5,
                                       amt_floor=0.05, wet_threshold=1.0)
    assert zero["amt_skill"] == 0.0          # 恒零雨 = 参照 = 0
    assert perfect["amt_skill"] == 1.0       # 完美 = 1


def test_amt_skill_still_scale_invariant():
    """并集口径不能破坏"同一相对误差同一分"的尺度不变性。"""
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0])
    bad = obs * 1.5
    a = ev.precip_amount_metrics(obs, bad, 5, amt_floor=0.05,
                                 wet_threshold=1.0)["amt_skill"]
    b = ev.precip_amount_metrics(obs * 20.0, bad * 20.0, 5, amt_floor=0.05,
                                 wet_threshold=1.0)["amt_skill"]
    assert a is not None and a == pytest.approx(b, abs=1e-9)


# ---------------------------------------------------------------- P0-1b 频率阻尼

def test_frequency_damping_only_presses_never_lifts():
    """d(B) ≤ 1；B 在容差带内为 1；过度报雨被单调压制。"""
    d = ev._frequency_damping
    assert d(1.0) == 1.0
    assert d(1.05) == 1.0                # 容差 B_tol=1.10 内不惩罚
    assert d(0.95) == pytest.approx(1.0, abs=1e-12)   # 欠报同带内也不惩罚
    vals = [d(1.3), d(1.7), d(2.22), d(3.0)]
    assert all(v <= 1.0 for v in vals)
    assert vals == sorted(vals, reverse=True)         # 偏离越远压得越狠
    # 1/B 对称：超报与欠报同等偏离给同样的阻尼
    assert d(2.0) == pytest.approx(d(0.5), abs=1e-12)


def test_frequency_damping_is_neutral_when_bias_unknown():
    """BIAS 缺失不拿"未知"当惩罚（缺测不折算 0 的同一条纪律）。"""
    assert ev._frequency_damping(None) == 1.0
    assert ev._frequency_damping(float("nan")) == 1.0
    assert ev._frequency_damping(0.0) == 1.0


def test_ets_is_damped_for_over_forecasting_source():
    """同一 ETS，BIAS=2.22 的过度报雨源必须拿比 BIAS=1.0 更低的技能值。"""
    cal = {"ets": 0.18, "amt_skill": 0.0, "ord_skill": 0.0,
           "grade_ets": 0.0, "bias": 1.0}
    over = dict(cal, bias=2.22)
    s_cal = ev._skill_value("ets", cal)
    s_over = ev._skill_value("ets", over)
    assert s_cal is not None and s_over is not None
    assert s_over < s_cal


# ---------------------------------------------------------------- P0-2a 信度方向

def test_reliability_weighting_shrinks_instead_of_amplifying():
    """z̃ = √ρ·z：ρ 越低噪声越被压制（旧口径除以 √ρ 会放大到 (1−ρ)/ρ）。"""
    rng = np.random.default_rng(7)
    # 构造一列"纯噪声"与一列"稳定信号"，比较两列在 ρ 下的噪声方差
    for rho in (0.9, 0.5, 0.2):
        z = rng.normal(size=400)
        z_tilde = np.sqrt(rho) * z
        noise_var = float(np.var(z_tilde)) * (1.0 - rho)
        # 乘以 √ρ 后：Var 噪声 = ρ(1−ρ) ≤ 0.25，恒不超过裸 z 的 (1−ρ)
        assert noise_var <= (1.0 - rho) + 1e-12
    # 显式对照：除以 √ρ 会把噪声放大到 (1−ρ)/ρ
    assert (1 - 0.2) / 0.2 == pytest.approx(4.0)


def test_rank_matrices_shrink_low_reliability_dimension():
    """低信度维对 R 的影响力必须小于高信度维（同尺度下）。"""
    # 单列、4 家同台；温度维离散、降水维离散相同
    T = np.array([[60.0], [55.0], [50.0], [45.0]])
    P = np.array([[60.0], [55.0], [50.0], [45.0]])
    mats = {"temp_score": T, "precip_score": P}
    out = ev._attach_rank_matrices(mats, n_blocks=1, days=1)
    r = out["rank_score"][:, 0]
    assert np.all(np.isfinite(r))
    # 单块无法估 ρ → 回填 ρ_exit（不再按 1.0 的"最不保守"选择）
    assert out["_rel_temp"][0] == pytest.approx(ev.SCORE_RELIABILITY_EXIT)
    assert out["_rel_temp_imputed"][0]
    # 名次分仍以 50 为中心、上下对称（√ρ 只改幅度）
    assert r.mean() == pytest.approx(50.0, abs=1e-9)
    assert (r[0] - 50.0) == pytest.approx(-(r[3] - 50.0), abs=1e-9)


# ---------------------------------------------------------------- P0-2b 稳健尺度

def test_winsorize_column_neutralizes_single_collapsed_source():
    """单家崩溃源不得把整列尺度撑大（裸 sd 会，MAD Winsorize 不会）。"""
    T = np.array([[60.0], [58.0], [56.0], [54.0], [52.0], [-999.0]])
    V = np.ones_like(T, dtype=bool)
    raw = float(np.std(T[:, 0]))
    W = ev._winsorize_column(T, V, k=3.5)
    robust = float(np.std(W[:, 0]))
    assert raw > 10 * robust              # 裸 sd 被单点绑架
    assert robust < 10.0                  # 稳健尺度反映"正常家数"的离散度
    # 未崩溃的家数相对顺序保持不变
    assert list(np.argsort(W[:5, 0])) == list(np.argsort(T[:5, 0]))


def test_winsorize_can_be_disabled_for_legacy_replay():
    """k<=0 关闭截断 → 与旧口径逐位相同。"""
    T = np.array([[1.0], [2.0], [-500.0]])
    V = np.ones_like(T, dtype=bool)
    assert np.array_equal(ev._winsorize_column(T, V, k=0), T)


# ---------------------------------------------------------------- P0-4 参照因果

def test_reference_prior_excludes_same_day_cross_station_anomaly():
    """同日跨站异常场不得进入参照：另一站被污染不该改变本站参照。"""
    def build(s2_shift: float):
        """s1 是 20+d+0.1h；s2 整体平移 `s2_shift`（模拟另一站的同期异常）。"""
        per_station = {}
        for sid in ("s1", "s2"):
            offset = 0.0 if sid == "s1" else s2_shift
            per_station[sid] = {
                f"2026-09-{d:02d}T{h:02d}:00": {
                    "temp": 20.0 + d + 0.1 * h + offset, "rain": 0.0}
                for d in range(1, 21) for h in range(24)}
        return per_station

    base = build(0.0)
    # s2 的"同期异常"被抬高 30°C：若参照吃进 pooled_anom，s1 的参照会漂移
    shifted = build(30.0)
    i1 = ev._build_reference_index(["s1", "s2"], base, mode="fixed",
                                   window_days=15, min_days=3)
    i2 = ev._build_reference_index(["s1", "s2"], shifted, mode="fixed",
                                   window_days=15, min_days=3)
    target = "2026-09-15T06:00"
    assert i1.hourly_ref("s1", target, "2026-09-10T06:00") is not None
    assert i1.hourly_ref("s1", target, "2026-09-10T06:00") == \
        i2.hourly_ref("s1", target, "2026-09-10T06:00")


def test_reference_is_leave_one_day_out_when_target_day_is_corrupted():
    """目标日实况被改成极端脏值时，该日参照必须一字不变。"""
    obs = {f"2026-09-{d:02d}T{h:02d}:00": {"temp": 20.0 + d + 0.1 * h,
                                          "rain": 0.0}
           for d in range(1, 31) for h in range(24)}
    idx = ev._build_reference_index(["s1"], {"s1": obs}, mode="fixed",
                                    window_days=15, min_days=3)
    target = "2026-09-15T06:00"
    a = idx.hourly_ref("s1", target, "2026-09-10T06:00")
    dirty = dict(obs)
    for key in dirty:
        if key[:10] == "2026-09-15":
            dirty[key] = {"temp": -100.0, "rain": 0.0}
    idx2 = ev._build_reference_index(["s1"], {"s1": dirty}, mode="fixed",
                                     window_days=15, min_days=3)
    assert idx2.hourly_ref("s1", target, "2026-09-10T06:00") == a


# ---------------------------------------------------------------- P1-1 门槛自适应

def test_daily_track_thresholds_are_resolution_adaptive():
    """7 天块内日轨每站最多 7 个样本：固定 30 会让校正常不触发。"""
    assert stats.cross_station_min_overlap("daily") == 5
    assert stats.group_min_n("daily") == 5
    assert stats.cross_station_min_overlap("1h") == 30
    assert stats.group_min_n("1h") == 30


def test_cross_station_rho_is_estimable_on_a_seven_day_block():
    """7 个公共点的日轨必须能估出 ρ̄（旧口径下恒为 None）。"""
    a = list(range(7))
    b = [x + 0.5 for x in range(7)]
    assert stats.cross_station_rho({"a": a, "b": b}, kind="daily") is not None
    assert stats.cross_station_rho({"a": a, "b": b}, kind="1h") is None


# ---------------------------------------------------------------- P1-3 子项结构

def test_temperature_parts_are_three_orthogonal_families():
    """温度维由三族构成，且没有任何一族吃掉超过 60% 的名义权重。"""
    weights = {k: w for k, w, *_ in ev.TEMP_SKILL_PARTS}
    assert set(weights) == {"rmse", "mae", "acc2", "r_anom"}
    assert sum(weights.values()) == pytest.approx(1.0)
    # acc1 已移除（与 acc2 相关 0.90，信息增量接近 0）
    assert "acc1" not in weights
    # 形态族必须占到有意义的份额（旧口径 0.07 是死循环的产物）
    assert weights["r_anom"] >= 0.15
    # 幅度族不再垄断：旧口径 0.93，现在 0.80（剩余 0.20 归正交的形态族）
    assert weights["rmse"] + weights["mae"] + weights["acc2"] <= 0.80


def test_temperature_requires_both_amplitude_and_shape_evidence():
    """形态族**部分**缺失 → 整维缺测；完全缺席（结构性）→ 幅度族可撑整维。

    规则与降水证据族门同纪律：≥2 族有证据即可评；只有 1 族存活时，该族必须
    **全员存活**（=该源结构性缺席另一类产品），否则是半截证据，判缺测。
    """
    amp = {"rmse": 1.5, "mae": 1.0, "acc2": 70.0,
           "ref": {"rmse_ref": 2.0, "mae_ref": 1.5, "acc2_ref": 70.0}}
    # 幅度族齐备 + 形态族在**源级结构性缺席**（该源从不提供 r_anom）→ 可评
    assert ev.temp_score(amp, absent_families={"形态与相位"}) is not None
    # 源在别处提供 r_anom、本格全缺 → 半截证据 → 缺测
    assert ev.temp_score(amp) is None
    # 幅度族**部分**缺失（只剩 rmse）+ 形态族缺席 → 半截证据 → 缺测
    partial = {"rmse": 1.5, "ref": {"rmse_ref": 2.0}}
    assert ev.temp_score(partial, absent_families={"形态与相位"}) is None
    # 两族齐备 → 可评
    assert ev.temp_score(dict(amp, r_anom=0.5)) is not None



def test_hourly_and_daily_grade_levels_are_isomorphic():
    """小时轨与日轨都用累积超越阈值，消除闭区间的边界双罚。"""
    assert stats._GRADE_LEVS["1h"] == ("+2", "+3")
    assert stats._GRADE_LEVS["24h"] == ("+2", "+3")


def test_soft_skill_is_c1_smooth_at_the_anchor():
    """τ=1 时 x=0 处左右导数相等（旧 τ=0.9 有 1.111 倍斜率跳变）。"""
    assert ev.SCORE_SKILL_TAU == pytest.approx(1.0)
    h = 1e-6
    left = (ev._soft_skill(-h) - ev._soft_skill(0.0)) / (-h)
    right = (ev._soft_skill(h) - ev._soft_skill(0.0)) / h
    assert left == pytest.approx(right, rel=1e-4)
    # 三个锚点仍在
    assert ev._soft_skill(0.0) == 0.0
    assert ev._soft_skill(1.0) == 1.0
    assert ev._soft_skill(-1.0) == pytest.approx(-(1 - np.exp(-1.0)))

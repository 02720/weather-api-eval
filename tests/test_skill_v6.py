"""skill-v6（2026-11 综合分优化）机制断言。

每条对应 `docs/综合分优化_第一性原理复核.md` §3.6 的 V6-n 一行，锁的是
**行为**而不是某个数字：把实现改回去就变红。

纪律同 test_skill_v5：只写"违反第一性原理且可复算"的断言，不写"和某个快照
数字相等"的快照测试（那种测试只会把 bug 一起锁死）。
"""
from __future__ import annotations

import numpy as np
import pytest

from weather_eval import evaluate as ev


def setup_function():
    ev.apply_score_scheme("skill-v6")
    ev.apply_score_composite_params(
        alpha=0.5, sd_temp=8.77, sd_precip=3.73, rank_scale=10.0,
        tau=1.0, display_gain=1.0,
        lead_weights={"1-3": 1.0, "4-7": 0.7, "8-14": 0.4, "15-16": 0.2})


def teardown_function():
    ev.apply_score_scheme("slope-v2")


# ------------------------------------------------------------------ V6-1 (D1)
#
# 判决性理由：全样本 L1 的最优常数预报是中位数，而小时降水 95% 为 0，
# 所以"永远报零雨"在 v5 口径下是**理论最优**、得 0 分（= 参照）。
# v6 换到湿域对数比后，它必须落到负半轴（漏掉了全部雨）。

def _wet_obs() -> np.ndarray:
    return np.array([0.0] * 40 + [1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0])


def test_v6_always_dry_scores_below_zero():
    """V6-1：`F≡0` 的 amt_skill 必须严格 < 0（当湿域非空）。"""
    obs = _wet_obs()
    out = ev.precip_amount_metrics(obs, np.zeros_like(obs), 5,
                                   amt_floor=0.05, wet_threshold=1.0)
    s = out["amt_skill"]
    assert s is not None and s < 0.0, f"恒零雨必须为负技巧，实得 {s}"


def test_v6_always_dry_is_dominated_by_every_real_forecast():
    """V6-1（强形式）：没有 do-nothing 占优策略。

    "永远报零雨"的得分不得高于**任何**恒定报雨量的预报。这在 v5 下不成立
    （恒零雨 = 0，而任何报了雨的源都落负半轴）。
    """
    obs = _wet_obs()
    dry = ev.precip_amount_metrics(obs, np.zeros_like(obs), 5,
                                   amt_floor=0.05, wet_threshold=1.0)["amt_skill"]
    for q in (1.0, 2.0, 3.5, 5.0, 8.0):
        r = ev.precip_amount_metrics(obs, np.full_like(obs, q), 5,
                                     amt_floor=0.05, wet_threshold=1.0)
        s = r["amt_skill"]
        assert s is not None
        assert dry < s or (q == 0.0), (
            f"恒报 {q}mm 得分 {s:.4f} 不高于恒零雨 {dry:.4f}")


def test_v6_perfect_anchor_is_one():
    """V6-1：完美预报仍严格为 1（换域不能动满分锚点）。"""
    obs = _wet_obs()
    out = ev.precip_amount_metrics(obs, obs.copy(), 5,
                                   amt_floor=0.05, wet_threshold=1.0)
    assert out["amt_skill"] == pytest.approx(1.0, abs=1e-9)


def test_v6_constant_reference_anchor_is_zero():
    """V6-1：**q_ref 常数额**必须恰为 0（0 分的语义 = 打平最优常数预报）。

    这条同时锁住 q_ref 必须是**对数域** L1 的极小点：若改用原始中位数或均值，
    该常数自身的得分会偏离 0，参照语义就漂了。
    """
    obs = _wet_obs()
    wet = obs >= 1.0
    q_ref = float(np.expm1(np.median(np.log1p(obs[wet]))))
    out = ev.precip_amount_metrics(obs, np.full_like(obs, q_ref), 5,
                                   amt_floor=0.05, wet_threshold=1.0)
    assert out["amt_skill"] == pytest.approx(0.0, abs=1e-9)


def test_v6_no_constant_forecast_beats_q_ref():
    """V6-1：q_ref 是常数预报族里的**全局最优**，没有任何常数能拿到正技巧。"""
    rng = np.random.default_rng(20261122)
    n = 4000
    obs = np.where(rng.random(n) < 0.10, rng.gamma(0.9, 4.0, n), 0.0)
    best = -np.inf
    for q in np.linspace(0.0, 60.0, 121):
        s = ev.precip_amount_metrics(obs, np.full(n, q), 5,
                                     amt_floor=0.05,
                                     wet_threshold=1.0)["amt_skill"]
        if s is not None:
            best = max(best, s)
    assert best <= 1e-6, f"某个常数额拿到正技巧 {best:.6f}：q_ref 不是最优锚点"


def test_v6_amount_skill_is_scale_monotone_not_invariant():
    """V6-1：`ln(1+x)` **不是**尺度不变的，这里锁住真实性质而不是一句错的声明。

    v5 文档曾把"尺度不变"当锚点性质；换到对数域后它不再成立：
    `ln(1+kF)−ln(1+kO)` 随 k 增大而**压缩**（倍率被 log1p 的 +1 稀释），
    所以同一"报 1.5 倍"在小雨量桶里罚得更重、暴雨桶里罚得更轻。
    这是**设计选择**（对数刻度天然如此），必须如实锁成"单调"而非"不变"。
    注意方向：观测固定时把预报整体放大 k 倍，`ln(1+kF)` 相对 `ln(1+O)`
    的绝对偏离随 k 增大而**增大**（+1 对 O 的影响远小于对 kF 的影响），
    所以 s 随 k 单调**下降**，而不是"倍率越大惩罚越轻"。
    """
    obs = _wet_obs()
    errs = []
    for k in (2.0, 5.0, 20.0, 100.0):
        s = ev.precip_amount_metrics(obs, obs * k, 5,
                                     amt_floor=0.05,
                                     wet_threshold=1.0)["amt_skill"]
        assert s is not None
        errs.append(s)
    # 单调：超报越多，分越低（严格递减）
    assert errs == sorted(errs, reverse=True), f"超报惩罚不单调：{errs}"


def test_v6_amount_over_forecast_is_penalised():
    """V6-1：超报必然扣分（不是"报得越多越好"）。"""
    obs = _wet_obs()
    perfect = ev.precip_amount_metrics(obs, obs.copy(), 5,
                                       amt_floor=0.05,
                                       wet_threshold=1.0)["amt_skill"]
    over = ev.precip_amount_metrics(obs, obs * 3.0, 5,
                                    amt_floor=0.05, wet_threshold=1.0)["amt_skill"]
    assert over < perfect


def test_v6_amount_under_forecast_is_penalised():
    """V6-1：漏报/欠报同样扣分（对称的另一半）。"""
    obs = _wet_obs()
    perfect = ev.precip_amount_metrics(obs, obs.copy(), 5,
                                       amt_floor=0.05,
                                       wet_threshold=1.0)["amt_skill"]
    under = ev.precip_amount_metrics(obs, obs * 0.3, 5,
                                     amt_floor=0.05, wet_threshold=1.0)["amt_skill"]
    assert under < perfect


def test_v6_empty_wet_domain_returns_none():
    """V6-1：整桶无湿样本 ⇒ None（缺测），绝不伪造 0 或 100。"""
    obs = np.zeros(200)
    out = ev.precip_amount_metrics(obs, np.zeros_like(obs), 5,
                                   amt_floor=0.05, wet_threshold=1.0)
    assert out["amt_skill"] is None


# ------------------------------------------------------------------ V6-2/3 (D2)
#
# d(B) ≤ 1 乘到**负**的 ETS 上会让它更接近 0（分数上升），违反"只压不抬"。

def test_v6_damping_never_lifts_negative_ets():
    """V6-2：阻尼不得抬高负 ETS 的得分（对任意 BIAS）。"""
    for bias in (0.3, 0.6, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0):
        s = ev._skill_value("ets", {"ets": -0.20, "bias": bias})
        undamped = ev._soft_skill(-0.20)
        assert s == pytest.approx(undamped, abs=1e-12), (
            f"BIAS={bias} 时负 ETS 被抬高：{s:.6f} vs {undamped:.6f}")


def test_v6_damping_still_damps_positive_ets():
    """V6-2：正 ETS 仍按频率偏差受阻尼（不能把修正做成"关掉阻尼"）。"""
    base = ev._skill_value("ets", {"ets": 0.20, "bias": 1.0})
    damped = ev._skill_value("ets", {"ets": 0.20, "bias": 3.0})
    assert base == pytest.approx(ev._soft_skill(0.20), abs=1e-12)
    assert damped < base, "BIAS=3 未受阻尼"


def test_v6_no_ordering_inversion_between_worse_ets_sources():
    """V6-2/3：ETS 更差的源，得分不得更高（序反转反例）。

    v5 下 A(ETS=−0.30, BIAS=5) 会因阻尼抬分而**高于** B(ETS=−0.28, BIAS=1)。
    """
    a = ev._skill_value("ets", {"ets": -0.30, "bias": 5.0})
    b = ev._skill_value("ets", {"ets": -0.28, "bias": 1.0})
    assert b > a, f"序反转仍在：A={a:.6f} B={b:.6f}"


# ------------------------------------------------------------------ V6-5 (D5)
#
# a_ref ≥ 1 时上侧分母 (1−a_ref) = 0 退化。v5 走 `return 1.0` 捷径，
# 把"与参照相同"记成"完美"，让四项全打平参照的源拿到 T=62.5。

def test_v6_degenerate_acc2_reference_returns_none():
    """V6-5：a_ref≥1 时 acc2 子项为 None（该格无可比信息）。"""
    t = {"acc2": 100.0, "ref": {"acc2_ref": 100.0}}
    assert ev._skill_value("acc2", t) is None


def test_v6_source_equal_to_reference_scores_fifty():
    """V6-5：四项全与参照相同的源，温度分必须恰为 50。"""
    t = {"acc2": 100.0, "rmse": 2.0, "mae": 1.5, "r_anom": 0.0,
         "ref": {"acc2_ref": 100.0, "rmse_ref": 2.0, "mae_ref": 1.5}}
    assert ev.temp_score(t) == pytest.approx(50.0, abs=1e-9)


def test_v6_perfect_source_still_scores_one_hundred():
    """V6-5：修锚点不能顺手把完美预报也判成缺测。"""
    t = {"acc2": 100.0, "rmse": 0.0, "mae": 0.0, "r_anom": 1.0,
         "ref": {"acc2_ref": 60.0, "rmse_ref": 2.0, "mae_ref": 1.5}}
    assert ev.temp_score(t) == pytest.approx(100.0, abs=1e-9)


# -------------------------------------------------------------------- V6-6 (D3)
#
# 名义权重 ≠ 话语权：方差份额 ∝ (w_i·sd_i)²。实测 amt_skill 在小时轨达 2.14×。

def test_v6_scale_norm_table_covers_every_precip_part():
    """V6-6：归一表必须覆盖权重表里的每一个子项、且每轨齐备。"""
    keys = {k for k, *_ in ev.PRECIP_SKILL_PARTS}
    for track in ("hourly", "daily"):
        table = ev.PRECIP_SCALE_NORM.get(track)
        assert table is not None, f"缺少 {track} 轨的归一表"
        assert set(table) == keys, f"{track} 轨归一表与权重表键集不一致"


def test_v6_scale_norm_keeps_weights_normalised():
    """V6-6：归一后的**分数权重**仍然归一（Σw = 1），不改变量表。

    注意区分两个量：**分数权重** w_i 与**方差份额** (w_i·sd_i)²/Σ。归一化的
    目标是把后者拉回名义值，因此前者**本来就不该**等于名义（除非 sd 全相等）。
    曾误以为"族间分数权重必须保持 0.60/0.40"——那会把份额再次推离名义。
    """
    nom = {k: w for k, w, *_ in ev.PRECIP_SKILL_PARTS}
    for track in ("hourly", "daily"):
        w = {k: nom[k] * ev.PRECIP_SCALE_NORM[track][k] for k in nom}
        total = sum(w.values())
        assert total == pytest.approx(1.0, abs=1e-6), (
            f"{track} 归一后总权重 {total} != 1")


def test_v6_scale_norm_brings_variance_share_back_to_nominal():
    """V6-6：归一后单一子项方差份额 ∈ [0.5, 1.5]× 名义权重。

    用合成数据直接验证归一的**数学效果**（与真实数据无关，因此不会随窗口漂移）。
    """
    rng = np.random.default_rng(20261122)
    n = 6000
    keys = ["ets", "amt_skill", "ord_skill", "grade_ets"]
    nom = {k: w for k, w, *_ in ev.PRECIP_SKILL_PARTS}
    # 每轨用**该轨自己的标定离散度**：归一表是按轨冻结的，用另一轨的 sd
    # 去检验它，测的就不是它对不对（这是 test 里最易犯的"用错尺子"）。
    sd_by_track = {
        "hourly": {"ets": 0.0656, "amt_skill": 0.2403,
                   "ord_skill": 0.0802, "grade_ets": 0.0421},
        "daily": {"ets": 0.1139, "amt_skill": 0.2479,
                  "ord_skill": 0.1048, "grade_ets": 0.1182},
    }
    for track in ("hourly", "daily"):
        sd = sd_by_track[track]
        X = np.column_stack([rng.normal(0.0, sd[k], n) for k in keys])
        w = np.array([nom[k] * ev.PRECIP_SCALE_NORM[track][k] for k in keys])
        w = w / w.sum()
        S = X @ w
        var = float(S.var())
        contrib = np.array([w[i] * float(np.cov(X[:, i], S, ddof=0)[0, 1])
                            for i in range(len(keys))])
        shares = contrib / var
        for i, k in enumerate(keys):
            ratio = shares[i] / nom[k]
            assert 0.5 <= ratio <= 1.5, (
                f"{track}/{k} 方差份额 {shares[i]:.3f} 是名义 {nom[k]:.2f} "
                f"的 {ratio:.2f} 倍，越界")


# --------------------------------------------------------------- family guards

def test_v6_empty_series_still_missing():
    """回归：换域不能把"没有降水可评"变成"零技巧"（缺测不折算 0）。"""
    assert ev.precip_score({}) is None


def test_v6_scores_stay_on_the_same_scale():
    """回归：v6 与 v5 共用 50=参照 / 100=完美 的量表（不是另起一套）。"""
    p = {"ets": 0.0, "amt_skill": 0.0, "ord_skill": 0.0, "grade_ets": 0.0,
         "bias": 1.0}
    assert ev.precip_score(p) == pytest.approx(50.0, abs=1e-9)
    p2 = {"ets": 1.0, "amt_skill": 1.0, "ord_skill": 1.0, "grade_ets": 1.0,
          "bias": 1.0}
    assert ev.precip_score(p2) == pytest.approx(100.0, abs=1e-9)


# -------------------------------------------------------------------- V6-4 (D4)
#
# R 的 z 分分母来自当期参评池 ⇒ 增删一个源会移动所有幸存者的分数（实测最高
# 7 分）。冻结 μ/σ/φ 到标定期后，这个依赖必须**逐位消失**。

def _frozen_from(T, P):
    base = ev._stats.rank_from_scores(T, P, alpha=0.5, rho_t=0.5, rho_p=0.4,
                                      winsor_k=3.5)
    return base, {k: base[k][0, 0] for k in ("mu_t", "sd_t", "mu_p", "sd_p",
                                             "phi")}


def test_v6_frozen_scale_reproduces_unfrozen_on_same_pool():
    """V6-4：同一批数据上，冻结尺度与当期尺度必须给出**相同**的 R。"""
    rng = np.random.default_rng(7)
    T = rng.normal(55.0, 7.0, (20, 1))
    P = rng.normal(48.0, 4.0, (20, 1))
    base, fz = _frozen_from(T, P)
    got = ev._stats.rank_from_scores(T, P, alpha=0.5, rho_t=0.5, rho_p=0.4,
                                     winsor_k=3.5, frozen_scale=fz)
    assert np.allclose(got["rank_score"], base["rank_score"], atol=1e-12)


def test_v6_frozen_scale_makes_rank_pool_independent():
    """V6-4：加入/删除任意一个源，幸存者的 R **逐位不变**。"""
    rng = np.random.default_rng(11)
    M = 20
    T = rng.normal(55.0, 7.0, (M, 1))
    P = rng.normal(48.0, 4.0, (M, 1))
    _, fz = _frozen_from(T, P)
    ref = ev._stats.rank_from_scores(T, P, alpha=0.5, rho_t=0.5, rho_p=0.4,
                                     winsor_k=3.5, frozen_scale=fz)["rank_score"][:, 0]
    # 加入强/弱/崩溃三种源
    for add in (30.0, 80.0, -200.0):
        T2 = np.vstack([T, [[add]]])
        P2 = np.vstack([P, [[add * 0.8]]])
        got = ev._stats.rank_from_scores(
            T2, P2, alpha=0.5, rho_t=0.5, rho_p=0.4,
            winsor_k=3.5, frozen_scale=fz)["rank_score"][:M, 0]
        assert np.abs(got - ref).max() < 1e-12, f"加入 {add} 后幸存者 R 变了"
    # 删除
    for drop in (0, 5, M - 1):
        idx = [i for i in range(M) if i != drop]
        got = ev._stats.rank_from_scores(
            T[idx], P[idx], alpha=0.5, rho_t=0.5, rho_p=0.4,
            winsor_k=3.5, frozen_scale=fz)["rank_score"][:, 0]
        assert np.abs(got - np.delete(ref, drop)).max() < 1e-12, (
            f"删除第 {drop} 名后幸存者 R 变了")


def test_v6_frozen_scale_is_opt_in():
    """V6-4：不传 frozen_scale 时行为与旧口径逐位相同（归档回放不受影响）。"""
    rng = np.random.default_rng(13)
    T = rng.normal(55.0, 7.0, (12, 3))
    P = rng.normal(48.0, 4.0, (12, 3))
    a = ev._stats.rank_from_scores(T, P, alpha=0.5, rho_t=0.5, rho_p=0.4)
    b = ev._stats.rank_from_scores(T, P, alpha=0.5, rho_t=0.5, rho_p=0.4,
                                   frozen_scale=None)
    assert np.allclose(a["rank_score"], b["rank_score"], equal_nan=True)


# -------------------------------------------------------------------- P3 专项卡
#
# 强降水（+4/+5/+6）**只披露、不进任何分数**：稀有事件不能靠平均把噪声洗成
# 能力，但也不能静默删除（"未检验" ≠ "能力为 0"）。

def _heavy(n_dry: int, obs_heavy: list[float], fcst_heavy: list[float]):
    obs = np.array([0.0] * n_dry + obs_heavy)
    fc = np.array([0.0] * n_dry + fcst_heavy)
    return obs, fc


def test_v6_heavy_rain_card_never_enters_composite():
    """P3：专项卡必须显式声明不进综合分，且不含任何合成分数字段。"""
    obs, fc = _heavy(90, [60.0, 70.0, 55.0], [40.0, 90.0, 10.0])
    card = ev.heavy_rain_card(obs, fc, "24h")
    assert card["enters_composite"] is False
    for level in card["levels"]:
        for k in ("score", "skill", "weight"):
            assert k not in level, f"专项卡混进了合成分字段 {k}"


def test_v6_heavy_rain_evidence_labels_follow_event_count():
    """P3：证据强度按**观测事件数**分档（untested / low_conf / formal）。"""
    # 5 次事件 → untested
    obs, fc = _heavy(200, [60.0] * 5, [60.0] * 5)
    assert ev.heavy_rain_card(obs, fc, "24h")["levels"][0]["evidence"] == "untested"
    # 15 次 → low_conf
    obs, fc = _heavy(200, [60.0] * 15, [60.0] * 15)
    assert ev.heavy_rain_card(obs, fc, "24h")["levels"][0]["evidence"] == "low_conf"
    # 40 次 → formal
    obs, fc = _heavy(200, [60.0] * 40, [60.0] * 40)
    assert ev.heavy_rain_card(obs, fc, "24h")["levels"][0]["evidence"] == "formal"


def test_v6_heavy_rain_zero_events_is_not_perfect_score():
    """P3：观测 0 事件时不得给出"全部正确"的满分（不给 ETS）。"""
    obs, fc = _heavy(300, [], [])
    card = ev.heavy_rain_card(obs, fc, "24h")
    for level in card["levels"]:
        assert level["obs_events"] == 0
        assert level["ets"] is None, "无观测事件却给出了 ETS"


def test_v6_heavy_rain_1h_excludes_level6():
    """P3：1h 没有独立第 6 级（cyeva 把它映射到第 5 级），不得重复汇报。"""
    obs, fc = _heavy(50, [30.0], [30.0])
    levs = [r["lev"] for r in ev.heavy_rain_card(obs, fc, "1h")["levels"]]
    assert "+6" not in levs, f"1h 不应出现 +6：{levs}"
    levs24 = [r["lev"] for r in ev.heavy_rain_card(obs, fc, "24h")["levels"]]
    assert "+6" in levs24


# ------------------------------------------------------------------ D4 接线
#
# 冻结尺度必须**按榜**取用：三张榜的列轴长度不同（K×H / K×D / K×(H+D)），
# 总榜是两条单轨的横向拼接。存一份通用的"每列尺度"会让总榜靠列数巧合误配。

def _frozen_payload():
    return {
        "hourly": {"mu_t": [1.0] * 4, "sd_t": [2.0] * 4,
                   "mu_p": [3.0] * 4, "sd_p": [4.0] * 4, "phi": [0.0] * 4},
        "daily": {"mu_t": [5.0] * 3, "sd_t": [6.0] * 3,
                  "mu_p": [7.0] * 3, "sd_p": [8.0] * 3, "phi": [0.1] * 3},
    }


def test_v6_frozen_scale_is_per_board():
    """D4：单轨按自己的列数取；总榜按 hourly+daily 拼接取。"""
    ev.apply_frozen_rank_scale(_frozen_payload())
    try:
        assert ev._frozen_scale_for(np.zeros((2, 4)), "hourly") is not None
        assert ev._frozen_scale_for(np.zeros((2, 3)), "daily") is not None
        all_scale = ev._frozen_scale_for(np.zeros((2, 7)), "all")
        assert all_scale is not None
        # 列序必须是"先全部小时列、再全部日列"（与 _panel_matrices 的 hstack 一致）
        assert all_scale["mu_t"].reshape(-1).tolist() == \
            [1.0] * 4 + [5.0] * 3
    finally:
        ev.apply_frozen_rank_scale(None)


def test_v6_frozen_scale_rejects_column_mismatch():
    """D4：列数对不上必须返回 None（宁可不冻结，也不静默错配）。"""
    ev.apply_frozen_rank_scale(_frozen_payload())
    try:
        assert ev._frozen_scale_for(np.zeros((2, 5)), "hourly") is None
        assert ev._frozen_scale_for(np.zeros((2, 6)), "all") is None
        assert ev._frozen_scale_for(np.zeros((2, 4)), "bogus") is None
    finally:
        ev.apply_frozen_rank_scale(None)


def test_v6_frozen_scale_disabled_by_default():
    """D4：缺省不冻结 ⇒ 逐位回到当期池标准化（归档回放不受影响）。"""
    assert ev.SCORE_RANK_FROZEN_SCALE is None


# --------------------------------------------------- 口径家族判定的一致性
#
# v6 上线时踩过两次"写死 skill-v5"的坑：页面"评分构成"表回落到 slope-v2 的
# 旧权重、诊断层的话语权披露整段消失、归档重渲染用错参数复算。三个都表现为
# **静默失效**（页面照样出，只是数字含义变了），所以用断言钉死。

def test_v6_is_in_every_scheme_family_gate():
    """所有"技能口径家族"的判定点都必须把 v6 算进去。"""
    import inspect
    from weather_eval.report import diagnostics, render

    assert "skill-v6" in ev.SKILL_SCHEMES
    # render._slim_report 的口径回放：v6 必须与 v5 同支（否则用旧权重复算）
    src = inspect.getsource(render._slim_report)
    assert "skill-v6" in src, "render._slim_report 漏了 v6 的口径回放"
    # 页面"评分构成"表选权重表：漏 v6 会回落到 slope-v2 的旧表
    src2 = inspect.getsource(render)
    assert '_scheme in ("skill-v3", "skill-v4", "skill-v5", "skill-v6")' in src2
    # 诊断层话语权披露：不能写死单个口径名
    src3 = inspect.getsource(diagnostics)
    assert 'SCORE_SCHEME == "skill-v5"' not in src3


def test_v6_apply_score_scheme_accepts_v6():
    ev.apply_score_scheme("skill-v6")
    assert ev.SCORE_SCHEME == "skill-v6"
    assert ev.is_v5_family() and ev.is_v6_family()
    with pytest.raises(ValueError):
        ev.apply_score_scheme("skill-v9")


def test_v6_render_uses_skill_parts_not_slope_parts():
    """页面评分构成表必须取技能权重表（v6 与 v5 同表），不是 slope-v2 的旧表。"""
    from weather_eval.report import render

    assert render.TEMP_SKILL_PARTS is ev.TEMP_SKILL_PARTS
    assert render.PRECIP_SKILL_PARTS is ev.PRECIP_SKILL_PARTS
    assert [k for k, *_ in render.PRECIP_SKILL_PARTS] == \
        ["ets", "amt_skill", "ord_skill", "grade_ets"]

"""统计内核的对抗式审查回归（P0-1/P0-2/P0-4/P0-5/P1-2/P1-4）。

每一条都对应审查报告里的一个具体缺陷，且都写成"实现改回去就变红"的形式——
这类缺陷的特点是**不会报错**，只会安静地给出另一个名次。
"""
import numpy as np

from weather_eval import stats


# ---------------------------------------------------------------- P0-5 ddof
def test_pearson_never_exceeds_one():
    """ddof 混用回归：`np.cov/(sqrt(var·var))` 会把 ρ 放大 n/(n−1)，
    完全相关序列会算出 1.0208 这种不可能的值，并在小样本上误触 RHO_CLAMP。"""
    for n in (3, 50, 591):
        x = np.arange(n, dtype=float)
        r = stats.pearson_r(x, x)
        assert r is not None and r <= 1.0 + 1e-12, (n, r)
        assert abs(r - 1.0) < 1e-9, (n, r)
    # 反向完全相关
    x = np.arange(50, dtype=float)
    assert abs(stats.pearson_r(x, -x) + 1.0) < 1e-9
    # 手写旧公式会 >1，这里作为"差异确实存在"的证据
    a, b = x[:-1], x[1:]
    legacy = float(np.cov(a, b)[0, 1] / np.sqrt(a.var() * b.var()))
    assert legacy > 1.0
    assert stats.pearson_r(a, b) <= 1.0


def test_pearson_degenerate_returns_none():
    assert stats.pearson_r(np.ones(10), np.arange(10.0)) is None      # 零方差
    assert stats.pearson_r(np.array([]), np.array([])) is None        # 空
    assert stats.pearson_r(np.arange(5.0), np.arange(6.0)) is None    # 长度不等
    # 含 NaN 时按成对有效值处理；有效点不足则 None
    assert stats.pearson_r(np.array([1.0, np.nan]), np.array([1.0, 2.0])) is None
    assert abs(stats.pearson_r(np.array([1.0, 2.0, np.nan, 4.0]),
                               np.array([1.0, 2.0, 3.0, 4.0])) - 1.0) < 1e-9


# ------------------------------------------------------- P0-4 跨站相关校正
def test_n_eff_cross_station_correction():
    """4 个由同一信号驱动的站，合起来只有独立情形的 1/(1+3ρ̄) 信息量。"""
    rng = np.random.default_rng(3)
    base = np.cumsum(rng.normal(size=400))          # 共同信号
    stations = {f"s{i}": list(base + rng.normal(size=400) * 0.05) for i in range(4)}
    independent = {f"s{i}": list(rng.normal(size=400)) for i in range(4)}
    times = {f"s{i}": [f"2026-01-01T{h:03d}" for h in range(400)] for i in range(4)}

    n_corr = stats.n_eff_from_station_series(stations, times)
    n_indep = stats.n_eff_from_station_series(independent, times)
    assert n_corr < n_indep, (n_corr, n_indep)
    # ρ̄≈1 时四个站≈一份信息：至少打折 3 倍
    assert n_corr < n_indep / 3.0
    rho = stats.cross_station_rho(stations, times)
    assert rho is not None and rho > 0.8


def test_n_eff_cross_station_falls_back_without_times():
    """给不出公共时刻时退回"按位置对齐"，但单站/样本不足时**不校正**（保守）。"""
    series = {"a": [1.0, 2.0, 3.0], "b": [1.0, 2.0, 3.0]}
    # 公共长度 3 < CROSS_STATION_MIN_OVERLAP → 不校正，等价于旧口径
    assert stats.n_eff_from_station_series(series) == sum(
        stats.effective_n(np.asarray(v)) for v in series.values())
    assert stats.n_eff_from_station_series({"a": [1.0, 2.0, 3.0]}) == 3


def test_effective_n_no_cliff_when_series_grows():
    """样本变多，n_eff 绝不许变少（UKMO"样本积累中"事故回归）。

    旧实现有 NEFF_MIN_SERIES=30 的硬开关：每站序列 ≤30 天完全不估自相关，
    ≥31 天全额折减。跨月累计窗口把日温度序列从 30 天养到 37 天时，估计器
    "换挡"，UKMO 的 n_eff 从 51（9 月月报，达标）骤跌到 18（掉出门槛）——
    **加了数据反而更缺样本**，违反信息量单调不减的第一性原理，把一个数据
    链路完全正常的源踢回"样本积累中"。改为连续噪声下限后：

    - 强持久序列（滞后 1 天相关 ≈ 1）在 30/37 天两个长度上都必须真折减
      （旧开关在 30 天处返回原始 n，正是悬崖的起跳点）；
    - 且 n_eff 随序列变长不减。
    改回硬开关或去掉噪声下限，本测试变红。
    """
    t = np.arange(60, dtype=float)
    smooth = np.sin(2 * np.pi * t / 100.0)      # 滞后 1 天相关 ≈ 0.998（RHO_CLAMP 触顶）
    n30 = stats.effective_n(smooth[:30])
    n37 = stats.effective_n(smooth[:37])
    assert n30 < 30, "强持久序列按原始 n 记数 = 旧硬开关复辟"
    assert n37 >= n30, (n30, n37)


def test_effective_n_small_sample_still_protected():
    """噪声下限保留了旧 guard 的本意：短序列的噪声 ρ̂ 不许凭空折损样本。

    白噪声短序列（n=10，下限 z/√n≈0.52）的任何滞后相关都按噪声抹平，
    n_eff 必须接近原始 n；长序列（n=600）下限趋近 0，全额校正照常生效
    （AR(1) 的定量断言见 test_evaluate.py::test_neff_effective_sample_size_and_gate）。
    """
    rng = np.random.default_rng(7)
    white_short = rng.normal(0, 1, 10)
    assert stats.effective_n(white_short) >= 8


# ------------------------------------------------------ P0-1 加权双向拟合
def test_two_way_fit_weights_change_the_answer():
    """等权与加权**必须**给出不同的行分——否则权重根本没进拟合。"""
    # 两行：A 在所有列都很稳；B 只有一列、且该列样本极薄（噪声大）
    S = np.array([[70.0, 65.0, 60.0], [99.0, np.nan, np.nan]])
    W = np.array([[100.0, 100.0, 100.0], [1.0, np.nan, np.nan]])
    equal = stats.two_way_adjust(S, min_col=1, min_row=1)["scores"]
    weighted = stats.two_way_adjust(S, min_col=1, min_row=1,
                                    weights=W)["scores"][1]
    assert np.isfinite(equal[1]) and np.isfinite(weighted)
    # 薄格子的虚高分在加权后被压下去（列效应由厚格子主导）
    assert weighted != equal[1]


def test_min_cell_weight_drops_thin_cells():
    """低于权重门槛的格子不进设计，且被如实计数（P0-1 建议 2）。

    第四轮 P1-1：旧实现只在 design_mask 之前剔薄格，_fit_parts 又按 isfinite
    把它请回来、按原权重参与 WLS——本测试曾"空转通过"（设计整体塌缩走早退
    分支），对真正的拟合路径零验收。现在断言三件事：计数、掩膜、且**分数
    确实因剔除而变**（薄格若还在拟合里，权重 0.5 的极端值会把行分拉偏）。
    """
    S = np.array([[70.0, 70.0, 70.0],
                  [60.0, 60.0, 60.0],
                  [10.0, np.nan, np.nan]])
    W = np.array([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0], [0.5, np.nan, np.nan]])
    # min_col=2 / min_row=1：让设计在"剔一格"后仍然立得住（缺省门槛下小例子
    # 整体塌缩走早退分支——旧测试正是这样空转通过的）
    out = stats.two_way_adjust(S, min_col=2, min_row=1, weights=W, min_cell_weight=1.0)
    assert out["dropped_thin_cells"] == 1
    assert not out["cell_valid"][2, 0]
    # 设计未塌缩：行 0/1 仍三格齐整可比
    assert bool(out["row_keep"][0]) and bool(out["row_keep"][1])
    assert bool(out["cell_valid"][0, 0])
    # 与"薄格仍在拟合"（min_cell_weight=0）的分数不同——否则剔除是假动作
    out_keep = stats.two_way_adjust(S, min_col=2, min_row=1, weights=W, min_cell_weight=0.0)
    assert not np.allclose(
        np.nan_to_num(out["scores"], nan=-999),
        np.nan_to_num(out_keep["scores"], nan=-999))
    # 薄格剔除后，行 2 的分数必须为 NaN（它在设计里不再有任何格子）
    assert np.isnan(out["scores"][2])


def test_bootstrap_fallback_reference_is_highest_mean():
    """未传 top_model 时，显著性参照必须是分布均值最高的入围者（第四轮 P1-2）。

    旧实现把参照系落在"第一个入围者"上：冠军频率明明说 best 100% 夺冠，
    † 却挂在别的家头上。
    """
    macro = np.column_stack([np.full(50, 10.0), np.full(50, 60.0),
                             np.full(50, 40.0), np.full(50, 5.0)])
    out = stats._summarize_bootstrap(macro, ["worst", "best", "mid", "low"],
                                     [True] * 4, top_model=None)
    assert out["best"]["champion_pct"] == 100.0
    assert out["best"]["p_vs_top"] == 0.0          # best 自己是参照系
    assert out["worst"]["p_vs_top"] is not None    # 其余家都与他比较


def test_ridge_shrinks_columns_not_rows():
    """ridge 只收缩列（难度），绝不收缩行（技巧）——第四轮 P1-4。

    行收缩会把"覆盖越短排名越低"的偏置从难度对齐要消掉的方向请回来。
    """
    rng = np.random.default_rng(11)
    S = 60 + rng.normal(0, 2, size=(4, 6)) + rng.normal(0, 2, size=(4, 1))
    W = np.full((4, 6), 20.0)
    base = stats.two_way_adjust(S, min_col=2, min_row=1, weights=W)
    ridge = stats.two_way_adjust(S, min_col=2, min_row=1, weights=W, ridge=50.0)
    # 行效应在收缩下基本不动（列被拉向 0，行吸收的是列让出来的份额）
    assert np.allclose(np.nan_to_num(base["row_effects"], nan=0),
                       np.nan_to_num(ridge["row_effects"], nan=0), atol=1.0)
    # 列确实被收缩
    assert (np.nanmax(np.abs(np.nan_to_num(ridge["col_effects"])))
            <= np.nanmax(np.abs(np.nan_to_num(base["col_effects"]))) + 1e-9)


def test_difficulty_adjusted_respects_explicit_valid_mask():
    """P1-3：给定 valid 掩膜时，派生列必须与综合分**逐格同集**。

    构造"格子 A 在指标矩阵里有值、但在综合分矩阵里不可用"的情形：显式掩膜必须
    把它排除，否则两列数字来自两批不同的格子。
    """
    S = np.array([[80.0, 74.0], [78.0, np.nan]])
    other = np.array([[90.0, 88.0], [85.0, 60.0]])       # 第二行第二列"看起来有值"
    design = np.isfinite(S)
    rows = np.array([True, True])
    cols = np.array([True, True])
    with_mask = stats.difficulty_adjusted(other[None, ...], rows, cols,
                                          valid=design[None, ...])[0]
    without = stats.difficulty_adjusted(other[None, ...], rows, cols)[0]
    assert np.isfinite(with_mask[1])
    assert with_mask[1] != without[1]                     # 掩膜确实生效了
    # 无掩膜时 60 分会把第二行的行效应拉下来；有掩膜时它不参与
    assert with_mask[1] > without[1]


def test_min_col_frac_excludes_long_tail_buckets():
    """P1-4：家数不足"最热闹桶的一半"的长尾桶不进主设计。"""
    V = np.zeros((10, 4), dtype=bool)
    V[:, 0] = True          # 10 家
    V[:6, 1] = True         # 6 家
    V[:2, 2] = True         # 2 家（长尾）
    rows, cols = stats.design_mask(V, min_col=2, min_row=1, min_col_frac=0.5)
    assert cols.tolist() == [True, True, False, False]
    # 关掉相对门槛时第三列（2 家）会进来
    _, cols0 = stats.design_mask(V, min_col=2, min_row=1, min_col_frac=0.0)
    assert cols0.tolist() == [True, True, True, False]


def test_ridge_shrinks_thin_column_effects():
    """P1-4：列收缩把薄桶的极端难度拉向平均值。"""
    S = np.array([[80.0, 30.0], [78.0, np.nan]])
    plain = stats.two_way_adjust(S, min_col=1, min_row=1)
    ridged = stats.two_way_adjust(S, min_col=1, min_row=1, ridge=10.0)
    assert abs(ridged["col_effects"][1]) < abs(plain["col_effects"][1]) + 1e-9


def test_equal_weight_path_is_bit_identical_to_legacy():
    """等权 + 不收缩时与旧实现逐位相同（回归锁定，避免"顺手改坏旧口径"）。"""
    rng = np.random.default_rng(11)
    S = np.where(rng.random((4, 5)) > 0.2,
                 70 + rng.normal(0, 4, (4, 5)), np.nan)
    out = stats.two_way_adjust(S, min_col=2, min_row=1)
    # 旧路径：design_mask + _fit_parts（无权重）
    V = np.isfinite(S)
    rows, cols = stats.design_mask(V, min_col=2, min_row=1)
    cr, cc = stats.largest_component_mask(rows, cols, V)
    rows, cols = (cr, cc) if not (np.array_equal(cr, rows) and np.array_equal(cc, cols)) \
        else (rows, cols)
    expect = stats.difficulty_adjusted(S[None, ...], rows, cols)[0]
    assert np.allclose(out["scores"], expect, atol=1e-12, equal_nan=True)


# ---------------------------------------------- P0-2 方差分解与名次稳定性
def test_variance_decomposition_exact_additive_model():
    """纯加法数据 → 残差份额 ≈ 0，且行列份额之和 ≈ 1。

    这条同时锁住一个曾经的真实 bug：把总平方和写成 Σw·S̄²（而不是离均差平方和）
    会让 total 虚高上百倍、残差被 max(0,·) 压成 0——"交互项占比"于是永远是 0，
    把"加法模型解释不了多少"这个关键披露变成噪声。
    """
    alpha = np.array([5.0, -3.0, 0.0, 8.0])
    beta = np.array([10.0, 4.0, 0.0, -6.0, -12.0])
    S = 60.0 + alpha[:, None] + beta[None, :]
    rows = np.ones(4, dtype=bool)
    cols = np.ones(5, dtype=bool)
    vd = stats.variance_decomposition(S, rows, cols)
    assert vd["residual_share"] is not None
    assert vd["residual_share"] < 1e-9, vd
    assert abs(vd["row_share"] + vd["col_share"] - 1.0) < 1e-6, vd
    # 份额必须与真实平方和一致（不只是"加起来等于 1"）
    ac = alpha - alpha.mean()
    bc = beta - beta.mean()
    ss = np.sum((S - S.mean()) ** 2)
    assert abs(vd["row_share"] - np.sum(np.outer(ac ** 2, np.ones(5))) / ss) < 1e-6
    assert abs(vd["col_share"] - np.sum(np.outer(np.ones(4), bc ** 2)) / ss) < 1e-6


def test_variance_decomposition_detects_interaction():
    """存在强交互时残差份额必须显著为正。"""
    m, b = 5, 6
    alpha = np.linspace(-5, 5, m)
    beta = np.linspace(8, -8, b)
    inter = np.outer(np.linspace(-6, 6, m), np.linspace(-1, 1, b))
    S = 60.0 + alpha[:, None] + beta[None, :] + inter
    vd = stats.variance_decomposition(S, np.ones(m, bool), np.ones(b, bool))
    assert vd["residual_share"] > 0.1, vd


def test_bucket_rank_stability_and_rank_sensitivity():
    # 完全同序 → ρ=1；翻转 → ρ=−1
    S = np.array([[90.0, 90.0], [80.0, 80.0], [70.0, 70.0], [60.0, 60.0]])
    st = stats.bucket_rank_stability(S, np.ones(4, bool), np.ones(2, bool),
                                     min_common=3)
    assert st["pairs"] and abs(st["pairs"][0]["rho"] - 1.0) < 1e-9
    S2 = np.array([[90.0, 60.0], [80.0, 70.0], [70.0, 80.0], [60.0, 90.0]])
    st2 = stats.bucket_rank_stability(S2, np.ones(4, bool), np.ones(2, bool),
                                      min_common=3)
    assert st2["pairs"][0]["rho"] < -0.9
    sens = stats.rank_sensitivity(np.array([90.0, 80.0, 70.0, 60.0]),
                                  np.array([90.0, 80.0, 70.0, 60.0]))
    assert sens["spearman"] == 1.0 and sens["top10_changed"] == 0
    assert sens["movers"][0]["rank_equal"] == 1


# ----------------------------------------------- P1-2 宏观权重进敏感性分析
def _toy_parts():
    temp_parts = (("acc2", 1.0, "", "", lambda v: v),)
    precip_parts = (("ets", 1.0, "", "", lambda v: v * 100),)
    return temp_parts, precip_parts


def test_weight_champion_distribution_perturbs_macro_weight():
    """温度:降水恒为 50:50 时，冠军必然是同温度分更高的那家；
    一旦把宏观权重扰动到 (0,1)，降水强的家有概率翻盘。

    旧实现把两维各自归一化后**恒定按 50:50 平均**，于是全榜最有争议的那个权重
    压根不在敏感性分析范围内。
    """
    temp_parts, precip_parts = _toy_parts()
    # 单桶、两模型：A 温度满分降水零分；B 温度零分降水满分（ets=1 → 100）
    temp_sub = np.array([[[100.0]], [[0.0]]])
    prec_sub = np.array([[[0.0]], [[100.0]]])

    fixed = stats.weight_champion_distribution(
        temp_sub, prec_sub, temp_parts, precip_parts, runs=200,
        macro_weight_range=(0.5, 0.5))
    names = {d["index"]: d["pct"] for d in fixed}
    assert names.get(0, 0) == 100.0        # 50:50 时按温度决胜

    # 同一份数据，只把宏观权重纳入扰动 → 冠军分布必须出现另一家
    varying = stats.weight_champion_distribution(
        temp_sub, prec_sub, temp_parts, precip_parts, runs=400,
        macro_weight_range=(0.05, 0.95))
    pcts = {d["index"]: d["pct"] for d in varying}
    assert len(pcts) == 2, pcts
    assert pcts[1] > 20.0 and pcts[0] > 20.0

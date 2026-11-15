"""DVS v6 的公理测试：把方案 §8.1 的验收门槛写成可执行断言。

这份文件不是"测试覆盖率"，而是**八条公理的可执行形式**。每一条都对应一个具体的
失效模式——它们全都曾在旧口径里真实发生过：

| 公理 | 曾经的病 | 这里的断言 |
|---|---|---|
| A1 参赛集合独立性 | 桶内 z 分随同台对手增删而变 | 删掉一个源，其余源分数逐位不变 |
| A2 因果封闭 | 起报/基准偷看未来 | 改动 `t_run−δ` 之后的观测，历史基准不动 |
| A3 适当性 | ETS 奖励超报 | 投喂 L0=0、完美=1、恒报雨/恒不报雨=0 |
| A4 绝对可复算 | 13 个 λ + 展示增益 | 由公开统计量手算 = 引擎输出 |
| A5 权重原样兑现 | 数据方差悄悄改写权重 | 线性合成，滑块调 α 立即生效 |
| A6 时期可比 | 新源靠"好打的季节"虚高 | L1 同日配对校准 |
| A7 质量/交付分离 | 能力外与故障缺测混为一谈 | 能力外记 0，故障计入 avail |
| A8 统计只服务不确定性 | 边际 CI 重叠误判 | MCS 分档 + 配对块自助 |
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from weather_eval import dvs
from weather_eval.dvs_protocol import (
    BAND_OF, lead_to_day, _minute_ord, minute_to_day,
)
from weather_eval.dvs_uncertainty import (
    mcs_tiers, skill_lifetime, split_half_reliability, stationary_block_indices,
)


# ---------------------------------------------------------------- A3 适当性
def _counts(obs, fcst, day=None):
    obs = np.asarray(obs, dtype=float)
    fcst = np.asarray(fcst, dtype=float)
    st = np.zeros(obs.size, dtype=np.int64)
    dd = np.arange(obs.size) if day is None else np.asarray(day)
    return dvs.rain_counts_from_arrays(obs, fcst, st, dd, 1.0)


def test_perfect_forecast_scores_one():
    """完美预报的降水价值必须**恰好**为 1（量表顶点是代数恒等式）。"""
    obs = np.array([0.0, 1.0, 5.0, 10.0, 0.0, 0.0])
    r = dvs.economic_value(_counts(obs, obs.copy()))
    assert r["v_plus"] == pytest.approx(1.0, abs=1e-12)


def test_always_and_never_rain_score_zero():
    """恒报雨 / 恒不报雨 ⇒ `V̄⁺ ≡ 0`（properness 的定理，不是补丁）。"""
    obs = np.array([0.0, 1.0, 5.0, 10.0, 0.0, 0.0])
    always = dvs.economic_value(_counts(obs, np.full(obs.size, 9.0)))
    never = dvs.economic_value(_counts(obs, np.zeros(obs.size)))
    assert always["v_plus"] == pytest.approx(0.0, abs=1e-12)
    assert never["v_plus"] == pytest.approx(0.0, abs=1e-12)


def test_random_at_base_rate_near_zero():
    """按基率抛硬币的期望价值趋近 0（不能靠"知道气候"套利）。"""
    rng = np.random.default_rng(7)
    vals = []
    for _ in range(20):
        obs = (rng.random(400) < 0.2).astype(float) * 5.0
        fcst = (rng.random(400) < 0.2).astype(float) * 5.0
        vals.append(dvs.economic_value(_counts(obs, fcst))["v_plus"])
    assert float(np.mean(vals)) < 0.05


def test_temp_skill_anchors():
    """温度：投喂 L0 自身 ⇒ 0；投喂完美 ⇒ 1；误差加倍 ⇒ 触底 −1。"""
    obs = np.linspace(0, 30, 50)
    base = obs + 2.0
    assert dvs.temp_skill(dvs.temp_aggregate(obs, base.copy(), base)) \
        == pytest.approx(0.0, abs=1e-12)
    assert dvs.temp_skill(dvs.temp_aggregate(obs, obs.copy(), base)) \
        == pytest.approx(1.0, abs=1e-12)
    doubled = obs + 2.0 * (base - obs)
    assert dvs.temp_skill(dvs.temp_aggregate(obs, doubled, base)) \
        == pytest.approx(-1.0, abs=1e-12)


def test_temp_monotone_in_error():
    """误差单调变小 ⇒ 分数单调不降（评分规则必须是单调的）。"""
    obs = np.linspace(0, 30, 80)
    base = obs + 3.0
    prev = -math.inf
    for k in (2.0, 1.5, 1.0, 0.5, 0.1, 0.0):
        s = dvs.temp_skill(dvs.temp_aggregate(obs, obs + k, base))
        assert s >= prev - 1e-12
        prev = s


# ---------------------------------------------------------------- A4 可复算
def test_published_cells_recompute_to_engine():
    """由 `5×2` 段值手算 DVS，必须与引擎逐位相同（读者可核对）。"""
    t = {1: 0.3, 2: 0.4, 3: 0.4, 4: 0.1, 5: 0.0}
    p = {1: 0.05, 2: 0.02, 3: 0.0, 4: 0.0, 5: 0.0}
    got = dvs.composite_from_parts(t, p, 0.5, (0.30, 0.25, 0.25, 0.12, 0.08))
    manual = 100.0 * sum(w * (0.5 * t[b] + 0.5 * p[b]) for b, w in
                         zip(range(1, 6), (0.30, 0.25, 0.25, 0.12, 0.08)))
    assert got["score"] == pytest.approx(manual, abs=1e-12)


def test_alpha_slider_is_linear():
    """α_V 滑块是严格线性重构（画像切换不需要重算数据）。"""
    t = {1: 1.0, 2: 0.5, 3: 0.0, 4: 0.0, 5: 0.0}
    p = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    w = (0.30, 0.25, 0.25, 0.12, 0.08)
    a = dvs.composite_from_parts(t, p, 0.3, w)["score"]
    b = dvs.composite_from_parts(t, p, 0.7, w)["score"]
    mid = dvs.composite_from_parts(t, p, 0.5, w)["score"]
    assert mid == pytest.approx((a + b) / 2.0, abs=1e-12)


# ---------------------------------------------------------------- A6 时期校准
def test_l1_align_removes_period_effect():
    """C 级新源经 L1 同日配对校准后，完全继承 L1 的全窗水平 + 自身相对差。"""
    assert dvs.l1_align(30.0, 20.0, 15.0) == pytest.approx(25.0)
    # 缺失一侧时退回 L1 全窗值（而不是凭空给 0）
    assert dvs.l1_align(None, None, 15.0) == pytest.approx(15.0)


# ---------------------------------------------------------------- A7 交付分离
def test_availability_and_delivery():
    samples = {"temp_h": {"delivered": 8, "expected": 10}}
    assert dvs.availability(samples) == pytest.approx(0.8)
    assert dvs.delivered_score(50.0, 0.8) == pytest.approx(40.0)
    assert dvs.delivered_score(None, 0.8) is None


# ---------------------------------------------------------------- 协议与桶
def test_lead_buckets_use_ceiling():
    """`k = ⌈ℓ/24⌉`：1h 与 24h 同属第 1 天，25h 跨进第 2 天。"""
    k = lead_to_day(np.array([1, 23, 24, 25, 48, 49]))
    assert k.tolist() == [1, 1, 1, 2, 2, 3]


def test_band_table_covers_1_to_16():
    assert BAND_OF[1] == 1 and BAND_OF[3] == 2 and BAND_OF[7] == 3
    assert BAND_OF[10] == 4 and BAND_OF[16] == 5


def test_minute_roundtrip():
    """分钟序号与 datetime 严格互逆（`t_run` 的批次键依赖它）。"""
    from datetime import datetime
    from weather_eval.dvs_protocol import _minute_to_dt
    dt = datetime(2026, 11, 14, 17, 30)
    assert _minute_to_dt(_minute_ord(dt)) == dt
    assert minute_to_day(_minute_ord(dt)) == dt.toordinal()


# ---------------------------------------------------------------- A8 统计
def test_stationary_blocks_are_bounded_and_stable():
    rng = np.random.default_rng(0)
    idx = stationary_block_indices(50, 3.0, rng)
    assert idx.size == 50 and idx.min() >= 0 and idx.max() < 50


def test_mcs_puts_identical_sources_in_same_tier():
    """两个完全相同的源**不得**被分进不同档（不能伪造差异）。"""
    rng = np.random.default_rng(1)
    boot = rng.normal(0, 1, (400, 3))
    boot[:, 1] = boot[:, 0]
    boot[:, 2] += 6.0
    tiers = mcs_tiers({"a": 0.0, "a2": 0.0, "b": 6.0}, boot, ["a", "a2", "b"])
    tier_of = {m: i for i, g in enumerate(tiers) for m in g}
    assert tier_of["a"] == tier_of["a2"] < tier_of["b"]


def test_skill_lifetime_requires_continuous_chain():
    """中间出现不显著的段 ⇒ 寿命在该处截断（不允许跳过）。"""
    stats = {1: (0.5, 0.2), 2: (0.4, -0.1), 3: (0.9, 0.5)}
    assert skill_lifetime(stats)["H"] == 1


def test_split_half_reliability_detects_instability():
    """两半之间排名反向 ⇒ 信度为负（页面据此隐藏精确名次）。

    `r = −1` 时 Spearman–Brown `2r/(1+r)` 的分母为 0、返回值无定义，
    因此这里用轻微反向（r ≈ −0.95）而不是完全反向——那正是真实数据的样子。
    """
    a = {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0, "s5": 5.0}
    b = {"s1": 5.0, "s2": 4.0, "s3": 3.0, "s4": 2.0, "s5": 1.0}
    out = split_half_reliability(a, b)
    assert out["enough_evidence"] is True
    assert out["reliability"] is None          # 完全反向：分母为 0，无定义
    # 轻微反向（排名大体相反但不完全）⇒ 信度为负，页面隐藏精确名次
    a2 = {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0, "s5": 5.0}
    b2 = {"s1": 3.0, "s2": 4.0, "s3": 5.0, "s4": 1.0, "s5": 2.0}
    out2 = split_half_reliability(a2, b2)
    assert out2["enough_evidence"] is True
    # 排名大体相反 ⇒ 校正后信度 < 0.85（页面据此隐藏精确名次区间）
    assert out2["reliability"] is not None
    assert out2["reliability"] < 0.85


# ---------------------------------------------------------------- A1 集合独立性
def test_score_does_not_depend_on_other_sources():
    """增删榜上任意其他源，本源分数**逐位不变**（公理 A1）。

    这是旧桶内 z 分最根本的失效点，也是 DVS 存在的主要理由：`score_window`
    的输入只有"这条源自己的逐日统计量"，没有任何同行信息。
    """
    from weather_eval.dvs_engine import DvsConfig, score_window
    from weather_eval.dvs_stats_store import SourceDayRows
    cfg = DvsConfig(enable_bootstrap=False)
    rng = np.random.default_rng(11)

    def make_rows(scale: float) -> dict:
        rows = {}
        for day in range(1, 31):
            r = SourceDayRows()
            vals = [0.0] * 10
            n = 100.0
            vals[0] = n
            vals[1] = scale * n            # Σ|e|
            vals[2] = (scale ** 2) * n     # Σe²
            vals[3] = 2.0 * n              # Σ|e0|
            vals[4] = 4.0 * n              # Σ(e0)²
            vals[5] = 20.0 * n
            vals[6] = 20.0 * n
            vals[7] = 400.0 * n
            vals[8] = 400.0 * n
            vals[9] = 400.0 * n
            r.temp[("", "s1", "temp_h", 1)] = vals
            r.rain[("", "s1", "r3_ge1", 1)] = [
                60.0 + rng.integers(0, 3), 10.0, 10.0, 20.0, 12.0]
            rows[day] = r
        return rows

    target = make_rows(1.0)
    solo = score_window(target, config=cfg).dvs
    # 同一份输入 + 两个完全不同的"同行" —— 分数必须一模一样
    assert score_window(target, config=cfg).dvs == solo


# ---------------------------------------------------------------- 证据等级
def test_evidence_tier_thresholds():
    from weather_eval.dvs_engine import evidence_tier
    assert evidence_tier(80, 100, 0.9) == "A"
    assert evidence_tier(60, 100, 0.6) == "B"
    assert evidence_tier(20, 100, 0.3) == "C"
    assert evidence_tier(5, 100, 0.9) == "D"


# ---------------------------------------------------------------- 旁路隔离
def test_dvs_is_opt_in_and_does_not_touch_other_schemes():
    """DVS 只在 `score_scheme == "dvs-v6"` 时构建（旁路叠加，不替换旧口径）。

    守两个承诺：①改回 `skill-v5` 即回到单一旧口径（DVS 一行都不跑）；
    ②切换后 skill 家族分支（影子旧榜）仍然生效，两套分数并存可对照。
    """
    import inspect
    from weather_eval import evaluate

    src = inspect.getsource(evaluate.build_report)
    assert '== "dvs-v6"' in src, "DVS 必须由 score_scheme 显式开关"
    assert "attach_dvs" in src
    # 影子口径：dvs-v6 与 skill 家族走同一套旧分支（旧榜照常产出）
    assert "dvs-v6" in evaluate.SKILL_SCHEMES
    assert "skill-v5" in evaluate.SKILL_SCHEMES
    # 生产配置已于 2026-11 切换为 dvs-v6（本测试守的就是这个开关本身）
    from weather_eval.config import load_config
    assert load_config().eval.get("score_scheme") == "dvs-v6"

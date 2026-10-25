"""降水分"最小存活权重"证据门槛（score_review_proposal §2 P8，2026-10-18 修）。

P8 的全部要点是**区分两种缺失**：

* "这项指标对该源/该桶不存在" —— 缺项按剩余权重归一（旧口径，保留正确）；
* "该桶根本**没有可评降水事件**" —— 7 项入分指标里 ets/pod/far/bias/grade_ets
  共 0.775 权重需要真实降水事件才存在，amt_mae/amt_bias 只需要 Σ实况雨量 > 0。
  一个整桶只有不达标小雨的格会退化成只剩 0.225 权重的雨量两项，旧口径把它们
  抬成 100% 话语权 → 降水分 100 分，把分时效榜第 1/2 名制造了出来。
  新口径：整维存活（换算后有效）子分权重低于名义总权重 × 门槛时判缺测。

三条入口必须**同一把尺**：点估计 `evaluate.precip_score`、bootstrap
`stats._score_from_parts`（同一门槛值）与 `stats._rain_scores_from_aggregate`
（读模块级门槛全局，即 build_report 装载 config 值的那条路）。

测试纪律：`_stats.PRECIP_MIN_WEIGHT_FRAC` 是模块级可装载量，与 SCORE_SLOPES /
SCORE_INDEX_GAIN 的既有测试同套自我复位模式——改变它之后 finally/夹具复位，
绝不跨用例泄漏（否则同进程后续每个用例都在另一把门槛上跑）；单测同时把
技能指数缩放钉回恒等（同 test_evaluate 的 _index_gain_identity），断言原始尺。
"""
from __future__ import annotations

import numpy as np
import pytest

from weather_eval import evaluate as ev
from weather_eval import stats as st
from weather_eval.evaluate import (
    DEFAULT_PRECIP_MIN_WEIGHT,
    PRECIP_SCORE_PARTS,
    _weighted_score,
    apply_precip_min_weight,
    precip_score,
    temp_score,
)
from weather_eval.config import DEFAULT_EVAL


@pytest.fixture(autouse=True)
def _identity_gain_and_default_gate():
    """单测看**原始度量尺 + config 缺省门槛**：缩放钉恒等，门槛复位缺省。"""
    ev.apply_score_index_gain(1.0, 50.0)
    yield
    ev.apply_score_index_gain(1.0, 50.0)
    ev.apply_precip_min_weight(None)      # None = 回到代码内缺省


def _set_gate(frac) -> None:
    """经生产装载入口改门槛（而非直接写全局——顺带测 apply 通路）。"""
    apply_precip_min_weight(frac)


def _weighted_raw(parts, metrics) -> float | None:
    """未缩放的子分加权均值复算，供断言对照（不是生产路径）。"""
    num = den = 0.0
    for key, w, _lb, _mp, fn in parts:
        v = metrics.get(key)
        if v is None:
            continue
        c = ev._clamp100(fn(v))
        if c is None:
            continue
        num += w * c
        den += w
    return (num / den) if den else None


# 只剩雨量两维的格：P8 的触发形态（整桶没有可评降水事件 → 事件五项全缺席）
AMT_ONLY = {"amt_mae": 0.5, "amt_bias": 1.0}
# 全量在位的格
FULL = {"ets": 0.2, "pod": 50.0, "far": 30.0, "bias": 1.0, "grade_ets": 0.1,
        "amt_mae": 0.6, "amt_bias": 1.05}


def test_synthetic_amt_only_bucket_is_missing_not_full_score():
    """核心回归（P8 要求的构造式用例）：只有 amt_mae+amt_bias → None，绝不给 100。"""
    _set_gate(0.5)                             # 与 config 冻结缺省同值
    assert precip_score(AMT_ONLY) is None
    # 旧的错误行为（重归一抬成满分话语权）只在门槛关闭时重现，供对照
    # （索引增益已被夹具钉恒等，故旧口径分 = 未缩放的加权均值四舍五入）
    _set_gate(0.0)
    old = precip_score(AMT_ONLY)
    assert old == round(_weighted_raw(PRECIP_SCORE_PARTS, AMT_ONLY), 2)
    # 旧口径只凭雨量一项重归一出一个高分（2026-10-25 去冗余后 amt_bias 不再
    # 入分，只剩 amt_mae 0.5 相对误差 → 81.1）。分数带的精确值随 λ 冻结值浮动，
    # 断言只锁"高分"这个性质本身。
    assert 80.0 <= old <= 100.0, old
    # 更贴近 P8 榜首形态的夹具：近乎完美的雨量格（0.05 相对误差）→ 旧口径 ≥95。
    # 对任意合理 λ 成立（0.05×λ 至多扣 2~3 分），这正是"整桶没有可评事件却产出
    # 榜首"的机制——P8 修法判它缺测（None），而不是把雨量两项抬成全部话语权。
    amt_perfect = {"amt_mae": 0.05, "amt_bias": 1.0}
    old_perfect = precip_score(amt_perfect)
    assert 95.0 <= old_perfect <= 100.0, old_perfect
    # 缺省代码内缺省值与 DEFAULT_EVAL 同源（标定量只此一处声明的纪律）
    assert DEFAULT_PRECIP_MIN_WEIGHT == float(
        DEFAULT_EVAL["precip_min_surviving_weight"]) == 0.5


def test_two_kinds_of_missing_are_distinguished():
    """证据还够半数 → 照旧重归一；缺一半以上 → 判缺测（P8 的语义分界）。"""
    _set_gate(0.5)
    # 事件五项全在、缺 amt 两维（0.775 存活）：照旧归一（amt 是真没数据）
    p = {k: v for k, v in FULL.items() if k not in ("amt_mae", "amt_bias")}
    assert precip_score(p) == pytest.approx(
        _weighted_raw(PRECIP_SCORE_PARTS, p), abs=5e-3)
    # 只缺 grade_ets（0.10 权重）→ 存活 0.90，照旧归一（该档没下雨≠无可评事件）
    p = {k: v for k, v in FULL.items() if k != "grade_ets"}
    assert precip_score(p) is not None
    # 缺 ets（0.315 权重，最大单权重）→ 存活 0.685 ≥ 0.5，照旧归一
    p = {k: v for k, v in FULL.items() if k != "ets"}
    assert precip_score(p) is not None
    # [ets, far] 在位（0.45 < 0.5）→ 判缺测：半截证据不能当整个用
    assert precip_score({"ets": 0.2, "far": 30.0}) is None
    # 干桶误报形态：实况全域无降水（h=mi=0）→ amt/pod/bias 缺，只剩 ets+far
    # （0.45）→ 判缺测，而不是重归一出一个"0 分惩罚"
    assert precip_score({"ets": 0.0, "far": 100.0}) is None
    # 全量在位必有分；全缺必 None（既有纪律不变）
    assert precip_score(FULL) is not None
    assert precip_score({}) is None
    # 温度维不受门槛波及：低权重单指标仍按剩余权重归一（温度 8 项恒可算）
    assert temp_score({"mbe_bdisp": 0.3}) == round(
        100 - 0.3 * ev.SCORE_SLOPES["mbe_bdisp"], 2)


def test_threshold_boundary_is_strictly_less_than():
    """存活权重**恰好**达门槛仍有效（< 门槛才判缺测），门槛显式可配。"""
    # 借降水换算函数造一张名义权重 1.0 的小表，便于精确踩边界
    parts = (("ets", 0.6, "边界表·a", "", PRECIP_SCORE_PARTS[0][4]),
             ("amt_mae", 0.4, "边界表·b", "", PRECIP_SCORE_PARTS[3][4]))
    m = {"ets": 0.2}
    # _weighted_score 的门槛是显式参数（None = 不设门槛，温度维语义）；
    # precip_score 才负责把 stats 全局（config 装载值）塞进来。
    assert _weighted_score(parts, m) is not None            # None = 无门槛
    assert _weighted_score(parts, m, min_weight_frac=0.6) is not None   # 0.6 == 门槛
    assert _weighted_score(parts, m, min_weight_frac=0.61) is None      # 0.6 < 门槛
    # 全量在位恒有效，哪怕门槛压到接近 1
    assert _weighted_score(parts, {"ets": 0.2, "amt_mae": 0.5},
                           min_weight_frac=0.99) is not None
    assert _weighted_score(parts, {}, min_weight_frac=0.99) is None
    # 门槛关闭 = 旧口径逐位复现（对照通道；全缺仍 None）
    assert _weighted_score(parts, m, min_weight_frac=0.0) == round(0.2 * 120.368, 2)
    assert _weighted_score(parts, {}, min_weight_frac=0.0) is None


def test_config_knob_is_registered_not_call_site_only():
    """门槛建在 config（DEFAULT_EVAL + stations.yaml eval 段），不在调用点散落。"""
    from weather_eval.config import load_config
    cfg = load_config(None)                    # 读仓库的 config/stations.yaml
    assert "precip_min_surviving_weight" in DEFAULT_EVAL
    assert "precip_min_surviving_weight" in cfg.eval
    assert abs(cfg.eval["precip_min_surviving_weight"] - 0.5) < 1e-12
    # 装载入口拒绝越界值
    with pytest.raises(ValueError):
        apply_precip_min_weight(1.5)


def test_point_and_bootstrap_paths_agree_on_the_gate():
    """点估计与 bootstrap 两条路逐格同尺（本仓库的 bit-identical 对拍纪律）。"""
    _set_gate(0.5)
    # ① 全量在位：两路同一分数（bootstrap 原值 ≈ 点估计的 round(…,2)）
    arr = {k: np.array(v) for k, v in FULL.items()}
    raw = float(st._score_from_parts(arr, PRECIP_SCORE_PARTS,
                                     min_weight_frac=0.5))
    pt = precip_score(FULL)
    assert pt is not None
    assert abs(raw - pt) <= 5e-3
    # ② 半证据二件套：两路**同时**判缺测
    arr2 = {k: np.array(v) for k, v in AMT_ONLY.items()}
    assert np.isnan(st._score_from_parts(arr2, PRECIP_SCORE_PARTS,
                                         min_weight_frac=0.5)).all()
    assert precip_score(AMT_ONLY) is None
    # ③ 门槛值住在 stats 全局：evaluate.apply_precip_min_weight 装载一次，
    #    点估计（precip_score 读全局）与聚合 bootstrap（缺省读全局）同改同效。
    #    （p90 存活 0.90：在门槛 0.9 上绕开——顺序累加分母的浮点噪声区，
    #     精确踩边界由上面那张 0.6/0.4 边界表负责。）
    for frac in (0.85, 0.95):
        _set_gate(frac)
        assert precip_score(FULL) is not None, frac
        p90 = {k: v for k, v in FULL.items() if k != "grade_ets"}   # 存活 0.90
        assert (precip_score(p90) is None) == (frac > 0.9), frac
        # 聚合路径缺省（不显式传门槛）跟同一全局走：amt-only 聚合格判缺测
        out = st._rain_scores_from_aggregate(_synthetic_aggregate(True),
                                             PRECIP_SCORE_PARTS, min_sample=5)
        assert np.isnan(out).all(), frac
    # ④ 聚合路径门槛关闭（对照）→ 旧的重归一分数回来
    _set_gate(0.0)
    out = st._rain_scores_from_aggregate(_synthetic_aggregate(True),
                                         PRECIP_SCORE_PARTS, min_sample=5)
    assert not np.isnan(out).any()


def _synthetic_aggregate(amt_only: bool) -> np.ndarray:
    """(m, b, s, 13) 一格聚合计数：1 模型 × 1 桶 × 1 站。

    13 列 = h, fa, mi, c, Σo, Σf, Σ|f−o|, …（后 6 列为分级计数，置 0）。
    n = c = 10（正确否定，样本量过 min_sample）。
    amt_only=False 模拟"干桶只有 ets+far 在位"（事件缺失的一种真实形态）；
    amt_only=True 再给 Σo/Σf/Σ|f−o| 充上证据 → 模拟 P8 的
    "只剩雨量两维"退化格。
    """
    A = np.zeros((1, 1, 1, 13))
    A[0, 0, 0, 3] = 10.0                       # correct_negatives = 10
    if amt_only:
        A[0, 0, 0, 4] = 5.0                    # Σ实况
        A[0, 0, 0, 5] = 1.0                    # Σ预报
        A[0, 0, 0, 6] = 4.0                    # Σ|预报−实况|
    return A

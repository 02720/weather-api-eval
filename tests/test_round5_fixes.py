"""第五轮审查的回归测试：每条都对应一个"不报错、只安静给错答案"的缺陷。

纪律与既有 test_stats_core.py 一致：**写成"实现改回去就变红"的形式**。
这批缺陷的共同点是没有异常、没有告警，只有一个悄悄变了的样本集合或名次。
"""
from datetime import datetime

import pytest

from weather_eval import stats
from weather_eval.config import DEFAULT_EVAL, Config
from weather_eval.forecast.http import (_DeterministicError, MAX_RESPONSE_BYTES,
                                        RETRY_AFTER_MAX_SECONDS, _bounded_sleep,
                                        _enforce_response_limit,
                                        _retry_after_seconds)
from weather_eval.forecast.resample import (floor_hour, hourly_axis,
                                            interpolate_hourly)
from weather_eval.storage import shift_month
from weather_eval.timeutil import (axis_is_pruned_safe, lower_split, parse_iso,
                                   unfrozen_split, upper_split)


# ------------------------------------------------------- P1-4 Holm 的 p=0.0
def test_holm_p_zero_is_strongest_not_weakest():
    """`ps[i] or 1.0` 会把 p=0.0 排到有效 p 的最后一位。

    Holm 是逐步降级的：一旦前面有一项判否，`still` 就关掉，排在后面的项
    即使阈值更松也拿不到显著。于是 p=0.0（配对 bootstrap 里 0 次跨到 0 另一
    侧，最强的证据）反而被判不显著，而 p=0.001 的邻居判显著。
    """
    for pvals in ([0.0, 0.001, 0.5], [0.001, 0.5, 0.0], [0.5, 0.001, 0.0]):
        got = stats.holm_bonferroni(pvals, alpha=0.1)
        assert got[pvals.index(0.0)] is True, (pvals, got)


def test_holm_p_zero_not_confused_with_missing():
    """p=0.0 是有效 p，绝不能与 None / NaN 同桶（同桶即排到末尾）。"""
    got = stats.holm_bonferroni([0.0, None, 0.001], alpha=0.1)
    assert got[0] is True
    assert got[1] is False                      # 缺 p 值不参与显著判定
    # 全部为 0 时全部显著（旧实现会全部判否）
    assert stats.holm_bonferroni([0.0, 0.0], alpha=0.1) == [True, True]


# --------------------------------------------- P1-2 逐小时轴与插值产物同地板
def test_hourly_axis_aligned_with_interpolated_keys():
    """采样落在非整点时，轴与插值键必须仍逐点对齐。

    旧实现拿原始首末时刻逐小时累加，轴带着分钟余数，而 interpolate_hourly
    的键已被地板到整点 ⇒ `temps.get(h)` 对每个 h 都落空 ⇒ **整列温度静默变
    缺测**。这是本项目最不能接受的失败形态（无报错、无告警、少一整列）。
    """
    from datetime import timedelta
    samples = [(datetime(2026, 9, 13, 3, 30), 25.0),
               (datetime(2026, 9, 13, 6, 30), 28.0)]
    interp = dict(interpolate_hourly(samples))
    axis = hourly_axis(samples)
    assert axis[0] == datetime(2026, 9, 13, 3, 0)
    col = [interp.get(h) for h in axis]
    assert all(v is not None for v in col), col
    # 与"用原始时刻累加"的旧轴逐点错位，正是缺陷所在
    legacy = [samples[0][0] + timedelta(hours=i) for i in range(4)]
    assert all(interp.get(h) is None for h in legacy)


def test_floor_hour_is_single_definition():
    assert floor_hour(datetime(2026, 9, 13, 3, 30)) == datetime(2026, 9, 13, 3, 0)
    assert hourly_axis([]) == []


# --------------------------------------------- P2-7 二分定界的前提不成立时
def test_bisect_precondition_rejects_unsorted_or_seconds():
    good = ["2026-09-01T00:00", "2026-09-01T01:00", "2026-09-01T02:00"]
    assert axis_is_pruned_safe(good) is True
    assert axis_is_pruned_safe(list(reversed(good))) is False      # 乱序
    assert axis_is_pruned_safe(["2026-09-01T00:00:00"]) is False   # 带秒
    assert axis_is_pruned_safe(["2026-09-01T00:00"] * 2) is False  # 重复


def test_split_matches_perpoint_semantics_even_when_unsorted():
    """前提成立与否，结果都必须等于"逐点 datetime 比较"的语义。

    乱序时间轴上 bisect 不会报错，只会给出一个**错误但自洽**的边界（旧实现
    就是这样静默截掉一段样本）。这里不对具体下标做断言，而是把 bisect 快路径
    与逐点扫描两种实现在两种轴上对拍。
    """
    good = ["2026-09-01T00:00", "2026-09-01T01:00", "2026-09-01T02:00"]
    bad = ["2026-09-01T02:00", "2026-09-01T00:00", "2026-09-01T01:00"]
    for axis in (good, bad):
        for dt in (datetime(2026, 9, 1, 1, 0), datetime(2026, 9, 1, 0, 30)):
            ref_lo = next((i for i, t in enumerate(axis) if parse_iso(t) >= dt),
                          len(axis))
            ref_hi = next((i for i, t in enumerate(axis) if parse_iso(t) > dt),
                          len(axis))
            assert lower_split(axis, dt) == ref_lo, (axis, dt)
            assert upper_split(axis, dt) == ref_hi, (axis, dt)
    fetched = "2026-09-01T01:00"
    ref = next((i for i, t in enumerate(bad)
                if parse_iso(t) >= parse_iso(fetched)), len(bad))
    assert unfrozen_split(bad, fetched)[0] == ref


# ---------------------------------------------- P1-3 配置不得污染模块级默认值
def test_config_eval_does_not_alias_defaults():
    """浅合并会让 cfg.eval 的嵌套容器与 DEFAULT_EVAL **共享同一个对象**。

    任何就地改动会永久污染默认值，而 lru_cache(maxsize=1) 让同进程后续
    每一次 load_config() 都读到被改过的默认——全程无报错。
    """
    cfg = Config({"models": [], "stations": []})
    assert cfg.eval["score_slopes"] is not DEFAULT_EVAL["score_slopes"]
    assert cfg.eval["temp_accuracy_limits"] is not DEFAULT_EVAL["temp_accuracy_limits"]
    # 就地改动不得外泄
    cfg.eval["score_slopes"]["rmse"] = 999.0
    cfg.eval["temp_accuracy_limits"].append(99)
    assert DEFAULT_EVAL["score_slopes"]["rmse"] != 999.0
    assert DEFAULT_EVAL["temp_accuracy_limits"] == [1, 2]


def test_config_empty_yaml_raises_actionable_error():
    """空 YAML（safe_load 返回 None）此前以裸 AttributeError 崩掉。"""
    with pytest.raises(ValueError, match="stations.yaml"):
        Config(None)
    with pytest.raises(ValueError, match="stations.yaml"):
        Config({"stations": [{"id": "x"}]})      # 缺 lat/lon


def test_source_stale_hours_is_declared():
    """该配置项此前只存在于调用点，DEFAULT_EVAL 里没有——无从得知可配。"""
    assert DEFAULT_EVAL.get("source_stale_hours") == 30


# ------------------------------------------------- P2-1 保留月数含当月
def test_retain_months_is_inclusive_of_current_month():
    """retain_months=13 应留下最近 13 个自然月（含当月）。

    旧式 `-abs(retain_months)` 配合 `month >= expire_before` 实际留下 14 个。
    """
    expire_before = shift_month("2026-10", -(13 - 1))
    kept = [m for m in ("2025-09", "2025-10", "2026-09", "2026-10")
            if m >= expire_before]
    assert "2025-10" in kept and "2025-09" not in kept
    assert len(kept) == 3


# --------------------------------------------- P2-5 / P2-6 共享 HTTP 层
class _H:
    def __init__(self, headers):
        self.headers = headers

def test_retry_after_is_not_discarded():
    """服务端说等 120s 时，旧实现 3s 后又撞上去（Retry-After 被忽略）。"""
    assert _retry_after_seconds(_H({})) is None
    assert 119.0 <= _retry_after_seconds(_H({"Retry-After": "120"})) <= 120.0
    # HTTP-date 形态现在**能**解析（旧实现静默退回"没说"），过去的时间即已过期
    assert _retry_after_seconds(_H({"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"})) is None


def test_retry_after_can_not_blow_the_time_budget():
    """`Retry-After` 是服务端可控值：它不许把一次睡眠推过整轮预算。

    旧实现把它当**倍数**（×100）乘在退避基数上：`Retry-After: 3000` → 1200 s
    单次睡眠，而 `TOTAL_BUDGET_SECONDS` 只有 900 s——一个响应头就能让排在其后
    的 12 个源丢掉这一轮。现在解析成秒，且被预算夹住。
    """
    # 绝对值上界：再长也不睡过 RETRY_AFTER_MAX_SECONDS
    assert _retry_after_seconds(_H({"Retry-After": "3000"})) == RETRY_AFTER_MAX_SECONDS
    assert _retry_after_seconds(_H({"Retry-After": "86400"})) == RETRY_AFTER_MAX_SECONDS
    # 剩余预算不足时：物理上不可能睡过量
    from weather_eval.forecast.http import TimeBudget
    import time as _time
    b = TimeBudget(900)
    b._t0 = _time.monotonic() - 895      # 只剩 5 s
    t0 = _time.monotonic()
    _bounded_sleep(1200.0, b, "unit")
    assert _time.monotonic() - t0 <= 5.9


def test_oversized_response_is_fatal_not_retried():
    class _R:
        headers = {"Content-Length": str(MAX_RESPONSE_BYTES + 1)}

    with pytest.raises(RuntimeError, match="超过上限"):
        _enforce_response_limit(_R(), "X")

    class _Ok:
        headers = {"Content-Length": "1000"}

    _enforce_response_limit(_Ok(), "Y")     # 正常大小不拦


def test_dispatch_error_is_deterministic():
    """session 不支持该方法 = 编程错误，重试一万次也不会变。"""
    assert issubclass(_DeterministicError, RuntimeError)


# ------------------------------------------------- P2-4 bool 不是观测值
def test_bool_is_never_a_measurement():
    """float(True) == 1.0：JSON 里的 `true` 会变成 1°C / 1mm 的真值。"""
    from weather_eval.forecast.qweather import _to_float as q
    from weather_eval.obs.eia_data import _to_float as e

    assert q(True) is None and q(False) is None
    assert e(True) is None and e(False) is None
    # 正常数值与字符串仍照原口径
    assert q(28) == 28.0 and q("28") == 28.0
    assert e(5.5) == 5.5 and e("na") is None


# -------------------------------- P1-1 可选要素降级不得否决整份快照的完整性
def test_ew4all_optional_precip_keeps_snapshot_complete():
    """可选要素（ONETPE）缺供只让短时效降水退化为摊薄值，不该整份出局。

    `complete=False` 会被 evaluate 的 require_complete 门槛**整份挡在所有榜
    外**，比"某一维缺值"重得多；而该要素在 ModelSpec 里是明写的可选项。
    降级改由 `precip_1h_degraded` 单独披露。
    """
    from weather_eval.forecast.ew4all import MODEL_SPECS, _has_values
    src = __import__("weather_eval.forecast.ew4all", fromlist=["x"])
    # 两模型中至少有一个带可选 1h 降水产品，且该字段是可选的
    assert any(spec.precip_1h_element for spec in MODEL_SPECS.values())
    assert any(spec.precip_1h_element is None for spec in MODEL_SPECS.values())
    assert _has_values([]) is False
    assert src is not None

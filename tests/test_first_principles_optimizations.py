"""第一性原理优化轮（2026-11）的回归锁。

这一轮改的不是评分公式，而是**让公式能被信任的前提**：缓存是否可能串味、
源码指纹是否会被运行时全局污染、分层边界是否守住缺测语义。
每条测试都对应一个已复现的静默失败，写成"改回旧实现就变红"的形式。
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------- 1. 身份缓存不得被地址复用骗过
def test_axis_cache_rejects_recycled_object_id():
    """`id()` 只在对象存活时唯一；旧实现只存 bool，跨构建会读到失效结论。

    复现：一条升序轴缓存 True → 释放 → 新建一条乱序轴恰好落在同一地址。
    旧实现返回 True，bisect 于是给出**静默错误**的窗口边界（样本被截断而
    报告毫无异常）。缓存持有原对象并核对身份后，这类串味在物理上不可能发生。
    """
    from weather_eval import timeutil as tu

    tu.clear_axis_cache()
    good = ["2026-01-01T00:00", "2026-01-01T01:00"]
    assert tu.axis_is_pruned_safe(good) is True
    # 缓存持有原对象：地址在原对象存活期间不可能被复用。
    hit = tu._AXIS_SAFE_CACHE[id(good)]
    assert hit[0] is good and hit[1] is True
    # 手工把"同地址、不同对象"的陈旧条目塞回去（模拟 id 回收复用）：
    # 身份核对必须拒绝命中，重新扫描内容而不是返回上一轮的 True。
    bad = ["2026-01-01T01:00", "2026-01-01T00:00"]
    tu._AXIS_SAFE_CACHE[id(good)] = ({"not": "the same object"}, True)
    assert tu.axis_is_pruned_safe(good) is True
    assert tu.axis_is_pruned_safe(bad) is False
    assert tu.axis_is_pruned_safe(bad) == tu._scan_axis_safe(bad)


def test_snapshot_sanitize_cache_keyed_by_identity(tmp_path, monkeypatch):
    """`evaluate._snaps_for` 的清洗缓存不得把上一轮释放的 dict 当成这一轮的。"""
    import weather_eval.evaluate as ev
    from weather_eval import storage

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    ev.clear_preload_cache()
    snap = {"issue_iso": "2026-08-01T00:00", "station_id": "s1",
            "source": "test", "models": ["m"], "hourly_time": ["2026-08-01T00:00"],
            "data": {"m": {"temperature_2m": [20.0], "precipitation": [0.0]}}}
    storage.save_forecast_snapshot("s1", "m", snap)
    snaps = {("s1", "m"): ev.list_forecast_snapshots("s1", "m")}
    ev._SNAP_SANITIZE_CACHE.clear()
    assert ev._snaps_for("s1", "m", snaps, require_complete=False)
    key = next(iter(ev._SNAP_SANITIZE_CACHE))
    # 手工构造一个"同地址、不同对象"的陈旧条目：身份核对必须拒绝命中。
    ev._SNAP_SANITIZE_CACHE[key] = ({"not": "the same object"}, False)
    assert ev._snaps_for("s1", "m", snaps, require_complete=False), \
        "陈旧身份被误当成有效缓存，快照会被静默排除"


def test_integrity_cache_rejects_recycled_id_tuple():
    """Merkle 根缓存必须核对整批对象身份，不能只看 id 元组。"""
    import weather_eval.evaluate as ev

    ev._INTEGRITY_CACHE.clear()
    a = {"issue_iso": "2026-01-01T00:00", "station_id": "s"}
    b = {"issue_iso": "2026-01-02T00:00", "station_id": "s"}
    key = (id(a), id(b))
    ev._INTEGRITY_CACHE[key] = ((a, {"wrong": "batch"}),
                                {"merkle_root": "STALE"})
    # 同地址、不同对象：必须重算，绝不返回 STALE
    out = ev._integrity_summary_cached([a, {"issue_iso": "2026-01-03T00:00"}])
    assert out.get("merkle_root") != "STALE"
    ev._INTEGRITY_CACHE.clear()


# ---------------------------------------------------- 2. 指纹稳定性
def test_fingerprint_is_independent_of_runtime_score_globals():
    """分块源码指纹不得随 apply_score_* 装载而变（否则历史块每轮重算）。"""
    import weather_eval.evaluate as ev
    from weather_eval.config import load_config

    cfg = load_config(None)
    before = ev._block_scoring_fingerprint(dict(cfg.eval), cfg.models,
                                           cfg.station_ids)
    try:
        ev.apply_score_scheme("skill-v5")
        ev.apply_score_composite_params(0.8, 9.0, 4.0, 20.0, 2.0, 3.0,
                                        {"1-3": 1.0})
        after = ev._block_scoring_fingerprint(dict(cfg.eval), cfg.models,
                                              cfg.station_ids)
        assert after == before
    finally:
        # 这些是模块级运行时全局：测试结束后必须全部还原，否则同进程后续用例
        # 会在另一把尺上跑（test_soft_skill_is_c1_smooth_at_the_anchor 的红灯
        # 就是这类状态泄漏，不是评分公式回归）。
        ev.apply_score_scheme(None)
        ev.apply_score_composite_params(
            None, None, None,
            __import__("weather_eval.config", fromlist=["DEFAULT_EVAL"])
            .DEFAULT_EVAL["score_rank_scale"],
            1.0, None, None)


# ---------------------------------------------------- 3. 分层边界的缺测语义
def test_r_slope_numpy_inf_is_quiet_and_missing():
    """inf 输入与 cyeva/scipy 同口径（NaN → 上层归 None），且不刷 RuntimeWarning。"""
    import warnings

    from weather_eval.stats import r_slope_numpy

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r, slope = r_slope_numpy(np.array([1.0, 2.0, 3.0, float("inf")]),
                                 np.array([1.0, 2.0, 3.0, 4.0]))
    assert r is not None and np.isnan(r)
    assert slope is not None and np.isnan(slope)
    assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]


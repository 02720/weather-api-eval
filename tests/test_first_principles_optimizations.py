"""第一性原理优化轮（2026-11）的回归锁。

这一轮改的不是评分公式，而是**让公式能被信任的前提**：缓存是否可能串味、
配置是否真的接线、分层边界是否守住缺测语义、生产口径是否真的被读者看见。
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


# ---------------------------------------------------- 2. 声明过的配置必须真的接线
def test_default_dvs_is_reachable_from_config():
    """文档说"缺项回落 DEFAULT_DVS"，旧实现里它只是个没人读的孤立常量。"""
    from weather_eval.config import DEFAULT_DVS, Config

    cfg = Config({"models": [], "stations": []})
    node = cfg.eval["dvs"]
    assert node == DEFAULT_DVS
    assert node is not DEFAULT_DVS             # 仍然不得共享可变对象
    node["alpha_v"] = 0.99
    assert DEFAULT_DVS["alpha_v"] != 0.99


def test_user_dvs_section_merges_with_defaults():
    """YAML 只写几个参数时，其余必须回落声明默认值而不是整段消失。"""
    from weather_eval.config import DEFAULT_DVS, Config

    cfg = Config({"models": [], "stations": [],
                  "eval": {"dvs": {"alpha_v": 0.7}}})
    assert cfg.eval["dvs"]["alpha_v"] == 0.7
    for key in ("rain_events", "scale_weights", "climatology_path",
                "cost_loss_prior", "bootstrap_runs"):
        assert cfg.eval["dvs"][key] == DEFAULT_DVS[key], key


def test_configured_cost_loss_prior_changes_score():
    """Beta 先验/格点数是**决策偏好**，改了必须改变经济价值分。

    旧实现把它们写进 `meta.dvs.config` 却始终用模块级 ALPHAS 算分：仪表盘
    说一套、分数算一套。生产值 (2,5)/20 与模块缺省逐位相同，所以平时看不出来；
    这条测试用另一套偏好把"没接线"逼出来。
    """
    from weather_eval import dvs

    counts = dvs.RainCounts(
        a=np.array([60]), b=np.array([10]), c=np.array([10]),
        d=np.array([20]), n_ev_days=2)
    base = dvs.economic_value(counts)
    alt = dvs.economic_value(counts, cost_loss_prior=(8.0, 2.0),
                             cost_loss_grid=80)
    assert alt["v_plus"] != base["v_plus"]
    assert alt["score"] != base["score"]
    # 显式传生产缺省必须与不传逐位相同（不破坏现有 golden 口径）
    same = dvs.economic_value(counts, cost_loss_prior=tuple(dvs.COST_LOSS_PRIOR),
                              cost_loss_grid=dvs.COST_LOSS_GRID)
    assert same == base


def test_fingerprint_is_independent_of_runtime_score_globals():
    """分块源码指纹不得随 apply_score_* 装载而变（否则历史块每轮重算）。"""
    import weather_eval.evaluate as ev
    from weather_eval.config import load_config

    cfg = load_config(None)
    before = ev._block_scoring_fingerprint(dict(cfg.eval), cfg.models,
                                           cfg.station_ids)
    try:
        ev.apply_score_scheme("dvs-v6")
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
def test_dvs_pick_treats_illegal_values_as_missing():
    """DVS 直读原始存档：字符串/哨兵必须记 NaN，绝不能炸掉整条旁路。"""
    from weather_eval.dvs_protocol import (_FCST_RAIN_RANGE,
                                           _FCST_TEMP_RANGE, _pick)

    idx = np.arange(6)
    temp = _pick([20.0, "x", None, 999.0, float("inf"), -95.0],
                 idx, 6, *_FCST_TEMP_RANGE)
    rain = _pick([0.0, 9999.0, -0.1, 1.5, None, 2.0],
                 idx, 6, *_FCST_RAIN_RANGE)
    assert temp[0] == 20.0 and np.isnan(temp[1:]).all()
    assert rain[0] == 0.0 and rain[3] == 1.5 and rain[5] == 2.0
    assert np.isnan(rain[[1, 2, 4]]).all()


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


# ---------------------------------------------------- 4. 生产口径必须真的发布
def test_dvs_board_is_server_rendered():
    """dvs-v6 的榜必须出现在 HTML 正文（服务端表格），不能只塞进内联 JSON。"""
    from weather_eval.report.render import render_report_html

    dvs_node = {
        "available": True, "scheme": "dvs-v6", "scoring_doc": "SCORING_DVS.md",
        "l1": "ecmwf_ifs025", "window": {"days": 82},
        "board": [{"model": "ecmwf_ifs025", "score": 27.87,
                   "temp": {1: 0.4}, "precip": {1: 0.25}, "cover": 27.87,
                   "evidence": "A", "days": 80, "window_days": 82,
                   "avail": 0.98, "ci90": [20.8, 32.1], "rank": [1, 5],
                   "p_best": 0.4, "mcs_tier": 5, "delta_l1": 3.2,
                   "aligned": None, "lifetime": {"H": 4},
                   "recent": 29.5, "delivered": 27.5,
                   "control": False, "errors": [], "detail": {}}],
        "controls": [{"model": "ctl:l0", "label": "L0（无技巧基准）",
                      "note": "量表零点", "score": 0.0}],
        "tiers": [["ecmwf_ifs025"]], "config": {"scheme": "dvs-v6"},
        "stations": {},
    }
    report = {
        "meta": {"score_scheme": "dvs-v6", "models": ["ecmwf_ifs025"],
                 "stations": ["s1"], "period_label": "test",
                 "is_monthly": False, "dvs": dvs_node,
                 "bootstrap_days": 82, "bootstrap_runs": 200},
        "leaderboards": {}, "scorecard": {}, "score_trend": {},
        "heatmap": [], "timeseries": {}, "per_station": {},
        # 模板的"读榜须知"与页脚无条件读取这些字段：合成夹具必须完整，
        # 否则测到的是 Jinja UndefinedError 而不是 DVS 榜有没有渲染。
        "coverage": {"first_obs": "2026-08-01", "last_obs": "2026-11-05"},
        "weight_sensitivity": {}, "difficulty_window": {}, "integrity": {},
        "provenance": {}, "diagnostics": {}, "daily_source_fallback": False,
        "require_complete_snapshots": True, "temp_hourly": {},
        "precip_hourly": {}, "temp_daily": {}, "precip_daily": {},
    }
    html = render_report_html(report)
    assert 'id="dvs-board"' in html
    assert 'id="dvsTable"' in html
    assert "ecmwf_ifs025" in html and "27.87" in html
    assert "L0（无技巧基准）" in html
    assert "skill-v5 影子口径" in html
    # 不可用时必须显式披露，而不是静默少一整节
    report["meta"]["dvs"] = {"available": False, "scheme": "dvs-v6",
                             "reason": "ValueError: boom"}
    html2 = render_report_html(report)
    assert "增量决策价值榜暂不可用" in html2 and "ValueError: boom" in html2

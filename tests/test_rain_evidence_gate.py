"""降水事件证据门（2026-10-23，score_review_proposal §2 P8 第 3 条的落地）。

n_eff 数的是"雨/无雨判定"，干燥小时占绝对多数：一个只覆盖 11 天、只有 14 个
≥1mm/h 观测事件的源，仍能凭 1,068 个"判定"通过 min_board_neff=30。事件门直接数
独立（站, 时刻）/（站, 日）事件，同一场雨被多轮起报覆盖时只算一次。

本测试锁三件事：
  * 计数按（站, 时刻）/（站, 日）去重——多轮起报不得把证据量乘起来；
  * 未达标原因的文案给出具体数字（cma_gowfs 形态：14 事件 / 6 雨日）；
  * 两个阈值在 config 的 DEFAULT_EVAL 与 stations.yaml 中都可配置且同源。
"""
from __future__ import annotations

from datetime import datetime, timedelta

from weather_eval import storage
from weather_eval.config import DEFAULT_EVAL, load_config
from weather_eval.evaluate import (_rain_evidence_by_model, _rain_evidence_note,
                                   build_report)
from weather_eval.timeutil import iso


def _h(station, valid, obs, fcst, bucket=1, model="m"):
    return {"station": station, "model": model, "valid_iso": valid,
            "bucket": bucket, "rain_obs": obs, "rain_fcst": fcst}


def _d(station, day, obs, fcst, model="m"):
    return {"station": station, "model": model, "valid_day": day,
            "rain_obs": obs, "rain_fcst": fcst}


def test_event_counts_are_independent_of_issue_frequency():
    """同一 (站, 时刻) 抓了多轮只算一个事件；阈值以下的样本不算。"""
    hourly = [
        _h("s1", "2026-10-01T00:00", 2.0, 0.0),
        _h("s1", "2026-10-01T00:00", 2.0, 0.0),      # 另一轮起报，重复
        _h("s1", "2026-10-01T01:00", 0.2, 1.5),      # 实况未达标、预报达标
        _h("s2", "2026-10-01T00:00", 0.0, 0.0),
        _h("s2", "2026-10-01T02:00", 1.0, 0.0),      # 恰好达标
    ]
    daily = [
        _d("s1", "2026-10-01", 3.0, 0.0),
        _d("s1", "2026-10-01", 3.0, 0.0),            # 另一轮起报，重复
        _d("s1", "2026-10-02", 0.5, 2.0),            # 实况未达标、预报达标
        _d("s2", "2026-10-01", 0.2, 0.0),
    ]
    ev = _rain_evidence_by_model(["m"], {"m": hourly}, {"m": daily}, 1.0, 1.0)["m"]
    assert ev["rain_events_h"] == 2          # s1 00:00 + s2 02:00（重复轮只算一次）
    assert ev["rain_events_fcst_h"] == 1     # s1 01:00
    assert ev["rain_days_d"] == 1            # s1 10-01
    assert ev["rain_days_fcst_d"] == 1       # s1 10-02


def test_shortfall_note_reports_the_actual_event_counts():
    """未达标原因必须带数字：读者能直接对照"样本积累中"的含金量。"""
    ev = {"rain_events_h": 14, "rain_days_d": 6}
    short, note = _rain_evidence_note("all", ev, 30, 8, 1.0, 1.0)
    assert short is True
    assert "14" in note and "≥1mm/h" in note
    assert "6" in note and "≥1mm" in note

    # 恰好达标不算不足（严格小于才失败）
    assert _rain_evidence_note("hourly", {"rain_events_h": 30, "rain_days_d": 0},
                               30, 8, 1.0, 1.0) == (False, "样本积累中")
    assert _rain_evidence_note("daily", {"rain_events_h": 0, "rain_days_d": 8},
                               30, 8, 1.0, 1.0) == (False, "样本积累中")
    # hourly 只看小时事件，daily 只看雨日——各榜不受另一轨证据影响
    assert _rain_evidence_note("hourly", {"rain_events_h": 30, "rain_days_d": 0},
                               30, 8, 1.0, 1.0)[0] is False
    assert _rain_evidence_note("daily", {"rain_events_h": 0, "rain_days_d": 8},
                               30, 8, 1.0, 1.0)[0] is False


def test_thresholds_are_registered_in_default_and_yaml():
    """阈值只此一处声明：代码缺省与仓库配置同值，且可被 eval 段覆盖。"""
    assert DEFAULT_EVAL["min_board_rain_events"] == 8
    assert DEFAULT_EVAL["min_board_rain_days"] == 3
    cfg = load_config(None)
    assert cfg.eval["min_board_rain_events"] == 8
    assert cfg.eval["min_board_rain_days"] == 3
    assert cfg.eval["board_block_min_rain_events"] == 8
    assert cfg.eval["board_block_min_rain_days"] == 3
    assert cfg.eval["board_min_qualified_days"] == 7
    assert cfg.eval["board_block_min_days"] == 5
    assert cfg.eval["board_block_min_cells"] == 3


def _perfect_snap(model: str, start: datetime, issue_day: int,
                  horizon_h: int, bias: float = 0.0) -> dict:
    """完美预报（可选择温度偏置）：用于让不同源有真实的技巧差异。"""
    issue = start + timedelta(days=issue_day)
    times = [iso(issue + timedelta(hours=h)) for h in range(1, horizon_h + 1)]
    temp = [20.0 + ((issue_day * 24 + h) % 5) + bias
            for h in range(1, horizon_h + 1)]
    rain = [2.0 if (issue_day * 24 + h) % 6 == 0 else 0.0
            for h in range(1, horizon_h + 1)]
    return {"issue_iso": iso(issue), "station_id": "s1", "source": "test",
            "models": [model], "grid_lat": 23.0, "grid_lon": 111.0,
            "elevation": 50, "hourly_time": times,
            "data": {model: {"temperature_2m": temp, "precipitation": rain}}}


def test_new_source_gets_official_score_after_seven_days(tmp_path, monkeypatch):
    """方案 A 的核心验收：7 天块模型让后加入的源在 7 天后拿**正式**分数。

    old 从 09-07 起连续 28 天；new 从 09-27 起（有效样本覆盖 09-28~10-04
    共 7 天）；too_new 从 09-30 起只有 5 天。窗口 = 4 个周一锚定的 7 天块。
    new 只有 1 个块，靠经验贝叶斯收缩向面板均值靠拢；它仍必须有
    score / rk / CI，而不是"参考分"。
    """
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 9, 7, 0, 0)        # 周一
    window_days = 28
    obs = []
    for h in range((window_days + 7) * 24):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + ((h) % 5),
                    "rain": 2.0 if h % 6 == 0 else 0.0})
    storage.save_obs("s1", obs)
    for day in range(window_days):
        storage.save_forecast_snapshot(
            "s1", "old", _perfect_snap("old", start, day, 168, bias=0.0))
    for day in range(20, window_days):        # 09-27 起，有效日覆盖 09-28~10-04
        storage.save_forecast_snapshot(
            "s1", "new", _perfect_snap("new", start, day, 168, bias=0.5))
    for day in range(23, window_days):        # 09-30 起，共 5 天
        storage.save_forecast_snapshot(
            "s1", "too_new", _perfect_snap("too_new", start, day, 168, bias=1.0))

    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "rain_daily_threshold_mm": 1.0, "rain_hourly_threshold_mm": 1.0,
           "hourly_lead_days": 16, "daily_max_offset_days": 16,
           "min_sample": 1, "min_cell_neff": 1,
           "board_cell_weighting": "equal", "board_min_col_frac": 0.0,
           "bootstrap_runs": 60, "sensitivity_runs": 30,
           "score_index_gain": 1.0, "score_composite_gain": 1.0}
    end = start + timedelta(hours=window_days * 24 - 1)
    data = build_report(["s1"], ["old", "new", "too_new"], cfg, start, end,
                        "2026-09 ~ 10")
    rows = {r["model"]: r for r in data["leaderboards"]["all"]}
    assert rows["old"]["qualified"] is True
    assert rows["new"]["qualified"] is True
    assert rows["new"]["score"] is not None
    assert rows["new"]["rk"] is not None
    assert rows["new"]["ci90"] is not None
    assert rows["new"]["gate_days"] == 7
    assert rows["new"]["gate_valid_block"] == "2026-09-28"
    # 证据量差异如实披露：新源 1 个块、老源 4 个块；单块新源的收缩不弱于老源
    eb = data["meta"]["difficulty_window"]["all"]["eb"]
    assert eb["n_blocks"]["new"] == 1 and eb["n_blocks"]["old"] == 4
    assert eb["shrink"]["new"] <= eb["shrink"]["old"]

    too = rows["too_new"]
    assert too["qualified"] is False
    assert too["score"] is None
    assert too["rk"] is None
    assert "样本积累中" in (too["evidence_note"] or "")
    # 小时榜同样给 new 正式分（两轨各自独立入围）
    hrows = {r["model"]: r for r in data["leaderboards"]["hourly"]}
    assert hrows["new"]["qualified"] is True and hrows["new"]["score"] is not None


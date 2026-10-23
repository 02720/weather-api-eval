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

from weather_eval.config import DEFAULT_EVAL, load_config
from weather_eval.evaluate import _rain_evidence_by_model, _rain_evidence_note


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
    assert DEFAULT_EVAL["min_board_rain_events"] == 30
    assert DEFAULT_EVAL["min_board_rain_days"] == 8
    cfg = load_config(None)
    assert cfg.eval["min_board_rain_events"] == 30
    assert cfg.eval["min_board_rain_days"] == 8

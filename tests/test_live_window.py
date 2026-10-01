"""实时总榜跨月累计（2026-10）的守卫测试。

口径分工：总榜 = 数据起点 → 现在（跨月累计，月初不清零）；月榜 = 单一自然月。
两榜共用同一套 build_report，差别只在窗口与 is_monthly 语义——本文件锁住：

1. 窗口探测（storage.available_months + __main__._live_window）的三个来源与回退；
2. 诊断层的月份契约：月份必须按窗口枚举，绝不能拿 period_label（跨月后是
   区间标签，不是任何一个月）当月份去读数据；
3. 跨月窗口下 build_report 的元信息（start/end/period_label）如实反映区间。
"""
from datetime import datetime, timedelta

from weather_eval import storage
from weather_eval.evaluate import _months_between, build_report
from weather_eval.__main__ import _live_window
from weather_eval.timeutil import iso, now_beijing

MIN_CFG = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}


# --------------------------------------------------------------- 窗口探测

def test_available_months_scans_obs_bundles_and_hot_layer(tmp_path, monkeypatch):
    """月份探测覆盖三个来源：观测月文件 / 冷层 bundle / 热层散装快照；
    锁文件与非月份命名的杂项不得混入。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    base = tmp_path / "obs" / "s1"
    base.mkdir(parents=True)
    for m in ("2026-08", "2026-09"):
        (base / f"{m}.json").write_text("{}", encoding="utf-8")
    (base / "2026-08.json.lock").write_text("", encoding="utf-8")   # 锁文件不算月份
    (base / "notes.txt").write_text("", encoding="utf-8")           # 杂项不算

    fc = tmp_path / "forecasts" / "s1" / "ecmwf_ifs"
    fc.mkdir(parents=True)
    (fc / "2026-08.json.gz").write_bytes(b"x")            # 冷层：月度 bundle
    (fc / "2026-10-01T0600.json").write_text("{}", encoding="utf-8")  # 热层：issue 命名
    (fc / "2026-10-01T0600.json.lock").write_text("", encoding="utf-8")

    months = storage.available_months()
    assert months == {"2026-08", "2026-09", "2026-10"}


def test_live_window_auto_starts_at_earliest_data_month(tmp_path, monkeypatch):
    """自动口径：起点 = 最早可用数据月的 1 号 00:00；标签如实反映区间。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    _touch_months(tmp_path, {"2026-08", "2026-09", "2026-10"})

    class Cfg:
        eval = {}

    start, end, label = _live_window(Cfg())
    assert (start.year, start.month, start.day) == (2026, 8, 1)
    assert start.hour == 0 and start.minute == 0
    assert label == f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d}"
    assert end >= start


def _touch_months(tmp_path, months):
    """只造文件名、不造内容——available_months/_live_window 只做文件名级扫描。"""
    for m in months:
        obs = tmp_path / "obs" / "s1" / f"{m}.json"
        obs.parent.mkdir(parents=True, exist_ok=True)
        obs.write_text("{}", encoding="utf-8")


def test_live_window_config_override_wins(tmp_path, monkeypatch):
    """eval.live_window_start 显式指定时压过自动探测（钉死起点做对照用）。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    _touch_months(tmp_path, {"2026-08"})

    class CfgMonth:
        eval = {"live_window_start": "2026-09"}

    class CfgDate:
        eval = {"live_window_start": "2026-09-15"}

    start_m, _, _ = _live_window(CfgMonth())
    assert (start_m.year, start_m.month, start_m.day) == (2026, 9, 1)
    start_d, _, _ = _live_window(CfgDate())
    assert (start_d.year, start_d.month, start_d.day) == (2026, 9, 15)


def test_live_window_empty_data_falls_back_to_current_month(tmp_path, monkeypatch):
    """data/ 里还没有任何存档时退回当月——空窗口不崩（首次部署/全新数据根）。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))

    class Cfg:
        eval = {}

    start, _, _ = _live_window(Cfg())
    now = now_beijing()
    assert (start.year, start.month) == (now.year, now.month)
    assert start.day == 1


def test_months_between_spans_cross_month_window():
    """跨月枚举：8/31 → 10/1 覆盖三个月；同月窗口只返回一个月。"""
    assert _months_between(datetime(2026, 8, 31, 23), datetime(2026, 10, 1, 0)) == \
        ["2026-08", "2026-09", "2026-10"]
    assert _months_between(datetime(2026, 9, 1), datetime(2026, 9, 30, 23)) == ["2026-09"]
    # 跨年（12 月 → 1 月）不因月序回绕而漏月
    assert _months_between(datetime(2025, 12, 1), datetime(2026, 1, 31)) == \
        ["2025-12", "2026-01"]


# --------------------------------------------------------------- 跨月窗口下的评估

def _populate_two_months(tmp_path, monkeypatch):
    """造 8、9 两个月的观测 + 两轮起报快照（形状与 test_render 的夹具同族）。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    for month, ndays in ((8, 31), (9, 30)):
        base = datetime(2026, month, 1, 0, 0)
        obs = []
        for h in range(ndays * 24):
            t = base + timedelta(hours=h)
            obs.append({"time": iso(t), "temp": 20.0 + (h % 5),
                        "rain": 1.0 if h % 12 == 0 else 0.0})
        storage.save_obs("s1", obs)
        for day in range(0, 6):
            issue = base + timedelta(days=day)
            times = [iso(base + timedelta(days=day, hours=hh)) for hh in range(24 * 4)]
            snap = {
                "issue_iso": iso(issue), "station_id": "s1", "source": "open-meteo",
                "models": ["ecmwf_ifs"], "grid_lat": 23.0, "grid_lon": 111.0,
                "elevation": 50,
                "hourly_time": times,
                "data": {"ecmwf_ifs": {
                    "temperature_2m": [20.0 + (day + hh % 5) % 5 + 0.5
                                       for hh in range(24 * 4)],
                    "precipitation": [1.0 if hh % 12 == 0 else 0.0
                                      for hh in range(24 * 4)],
                }},
            }
            storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap)


def test_build_report_cross_month_window_meta_and_diagnostics(tmp_path, monkeypatch):
    """跨月窗口的 build_report：meta 如实报区间；诊断层按窗口月份读到数据。

    回归锚点：诊断层曾拿 period_label 当月份找观测月文件——旧口径下"实时总榜的
    period_label 恰好等于窗口月份"才碰巧正确；period_label 改成区间标签后会静默
    读不到任何数据（诊断层整体退化为不可用）。本测试钉住"区间标签下诊断层仍然
    活着"这条契约。"""
    _populate_two_months(tmp_path, monkeypatch)
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 9, 30, 23, 0)
    data = build_report(["s1"], ["ecmwf_ifs"], MIN_CFG, start, end,
                        "2026-08-01 ~ 2026-09-30")
    meta = data["meta"]
    assert meta["start"] == "2026-08-01 00:00"
    assert meta["end"] == "2026-09-30 23:00"
    assert meta["period_label"] == "2026-08-01 ~ 2026-09-30"
    assert meta["is_monthly"] is False
    # 榜单照常产出（跨月窗口不是特殊路径，只是更宽的窗口）
    assert data["leaderboards"]["all"] and data["leaderboards"]["all"][0]["score"] is not None
    # 诊断层活着，且确实读到了两个月的样本
    diag = meta["diagnostics"]
    assert diag["available"] is True
    assert (diag.get("_debug") or {}).get("samples", 0) > 0


def test_monthly_window_semantics_untouched(tmp_path, monkeypatch):
    """月榜口径不变：自然月窗口 + is_monthly + period_label = 月份号。"""
    _populate_two_months(tmp_path, monkeypatch)
    start = datetime(2026, 9, 1, 0, 0)
    end = datetime(2026, 9, 30, 23, 0)
    data = build_report(["s1"], ["ecmwf_ifs"], MIN_CFG, start, end,
                        "2026-09", is_monthly=True)
    assert data["meta"]["period_label"] == "2026-09"
    assert data["meta"]["is_monthly"] is True
    assert data["leaderboards"]["all"][0]["score"] is not None

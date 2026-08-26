from datetime import datetime, timedelta

from weather_eval import storage
from weather_eval.evaluate import build_report, temp_metrics, precip_binary_metrics
from weather_eval.timeutil import iso


def test_temp_metrics_handcalc():
    # 误差 [0, 2, 0, 0] -> |err|<=2 全部成立；|err|<=1 仅有 3/4
    m = temp_metrics([20, 21, 19, 22], [20, 23, 19, 22], [1, 2], 0)
    assert m["acc2"] == 100.0
    assert m["acc1"] == 75.0
    assert abs(m["rmse"] - 1.0) < 1e-6
    assert abs(m["mae"] - 0.5) < 1e-6


def test_precip_binary_handcalc():
    # obs=[0,1,1,0,0,1] fcst=[0,1,0,0,1,1]  (>=0.1)
    # 命中 idx1,idx5 -> a=2；空报 idx4 -> b=1；漏报 idx2 -> c=1；正确否定 idx0,idx3 -> d=2
    # TS(a/(a+b+c))=0.5（比值）；acc/pod/far/miss 为百分数；bias 为比值
    m = precip_binary_metrics([0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1], 0.1, 0)
    assert abs(m["ts"] - 0.5) < 1e-6
    assert abs(m["acc"] - 100 * 2 / 3) < 1e-2
    assert abs(m["pod"] - 100 * 2 / 3) < 1e-2
    assert abs(m["far"] - 100 * 1 / 3) < 1e-2
    assert abs(m["bias"] - 1.0) < 1e-6


def test_min_sample_suppresses():
    m = temp_metrics([20, 21], [20, 23], [1, 2], 5)
    assert m["rmse"] is None
    assert m["n"] == 2


def test_build_report_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    obs = []
    for h in range(48):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 3), "rain": 1.0 if h % 6 == 0 else 0.0})
    storage.save_obs("s1", obs)

    times = [iso(start + timedelta(hours=h)) for h in range(48)]
    snap = {
        "issue_iso": iso(start), "station_id": "s1", "source": "open-meteo",
        "models": ["ecmwf_ifs"], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
        "hourly_time": times,
        "data": {"ecmwf_ifs": {
            "temperature_2m": [20.0 + (h % 3) + 1.0 for h in range(48)],
            "precipitation": [1.0 if h % 6 == 0 else 0.0 for h in range(48)],
        }},
    }
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap)

    end = start + timedelta(hours=47)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08")

    # 逐小时 24h 桶：lead 1..24 共 24 对，且预报=观测+1 -> 误差恒为 1 -> ±2°C 准确率 100%，RMSE=1
    sc_t = data["scorecard"]["ecmwf_ifs"]["temp_24h"]
    assert sc_t["n"] == 24
    assert sc_t["acc2"] == 100.0
    assert abs(sc_t["rmse"] - 1.0) < 1e-6

    # 降水：预报与观测逐小时完全一致 -> TS=1，准确率=100
    pb = data["scorecard"]["ecmwf_ifs"]["precip_24h"]
    assert abs(pb["ts"] - 1.0) < 1e-6
    assert pb["acc"] == 100.0

    # 逐小时各天桶都有数据（lead 上限 47，故 2d 桶含 lead 25..47 共 23 个样本）
    assert data["temp_hourly"]["ecmwf_ifs"]["1d"]["n"] == 24
    assert data["temp_hourly"]["ecmwf_ifs"]["2d"]["n"] == 23

    # 按天：offset 1（有效日 08-25）存在样本
    assert data["temp_daily"]["ecmwf_ifs"]["1d"]["max"]["n"] == 1

    # 时间序列与热力图结构存在
    assert "s1" in data["timeseries"]
    assert isinstance(data["heatmap"], list)


def test_build_report_empty_is_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 2, 0, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08")
    # 无数据时不崩溃，指标均为 None/空
    assert data["scorecard"]["ecmwf_ifs"]["temp_24h"]["n"] == 0
    assert data["temp_hourly"]["ecmwf_ifs"]["1d"]["rmse"] is None

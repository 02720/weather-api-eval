from weather_eval import storage


def test_obs_dedup_and_merge(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    r1 = [
        {"time": "2026-08-26T20:00", "temp": 27.0, "rain": 0.0},
        {"time": "2026-08-26T19:00", "temp": 26.0, "rain": 1.0},
    ]
    assert storage.save_obs("s1", r1) == 2
    # 重复保存相同数据：不新增
    assert storage.save_obs("s1", r1) == 0
    # 更新一条
    assert storage.save_obs("s1", [{"time": "2026-08-26T20:00", "temp": 28.0, "rain": 0.0}]) == 1
    loaded = storage.load_obs("s1", "2026-08")
    assert loaded["2026-08-26T20:00"]["temp"] == 28.0
    assert loaded["2026-08-26T19:00"]["rain"] == 1.0


def test_forecast_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    snap = {
        "issue_iso": "2026-08-26T21:00", "station_id": "s1", "source": "open-meteo",
        "models": ["ecmwf_ifs"], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
        "hourly_time": ["2026-08-26T21:00"],
        "data": {"ecmwf_ifs": {"temperature_2m": [20.0], "precipitation": [0.0]}},
    }
    assert storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap) is True
    # 同站同模型同起报时刻：幂等
    assert storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap) is False
    assert len(storage.list_forecast_snapshots("s1", "ecmwf_ifs")) == 1

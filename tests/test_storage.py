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


def test_corrupt_json_skipped_not_fatal(tmp_path, monkeypatch):
    """损坏的存档文件（git 冲突残留等）应告警跳过，不拖垮观测/快照读取。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    # 好文件 + 坏文件并存
    good_obs = tmp_path / "obs" / "s1" / "2026-08.json"
    good_obs.parent.mkdir(parents=True)
    good_obs.write_text('{"2026-08-26T20:00": {"time": "2026-08-26T20:00", "temp": 27.0}}',
                        encoding="utf-8")
    bad = tmp_path / "obs" / "s1" / "2026-07.json"
    bad.write_text('{"2026-07-01T10:00": {"time": ', encoding="utf-8")
    loaded = storage.load_obs("s1")
    assert set(loaded) == {"2026-08-26T20:00"}
    # 坏文件被跳过后仍可正常合并写入该月
    storage.save_obs("s1", [{"time": "2026-07-01T10:00", "temp": 25.0, "rain": 0.0}])
    assert storage.load_obs("s1", "2026-07")["2026-07-01T10:00"]["temp"] == 25.0

    # 预报快照同理
    good_snap_dir = tmp_path / "forecasts" / "s1" / "ecmwf_ifs"
    good_snap_dir.mkdir(parents=True)
    snap = {"issue_iso": "2026-08-26T21:00", "models": ["ecmwf_ifs"],
            "hourly_time": ["2026-08-26T21:00"], "data": {"ecmwf_ifs": {}}}
    (good_snap_dir / "2026-08-26T2100.json").write_text(
        __import__("json").dumps(snap), encoding="utf-8")
    (good_snap_dir / "broken.json").write_text("{oops", encoding="utf-8")
    assert len(storage.list_forecast_snapshots("s1", "ecmwf_ifs")) == 1

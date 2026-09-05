import json

from conftest import FakeSession

from weather_eval.forecast.open_meteo import OpenMeteoProvider, _model_key


def test_model_key_multi_and_single():
    units = {"temperature_2m_ecmwf_ifs": "°C", "precipitation_ecmwf_ifs": "mm"}
    assert _model_key(units, "temperature_2m", "ecmwf_ifs") == "temperature_2m_ecmwf_ifs"
    single = {"temperature_2m": "°C"}
    assert _model_key(single, "temperature_2m", "ecmwf_ifs") == "temperature_2m"
    assert _model_key({"temperature_2m_x": "°C"}, "temperature_2m", "ecmwf_ifs") is None


def test_model_key_bare_key_only_for_single_model():
    """裸键回退仅限单模型请求：多模型响应若混入裸键，不得让多模型共享同一数组。"""
    units = {"temperature_2m": "°C"}  # 混入裸键的多模型响应（理论情形）
    assert _model_key(units, "temperature_2m", "ecmwf_ifs", allow_bare=False) is None
    assert _model_key(units, "temperature_2m", "ecmwf_ifs", allow_bare=True) == "temperature_2m"


def test_fetch_snapshot_parses_daily_block():
    """逐日预报块：与逐小时同一次请求取回，按模型拆列、日期取到日（北京时自然日）。"""
    payload = {
        "latitude": 23.5, "longitude": 111.3, "elevation": 100,
        "hourly": {
            "time": ["2026-08-24T00:00", "2026-08-24T01:00"],
            "temperature_2m_ecmwf_ifs": [20.0, 21.0],
            "precipitation_ecmwf_ifs": [0.0, 0.1],
        },
        "hourly_units": {"temperature_2m_ecmwf_ifs": "°C", "precipitation_ecmwf_ifs": "mm"},
        "daily": {
            "time": ["2026-08-24", "2026-08-25"],
            "temperature_2m_max_ecmwf_ifs": [30.0, 31.0],
            "temperature_2m_min_ecmwf_ifs": [20.0, 21.0],
            "precipitation_sum_ecmwf_ifs": [0.0, 5.5],
        },
        "daily_units": {"temperature_2m_max_ecmwf_ifs": "°C",
                        "temperature_2m_min_ecmwf_ifs": "°C",
                        "precipitation_sum_ecmwf_ifs": "mm"},
    }
    src = OpenMeteoProvider(session=FakeSession(json.dumps(payload)))

    class S:
        id = "s1"
        lat = 23.5
        lon = 111.3

    snap = src.fetch_snapshot(S(), ["ecmwf_ifs"])
    assert snap["daily_time"] == ["2026-08-24", "2026-08-25"]
    assert snap["daily"]["ecmwf_ifs"] == {"temp_max": [30.0, 31.0],
                                          "temp_min": [20.0, 21.0],
                                          "precipitation": [0.0, 5.5]}
    # 逐小时主干不受影响
    assert snap["hourly_time"] == ["2026-08-24T00:00", "2026-08-24T01:00"]
    assert snap["data"]["ecmwf_ifs"]["temperature_2m"] == [20.0, 21.0]


def test_fetch_snapshot_without_daily_block_is_backward_compatible():
    """契约漂移（响应无 daily 组）只丢延长线，绝不丢整份起报快照。"""
    payload = {
        "latitude": 23.5, "longitude": 111.3, "elevation": 100,
        "hourly": {"time": ["2026-08-24T00:00"], "temperature_2m_ecmwf_ifs": [20.0],
                   "precipitation_ecmwf_ifs": [0.0]},
        "hourly_units": {"temperature_2m_ecmwf_ifs": "°C", "precipitation_ecmwf_ifs": "mm"},
    }
    src = OpenMeteoProvider(session=FakeSession(json.dumps(payload)))

    class S:
        id = "s1"
        lat = 23.5
        lon = 111.3

    snap = src.fetch_snapshot(S(), ["ecmwf_ifs"])
    assert "daily" not in snap and "daily_time" not in snap
    assert snap["data"]["ecmwf_ifs"]["temperature_2m"] == [20.0]


def test_daily_block_multimodel_uses_own_column():
    """多模型响应的裸键混淆在 daily 组同样危险：不得让两家共用同一列日产品。"""
    payload = {
        "latitude": 23.5, "longitude": 111.3, "elevation": 100,
        "hourly": {"time": ["2026-08-24T00:00"],
                   "temperature_2m_ecmwf_ifs": [20.0], "precipitation_ecmwf_ifs": [0.0],
                   "temperature_2m_ncep_gfs_global": [19.0],
                   "precipitation_ncep_gfs_global": [0.0]},
        "hourly_units": {"temperature_2m_ecmwf_ifs": "°C", "precipitation_ecmwf_ifs": "mm",
                         "temperature_2m_ncep_gfs_global": "°C",
                         "precipitation_ncep_gfs_global": "mm"},
        "daily": {"time": ["2026-08-24"],
                  "temperature_2m_max_ecmwf_ifs": [30.0],
                  "temperature_2m_min_ecmwf_ifs": [20.0],
                  "precipitation_sum_ecmwf_ifs": [0.0],
                  "temperature_2m_max_ncep_gfs_global": [31.0],
                  "temperature_2m_min_ncep_gfs_global": [21.0],
                  "precipitation_sum_ncep_gfs_global": [1.0]},
        "daily_units": {"temperature_2m_max_ecmwf_ifs": "°C",
                        "temperature_2m_min_ecmwf_ifs": "°C",
                        "precipitation_sum_ecmwf_ifs": "mm",
                        "temperature_2m_max_ncep_gfs_global": "°C",
                        "temperature_2m_min_ncep_gfs_global": "°C",
                        "precipitation_sum_ncep_gfs_global": "mm"},
    }
    src = OpenMeteoProvider(session=FakeSession(json.dumps(payload)))

    class S:
        id = "s1"
        lat = 23.5
        lon = 111.3

    snap = src.fetch_snapshot(S(), ["ecmwf_ifs", "ncep_gfs_global"])
    assert snap["daily"]["ecmwf_ifs"]["temp_max"] == [30.0]
    assert snap["daily"]["ncep_gfs_global"]["temp_max"] == [31.0]
    assert snap["daily"]["ncep_gfs_global"]["precipitation"] == [1.0]


def test_fetch_snapshot_multi_model():
    payload = {
        "latitude": 23.5, "longitude": 111.3, "elevation": 100,
        "hourly": {
            "time": ["2026-08-24T00:00", "2026-08-24T01:00"],
            "temperature_2m_ecmwf_ifs": [20.0, 21.0],
            "precipitation_ecmwf_ifs": [0.0, 0.1],
            "temperature_2m_ncep_gfs_global": [19.0, 22.0],
            "precipitation_ncep_gfs_global": [0.0, 0.0],
        },
        "hourly_units": {
            "temperature_2m_ecmwf_ifs": "°C", "precipitation_ecmwf_ifs": "mm",
            "temperature_2m_ncep_gfs_global": "°C", "precipitation_ncep_gfs_global": "mm",
        },
    }
    src = OpenMeteoProvider(session=FakeSession(json.dumps(payload)))

    class S:
        id = "s1"
        lat = 23.5
        lon = 111.3

    snap = src.fetch_snapshot(S(), ["ecmwf_ifs", "ncep_gfs_global"])
    assert "ecmwf_ifs" in snap["data"] and "ncep_gfs_global" in snap["data"]
    assert snap["data"]["ecmwf_ifs"]["temperature_2m"] == [20.0, 21.0]
    assert snap["data"]["ncep_gfs_global"]["precipitation"] == [0.0, 0.0]
    assert snap["grid_lat"] == 23.5 and snap["elevation"] == 100
    assert snap["issue_iso"].endswith(":00")
    assert len(snap["hourly_time"]) == 2

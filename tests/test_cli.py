"""CLI 杂项回归：monthly --month 参数校验与归档冻结保护、快照拆分的模型分账。"""
import pytest

from weather_eval import storage
from weather_eval.__main__ import main


def test_monthly_rejects_invalid_month():
    # L7 回归：非法月份给出可读错误并以非零码退出，而非裸 traceback
    with pytest.raises(SystemExit) as ei:
        main(["monthly", "--month", "2026-13"])
    assert ei.value.code != 0
    assert "无效月份" in str(ei.value)


@pytest.mark.parametrize("bad", ["2026-1", "26-08", "2026/08", "abc", "2026-00"])
def test_monthly_rejects_malformed_months(bad):
    with pytest.raises(SystemExit):
        main(["monthly", "--month", bad])


def test_fetch_forecast_splits_daily_block_per_model(tmp_path, monkeypatch):
    """共享时间轴的快照按模型拆分存档时，逐日预报块必须跟着拆——
    否则每个模型的文件里都会带上别家的日产品，按天评估会串账。"""
    import weather_eval.__main__ as m

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    models = ["ecmwf_ifs", "ncep_gfs_global"]

    class FakeProvider:
        def fetch_snapshot(self, station, model_list):
            return {
                "issue_iso": "2026-08-24T00:00", "station_id": station.id,
                "source": "open-meteo", "models": models,
                "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
                "hourly_time": ["2026-08-24T00:00", "2026-08-24T01:00"],
                "data": {mod: {"temperature_2m": [20.0, 21.0], "precipitation": [0.0, 0.0]}
                         for mod in models},
                "daily_time": ["2026-08-24"],
                "daily": {mod: {"temp_max": [30.0], "temp_min": [20.0],
                                "precipitation": [0.0 if mod == "ecmwf_ifs" else 9.0]}
                          for mod in models},
            }

    # 零参工厂注入（CLI 可注入性契约：见 __main__.SOURCE_SPECS 注释）
    monkeypatch.setattr(m, "OpenMeteoProvider", lambda: FakeProvider())
    monkeypatch.setattr(m, "NON_OPENMETEO_MODELS", set(models) - set(models))

    assert main(["fetch-forecast", "--source", "open_meteo"]) in (None, 0)

    for mod in models:
        snaps = storage.list_forecast_snapshots("wuzhou", mod)
        assert snaps, mod
        for snap in snaps:
            assert list(snap["data"]) == [mod]
            assert list(snap["daily"]) == [mod], "日产品块必须按模型拆分"
    # 两家的日产品值各不相同，拆分正确时不会串味
    assert (storage.list_forecast_snapshots("wuzhou", "ecmwf_ifs")[0]["daily"]["ecmwf_ifs"]
            ["precipitation"] == [0.0])
    assert (storage.list_forecast_snapshots("wuzhou", "ncep_gfs_global")[0]["daily"]
            ["ncep_gfs_global"]["precipitation"] == [9.0])

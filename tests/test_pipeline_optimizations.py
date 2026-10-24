"""流水线优化轮的回归测试。

覆盖六件事：
  1. Open-Meteo 痕迹量负降水的钳制（活跃数据 bug：-0.1mm 曾原样入库）；
  2. 落盘闸门统一清洗（非法值在写入时刻归一，读侧不再每轮重洗重报）；
  3. 告警去重（同一份历史脏存档一轮进程内只报一次）；
  4. 进程内 _preload 缓存（命中不重读盘；写路径落盘后自动失效）；
  5. health 陈旧判定的状态复用（新鲜即复用，陈旧/口径变更即重算）；
  6. 多站按天聚合的观测归属（按天段的观测查表必须绑定**本站**）。
"""
import json
import logging


from weather_eval import storage
from weather_eval.forecast.open_meteo import _clamp_trace_precip


# ------------------------------------------------------------------ 1. 钳制
def test_clamp_trace_precip():
    src = [0.0, -0.1, 1.5, -0.25, None, -5.0]
    out, n = _clamp_trace_precip(src)
    assert n == 2
    assert out == [0.0, 0.0, 1.5, 0.0, None, -5.0]
    # 大幅负值不属于"痕迹量"：静默改写会掩盖数据事故，交由落盘闸门按缺测处理
    # 干净数组零拷贝
    clean = [0.0, 1.0, None]
    out2, n2 = _clamp_trace_precip(clean)
    assert n2 == 0 and out2 is clean


def test_open_meteo_fetch_clamps_trace_precip(tmp_path, monkeypatch):
    """抓取路径端到端：-0.1 的逐小时与逐日降水入库前已变 0.0。"""
    import weather_eval.forecast.open_meteo as om
    times = [f"2026-08-26T{h:02d}:00" for h in range(3)]
    payload = {
        "hourly": {
            "time": times,
            "temperature_2m": [20.0, 21.0, 22.0],
            "precipitation": [0.0, -0.1, 2.0],
        },
        "hourly_units": {"temperature_2m": "°C", "precipitation": "mm"},
        "daily": {
            "time": ["2026-08-26"],
            "temperature_2m_max": [30.0],
            "temperature_2m_min": [20.0],
            "precipitation_sum": [-0.1],
        },
        "daily_units": {
            "temperature_2m_max": "°C", "temperature_2m_min": "°C",
            "precipitation_sum": "mm",
        },
        "latitude": 23.0, "longitude": 111.0, "elevation": 50.0,
    }

    class FakeResp:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return payload

    prov = om.OpenMeteoProvider(session=_FixedSession(FakeResp()))
    snap = prov.fetch_snapshot(_Station("s1", 23.0, 111.0), ["ecmwf_ifs"])
    assert snap["data"]["ecmwf_ifs"]["precipitation"] == [0.0, 0.0, 2.0]
    assert snap["daily"]["ecmwf_ifs"]["precipitation"] == [0.0]


class _FixedSession:
    def __init__(self, resp):
        self._resp = resp
    def get(self, *a, **kw):
        return self._resp
    def post(self, *a, **kw):
        return self._resp


class _Station:
    def __init__(self, id_, lat, lon):
        self.id, self.lat, self.lon = id_, lat, lon


# -------------------------------------------------------------- 2. 落盘闸门
def _dirty_snapshot():
    return {
        "issue_iso": "2026-08-26T21:00", "station_id": "s1", "source": "test",
        "models": ["m"], "hourly_time": ["2026-08-26T21:00", "2026-08-26T22:00"],
        "data": {"m": {"temperature_2m": [20.0, 999.9],
                       "precipitation": [0.5, -1092.2]}},
    }


def test_save_gate_sanitizes_illegal_values(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    with caplog.at_level(logging.WARNING, logger="weather_eval.storage"):
        assert storage.save_forecast_snapshot("s1", "m", _dirty_snapshot()) is True
    loaded = storage.list_forecast_snapshots("s1", "m")[0]
    assert loaded["data"]["m"]["temperature_2m"] == [20.0, None]
    assert loaded["data"]["m"]["precipitation"] == [0.5, None]
    assert any("清洗" in r.message for r in caplog.records)
    # 调用方持有的原对象不被原地改写（清洗返回副本）
    assert _dirty_snapshot()["data"]["m"]["temperature_2m"] == [20.0, 999.9]


def test_save_gate_clean_snapshot_untouched(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    snap = {
        "issue_iso": "2026-08-26T21:00", "station_id": "s1", "source": "test",
        "models": ["m"], "hourly_time": ["2026-08-26T21:00"],
        "data": {"m": {"temperature_2m": [20.0], "precipitation": [0.0]}},
    }
    with caplog.at_level(logging.WARNING, logger="weather_eval.storage"):
        assert storage.save_forecast_snapshot("s1", "m", snap) is True
    assert not any("清洗" in r.message for r in caplog.records)


# -------------------------------------------------------------- 3. 告警去重
def test_data_quality_warning_dedup(caplog):
    from weather_eval import evaluate as ev
    ev.reset_data_quality_warnings()
    with caplog.at_level(logging.WARNING, logger="weather_eval.evaluate"):
        snap = _dirty_snapshot()
        ev._sanitize_hourly_arrays(snap)
        ev._sanitize_hourly_arrays(json.loads(json.dumps(snap)))   # 同键再来一遍
    warned = [r for r in caplog.records if "非法值" in r.message]
    assert len(warned) == 1
    # 告警带上站/模型上下文（旧版只有 issue，无法定位来源）
    assert "s1" in warned[0].getMessage() and "issue=" in warned[0].getMessage()
    ev.reset_data_quality_warnings()


# ------------------------------------------------------------ 4. 预加载缓存
def test_preload_cache_hit_and_invalidate(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    from weather_eval import evaluate as ev
    ev.clear_preload_cache()
    snap = {
        "issue_iso": "2026-08-26T21:00", "station_id": "s1", "source": "test",
        "models": ["m"], "hourly_time": ["2026-08-26T21:00"],
        "data": {"m": {"temperature_2m": [20.0], "precipitation": [0.0]}},
    }
    storage.save_forecast_snapshot("s1", "m", snap)
    o1, s1 = ev._preload(["s1"], ["m"])
    o2, s2 = ev._preload(["s1"], ["m"])
    assert s1 is s2 and o1 is o2                     # 命中：同一批对象
    # 写路径落盘后版本递增 → 缓存失效，读到新数据
    snap2 = dict(snap, issue_iso="2026-08-27T21:00",
                 hourly_time=["2026-08-27T21:00"])
    storage.save_forecast_snapshot("s1", "m", snap2)
    _, s3 = ev._preload(["s1"], ["m"])
    assert s3 is not s1 and len(s3[("s1", "m")]) == 2
    ev.clear_preload_cache()


# ------------------------------------------------- 5. health 状态复用
def test_health_state_reuse(tmp_path, monkeypatch):
    import weather_eval.__main__ as main
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("WEATHER_EVAL_REPORTS_ROOT", str(tmp_path / "reports"))

    class Cfg:
        station_ids = ["s1"]
        models = ["m"]
        eval = {}
        class stations:
            pass
    # 造一份"新鲜"状态：无陈旧源
    state_path = tmp_path / "health" / "source_staleness.json"
    state_path.parent.mkdir(parents=True)
    from weather_eval.timeutil import now_beijing
    state_path.write_text(json.dumps({
        "generated_at": now_beijing().strftime("%Y-%m-%dT%H:%M:%S"),
        "stale_hours": 30, "stale_sources": [],
    }), encoding="utf-8")

    called = {"n": 0}

    def _boom(cfg, stale_hours=30):
        called["n"] += 1
        raise AssertionError("新鲜状态不应触发全量重算")

    monkeypatch.setattr(main, "_compute_and_write_health", _boom)
    rc = main.cmd_health(_Args(30))
    assert rc == 0 and called["n"] == 0

    # 状态过期 → 回退全量重算（保留 CI 的 if:always() 语义）
    state_path.write_text(json.dumps({
        "generated_at": "2026-01-01T00:00:00",
        "stale_hours": 30, "stale_sources": ["m"],
    }), encoding="utf-8")
    recomputed = {"n": 0}

    def _recompute(cfg, stale_hours):
        recomputed["n"] += 1
        return ["m"]

    monkeypatch.setattr(main, "_compute_and_write_health", _recompute)
    rc = main.cmd_health(_Args(30))
    assert rc == 1 and recomputed["n"] == 1


class _Args:
    def __init__(self, stale_hours):
        self.stale_hours = stale_hours
        self.config = None


def test_write_health_report_persists_state(tmp_path, monkeypatch):
    import weather_eval.__main__ as main
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("WEATHER_EVAL_REPORTS_ROOT", str(tmp_path / "reports"))
    monkeypatch.setattr(main, "load_config", lambda *a, **k: _FakeCfg())
    n = main.write_health_report(_FakeCfg(), stale_hours=30)
    assert n == 0
    state = json.loads((tmp_path / "health" / "source_staleness.json").read_text())
    assert state["stale_hours"] == 30 and state["stale_sources"] == []


class _FakeCfg:
    station_ids = ["s1"]
    models = ["m"]
    eval = {"source_stale_hours": 30}
    stations = []


# --------------------------------------- 6. 多站按天聚合的观测归属
def test_daily_collect_uses_own_station_obs(tmp_path, monkeypatch):
    """按天段读观测必须绑定本站：站序变化不得让样本串站。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    from weather_eval import evaluate as ev
    from weather_eval.timeutil import parse_iso
    hours = [f"2026-08-27T{h:02d}:00" for h in range(24)]

    def make_obs(temp):
        return {t: {"time": t, "temp": temp, "rain": 0.0} for t in hours}

    def make_snap(sid):
        return {
            "issue_iso": "2026-08-26T00:00", "station_id": sid, "source": "test",
            "issue_source": "model_run",
            "models": ["m"], "hourly_time": ["2026-08-26T00:00"] + hours,
            "fetched_at_bj": "2026-08-26T00:05",
            "data": {"m": {
                "temperature_2m": [20.0] + [25.0] * 24,
                "precipitation": [0.0] + [0.0] * 24}},
        }

    obs_maps = {"a": make_obs(30.0), "b": make_obs(10.0)}
    storage.save_obs("a", list(obs_maps["a"].values()))
    storage.save_obs("b", list(obs_maps["b"].values()))
    snapshots = {
        ("a", "m"): [make_snap("a")], ("b", "m"): [make_snap("b")],
    }
    for sid in ("a", "b"):
        storage.save_forecast_snapshot(sid, "m", make_snap(sid))
    for order in (("a", "b"), ("b", "a")):
        _h, daily = ev.collect(list(order), ["m"],
                               parse_iso("2026-08-26T00:00"),
                               parse_iso("2026-08-28T00:00"),
                               3, 3, obs_maps=obs_maps, snapshots=snapshots)
        by_station = {r["station"]: r for r in daily if r["valid_day"] == "2026-08-27"}
        # 各站的日最高观测必须来自本站（30 vs 10 差异足够大，串站必被抓住）
        assert by_station["a"]["temp_max_obs"] == 30.0
        assert by_station["b"]["temp_max_obs"] == 10.0

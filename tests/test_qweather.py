"""和风天气（QWeather）weather/v1 接入测试。

覆盖：UTC 时间戳转北京时下取整、坐标取整契约（≤2 位小数）、weather/v1 主路解析、
路由不可用时自动降级旧版 v7、鉴权失败快速终止、数值字符串/null 归一、重复整点
合并、返回点数截断告警、API Key 日志脱敏，以及与评估引擎端到端配对。
"""
import json
import logging
from datetime import datetime, timedelta

import pytest
import requests

from conftest import FakeResp

from weather_eval.forecast.qweather import (
    QWeatherProvider,
    _parse_qweather_dt,
    _to_float,
    DEFAULT_HOURS,
)
from weather_eval import storage
from weather_eval.evaluate import build_report
from weather_eval.timeutil import iso


def _station():
    class S:
        id = "s1"
        lat = 23.4783
        lon = 111.304
    return S()


def _bjt(dt_beijing: datetime) -> str:
    """把北京时 naive datetime 转成和风新版接口的 UTC 字符串（'Z' 结尾）。"""
    return (dt_beijing - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%MZ")


def _v1_hours(base_bj, temps, precips):
    """按官方响应示例的真实 schema 构造 hours 数组。

    带单位的量纲统一为 value/unit 对象：temperature={"value":..,"unit":"°C"}、
    precipitation={"amount":{"value":..,"unit":"mm"},"intensity":{...},...}。
    temp/precip 为 None 时模拟缺测（对应子对象为空 dict）。"""
    items = []
    for i, (tv, pv) in enumerate(zip(temps, precips)):
        items.append({
            "forecastTime": _bjt(base_bj + timedelta(hours=i)),
            "temperature": {} if tv is None else {"value": tv, "unit": "°C"},
            "precipitation": (
                {} if pv is None else
                {"amount": {"value": pv, "unit": "mm"},
                 "intensity": {"value": pv / 3 or 0.05, "unit": "mm/h"},
                 "probability": 0.31, "type": "rain" if pv else "none"}
            ),
        })
    return items


def _v1_payload(base_bj, temps, precips):
    return {
        "metadata": {"tag": "f" * 64, "attributions": ["qweather"]},
        "hours": _v1_hours(base_bj, temps, precips),
    }


def _v7_payload(base_bj, temps, precips):
    """构造旧版 v7 响应：数值为字符串；precip 为 None 时用空串模拟缺测。"""
    items = []
    for i, (tv, pv) in enumerate(zip(temps, precips)):
        t = (base_bj + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") + "+08:00"
        items.append({
            "fxTime": t,
            "temp": "" if tv is None else str(tv),
            "precip": "" if pv is None else str(pv),
            "text": "多云", "icon": "101",
        })
    update_time = base_bj.strftime("%Y-%m-%dT%H:%M") + "+08:00"
    return {"code": "200", "updateTime": update_time,
            "fxLink": "http://hfx.link/2ax1", "hourly": items}


class _RoutingSession:
    """按 URL 分发响应的假 session，记录全部请求用于断言契约。"""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def get(self, url, params=None, headers=None, **kwargs):
        self.calls.append({"url": url, "params": params, "headers": headers})
        resp = self.handler(url, params)
        status, payload = resp
        return FakeResp(payload if isinstance(payload, str) else json.dumps(payload), status)


def _ok(payload):
    return 200, payload


# ----------------------------------------------------------------- 时间解析
def test_parse_utc_converts_to_beijing_and_floors():
    # 07:10Z -> 北京时 15:10 -> 下取整 15:00
    dt = _parse_qweather_dt("2026-08-27T07:10Z")
    assert dt.strftime("%Y-%m-%dT%H:%M") == "2026-08-27T15:00"
    # 已带 +08:00 的形态同样处理（旧版 fxTime）
    dt2 = _parse_qweather_dt("2026-08-27T15:59+08:00")
    assert dt2.strftime("%Y-%m-%dT%H:%M") == "2026-08-27T15:00"
    # 小写 z 与无偏移兜底
    assert _parse_qweather_dt("2026-08-27T02:03z").strftime("%H") == "10"
    assert _parse_qweather_dt("2026-08-27T15:10").strftime("%H:%M") == "15:00"


def test_to_float_normalizes_all_shapes():
    assert _to_float(28.5) == 28.5          # 新版 number
    assert _to_float("28") == 28.0          # 旧版字符串
    assert _to_float("0.5") == 0.5
    assert _to_float("") is None            # v7 空串缺测
    assert _to_float(None) is None
    assert _to_float({}) is None            # 异常类型不抛错
    assert _to_float(float("nan")) is None


def test_v1_scalar_amount_shape_also_supported():
    """个别字段形态可能直接给标量（amount=数字），实现须双形态兼容。"""
    hours = [{
        "forecastTime": _bjt(datetime(2026, 8, 27, 7, 0)),
        "temperature": {"value": 28.5},
        "precipitation": {"amount": 2.5},
    }]
    sess = _RoutingSession(lambda u, p: _ok({"metadata": {}, "hours": hours}))
    snap = QWeatherProvider(key="d", session=sess, retries=0).fetch_snapshot(_station())
    assert snap["data"]["qweather_v1"]["precipitation"] == [2.5]
    assert snap["data"]["qweather_v1"]["temperature_2m"] == [28.5]


# ----------------------------------------------------------------- 主路解析
def test_fetch_v1_parses_snapshot(monkeypatch):
    monkeypatch.delenv("QWEATHER_API_HOST", raising=False)
    base = datetime(2026, 8, 27, 15, 0)
    payload = _v1_payload(base, [28.5, 27.0, 26.0], [0.0, 0.2, None])
    sess = _RoutingSession(lambda url, params: _ok(payload))
    src = QWeatherProvider(key="dummy", session=sess, retries=0)
    snap = src.fetch_snapshot(_station(), ["qweather_v1"])

    assert snap["source"] == "qweather"
    assert snap["models"] == ["qweather_v1"]
    # UTC 07:00Z/08:00Z/09:00Z -> 北京时 15:00/16:00/17:00
    assert snap["hourly_time"] == ["2026-08-27T15:00", "2026-08-27T16:00", "2026-08-27T17:00"]
    assert snap["data"]["qweather_v1"]["temperature_2m"] == [28.5, 27.0, 26.0]
    assert snap["data"]["qweather_v1"]["precipitation"] == [0.0, 0.2, None]  # 缺测归 None
    assert snap["issue_iso"] == "2026-08-27T15:00"
    assert snap["elevation"] is None
    # 坐标契约：路径取 2 位小数取整值；快照记录 requested 与 grid 两份
    assert snap["requested_lat"] == 23.4783 and snap["requested_lon"] == 111.304
    assert snap["grid_lat"] == 23.48 and snap["grid_lon"] == round(111.304, 2)


def test_v1_request_contract(monkeypatch):
    """锁定新版的请求契约：纬度在前、坐标 ≤2 位小数、hours 参数、Key 走请求头。"""
    monkeypatch.delenv("QWEATHER_API_HOST", raising=False)
    base = datetime(2026, 8, 27, 15, 0)
    sess = _RoutingSession(lambda url, params: _ok(_v1_payload(base, [1.0], [0.0])))
    QWeatherProvider(key="dummykey123", session=sess, retries=0).fetch_snapshot(_station())

    call = sess.calls[0]
    assert "/weather/v1/hourly/23.48/" in call["url"]
    assert call["url"].rstrip("/").endswith(str(round(111.304, 2)))
    assert call["params"]["hours"] == DEFAULT_HOURS
    assert call["headers"]["X-QW-Api-Key"] == "dummykey123"
    # Key 绝不允许出现在 URL/query 里
    assert "dummykey123" not in call["url"]
    assert all("dummykey123" not in str(v) for v in call["params"].values())


def test_duplicate_times_merged_keep_first(caplog):
    """同一整点的两条时间戳（下取整后碰撞）应去重并保留首见条目。"""
    base = datetime(2026, 8, 27, 15, 0)
    hours = [
        # 两个不同 UTC 分钟均落在北京时 15 点档
        {"forecastTime": _bjt(datetime(2026, 8, 27, 14, 5)),
         "temperature": {"value": 28.5}, "precipitation": {"amount": 0.0}},
        {"forecastTime": _bjt(datetime(2026, 8, 27, 14, 55)),
         "temperature": {"value": 99.0}, "precipitation": {"amount": 9.9}},
        {"forecastTime": _bjt(datetime(2026, 8, 27, 15, 0)),
         "temperature": {"value": 27.0}, "precipitation": {"amount": 0.1}},
    ]
    sess = _RoutingSession(lambda u, p: _ok({"metadata": {}, "hours": hours}))
    snap = QWeatherProvider(key="dummy", session=sess, retries=0).fetch_snapshot(_station())
    assert snap["hourly_time"] == ["2026-08-27T14:00", "2026-08-27T15:00"]
    assert snap["data"]["qweather_v1"]["temperature_2m"] == [28.5, 27.0]
    assert snap["data"]["qweather_v1"]["precipitation"] == [0.0, 0.1]


# ----------------------------------------------------------------- 降级与鉴权
def test_fallback_to_v7_when_route_missing(monkeypatch, caplog):
    """weather/v1 路由不存在（404）时自动降级 v7，且字符串数值被正确转换。"""
    monkeypatch.delenv("QWEATHER_API_HOST", raising=False)
    base = datetime(2026, 8, 27, 15, 0)

    def route(url, params):
        if "weather/v1" in url:
            return 404, {"error": "route not found"}
        assert "/v7/weather/" in url
        return _ok(_v7_payload(base, ["28.5", "27"], [None, "0.2"]))  # None->"" 模拟

    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.qweather"):
        snap = QWeatherProvider(key="dummy", session=_RoutingSession(route), retries=0) \
            .fetch_snapshot(_station())

    assert any("改用旧版" in r.message for r in caplog.records)
    assert snap["source"] == "qweather"
    assert snap["hourly_time"][0] == "2026-08-27T15:00"
    vals = snap["data"]["qweather_v1"]
    assert vals["temperature_2m"][0] == 28.5
    assert vals["temperature_2m"][1] == 27.0
    assert vals["precipitation"][0] is None      # 空串 -> None
    assert vals["precipitation"][1] == 0.2


def test_fallback_picks_largest_tier_within_requested_hours(monkeypatch):
    """默认请求 240h，v7 只有 24/72/168 三档 -> 应选 168h。"""
    monkeypatch.delenv("QWEATHER_API_HOST", raising=False)
    base = datetime(2026, 8, 27, 15, 0)

    def route(url, params):
        if "weather/v1" in url:
            return 404, {}
        assert "/v7/weather/168h" in url
        assert params["location"] == f"{round(111.304, 2)},{round(23.4783, 2)}"
        return _ok(_v7_payload(base, ["1.0"], ["0.0"]))

    sess = _RoutingSession(route)
    QWeatherProvider(key="d", session=sess, retries=0).fetch_snapshot(_station())
    assert len(sess.calls) == 2


def test_auth_failure_terminates_without_fallback(monkeypatch):
    """401 是账号级问题，直接失败且不做第二次（v7）请求。"""
    monkeypatch.delenv("QWEATHER_API_HOST", raising=False)
    sess = _RoutingSession(lambda url, params: (401, {"error": "unauthorized"}))
    with pytest.raises(RuntimeError) as exc:
        QWeatherProvider(key="k", session=sess, retries=0).fetch_snapshot(_station())
    assert "401" in str(exc.value)
    assert len(sess.calls) == 1


def test_both_generations_failing_raises_joint_error(monkeypatch):
    """v1 与 v7 都失败时，错误信息应同时携带两侧线索，便于定位凭据/Host 问题。"""
    monkeypatch.delenv("QWEATHER_API_HOST", raising=False)

    def route(url, params):
        if "weather/v1" in url:
            return 404, {"error": "no route"}
        return 200, {"code": "402"}  # v7 业务错误码（如额度不足）

    with pytest.raises(RuntimeError) as exc:
        QWeatherProvider(key="k", session=_RoutingSession(route), retries=0) \
            .fetch_snapshot(_station())
    msg = str(exc.value)
    assert "weather/v1" in msg and "402" in msg and "v7" in msg


def test_empty_hours_raises():
    payload = {"metadata": {}, "hours": []}
    empty_v7 = {"code": "200", "hourly": []}
    src = QWeatherProvider(
        key="d", retries=0,
        session=_RoutingSession(
            lambda u, p: _ok(payload) if "weather/v1" in u else _ok(empty_v7)),
    )
    # hours 为空数组 -> 走降级链 -> v7 也为空 -> 最终报错而非产出空快照
    with pytest.raises(RuntimeError):
        src.fetch_snapshot(_station())


# ----------------------------------------------------------------- 截断告警
def test_truncated_response_warns(caplog):
    """免费档等场景只回 24 点而请求 240 点 -> 必须以 WARNING 暴露静默缩短。"""
    base = datetime(2026, 8, 27, 15, 0)
    payload = _v1_payload(base, list(range(24)), [0.0] * 24)
    src = QWeatherProvider(key="d", hours=DEFAULT_HOURS,
                           session=_RoutingSession(lambda u, p: _ok(payload)), retries=0)
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.qweather"):
        snap = src.fetch_snapshot(_station())
    assert len(snap["hourly_time"]) == 24
    assert any("24 个逐小时点" in r.message for r in caplog.records)


# ----------------------------------------------------------------- 配置与安全
def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv("QWEATHER_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        QWeatherProvider(session=requests.Session())


def test_key_arg_beats_env(monkeypatch):
    monkeypatch.setenv("QWEATHER_API_KEY", "from_env")
    prov = QWeatherProvider(key="from_arg")
    assert prov.key == "from_arg"


def test_host_env_is_used_and_normalized(monkeypatch):
    monkeypatch.setenv("QWEATHER_API_HOST", "https://MyHost.qweatherapi.com/")
    base = datetime(2026, 8, 27, 15, 0)
    sess = _RoutingSession(lambda u, p: _ok(_v1_payload(base, [1.0], [0.0])))
    QWeatherProvider(key="d", session=sess, retries=0).fetch_snapshot(_station())
    assert sess.calls[0]["url"].startswith("https://myhost.qweatherapi.com/weather/v1/")


def test_unreachable_host_defaults_to_deprecated_with_warning(monkeypatch, caplog):
    """未提供 Host 时退回 devapi 并显式告警（旧公共地址 2026 年起逐步停服）。"""
    monkeypatch.delenv("QWEATHER_API_HOST", raising=False)
    base = datetime(2026, 8, 27, 15, 0)
    sess = _RoutingSession(lambda u, p: _ok(_v1_payload(base, [1.0], [0.0])))
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.qweather"):
        QWeatherProvider(key="d", session=sess, retries=0).fetch_snapshot(_station())
    assert sess.calls[0]["url"].startswith("https://devapi.qweather.com/weather/v1/")
    assert any("逐步停止服务" in r.message for r in caplog.records)


def test_key_redacted_in_logs_and_errors(monkeypatch, caplog):
    """网络异常信息可能包含 URL/请求头上下文，日志与最终异常都必须脱敏。"""
    monkeypatch.delenv("QWEATHER_API_HOST", raising=False)
    sentinel = "FAKEQWKEY456"

    class _BoomSession:
        def get(self, url, **kwargs):
            headers = kwargs.get("headers") or {}
            raise ConnectionError(f"refused while GET {url} with {headers}")

    prov = QWeatherProvider(key=sentinel, session=_BoomSession(), retries=0)
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.qweather"):
        with pytest.raises(RuntimeError) as exc:
            prov.fetch_snapshot(_station())
    assert sentinel not in str(exc.value)
    assert all(sentinel not in r.message for r in caplog.records)


# ----------------------------------------------------------------- CLI 集成
def test_cmd_fetch_forecast_open_meteo_excludes_all_standalone_sources(monkeypatch):
    """Open-Meteo 分支必须同时排除 caiyun_v2_6 与 qweather_v1，避免刷缺失模型警告。"""
    import weather_eval.__main__ as m

    captured = {}

    class FakeOM:
        def fetch_snapshot(self, station, models):
            captured["models"] = list(models)
            return {
                "issue_iso": "2026-08-27T00:00", "station_id": station.id,
                "source": "open-meteo", "models": list(models),
                "grid_lat": 0.0, "grid_lon": 0.0, "elevation": 0.0,
                "hourly_time": [],
                "data": {mm: {"temperature_2m": [], "precipitation": []} for mm in models},
            }

    monkeypatch.setattr(m, "OpenMeteoProvider", lambda: FakeOM())
    monkeypatch.setattr(m, "save_forecast_snapshot", lambda *a, **k: True)

    class Station:
        id = "s1"
        lat = 23.0
        lon = 111.0

    class Cfg:
        models = ["ecmwf_ifs", "caiyun_v2_6", "qweather_v1"]
        stations = [Station()]

    monkeypatch.setattr(m, "load_config", lambda cfg: Cfg())

    class Args:
        source = "open_meteo"
        config = None

    m.cmd_fetch_forecast(Args())
    assert captured["models"] == ["ecmwf_ifs"]


def test_cmd_fetch_forecast_routes_to_qweather(monkeypatch):
    """--source qweather 应构造 QWeatherProvider 并用其单模型抓取存档。"""
    import weather_eval.__main__ as m

    made = {}

    class FakeQW:
        name = "qweather_v1"

        def fetch_snapshot(self, station, models):
            made["models"] = list(models)
            return {
                "issue_iso": "2026-08-27T15:00", "station_id": station.id,
                "source": "qweather", "models": ["qweather_v1"],
                "grid_lat": 0.0, "grid_lon": 0.0, "elevation": None,
                "hourly_time": [],
                "data": {"qweather_v1": {"temperature_2m": [], "precipitation": []}},
            }

    monkeypatch.setattr(m, "QWeatherProvider", lambda: FakeQW())
    saved = []
    monkeypatch.setattr(m, "save_forecast_snapshot",
                        lambda sid, mod, sub: saved.append((sid, mod)) or True)

    class Station:
        id = "s1"
        lat = 23.0
        lon = 111.0

    class Cfg:
        models = ["ecmwf_ifs", "qweather_v1"]
        stations = [Station()]

    monkeypatch.setattr(m, "load_config", lambda cfg: Cfg())

    class Args:
        source = "qweather"
        config = None

    m.cmd_fetch_forecast(Args())
    assert made["models"] == ["qweather_v1"]  # 单模型列表而非全量 cfg.models
    assert saved == [("s1", "qweather_v1")]


# ----------------------------------------------------------------- 端到端
def _e2e_setup(tmp_path, monkeypatch, n_hours=3):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 27, 15, 0)
    temps = [28.5 - i * 0.5 for i in range(n_hours)]
    precips = [0.0 if i % 2 == 0 else 0.2 for i in range(n_hours)]
    obs = [{"time": iso(start + timedelta(hours=i)), "temp": temps[i], "rain": precips[i]}
           for i in range(n_hours)]
    storage.save_obs("s1", obs)
    payload = _v1_payload(start, temps, precips)
    sess = _RoutingSession(lambda u, p: _ok(payload))
    snap = QWeatherProvider(key="d", session=sess, retries=0).fetch_snapshot(_station())
    storage.save_forecast_snapshot("s1", "qweather_v1", snap)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 0}
    return start, temps, precips, cfg


def test_end_to_end_with_obs(tmp_path, monkeypatch):
    """UTC 快照与整点观测完美匹配 -> 温度±2°C 准确率100%、降水TS=1。lead0 排除。"""
    start, _, _, cfg = _e2e_setup(tmp_path, monkeypatch, n_hours=3)
    end = start + timedelta(hours=2)
    data = build_report(["s1"], ["qweather_v1"], cfg, start, end, "2026-08")

    sc_t = data["scorecard"]["qweather_v1"]["temp_24h"]
    assert sc_t["n"] == 2  # lead0 被排除，剩 16:00/17:00 两对
    assert sc_t["acc2"] == 100.0
    assert abs(sc_t["rmse"]) < 1e-6
    pb = data["scorecard"]["qweather_v1"]["precip_24h"]
    assert pb["acc"] == 100.0
    assert abs(pb["ts"] - 1.0) < 1e-6


def test_end_to_end_spans_day_for_daily(tmp_path, monkeypatch):
    """30 小时序列跨日 -> 评估引擎能产出 offset=1 的按天样本且逐日一致。"""
    start, _, _, cfg = _e2e_setup(tmp_path, monkeypatch, n_hours=30)
    end = start + timedelta(hours=29)
    data = build_report(["s1"], ["qweather_v1"], cfg, start, end, "2026-08")
    daily_max = data["temp_daily"]["qweather_v1"]["1d"]["max"]
    assert daily_max["n"] >= 1
    assert daily_max["acc2"] == 100.0

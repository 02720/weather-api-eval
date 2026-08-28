"""伏羲中期（FuxiC88Provider）单测。

mock 契约来自 2026-08 线上实测 + 前端 JS 逆向：
- GET /gw/weather/api/v1/weather/queryWeatherTile → data[].{forecastType,startTime,...}
- POST /gw/weather/api/v1/weather/queryWeatherInfo body {lat,lon,forecastType}
  → data.weatherInfoList[{step,t2m,tp,...}]，值为字符串，无绝对时间。
- startTime 为 UTC YYYYMMDDHH（北京时 = +8h），时刻 = 北京时起报 + step 小时。
"""
import json

import pytest
from conftest import FakeResp

from weather_eval.forecast.fuxi import (
    EXPECTED_HOURS,
    FuxiC88Provider,
    INFO_URL,
    TILE_URL,
    parse_tile_start_time,
    parse_weather_info,
    tile_start_to_issue_iso,
)

TILE_PAYLOAD = json.dumps({
    "traceId": "x", "msgCode": "10000", "msg": "ok", "success": True, "total": 2,
    "data": [
        {"ossPrefix": "s2s", "startTime": "2026082712", "totalStep": 1440,
         "range": 24, "forecastType": "2"},
        {"ossPrefix": "c88", "startTime": "2026082712", "totalStep": 360,
         "range": 1, "forecastType": "1"},
    ],
})


def _info_payload(steps):
    """构造点位响应：steps = [(step, t2m, tp), ...]，值按线上为字符串。"""
    return json.dumps({
        "success": True, "msgCode": "10000",
        "data": {"stepRange": 1,
                 "weatherInfoList": [
                     {"step": s, "t2m": t2m, "tp": tp, "q850": "16.3",
                      "ssrd": "-0.2", "u10m": "-0.6", "v10m": "0.4",
                      "t2mAno": None, "tpAnoPercent": None}
                     for s, t2m, tp in steps]},
    })


class RoutingSession:
    """按 (method, url) 路由响应的假 session。"""

    def __init__(self, routes):
        self.routes = routes          # fn(method, url, kwargs) -> (status, text)
        self.calls: list[tuple] = []

    def get(self, url, **kwargs):
        self.calls.append(("GET", url, kwargs))
        s, t = self.routes("GET", url, kwargs)
        return FakeResp(t, status_code=s)

    def post(self, url, **kwargs):
        self.calls.append(("POST", url, kwargs))
        s, t = self.routes("POST", url, kwargs)
        return FakeResp(t, status_code=s)


class Station:
    id, name, lat, lon = "wuzhou", "梧州", 23.4783, 111.304


def _session(steps=None):
    steps = steps or [(i, str(20 + i % 5), "0.08") for i in range(1, EXPECTED_HOURS + 1)]

    def routes(method, url, kwargs):
        if url == TILE_URL:
            return 200, TILE_PAYLOAD
        if url == INFO_URL:
            body = (kwargs.get("json") or {})
            assert body["forecastType"] == "1"
            assert body["lat"] == Station.lat and body["lon"] == Station.lon
            return 200, _info_payload(steps)
        return 404, "{}"

    return RoutingSession(routes)


def test_tile_start_time_parsing():
    assert parse_tile_start_time(json.loads(TILE_PAYLOAD)) == "2026082712"
    bad = json.loads(TILE_PAYLOAD)
    bad["data"] = [dict(forecastType="2", startTime="2026082712")]
    with pytest.raises(RuntimeError, match="forecastType=1"):
        parse_tile_start_time(bad)


def test_tile_start_time_is_utc_beijing_plus_8():
    # 契约锚点：startTime 是 UTC；12Z → 北京时 20 时（模块 docstring 第 2 点）
    assert tile_start_to_issue_iso("2026082712") == "2026-08-27T20:00"
    assert tile_start_to_issue_iso("2026082800") == "2026-08-28T08:00"
    # 跨日/跨月边界
    assert tile_start_to_issue_iso("2026083118") == "2026-09-01T02:00"


def test_parse_weather_info_anchors_steps_to_issue():
    payload = json.loads(_info_payload([(1, "27.5", "0.08"), (3, "26.9", None),
                                        (2, "27.1", "abc")]))
    series = parse_weather_info(payload, "2026-08-27T20:00")
    assert series["time"] == ["2026-08-27T21:00", "2026-08-27T22:00", "2026-08-27T23:00"]
    assert series["temperature_2m"] == [27.5, 27.1, 26.9]
    assert series["precipitation"] == [0.08, None, None]  # 非法字符串 → None


def test_parse_weather_info_rejects_empty():
    with pytest.raises(RuntimeError, match="weatherInfoList 为空"):
        parse_weather_info(json.loads(_info_payload([])), "2026-08-27T20:00")


def test_all_steps_invalid_rejected_before_archive():
    # P1 回归：weatherInfoList 非空但 step 全非法 → 空时间轴快照必须拒绝入库
    #（空快照一旦存档会被同 issue 幂等锁死，正常数据永远进不来）
    def routes(method, url, kwargs):
        if url == TILE_URL:
            return 200, TILE_PAYLOAD
        return 200, _info_payload([("x", "27.5", "0.0"), (None, "27.1", "0.0")])

    with pytest.raises(RuntimeError, match="拒绝入库"):
        FuxiC88Provider(session=RoutingSession(routes)).fetch_snapshot(Station)


def test_fetch_snapshot_structure():
    prov = FuxiC88Provider(session=_session())
    snap = prov.fetch_snapshot(Station, ["fuxi_c88"])
    assert snap["issue_iso"] == "2026-08-27T20:00"
    assert snap["station_id"] == "wuzhou"
    assert snap["source"] == "fuxi"
    assert snap["models"] == ["fuxi_c88"]
    assert len(snap["hourly_time"]) == EXPECTED_HOURS
    assert snap["hourly_time"][0] == "2026-08-27T21:00"   # 起报后 1h
    assert snap["hourly_time"][-1] == "2026-09-11T20:00"  # +360h
    data = snap["data"]["fuxi_c88"]
    assert len(data["temperature_2m"]) == EXPECTED_HOURS
    assert data["precipitation"][0] == 0.08


def test_tile_anchor_cached_across_stations():
    sess = _session()
    prov = FuxiC88Provider(session=sess)
    prov.fetch_snapshot(Station, ["fuxi_c88"])
    n_first = len(sess.calls)
    prov.fetch_snapshot(Station, ["fuxi_c88"])
    # 第二站不再请求 tile（锚点为产品级属性）
    assert not any(m == "GET" and u == TILE_URL for m, u, _ in sess.calls[n_first:])


def test_point_series_shorter_than_expected_warns(caplog):
    sess = _session(steps=[(i, "27.0", "0.0") for i in range(1, 49)])
    with caplog.at_level("WARNING"):
        FuxiC88Provider(session=sess).fetch_snapshot(Station, ["fuxi_c88"])
    assert "长时效可能被截断" in caplog.text


def test_http_400_is_deterministic_no_retry():
    def routes(method, url, kwargs):
        return 400, '{"msg":"bad"}'

    sess = RoutingSession(routes)
    with pytest.raises(RuntimeError, match="请求被拒"):
        FuxiC88Provider(session=sess, retries=3).fetch_snapshot(Station, ["fuxi_c88"])
    assert len(sess.calls) == 1  # tile 请求即被拒，无重试


def test_success_false_is_error():
    def routes(method, url, kwargs):
        if url == TILE_URL:
            return 200, json.dumps({"success": False, "msg": "系统繁忙"})
        return 200, _info_payload([(1, "27.5", "0.0")])

    with pytest.raises(RuntimeError, match="tile 响应异常"):
        FuxiC88Provider(session=RoutingSession(routes)).fetch_snapshot(Station)

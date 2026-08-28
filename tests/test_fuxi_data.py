"""伏羲确定性（FuxiDetProvider）单测。

mock 契约来自 2026-08 线上实测 + 官方页面文档：
- POST /initTime/isAvail body {modelId:"4", initTime:"YYYY-MM-DD"}（UTC 日期）
  → data: ["00","06","12","18"]（该日可用 UTC 小时）；游客可用。
- POST /queryWeatherInfo body {lon,lat,vars,initTime:"YYYY-MM-DD HH:00:00",model:"FuXi-Det"}
  headers Authorization=<token>（原始值无 Bearer）→ data{location,time_fcst,
  timestamp(UTC 时刻数组),var_names,units,values[i][j]}。
"""
import json
from datetime import datetime

import pytest
from conftest import FakeResp

from weather_eval.forecast.fuxi_data import (
    AVAIL_URL,
    QUERY_URL,
    FuxiDetProvider,
    candidate_dates,
    looks_like_running_accumulation,
    parse_avail_hours,
    parse_query_response,
    utc_now,
)

TOKEN = "tok-abc123"


def _avail_payload(hours):
    return json.dumps({"traceId": "x", "msg": "获取时次成功", "msgCode": "10000",
                       "data": hours, "success": True})


def _query_payload(var_names, units, timestamps, values, location=None):
    return json.dumps({
        "traceId": "x", "msg": "success", "msgCode": "10000",
        "data": {
            "location": location or [23.5, 111.3],
            "time_fcst": "2026-08-28 00:00:00",
            "timestamp": timestamps,
            "var_names": var_names,
            "units": units,
            "values": values,
        },
    }, ensure_ascii=False)


class RoutingSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls: list[tuple] = []

    def request(self, method, url, json=None, headers=None, timeout=None, **kw):
        self.calls.append((method, url, json, dict(headers or {})))
        s, t = self.routes(method, url, json, headers)
        return FakeResp(t, status_code=s)


class Station:
    id, name, lat, lon = "wuzhou", "梧州", 23.4783, 111.304


def _routes(avail_by_date, query_payload=None):
    def routes(method, url, body, headers):
        if url == AVAIL_URL:
            assert body["modelId"] == "4"          # 契约：字符串
            assert " " not in body["initTime"]     # 契约：仅日期
            return 200, _avail_payload(avail_by_date.get(body["initTime"], []))
        if url == QUERY_URL:
            assert headers.get("Authorization") == TOKEN  # 原始 token，无 Bearer
            return 200, query_payload
        return 404, "{}"
    return routes


def _query_payload_k():
    """t2m 单位 K（官方文档示例 m/s、K、Pa），tp 单位 mm，逐小时 4 点。"""
    return _query_payload(
        var_names=["t2m", "tp"], units=["K", "mm"],
        timestamps=["2026-08-28T01:00:00Z", "2026-08-28T02:00:00Z",
                    "2026-08-28T03:00:00Z", "2026-08-28T04:00:00Z"],
        values=[[300.15, 301.15, 299.15, 300.65], [0.0, 0.5, 1.2, 0.0]],
    )


def test_missing_token_raises():
    import os
    old = os.environ.pop("FUXI_DATA_TOKEN", None)
    try:
        with pytest.raises(RuntimeError, match="FUXI_DATA_TOKEN"):
            FuxiDetProvider()
    finally:
        if old is not None:
            os.environ["FUXI_DATA_TOKEN"] = old


def test_candidate_dates_descends():
    dates = candidate_dates(datetime(2026, 8, 28, 7), 3)
    assert dates == ["2026-08-28", "2026-08-27", "2026-08-26"]


def test_parse_avail_hours():
    assert parse_avail_hours(json.loads(_avail_payload(["00", "06", "12", "18"]))) \
        == ["00", "06", "12", "18"]
    assert parse_avail_hours(json.loads(_avail_payload([]))) == []
    with pytest.raises(RuntimeError, match="业务错误"):
        parse_avail_hours({"msgCode": "400", "msg": "日期格式错误"})


def test_parse_query_response_kelvin_to_celsius_and_bj_times():
    parsed = parse_query_response(json.loads(_query_payload_k()))
    assert parsed["time"] == ["2026-08-28T09:00", "2026-08-28T10:00",
                              "2026-08-28T11:00", "2026-08-28T12:00"]  # UTC+8
    assert parsed["t2m"] == [27.0, 28.0, 26.0, 27.5]  # 300.15-273.15=27.0
    assert parsed["tp"] == [0.0, 0.5, 1.2, 0.0]
    assert parsed["grid_lat"] == 23.5 and parsed["grid_lon"] == 111.3
    assert parsed["issue_echo"] == "2026-08-28 00:00:00"


def test_parse_query_response_celsius_passthrough():
    payload = _query_payload(["t2m", "tp"], ["℃", "mm"],
                             ["2026-08-28T01:00:00Z"], [[27.5], [0.3]])
    parsed = parse_query_response(json.loads(payload))
    assert parsed["t2m"] == [27.5]  # 单位已是 ℃，不换算


def test_parse_query_response_missing_vars():
    payload = _query_payload(["u10", "tp"], ["m/s", "mm"],
                             ["2026-08-28T01:00:00Z"], [[1.0], [0.0]])
    with pytest.raises(RuntimeError, match="缺 t2m"):
        parse_query_response(json.loads(payload))


def test_parse_query_response_bad_values_to_none():
    payload = _query_payload(["t2m", "tp"], ["K", "mm"],
                             ["2026-08-28T01:00:00Z", "2026-08-28T02:00:00Z"],
                             [["bad", None], [0.5, 1.0]])
    parsed = parse_query_response(json.loads(payload))
    assert parsed["t2m"] == [None, None]
    assert parsed["tp"] == [0.5, 1.0]


def test_millisecond_timestamp_and_offset_variants():
    # 契约漂移容忍：毫秒/小写 z/+00:00 均可解析（UTC）
    payload = _query_payload(["t2m", "tp"], ["K", "mm"],
                             ["2026-08-28T01:00:00.000Z", "2026-08-28T02:00:00z",
                              "2026-08-28T03:00:00+00:00"],
                             [[300.0, 301.0, 302.0], [0.0, 0.1, 0.2]])
    parsed = parse_query_response(json.loads(payload))
    assert parsed["time"] == ["2026-08-28T09:00", "2026-08-28T10:00", "2026-08-28T11:00"]


def test_unparseable_timestamp_column_skipped(caplog):
    # P0 回归：无法解析的时刻必须整列剔除，绝不允许空串时间键入库毒化评估
    payload = _query_payload(["t2m", "tp"], ["K", "mm"],
                             ["2026-08-28T01:00:00Z", "garbage", "2026-08-28T03:00:00Z"],
                             [[300.0, 999.0, 302.0], [0.0, 9.9, 0.2]])
    with caplog.at_level("WARNING"):
        parsed = parse_query_response(json.loads(payload))
    assert parsed["time"] == ["2026-08-28T09:00", "2026-08-28T11:00"]
    assert parsed["t2m"] == [pytest.approx(26.85), pytest.approx(28.85)]  # 300/302 K
    assert parsed["tp"] == [0.0, 0.2]   # 中列数值一并剔除
    assert "garbage" in caplog.text


def test_unit_missing_warns(caplog):
    payload = _query_payload(["t2m", "tp"], [None, "mm"],
                             ["2026-08-28T01:00:00Z"] * 8,
                             [[27.0] * 8, [0.0] * 8])
    with caplog.at_level("WARNING"):
        parsed = parse_query_response(json.loads(payload))
    assert parsed["t2m_unit"] is None
    assert "未提供 t2m 单位" in caplog.text


def test_kelvin_magnitude_sentinel(caplog):
    # 未声明 K 但数值普遍 >150（开尔文量级）→ 预警口径漂移
    payload = _query_payload(["t2m", "tp"], ["", "mm"],
                             [f"2026-08-28T0{h}:00:00Z" for h in range(1, 9)],
                             [[300.0 + i for i in range(8)], [0.0] * 8])
    with caplog.at_level("WARNING"):
        parse_query_response(json.loads(payload))
    assert "K 量级" in caplog.text


def test_400_no_retry():
    calls = {"n": 0}

    def routes(method, url, body, headers):
        calls["n"] += 1
        return 400, '{"msg":"bad request"}'

    prov = FuxiDetProvider(session=RoutingSession(routes), token=TOKEN,
                           now=datetime(2026, 8, 28, 7))
    with pytest.raises(RuntimeError, match="请求被拒"):
        prov._resolve_init_time()
    assert calls["n"] == 1  # 4xx 确定性失败，无重试


def test_non_zero_padded_hour_picks_latest():
    # P2 回归：服务端若返回非零填充小时串，按整数值取最晚轮次
    routes = _routes(avail_by_date={"2026-08-28": ["0", "6", "12", "18"]},
                     query_payload=_query_payload_k())
    prov = FuxiDetProvider(session=RoutingSession(routes), token=TOKEN,
                           now=datetime(2026, 8, 28, 7))
    assert prov._resolve_init_time() == "2026-08-28 18:00:00"


def test_avail_response_without_msgcode_raises():
    routes = _routes(avail_by_date={})
    sess = RoutingSession(routes)
    with pytest.raises(RuntimeError, match="无 msgCode"):
        parse_avail_hours({"success": False, "msg": "token 无效", "data": None})


def test_resolve_init_time_falls_back_to_earlier_date():
    # 今天（UTC）无可用轮次，昨天有 → 取昨日最大小时 18
    routes = _routes(avail_by_date={"2026-08-27": ["00", "06", "12", "18"]},
                     query_payload=_query_payload_k())
    sess = RoutingSession(routes)
    prov = FuxiDetProvider(session=sess, token=TOKEN,
                           now=datetime(2026, 8, 28, 7))
    assert prov._resolve_init_time() == "2026-08-27 18:00:00"


def test_resolve_init_time_picks_latest_hour_of_today():
    routes = _routes(avail_by_date={"2026-08-28": ["00", "06"]},
                     query_payload=_query_payload_k())
    prov = FuxiDetProvider(session=RoutingSession(routes), token=TOKEN,
                           now=datetime(2026, 8, 28, 7))
    assert prov._resolve_init_time() == "2026-08-28 06:00:00"


def test_resolve_init_time_exhausted():
    routes = _routes(avail_by_date={})
    prov = FuxiDetProvider(session=RoutingSession(routes), token=TOKEN,
                           now=datetime(2026, 8, 28, 7))
    with pytest.raises(RuntimeError, match="均无可用起报"):
        prov._resolve_init_time()


def test_fetch_snapshot_end_to_end():
    routes = _routes(avail_by_date={"2026-08-28": ["00"]},
                     query_payload=_query_payload_k())
    sess = RoutingSession(routes)
    snap = FuxiDetProvider(session=sess, token=TOKEN,
                           now=datetime(2026, 8, 28, 7)).fetch_snapshot(
        Station, ["fuxi_det"])
    # 请求契约
    method, url, body, headers = [c for c in sess.calls if c[1] == QUERY_URL][0]
    assert body == {"lon": 111.304, "lat": 23.4783, "vars": ["t2m", "tp"],
                    "initTime": "2026-08-28 00:00:00", "model": "FuXi-Det"}
    # 快照结构
    assert snap["issue_iso"] == "2026-08-28T08:00"  # UTC 00z → 北京时 08
    assert snap["models"] == ["fuxi_det"]
    assert snap["source"] == "fuxi-data"
    assert snap["grid_lat"] == 23.5                 # 服务端吸附坐标
    assert snap["hourly_time"][0] == "2026-08-28T09:00"
    d = snap["data"]["fuxi_det"]
    assert d["temperature_2m"][0] == 27.0
    assert d["precipitation"][1] == 0.5
    assert len(d["temperature_2m"]) == len(snap["hourly_time"])


def test_issue_echo_mismatch_warns(caplog):
    payload = json.loads(_query_payload_k())
    payload["data"]["time_fcst"] = "2026-08-27 12:00:00"
    routes = _routes(avail_by_date={"2026-08-28": ["00"]}, query_payload=json.dumps(payload))
    with caplog.at_level("WARNING"):
        FuxiDetProvider(session=RoutingSession(routes), token=TOKEN,
                        now=datetime(2026, 8, 28, 7)).fetch_snapshot(Station)
    assert "不一致" in caplog.text


def test_running_accumulation_guard_warns(caplog):
    # 单调不减且显著增长 → 提示口径存疑
    vals = [300.0 + i for i in range(24)]
    payload = _query_payload(["t2m", "tp"], ["K", "mm"],
                             [f"2026-08-28T{h:02d}:00:00Z" for h in range(1, 25)],
                             [vals, list(range(0, 24))])
    routes = _routes(avail_by_date={"2026-08-28": ["00"]}, query_payload=payload)
    with caplog.at_level("WARNING"):
        FuxiDetProvider(session=RoutingSession(routes), token=TOKEN,
                        now=datetime(2026, 8, 28, 7)).fetch_snapshot(Station)
    assert "单调不减" in caplog.text


def test_looks_like_running_accumulation():
    assert looks_like_running_accumulation(list(range(0, 24))) is True
    assert looks_like_running_accumulation([0.1] * 24) is False      # 无增长
    assert looks_like_running_accumulation([0.5, 0.0, 2.0, 0.1] * 4) is False
    assert looks_like_running_accumulation([None] * 24) is False


def test_401_gives_actionable_error():
    def routes(method, url, body, headers):
        return 401, '{"detail":"请求头中缺少id字段"}'

    prov = FuxiDetProvider(session=RoutingSession(routes), token=TOKEN,
                           now=datetime(2026, 8, 28, 7))
    with pytest.raises(RuntimeError, match="鉴权失败"):
        prov.fetch_snapshot(Station, ["fuxi_det"])

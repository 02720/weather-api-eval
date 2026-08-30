"""中科天机（TianjiProvider）单测。

mock 契约来自 2026-08 线上实测：
- 单点查询 GET /meteorological/spas/single-point/query，参数 lat/lon/mode/baseTime/
  production/region/factorCode；游客态无鉴权。
- baseTime 与 forecastTimeString 均为北京时 YYYYMMDDHH；起报后 1h 起逐小时步进。
- 最新轮次未发布时返回 200 + 空 forecastDetails（探测回退的依据）。
"""
import json
from datetime import datetime

import pytest

from conftest import FakeResp

from weather_eval.forecast.tianji import (
    ENDPOINT,
    MODEL_SPECS,
    TianjiProvider,
    _parse_tj_response,
    _value_of,
    candidate_base_times,
)

STATION_LAT, STATION_LON = 23.4783, 111.304


class RoutingSession:
    """按请求参数路由响应的假 session（记录每次调用的参数）。"""

    def __init__(self, responder):
        self.responder = responder  # fn(params: dict) -> (status_code, json_text)
        self.calls: list[dict] = []

    def get(self, url, params=None, headers=None, timeout=None):
        p = dict(params or {})
        self.calls.append(p)
        status, text = self.responder(p)
        return FakeResp(text, status_code=status)


def _details(start_yyyymmddhh, values, start_utc_iso):
    """构造 forecastDetails：逐小时（北京时）递增，value 为标量数组。"""
    from datetime import datetime, timedelta

    base = datetime.strptime(start_yyyymmddhh, "%Y%m%d%H")
    out = []
    for i, v in enumerate(values):
        t = base + timedelta(hours=i + 1)  # 起报后 1h 起
        utc = datetime.strptime(start_utc_iso, "%Y-%m-%dT%H:%M:%S.000+00:00") + timedelta(hours=i + 1)
        out.append({
            "forecastTimeString": t.strftime("%Y%m%d%H"),
            "forecastTime": utc.strftime("%Y-%m-%dT%H:%M:%S.000+00:00"),
            "value": [v],
            "icon": None,
        })
    return out


def _payload(mode, factor, details, base="2026082708"):
    return json.dumps({
        "code": 200, "message": "成功",
        "data": {
            "lon": STATION_LON, "lat": STATION_LAT,
            "baseTimeString": base, "baseTime": "2026-08-27T00:00:00.000+00:00",
            "mode": mode,
            "forecast": [{"factorCode": factor, "forecastDetails": details}],
        },
    }, ensure_ascii=False)


def _ok_responder(temp_values=(27.0, 27.5, 28.0), precip_values=(0.0, 0.1, None)):
    """默认响应器：任意合法查询都返回 3 点温度 + 3 点降水（首点起报 2026082708）。"""
    def responder(p):
        factor = p["factorCode"]
        if factor in ("tmp2m", "t2mz"):
            return 200, _payload(p["mode"], factor,
                                 _details("2026082708", list(temp_values),
                                          "2026-08-27T00:00:00.000+00:00"))
        return 200, _payload(p["mode"], factor,
                             _details("2026082708", list(precip_values),
                                      "2026-08-27T00:00:00.000+00:00"))
    return responder


class _Station:
    id = "s1"
    lat = STATION_LAT
    lon = STATION_LON


NOW = datetime(2026, 8, 27, 13, 0)  # 北京时；首个候选轮次 2026082708


def test_candidate_base_times():
    assert candidate_base_times(datetime(2026, 8, 27, 6, 0)) == [
        "2026082620", "2026082608", "2026082520", "2026082508"]
    assert candidate_base_times(datetime(2026, 8, 27, 8, 0)) == [
        "2026082708", "2026082620", "2026082608", "2026082520"]
    assert candidate_base_times(datetime(2026, 8, 27, 20, 30)) == [
        "2026082720", "2026082708", "2026082620", "2026082608"]


def test_fetch_returns_independent_snapshot_per_model():
    sess = RoutingSession(_ok_responder())
    prov = TianjiProvider(session=sess, now=NOW, retries=0)
    snaps = prov.fetch_snapshot(_Station(), list(MODEL_SPECS))

    assert len(snaps) == len(MODEL_SPECS)
    by_model = {s["models"][0]: s for s in snaps}
    assert set(by_model) == set(MODEL_SPECS)

    for name, s in by_model.items():
        # 每份快照：独立 issue（=起报轮次）、北京时整点时间轴、单模型 data
        assert s["issue_iso"] == "2026-08-27T08:00"
        assert s["source"] == "tianji"
        assert s["station_id"] == "s1"
        assert s["elevation"] is None
        assert s["grid_lat"] == STATION_LAT and s["grid_lon"] == STATION_LON
        times = s["hourly_time"]
        assert times == ["2026-08-27T09:00", "2026-08-27T10:00", "2026-08-27T11:00"]
        data = s["data"][name]
        assert data["temperature_2m"] == [27.0, 27.5, 28.0]
        # 降水 value 中的 None 保留为 None（缺测不伪装成 0）
        assert data["precipitation"] == [0.0, 0.1, None]
        assert list(s["data"].keys()) == [name]


def test_probe_cached_across_stations():
    sess = RoutingSession(_ok_responder())
    prov = TianjiProvider(session=sess, now=NOW, retries=0)
    prov.fetch_snapshot(_Station(), ["tj_t2"])
    n_first = len(sess.calls)
    # 第二个站点：起报探测应命中实例缓存，仅温度+降水两次请求
    class S2(_Station):
        id = "s2"
    prov.fetch_snapshot(S2(), ["tj_t2"])
    assert len(sess.calls) == n_first + 2


def test_base_time_fallback_when_latest_cycle_not_published():
    def responder(p):
        if p["baseTime"] == "2026082708":
            # 最新轮次尚未发布：200 + 空序列
            return 200, _payload(p["mode"], p["factorCode"], [], base="2026082708")
        return 200, _payload(p["mode"], p["factorCode"],
                             _details("2026082620", [25.0, 25.5],
                                      "2026-08-26T12:00:00.000+00:00"),
                             base="2026082620")

    sess = RoutingSession(responder)
    prov = TianjiProvider(session=sess, now=NOW, retries=0)
    snaps = prov.fetch_snapshot(_Station(), ["tj_t2"])
    assert len(snaps) == 1
    assert snaps[0]["issue_iso"] == "2026-08-26T20:00"
    assert snaps[0]["hourly_time"] == ["2026-08-26T21:00", "2026-08-26T22:00"]
    # 探测请求的 baseTime 逐轮回退
    probed = [c["baseTime"] for c in sess.calls]
    assert probed[0] == "2026082708" and probed[1] == "2026082620"


def test_value_normalization():
    assert _value_of({"value": [27.0]}) == 27.0
    assert _value_of({"value": []}) is None
    assert _value_of({"value": [None]}) is None
    assert _value_of({"value": ["x"]}) is None
    assert _value_of({"value": [float("nan")]}) is None
    assert _value_of({}) is None
    assert _value_of(None) is None


def test_parse_response_error_and_missing_factor():
    with pytest.raises(RuntimeError):
        _parse_tj_response({"code": 11001, "message": "缺失参数:[lat]"}, "tmp2m")
    with pytest.raises(RuntimeError):
        _parse_tj_response(None, "tmp2m")
    # 要素缺失 → 空序列（用于起报轮次探测）
    out = _parse_tj_response(json.loads(_payload("t2", "t2mz", [])), "tmp2m")
    assert out["time"] == [] and out["value"] == []


def test_http_4xx_no_retry():
    sess = RoutingSession(lambda p: (403, json.dumps({"detail": "forbidden"})))
    prov = TianjiProvider(session=sess, now=NOW, retries=3)
    with pytest.raises(RuntimeError, match="HTTP 403"):
        prov.fetch_snapshot(_Station(), ["tj_t2"])
    assert len(sess.calls) == 1  # 4xx 不重试


def test_unknown_model_rejected():
    prov = TianjiProvider(session=RoutingSession(_ok_responder()), now=NOW)
    with pytest.raises(RuntimeError):
        prov.fetch_snapshot(_Station(), ["not_a_tj_model"])


def test_endpoint_url():
    assert ENDPOINT.startswith("https://www.tjweather.com/")
    assert "/spas/single-point/query" in ENDPOINT


# --------------------------------------------------------------------------
# 对抗式审查补充用例
# --------------------------------------------------------------------------

# 硬编码的线上实测契约（与 MODEL_SPECS 独立，防 mock 假绿：映射漂移即红）
_EXPECTED_CONTRACT = {
    ("nextgen", "c1km", "tmp2m"), ("nextgen", "c2_5km", "pratesfc"),
    ("early", "t2", "t2mz"), ("early", "t2", "pratesfc"),
    ("late", "t2", "t2mz"), ("late", "t2", "pratesfc"),
    ("t1_ai", "t1", "t2mz"), ("t1_ai", "t1", "pratesfc"),
    ("early", "t1h", "t2mz"), ("early", "t1h", "pratesfc"),
}


def test_request_params_follow_online_contract():
    """每个请求的 mode/production/factorCode/region 都必须匹配线上实测契约。"""
    seen = []

    def responder(p):
        key = (p["mode"], p["production"], p["factorCode"])
        seen.append(key)
        assert p["region"] == "global"
        assert len(p["baseTime"]) == 10 and p["baseTime"].isdigit()
        assert p["lat"] == STATION_LAT and p["lon"] == STATION_LON
        assert key in _EXPECTED_CONTRACT, f"请求参数组合偏离线上契约: {key}"
        return _ok_responder()(p)

    sess = RoutingSession(responder)
    prov = TianjiProvider(session=sess, now=NOW, retries=0)
    prov.fetch_snapshot(_Station(), list(MODEL_SPECS))
    # 探测请求（温度要素）也必须合规
    assert set(seen) <= _EXPECTED_CONTRACT


def test_probe_exhaustion_raises():
    """4 个候选起报轮次全空 → 该模型失败；仅请求该模型时报错上抛。"""
    sess = RoutingSession(lambda p: (200, _payload(p["mode"], p["factorCode"], [])))
    prov = TianjiProvider(session=sess, now=NOW, retries=0)
    with pytest.raises(RuntimeError, match="全部模型抓取失败"):
        prov.fetch_snapshot(_Station(), ["tj_t2"])
    # 每模型 1 次温度探测 ×4 轮（探测失败无降水请求）
    assert len(sess.calls) == 4


def test_partial_model_failure_keeps_other_models():
    """t1_ai 产品停发（探测穷尽）→ 其余 4 模型快照照常返回。"""
    def responder(p):
        if p["mode"] == "t1_ai":
            return 200, _payload(p["mode"], p["factorCode"], [])
        return _ok_responder()(p)

    sess = RoutingSession(responder)
    prov = TianjiProvider(session=sess, now=NOW, retries=0)
    snaps = prov.fetch_snapshot(_Station(), list(MODEL_SPECS))
    assert {s["models"][0] for s in snaps} == set(MODEL_SPECS) - {"tj_t1"}
    # 已知失败模型在后续站点被跳过（不再发起任何请求）
    n = len(sess.calls)

    class S2(_Station):
        id = "s2"
    snaps2 = prov.fetch_snapshot(S2(), list(MODEL_SPECS))
    assert {s["models"][0] for s in snaps2} == set(MODEL_SPECS) - {"tj_t1"}
    assert len(sess.calls) == n + 8  # 4 模型 × (温度+降水)


def test_precip_product_empty_warns_but_archives_temperature(caplog):
    """降水产品（另一条产品线）空序列 → 温度照常入库、降水全 None，且 WARNING 可见。"""
    import logging

    def responder(p):
        if p["factorCode"] == "pratesfc":
            return 200, _payload(p["mode"], p["factorCode"], [])
        return _ok_responder()(p)

    sess = RoutingSession(responder)
    prov = TianjiProvider(session=sess, now=NOW, retries=0)
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.tianji"):
        snaps = prov.fetch_snapshot(_Station(), ["tj_km_fusion"])
    assert len(snaps) == 1
    assert snaps[0]["data"]["tj_km_fusion"]["temperature_2m"] == [27.0, 27.5, 28.0]
    assert snaps[0]["data"]["tj_km_fusion"]["precipitation"] == [None, None, None]
    assert any("降水产品" in r.getMessage() for r in caplog.records)


def test_candidate_base_times_boundaries():
    """月末/年末跨天边界。"""
    assert candidate_base_times(datetime(2026, 9, 1, 5, 0))[0] == "2026083120"
    assert candidate_base_times(datetime(2027, 1, 1, 0, 30))[0] == "2026123120"
    assert candidate_base_times(datetime(2026, 3, 1, 8, 0))[0] == "2026030108"


def test_http_500_business_code_no_retry():
    """服务端把确定性错误包在 HTTP 500 + 业务 code 里：不应重试。"""
    sess = RoutingSession(
        lambda p: (500, json.dumps({"code": 11001, "message": "缺失参数:[lat]"})))
    prov = TianjiProvider(session=sess, now=NOW, retries=3)
    with pytest.raises(RuntimeError, match="11001"):
        prov.fetch_snapshot(_Station(), ["tj_t2"])
    assert len(sess.calls) == 1


def test_http_500_transient_retries_then_success():
    """无业务错误码的 5xx 视为瞬时故障：退避重试后成功。"""
    state = {"n": 0}

    def responder(p):
        state["n"] += 1
        if state["n"] == 1:
            return 500, "<html>oops</html>"
        return _ok_responder()(p)

    sess = RoutingSession(responder)
    prov = TianjiProvider(session=sess, now=NOW, retries=2)
    snaps = prov.fetch_snapshot(_Station(), ["tj_t2"])
    assert len(snaps) == 1
    # 1 次探测失败 + 1 次探测重试成功（序列复用）+ 1 次降水请求
    assert state["n"] == 3


def test_cli_fetch_forecast_saves_list_snapshots(tmp_path, monkeypatch):
    """CLI 对 list 返回形态的集成：按模型逐份存档。"""
    import argparse

    from weather_eval import __main__ as cli

    class FakeCfg:
        models = ["tj_t2"]
        stations = [_Station()]

    class FakeProv:
        def fetch_snapshot(self, station, models):
            return [{
                "issue_iso": "2026-08-27T08:00", "station_id": station.id,
                "source": "tianji", "models": ["tj_t2"],
                "grid_lat": 1.0, "grid_lon": 2.0, "elevation": None,
                "requested_lat": 1.0, "requested_lon": 2.0,
                "hourly_time": ["2026-08-27T09:00"],
                "data": {"tj_t2": {"temperature_2m": [27.0], "precipitation": [0.0]}},
            }]

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(cli, "load_config", lambda p: FakeCfg())
    monkeypatch.setattr(cli, "_build_provider", lambda source, cfg: (FakeProv(), ["tj_t2"]))
    rc = cli.cmd_fetch_forecast(argparse.Namespace(config=None, source="tianji"))
    assert rc == 0
    files = list((tmp_path / "forecasts" / "s1" / "tj_t2").glob("*.json"))
    assert len(files) == 1
    assert files[0].name == "2026-08-27T0800.json"


def test_echoed_base_time_wins_over_requested():
    # L4 回归：服务端"就近替换"（请求 08 轮、回显 02 轮的数据）时，
    # issue_iso 必须取回显轮次——否则 lead 为负、样本全被评估引擎丢弃。
    def responder(p):
        factor = p["factorCode"]
        vals = (27.0, 27.5, 28.0) if factor in ("tmp2m", "t2mz") else (0.0, 0.1, None)
        return 200, _payload(p["mode"], factor,
                             _details("2026082702", list(vals),
                                      "2026-08-26T18:00:00.000+00:00"),
                             base="2026082702")
    sess = RoutingSession(responder)
    prov = TianjiProvider(session=sess, now=NOW, retries=0)
    snaps = prov.fetch_snapshot(_Station(), ["tj_t2"])
    assert snaps[0]["issue_iso"] == "2026-08-27T02:00"
    assert snaps[0]["hourly_time"] == [
        "2026-08-27T03:00", "2026-08-27T04:00", "2026-08-27T05:00"]


def test_malformed_echo_ignored_keeps_requested_base():
    # 第二轮审查回归：回显 baseTimeString 非 YYYYMMDDHH 格式时必须忽略，
    # 绝不让垃圾回显污染 issue_iso（否则 parse_iso 在评估期崩溃）
    def responder(p):
        factor = p["factorCode"]
        vals = (27.0, 27.5, 28.0) if factor in ("tmp2m", "t2mz") else (0.0, 0.1, None)
        return 200, _payload(p["mode"], factor,
                             _details("2026082708", list(vals),
                                      "2026-08-27T00:00:00.000+00:00"),
                             base="not-a-time")
    sess = RoutingSession(responder)
    prov = TianjiProvider(session=sess, now=NOW, retries=0)
    snaps = prov.fetch_snapshot(_Station(), ["tj_t2"])
    assert snaps[0]["issue_iso"] == "2026-08-27T08:00"   # 回退为请求轮次

"""AccuWeather 逐小时预报接入测试。

mock 契约依据官方文档（developer.accuweather.com，2026-08-29 核对现行版）：
- 鉴权：Key 经 Authorization: Bearer 头传递（2026-06-10 修订契约；
  旧版 ?apikey= query 参数已停用——实测有效 Key 走 query 也 401）；
- Locations API geoposition search：q 为"纬度,经度"，返回 Key/GeoPosition；
- Forecast API v1 hourly：URL 路径 /forecasts/v1/hourly/{hours}hour/{locationKey}，
  details=true 才有 TotalLiquid（该小时液态降水总量，metric=true 时 mm），
  DateTime 为 ISO8601 带当地偏移；
- 免费档 50 次/天，超限 503；超出订阅的时效档位 403/400。

覆盖：时间换算与下取整、单位自适应换算（F→℃/inch→mm/未知单位）、
降水 -1h 移位口径（含缺口防错配）、最近城市吸附留档、档位梯子回退与
跨运行状态缓存、401 快速失败、503 退避穷尽后的配额熔断、日志脱敏、
CLI 分发与 Open-Meteo 排除链、与评估引擎端到端配对。
"""
import json
import logging
from datetime import datetime, timedelta

import pytest
import requests
from conftest import FakeResp

from weather_eval.forecast.accuweather import (
    DEFAULT_HOURS,
    KEY_ENV,
    LOCATION_URL,
    MODEL_NAME,
    TIERS,
    AccuWeatherProvider,
    _parse_dt,
    _tier_for,
    parse_hourly_payload,
)
from weather_eval import storage
from weather_eval.evaluate import build_report
from weather_eval.timeutil import iso

KEY = "FAKEAWKEY123"


def _station():
    class S:
        id = "s1"
        lat = 23.4783
        lon = 111.304
    return S()


# ----------------------------------------------------------------- 夹具
LOCATION_PAYLOAD = {
    "Version": 1, "Key": "329260", "Type": "City", "Rank": 15,
    "LocalizedName": "梧州", "EnglishName": "Wuzhou",
    "TimeZone": {"Code": "CST", "Name": "Asia/Shanghai", "GmtOffset": 8.0},
    "GeoPosition": {
        "Latitude": 23.477, "Longitude": 111.279,
        "Elevation": {"Imperial": {"Value": 36.0, "Unit": "ft"},
                      "Metric": {"Value": 11.0, "Unit": "m"}},
    },
}


def _fc_payload(base_bj, temps, liquids, unit_t="C", unit_l="mm",
                include_liquid=True):
    """构造逐小时预报数组：DateTime 带当地偏移，Value 为 number。

    temps/liquids 中 None 模拟缺测（Value 为 null 或 TotalLiquid 键缺失）。"""
    entries = []
    for i, (tv, lv) in enumerate(zip(temps, liquids)):
        t = (base_bj + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
        ent = {
            "DateTime": f"{t}:00+08:00",
            "EpochDateTime": 1787900000 + i * 3600,
            "WeatherIcon": 1, "IconPhrase": " Mostly sunny", "IsDaylight": True,
            "HasPrecipitation": bool(lv), "Temperature": {"Value": tv, "Unit": unit_t},
        }
        if include_liquid:
            ent["TotalLiquid"] = None if lv is None else {"Value": lv, "Unit": unit_l}
        entries.append(ent)
    return entries


class RoutingSession:
    """按 URL 分发响应的假 session，记录全部请求用于断言契约。"""

    def __init__(self, handler):
        self.handler = handler
        self.calls = []

    def get(self, url, params=None, headers=None, **kwargs):
        self.calls.append({"url": url, "params": params, "headers": headers})
        payload, status = self.handler(url, params)
        return FakeResp(payload if isinstance(payload, str) else json.dumps(payload),
                        status)


def _ok(payload):
    return payload, 200


def _routes(location=_ok(LOCATION_PAYLOAD), forecast=None):
    """默认路由：定位成功 + 预报走 forecast(url, params)。"""
    def handler(url, params):
        if "geoposition" in url:
            return location(url, params) if callable(location) else location
        return forecast(url, params)
    return handler


BASE = datetime(2026, 8, 28, 15, 0)


def _provider(routes, **kw):
    kw.setdefault("retries", 0)
    return AccuWeatherProvider(api_key=kw.pop("key", KEY),
                               session=RoutingSession(routes), **kw)


@pytest.fixture(autouse=True)
def _isolated_data_root(tmp_path, monkeypatch):
    """每个用例独立的数据根（快照/观测存档互不污染）。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))


# ----------------------------------------------------------------- 时间解析
def test_parse_dt_converts_to_beijing_and_floors():
    # +08:00 直接转北京时并下取整
    assert _parse_dt("2026-08-28T15:10:30+08:00").strftime("%Y-%m-%dT%H:%M") == "2026-08-28T15:00"
    # 文档中的两数字偏移形态 +08（3.12 fromisoformat 原生支持）
    assert _parse_dt("2026-08-28T07:10+08").strftime("%Y-%m-%dT%H:%M") == "2026-08-28T07:00"
    # 其他偏移（如海外属地城市）同样换算：00:30+00:00 -> 08:00 北京时
    assert _parse_dt("2026-08-28T00:30:00+00:00").strftime("%Y-%m-%dT%H:%M") == "2026-08-28T08:00"
    assert _parse_dt("2026-08-28T15:00:00Z").strftime("%Y-%m-%dT%H:%M") == "2026-08-28T23:00"


def test_parse_dt_rejects_garbage():
    for bad in (None, 5, "", "  ", "not-a-time"):
        with pytest.raises(ValueError):
            _parse_dt(bad)


# ----------------------------------------------------------------- 档位吸附
def test_tier_for_snaps_to_official_tiers():
    assert [_tier_for(h) for h in (120, 119, 73, 72, 49, 25, 24, 13, 1, 0)] == \
           [120, 72, 72, 72, 48, 24, 24, 12, 1, 1]


# ----------------------------------------------------------------- 主路解析
def test_fetch_full_snapshot(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    temps = [30.0, 29.5, 29.0]
    liquids = [0.0, 0.0, 1.2]
    src = _provider(_routes(forecast=lambda u, p: _ok(
        _fc_payload(BASE, temps, liquids))))

    class St:
        id, lat, lon = "s1", 23.4783, 111.304

    snap = src.fetch_snapshot(St(), [MODEL_NAME])
    assert snap["source"] == "accuweather"
    assert snap["models"] == [MODEL_NAME]
    assert snap["issue_iso"] == "2026-08-28T15:00"
    assert snap["hourly_time"] == ["2026-08-28T15:00", "2026-08-28T16:00", "2026-08-28T17:00"]
    assert snap["data"][MODEL_NAME]["temperature_2m"] == temps
    # 降水整体后移 1 小时：slot 15:00 无对应窗记 None，16:00/17:00 取前一窗
    assert snap["data"][MODEL_NAME]["precipitation"] == [None, 0.0, 0.0]
    # 最近城市吸附留档：grid 为城市坐标、requested 为站点坐标、距离量化偏离
    assert snap["grid_lat"] == 23.477 and snap["grid_lon"] == 111.279
    assert snap["requested_lat"] == 23.4783 and snap["requested_lon"] == 111.304
    assert snap["location_key"] == "329260"
    assert snap["location_name"] == "梧州"
    assert 0 < snap["location_distance_km"] < 5
    assert snap["elevation"] == 11.0
    assert snap["tier"] == DEFAULT_HOURS
    assert snap["hours"] == 3  # 实际点数，而非请求档位
    assert "TotalLiquid" in snap["precip_alignment"]


def test_request_contract(monkeypatch):
    """锁定请求契约：定位 q=纬度,经度；预报路径档位/locationKey、
    details/metric 为字符串 true、鉴权走 Authorization: Bearer 头且
    绝不回落 query 参数（旧契约已停用）、UA 固定。"""
    monkeypatch.delenv(KEY_ENV, raising=False)
    sess = RoutingSession(_routes(forecast=lambda u, p: _ok(
        _fc_payload(BASE, [30.0], [0.0]))))
    AccuWeatherProvider(api_key=KEY, session=sess, retries=0).fetch_snapshot(_station())

    loc_call, fc_call = sess.calls[0], sess.calls[1]
    assert loc_call["url"] == LOCATION_URL
    assert loc_call["params"]["q"] == "23.4783,111.304"   # 纬度在前（与和风 v7 相反）
    assert "apikey" not in loc_call["params"]
    assert loc_call["headers"]["Authorization"] == f"Bearer {KEY}"
    assert fc_call["url"] == f"https://dataservice.accuweather.com/forecasts/v1/hourly/{DEFAULT_HOURS}hour/329260"
    assert fc_call["params"]["details"] == "true"          # 布尔必须传字符串，requests 的 True 会变 "True"
    assert fc_call["params"]["metric"] == "true"
    assert "apikey" not in fc_call["params"]
    assert fc_call["headers"]["Authorization"] == f"Bearer {KEY}"
    assert fc_call["headers"]["User-Agent"] == "weather-api-eval/0.1 (+https://github.com/)"


def test_key_whitespace_stripped_into_bearer_header(monkeypatch):
    """CI Secret/.env 常携带首尾空白或换行：必须剥除后再进 Authorization 头。"""
    monkeypatch.delenv(KEY_ENV, raising=False)
    sess = RoutingSession(_routes(forecast=lambda u, p: _ok(
        _fc_payload(BASE, [30.0], [0.0]))))
    prov = AccuWeatherProvider(api_key=f"  {KEY}\n", session=sess, retries=0)
    assert prov.key == KEY
    prov.fetch_snapshot(_station())
    assert all(c["headers"]["Authorization"] == f"Bearer {KEY}"
               for c in sess.calls)


def test_env_key_whitespace_stripped(monkeypatch):
    monkeypatch.setenv(KEY_ENV, " from_env\n")
    assert AccuWeatherProvider().key == "from_env"


def test_key_with_control_chars_rejected_cleanly():
    """带控制字符的 Key 会在 requests 头编码处炸出天书异常——提前给出可操作错误。
    Cc 类须覆盖 C0(0x00-0x1F)/DEL(0x7F)/C1(0x80-0x9F) 全部码位。"""
    # \x85(NEL) 属 Unicode 空白、会被 strip() 先剥除，不算到达头的控制字符
    for bad in (KEY + "\x00", KEY + "\x7f", KEY + "\x9f"):
        with pytest.raises(RuntimeError, match="控制字符"):
            AccuWeatherProvider(api_key=bad)


def test_precip_shift_gap_does_not_misassign():
    """序列缺口时，降水必须按整点键回填：T3 槽不得错拿 T1 窗的雨。"""
    payload = _fc_payload(BASE, [30.0, 29.5, None, 28.5], [0.0, 0.0, None, 2.0])
    # 删掉 T2 条目制造缺口
    del payload[2]
    out = parse_hourly_payload(payload, "s1", DEFAULT_HOURS)
    assert out["time"] == ["2026-08-28T15:00", "2026-08-28T16:00", "2026-08-28T18:00"]
    # slot15=None、slot16=T15 窗、slot18：T17 缺失 -> None（而非 T16 窗）
    assert out["precipitation"] == [None, 0.0, None]
    assert out["temperature_2m"] == [30.0, 29.5, 28.5]


def test_duplicate_times_keep_first():
    """同整点重复条目保留首见：温度取 30 而非 99，降水窗取 0.0 而非 9.9。"""
    payload = _fc_payload(BASE, [30.0, 29.5, 99.0, 28.5], [0.0, 0.0, 9.9, 2.0])
    payload[2]["DateTime"] = payload[1]["DateTime"]  # 第 3 条改为与 16:00 同刻
    out = parse_hourly_payload(payload, "s1", DEFAULT_HOURS)
    assert out["time"] == ["2026-08-28T15:00", "2026-08-28T16:00", "2026-08-28T18:00"]
    assert out["temperature_2m"] == [30.0, 29.5, 28.5]   # 99.0 被丢弃（保首见）
    assert out["precipitation"] == [None, 0.0, None]     # slot16 取 0.0 而非 9.9


def test_truncated_response_warns(caplog, monkeypatch):
    """档位 120h 但仅返回 24 点 -> 以 WARNING 暴露时效被截断。"""
    monkeypatch.delenv(KEY_ENV, raising=False)
    payload = _fc_payload(BASE, [30.0] * 24, [0.0] * 24)
    src = _provider(_routes(forecast=lambda u, p: _ok(payload)))
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.accuweather"):
        snap = src.fetch_snapshot(_station())
    assert len(snap["hourly_time"]) == 24
    assert any("24 个逐小时点" in r.message for r in caplog.records)


def test_empty_forecast_raises():
    src = _provider(_routes(forecast=lambda u, p: _ok([])))
    with pytest.raises(RuntimeError, match="响应为空"):
        src.fetch_snapshot(_station())


def test_unparseable_datetimes_only_raises():
    payload = _fc_payload(BASE, [30.0, 29.5], [0.0, 0.0])
    payload[0]["DateTime"] = "garbage"
    payload[1]["DateTime"] = "garbage"
    src = _provider(_routes(forecast=lambda u, p: _ok(payload)))
    with pytest.raises(RuntimeError, match="未解析出任何有效"):
        src.fetch_snapshot(_station())


# ----------------------------------------------------------------- 单位自适应
def test_imperial_units_converted(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    payload = _fc_payload(BASE, [86.0, None], [0.1, None], unit_t="F", unit_l="Inch")
    out = parse_hourly_payload(payload, "s1", DEFAULT_HOURS)
    assert out["temperature_2m"][0] == pytest.approx(30.0)
    assert out["precipitation"][1] == pytest.approx(0.1 * 25.4)


def test_unknown_unit_becomes_none_and_warns(caplog, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    payload = _fc_payload(BASE, [30.0, 29.0], [0.1, 0.0], unit_t="K", unit_l="gallon")
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.accuweather"):
        out = parse_hourly_payload(payload, "s1", DEFAULT_HOURS)
    assert out["temperature_2m"] == [None, None]
    assert out["precipitation"] == [None, None]
    assert any("单位 'k' 未知" in r.message and "按缺测处理" in r.message
               for r in caplog.records)
    assert any("单位 'gallon' 未知" in r.message for r in caplog.records)


def test_missing_unit_assumed_metric_and_warns(caplog, monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    payload = _fc_payload(BASE, [30.0, 29.0], [0.1, 0.0])
    for ent in payload:
        del ent["Temperature"]["Unit"]
        del ent["TotalLiquid"]["Unit"]
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.accuweather"):
        out = parse_hourly_payload(payload, "s1", DEFAULT_HOURS)
    assert out["temperature_2m"] == [30.0, 29.0]   # 按公制保留数据
    assert out["precipitation"][1] == 0.1
    assert any("未声明单位" in r.message and "气温" in r.message for r in caplog.records)


def test_all_missing_values_sentinels(caplog, monkeypatch):
    """details=true 未生效 / 数值全 null 都必须高声告警，不得静默产出空数据。"""
    monkeypatch.delenv(KEY_ENV, raising=False)
    no_liquid = _fc_payload(BASE, [30.0, 29.0], [None, None], include_liquid=False)
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.accuweather"):
        out = parse_hourly_payload(no_liquid, "s1", DEFAULT_HOURS)
    assert out["precipitation"] == [None, None]
    assert any("TotalLiquid" in r.message for r in caplog.records)

    all_null = _fc_payload(BASE, [None, None], [None, None])
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.accuweather"):
        out2 = parse_hourly_payload(all_null, "s1", DEFAULT_HOURS)
    assert out2["temperature_2m"] == [None, None]
    assert any("温度序列全部缺测" in r.message for r in caplog.records)
    assert any("降水序列全部缺测" in r.message for r in caplog.records)
    # 缺测条目虽带 Unit=C，但不得被误报为"未声明单位"
    assert not any("未声明单位" in r.message for r in caplog.records)


# ----------------------------------------------------------------- 档位回退与熔断
def test_tier_fallback_steps_down_and_caches(monkeypatch):
    """120/72 被 403 拒 -> 48 成功；第二个站点直接走 48，不再重探。"""
    monkeypatch.delenv(KEY_ENV, raising=False)
    state = {"forecast_calls": []}

    def forecast(url, params):
        state["forecast_calls"].append(url)
        if "/hourly/48hour/" in url:
            return _ok(_fc_payload(BASE, [30.0], [0.0]))
        return json.dumps({"code": "403", "message": "Access denied"}), 403

    src = _provider(_routes(forecast=forecast))
    snap1 = src.fetch_snapshot(_station())
    assert snap1["tier"] == 48
    src.fetch_snapshot(_station())
    assert [u.split("/hourly/")[1].split("/")[0] for u in state["forecast_calls"]] == \
           ["120hour", "72hour", "48hour", "48hour"]


def test_tier_cache_is_in_process_only(monkeypatch):
    """档位缓存仅进程内：新实例（新一次运行）从默认档重新探测，双向自愈。"""
    monkeypatch.delenv(KEY_ENV, raising=False)

    def forecast(url, params):
        if "/hourly/24hour/" in url:
            return _ok(_fc_payload(BASE, [30.0], [0.0]))
        return json.dumps({"code": "403"}), 403

    first = _provider(_routes(forecast=forecast))
    assert first.fetch_snapshot(_station())["tier"] == 24
    # 同实例第二站复用缓存，不再重探
    first.fetch_snapshot(_station())

    # 新实例（模拟下一次运行）：重新从 120 探测（瞬时降级不会被永久封顶）
    sess = RoutingSession(_routes(forecast=forecast))
    second = AccuWeatherProvider(api_key=KEY, session=sess, retries=0)
    assert second.fetch_snapshot(_station())["tier"] == 24
    assert [c["url"].split("/hourly/")[1].split("/")[0] for c in sess.calls[1:]] == \
        ["120hour", "72hour", "48hour", "24hour"]


def test_all_tiers_rejected_breaker_blocks_rest_of_run(monkeypatch):
    """梯子穷尽（含配额以 403 形态耗尽）也置熔断：后续站点 0 次预报请求。"""
    monkeypatch.delenv(KEY_ENV, raising=False)
    sess = RoutingSession(_routes(forecast=lambda u, p: (json.dumps({"code": "403"}), 403)))
    src = AccuWeatherProvider(api_key=KEY, session=sess, retries=0)
    with pytest.raises(RuntimeError, match="各档位均被拒绝"):
        src.fetch_snapshot(_station())
    n_after_first = len(sess.calls)
    assert src._tiers_exhausted is True
    with pytest.raises(RuntimeError, match="熔断"):
        src.fetch_snapshot(_station())
    assert len(sess.calls) == n_after_first  # 后续站点不再发任何请求


# ----------------------------------------------------------------- 鉴权与配额
def test_auth_failure_fast_fails_without_tier_stepping(monkeypatch):
    """401 是账号级鉴权问题：不做档位回退、不做无意义重试，直接失败。"""
    monkeypatch.delenv(KEY_ENV, raising=False)
    sess = RoutingSession(_routes(forecast=lambda u, p: (json.dumps({"code": "401"}), 401)))
    with pytest.raises(RuntimeError, match="401"):
        AccuWeatherProvider(api_key=KEY, session=sess, retries=0).fetch_snapshot(_station())
    assert len(sess.calls) == 2  # 1 定位 + 1 预报


def test_location_auth_failure_has_actionable_message(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    sess = RoutingSession(_routes(location=(json.dumps({"code": "401"}), 401),
                                  forecast=lambda u, p: _ok([])))
    with pytest.raises(RuntimeError, match="鉴权失败"):
        AccuWeatherProvider(api_key=KEY, session=sess, retries=0).fetch_snapshot(_station())


def test_all_tiers_rejected_error_carries_attempts(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    sess = RoutingSession(_routes(forecast=lambda u, p: (json.dumps({"code": "403"}), 403)))
    with pytest.raises(RuntimeError) as exc:
        AccuWeatherProvider(api_key=KEY, session=sess, retries=0).fetch_snapshot(_station())
    msg = str(exc.value)
    for tier in TIERS:
        assert f"{tier}h" in msg
    assert "403" in msg


def test_503_exhausts_then_breaker_blocks_rest_of_run(monkeypatch, caplog):
    """503 退避重试穷尽 -> 带配额说明抛错 + 置熔断；后续站点 0 次请求快速失败。"""
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.setattr("weather_eval.forecast.accuweather.time.sleep", lambda s: None)
    sess = RoutingSession(_routes(forecast=lambda u, p: ("Service Unavailable", 503)))
    src = AccuWeatherProvider(api_key=KEY, session=sess, retries=2)
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.accuweather"):
        with pytest.raises(RuntimeError, match="50 次/天"):
            src.fetch_snapshot(_station())
    assert src._quota_suspect is True
    assert len(sess.calls) == 1 + 3  # 定位 1 次（200）+ 预报重试穷尽 3 次
    with pytest.raises(RuntimeError, match="熔断"):
        src.fetch_snapshot(_station())
    assert len(sess.calls) == 1 + 3  # 熔断后不再发请求


def test_non_503_5xx_retries_but_no_breaker(monkeypatch):
    """500 过载重试穷尽后抛错，但不触发配额熔断（下次运行/下一站仍应尝试）。"""
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.setattr("weather_eval.forecast.accuweather.time.sleep", lambda s: None)
    sess = RoutingSession(_routes(forecast=lambda u, p: ("boom", 500)))
    src = AccuWeatherProvider(api_key=KEY, session=sess, retries=1)
    with pytest.raises(RuntimeError, match="最终失败"):
        src.fetch_snapshot(_station())
    assert src._quota_suspect is False


def test_key_redacted_in_logs_and_errors(monkeypatch, caplog):
    """Key 已走 Authorization 头，URL 不再携带凭据；但底层异常/服务端回显仍
    可能外带 Key——日志与最终异常都必须脱敏（防御纵深）。"""
    monkeypatch.delenv(KEY_ENV, raising=False)

    class _BoomSession:
        # 模拟中间层/服务端回显凭据的真实异常形态：消息携带 Bearer Key 明文
        def get(self, url, params=None, headers=None, **kwargs):
            raise ConnectionError(
                f"refused while sending 'Authorization: Bearer {KEY}' to {url}")

    prov = AccuWeatherProvider(api_key=KEY, session=_BoomSession(), retries=0)
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.accuweather"):
        with pytest.raises(RuntimeError) as exc:
            prov.fetch_snapshot(_station())
    assert KEY not in str(exc.value)
    # __cause__ 链也不得携带明文：原始网络异常须先归一为脱敏 RuntimeError
    assert exc.value.__cause__ is not None
    assert KEY not in str(exc.value.__cause__)
    assert all(KEY not in r.message for r in caplog.records)


# ----------------------------------------------------------------- 定位解析
def test_location_missing_geoposition_falls_back_with_warning(monkeypatch, caplog):
    monkeypatch.delenv(KEY_ENV, raising=False)
    loc = {"Key": "42", "LocalizedName": "某地"}  # 无 GeoPosition
    sess = RoutingSession(_routes(location=_ok(loc),
                                  forecast=lambda u, p: _ok(_fc_payload(BASE, [30.0], [0.0]))))
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.accuweather"):
        snap = AccuWeatherProvider(api_key=KEY, session=sess, retries=0) \
            .fetch_snapshot(_station())
    assert snap["grid_lat"] == 23.4783 and snap["grid_lon"] == 111.304
    assert snap["elevation"] is None
    assert snap["location_distance_km"] == 0.0
    assert any("GeoPosition" in r.message for r in caplog.records)


def test_location_missing_key_raises(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    sess = RoutingSession(_routes(location=_ok({"LocalizedName": "x"}),
                                  forecast=lambda u, p: _ok([])))
    with pytest.raises(RuntimeError, match="locationKey"):
        AccuWeatherProvider(api_key=KEY, session=sess, retries=0).fetch_snapshot(_station())


def test_location_4xx_fails_with_status(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    sess = RoutingSession(_routes(location=("bad q", 400), forecast=lambda u, p: _ok([])))
    with pytest.raises(RuntimeError, match="定位查询失败"):
        AccuWeatherProvider(api_key=KEY, session=sess, retries=0).fetch_snapshot(_station())


# ----------------------------------------------------------------- 构造与配置
def test_missing_key_raises(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    with pytest.raises(RuntimeError, match=KEY_ENV):
        AccuWeatherProvider(session=requests.Session())


def test_key_arg_beats_env(monkeypatch):
    monkeypatch.setenv(KEY_ENV, "from_env")
    assert AccuWeatherProvider(api_key="from_arg").key == "from_arg"
    assert AccuWeatherProvider().key == "from_env"


def test_hours_snapped_to_tier(monkeypatch):
    monkeypatch.delenv(KEY_ENV, raising=False)
    assert AccuWeatherProvider(api_key="k", hours=48).hours == 48
    assert AccuWeatherProvider(api_key="k", hours=100).hours == 72
    assert AccuWeatherProvider(api_key="k", hours=0).hours == 1


# ----------------------------------------------------------------- CLI 集成
def test_cmd_fetch_forecast_routes_to_accuweather(monkeypatch):
    """--source accuweather 应构造 AccuWeatherProvider 并存档其单模型快照。"""
    import weather_eval.__main__ as m

    made = {}
    saved = []

    class FakeAW:
        def fetch_snapshot(self, station, models):
            made["models"] = list(models)
            return {
                "issue_iso": "2026-08-28T15:00", "station_id": station.id,
                "source": "accuweather", "models": [MODEL_NAME],
                "grid_lat": 0.0, "grid_lon": 0.0, "elevation": None,
                "hourly_time": [],
                "data": {MODEL_NAME: {"temperature_2m": [], "precipitation": []}},
            }

    monkeypatch.setattr(m, "AccuWeatherProvider", lambda: FakeAW())
    monkeypatch.setattr(m, "save_forecast_snapshot",
                        lambda sid, mod, sub: saved.append((sid, mod)) or True)

    class Station:
        id = "s1"
        lat = 23.0
        lon = 111.0

    class Cfg:
        models = ["ecmwf_ifs", MODEL_NAME]
        stations = [Station()]

    monkeypatch.setattr(m, "load_config", lambda cfg: Cfg())

    class Args:
        source = "accuweather"
        config = None

    m.cmd_fetch_forecast(Args())
    assert made["models"] == [MODEL_NAME]
    assert saved == [("s1", MODEL_NAME)]


def test_cmd_fetch_forecast_accuweather_config_missing_fails_cleanly(monkeypatch, caplog):
    """config 未登记 accuweather_v1 时应给出可操作错误而非裸 traceback。"""
    import weather_eval.__main__ as m

    class Station:
        id = "s1"
        lat = 23.0
        lon = 111.0

    class Cfg:
        models = ["ecmwf_ifs"]
        stations = [Station()]

    monkeypatch.setattr(m, "load_config", lambda cfg: Cfg())

    class Args:
        source = "accuweather"
        config = None

    with caplog.at_level(logging.ERROR):
        with pytest.raises(SystemExit):
            m.cmd_fetch_forecast(Args())
    assert any("accuweather" in r.message for r in caplog.records)


def test_open_meteo_branch_excludes_accuweather(monkeypatch):
    """Open-Meteo 分支必须排除 accuweather_v1，避免刷缺失模型警告。"""
    import weather_eval.__main__ as m

    captured = {}

    class FakeOM:
        def fetch_snapshot(self, station, models):
            captured["models"] = list(models)
            return {
                "issue_iso": "2026-08-28T00:00", "station_id": station.id,
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
        models = ["ecmwf_ifs", MODEL_NAME]
        stations = [Station()]

    monkeypatch.setattr(m, "load_config", lambda cfg: Cfg())

    class Args:
        source = "open_meteo"
        config = None

    m.cmd_fetch_forecast(Args())
    assert captured["models"] == ["ecmwf_ifs"]


def test_cli_main_parses_and_dispatches_accuweather(monkeypatch):
    """真 argparse 解析 + 分发到 cmd_fetch_forecast（复制一份 parser 的测试会随实现漂移）。"""
    import weather_eval.__main__ as m

    called = {}

    def _fake_fetch(args):
        called["source"] = args.source
        return 0  # 0 失败 -> 不触发退出码路径

    monkeypatch.setattr(m, "cmd_fetch_forecast", _fake_fetch)
    m.main(["fetch-forecast", "--source", "accuweather"])
    assert called["source"] == "accuweather"


# ----------------------------------------------------------------- 端到端
def _e2e_setup(tmp_path, monkeypatch, n_hours=3):
    """快照（含 -1h 降水移位）与整点观测完美配对 -> 温度/降水指标全对。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv(KEY_ENV, raising=False)
    temps = [28.5 - i * 0.5 for i in range(n_hours)]
    liquids = [0.0 if i % 2 == 0 else 0.2 for i in range(n_hours)]
    payload = _fc_payload(BASE, temps, liquids)
    sess = RoutingSession(_routes(forecast=lambda u, p: _ok(payload)))
    snap = AccuWeatherProvider(api_key=KEY, session=sess, retries=0) \
        .fetch_snapshot(_station())
    storage.save_forecast_snapshot("s1", MODEL_NAME, snap)
    # 观测雨量按移位后的预报逐点对齐：rain@t_i = liquids[i-1]（i>=1）
    obs = []
    for i in range(1, n_hours):
        obs.append({"time": iso(BASE + timedelta(hours=i)),
                    "temp": temps[i], "rain": liquids[i - 1]})
    storage.save_obs("s1", obs)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 0}
    return temps, liquids, cfg


def test_end_to_end_with_obs(tmp_path, monkeypatch):
    temps, _, cfg = _e2e_setup(tmp_path, monkeypatch, n_hours=3)
    data = build_report(["s1"], [MODEL_NAME], cfg, BASE, BASE + timedelta(hours=2), "2026-08")

    sc_t = data["scorecard"][MODEL_NAME]["temp_24h"]
    assert sc_t["n"] == 2
    assert sc_t["acc2"] == 100.0
    assert abs(sc_t["rmse"]) < 1e-6
    pb = data["scorecard"][MODEL_NAME]["precip_24h"]
    assert pb["acc"] == 100.0
    assert abs(pb["ts"] - 1.0) < 1e-6


def test_end_to_end_misaligned_rain_degrades_ts(tmp_path, monkeypatch):
    """若不做 -1h 移位（观测雨量错拿同刻窗），晴雨 TS 应崩为 0——锁定移位口径。"""
    temps, liquids, cfg = _e2e_setup(tmp_path, monkeypatch, n_hours=5)
    from weather_eval.timeutil import parse_iso

    def _obs_with(rain_at):
        # rain_at(i) 返回观测时刻 t_i 应有的雨量；快照降水（已移位）固定为 liquids[i-1]
        return [{"time": iso(parse_iso("2026-08-28T15:00") + timedelta(hours=i)),
                 "temp": temps[i], "rain": rain_at(i)} for i in range(1, 5)]

    # 未移位口径：rain@t_i = liquids[i]，与已移位的快照降水错开 1 小时
    storage.save_obs("s1", _obs_with(lambda i: liquids[i]))
    mis = build_report(["s1"], [MODEL_NAME], cfg, BASE, BASE + timedelta(hours=4), "2026-08")
    ts_misaligned = mis["scorecard"][MODEL_NAME]["precip_24h"]["ts"]

    # 移位口径：rain@t_i = liquids[i-1]，与快照降水逐窗精确对齐
    storage.save_obs("s1", _obs_with(lambda i: liquids[i - 1]))
    al = build_report(["s1"], [MODEL_NAME], cfg, BASE, BASE + timedelta(hours=4), "2026-08")
    ts_aligned = al["scorecard"][MODEL_NAME]["precip_24h"]["ts"]

    assert ts_aligned == 1.0
    assert ts_misaligned == 0.0


def test_snapshot_is_idempotent_per_issue(tmp_path, monkeypatch):
    temps, liquids, cfg = _e2e_setup(tmp_path, monkeypatch, n_hours=3)
    sess = RoutingSession(_routes(forecast=lambda u, p: _ok(
        _fc_payload(BASE, temps, liquids))))
    snap = AccuWeatherProvider(api_key=KEY, session=sess, retries=0) \
        .fetch_snapshot(_station())
    assert storage.save_forecast_snapshot("s1", MODEL_NAME, snap) is False  # 同 issue 幂等

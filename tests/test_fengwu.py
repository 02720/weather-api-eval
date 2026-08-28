"""FengWuProvider（风乌 GHR-9km）单测。

mock 契约来自 2026-08 线上实测 + 前端 JS 逆向：
- GET /api/v1/weather/availability → {api_end_time: "2026-08-28T00:00:00Z", ...}
- GET /api/open/v1/weather/visual/query?longitude&latitude&model_type&region&forecast_time
  → {longitude, latitude, data:[{time:"...Z", values:{t2m(K), tp6h(mm/6h), ...}}],
     forecast_time, forecast_hours}
- 游客态 3h 步长 56 点（+1h..+166h）；Authorization: Bearer <key> 可解锁逐小时 360h。
"""
import json
from datetime import datetime, timedelta

import pytest
from conftest import FakeResp

from weather_eval.forecast.fengwu import (
    AVAIL_URL,
    QUERY_URL,
    FengWuProvider,
    interpolate_hourly,
    parse_iso_z,
    spread_precip_6h,
)
from weather_eval.forecast.fuxi import _num  # noqa: F401  (仅确保模块独立可导入)

KEY = "key-xyz"


class RoutingSession:
    def __init__(self, routes):
        self.routes = routes
        self.calls: list[tuple] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {}), dict(headers or {})))
        s, t = self.routes(url, params, headers)
        return FakeResp(t, status_code=s)


class Station:
    id, name, lat, lon = "wuzhou", "梧州", 23.4783, 111.304


def _samples_3h():
    """游客态形态：起报 2026-08-28T00Z，+1/+4/+7h 三个采样点，3h 步长。"""
    return [
        {"time": "2026-08-28T01:00:00Z", "values": {"t2m": 300.15, "tp6h": 6.0}},
        {"time": "2026-08-28T04:00:00Z", "values": {"t2m": 303.15, "tp6h": 3.0}},
        {"time": "2026-08-28T07:00:00Z", "values": {"t2m": 297.15, "tp6h": 0.0}},
    ]


def _query_payload(samples):
    return json.dumps({
        "longitude": 111.33, "latitude": 23.49,   # 9km 网格吸附回显
        "data": samples,
        "forecast_time": "2026-08-28T00:00:00Z",
        "forecast_hours": 166,
    })


def _routes(query_payload=None, fail_issues=0, avail_end="2026-08-28T00:00:00Z"):
    """fail_issues：前 N 个起报的查询返回 400（测回退）。"""
    state = {"fails": 0}

    def routes(url, params, headers):
        if url == AVAIL_URL:
            assert params["model_type"] == "FengWu-GHR-9km"
            assert params["region"] == "cn"
            return 200, json.dumps({"api_start_time": "2026-01-01T00:00:00Z",
                                    "api_end_time": avail_end})
        if url == QUERY_URL:
            assert params["model_type"] == "FengWu-GHR-9km"
            assert params["longitude"] == Station.lon
            assert params["latitude"] == Station.lat
            if state["fails"] < fail_issues:
                state["fails"] += 1
                return 400, '{"message":"仅支持 ... 的起报时间"}'
            return 200, query_payload
        return 404, "{}"

    return routes


# ------------------------------------------------------------------ 纯函数
def test_parse_iso_z():
    assert parse_iso_z("2026-08-28T00:00:00Z") == datetime(2026, 8, 28, 0, 0)
    assert parse_iso_z("2026-08-28T00:00Z") == datetime(2026, 8, 28, 0, 0)


def test_interpolate_hourly_fills_gaps_linearly():
    samples = [(datetime(2026, 8, 28, 9), 26.0),
               (datetime(2026, 8, 28, 12), 29.0)]   # 3h 间隔
    out = interpolate_hourly(samples)
    assert [t.hour for t, _ in out] == [9, 10, 11, 12]
    assert [v for _, v in out] == [26.0, 27.0, 28.0, 29.0]


def test_interpolate_hourly_identity_for_hourly_samples():
    samples = [(datetime(2026, 8, 28, 9) + timedelta(hours=i), float(i))
               for i in range(3)]
    out = interpolate_hourly(samples)
    assert out == samples


def test_interpolate_hourly_none_endpoints_not_extrapolated():
    samples = [(datetime(2026, 8, 28, 9), None),
               (datetime(2026, 8, 28, 12), 29.0),
               (datetime(2026, 8, 28, 15), None)]
    out = interpolate_hourly(samples)
    # 端点缺测的区段不插值：仅保留采样点本身
    assert [t.hour for t, _ in out] == [9, 12, 15]
    vals = {t.hour: v for t, v in out}
    assert vals[9] is None and vals[12] == 29.0 and vals[15] is None


def test_spread_precip_6h_window_math():
    # 采样 09(6mm)/12(3mm)/15(3mm)，相位平铺子集 = {09, 15}（中间的 12 被排除，
    # 否则相邻窗口重叠 3h 会重复计总量）→ 窗口 (03,09] 与 (09,15] 无缝拼接
    samples = [(datetime(2026, 8, 28, 9), 6.0),
               (datetime(2026, 8, 28, 12), 3.0),
               (datetime(2026, 8, 28, 15), 3.0)]
    hours = [datetime(2026, 8, 28, h) for h in (9, 10, 11, 12, 13, 14, 15, 16)]
    out = spread_precip_6h(samples, hours)
    # h=09 → 窗口@09（03-09 累计 6mm）→ 1.0；h=10..15 → 窗口@15（09-15 累计 3mm）→ 0.5
    # h=16 → 窗口@15 (09,15] 不含 16，且无更晚平铺端点 → None
    assert out == [1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, None]


def test_spread_precip_6h_conserves_total():
    samples = [(datetime(2026, 8, 28, 9) + timedelta(hours=3 * i), float(i + 1) * 3)
               for i in range(10)]                    # 每 3h 采样，值=窗口累计
    # hours 覆盖所有平铺窗口的全部小时（首窗 (03,09] 至末窗 (27,33]）
    hours = [datetime(2026, 8, 28, 4) + timedelta(hours=i) for i in range(33)]
    out = spread_precip_6h(samples, hours)
    covered = [v for v in out if v is not None]
    # 总量守恒：Σ 逐小时 == Σ 平铺子集窗口的 tp6h（重叠的中间采样被排除，不重复计）
    sub = [v for t, v in samples
           if (t - samples[0][0]) % timedelta(hours=6) == timedelta(0)]
    assert sum(covered) == pytest.approx(sum(sub), rel=1e-9)
    # 且每个平铺窗口恰好摊满 6 小时
    assert len(covered) == 6 * len(sub)


# ------------------------------------------------------------------ 集成
def test_fetch_snapshot_guest_end_to_end():
    sess = RoutingSession(_routes(query_payload=_query_payload(_samples_3h())))
    prov = FengWuProvider(session=sess)
    snap = prov.fetch_snapshot(Station, ["fengwu_ghr_9km"])
    assert snap["issue_iso"] == "2026-08-28T08:00"        # 00Z → 北京时 08
    assert snap["source"] == "fengwu"
    assert snap["models"] == ["fengwu_ghr_9km"]
    assert snap["grid_lat"] == 23.49 and snap["grid_lon"] == 111.33
    # 3h 采样 → 逐小时插值：09/12/15 为采样整点，10/11/13/14 为插值点
    assert snap["hourly_time"][0] == "2026-08-28T09:00"
    assert snap["hourly_time"][-1] == "2026-08-28T15:00"
    d = snap["data"]["fengwu_ghr_9km"]
    temps = dict(zip(snap["hourly_time"], d["temperature_2m"]))
    assert temps["2026-08-28T09:00"] == pytest.approx(27.0)   # 300.15K
    assert temps["2026-08-28T12:00"] == pytest.approx(30.0)   # 303.15K
    assert temps["2026-08-28T10:00"] == pytest.approx(28.0)   # 线性插值
    prec = dict(zip(snap["hourly_time"], d["precipitation"]))
    # 平铺窗口 = {+1h(6mm), +7h(0mm)}：h=09→6/6=1.0；h=10..15→0/6=0；h=16+→None
    assert prec["2026-08-28T09:00"] == pytest.approx(1.0)
    assert prec["2026-08-28T12:00"] == pytest.approx(0.0)
    assert prec["2026-08-28T15:00"] == pytest.approx(0.0)
    # 口径留档
    assert "tp6h" in snap["expansion"]


def test_api_key_sent_as_bearer():
    sess = RoutingSession(_routes(query_payload=_query_payload(_samples_3h())))
    FengWuProvider(session=sess, api_key=KEY).fetch_snapshot(Station)
    for url, _, headers in sess.calls:
        assert headers.get("Authorization") == f"Bearer {KEY}"


def test_no_key_sends_no_auth_header():
    sess = RoutingSession(_routes(query_payload=_query_payload(_samples_3h())))
    FengWuProvider(session=sess, api_key="").fetch_snapshot(Station)
    assert all("Authorization" not in h for _, _, h in sess.calls)


def test_issue_fallback_on_400(caplog):
    # availability 给出 00Z，但查询 400 两次 → 回退 12Z、06Z
    payload = _query_payload(_samples_3h()).replace("2026-08-28T00:00:00Z",
                                                    "2026-08-27T12:00:00Z")
    sess = RoutingSession(_routes(query_payload=payload, fail_issues=2,
                                  avail_end="2026-08-28T00:00:00Z"))
    with caplog.at_level("WARNING"):
        snap = FengWuProvider(session=sess).fetch_snapshot(Station)
    assert snap["issue_iso"] == "2026-08-27T20:00"        # 12Z → 北京时 20
    assert "尝试更早轮次" in caplog.text


def test_all_issues_fail_raises():
    sess = RoutingSession(_routes(query_payload=_query_payload(_samples_3h()),
                                  fail_issues=99))
    with pytest.raises(RuntimeError, match="均不可查"):
        FengWuProvider(session=sess, retries=0).fetch_snapshot(Station)


def test_401_with_key_gives_actionable_error():
    def routes(url, params, headers):
        if url == AVAIL_URL:
            return 200, json.dumps({"api_end_time": "2026-08-28T00:00:00Z"})
        return 401, '{"message":"invalid token"}'

    with pytest.raises(RuntimeError, match="FENGWU_API_KEY"):
        FengWuProvider(session=RoutingSession(routes), api_key="bad").fetch_snapshot(Station)


def test_401_does_not_fall_back():
    # P1 回归：Key 无效是账号级错误，必须立即失败而非逐轮回退 4 轮
    sess = RoutingSession(_routes_for_401())
    with pytest.raises(RuntimeError, match="鉴权失败"):
        FengWuProvider(session=sess, api_key="bad").fetch_snapshot(Station)
    query_calls = [c for c in sess.calls if c[0] == QUERY_URL]
    assert len(query_calls) == 1  # 仅 1 次查询，无回退


def _routes_for_401():
    def routes(url, params, headers):
        if url == AVAIL_URL:
            return 200, json.dumps({"api_end_time": "2026-08-28T00:00:00Z"})
        return 401, '{"message":"invalid token"}'

    return routes


def test_other_4xx_does_not_fall_back():
    # 403/404 等非 400 的 4xx 同样不参与起报回退
    def routes(url, params, headers):
        if url == AVAIL_URL:
            return 200, json.dumps({"api_end_time": "2026-08-28T00:00:00Z"})
        return 404, '{"message":"no route"}'

    sess = RoutingSession(routes)
    with pytest.raises(RuntimeError, match="请求被拒"):
        FengWuProvider(session=sess).fetch_snapshot(Station)
    assert len([c for c in sess.calls if c[0] == QUERY_URL]) == 1


def test_nonzero_offset_rejected():
    # P2 回归：非零时区偏移会整体错 8h，必须显式失败
    with pytest.raises(ValueError, match="非零时区偏移"):
        parse_iso_z("2026-08-28T08:00:00+08:00")


def test_nan_and_invalid_values_to_none():
    samples = [
        {"time": "2026-08-28T01:00:00Z", "values": {"t2m": "NaN", "tp6h": 6.0}},
        {"time": "2026-08-28T04:00:00Z", "values": {"t2m": "abc", "tp6h": "NaN"}},
        {"time": "2026-08-28T07:00:00Z", "values": {"t2m": 300.0, "tp6h": "inf"}},
    ]
    payload = _query_payload(samples)
    snap = FengWuProvider(session=RoutingSession(_routes(query_payload=payload))) \
        .fetch_snapshot(Station)
    d = snap["data"]["fengwu_ghr_9km"]
    # 端点缺测的区段不插值：仅保留采样点（09/12 为 None，15=300K→26.85℃）
    temps = dict(zip(snap["hourly_time"], d["temperature_2m"]))
    assert temps["2026-08-28T09:00"] is None
    assert temps["2026-08-28T12:00"] is None
    assert temps["2026-08-28T15:00"] == pytest.approx(26.85)
    prec = dict(zip(snap["hourly_time"], d["precipitation"]))
    # 平铺窗口 {+1h(6.0), +7h(inf→None)}：h=09→1.0；h=10..15 → None（缺测窗不虚构 0）
    assert prec["2026-08-28T09:00"] == pytest.approx(1.0)
    assert prec["2026-08-28T12:00"] is None
    assert prec["2026-08-28T15:00"] is None


def test_empty_data_treated_as_failure_and_falls_back():
    payload = json.dumps({"longitude": 111.33, "latitude": 23.49, "data": [],
                          "forecast_time": "2026-08-28T00:00:00Z"})
    good = _query_payload(_samples_3h()).replace(
        "2026-08-28T00:00:00Z", "2026-08-27T18:00:00Z")
    state = {"n": 0}

    def routes(url, params, headers):
        if url == AVAIL_URL:
            return 200, json.dumps({"api_end_time": "2026-08-28T00:00:00Z"})
        state["n"] += 1
        return (400, "{}") if state["n"] == 1 else (200, good)

    snap = FengWuProvider(session=RoutingSession(routes)).fetch_snapshot(Station)
    assert snap["issue_iso"] == "2026-08-28T02:00"        # 18Z → 北京时 02（次日）

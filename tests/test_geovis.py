"""中科星图（GevisProvider）单测。

mock 契约依据官方文档（datacloud.geovisearth.com/support/meteorological/
chinaCity120HourForecast，2026-08 版）：
- GET /meteorology/v1/weather/cn/forecast/hour/area[/professional|/basic]
  ?location={lon,lat}&token=...
- status!=0 业务失败；result.start/datas[].fc_time 为 yyyyMMddHH 当地时间；
  tem(℃)/pre(1h 降水 mm)，异常值 999999。
"""
import json
import logging

import pytest
from conftest import FakeResp

from weather_eval.forecast.geovis import (
    AREA_URL,
    DAY_URL,
    GevisProvider,
    _num_or_none,
    parse_area_response,
    parse_fc_time,
)

TOKEN = "gevis-token"


def _area_payload(tier="", tems=None, pres=None, start="2026082815", size=None):
    tems = tems if tems is not None else [30, 29, 28]
    pres = pres if pres is not None else [0, 0.5, 1.2]
    n = size or len(tems)
    datas = [{"fc_time": f"20260828{15 + i:02d}" if 15 + i < 24
              else f"20260829{(15 + i) % 24:02d}",
              "tem": tems[i], "pre": pres[i]} for i in range(n)]
    body = {
        "status": 0, "version": "v1",
        "date": {"time": "20260828153000", "timeZone": "Asia/Shanghai"},
        "result": {"start": start, "end": datas[-1]["fc_time"],
                   "size": str(n), "datas": datas},
    }
    return json.dumps(body)


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


def _routes(tier_payloads=None, fail_tiers=0):
    """tier_payloads 按 URL 后缀索引：{"/professional": text, "": text, "/basic": text}。"""
    tier_payloads = tier_payloads if tier_payloads is not None else \
        {"/professional": _area_payload()}
    state = {"fails": 0}

    def routes(url, params, headers):
        assert params["location"] == f"{Station.lon},{Station.lat}"  # 经度在前
        assert params["token"] == TOKEN
        if not url.startswith(AREA_URL):
            return 404, "{}"
        suffix = url[len(AREA_URL):]
        if state["fails"] < fail_tiers:
            state["fails"] += 1
            return 403, '{"status":1001,"msg":"no permission"}'
        if suffix in tier_payloads:
            return 200, tier_payloads[suffix]
        return 403, '{"status":1001,"msg":"no permission"}'

    return routes


def test_parse_fc_time():
    assert parse_fc_time("2026082815") == "2026-08-28T15:00"
    assert parse_fc_time("2026083123") == "2026-08-31T23:00"
    assert parse_fc_time("bad") is None
    assert parse_fc_time(None) is None
    assert parse_fc_time(2026082815) is None   # 非字符串拒绝


def test_num_or_none_missing_values():
    assert _num_or_none(999999) is None
    assert _num_or_none(-999999) is None       # 负向异常同样剔除
    assert _num_or_none(99999) is None         # 变体容差
    assert _num_or_none(9999) is None          # P1 回归：9999 缺测变体必须剔除
    assert _num_or_none(-9999) is None
    assert _num_or_none(28.5) == 28.5
    assert _num_or_none(0) == 0.0
    assert _num_or_none(None) is None


def test_parse_area_response_structure():
    parsed = parse_area_response(json.loads(_area_payload()))
    assert parsed["issue_iso"] == "2026-08-28T15:00"   # start 即起报（查询时刻）
    assert parsed["time"] == ["2026-08-28T15:00", "2026-08-28T16:00", "2026-08-28T17:00"]
    assert parsed["temperature_2m"] == [30.0, 29.0, 28.0]
    assert parsed["precipitation"] == [0.0, 0.5, 1.2]


def test_parse_area_response_status_error():
    with pytest.raises(RuntimeError, match="status"):
        parse_area_response({"status": 1001, "msg": "token 无效"})


def test_parse_area_response_empty_datas():
    body = json.loads(_area_payload())
    body["result"]["datas"] = []
    with pytest.raises(RuntimeError, match="datas 为空"):
        parse_area_response(body)


def test_fetch_snapshot_end_to_end():
    sess = RoutingSession(_routes())
    snap = GevisProvider(session=sess, token=TOKEN).fetch_snapshot(Station, ["geovis_v1"])
    assert snap["issue_iso"] == "2026-08-28T15:00"
    assert snap["models"] == ["geovis_v1"]
    assert snap["source"] == "geovis"
    assert snap["tier"] == "professional"
    assert snap["grid_lat"] == Station.lat      # 无吸附语义，回显请求坐标
    d = snap["data"]["geovis_v1"]
    assert d["temperature_2m"][1] == 29.0
    assert d["precipitation"][2] == 1.2


def test_tier_fallback_to_48h_and_cached():
    # professional 无权限（403）→ 进阶版（无后缀）可用
    sess = RoutingSession(_routes(tier_payloads={"": _area_payload()},
                                  fail_tiers=1))
    prov = GevisProvider(session=sess, token=TOKEN)
    snap = prov.fetch_snapshot(Station)
    assert snap["tier"] == "48h"
    # 只看逐小时端点：逐日块（day/area）是另一次探测，另有专测
    urls_tried = [u for u, _, _ in sess.calls if u.startswith(AREA_URL)]
    assert urls_tried == [AREA_URL + "/professional", AREA_URL]
    # 同实例第二站：直接走已固定的档位，不再探测
    n = len([u for u, _, _ in sess.calls if u.startswith(AREA_URL)])
    prov.fetch_snapshot(Station)
    assert [u for u, _, _ in sess.calls
            if u.startswith(AREA_URL)][n:] == [AREA_URL]


def test_tier_fallback_exhausted():
    sess = RoutingSession(_routes(fail_tiers=99))
    with pytest.raises(RuntimeError, match="全部档位查询失败"):
        GevisProvider(session=sess, token=TOKEN, retries=0).fetch_snapshot(Station)


def test_all_missing_values_warn(caplog):
    sess = RoutingSession(_routes(tier_payloads={
        "/professional": _area_payload(tems=[999999] * 3, pres=[999999] * 3)}))
    with caplog.at_level("WARNING"):
        GevisProvider(session=sess, token=TOKEN).fetch_snapshot(Station)
    assert "温度序列全部缺测" in caplog.text
    assert "降水序列全部缺测" in caplog.text


def test_missing_token_raises():
    import os
    old = os.environ.pop("GEVIS_TOKEN", None)
    try:
        with pytest.raises(RuntimeError, match="GEVIS_TOKEN"):
            GevisProvider()
    finally:
        if old is not None:
            os.environ["GEVIS_TOKEN"] = old


def test_network_error_masks_token(caplog):
    # P1 回归：token 走 URL query，网络异常消息入日志前必须掩码
    class BrokenSession:
        def get(self, url, params=None, headers=None, timeout=None):
            raise ConnectionError(
                f"Max retries exceeded with url: {url}?token={TOKEN} (/area/professional)")

    with caplog.at_level("WARNING"):
        with pytest.raises(RuntimeError, match="最终失败"):
            GevisProvider(session=BrokenSession(), token=TOKEN, retries=1) \
                .fetch_snapshot(Station)
    assert TOKEN not in caplog.text
    assert "***" in caplog.text


def test_400_no_retry_within_tier():
    calls = {"n": 0}

    class CountingSession:
        def get(self, url, params=None, headers=None, timeout=None):
            calls["n"] += 1
            return FakeResp('{"msg":"bad"}', status_code=400)

    with pytest.raises(RuntimeError, match="请求被拒|全部档位"):
        GevisProvider(session=CountingSession(), token=TOKEN, retries=2) \
            .fetch_snapshot(Station)
    # 每档一次、不重试（3 档探测各 1 次）
    assert calls["n"] == 3


def test_unparseable_fc_time_rejected():
    # P1 回归：fc_time 全非法 → 空时间轴快照必须拒绝入库（防幂等锁死）
    payload = json.loads(_area_payload())
    for d in payload["result"]["datas"]:
        d["fc_time"] = "garbage"
    sess = RoutingSession(_routes(tier_payloads={"/professional": json.dumps(payload)}))
    with pytest.raises(RuntimeError, match="拒绝入库"):
        GevisProvider(session=sess, token=TOKEN).fetch_snapshot(Station)


def test_business_error_downgrades_tier():
    # L3 回归：档位无权限以 HTTP 200 + status!=0 表达时，同样降档而非整源失败
    sess = RoutingSession(_routes(tier_payloads={
        "/professional": json.dumps({"status": 1001, "msg": "no permission"}),
        "": _area_payload(),
    }))
    prov = GevisProvider(session=sess, token=TOKEN)
    snap = prov.fetch_snapshot(Station)
    assert snap["tier"] == "48h"
    assert [u for u, _, _ in sess.calls
            if u.startswith(AREA_URL)] == [AREA_URL + "/professional", AREA_URL]


def test_cached_tier_business_error_invalidates_cache():
    # 已固定档位中途变为业务不可用：作废缓存回梯子自愈，不把失败档位当常量
    payloads = {"/professional": _area_payload(), "": _area_payload()}
    state = {"professional_ok": True}

    def routes(url, params, headers):
        if url.startswith(DAY_URL):      # 逐日端点另有专测；此处固定成功避免烧退避
            return 200, _day_payload()
        suffix = url[len(AREA_URL):]
        if suffix == "/professional" and not state["professional_ok"]:
            return 200, json.dumps({"status": 1002, "msg": "expired mid-run"})
        return 200, payloads[suffix]

    sess = RoutingSession(routes)
    prov = GevisProvider(session=sess, token=TOKEN)
    assert prov.fetch_snapshot(Station)["tier"] == "professional"
    state["professional_ok"] = False
    assert prov.fetch_snapshot(Station)["tier"] == "48h"


# ----------------------------------------------------------------- 逐日预报块
def _day_payload(days=None):
    """构造 15 日逐日响应（fc_time 为 yyyyMMdd；tem_max/min + 量级码式 pre 分量）。"""
    days = days or [("20260828", 30, 24), ("20260829", 31, 24), ("20260830", 33, 25)]
    datas = [{"fc_time": d, "tem_max": mx, "tem_min": mn,
              "pre_day": 5.0, "pre_night": 5.0, "pre_pro_day": 100}
             for d, mx, mn in days]
    last = days[-1][0]
    end = f"{last[0:4]}-{last[4:6]}-{last[6:8]}"
    body = {
        "status": 0, "version": "v1",
        "date": {"time": "20260828153000", "timeZone": "Asia/Shanghai"},
        "result": {"start": days[0][0], "end": end, "size": str(len(datas)),
                   "datas": datas},
    }
    return json.dumps(body)


def _day_routes(hour_suffix_payload=None, day_handler=None):
    """同时路由逐小时与逐日端点的假 session 路由。"""
    hour = hour_suffix_payload if hour_suffix_payload is not None else _area_payload()

    def routes(url, params, headers):
        assert params["location"] == f"{Station.lon},{Station.lat}"
        assert params["token"] == TOKEN
        if url.startswith(DAY_URL):
            if day_handler is None:
                return 403, '{"status":1001,"msg":"no permission"}'
            ret = day_handler(url)
            return ret if isinstance(ret, tuple) else (200, ret)
        if url.startswith(AREA_URL):
            suffix = url[len(AREA_URL):]
            if isinstance(hour, dict):
                return 200, hour.get(suffix, "403")
            return 200, hour
        return 404, "{}"

    return routes


def test_daily_block_parsed_temp_only():
    """day/area 15 天逐日：tem_max/min 入块、日期转自然日；量级码式降水不入块。"""
    sess = RoutingSession(_day_routes(day_handler=lambda u: _day_payload()))
    snap = GevisProvider(session=sess, token=TOKEN).fetch_snapshot(Station, ["geovis_v1"])
    assert snap["daily_time"] == ["2026-08-28", "2026-08-29", "2026-08-30"]
    d = snap["daily"]["geovis_v1"]
    assert d["temp_max"] == [30.0, 31.0, 33.0]
    assert d["temp_min"] == [24.0, 24.0, 25.0]
    assert d["precipitation"] == [None, None, None]   # pre_day/night 是量级码，绝不入库
    assert sess.calls[1][0] == DAY_URL + "/professional"   # 逐日首选专业档（call0 是逐小时）
    # 逐小时主干不受影响
    assert snap["hourly_time"][0] == "2026-08-28T15:00"


def test_daily_999999_missing_marker_becomes_none():
    sess = RoutingSession(_day_routes(day_handler=lambda u: _day_payload(
        days=[("20260828", 999999, 24), ("20260829", 31, -9999)])))
    snap = GevisProvider(session=sess, token=TOKEN).fetch_snapshot(Station, ["geovis_v1"])
    d = snap["daily"]["geovis_v1"]
    assert d["temp_max"] == [None, 31.0]
    assert d["temp_min"] == [24.0, None]


def test_daily_tier_descends_and_caches():
    """逐日 professional 被拒 → 进阶（无后缀）成功；第二站直走已固定档位。"""
    tried = []

    def day_handler(url):
        tried.append(url)
        if url == DAY_URL + "/professional":
            return 403, '{"status":1001}'
        assert url == DAY_URL
        return _day_payload()

    sess = RoutingSession(_day_routes(day_handler=day_handler))
    prov = GevisProvider(session=sess, token=TOKEN)
    snap = prov.fetch_snapshot(Station, ["geovis_v1"])
    assert snap["daily_time"] == ["2026-08-28", "2026-08-29", "2026-08-30"]
    prov.fetch_snapshot(Station, ["geovis_v1"])
    assert tried == [DAY_URL + "/professional", DAY_URL, DAY_URL]


def test_daily_all_tiers_fail_leaves_snapshot_intact(caplog):
    """逐日各档位均被拒：快照照常产出但不带逐日块（延长线绝不拖垮主干）。"""
    sess = RoutingSession(_day_routes())     # day 全部 403
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.geovis"):
        snap = GevisProvider(session=sess, token=TOKEN, retries=0) \
            .fetch_snapshot(Station, ["geovis_v1"])
    assert "daily" not in snap and "daily_time" not in snap
    assert snap["data"]["geovis_v1"]["temperature_2m"][0] == 30.0
    assert any("逐日预报各档位均不可用" in r.message for r in caplog.records)
    # 三个档位都试过且 4xx 不烧退避
    day_urls = [u for u, _, _ in sess.calls if u.startswith(DAY_URL)]
    assert day_urls == [DAY_URL + "/professional", DAY_URL, DAY_URL + "/basic"]


def test_daily_business_error_and_bad_fc_time_rejected(caplog):
    """200 + status!=0 属档位不可用（降档）；fc_time 全非法属契约漂移（拒绝入块）。"""
    # status!=0：三档全部业务失败 → 放弃逐日块
    sess = RoutingSession(_day_routes(day_handler=lambda u: (200, json.dumps({"status": 1001}))))
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.geovis"):
        snap = GevisProvider(session=sess, token=TOKEN, retries=0) \
            .fetch_snapshot(Station, ["geovis_v1"])
    assert "daily" not in snap
    assert any("逐日预报各档位均不可用" in r.message for r in caplog.records)

    # fc_time 全非法：业务成功但解析不出任何日 → 该次降档后同样放弃
    bad = json.dumps({"status": 0, "result": {"datas": [{"fc_time": "bad", "tem_max": 30}]}})
    sess2 = RoutingSession(_day_routes(day_handler=lambda u: bad))
    snap2 = GevisProvider(session=sess2, token=TOKEN, retries=0) \
        .fetch_snapshot(Station, ["geovis_v1"])
    assert "daily" not in snap2

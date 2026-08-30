"""中科星图（GevisProvider）单测。

mock 契约依据官方文档（datacloud.geovisearth.com/support/meteorological/
chinaCity120HourForecast，2026-08 版）：
- GET /meteorology/v1/weather/cn/forecast/hour/area[/professional|/basic]
  ?location={lon,lat}&token=...
- status!=0 业务失败；result.start/datas[].fc_time 为 yyyyMMddHH 当地时间；
  tem(℃)/pre(1h 降水 mm)，异常值 999999。
"""
import json

import pytest
from conftest import FakeResp

from weather_eval.forecast.geovis import (
    AREA_URL,
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
    urls_tried = [u for u, _, _ in sess.calls]
    assert urls_tried == [AREA_URL + "/professional", AREA_URL]
    # 同实例第二站：直接走已固定的档位，不再探测
    n = len(sess.calls)
    prov.fetch_snapshot(Station)
    assert [u for u, _, _ in sess.calls[n:]] == [AREA_URL]


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
    assert [u for u, _, _ in sess.calls] == [AREA_URL + "/professional", AREA_URL]


def test_cached_tier_business_error_invalidates_cache():
    # 已固定档位中途变为业务不可用：作废缓存回梯子自愈，不把失败档位当常量
    payloads = {"/professional": _area_payload(), "": _area_payload()}
    state = {"professional_ok": True}

    def routes(url, params, headers):
        suffix = url[len(AREA_URL):]
        if suffix == "/professional" and not state["professional_ok"]:
            return 200, json.dumps({"status": 1002, "msg": "expired mid-run"})
        return 200, payloads[suffix]

    sess = RoutingSession(routes)
    prov = GevisProvider(session=sess, token=TOKEN)
    assert prov.fetch_snapshot(Station)["tier"] == "professional"
    state["professional_ok"] = False
    assert prov.fetch_snapshot(Station)["tier"] == "48h"

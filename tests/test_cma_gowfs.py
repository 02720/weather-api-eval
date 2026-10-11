"""CmaGowfsProvider（中国气象数据网 data.cma.cn 智能网格预报）接入测试。

mock 契约来自 2026-10-11 线上实测（18 个经纬点 × 两个端点逐点核对，见
forecast/cma_gowfs.py docstring，扫描脚本 scripts/calibrate_cma_gowfs.py 可复现）：

  GET https://data.cma.cn/rest/gowfs/hour?lat=..&lon=..
  → {"hourTitle":...,"station":"","detail":[
       {"time":"2026-10-11 09:00","temperature":"24.4","total_rain":"0.0",...},
       ... 96 点：前 24 点逐 1h、其后逐 3h ...]}

  GET https://data.cma.cn/rest/gowfs/day?lat=..&lon=..
  → {"detail":[{"date":"2026-10-11","pt":"2026-10-11 08:00",
      "day":{"weather":{"info":"晴","temperature":"33.0"},...},
      "night":{"weather":{"info":"晴","temperature":"22.0"},...},
      "precipitation":""}, ... 30 天 ...]}

重点覆盖六处容易**静默**出错的契约：
1. 时区（接口原生北京时，非 UTC）——错 8 小时不报错，只让样本静默归零；
2. **网格分段**（前 24h 逐小时 + 其后 3h）——把整条序列当同一分辨率处理会让温度
   台阶化、降水放大 3 倍，两者都不报错；
3. **降水分段口径**（近场 1h 累计透传 / 远场 3h 累计均摊）——错一倍即污染全部降水样本；
4. **交接点不重复计数**——总降水必须严格等于原始点值之和；
5. **起报锚点**由逐日端点自述的 `pt` 反解（== 首点−1h == 末点−240h）；
6. **逐日日界是昼夜窗而非自然日**——temp_min 必须移一天取 night(D−1)。
"""
import json
import logging
from datetime import datetime, timedelta

import pytest
from conftest import FakeResp

from weather_eval.forecast.cma_gowfs import (
    BASE_URL,
    DAY_PATH,
    GowfsPayloadError,
    HOUR_PATH,
    MODEL_NAME,
    NATIVE_1H_POINTS,
    SOURCE,
    CmaGowfsProvider,
    build_daily_block,
    build_hourly_series,
    daily_precip_from_hourly,
    derive_issue,
    extract_day_publish,
    extract_hour_points,
    native_1h_prefix_len,
    parse_dt_bj,
)

ISSUE = datetime(2026, 10, 11, 8, 0)      # 产品发布时间（逐日端点 pt）
FIRST = datetime(2026, 10, 11, 9, 0)      # 首个逐小时有效时刻（= 起报 +1h）
SEAM = datetime(2026, 10, 12, 8, 0)       # 近/远场交接点（= 起报 +24h）
LAST = datetime(2026, 10, 21, 8, 0)       # 末个有效时刻（= 起报 +240h）


class Station:
    id, name, lat, lon = "wuzhou", "梧州气象站", 23.4783, 111.304
    cma_id = "59265"

def _hour_rows(native_hours: int = NATIVE_1H_POINTS,
               far_hours: int = 216, base: float = 20.0,
               rain_near: dict | None = None,
               rain_far: dict | None = None) -> list[dict]:
    """合成逐小时响应点列：前 `native_hours` 点逐 1h，其后逐 3h。"""
    rows: list[dict] = []
    t = FIRST
    hours = native_hours + far_hours // 3
    for i in range(hours):
        temp = base + 8.0 * ((t.hour - 5 + 24) % 24) / 24.0
        rain = 0.0
        if rain_near and t in rain_near:
            rain = rain_near[t]
        elif rain_far and t in rain_far:
            rain = rain_far[t]
        rows.append({"time": t.strftime("%Y-%m-%d %H:%M"),
                     "temperature": f"{temp:.2f}", "total_rain": f"{rain:.2f}",
                     "humidity": 60.0, "visibility": 20.0, "air_pressure": 9999.0,
                     "total_cloud": 0.0, "weather_icon": "0", "weather_info": "晴",
                     "wind_direction": 90.0, "wind_speed": 1.5,
                     "timelimit": 0.0, "dealWith": False})
        t = t + timedelta(hours=1) if i + 1 < native_hours \
            else t + timedelta(hours=3)
    return rows


def _hour_payload(rows: list[dict]) -> dict:
    return {"hourTitle": "数据来源于全国智能网格预报", "station": "", "detail": rows}


def _day_payload(days: int = 30, pt: str = "2026-10-11 08:00",
                 day_t: float = 33.0, night_t: float = 22.0,
                 precip: str = "") -> dict:
    out = []
    for i in range(days):
        d = ISSUE + timedelta(days=i)
        out.append({"date": d.date().isoformat(), "pt": pt,
                    "day": {"weather": {"info": "晴", "img": "0",
                                         "temperature": str(day_t)},
                            "wind": {"direct": "90", "power": "2"}},
                    "night": {"weather": {"info": "晴", "img": "0",
                                          "temperature": str(night_t)},
                              "wind": {"direct": "90", "power": "2"}},
                    "precipitation": precip})
    return {"dayTitle": "", "station": "", "publishTime": "", "detail": out}


class RoutingSession:
    """按 URL 路由 hour/day 两个端点的假会话；记录调用次数供断言。"""

    def __init__(self, hour=None, day=None, hour_status: int = 200,
                 day_status: int = 200, day_text: str | None = None):
        self.hour = hour
        self.day = day
        self.hour_status = hour_status
        self.day_status = day_status
        self.day_text = day_text
        self.calls: list[str] = []
        self.params: list[dict] = []

    def request(self, method, url, **kw):
        self.calls.append(url)
        self.params.append(dict(kw.get("params") or {}))
        if url.startswith(BASE_URL + HOUR_PATH):
            if self.hour_status != 200:
                return FakeResp("<html>417</html>", status_code=self.hour_status)
            return FakeResp(json.dumps(self.hour), status_code=200)
        if url.startswith(BASE_URL + DAY_PATH):
            if self.day_text is not None:
                return FakeResp(self.day_text, status_code=self.day_status)
            if self.day_status != 200:
                return FakeResp("<html>err</html>", status_code=self.day_status)
            return FakeResp(json.dumps(self.day), status_code=200)
        raise AssertionError(f"未预期的 URL: {url}")

    def get(self, url, **kw):
        return self.request("GET", url, **kw)


def _provider(session=None, **kw):
    return CmaGowfsProvider(session=session or RoutingSession(**kw))


# ------------------------------------------------------------ 解析与外壳
def test_parse_dt_bj_accepts_contract_format():
    assert parse_dt_bj("2026-10-11 09:00") == FIRST
    assert parse_dt_bj(" 2026-10-11 09:00 ") == FIRST      # 容忍首尾空白


@pytest.mark.parametrize("bad", ["2026-10-11T09:00", "2026/10/11 09:00",
                                "2026-10-11 09:00:00", "2026-10-11", "",
                                "2026-10-11 09:00+08:00", None])
def test_parse_dt_bj_rejects_other_formats(bad):
    """非契约形态一律响亮失败——UTC/ISO 串被当北京时读会静默错 8 小时。"""
    with pytest.raises(GowfsPayloadError):
        parse_dt_bj(bad)


def test_extract_hour_points_rejects_empty_detail():
    with pytest.raises(GowfsPayloadError):
        extract_hour_points({"detail": []})
    with pytest.raises(GowfsPayloadError):
        extract_hour_points({"detail": ""})
    with pytest.raises(GowfsPayloadError):
        extract_hour_points({})
    with pytest.raises(GowfsPayloadError):
        extract_hour_points("not json")


def test_extract_hour_points_counts_bad_and_sorts():
    rows = _hour_rows()
    rows.append({"time": "garbage", "temperature": "1", "total_rain": "0"})
    rows.append("not a dict")
    points, meta = extract_hour_points(_hour_payload(rows))
    assert meta["bad_time"] == 2
    assert len(points) == NATIVE_1H_POINTS + 72
    assert points[0][0] == FIRST and points[-1][0] == LAST


def test_extract_day_publish_single_pt():
    pt, raws = extract_day_publish(_day_payload())
    assert pt == ISSUE and raws == {"2026-10-11 08:00"}


def test_extract_day_publish_rejects_multiple_pt():
    payload = _day_payload()
    payload["detail"][5]["pt"] = "2026-10-11 20:00"
    pt, raws = extract_day_publish(payload)
    assert pt is None and len(raws) == 2


# ------------------------------------------------------------ 分段点与起报锚点
def test_native_1h_prefix_len_detects_seam():
    points, _ = extract_hour_points(_hour_payload(_hour_rows()))
    assert native_1h_prefix_len(points) == NATIVE_1H_POINTS


def test_native_1h_prefix_len_follows_product_change():
    """产品若把原生逐小时段改成 48 点，分段点必须自动跟随（不写死 24）。"""
    points, _ = extract_hour_points(
        _hour_payload(_hour_rows(native_hours=48, far_hours=192)))
    assert native_1h_prefix_len(points) == 48


def test_derive_issue_uses_day_pt():
    points, _ = extract_hour_points(_hour_payload(_hour_rows()))
    notes: list[str] = []
    issue, detail = derive_issue(points, NATIVE_1H_POINTS, ISSUE, notes)
    assert issue == ISSUE and detail == "day_pt"
    assert notes == []


def test_derive_issue_falls_back_to_axis_minus_1h_without_day():
    """逐日端点挂了也要给出**同一个** issue（lead 分布不变，天桶不错位）。"""
    points, _ = extract_hour_points(_hour_payload(_hour_rows()))
    notes: list[str] = []
    issue, detail = derive_issue(points, NATIVE_1H_POINTS, None, notes)
    assert issue == ISSUE and detail == "axis_minus_1h"
    assert notes


def test_derive_issue_rejects_pt_after_first_valid():
    """pt 晚于首点会产生负 lead（评估侧静默丢弃），必须退回结构恒等式。"""
    points, _ = extract_hour_points(_hour_payload(_hour_rows()))
    notes: list[str] = []
    issue, detail = derive_issue(points, NATIVE_1H_POINTS,
                                 datetime(2026, 10, 11, 12, 0), notes)
    assert issue == ISSUE and detail == "axis_minus_1h"


def test_derive_issue_rejects_stale_pt():
    points, _ = extract_hour_points(_hour_payload(_hour_rows()))
    notes: list[str] = []
    issue, detail = derive_issue(points, NATIVE_1H_POINTS,
                                 datetime(2026, 10, 9, 8, 0), notes)
    assert issue == ISSUE and detail == "axis_minus_1h"
    assert any("不同源" in n for n in notes)


# ------------------------------------------------------------ 逐小时序列
def test_hourly_axis_covers_full_horizon():
    points, _ = extract_hour_points(_hour_payload(_hour_rows()))
    axis, temps, precips, _ = build_hourly_series(points, NATIVE_1H_POINTS)
    assert axis[0] == FIRST and axis[-1] == LAST
    assert len(axis) == 240
    assert all(t.minute == 0 for t in axis)


def test_temperature_is_interpolated_in_far_field_only():
    """近场逐小时原生值原样保留；远场 3h 采样线性插值到逐小时。"""
    points, _ = extract_hour_points(_hour_payload(_hour_rows()))
    axis, temps, _, _ = build_hourly_series(points, NATIVE_1H_POINTS)
    raw = {t: float(e["temperature"]) for t, e in points}
    # 近场 24 个采样点逐位一致（插值不改既有采样）
    for h in axis[:NATIVE_1H_POINTS]:
        assert abs(temps[axis.index(h)] - raw[h]) < 1e-9
    # 远场：相邻 3h 采样之间的中点应恰为两端均值
    seam_i = axis.index(SEAM)
    t0, t1 = raw[SEAM], raw[SEAM + timedelta(hours=3)]
    assert abs(temps[seam_i + 1] - (t0 * 2 + t1) / 3) < 1e-9
    assert abs(temps[seam_i + 2] - (t0 + t1 * 2) / 3) < 1e-9
    assert abs(temps[seam_i + 3] - t1) < 1e-9


def test_precipitation_near_field_is_passthrough():
    """近场 1h 累计直接透传（与观测 rain@t=(t−1h, t] 同字）。"""
    rain_near = {FIRST + timedelta(hours=h): 0.25 + 0.1 * h for h in range(6)}
    points, _ = extract_hour_points(
        _hour_payload(_hour_rows(rain_near=rain_near)))
    axis, _, precips, _ = build_hourly_series(points, NATIVE_1H_POINTS)
    for h, v in rain_near.items():
        assert abs(precips[axis.index(h)] - v) < 1e-9


def test_precipitation_far_field_is_three_hour_accumulation_divided_by_three():
    """远场 3h 累计按 1/3 摊到每个小时（spread_accumulation 语义）。"""
    # 交接点 SEAM 的值按"近场 1h 累计"透传；SEAM+3k 的值覆盖 (SEAM+3k-3h, SEAM+3k]
    rain_far = {SEAM + timedelta(hours=3 * k): 0.6 + 0.3 * k for k in range(4)}
    points, _ = extract_hour_points(
        _hour_payload(_hour_rows(rain_far=rain_far)))
    axis, _, precips, _ = build_hourly_series(points, NATIVE_1H_POINTS)
    assert abs(precips[axis.index(SEAM)] - rain_far[SEAM]) < 1e-9
    for k in (1, 2, 3):
        t = SEAM + timedelta(hours=3 * k)
        for j in (2, 1, 0):                    # 该 3h 窗覆盖的 3 个小时
            h = t - timedelta(hours=j)
            assert abs(precips[axis.index(h)] - rain_far[t] / 3) < 1e-9


def test_precipitation_total_is_exactly_conserved():
    """交接点处理的核心不变量：逐小时合计 == 原始点值合计（无重复、无缺口）。"""
    rain_near = {FIRST + timedelta(hours=h): 0.3 for h in range(3)}
    rain_far = {SEAM + timedelta(hours=3 * k): 1.2 for k in range(3)}
    points, _ = extract_hour_points(
        _hour_payload(_hour_rows(rain_near=rain_near, rain_far=rain_far)))
    axis, _, precips, _ = build_hourly_series(points, NATIVE_1H_POINTS)
    raw = {t: float(e["total_rain"]) for t, e in points}
    assert abs(sum(v for v in precips if v is not None)
               - sum(raw.values())) < 1e-9


def test_precipitation_missing_near_field_hour_is_none_not_zero():
    """缺测绝不伪装成 0.0（否则"预报无雨"与"没数据"混为一谈）。"""
    rows = _hour_rows()
    rows[2]["total_rain"] = ""            # 第三个小时缺测
    points, _ = extract_hour_points(_hour_payload(rows))
    axis, _, precips, _ = build_hourly_series(points, NATIVE_1H_POINTS)
    assert precips[2] is None
    assert precips[1] == 0.0 and precips[3] == 0.0


def test_negative_and_sentinel_precip_become_missing():
    rows = _hour_rows()
    rows[1]["total_rain"] = "-0.5"
    rows[2]["total_rain"] = "9999"
    points, meta = extract_hour_points(_hour_payload(rows))
    axis, _, precips, stats = build_hourly_series(points, NATIVE_1H_POINTS)
    assert precips[1] is None and precips[2] is None
    assert stats["negative"] == 1 and stats["sentinel"] == 1


def test_implausible_temperature_becomes_missing():
    rows = _hour_rows()
    rows[3]["temperature"] = "999.0"
    points, _ = extract_hour_points(_hour_payload(rows))
    axis, temps, _, stats = build_hourly_series(points, NATIVE_1H_POINTS)
    assert temps[3] is None
    assert stats["temp_implausible"] == 1


def test_all_3h_grid_degrades_without_pretending_near_field():
    """没有逐小时前缀时按全 3h 网格处理——绝不把 3h 累计误当 1h 透传。"""
    points, _ = extract_hour_points(
        _hour_payload(_hour_rows(native_hours=1, far_hours=216)))
    axis, temps, precips, _ = build_hourly_series(points, 1)
    raw = {t: float(e["total_rain"]) for t, e in points}
    # 全 3h 网格：首点值也按 1/3 摊（绝不当作 1h 透传而放大 3 倍）
    first_t = points[0][0]
    assert precips[0] == raw[first_t] / 3
    assert precips[1] == raw[first_t + timedelta(hours=3)] / 3
    assert precips[2] == raw[first_t + timedelta(hours=3)] / 3


# ------------------------------------------------------------ 逐日块
def test_daily_block_shift_night_temp_by_one_day():
    """temp_min(D) 必须取 night(D−1)（自然日 D 的低温在 D 凌晨）。"""
    block = build_daily_block(_day_payload(days=4))
    assert block is not None
    times = block["time"]
    assert times == [ISSUE.date().isoformat(),
                     (ISSUE + timedelta(days=1)).date().isoformat(),
                     (ISSUE + timedelta(days=2)).date().isoformat(),
                     (ISSUE + timedelta(days=3)).date().isoformat()]
    d = block["data"][MODEL_NAME]
    assert d["temp_max"] == [33.0, 33.0, 33.0, 33.0]
    # 首日没有 D−1 的夜间条目 → 缺测；其后都是 22.0
    assert d["temp_min"] == [None, 22.0, 22.0, 22.0]


def test_daily_block_night_temp_actually_follows_previous_day():
    """night 温度按日变化时，temp_min 必须跟着 D−1 走而不是 D。"""
    payload = _day_payload(days=3)
    payload["detail"][0]["night"]["weather"]["temperature"] = "10.0"
    payload["detail"][1]["night"]["weather"]["temperature"] = "11.0"
    d = build_daily_block(payload)["data"][MODEL_NAME]
    assert d["temp_min"] == [None, 10.0, 11.0]


def test_daily_block_precipitation_is_all_null_without_hourly_input():
    """未提供逐小时聚合值时全 null——绝不回退去读 /day 的空串字段或折算现象码。"""
    d = build_daily_block(_day_payload(precip=""))["data"][MODEL_NAME]
    assert set(d["precipitation"]) == {None}


def _day_axis(day: str, hours: int = 24) -> list[datetime]:
    return [datetime.strptime(day + " " + str(h).zfill(2) + ":00",
                               "%Y-%m-%d %H:%M") for h in range(hours)]


def test_daily_precip_is_sum_of_24_hourly_hours():
    """日降水 = 该自然日 24 个整点降水之和（无损聚合，非现象码折算）。"""
    axis, precips = [], []
    for day, rate in (("2026-10-11", 0.25), ("2026-10-12", 1.0)):
        axis += _day_axis(day)
        precips += [rate] * 24
    out = daily_precip_from_hourly(axis, precips)
    assert out["2026-10-11"] == round(0.25 * 24, 4)
    assert out["2026-10-12"] == round(1.0 * 24, 4)


def test_daily_precip_partial_day_is_missing_not_partial_sum():
    """当天不足 24 个整点 → None，绝不拿"部分小时的和"冒充全天量。"""
    out = daily_precip_from_hourly(_day_axis("2026-10-11", 20), [0.5] * 20)
    assert out["2026-10-11"] is None


def test_daily_precip_any_missing_hour_makes_day_missing():
    """任一小时缺测即整日缺测——部分和不是全天量（否则会低估日雨量）。"""
    precips = [0.5] * 24
    precips[7] = None
    got = daily_precip_from_hourly(_day_axis("2026-10-11"), precips)
    assert got["2026-10-11"] is None


def test_daily_block_consumes_aggregated_hourly_precip():
    """逐日块的 precipitation 列与 daily_time 索引对齐，逐日累加值正确落位。"""
    dp = {"2026-10-11": None, "2026-10-12": 6.0, "2026-10-13": 12.5}
    block = build_daily_block(_day_payload(days=3), dp)
    assert block["time"] == ["2026-10-11", "2026-10-12", "2026-10-13"]
    assert block["data"][MODEL_NAME]["precipitation"] == [None, 6.0, 12.5]


def test_daily_block_returns_none_on_broken_payload():
    assert build_daily_block(None) is None
    assert build_daily_block({}) is None
    assert build_daily_block({"detail": []}) is None


# ------------------------------------------------------------ 快照外壳
def test_snapshot_issue_and_hourly_axis():
    snap = _provider(hour=_hour_payload(_hour_rows()),
                     day=_day_payload()).fetch_snapshot(Station, [MODEL_NAME])
    assert snap["issue_iso"] == "2026-10-11T08:00"
    assert snap["issue_source"] == "model_run"
    assert snap["issue_source_detail"] == "day_pt"
    assert snap["source"] == SOURCE
    assert snap["models"] == [MODEL_NAME]
    assert len(snap["hourly_time"]) == 240
    assert snap["hourly_time"][0] == "2026-10-11T09:00"
    assert snap["hourly_time"][-1] == "2026-10-21T08:00"
    # 口径披露：分段分辨率、降水累计窗口、交接点
    assert snap["resolution_hours"] == 3
    assert snap["precip_accum_window_hours"] == 3
    assert snap["precip_unit"] == "mm"
    assert snap["precip_1h_until"] == "2026-10-12T08:00"
    assert snap["native_1h_points"] == NATIVE_1H_POINTS
    assert snap["horizon_hours"] == 240
    assert "native 1h passthrough" in snap["expansion"]


def test_snapshot_requests_both_endpoints_with_lat_lon():
    """必须以经纬度（而非 WMO 站号）寻址，且逐小时与逐日都请求。"""
    session = RoutingSession(hour=_hour_payload(_hour_rows()),
                             day=_day_payload())
    _provider(session).fetch_snapshot(Station, [MODEL_NAME])
    assert len(session.calls) == 2
    assert session.calls == [BASE_URL + HOUR_PATH, BASE_URL + DAY_PATH]
    assert session.params == [{"lat": 23.4783, "lon": 111.304}] * 2


def test_day_failure_degrades_to_hour_only_snapshot():
    """逐日端点失败绝不拖垮逐小时快照，且 issue 与逐日可用时一致。"""
    session = RoutingSession(hour=_hour_payload(_hour_rows()),
                             day=_hour_payload(_hour_rows()),
                             day_status=500)
    snap = _provider(session).fetch_snapshot(Station, [MODEL_NAME])
    assert snap["issue_iso"] == "2026-10-11T08:00"
    assert snap["issue_source_detail"] == "axis_minus_1h"
    assert "daily" not in snap
    assert any("逐日端点不可用" in n for n in snap["issue_notes"])
    assert len(snap["hourly_time"]) == 240


def test_day_5xx_is_not_retried_forever(caplog):
    """逐日端点 5xx：重试由共享助手处理，但绝不因此抛掉逐小时主干。"""
    session = RoutingSession(hour=_hour_payload(_hour_rows()),
                             day=_hour_payload(_hour_rows()), day_status=503)
    with caplog.at_level(logging.WARNING):
        snap = _provider(session).fetch_snapshot(Station, [MODEL_NAME])
    assert snap["issue_source_detail"] == "axis_minus_1h"


def test_hour_empty_detail_raises():
    with pytest.raises(GowfsPayloadError):
        _provider(hour={"detail": []}).fetch_snapshot(Station, [MODEL_NAME])


def test_hour_417_is_fatal_and_not_retried():
    """经纬度参数非法返回 417（确定性失败）——重试不改变结果，立即熔断。"""
    session = RoutingSession(hour=_hour_payload(_hour_rows()),
                             day=_day_payload(), hour_status=417)
    prov = _provider(session)
    with pytest.raises(GowfsPayloadError):
        prov.fetch_snapshot(Station, [MODEL_NAME])
    assert len(session.calls) == 1          # 一次都没重试


def test_hour_4xx_is_fatal():
    session = RoutingSession(hour=_hour_payload(_hour_rows()),
                             day=_day_payload(), hour_status=403)
    with pytest.raises(GowfsPayloadError):
        _provider(session).fetch_snapshot(Station, [MODEL_NAME])


def test_all_missing_refuses_to_store():
    rows = _hour_rows()
    for r in rows:
        r["temperature"] = ""
        r["total_rain"] = ""
    with pytest.raises(RuntimeError):
        _provider(hour=_hour_payload(rows), day=_day_payload()) \
            .fetch_snapshot(Station, [MODEL_NAME])


def test_station_without_coords_raises_actionably():
    class NoGeo:
        id, name, lat, lon = "nogeo", "无坐标站", None, None
    with pytest.raises(Exception):
        _provider(hour=_hour_payload(_hour_rows()), day=_day_payload()) \
            .fetch_snapshot(NoGeo(), [MODEL_NAME])


# ------------------------------------------------------------ 端到端配对
def test_snapshot_pairs_with_observations_hourly(tmp_path, monkeypatch):
    """**时区/口径的终局回归**：入库快照必须与实况按北京时整点配上。"""
    from weather_eval import storage
    from weather_eval.evaluate import build_report
    from weather_eval.timeutil import parse_iso

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    snap = _provider(hour=_hour_payload(_hour_rows()),
                     day=_day_payload()).fetch_snapshot(Station, [MODEL_NAME])
    storage.save_forecast_snapshot(Station.id, MODEL_NAME, snap)

    obs = []
    for t, v in zip(snap["hourly_time"], snap["data"][MODEL_NAME]["temperature_2m"]):
        if v is not None:
            obs.append({"time": t, "temp": v + 1.0, "rain": 0.0})
    assert len(obs) == 240
    storage.save_obs(Station.id, obs)

    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 1}
    data = build_report([Station.id], [MODEL_NAME], cfg,
                        parse_iso(snap["hourly_time"][0]),
                        parse_iso(snap["hourly_time"][-1]), "2026-10")
    sc = data["scorecard"][MODEL_NAME]["temp_all"]
    assert sc["n"] == 240, "时刻必须逐点配上（错 8 小时则 n=0）"
    assert abs(sc["rmse"] - 1.0) < 1e-6


def test_second_fetch_in_same_issue_is_idempotent(tmp_path, monkeypatch):
    from weather_eval import storage

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    snap = _provider(hour=_hour_payload(_hour_rows()),
                     day=_day_payload()).fetch_snapshot(Station, [MODEL_NAME])
    assert storage.save_forecast_snapshot(Station.id, MODEL_NAME, snap) is True
    assert storage.save_forecast_snapshot(Station.id, MODEL_NAME, snap) is False
    assert len(storage.list_forecast_snapshots(Station.id, MODEL_NAME)) == 1


# ------------------------------------------------------------ 登记与健康哨兵
def test_cli_cma_gowfs_is_registered():
    import weather_eval.__main__ as m
    assert "cma_gowfs" in m.SOURCE_SPECS
    assert m.SOURCE_MODELS["cma_gowfs"] == {MODEL_NAME}


def test_health_warns_on_short_horizon(caplog):
    points, _ = extract_hour_points(
        _hour_payload(_hour_rows(native_hours=24, far_hours=24)))
    snap = _provider(hour=_hour_payload(_hour_rows(native_hours=24, far_hours=24)),
                     day=_day_payload()).fetch_snapshot(Station, [MODEL_NAME])
    with caplog.at_level(logging.WARNING):
        _provider(hour=_hour_payload(_hour_rows(native_hours=24, far_hours=24)),
                  day=_day_payload()).fetch_snapshot(Station, [MODEL_NAME])
    assert any("最大时效" in r.message for r in caplog.records)
    assert snap["native_1h_points"] == NATIVE_1H_POINTS


def test_health_warns_on_short_native_block(caplog):
    with caplog.at_level(logging.WARNING):
        _provider(hour=_hour_payload(_hour_rows(native_hours=10, far_hours=216)),
                  day=_day_payload()).fetch_snapshot(Station, [MODEL_NAME])
    assert any("原生逐小时段仅" in r.message for r in caplog.records)


def test_health_warns_on_conflicting_duplicates(caplog):
    rows = _hour_rows()
    rows.append(dict(rows[0], total_rain="1.23"))
    points, meta = extract_hour_points(_hour_payload(rows))
    assert meta["conflicts"] == 1
    with caplog.at_level(logging.WARNING):
        _provider(hour=_hour_payload(rows), day=_day_payload()) \
            .fetch_snapshot(Station, [MODEL_NAME])
    assert any("数值不一致" in r.message for r in caplog.records)
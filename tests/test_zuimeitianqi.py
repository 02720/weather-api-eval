"""ZuimeitianqiProvider（最美天气 PC 网页接口）接入测试。

mock 契约来自 2026-10-25 线上实测（4 站逐点核对 + 全国 20 余城抽样，
见 forecast/zuimeitianqi.py docstring）：

  GET https://h5-api.zuimeitianqi.com/h5zh/api/pc/actual?cityId={cityCode}
      → {"code":0,"data":{"hourly":[{date, dateFmt, time, temp, precipitation,...},
         ...共 360 点...],"actual":{...},"cityInfo":{"cityCode":...}}}
  GET https://h5-api.zuimeitianqi.com/h5zh/api/pc/forecast?cityId={cityCode}
      → {"code":0,"data":{"forecast":[{date, spanDaysNew, maxtemp, mintemp,
         rain, totalLiquid, ...}],"cityInfo":{...}}}

重点覆盖五处容易**静默**出错的契约：
1. 无效 cityId 不报错而是静默回退北京——只有 cityInfo.cityCode 回显能暴露它，
   回显不一致必须响亮失败（绝不让别家城市的预报冒充目标站点入库）；
2. 逐日列表混排历史日历与 +90 天趋势填充——不滤 spanDaysNew 会把平滑气候值
   冒充逐日起报，污染日轨样本；
3. 逐小时时刻的 epoch/dateFmt/time 三重声明互证——时区错 8 小时不报任何错，
   只会让样本静默归零；
4. 同义重复字段（maxtemp/max、rain/totalLiquid）不一致必须计数留痕，绝不
   静默二选一；
5. "HTTP 200 + 空 data" 的失效形态——空快照会被幂等键永久锁死。
"""
import json
from datetime import datetime, timedelta, timezone

import pytest
from conftest import FakeResp

from weather_eval.forecast.zuimeitianqi import (
    BASE_URL,
    DAILY_HORIZON_DAYS,
    HOURLY_POINTS,
    MODEL_NAME,
    ZmtqCityEchoError,
    ZmtqPayloadError,
    ZuimeitianqiProvider,
    check_city_echo,
    extract_daily_block,
    extract_hour_points,
)

NOW = datetime(2026, 10, 25, 15, 0)          # 抓取时刻（北京时整点）
CITY = "0101081307"
CITY_NAME = "万秀区"
CITY_LAT, CITY_LON = 23.472961, 111.32054
_BJ = timezone(timedelta(hours=8))


class Station:
    id, name, lat, lon = "wuzhou", "梧州", 23.4783, 111.304
    zmtq_city_id = CITY


class StationWithoutCity:
    id, name, lat, lon = "nodefault", "无码站", 23.0, 111.0
    zmtq_city_id = None


class RoutingSession:
    """按 URL 路由的假会话；记录调用次数供"确定性失败不重试"断言。"""

    def __init__(self, responder):
        self.responder = responder
        self.calls: list[str] = []

    def request(self, method, url, **kw):
        self.calls.append(url)
        status, text = self.responder(url)
        return FakeResp(text, status_code=status)

    def get(self, url, **kw):
        return self.request("GET", url, **kw)


def _hour_entry(dt: datetime, temp=30.0, precip=0.0) -> dict:
    """一个逐小时条目：epoch/dateFmt/time 三重声明一致（契约第 3 条）。"""
    return {
        "date": int(dt.replace(tzinfo=_BJ).timestamp() * 1000),
        "dateFmt": dt.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
        "time": dt.strftime("%Y-%m-%d %H:%M:%S"),
        "temp": temp, "precipitation": precip,
        "humidity": 48, "realfeel": 33, "weatherid": 1,
    }


def _index_payload(hours: list[dict], city: str = CITY,
                   name: str = CITY_NAME) -> dict:
    # /actual 的逐小时键是 hourly（原 /index 为 hour，2026-10-25 换端点）
    return {"code": 0, "data": {
        "hourly": hours,
        "actual": {"temperature": 30, "precipitation": 0},
        "cityInfo": {"cityCode": city, "localizedName": name,
                     "lat": CITY_LAT, "lng": CITY_LON},
    }}


def _day_entry(date_str: str, span_new, tmax=30, tmin=20, rain=0.0) -> dict:
    return {"date": date_str, "spanDaysNew": span_new,
            "maxtemp": tmax, "mintemp": tmin, "max": tmax, "min": tmin,
            "rain": rain, "totalLiquid": rain, "precProb": 10,
            "conditionDay": 1, "conditionNight": 2}


def _forecast_payload(days: list[dict], city: str = CITY,
                      name: str = CITY_NAME) -> dict:
    return {"code": 0, "data": {"forecast": days,
                                "cityInfo": {"cityCode": city,
                                             "localizedName": name}}}


def _happy_hours() -> list[dict]:
    """360 点连续轴：NOW..NOW+359h（契约第 4 条，2026-10-25 起 /actual 15 天）。"""
    return [_hour_entry(NOW + timedelta(hours=i), temp=30.0 - i * 0.1,
                        precip=1.0 if i == 5 else 0.0)
            for i in range(HOURLY_POINTS)]


def _happy_days(issue_day: str) -> list[dict]:
    """16 个真预报日（spanDaysNew 15..0）+ 历史日历与趋势填充（必须被滤除）。"""
    base = datetime.strptime(issue_day, "%Y-%m-%d")
    days = []
    for off in range(-3, 0):   # 历史日历（spanDaysNew=-1）
        d = (base + timedelta(days=off)).strftime("%Y-%m-%d")
        days.append(_day_entry(d, -1, tmax=99, tmin=99, rain=88.0))
    for span_new in range(DAILY_HORIZON_DAYS - 1, -1, -1):   # 15..0
        d = (base + timedelta(days=DAILY_HORIZON_DAYS - 1 - span_new)).strftime("%Y-%m-%d")
        days.append(_day_entry(d, span_new, tmax=30, tmin=20,
                               rain=2.5 if span_new % 3 == 0 else 0.0))
    for off in range(DAILY_HORIZON_DAYS, DAILY_HORIZON_DAYS + 5):  # +90 天趋势
        d = (base + timedelta(days=off)).strftime("%Y-%m-%d")
        days.append(_day_entry(d, -1, tmax=22, tmin=13, rain=1.4))
    return days


def _routes(actual_payload, forecast_payload, forecast_status=200):
    def responder(url: str):
        if url.startswith(f"{BASE_URL}/actual"):
            return 200, json.dumps(actual_payload)
        if url.startswith(f"{BASE_URL}/forecast"):
            return forecast_status, json.dumps(forecast_payload)
        return 404, "{}"
    return responder


def _provider(responder, **kw) -> ZuimeitianqiProvider:
    kw.setdefault("retries", 0)
    return ZuimeitianqiProvider(session=RoutingSession(responder), **kw)


def _happy_provider(forecast_city=CITY) -> ZuimeitianqiProvider:
    return _provider(_routes(_index_payload(_happy_hours()),
                             _forecast_payload(_happy_days("2026-10-25"),
                                               city=forecast_city)))


# ------------------------------------------------------------------ 纯函数：城市回显
def test_check_city_echo_accepts_matching_city():
    info = check_city_echo(_index_payload(_happy_hours()), CITY)
    assert info["localizedName"] == CITY_NAME


def test_check_city_echo_rejects_beijing_fallback():
    """实测无效 cityId 静默回退北京（01010101）且业务码仍为 0——必须响亮失败。"""
    payload = _index_payload(_happy_hours(), city="01010101", name="北京市")
    with pytest.raises(ZmtqCityEchoError, match="01010101"):
        check_city_echo(payload, CITY)


def test_check_city_echo_rejects_missing_echo():
    payload = {"code": 0, "data": {"hour": _happy_hours()}}
    with pytest.raises(ZmtqPayloadError, match="cityInfo"):
        check_city_echo(payload, CITY)


def test_check_city_echo_rejects_bad_envelope():
    with pytest.raises(ZmtqPayloadError, match="业务码"):
        check_city_echo({"code": 1, "message": "err"}, CITY)
    with pytest.raises(ZmtqPayloadError, match="data"):
        check_city_echo({"code": 0, "data": ""}, CITY)


# ------------------------------------------------------------------ 纯函数：逐小时
def test_extract_hour_points_parses_and_orders():
    pts, meta = extract_hour_points(_index_payload(_happy_hours()), CITY, {})
    assert len(pts) == HOURLY_POINTS
    assert pts[0][0] == NOW and pts[-1][0] == NOW + timedelta(hours=HOURLY_POINTS - 1)
    assert pts[5][1]["precipitation"] == 1.0
    assert meta["city_name"] == CITY_NAME and meta["city_lat"] == CITY_LAT


def test_extract_hour_points_skips_time_disagreement():
    """dateFmt/time 与 epoch 不一致的点位跳过并计数（系统性漂移 → 0 点 → 响亮失败）。"""
    hours = _happy_hours()
    hours[3]["time"] = "2026-10-25 03:00:00"      # 墙钟串错 8 小时（UTC 形态）
    hours[7]["dateFmt"] = "2026-10-25T15:00:00Z"  # 偏移标注漂移
    stats: dict = {}
    pts, _ = extract_hour_points(_index_payload(hours), CITY, stats)
    assert len(pts) == HOURLY_POINTS - 2 and stats["time_disagreement"] == 2
    assert pts[3][0] == NOW + timedelta(hours=4)  # 跳点后轴仍升序衔接


def test_extract_hour_points_all_disagreement_raises():
    """三重声明整体漂移（如整体换 UTC）→ 0 点 → 调用方拒绝空快照。"""
    hours = _happy_hours()
    for h in hours:
        h["dateFmt"] = h["dateFmt"].replace("+08:00", "+00:00")
    stats: dict = {}
    pts, _ = extract_hour_points(_index_payload(hours), CITY, stats)
    assert pts == [] and stats["time_disagreement"] == HOURLY_POINTS


def test_extract_hour_points_rejects_empty_hour_list():
    with pytest.raises(ZmtqPayloadError, match="hour"):
        extract_hour_points(_index_payload([]), CITY, {})


# ------------------------------------------------------------------ 纯函数：逐日
def test_extract_daily_block_filters_filler_days():
    """/forecast 混排的历史日历与趋势填充必须被滤除（契约第 6 条的最隐蔽陷阱）。"""
    days = extract_daily_block(_forecast_payload(_happy_days("2026-10-25")),
                               CITY, "2026-10-25", {})
    assert [d for d, _ in days] == [
        (datetime(2026, 10, 25) + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(DAILY_HORIZON_DAYS)]
    # 填充项的数值（99/88/1.4）一个都不许混进来
    assert all(row["temp_max"] == 30 and row["precipitation"] in (0.0, 2.5)
               for _, row in days)


def test_extract_daily_block_counts_missing_span_new():
    """spanDaysNew 缺失的条目按非真预报排除并计数（绝不猜测语义）。"""
    payload = _forecast_payload(
        [_day_entry("2026-10-25", 15), _day_entry("2026-10-26", None),
         _day_entry("2026-10-27", True)])   # bool 是 int 子类，必须挡
    stats: dict = {}
    days = extract_daily_block(payload, CITY, "2026-10-25", stats)
    assert [d for d, _ in days] == ["2026-10-25"]
    assert stats["span_new_excluded"] == 2


def test_extract_daily_block_drops_pre_issue_days():
    stats: dict = {}
    days = extract_daily_block(
        _forecast_payload([_day_entry("2026-10-24", 16, tmax=35),
                           _day_entry("2026-10-25", 15)]),
        CITY, "2026-10-25", stats)
    assert [d for d, _ in days] == ["2026-10-25"]
    assert stats["pre_issue_days"] == 1


def test_extract_daily_block_counts_synonym_field_mismatch():
    """maxtemp/max、rain/totalLiquid 不一致：以 maxtemp/rain 为准并计数留痕。"""
    entries = [_day_entry("2026-10-25", 15, tmax=30, rain=3.3)]
    entries[0]["max"] = 25        # 同义字段打架
    entries[0]["totalLiquid"] = 0.0
    stats: dict = {}
    days = extract_daily_block(_forecast_payload(entries), CITY, "2026-10-25", stats)
    assert days[0][1]["temp_max"] == 30 and days[0][1]["precipitation"] == 3.3
    assert stats["mismatch_temp_max"] == 1 and stats["mismatch_precipitation"] == 1


def test_extract_daily_block_nonphysical_values_missing():
    entries = [_day_entry("2026-10-25", 15, tmax=150, tmin=-200, rain=-0.5),
               _day_entry("2026-10-26", 14, rain=999.9)]
    stats: dict = {}
    days = extract_daily_block(_forecast_payload(entries), CITY, "2026-10-25", stats)
    assert days[0][1] == {"temp_max": None, "temp_min": None, "precipitation": None}
    assert days[1][1]["precipitation"] is None
    assert stats["temp_implausible"] == 2 and stats["negative"] == 1
    assert stats["sentinel"] == 1


def test_extract_daily_block_rejects_empty_forecast():
    with pytest.raises(ZmtqPayloadError, match="forecast"):
        extract_daily_block(_forecast_payload([]), CITY, "2026-10-25", {})


# ------------------------------------------------------------------ 快照整体
def test_snapshot_shape_and_contract():
    prov = _happy_provider()
    snap = prov.fetch_snapshot(Station, [MODEL_NAME])
    assert snap["issue_iso"] == "2026-10-25T15:00"
    # 滚动 24h 产品锚点 ≈ 抓取时刻下取整（契约第 4 条，同 AccuWeather 的处置）
    assert snap["issue_source"] == "request_floor"
    assert snap["issue_raw"] == "2026-10-25T15:00"
    assert snap["source"] == "zuimeitianqi" and snap["models"] == [MODEL_NAME]
    assert snap["resolution_hours"] == 1 and snap["precip_unit"] == "mm"
    assert snap["precip_accum_window_hours"] == 1
    # grid_* 是吸附城市坐标；requested_* 是站点坐标（契约第 11 条）
    assert snap["grid_lat"] == CITY_LAT and snap["grid_lon"] == CITY_LON
    assert snap["requested_lat"] == Station.lat and snap["requested_lon"] == Station.lon
    assert snap["city_code"] == CITY and snap["city_name"] == CITY_NAME
    # 逐小时轴 24 点、首尾对齐、缺测列与轴等长
    assert snap["hourly_time"] == [
        (NOW + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
        for i in range(HOURLY_POINTS)]
    series = snap["data"][MODEL_NAME]
    assert len(series["temperature_2m"]) == HOURLY_POINTS
    assert series["precipitation"][5] == 1.0
    # 逐日块：16 天、与轴索引对齐
    assert snap["daily_time"] == [
        (datetime(2026, 10, 25) + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(DAILY_HORIZON_DAYS)]
    daily = snap["daily"][MODEL_NAME]
    for key in ("temp_max", "temp_min", "precipitation"):
        assert len(daily[key]) == len(snap["daily_time"])
    assert daily["temp_max"][0] == 30 and daily["precipitation"][0] == 2.5


def test_snapshot_wrong_index_city_fails_loudly_without_retry():
    """/index 回显北京（静默回退形态）：响亮失败且确定性失败不重试。"""
    prov = ZuimeitianqiProvider(session=RoutingSession(_routes(
        _index_payload(_happy_hours(), city="01010101", name="北京市"),
        _forecast_payload(_happy_days("2026-10-25")))), retries=3)
    with pytest.raises(ZmtqCityEchoError, match="北京市"):
        prov.fetch_snapshot(Station, [MODEL_NAME])
    assert len(prov.session.calls) == 1


def test_snapshot_forecast_city_mismatch_drops_daily_only():
    """/forecast 回显不一致：丢弃日块保逐小时主干（错误数据绝不入库）。"""
    prov = _happy_provider(forecast_city="01010101")
    snap = prov.fetch_snapshot(Station, [MODEL_NAME])
    assert "daily_time" not in snap and "daily" not in snap
    assert any("城市回显不一致" in n for n in snap["issue_notes"])


def test_snapshot_forecast_http_error_drops_daily_only():
    """逐日端点 5xx 穷尽（retries=0）：日块丢弃、逐小时快照照常入库。"""
    prov = _provider(_routes(_index_payload(_happy_hours()),
                             _forecast_payload(_happy_days("2026-10-25")),
                             forecast_status=500))
    snap = prov.fetch_snapshot(Station, [MODEL_NAME])
    assert "daily_time" not in snap
    assert any("逐日端点不可用" in n for n in snap["issue_notes"])


def test_snapshot_daily_without_issue_day_dropped():
    """日界翻转形态（日块已滚到次日、不含起报日）：日块丢弃并留痕。"""
    payload = _forecast_payload(_happy_days("2026-10-26"))  # 首日=明天
    prov = _provider(_routes(_index_payload(_happy_hours()), payload))
    snap = prov.fetch_snapshot(Station, [MODEL_NAME])
    assert "daily_time" not in snap
    assert any("不含起报日" in n for n in snap["issue_notes"])


def test_snapshot_rejects_empty_hour_list():
    """HTTP 200 + 空 hourly 列表（cityId 无逐小时产品或契约漂移）：响亮失败。"""
    prov = _provider(_routes(_index_payload([]),
                             _forecast_payload(_happy_days("2026-10-25"))))
    with pytest.raises(RuntimeError, match="hour"):
        prov.fetch_snapshot(Station, [MODEL_NAME])


def test_snapshot_rejects_all_points_skipped():
    """三重时间声明整体漂移 → 0 个有效时刻 → 拒绝空快照（幂等键保护）。"""
    hours = _happy_hours()
    for h in hours:
        h["dateFmt"] = h["dateFmt"].replace("+08:00", "+00:00")
        h["time"] = h["time"][:11] + "0" + h["time"][12:]  # 墙钟串同样错位
    prov = _provider(_routes(_index_payload(hours),
                             _forecast_payload(_happy_days("2026-10-25"))))
    with pytest.raises(RuntimeError, match="空快照"):
        prov.fetch_snapshot(Station, [MODEL_NAME])


def test_snapshot_rejects_all_null_values():
    hours = [_hour_entry(NOW + timedelta(hours=i), temp=None, precip=None)
             for i in range(HOURLY_POINTS)]
    prov = _provider(_routes(_index_payload(hours),
                             _forecast_payload(_happy_days("2026-10-25"))))
    with pytest.raises(RuntimeError, match="空快照"):
        prov.fetch_snapshot(Station, [MODEL_NAME])


def test_snapshot_missing_city_id_fails_loudly():
    prov = _happy_provider()
    with pytest.raises(RuntimeError, match="zmtq_city_id"):
        prov.fetch_snapshot(StationWithoutCity, [MODEL_NAME])
    assert prov.session.calls == []          # 请求根本不该发出


def test_snapshot_hourly_gap_kept_as_is():
    """原生 1h 产品缺点即真实缺口：绝不插值（契约第 10 条）。"""
    hours = _happy_hours()
    del hours[10]                            # 缺一个整点
    prov = _provider(_routes(_index_payload(hours),
                             _forecast_payload(_happy_days("2026-10-25"))))
    snap = prov.fetch_snapshot(Station, [MODEL_NAME])
    assert len(snap["hourly_time"]) == HOURLY_POINTS - 1
    assert any("缺口" in n for n in snap["issue_notes"])


def test_snapshot_hourly_count_drift_warns():
    """逐小时轴不是 360 点（产品形态漂移）必须可见。"""
    hours = _happy_hours()[:20]
    prov = _provider(_routes(_index_payload(hours),
                             _forecast_payload(_happy_days("2026-10-25"))))
    snap = prov.fetch_snapshot(Station, [MODEL_NAME])
    assert len(snap["hourly_time"]) == 20
    assert any("20 点" in n for n in snap["issue_notes"])


def test_snapshot_daily_horizon_growth_warns():
    """真预报天数超过实测的 16 天（视野扩展）必须可见。"""
    days = _happy_days("2026-10-25")
    days.append(_day_entry("2026-11-10", 0))  # spanDaysNew=0 但日期在 +16 天外
    prov = _provider(_routes(_index_payload(_happy_hours()),
                             _forecast_payload(days)))
    snap = prov.fetch_snapshot(Station, [MODEL_NAME])
    assert len(snap["daily_time"]) == DAILY_HORIZON_DAYS + 1
    assert any("视野可能已扩展" in n for n in snap["issue_notes"])


# ------------------------------------------------------------------ 接线
def test_source_registered_for_cli():
    """CLI SOURCE_SPECS 与 forecast/__init__ 的登记必须同步（缺一即运行期 KeyError）。"""
    from weather_eval.__main__ import SOURCE_MODELS
    assert "zuimeitianqi" in SOURCE_MODELS
    assert SOURCE_MODELS["zuimeitianqi"] == {MODEL_NAME}


def test_station_config_carries_zmtq_city_id():
    """stations.yaml 的 zmtq_city_id 必须被 config 层接收（漏配=运行期响亮失败）。"""
    from weather_eval.config import load_config
    cfg = load_config()
    by_id = {s.id: s for s in cfg.stations}
    assert by_id["wuzhou"].zmtq_city_id == "0101081307"
    assert by_id["wanning"].zmtq_city_id == "01011016"
    assert MODEL_NAME in cfg.models

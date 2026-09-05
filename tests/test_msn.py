"""MSN 天气（msn.cn/zh-cn/weather，底层中国天气网）接入测试。

覆盖：SSR 状态抽取、loc 参数（x=经度/y=纬度）、逐日分片合并、字段映射（降水
必须取 rainAmount 而非降水概率 precipitation）、起报时刻锚定、版本漂移、降水守恒
哨兵、确定性失败不重试，以及与评估引擎端到端配对。
"""
import json
import logging
import re
import types
from datetime import datetime, timedelta

import pytest

from weather_eval.forecast.msn import (
    MAX_DAY, MsnProvider, ReduxStateMissing, check_rain_conservation,
    extract_state, forecast_url, iter_hourly, loc_param, _parse_entry_time,
)
from weather_eval import storage
from weather_eval.evaluate import build_report
from weather_eval.timeutil import iso


LU = "2026-09-05T15:22:15+08:00"   # lastUpdated：起报锚点来源


def _station():
    class S:
        id = "s1"
        lat = 23.4783
        lon = 111.304
    return S()


def _entry(t_local, temp, rain=None, accu=None, prob=None):
    """构造一个逐小时条目（time.dataValue 为 UTC、timeStr 为 +08:00 当地时）。

    rain=rainAmount(mm)、accu=raAccu(日内累计)、prob=precipitation(降水概率%，
    具有误导性的字段，实现绝不可误取)。
    """
    utc = t_local - timedelta(hours=8)
    e = {
        "temperature": temp,
        "time": {"dataType": "Date",
                 "dataValue": utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")},
        "timeStr": t_local.strftime("%Y-%m-%dT%H:%M:%S+08:00"),
    }
    if rain is not None:
        e["rainAmount"] = rain
    if accu is not None:
        e["raAccu"] = accu
    if prob is not None:
        e["precipitation"] = prob
    return e


def _state(days, last_updated=LU, source=None, provider=None):
    """days: 逐日 hourly 条目列表的列表。"""
    return {"WeatherData": {"_@STATE@_": {
        "lastUpdated": last_updated,
        "forecast": [{"hourly": h} for h in days],
        "source": source or {"id": "101300603", "location": {"Name": "万秀"},
                             "coordinates": {"lat": 23.4729, "lon": 111.3210}},
        "provider": provider or {"name": "中国天气网",
                                 "url": "http://www.weather.com.cnweather/101300601.shtml"},
    }}}


def _html(state):
    return ('<html><body><script id="redux-data" type="application/json">'
            + json.dumps(state, ensure_ascii=False)
            + '</script></body></html>')


class _PageSession:
    """按 URL 的 day 参数返回不同页面的假 session（模拟 SSR 一次只渲染一天）。

    pages: {day: html文本}；未登记的 day 回落到 default（None 时抛错模拟抓取失败）。
    """

    def __init__(self, pages: dict[int, str], default: str | None = None):
        self.pages = pages
        self.default = default
        self.calls = 0
        self.urls: list[str] = []

    def get(self, url, **kwargs):
        self.calls += 1
        self.urls.append(url)
        m = re.search(r"day=(\d+)", url)
        day = int(m.group(1)) if m else 1
        text = self.pages.get(day, self.default)
        if text is None:
            raise RuntimeError(f"day={day} 抓取失败")
        return types.SimpleNamespace(status_code=200, text=text)


class _StatusSession:
    """固定返回某 HTTP 状态码的假 session（用于验证确定性失败不重试）。"""

    def __init__(self, status):
        self.status = status
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return types.SimpleNamespace(status_code=self.status, text="error body")


# ----------------------------------------------------------------- loc / URL
def test_loc_param_puts_longitude_in_x():
    """x=经度、y=纬度——写反会静默抓到错误的地点，是本接口最易错的细节。"""
    import base64
    raw = json.loads(base64.urlsafe_b64decode(loc_param(23.4783, 111.304) + "=="))
    assert raw == {"x": "111.304", "y": "23.4783"}


def test_forecast_url_carries_celsius_and_day():
    url = forecast_url(23.4783, 111.304, 3)
    assert url.startswith("https://www.msn.cn/zh-cn/weather/forecast/in-23.4783,111.304?loc=")
    assert "weadegreetype=C" in url          # 缺此参数有返回华氏的风险
    assert url.endswith("&day=3")
    assert "day=" not in forecast_url(23.4783, 111.304)


# ----------------------------------------------------------------- 状态抽取
def test_extract_state_happy_path():
    st = extract_state(_html(_state([[{"temperature": 30}]])))
    assert st["lastUpdated"] == LU


def test_extract_state_missing_block_raises():
    """页面结构改版/被风控拦截时必须明确报错，绝不退化为空快照。"""
    with pytest.raises(ReduxStateMissing):
        extract_state("<html><body>no state here</body></html>")
    with pytest.raises(ReduxStateMissing):
        extract_state("")


def test_extract_state_bad_json_raises():
    html = '<script id="redux-data" type="application/json">{not json}</script>'
    with pytest.raises(ReduxStateMissing):
        extract_state(html)


def test_extract_state_missing_marker_raises():
    html = '<script id="redux-data" type="application/json">{"WeatherData":{}}</script>'
    with pytest.raises(ReduxStateMissing):
        extract_state(html)


# ----------------------------------------------------------------- 时间解析
def test_parse_entry_time_prefers_utc_data_value():
    e = {"time": {"dataValue": "2026-09-05T07:00:00.000Z"},
         "timeStr": "2026-09-05T15:00:00.000+08:00"}
    assert _parse_entry_time(e) == datetime(2026, 9, 5, 15, 0)


def test_parse_entry_time_falls_back_to_timeStr_both_forms():
    """timeStr 实测混用带/不带毫秒两种形态，不可按固定长度切片。"""
    assert _parse_entry_time({"timeStr": "2026-09-05T20:00:00+08:00"}) == \
        datetime(2026, 9, 5, 20, 0)
    assert _parse_entry_time({"timeStr": "2026-09-05T15:00:00.000+08:00"}) == \
        datetime(2026, 9, 5, 15, 0)


def test_parse_entry_time_unparseable_returns_none():
    assert _parse_entry_time({}) is None
    assert _parse_entry_time({"timeStr": "不是时间"}) is None


def test_iter_hourly_sorts_and_dedups():
    # 经真实链路取状态（iter_hourly 接收的是 _@STATE@_ 子树，不是整块 redux-data）
    st = extract_state(_html(_state([[
        _entry(datetime(2026, 9, 5, 17, 0), 28),
        _entry(datetime(2026, 9, 5, 15, 0), 30),
        _entry(datetime(2026, 9, 5, 15, 0), 31),   # 重复整点，保留首见
        {"temperature": 99},                        # 无时间，丢弃
    ]])))
    out = iter_hourly(st)
    assert [dt.strftime("%Y-%m-%dT%H:%M") for dt, _ in out] == \
        ["2026-09-05T15:00", "2026-09-05T17:00"]
    assert out[0][1]["temperature"] == 30


# ----------------------------------------------------------------- 降水守恒哨兵
def test_rain_conservation_detects_mismatch():
    """日内 raAccu 增量必须等于当小时 rainAmount，跨日边界不差分。"""
    t0 = datetime(2026, 9, 7, 15, 0)
    pts = [
        (t0, 0.32, 0.32),                              # 日内首点：重置为自身 ✓
        (t0 + timedelta(hours=1), 0.36, 0.68),         # ✓
        (t0 + timedelta(hours=2), 0.29, 0.97),         # ✓
        (t0 + timedelta(hours=3), 0.50, 9.99),         # ✗
        (datetime(2026, 9, 8, 0, 0), 1.0, 1.0),        # 新的一天：重置 ✓
    ]
    problems = check_rain_conservation(pts)
    assert len(problems) == 1
    assert "09-07T18:00" in problems[0]


def test_rain_conservation_no_false_positive_on_clean_series():
    """哨兵的价值取决于不误报：一串完全自洽的样本必须零问题（否则告警疲劳）。"""
    t0 = datetime(2026, 9, 7, 15, 0)
    pts, acc = [], 0.0
    for i in range(30):
        r = round(0.1 * (i % 5), 2)
        acc = r if (t0 + timedelta(hours=i)).strftime("%Y-%m-%d") != \
            (t0 + timedelta(hours=i - 1)).strftime("%Y-%m-%d") else round(acc + r, 2)
        pts.append(((t0 + timedelta(hours=i)), r, acc))
    assert check_rain_conservation(pts) == []


def test_rain_conservation_gap_does_not_produce_false_positive():
    """raAccu 缺测后基准置空：宁可让下一点无法校验，也不跨缺口差分产假告警。"""
    t0 = datetime(2026, 9, 7, 15, 0)
    pts = [
        (t0, 0.32, 0.32),
        (t0 + timedelta(hours=1), 0.36, None),     # 缺测
        (t0 + timedelta(hours=2), 0.29, 0.97),     # 若跨缺口差分：0.97-0.32=0.65≠0.29 会误报
    ]
    assert check_rain_conservation(pts) == []


def test_rain_conservation_tolerates_rounding():
    t0 = datetime(2026, 9, 7, 15, 0)
    pts = [(t0, 0.32, 0.32), (t0 + timedelta(hours=1), 0.36, 0.67)]  # 差 0.01
    assert check_rain_conservation(pts) == []


# ----------------------------------------------------------------- 快照装配
def _one_day_state(hourly, **kw):
    return _state([hourly], **kw)


def test_fetch_snapshot_maps_temperature_and_rain_amount():
    """降水必须取 rainAmount；precipitation 是降水概率%，误取会摧毁降水评分。"""
    base = datetime(2026, 9, 5, 15, 0)
    hourly = [
        _entry(base, 30, rain=0.0, accu=0.0, prob="10"),
        _entry(base + timedelta(hours=1), 29, rain=1.7, accu=1.7, prob="90"),
    ]
    sess = _PageSession({1: _html(_one_day_state(hourly))}, default=None)
    snap = MsnProvider(session=sess, retries=0).fetch_snapshot(_station())

    assert snap["source"] == "msn"
    assert snap["models"] == ["msn_v1"]
    assert snap["hourly_time"] == ["2026-09-05T15:00", "2026-09-05T16:00"]
    assert snap["data"]["msn_v1"]["temperature_2m"] == [30.0, 29.0]
    assert snap["data"]["msn_v1"]["precipitation"] == [0.0, 1.7]   # 不是 [10, 90]


def test_fetch_snapshot_issue_is_floor_hour_of_last_updated():
    base = datetime(2026, 9, 5, 15, 0)
    hourly = [_entry(base, 30), _entry(base + timedelta(hours=1), 29)]
    sess = _PageSession({1: _html(_one_day_state(hourly))})
    snap = MsnProvider(session=sess, retries=0).fetch_snapshot(_station())
    # lastUpdated=15:22:15 → 起报 15:00；首点 15:00 的 lead=0（由评估引擎排除）
    assert snap["issue_iso"] == "2026-09-05T15:00"
    assert snap["last_updated"] == LU


def test_fetch_snapshot_issues_ten_requests_and_merges():
    """SSR 一次只渲染一天：完整快照需 day=1..10，合并后时间轴连续。"""
    base = datetime(2026, 9, 5, 15, 0)
    pages = {}
    for day in range(1, MAX_DAY + 1):
        start = base + timedelta(days=day - 1)
        if day == 1:
            start = base
        hourly = [_entry(start + timedelta(hours=h), 20 + h % 5)
                  for h in range(24)]
        pages[day] = _html(_one_day_state(hourly))
    sess = _PageSession(pages)
    snap = MsnProvider(session=sess, retries=0).fetch_snapshot(_station())
    assert sess.calls == MAX_DAY
    assert snap["days_fetched"] == MAX_DAY
    assert len(snap["hourly_time"]) == 24 * MAX_DAY


def test_max_day_is_clamped_to_server_limit():
    """day>10 会被服务端静默回退为滚动窗口，上界不可越过。"""
    assert MsnProvider(max_day=99).max_day == MAX_DAY
    assert MsnProvider(max_day=0).max_day == 1


def test_version_drift_restarts_on_new_version(caplog):
    """抓取途中数据版本推进：舍弃旧版本整体重抓，快照内绝不跨版本。"""
    base = datetime(2026, 9, 5, 15, 0)
    new_lu = "2026-09-05T16:05:00+08:00"

    def page(t0, temp, lu):
        return _html(_one_day_state([_entry(t0, temp),
                                     _entry(t0 + timedelta(hours=1), temp)],
                                    last_updated=lu))

    # 第一次遍历：day=1 是旧版本，day>=2 全是新版本 → 触发一次整体重抓
    old_first = {1: page(base, 30, LU)}
    calls = {"n": 0}

    class _RollSession:
        def get(self, url, **kwargs):
            m = re.search(r"day=(\d+)", url)
            day = int(m.group(1)) if m else 1
            calls["n"] += 1
            # 前 2 次请求（day=1 与 day=2）仍属旧版本，之后服务端已滚动到新版本
            text = page(base, 30, LU) if calls["n"] <= 2 else page(base, 29, new_lu)
            return types.SimpleNamespace(status_code=200, text=text)

    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.msn"):
        snap = MsnProvider(session=_RollSession()).fetch_snapshot(_station())

    assert any("整体重抓" in r.message for r in caplog.records)
    # 重抓后整份快照都来自新版本：起报锚点、数值均为新版本
    assert snap["issue_iso"] == "2026-09-05T16:00"
    assert snap["last_updated"] == new_lu
    assert snap["days_fetched"] == MAX_DAY
    # 重抓成功 → 快照并不残缺；漂移史单独留档
    assert snap["dropped_days"] == []
    assert len(snap["drift_history"]) == 1
    assert snap["version_restarts"] == 1
    # 旧版本温度 30 / 新版本 29：快照内不得残留旧版本数值
    assert set(snap["data"]["msn_v1"]["temperature_2m"]) == {29.0}


def test_version_drift_beyond_restart_limit_keeps_partial(caplog):
    """重抓后再次漂移（极罕见）：中止并保留**新版本**的残缺快照，残缺留档可见。

    版本脚本：A×4 → B（day5 漂移，整体重抓）→ B×3 → C（day4 再漂移，重抓已用尽）
    → 中止，此时持有 B 版的 3 个分片。
    """
    base = datetime(2026, 9, 5, 15, 0)
    script = [LU] * 4 + ["2026-09-05T16:05:00+08:00"] * 4 + ["2026-09-05T17:05:00+08:00"]

    class _ScriptedSession:
        def __init__(self):
            self.n = 0

        def get(self, url, **kwargs):
            i = self.n
            self.n += 1
            lu = script[i] if i < len(script) else script[-1]
            st = _one_day_state([_entry(base + timedelta(hours=i), 30)], last_updated=lu)
            return types.SimpleNamespace(status_code=200, text=_html(st))

    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.msn"):
        snap = MsnProvider(session=_ScriptedSession()).fetch_snapshot(_station())

    assert any("重抓上限" in r.message for r in caplog.records)
    assert snap["days_fetched"] == 3                       # 中止在 B 版第 4 个分片
    assert snap["last_updated"] == "2026-09-05T16:05:00+08:00"   # 锚点属 B 版
    assert snap["issue_iso"] == "2026-09-05T16:00"
    assert len(snap["dropped_days"]) == 1                  # 仅第二次漂移属实际损失
    assert len(snap["drift_history"]) == 2                 # 两次漂移均留档


def test_version_rolling_every_request_fails_loudly():
    """服务端每次请求都换版本：永远凑不出一致快照 → 明确报错，绝不产出混合数据。"""
    base = datetime(2026, 9, 5, 15, 0)

    class _AlwaysRollSession:
        def __init__(self):
            self.n = 0

        def get(self, url, **kwargs):
            self.n += 1
            st = _one_day_state([_entry(base + timedelta(hours=self.n), 30)],
                                last_updated=f"2026-09-05T{15 + self.n:02d}:05:00+08:00")
            return types.SimpleNamespace(status_code=200, text=_html(st))

    with pytest.raises(RuntimeError, match="未能取得任何逐小时预报"):
        MsnProvider(session=_AlwaysRollSession()).fetch_snapshot(_station())


def test_rain_conservation_warns_on_inconsistent_series(caplog):
    base = datetime(2026, 9, 7, 15, 0)
    hourly = [
        _entry(base, 28, rain=0.32, accu=0.32),
        _entry(base + timedelta(hours=1), 28, rain=0.36, accu=9.99),
    ]
    sess = _PageSession({1: _html(_one_day_state(hourly))})
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.msn"):
        MsnProvider(session=sess, retries=0).fetch_snapshot(_station())
    assert any("raAccu" in r.message for r in caplog.records)


def test_missing_precipitation_field_yields_none():
    base = datetime(2026, 9, 5, 15, 0)
    hourly = [_entry(base, 30)]          # 无 rainAmount
    sess = _PageSession({1: _html(_one_day_state(hourly))})
    snap = MsnProvider(session=sess, retries=0).fetch_snapshot(_station())
    assert snap["data"]["msn_v1"]["precipitation"] == [None]


def test_all_slices_failing_raises():
    sess = _PageSession({}, default=None)
    with pytest.raises(RuntimeError, match="未能取得任何逐小时预报"):
        MsnProvider(session=sess, retries=0).fetch_snapshot(_station())


def test_partial_failure_circuit_breaks_and_archives(caplog):
    """残缺入库是显式决策（起报快照错过即无法追补），残缺必须可见、且不再空耗。

    熔断：day=2 失败后不应继续为 day=3..10 各烧一轮退避（同一端点同参数形态，
    剩余分片必然同样失败）。
    """
    base = datetime(2026, 9, 5, 15, 0)
    sess = _PageSession({1: _html(_one_day_state([_entry(base, 30)]))}, default=None)
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.msn"):
        snap = MsnProvider(session=sess, retries=0).fetch_snapshot(_station())
    assert snap["days_fetched"] == 1
    assert len(snap["dropped_days"]) == 1          # 熔断：只记录了首个失败分片
    assert sess.calls == 2                          # day=1 成功 + day=2 失败即止
    assert any("快照残缺" in r.message for r in caplog.records)
    assert any("中止该站剩余分片" in r.message for r in caplog.records)


def test_records_snapping_metadata():
    base = datetime(2026, 9, 5, 15, 0)
    sess = _PageSession({1: _html(_one_day_state([_entry(base, 30)]))})
    snap = MsnProvider(session=sess, retries=0).fetch_snapshot(_station())
    assert snap["location_name"] == "万秀"
    assert snap["location_id"] == "101300603"
    assert snap["provider_name"] == "中国天气网"
    # 最近城市吸附语义：留档距离供报告披露，绝不当作点预报
    assert 0 < snap["location_distance_km"] < 20
    assert snap["grid_lat"] == 23.4729 and snap["grid_lon"] == 111.3210
    assert snap["requested_lat"] == 23.4783


def test_http_4xx_is_deterministic_no_retry():
    sess = _StatusSession(404)
    with pytest.raises(RuntimeError):
        MsnProvider(session=sess, retries=3).fetch_snapshot(_station())
    assert sess.calls == 1      # 确定性失败不烧退避


# ----------------------------------------------------------------- CLI 接线
def test_build_provider_selects_msn_and_excludes_it_from_open_meteo(monkeypatch):
    import weather_eval.__main__ as m
    from weather_eval.forecast.msn import MsnProvider as MP

    class Station:
        id = "s1"
        lat = 23.0
        lon = 111.0

    class Cfg:
        models = ["ecmwf_ifs", "msn_v1", "accuweather_v1"]
        stations = [Station()]

    prov, models = m._build_provider("msn", Cfg())
    assert isinstance(prov, MP) and models == ["msn_v1"]

    class OM:
        def fetch_snapshot(self, station, models):
            raise AssertionError("不应真的请求")

    monkeypatch.setattr(m, "OpenMeteoProvider", lambda: OM())
    om_prov, om_models = m._build_provider("open_meteo", Cfg())
    assert om_models == ["ecmwf_ifs"]      # msn_v1 不得被当作 Open-Meteo 模型


# ----------------------------------------------------------------- 端到端
def test_end_to_end_pairs_with_obs(tmp_path, monkeypatch):
    """MSN 快照与整点观测完美匹配 → lead=0 被排除，lead≥1 的样本全部命中。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    base = datetime(2026, 9, 5, 15, 0)
    temps = [30.0, 29.0, 28.0, 27.0]
    rains = [0.0, 1.7, 0.0, 0.0]

    obs = []
    for i in range(len(temps)):
        t = base + timedelta(hours=i)
        obs.append({"time": iso(t), "temp": temps[i], "rain": rains[i]})
    storage.save_obs("s1", obs)

    hourly = [_entry(base + timedelta(hours=i), temps[i], rain=rains[i],
                     accu=(rains[i] if i == 0 else None), prob="50")
              for i in range(len(temps))]
    sess = _PageSession({1: _html(_one_day_state(hourly))})
    snap = MsnProvider(session=sess, retries=0).fetch_snapshot(_station())
    storage.save_forecast_snapshot("s1", "msn_v1", snap)

    end = base + timedelta(hours=3)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 0}
    data = build_report(["s1"], ["msn_v1"], cfg, base, end, "2026-09")

    # 15:00(lead=0) 被评估引擎排除 → 16:00/17:00/18:00 三对
    sc = data["scorecard"]["msn_v1"]["temp_24h"]
    assert sc["n"] == 3
    assert sc["acc2"] == 100.0
    pb = data["scorecard"]["msn_v1"]["precip_24h"]
    assert abs(pb["ts"] - 1.0) < 1e-6

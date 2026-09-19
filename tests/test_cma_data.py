"""中国气象数据网实况源（obs/cma_data.py）的回归测试。

这里的用例把**逆向得到的接口契约**钉死，每一条都对应一次实测（见模块 docstring）：
请求参数按 UTC 解释、时间键取响应自述的 D_datetime、空 content 是缺测而非 0.0、
整窗全空视为抓取失败、缺 cma_id 响亮失败、UA 必须在白名单内。
契约哪天变了，这些测试应当先红，而不是等观测被静默错标后才被人发现。
"""
from __future__ import annotations

import json
import logging

import pytest

from conftest import FakeResp

from weather_eval.obs.cma_data import (
    ELEMENT_MAP, HEADERS, PARAM_UTC_OFFSET_HOURS, SOURCE_TAG, CmaDataObsSource,
    _target_hours, _to_float,
)
from weather_eval.timeutil import now_beijing


class _Station:
    id = "wuzhou"
    name = "梧州气象站"
    cma_id = "59265"


class _NoCmaStation:
    id = "ghost"
    name = "无站号站"
    cma_id = None


def _content(**over) -> dict:
    c = {
        "D_datetime": "2026-09-19 22:00:00",
        "V12001": 28.9, "V13003": 80, "V10004": 1002.7,
        "V11292T": "东北风", "V11293T": "1级", "V11293": 1.4,
        "V20003T": "霾", "V13019": 0.0, "V20001": 7100,
    }
    c.update(over)
    return c


def _payload(content) -> str:
    return json.dumps({"message": "成功！", "status": 0, "code": 200, "type": 1,
                       "content": content}, ensure_ascii=False)


class _StationSession:
    """按 datetime 参数返回预设 content 的假会话；记录每次请求的参数。"""

    def __init__(self, content_for: dict | None = None, default=None):
        self.content_for = content_for or {}
        self.default = default
        self.params: list[str] = []
        self.urls: list[str] = []
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        self.urls.append(url)
        dt = (kwargs.get("params") or {}).get("datetime")
        self.params.append(dt)
        content = self.content_for.get(dt, self.default)
        if content is None:
            content = {}
        return FakeResp(_payload(content))


# --------------------------------------------------------------- 要素码与转换
def test_element_map_covers_verified_codes():
    """要素码映射是实测对拍确认的口径（气温 ΔT=0.00℃、降水 ΔR=0.00mm）。"""
    assert ELEMENT_MAP["V12001"] == "temp"
    assert ELEMENT_MAP["V13019"] == "rain"
    assert ELEMENT_MAP["V13003"] == "humidity"
    assert ELEMENT_MAP["V10004"] == "pressure"
    assert ELEMENT_MAP["V11293"] == "wind_speed"
    assert ELEMENT_MAP["V20001"] == "visibility"


def test_to_float_never_fabricates_zero():
    """缺测绝不折算成 0.0——本项目最重要的地基之一。"""
    for empty in (None, "", "   ", "-", "null", "nan", "NA", "—"):
        assert _to_float(empty) is None
    assert _to_float("0") == 0.0        # 真正的 0 要保留
    assert _to_float(0.0) == 0.0
    assert _to_float("4.6") == 4.6
    assert _to_float(True) is None      # bool 不是观测值


def test_uheaders_use_browser_ua():
    """实测：空 UA → HTTP 403；python-requests UA → 服务端直接断连。

    UA 被改成一个会被反爬名单命中的值，抓取会整体失效且失败形态很吵（403/断连），
    但根因（UA）离现象很远。这里把"必须是浏览器 UA"钉成回归。
    """
    ua = HEADERS["User-Agent"]
    assert ua.startswith("Mozilla/5.0")
    assert "python-requests" not in ua and "curl" not in ua


# ------------------------------------------------------------------- 时间语义
def test_target_hours_window_is_hourly_and_inclusive():
    now = now_beijing().replace(minute=37, second=5, microsecond=0)
    hours = _target_hours(3, now)
    assert len(hours) == 4                      # 含两端
    assert all(h.minute == 0 and h.second == 0 for h in hours)
    assert hours[-1] == now.replace(minute=0, second=0, microsecond=0)
    assert (hours[-1] - hours[0]).total_seconds() == 3 * 3600


def test_request_param_is_target_minus_utc_offset():
    """接口的 datetime 参数按 UTC 解释：请求 = 目标北京时 − 8h。

    这条错了不会报错——只会把每一个小时都错标 8 小时（本项目对 EW4ALL 吃过同型的亏）。
    """
    target = "2026-09-19 22:00:00"
    c = _content(D_datetime=target)
    sess = _StationSession({"20260919140000": c})
    src = CmaDataObsSource(lookback_hours=0, sleep_seconds=0.0, session=sess)

    # 把"当前时刻"钉在目标整点上，使唯一的目标小时就是 target
    import weather_eval.obs.cma_data as mod
    real = mod.now_beijing
    mod.now_beijing = lambda: mod.parse_obs_time(target)
    try:
        recs = src.fetch(_Station())
    finally:
        mod.now_beijing = real

    assert PARAM_UTC_OFFSET_HOURS == 8
    assert sess.params == ["20260919140000"], "请求参数必须是目标北京时减 8 小时"
    assert recs[0]["time"] == "2026-09-19T22:00"


def test_time_key_comes_from_payload_not_request():
    """时间键取响应自述的 D_datetime，绝不使用请求时刻。

    接口"回显晚于请求"是语义漂移的信号；即便如此，宁可把观测记在它**自称**的时刻上
    （并告警），也不能按请求时刻落库——错标的观测会污染全部评估。
    """
    c = _content(D_datetime="2026-09-19 23:00:00")   # 回显比请求晚 1 小时
    sess = _StationSession(default=c)
    src = CmaDataObsSource(lookback_hours=0, sleep_seconds=0.0, session=sess)

    import weather_eval.obs.cma_data as mod
    real = mod.now_beijing
    mod.now_beijing = lambda: mod.parse_obs_time("2026-09-19 22:00:00")
    try:
        recs = src.fetch(_Station())
    finally:
        mod.now_beijing = real
    assert any(r["time"] == "2026-09-19T23:00" for r in recs)
    assert not any(r["time"] == "2026-09-19T22:00" for r in recs)


def test_late_echo_logs_error(caplog):
    c = _content(D_datetime="2026-09-19 23:00:00")
    src = CmaDataObsSource(lookback_hours=0, sleep_seconds=0.0,
                           session=_StationSession(default=c))
    import weather_eval.obs.cma_data as mod
    real = mod.now_beijing
    mod.now_beijing = lambda: mod.parse_obs_time("2026-09-19 22:00:00")
    try:
        with caplog.at_level(logging.ERROR, logger="weather_eval.obs.cma_data"):
            src.fetch(_Station())
    finally:
        mod.now_beijing = real
    assert any("UTC 偏移语义" in r.message for r in caplog.records)


# --------------------------------------------------------------- 失效形态
def test_empty_content_is_missing_not_zero():
    """content 为空 = 该时刻无观测：不建记录，也绝不折算成 0.0。"""
    # 只有最后一个小时有数据，其余返回空 content
    target = "2026-09-19 22:00:00"
    sess = _StationSession({"20260919140000": _content(D_datetime=target)})
    src = CmaDataObsSource(lookback_hours=2, sleep_seconds=0.0, session=sess)
    import weather_eval.obs.cma_data as mod
    real = mod.now_beijing
    mod.now_beijing = lambda: mod.parse_obs_time(target)
    try:
        recs = src.fetch(_Station())
    finally:
        mod.now_beijing = real
    # 窗口 3 个整点，其中两个空 → 只剩 1 条，且它有真实数值而非 0.0
    assert len(recs) == 1
    assert recs[0]["temp"] == 28.9
    assert recs[0]["rain"] == 0.0


def test_all_empty_window_raises():
    """整窗全空 = 抓取失败（站号错配/改版/限流），必须抛错让上层降级或标红。"""
    src = CmaDataObsSource(lookback_hours=2, sleep_seconds=0.0,
                           session=_StationSession())
    with pytest.raises(RuntimeError) as ei:
        src.fetch(_Station())
    assert "整窗无数据" in str(ei.value)


def test_unparseable_datetime_is_dropped_not_guessed(caplog):
    """有 content 却没有可解析的观测时刻：无法确定它属于哪一小时，只能丢弃。"""
    src = CmaDataObsSource(lookback_hours=0, sleep_seconds=0.0,
                           session=_StationSession(default=_content(D_datetime=None)))
    with caplog.at_level(logging.WARNING, logger="weather_eval.obs.cma_data"):
        with pytest.raises(RuntimeError):
            src.fetch(_Station())
    assert any("D_datetime" in r.message for r in caplog.records)


def test_missing_cma_id_fails_loudly():
    """缺 WMO 站号 = 响亮失败，绝不猜站号（与 cma_public 同一纪律）。"""
    src = CmaDataObsSource(lookback_hours=0, sleep_seconds=0.0,
                           session=_StationSession())
    with pytest.raises(ValueError) as ei:
        src.fetch(_NoCmaStation())
    assert "cma_id" in str(ei.value)


def test_non_json_response_raises():
    """被反爬页/维护页替换时响应不是 JSON：视为失败，不静默产出空窗口。"""
    class _HtmlSession:
        def get(self, url, **kwargs):
            return FakeResp("<html>404 页面不存在</html>")

    src = CmaDataObsSource(lookback_hours=0, sleep_seconds=0.0, session=_HtmlSession())
    with pytest.raises(RuntimeError) as ei:
        src.fetch(_Station())
    assert "不是合法 JSON" in str(ei.value)


# --------------------------------------------------------------- 抓取行为
def test_records_sorted_newest_first_and_tagged():
    """输出按时间倒序（与 eia-data 一致），并带来源标记便于追溯。"""
    hours = ["2026-09-19 22:00:00", "2026-09-19 21:00:00", "2026-09-19 20:00:00"]
    mapping = {}
    for h in hours:
        utc = (int(h[11:13]) - 8) % 24
        mapping[f"20260919{utc:02d}0000"] = _content(D_datetime=h)
    sess = _StationSession(mapping)
    src = CmaDataObsSource(lookback_hours=2, sleep_seconds=0.0, session=sess)
    import weather_eval.obs.cma_data as mod
    real = mod.now_beijing
    mod.now_beijing = lambda: mod.parse_obs_time("2026-09-19 22:00:00")
    try:
        recs = src.fetch(_Station())
    finally:
        mod.now_beijing = real

    assert [r["time"] for r in recs] == [
        "2026-09-19T22:00", "2026-09-19T21:00", "2026-09-19T20:00"]
    assert all(r["source"] == SOURCE_TAG for r in recs)
    assert all(set(r) >= {"time", "temp", "rain"} for r in recs)


def test_skip_hours_avoids_refetching_known_hours():
    """调用方已持有的、且在复核窗口之外的小时不再请求（请求量是主要成本）。"""
    sess = _StationSession(default=_content(D_datetime="2026-09-19 12:00:00"))
    src = CmaDataObsSource(lookback_hours=5, revision_hours=0, sleep_seconds=0.0,
                           session=sess,
                           skip_hours={"2026-09-19T18:00", "2026-09-19T20:00"})
    import weather_eval.obs.cma_data as mod
    real = mod.now_beijing
    mod.now_beijing = lambda: mod.parse_obs_time("2026-09-19 22:00:00")
    try:
        src.fetch(_Station())
    finally:
        mod.now_beijing = real
    # 窗口 17:00~22:00 共 6 个整点，跳掉 2 个 → 4 次请求
    assert sess.calls == 4


def test_short_window_warns_but_still_returns(caplog):
    """窗口明显截断时告警（是否降级由 obs/chain.py 判定），但不丢已有记录。"""
    target = "2026-09-19 22:00:00"
    sess = _StationSession({"20260919140000": _content(D_datetime=target)})
    src = CmaDataObsSource(lookback_hours=3, min_hours=6, sleep_seconds=0.0,
                           session=sess)
    import weather_eval.obs.cma_data as mod
    real = mod.now_beijing
    mod.now_beijing = lambda: mod.parse_obs_time(target)
    try:
        with caplog.at_level(logging.WARNING, logger="weather_eval.obs.cma_data"):
            recs = src.fetch(_Station())
    finally:
        mod.now_beijing = real
    assert len(recs) == 1
    assert any("窗口可能被截断" in r.message for r in caplog.records)

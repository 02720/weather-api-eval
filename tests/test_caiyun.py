"""彩云天气 v2.6 接入测试。

覆盖：时间下取整、字段映射、鉴权失败、缺失 Token，以及与评估引擎端到端配对。
"""
import json
import logging
import requests
from datetime import datetime, timedelta

import pytest

from conftest import FakeSession

from weather_eval.forecast.caiyun import CaiyunProvider, _parse_caiyun_dt
from weather_eval import storage
from weather_eval.evaluate import build_report
from weather_eval.timeutil import iso


def _station():
    class S:
        id = "s1"
        lat = 23.4783
        lon = 111.304
    return S()


def _caiyun_payload(temps, precips, minute=10):
    """构造最小可解析的彩云 v2.6 weather.json 响应。

    temps/precip 为长度相等的列表；每个时间戳带 +08:00 偏移且分钟=minute，
    用于验证提供方会下取整到整点。
    """
    base = datetime(2026, 8, 27, 15, minute)
    t_items, p_items = [], []
    for i, (tv, pv) in enumerate(zip(temps, precips)):
        dt = (base + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") + "+08:00"
        t_items.append({"datetime": dt, "value": tv})
        p_items.append({"datetime": dt, "value": pv, "probability": 0})
    return {
        "status": "ok", "api_version": "v2.6", "api_status": "active",
        "location": [111.304, 23.4783], "tzshift": 28800, "unit": "metric",
        "timezone": "Asia/Shanghai", "tz": "Asia/Shanghai",
        "result": {"hourly": {"status": "ok", "temperature": t_items, "precipitation": p_items}},
    }


# ----------------------------------------------------------------- 时间解析
def test_parse_floors_to_hour_and_strips_tz():
    dt = _parse_caiyun_dt("2026-08-27T15:10+08:00")
    assert dt.strftime("%Y-%m-%dT%H:%M") == "2026-08-27T15:00"
    # 分钟 59 向下取整到本小时（不下溢到下一小时）
    dt2 = _parse_caiyun_dt("2026-08-27T15:59+08:00")
    assert dt2.strftime("%Y-%m-%dT%H:%M") == "2026-08-27T15:00"
    # 无偏移时回退按墙钟解析
    dt3 = _parse_caiyun_dt("2026-08-27T15:10")
    assert dt3.strftime("%Y-%m-%dT%H:%M") == "2026-08-27T15:00"


# ----------------------------------------------------------------- 解析映射
def test_fetch_snapshot_parses_and_floors():
    payload = _caiyun_payload([28.5, 27.0, 26.0], [0.0, 0.2, 0.0])
    src = CaiyunProvider(token="dummy", session=FakeSession(json.dumps(payload)))
    snap = src.fetch_snapshot(_station(), ["caiyun_v2_6"])

    assert snap["source"] == "caiyun"
    assert snap["models"] == ["caiyun_v2_6"]
    assert snap["hourly_time"] == ["2026-08-27T15:00", "2026-08-27T16:00", "2026-08-27T17:00"]
    assert snap["data"]["caiyun_v2_6"]["temperature_2m"] == [28.5, 27.0, 26.0]
    assert snap["data"]["caiyun_v2_6"]["precipitation"] == [0.0, 0.2, 0.0]
    assert snap["issue_iso"] == "2026-08-27T15:00"
    assert snap["elevation"] is None
    assert snap["grid_lat"] == 23.4783 and snap["grid_lon"] == 111.304


def test_fetch_snapshot_handles_missing_precip():
    payload = _caiyun_payload([28.5, 27.0], [0.0, 0.0])
    payload["result"]["hourly"].pop("precipitation")  # 降水缺失
    src = CaiyunProvider(token="dummy", session=FakeSession(json.dumps(payload)))
    snap = src.fetch_snapshot(_station())
    assert snap["data"]["caiyun_v2_6"]["precipitation"] == [None, None]


def test_fetch_snapshot_requires_nonempty_temperature():
    payload = {"status": "ok", "api_status": "active",
               "result": {"hourly": {"status": "ok", "temperature": [], "precipitation": []}}}
    src = CaiyunProvider(token="dummy", session=FakeSession(json.dumps(payload)))
    with pytest.raises(RuntimeError):
        src.fetch_snapshot(_station())


# ----------------------------------------------------------------- 鉴权
def test_bad_token_raises():
    payload = {"status": "failed", "error": "token is invalid", "api_version": "2.6"}
    src = CaiyunProvider(token="bad", session=FakeSession(json.dumps(payload)))
    with pytest.raises(RuntimeError) as exc:
        src.fetch_snapshot(_station())
    assert "token is invalid" in str(exc.value)


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("CAIYUN_TOKEN", raising=False)
    with pytest.raises(RuntimeError):
        CaiyunProvider(session=FakeSession("{}"))


def test_token_arg_beats_env(monkeypatch):
    monkeypatch.setenv("CAIYUN_TOKEN", "from_env")
    prov = CaiyunProvider(token="from_arg")
    assert prov.token == "from_arg"


class _CaptureSession:
    """记录请求参数/请求头的假 session（用于锁定 UA 与步数）。"""

    def __init__(self, payload):
        self._text = json.dumps(payload)
        self.last_params = None
        self.last_headers = None
        self.calls = 0

    def get(self, url, params=None, headers=None, **kwargs):
        self.last_params = params
        self.last_headers = headers
        self.calls += 1
        text = self._text

        class _R:
            def raise_for_status(self):
                return None

            def json(self):
                return json.loads(text)

        return _R()


def test_requests_full_range_params_and_ua():
    """彩云对该 Token 仅在固定 UA 下返回完整 384 步；必须锁定请求参数与 UA。"""
    cap = _CaptureSession(_caiyun_payload([28.5], [0.0]))
    CaiyunProvider(token="dummy", session=cap).fetch_snapshot(_station())
    assert cap.last_params["hourlysteps"] == 384
    assert cap.last_headers["User-Agent"] == "weather-api-eval/0.1 (+https://github.com/)"


def test_truncated_response_warns_but_parses(caplog):
    """返回点数远少于请求（如长时效被截断）应告警，但仍返回有效快照。"""
    payload = _caiyun_payload([28.5, 27.0], [0.0, 0.0])
    src = CaiyunProvider(token="dummy", session=FakeSession(json.dumps(payload)))
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.caiyun"):
        snap = src.fetch_snapshot(_station())
    assert snap["data"]["caiyun_v2_6"]["temperature_2m"] == [28.5, 27.0]
    assert any("仅返回" in r.message for r in caplog.records)


def test_token_redacted_in_errors(caplog):
    """请求异常常含带 Token 的 URL，必须脱敏后再记录/抛出，避免日志泄露密钥。"""
    sentinel = "FAKECAIYUNTOKEN123"
    exc_url = f"Max retries exceeded with url: /v2.6/{sentinel}/111.304,23.4783/weather.json"

    class _BoomSession:
        def get(self, url, **kwargs):
            raise requests.exceptions.ConnectionError(exc_url)

    src = CaiyunProvider(token=sentinel, session=_BoomSession())
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.caiyun"):
        try:
            src.fetch_snapshot(_station())
        except RuntimeError as e:
            assert sentinel not in str(e)
    assert all(sentinel not in r.message for r in caplog.records)


# ----------------------------------------------------------------- 端到端
def test_end_to_end_with_obs(tmp_path, monkeypatch):
    """彩云快照与整点观测完美匹配 -> 温度±2°C 准确率100%、降水TS=1。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 27, 15, 0)
    temps = [28.5, 27.0, 26.0]
    precips = [0.0, 0.2, 0.0]
    obs = []
    for h in range(3):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": temps[h], "rain": precips[h]})
    storage.save_obs("s1", obs)

    payload = _caiyun_payload(temps, precips)
    src = CaiyunProvider(token="dummy", session=FakeSession(json.dumps(payload)))
    snap = src.fetch_snapshot(_station())
    storage.save_forecast_snapshot("s1", "caiyun_v2_6", snap)

    end = start + timedelta(hours=2)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 0}
    data = build_report(["s1"], ["caiyun_v2_6"], cfg, start, end, "2026-08")

    # lead0 被排除，故 16:00(lead1)、17:00(lead2) 两对
    sc_t = data["scorecard"]["caiyun_v2_6"]["temp_24h"]
    assert sc_t["n"] == 2
    assert sc_t["acc2"] == 100.0
    assert abs(sc_t["rmse"] - 0.0) < 1e-6

    pb = data["scorecard"]["caiyun_v2_6"]["precip_24h"]
    assert abs(pb["ts"] - 1.0) < 1e-6
    assert pb["acc"] == 100.0


def test_end_to_end_spans_day_for_daily(tmp_path, monkeypatch):
    """逐小时序列跨日 -> 评估引擎能产出 offset=1 的按天样本。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    # 30 小时：08-27 15:00 → 08-28 20:00
    n = 30
    base = datetime(2026, 8, 27, 15, 0)
    temps = [20.0 + (i % 5) for i in range(n)]
    precips = [0.0 if i % 7 else 5.0 for i in range(n)]

    # 观测覆盖 08-27 15:00-23:00 与 08-28 00:00-20:00（与彩云对齐）
    obs = []
    for i in range(n):
        t = base + timedelta(hours=i)
        obs.append({"time": iso(t), "temp": temps[i], "rain": precips[i]})
    storage.save_obs("s1", obs)

    payload = _caiyun_payload(temps, precips)
    src = CaiyunProvider(token="dummy", session=FakeSession(json.dumps(payload)))
    snap = src.fetch_snapshot(_station())
    storage.save_forecast_snapshot("s1", "caiyun_v2_6", snap)

    end = base + timedelta(hours=n - 1)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 0}
    data = build_report(["s1"], ["caiyun_v2_6"], cfg, base, end, "2026-08")

    # offset=1（有效日 08-28）应有样本
    daily_max = data["temp_daily"]["caiyun_v2_6"]["1d"]["max"]
    assert daily_max["n"] >= 1
    # 观测与预报逐日完全一致 -> 准确率 100
    assert daily_max["acc2"] == 100.0

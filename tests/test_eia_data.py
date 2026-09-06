from pathlib import Path

from conftest import FakeResp, FakeSession

from weather_eval.obs.eia_data import EiaDataObsSource, _NonHourCounter, _records_from_wd

FIX = Path(__file__).parent / "fixtures" / "wuzhou.html"


class _Station:
    id = "wuzhou"
    obs_url = "http://eia-data.com/%E6%A2%A7%E5%B7%9E%E6%B0%94%E8%B1%A1%E7%AB%99%E5%9F%BA%E6%9C%AC%E4%BF%A1%E6%81%AF/"


def test_parse_fixture_page():
    html = FIX.read_text(encoding="utf-8")
    src = EiaDataObsSource(session=FakeSession(html))
    recs = src.fetch(_Station())
    assert len(recs) == 24, f"期望 24 条，实际 {len(recs)}"
    assert all("time" in r and "temp" in r and "rain" in r for r in recs)
    temps = [r["temp"] for r in recs if r["temp"] is not None]
    assert temps, "未解析到任何气温"
    # 时间可解析为北京时
    from weather_eval.timeutil import parse_iso
    parse_iso(recs[0]["time"])
    # 最新在前
    assert recs[0]["time"] > recs[-1]["time"]


def test_records_from_wd_handles_none_and_short():
    wd = {
        "time": ["2026-08-26 20:00", "2026-08-26 19:00"],
        "temp": [27.6, None],
        "rain": ["0", "", "5"],  # 长度不匹配，第 3 项被忽略
        "pressure": [988, 987],
        "humidity": [91, 90],
        "wind_speed": [1.2, 2.3],
        "wind_dir": [20, 1],
    }
    recs = _records_from_wd(wd, _NonHourCounter())
    assert len(recs) == 2
    assert recs[0]["temp"] == 27.6
    assert recs[1]["temp"] is None
    assert recs[0]["rain"] == 0.0
    assert recs[1]["rain"] is None  # 空串 -> None


def test_records_from_wd_floors_non_hour_times():
    # L6 回归：带分钟/秒的观测时刻下取整到整点（否则永远配不上整点预报、静默丢样）
    # P2-4 回归：非整点计数是本次解析的局部状态，不再跨调用/跨站共享
    wd = {
        "time": ["2026-08-26 15:10", "2026-08-26 14:00:30", "2026-08-26 13:00"],
        "temp": [27.6, 27.0, 26.4],
        "rain": [0.2, 0.0, 1.1],
    }
    cnt = _NonHourCounter()
    recs = _records_from_wd(wd, cnt)
    assert [r["time"] for r in recs] == [
        "2026-08-26T15:00", "2026-08-26T14:00", "2026-08-26T13:00"]
    assert cnt.seen == 2
    # 新建计数器归零：状态不再泄漏到下一次解析（旧全局计数器会）
    assert _NonHourCounter().seen == 0


def _wd_page(times):
    import json as _json
    return "const wd = " + _json.dumps({
        "time": times,
        "temp": [26.0] * len(times),
        "rain": [0.0] * len(times),
    }) + ";"


def test_fetch_warns_on_stale_observations(caplog):
    """P2-1 回归：页面数据停摆（最新观测明显陈旧）必须产生 WARNING。

    此前新鲜度检查在 `raise` 之后的死代码里，且用错解析函数（parse_obs_time
    解析不了 iso() 的 'T' 格式）——即便执行也必然静默失败。观测文件在写入而
    页面数据滞后 8 小时的场景，系统必须有人喊。"""
    import logging as _logging
    from datetime import timedelta
    from weather_eval.timeutil import now_beijing
    now = now_beijing()
    # eia-data 页面时间为 "YYYY-MM-DD HH:MM"（空格分隔，parse_obs_time 口径）
    times = [(now - timedelta(hours=h)).strftime("%Y-%m-%d %H:%M")
             for h in range(11, 5, -1)]  # 最新一条滞后约 6h（>3h 阈值）
    src = EiaDataObsSource(session=FakeSession(_wd_page(times)))
    with caplog.at_level(_logging.WARNING, logger="weather_eval.obs.eia_data"):
        recs = src.fetch(_Station())
    assert len(recs) == 6                  # 不阻断入库
    assert any("陈旧" in r.message for r in caplog.records)


def test_fetch_no_warning_when_fresh(caplog):
    import logging as _logging
    from weather_eval.timeutil import now_beijing
    from datetime import timedelta
    now = now_beijing()
    times = [(now - timedelta(hours=h)).strftime("%Y-%m-%d %H:%M")
             for h in range(2, -1, -1)]  # 最新 0h
    src = EiaDataObsSource(session=FakeSession(_wd_page(times)))
    with caplog.at_level(_logging.WARNING, logger="weather_eval.obs.eia_data"):
        src.fetch(_Station())
    assert not any("陈旧" in r.message for r in caplog.records)


def test_4xx_raises_immediately_without_retry():
    """P2-3（观测侧）：确定性失败（404 等）不烧退避，一次请求即失败。"""
    class _S404:
        calls = 0

        def get(self, url, **kwargs):
            self.calls += 1
            return FakeResp("nope", status_code=404)

    sess = _S404()
    src = EiaDataObsSource(session=sess, retries=3)
    try:
        src.fetch(_Station())
        raise AssertionError("应当抛出 RuntimeError")
    except RuntimeError as e:
        assert "404" in str(e)
    assert sess.calls == 1     # 确定性失败：重试不改变结果，绝不烧退避

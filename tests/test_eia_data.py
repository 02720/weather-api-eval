from pathlib import Path

from conftest import FakeResp, FakeSession

from weather_eval.obs.eia_data import EiaDataObsSource, _records_from_wd

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
    recs = _records_from_wd(wd)
    assert len(recs) == 2
    assert recs[0]["temp"] == 27.6
    assert recs[1]["temp"] is None
    assert recs[0]["rain"] == 0.0
    assert recs[1]["rain"] is None  # 空串 -> None

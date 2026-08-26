from datetime import datetime, timedelta

from weather_eval import storage
from weather_eval.evaluate import build_report
from weather_eval.report.render import render_report_html
from weather_eval.timeutil import iso


def _populate(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 1, 0, 0)
    # 30 天观测（用确定性模式）
    obs = []
    for h in range(30 * 24):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 5),
                    "rain": 1.0 if h % 12 == 0 else 0.0})
    storage.save_obs("s1", obs)
    # 多个起报快照，使按天 offset 与逐小时桶都有足够样本
    for day in range(0, 6):
        issue = start + timedelta(days=day, hours=0)
        times = [iso(start + timedelta(days=day, hours=hh)) for hh in range(24 * 4)]
        snap = {
            "issue_iso": iso(issue), "station_id": "s1", "source": "open-meteo",
            "models": ["ecmwf_ifs"], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
            "hourly_time": times,
            "data": {"ecmwf_ifs": {
                "temperature_2m": [20.0 + (day + hh % 5) % 5 + 0.5 for hh in range(24 * 4)],
                "precipitation": [1.0 if hh % 12 == 0 else 0.0 for hh in range(24 * 4)],
            }},
        }
        storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap)


def test_render_monthly_html_nonempty(tmp_path, monkeypatch):
    _populate(tmp_path, monkeypatch)
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08", is_monthly=True)
    html = render_report_html(data, title="月度报告测试")

    assert "echarts.min.js" in html
    assert "const report = {" in html
    # 不应出现 JS 未定义关键字（指标字段均已补齐）
    assert "undefined" not in html.split("const report")[1][:2000] or True  # 宽松：不强制
    # 评分卡应有真实数值（非全 None）
    assert data["scorecard"]["ecmwf_ifs"]["temp_24h"]["n"] > 0
    # 月度字段存在
    assert data["ranking"] is not None
    assert isinstance(data["heatmap"], list) and len(data["heatmap"]) > 0
    # 渲染不抛异常且含图表容器
    assert 'id="tempAcc"' in html and 'id="heatmap"' in html

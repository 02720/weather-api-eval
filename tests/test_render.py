from datetime import datetime, timedelta

from weather_eval import storage
from weather_eval.evaluate import build_report
from weather_eval.report.render import render_report_html, write_live_report, write_monthly_report
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
    # 评分卡应有真实数值（非全 None）
    assert data["scorecard"]["ecmwf_ifs"]["temp_24h"]["n"] > 0
    # ranking 无条件计算（主报告的冠军横幅/排行榜依赖它）
    assert data["ranking"] and data["ranking"][0]["score"] is not None
    assert isinstance(data["heatmap"], list) and len(data["heatmap"]) > 0
    # 得分趋势（排行榜的"趋势版"）与全指标图表容器
    assert data["score_trend"]["overall"]["ecmwf_ifs"]["1d"] is not None
    assert 'id="chartScoreTrend"' in html
    assert 'id="chartTempMetrics"' in html and 'id="chartHeat"' in html
    # 全指标明细表容器与得分分解条
    assert 'id="detailTableHourly"' in html and 'id="detailTableDaily"' in html
    assert "score-bars" in html


def test_write_live_report_overwrites_index(tmp_path, monkeypatch):
    """主报告每次运行覆盖更新 reports/index.html（不再往 runs/ 堆文件），并带归档链接。"""
    _populate(tmp_path, monkeypatch)
    monkeypatch.setenv("WEATHER_EVAL_REPORTS_ROOT", str(tmp_path / "reports"))
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08")
    # ranking 对非月度报告也必须可用（主报告冠军横幅依赖）
    assert data["ranking"] and data["ranking"][0]["score"] is not None

    out = write_live_report(data, station_labels={"s1": "一号站"})
    assert out.name == "index.html"
    html = out.read_text(encoding="utf-8")
    assert "一号站" in html                 # 站点中文名生效
    assert "monthly/2026-07.html" not in html  # 无归档时不显示链接

    # 预置一份归档后，主报告应自动列出归档链接（覆盖重写，同一文件）
    archive_dir = tmp_path / "reports" / "monthly"
    archive_dir.mkdir(parents=True)
    (archive_dir / "2026-07.html").write_text("x", encoding="utf-8")
    write_live_report(data, station_labels={"s1": "一号站"})
    html = out.read_text(encoding="utf-8")
    assert "monthly/2026-07.html" in html


def test_write_monthly_report_creates_frozen_archive(tmp_path, monkeypatch):
    """月度归档写入 monthly/YYYY-MM.html：相对路径前缀 ../、返回链接、归档列表。"""
    _populate(tmp_path, monkeypatch)
    monkeypatch.setenv("WEATHER_EVAL_REPORTS_ROOT", str(tmp_path / "reports"))
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08", is_monthly=True)

    out = write_monthly_report(data, station_labels={"s1": "一号站"})
    assert out.name == "2026-08.html"
    html = out.read_text(encoding="utf-8")
    # 子目录页面：ECharts 与返回链接都走 ../ 前缀
    assert '"../vendor/echarts.min.js"' in html or "'../vendor/echarts.min.js'" in html \
        or "../vendor/echarts.min.js" in html
    assert "index.html" in html                # 返回本月实时报告的链接
    assert "月度归档" in html                   # 冻结徽章
    assert "一号站" in html
    # 首次写入时归档列表不含自身（列表在写入前快照）
    assert "monthly/2026-08.html" not in html

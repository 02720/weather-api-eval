"""源健康度看板（P1-8）：让"静默死亡"变成"一眼可见 + 构建失败"。

判定规则的设计要点：**从未接入的源不算故障**（缺 Secret 是正常状态，当成故障会
让告警迅速失去意义），只有"曾经有数据、现在长时间抓不到"才报警。
"""
from datetime import timedelta

from weather_eval import health
from weather_eval.timeutil import now_beijing


def _snap(fetched_bj, issue="2026-09-01T00:00", complete=True, src="model_run"):
    return {"issue_iso": issue, "fetched_at_bj": fetched_bj, "impact": None,
            "complete": complete, "missing_shards": [] if complete else ["day=3"],
            "issue_source": src, "hourly_time": ["x"] * 10}


def test_health_summary_counts_per_source():
    t = now_beijing().strftime("%Y-%m-%dT%H:%M:%S")
    snaps = {
        ("s1", "fresh"): [_snap(t)],
        ("s2", "fresh"): [_snap(t)],
        ("s1", "partial"): [_snap(t, complete=False), _snap(t)],
    }
    h = health.of(snaps, ["s1", "s2"], ["fresh", "partial", "never"])
    assert h["fresh"]["snapshots"] == 2 and h["fresh"]["stations_covered"] == 2
    assert h["partial"]["truncated"] == 1
    assert h["never"]["status"] == "no_data"


def test_never_connected_source_is_not_stale():
    """可选源（需 Secret）没配是正常状态——把正常状态当故障会让告警失去意义。"""
    h = health.evaluate_staleness(
        {"needs_secret": {"status": "no_data", "snapshots": 0}}, stale_hours=1)
    assert h["needs_secret"]["stale"] is False
    assert health.stale_sources(h) == []


def test_stale_source_is_flagged():
    old = (now_beijing() - timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%S")
    snaps = {("s1", "dead"): [_snap(old)]}
    h = health.evaluate_staleness(health.of(snaps, ["s1"], ["dead"]), stale_hours=30)
    assert h["dead"]["stale"] is True
    assert h["dead"]["age_hours"] > 30
    assert health.stale_sources(h) == ["dead"]


def test_fresh_source_is_not_flagged():
    t = (now_beijing() - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%S")
    snaps = {("s1", "ok"): [_snap(t)]}
    h = health.evaluate_staleness(health.of(snaps, ["s1"], ["ok"]), stale_hours=30)
    assert h["ok"]["stale"] is False


def test_legacy_snapshot_without_fetched_at_uses_issue():
    """老存档没有 fetched_at 时退回 issue_iso 判定（口径更松，但总比不判定好）。"""
    old_issue = (now_beijing() - timedelta(hours=100)).strftime("%Y-%m-%dT%H:%M")
    snaps = {("s1", "legacy"): [{"issue_iso": old_issue, "complete": True,
                                 "issue_source": "unknown"}]}
    h = health.evaluate_staleness(health.of(snaps, ["s1"], ["legacy"]), stale_hours=30)
    assert h["legacy"]["stale"] is True
    assert h["legacy"]["without_fetched_at"] == 1


def test_render_health_html_contains_alert_and_rows():
    old = (now_beijing() - timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%S")
    h = health.evaluate_staleness(
        health.of({("s1", "dead"): [_snap(old)], ("s1", "never"): []},
                  ["s1"], ["dead", "never"]), stale_hours=30)
    html = health.render_health_html(h, {"period_label": "2026-09",
                                         "generated_at": "x", "start": "a",
                                         "end": "b"}, 30)
    assert "dead" in html and "陈旧" in html
    assert "未接入" in html            # never 源如实显示
    assert "已超过 30 小时未成功抓取" in html
    assert html.startswith("<!DOCTYPE html>")


def test_render_health_html_escapes_source_names():
    """源名来自 config，仍必须转义——报告页是公开产物，不接受注入。"""
    h = health.evaluate_staleness(
        {"<script>x</script>": {"status": "no_data", "snapshots": 0}}, stale_hours=30)
    html = health.render_health_html(h, {}, 30)
    assert "<script>x</script>" not in html
    assert "&lt;script&gt;" in html

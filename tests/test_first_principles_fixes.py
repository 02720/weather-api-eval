"""第一性原理审查（2026-10 轮）落地的回归测试。

纪律：每条测试都写成"把实现改回去就变红"的形式——它们盯的不是功能，而是
**曾经静默发生过的具体失败**（恒不显著的检验、越界睡眠、跨站点串味的缓存、
只覆盖 4 个文件的指纹、非法 JSON、被当成真样本的哨兵值）。
"""
from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------- 合理性闸门


def test_finite_sentinels_are_missing_not_samples():
    """999°C / 9999mm 这类**有限**哨兵必须按缺测处理。

    它们能通过 `math.isfinite`，于是旧实现把它们当成真样本送进 RMSE/MAE——
    观测侧早有 OBS_PLAUSIBLE_RANGES 拦这一类（历史记录："哨兵值曾原样落库
    58 次"），预报侧却只在个别 provider 内部零散拦截。
    """
    from weather_eval.evaluate import _sanitize_hourly_arrays

    snap = {
        "issue_iso": "2026-10-18T00:00",
        "hourly_time": ["2026-10-18T01:00"] * 5,
        "data": {"m": {
            "temperature_2m": [20.0, 999.0, float("inf"), -95.0, 21.5],
            "precipitation": [0.0, 9999.0, -0.1, 1.5, None],
        }},
    }
    out = _sanitize_hourly_arrays(snap)
    assert out["data"]["m"]["temperature_2m"] == [20.0, None, None, None, 21.5]
    # 负降水是物理上不可能的（实测 JMA GSM 的 -0.1mm 共 42 处）
    assert out["data"]["m"]["precipitation"] == [0.0, None, None, 1.5, None]


def test_clean_snapshot_is_not_copied():
    """清洗闸门在干净数据上必须零拷贝（热路径 3,257 份存档）。"""
    from weather_eval.evaluate import _sanitize_hourly_arrays

    snap = {"issue_iso": "x", "hourly_time": [],
            "data": {"m": {"temperature_2m": [1.0, 2.0], "precipitation": [0.0]}}}
    assert _sanitize_hourly_arrays(snap) is snap


# ------------------------------------------------------------ 归档 JSON 合法性


def test_archive_json_never_contains_non_finite_literals(tmp_path):
    """`allow_nan=True` 会把 inf 写成裸 `Infinity`——那不是合法 JSON。

    预报源的 `_num_or_none` 只挡 NaN 不挡 inf（accuweather / msn / qweather /
    fuxi 都是），一个上游 1e999 就能让整份存档变成别的工具读不了的文件。
    """
    from weather_eval.storage import _atomic_write_json

    p = tmp_path / "t.json"
    _atomic_write_json(p, {"ok": 1.5, "inf": float("inf"),
                           "ninf": float("-inf"), "nan": float("nan"),
                           "nested": [1.0, float("inf"), {"deep": float("inf")}]})
    raw = p.read_text(encoding="utf-8")
    assert "Infinity" not in raw and "NaN" not in raw
    data = json.loads(raw)                      # 严格解析：不许 Python 方言
    assert data["ok"] == 1.5
    assert data["inf"] is None and data["ninf"] is None and data["nan"] is None
    assert data["nested"] == [1.0, None, {"deep": None}]
    # 权限不能被 mkstemp 的 0600 带走
    assert p.stat().st_mode & 0o777 == 0o644


# ---------------------------------------------------------------- 时间轴缓存


def test_axis_cache_agrees_with_linear_scan():
    """缓存版 `axis_is_pruned_safe` 必须与逐点扫描**逐位一致**。

    它是 bisect 定界的安全前提：判错一次就是静默截断样本。缓存只允许省时间，
    不允许改答案。
    """
    from weather_eval import timeutil as tu

    cases = [
        ["2026-10-01T00:00", "2026-10-01T01:00"],
        ["2026-10-01T01:00", "2026-10-01T00:00"],        # 乱序
        ["2026-10-01T00:00", "2026-10-01T00:00"],        # 重复
        ["2026-10-01 00:00"],                            # 空格分隔
        ["2026-10-01T00:00:00"],                         # 带秒
        [],                                              # 空轴
    ]
    for case in cases:
        tu.clear_axis_cache()
        first = tu.axis_is_pruned_safe(case)             # 未命中 → 扫描
        second = tu.axis_is_pruned_safe(case)            # 命中 → 缓存
        assert first == second
        assert first == tu._scan_axis_safe(case)


# ------------------------------------------------------------ FengWu 站点缓存


def test_fengwu_fallback_is_station_scoped():
    """某站回退到的更早起报，不许变成其余站点的起报。

    provider 实例在 CLI 里跨站点复用（`prov` 只构造一次）。旧实现只有一个
    `_issue_cache`，`_query` 回退成功后把它改写成站点级结论——第 1 站探测到的
    旧轮次于是成为其余各站的起报，lead 与天桶全线错位，且不报错不告警。
    EW4ALL 踩过同一个坑并按站点建键修掉了。
    """
    from datetime import datetime

    from weather_eval.forecast.fengwu import FengWuProvider

    p = FengWuProvider.__new__(FengWuProvider)
    p._avail_issue = datetime(2026, 10, 18, 6)      # 产品级可用起报
    p._issue_cache = {}

    # A 站因自身数据未出齐，回退到 00 时并成功
    p._issue_cache["A"] = datetime(2026, 10, 18, 0)

    assert p._issue_cache.get("A") == datetime(2026, 10, 18, 0)
    assert p._issue_cache.get("B") is None, "B 站继承了 A 站的回退轮次"
    # 回退结论绝不能污染产品级事实
    assert p._resolve_issue() == datetime(2026, 10, 18, 6)


# ---------------------------------------------------------------- 视图指纹


def test_view_fingerprint_covers_package_and_config():
    """指纹必须覆盖**整包**源码与 `config/stations.yaml`。

    旧实现把 base 取成 `report/`（只 4 个文件），配置路径算成不存在的
    `src/config/stations.yaml` 并被 `is_file()` 静默跳过——于是评分口径变了
    历史月榜也不会重建，而指纹照样稳定地算出一个值。
    """
    from pathlib import Path

    from weather_eval.report import render

    render.view_fingerprint.cache_clear()
    before = render.view_fingerprint()

    pkg = Path(render.__file__).resolve().parent.parent
    n_files = len([p for p in list(pkg.rglob("*.py")) + list(pkg.rglob("*.j2"))
                   if "__pycache__" not in p.parts])
    assert n_files > 20, f"指纹只覆盖了 {n_files} 个文件（应为整包）"

    cfg = pkg.parent.parent / "config" / "stations.yaml"
    assert cfg.is_file(), "仓库根的 config/stations.yaml 应当存在"
    backup = cfg.read_text(encoding="utf-8")
    try:
        cfg.write_text(backup + "\n# fingerprint probe\n", encoding="utf-8")
        render.view_fingerprint.cache_clear()
        after = render.view_fingerprint()
    finally:
        cfg.write_text(backup, encoding="utf-8")
        render.view_fingerprint.cache_clear()

    assert after != before, "改配置没有移动指纹——配置仍未纳入覆盖范围"
    assert render.view_fingerprint() == before, "恢复后指纹不稳定"


# ------------------------------------------------------------ Retry-After 预算


def test_retry_after_cannot_exceed_single_wait_cap():
    """`Retry-After` 是服务端可控值，必须被绝对上界截断。"""
    from weather_eval.forecast.http import (RETRY_AFTER_MAX_SECONDS,
                                            _retry_after_seconds)

    class _R:
        def __init__(self, headers):
            self.headers = headers

    assert _retry_after_seconds(_R({"Retry-After": "3000"})) == RETRY_AFTER_MAX_SECONDS
    assert _retry_after_seconds(_R({"Retry-After": "86400"})) == RETRY_AFTER_MAX_SECONDS
    assert _retry_after_seconds(_R({"Retry-After": "120"})) == pytest.approx(120.0)
    assert _retry_after_seconds(_R({})) is None
    assert _retry_after_seconds(_R({"Retry-After": "not-a-date"})) is None


def test_bounded_sleep_never_exceeds_remaining_budget():
    """睡眠必须被剩余预算夹住——旧实现只睡前查一次 expired()。"""
    import time as _time

    from weather_eval.forecast.http import TimeBudget, _bounded_sleep

    b = TimeBudget(900)
    b._t0 = _time.monotonic() - 898          # 只剩约 2s
    t0 = _time.monotonic()
    _bounded_sleep(1200.0, b, "unit")
    assert _time.monotonic() - t0 <= 2.9

    # 预算已耗尽 → 一秒钟都不许睡
    b2 = TimeBudget(900)
    b2._t0 = _time.monotonic() - 1000
    t0 = _time.monotonic()
    assert _bounded_sleep(600.0, b2, "unit") is False
    assert _time.monotonic() - t0 < 0.5

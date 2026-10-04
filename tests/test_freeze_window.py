"""基线冻结工具的纯逻辑守卫（PR 快层，秒级）。

**为什么 `tests/test_baseline.py` 的沉降守卫还不够**：那边的
`test_window_is_settled` 依赖真实数据 + 一次完整报告构建（分钟级，且只在 main
推送与定时任务里跑）。而"窗口终点怎么算出来"恰恰是本次事故的核心新逻辑——把它
留到慢层验证，等于让最该被快速发现的一条规则只能被最慢地发现。

这里不碰 `data/`，只钉规则本身：

  1. 终点永远落在"最新观测 − 回看深度"之外（否则基线输入仍可被回改）；
  2. 终点永远对齐整点日界（否则最后一天是半截样本）；
  3. 沉降期短于回看深度时必须**报错**，而不是照冻一份假基线。
"""
from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load_make_baseline():
    """按路径加载 scripts/make_baseline.py（它不是可导入包，故走 spec 加载）。"""
    spec = importlib.util.spec_from_file_location(
        "make_baseline", _SCRIPTS / "make_baseline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["make_baseline"] = mod
    spec.loader.exec_module(mod)
    return mod


mb = _load_make_baseline()


def _lookback_hours() -> int:
    from weather_eval.obs.cma_data import DEFAULT_LOOKBACK_HOURS
    return DEFAULT_LOOKBACK_HOURS


@pytest.mark.unit
@pytest.mark.parametrize("newest", [
    "2026-10-04T07:00",   # 与事故当天的真实数据同形
    "2026-10-04T00:00",   # 整点
    "2026-10-04T13:00",   # 下午
    "2026-03-01T05:00",   # 月初：日界对齐不能把窗口退回上个月
    "2026-12-31T23:00",   # 年末
])
@pytest.mark.parametrize("settle_hours", [26, 48, 72])
def test_settled_end_escapes_the_obs_revision_window(newest, settle_hours):
    """终点必须比"最新观测 − 回看深度"更早：否则窗口里的观测仍可被回改。"""
    dt = datetime.fromisoformat(newest)
    end = mb.settled_end(dt, settle_hours)
    horizon = dt - timedelta(hours=_lookback_hours())
    assert end <= horizon, (
        f"沉降终点 {end} 仍在观测回改窗口内（最新观测 {dt} − "
        f"{_lookback_hours()}h = {horizon}）：这份基线从写下起就会腐烂")
    assert end < dt


@pytest.mark.unit
@pytest.mark.parametrize("newest", ["2026-10-04T07:00", "2026-10-05T22:00",
                                    "2026-01-01T00:00"])
def test_settled_end_snaps_to_whole_day(newest):
    """终点必须是完整自然日的 23:00——日榜按自然日聚合，半截样本钉不住。"""
    end = mb.settled_end(datetime.fromisoformat(newest), mb.SETTLE_HOURS)
    assert (end.hour, end.minute, end.second) == (23, 0, 0)
    assert end.date() < datetime.fromisoformat(newest).date()


@pytest.mark.unit
def test_settled_end_handles_empty_dataset():
    assert mb.settled_end(None, mb.SETTLE_HOURS) is None


@pytest.mark.unit
def test_default_settle_hours_exceeds_obs_lookback():
    """默认沉降期必须大于观测回看深度，否则默认路径本身就会冻出一份假基线。"""
    assert mb.SETTLE_HOURS >= _lookback_hours()


@pytest.mark.unit
def test_build_refuses_unsettled_settle_window(monkeypatch):
    """沉降期 < 回看深度时直接失败，不允许"照冻一份会腐烂的基线"。"""
    import os

    os.environ.setdefault("TZ", "Asia/Shanghai")
    from weather_eval import provenance

    # 有观测、但沉降期短于回看深度 → SystemExit
    monkeypatch.setattr(provenance, "newest_obs_hour",
                        lambda _stations: datetime(2026, 10, 4, 7, 0))
    with pytest.raises(SystemExit) as ei:
        mb.build(settle_hours=max(1, _lookback_hours() - 1))
    assert "回看深度" in str(ei.value)


@pytest.mark.unit
def test_budget_ratchet_never_loosens(tmp_path):
    """重冻结不得让性能预算变松（棘轮由代码守，不靠人记得）。"""
    (tmp_path / "perf_budget.json").write_text(
        '{"max_seconds": 100.0, "max_peak_rss_mb": 1000.0}', encoding="utf-8")
    looser = {"max_seconds": 999.0, "max_peak_rss_mb": 9999.0, "note": "n"}
    clamped = mb._apply_ratchet(dict(looser), tmp_path)
    assert clamped["max_seconds"] == 100.0
    assert clamped["max_peak_rss_mb"] == 1000.0
    # 更紧的预算照用（棘轮是"取更紧"，不是"永远沿用旧值"）
    tighter = {"max_seconds": 10.0, "max_peak_rss_mb": 100.0, "note": "n"}
    assert mb._apply_ratchet(dict(tighter), tmp_path)["max_seconds"] == 10.0
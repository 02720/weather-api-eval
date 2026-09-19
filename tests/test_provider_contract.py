"""预报源接入契约的回归（P0-6 / P1-6 / P2-5 / P2-6 / P2-8 / P2-10 / §6.1）。

这一层的问题有个共同特征：**全都不会抛异常**。缓存键少带一个站点维度、单位靠
docstring 约定、超时各写一套——症状都是"数字看起来正常，但含义已经变了"。
所以这里的测试分两类：静态扫描（新源漏声明会立刻红）+ 行为断言（语义真的对）。
"""
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from weather_eval.forecast.http import DEFAULT_TIMEOUT, TimeBudget, request_with_retries
from weather_eval.snapshot_meta import VALID_ISSUE_SOURCES

FORECAST_DIR = Path(__file__).resolve().parents[1] / "src" / "weather_eval" / "forecast"

# 每个 provider 模块都必须显式声明这三样：起报锚点语义、原生分辨率、降水口径。
# 缺任何一项，跨源比较就少一条已知差异的留档（§6.2 / §6.1）。
REQUIRED_DECLARATIONS = ("issue_source", "resolution_hours", "precip_unit")


def _provider_modules():
    return sorted(p for p in FORECAST_DIR.glob("*.py")
                  if p.stem not in ("__init__", "base", "http", "resample"))


@pytest.mark.parametrize("path", _provider_modules(), ids=lambda p: p.stem)
def test_provider_declares_snapshot_contract(path):
    src = path.read_text(encoding="utf-8")
    for key in REQUIRED_DECLARATIONS:
        assert f'"{key}"' in src, f"{path.stem} 未声明 {key}（§6.1/§6.2 的留档要求）"
    # 锚点语义必须取自契约枚举，不能自造词表
    for m in re.findall(r'"issue_source":\s*"([a-z_]+)"', src):
        assert m in VALID_ISSUE_SOURCES, f"{path.stem} 使用了非法锚点语义 {m}"


@pytest.mark.parametrize("path", _provider_modules(), ids=lambda p: p.stem)
def test_provider_uses_unified_timeout(path):
    """P2-6：超时统一到 http.DEFAULT_TIMEOUT，不再各源各写一套。"""
    src = path.read_text(encoding="utf-8")
    if "timeout" not in src:
        return
    # 不允许再出现硬编码的 tuple 超时字面量
    assert not re.search(r"timeout[^=\n]*=\s*\(\s*\d+\s*,\s*\d+\s*\)", src), \
        f"{path.stem} 仍在硬编码连接/读取超时"


def test_time_budget_semantics():
    b = TimeBudget(0.0)      # 0 → 不限
    assert b.remaining() is None and not b.expired()
    b2 = TimeBudget(None)
    assert b2.remaining() is None
    b3 = TimeBudget(1e6)
    assert not b3.expired() and b3.remaining() > 0
    b4 = TimeBudget(-1.0)    # 立即过期
    assert b4.expired()


def test_budget_stops_retries():
    """预算耗尽时不再退避重试：把"这个源这轮废了"变成可见的失败，
    而不是一个看起来正常、实际只抓了一半的慢速成功。"""
    class Dead:
        def request(self, *a, **kw):
            raise OSError("connection refused")

    sleeps = []
    import weather_eval.forecast.http as h
    orig = h.time.sleep
    h.time.sleep = lambda s: sleeps.append(s)
    try:
        with pytest.raises(RuntimeError):
            request_with_retries(Dead(), "http://x", retries=3, source="T",
                                 budget=TimeBudget(-1.0))
    finally:
        h.time.sleep = orig
    assert sleeps == []          # 一次都没等


# ------------------------------------------------------- P1-6 EW4ALL 轮次探测
def test_ew4all_run_probe_is_per_station(monkeypatch):
    """探测站选中的轮次不得被提升为"模式级事实"供其余站沿用。

    否则若探测站恰好落在偏旧/偏新的轮次上，4 站的快照会一起打上错位的 issue_iso，
    而且不报错、不告警（对抗式审查 P1-6）。
    """
    import tests.test_ew4all as T
    from weather_eval.forecast.ew4all import (
        FIND_BY_POINT_URL, MODEL_TIME_LIST_URL, Ew4allProvider,
    )

    class TwoStations:
        pass

    A = type("A", (), {"id": "a", "name": "A", "lat": 23.4783, "lon": 111.304})
    B = type("B", (), {"id": "b", "name": "B", "lat": 22.0, "lon": 110.0})

    def routes(method, url, params, body):
        if url == MODEL_TIME_LIST_URL:
            return 200, json.dumps({"code": 200, "success": "true",
                                    "data": [{"data_time": "20260912000000"},
                                             {"data_time": "20260911120000"}]})
        if url == FIND_BY_POINT_URL:
            mode, el, dt = body["mode"], body["elements"], body["dataTime"]
            lon = body["point"][0][0]
            run = datetime.strptime(dt, "%Y%m%d%H")
            # 站点 B 在最新轮（00Z）的降水要素尚未出数 -> 该轮对它不可用
            if lon == 110.0 and dt == "2026091200" and el != "TEM":
                return 200, json.dumps({"code": 200, "data": [], "success": "true"})
            if el == "TEM":
                rows = T._rows("TEM", run + timedelta(hours=1), 4, 1, lambda i: 20.0 + i)
            elif el == "ONETPE":
                rows = T._rows(el, run + timedelta(hours=1), 4, 1, lambda i: 0.5)
            else:
                step = 3 if el == "HOURTPE" else 6
                rows = T._rows(el, run + timedelta(hours=step), 2, step, lambda i: 3.0)
            return 200, json.dumps({"code": 200, "message": "ok", "data": rows,
                                    "success": "true"})
        return 404, "{}"

    prov = Ew4allProvider(session=T.RoutingSession(routes))
    snap_a = prov.fetch_snapshot(A, ["cma_ndfs"])[0]
    snap_b = prov.fetch_snapshot(B, ["cma_ndfs"])[0]
    assert snap_a["issue_iso"] == "2026-09-12T08:00"     # A：最新轮可用
    assert snap_b["issue_iso"] == "2026-09-11T20:00"     # B：回退到更早的齐备轮
    assert len(prov._issue_cache) == 2                   # 缓存按 (模型, 站点) 建键
    assert (("cma_ndfs", "a") in prov._issue_cache
            and ("cma_ndfs", "b") in prov._issue_cache)


def test_ew4all_negative_precip_becomes_missing(monkeypatch):
    """P2-8：负降水是非物理值，超过浮点容差即按缺测处理——既不伪装成 0，
    也不让它污染降水指标。"""
    import tests.test_ew4all as T
    from weather_eval.forecast.ew4all import (
        FIND_BY_POINT_URL, MODEL_TIME_LIST_URL, Ew4allProvider,
    )

    def routes(method, url, params, body):
        if url == MODEL_TIME_LIST_URL:
            return 200, json.dumps({"code": 200, "success": "true",
                                    "data": [{"data_time": "20260912000000"}]})
        if url == FIND_BY_POINT_URL:
            el, dt = body["elements"], body["dataTime"]
            run = datetime.strptime(dt, "%Y%m%d%H")
            if el == "TEM":
                rows = T._rows("TEM", run + timedelta(hours=1), 4, 1, lambda i: 20.0 + i)
            elif el == "ONETPE":
                # 混入负值
                rows = T._rows(el, run + timedelta(hours=1), 4, 1,
                               lambda i: -0.5 if i == 1 else 0.5)
            else:
                rows = T._rows(el, run + timedelta(hours=3), 2, 3, lambda i: 3.0)
            return 200, json.dumps({"code": 200, "message": "ok", "data": rows,
                                    "success": "true"})
        return 404, "{}"

    prov = Ew4allProvider(session=T.RoutingSession(routes))
    snap = prov.fetch_snapshot(T.Station, ["cma_ndfs"])[0]
    assert None in snap["data"]["cma_ndfs"]["precipitation"]
    assert all(v is None or v >= 0 for v in snap["data"]["cma_ndfs"]["precipitation"])


def test_ew4all_declares_model_run_anchor():
    """EW4ALL 的锚点是真实模式轮次——本项目可复核性最强的一类，必须如实声明。"""
    import tests.test_ew4all as T
    from weather_eval.forecast.ew4all import Ew4allProvider

    prov = Ew4allProvider(session=T.RoutingSession(T._routes()))
    snap = prov.fetch_snapshot(T.Station, ["cma_ndfs"])[0]
    assert snap["issue_source"] == "model_run"
    assert snap["issue_raw"] == snap["run_label_utc"]
    assert snap["precip_accum_window_hours"] == 3
    assert snap["resolution_hours"] == 1

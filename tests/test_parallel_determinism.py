"""并行的确定性契约（TASK-02）：worker=1 与 worker=N 的输出必须逐位相同。

并行化最危险的失败不是慢，是**静默改变结果**：如果汇总顺序随调度漂移，
浮点求和顺序就会变，round4 之后的展示值可能翻一位——那不是性能问题，是把
榜单悄悄改了（§11 负面清单第 15 条）。

本测试把"顺序无关"钉成可执行的事实：同一个批量分别用 1 个和 4 个 worker 跑，
逐格比对全部指标字段。

同时验证两条工程纪律：
  * 小批量自动走串行——测试用的小数据不该付进程启动的固定开销；
  * 并行失败整批退回串行——绝不返回"部分并行 + 部分串行"的混合结果。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from weather_eval import parallel

pytestmark = pytest.mark.unit

LIMITS = [1, 2]


def _jobs(n_cells: int, n: int = 300):
    """造一批确定性的 (obs, fcst) 调用任务。"""
    rng = np.random.default_rng(11)
    out = []
    for i in range(n_cells):
        o = np.round(rng.normal(22, 3, n), 3)
        f = o + np.round(rng.normal(0, 1.5, n), 3)
        out.append(((o, f, LIMITS, 5), {"n_eff": 60}))
    return out


def _same(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k, va in a.items():
        vb = b[k]
        if va is None or vb is None:
            if va is not None or vb is not None:
                return False
            continue
        if isinstance(va, float) and (math.isnan(va) or math.isnan(vb)):
            if not (math.isnan(va) and math.isnan(vb)):
                return False
            continue
        if va != vb:
            return False
    return True


def test_parallel_output_matches_serial(monkeypatch):
    """worker=1 与 worker=4 对同一批任务给出逐位相同的结果。"""
    jobs = _jobs(48)
    monkeypatch.setenv(parallel.WORKERS_ENV, "1")
    serial = parallel.map_metric_calls("temp_metrics", jobs)
    monkeypatch.setenv(parallel.WORKERS_ENV, "4")
    par = parallel.map_metric_calls("temp_metrics", jobs)
    assert len(serial) == len(par)
    for i, (a, b) in enumerate(zip(serial, par)):
        assert _same(a, b), f"第 {i} 个格子的结果在并行下发生了变化"


def test_precip_parallel_matches_serial(monkeypatch):
    """降水（含分级快速路径）同样逐位相同。"""
    from weather_eval.evaluate import DAILY_GRADED_LEVS
    rng = np.random.default_rng(13)
    jobs = []
    for _ in range(40):
        o = np.round(np.abs(rng.normal(0.5, 2, 260)), 3)
        f = np.round(np.abs(rng.normal(0.7, 2, 260)), 3)
        jobs.append(((o, f, 1.0, 5),
                     {"kind": "24h", "graded_levs": DAILY_GRADED_LEVS,
                      "graded_backend": "cyeva"}))
    monkeypatch.setenv(parallel.WORKERS_ENV, "1")
    serial = parallel.map_metric_calls("precip_metrics", jobs)
    monkeypatch.setenv(parallel.WORKERS_ENV, "4")
    par = parallel.map_metric_calls("precip_metrics", jobs)
    for i, (a, b) in enumerate(zip(serial, par)):
        assert _same(a, b), f"第 {i} 个降水格子在并行下发生了变化"
        for lev in DAILY_GRADED_LEVS:
            ga, gb = a["graded"][lev], b["graded"][lev]
            assert (ga is None) == (gb is None)
            if ga is not None:
                assert _same(ga, gb), f"第 {i} 个格子 {lev} 级分级指标发生了变化"


def test_small_batch_stays_serial(monkeypatch):
    """小批量走串行：测试数据不该付进程启动开销（也保证小场景行为与改造前一致）。"""
    monkeypatch.setenv(parallel.WORKERS_ENV, "4")
    assert parallel.worker_count() == 4
    jobs = _jobs(3)
    got = parallel.map_metric_calls("temp_metrics", jobs)
    monkeypatch.setenv(parallel.WORKERS_ENV, "1")
    assert _same(got[0], parallel.map_metric_calls("temp_metrics", jobs)[0])


def test_falls_back_to_serial_on_worker_failure(monkeypatch):
    """worker 侧任何失败都整批退回串行——绝不产出混合来源的结果。"""
    import concurrent.futures as cf

    def boom(*a, **k):
        raise RuntimeError("worker 挂了")

    monkeypatch.setenv(parallel.WORKERS_ENV, "4")
    monkeypatch.setattr(cf, "ProcessPoolExecutor", boom)
    jobs = _jobs(40)
    got = parallel.map_metric_calls("temp_metrics", jobs)
    monkeypatch.setenv(parallel.WORKERS_ENV, "1")
    serial = parallel.map_metric_calls("temp_metrics", jobs)
    assert all(_same(a, b) for a, b in zip(got, serial)), "退回串行后结果应与串行一致"


def test_empty_batch_is_empty():
    assert parallel.map_metric_calls("temp_metrics", []) == []

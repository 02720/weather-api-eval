"""Golden Master 对拍（I8：口径逐位不变）与性能预算（I12：单轮成本有界）。

这两个测试是本项目全部重构的**安全网**：后面每一个 Phase 都在改代码组织、
存储与计算路径，而读者（与复核者）唯一关心的是"报告里的数字没变"。

对拍判据（§3.1）：
  * 名次**零容差**——排名是离散量，差一位就是结论变了；
  * 展示数字**逐位相同**（已 round 到展示精度）；
  * 内部浮点允许 1e-9 相对容差（跨进程/跨 BLAS 的求和顺序差异）。

为什么基线里必须冻结 `eval_days`（window.json）：`build_day_stat_tables` 的天数轴
取 hourly 与 daily 记录的天并集，`day_block_weights` 的形状依赖 `n_days`。新增一天
会改变整个权重矩阵，bootstrap 的 CI 随之变化——这是**正确行为**。若不冻结天数轴，
增量路径与全量路径会在不同轴长上对拍，被误判成口径漂移，于是"增量计算"这个
正确方向会被一个假红灯永久挡住。

跑法：
    pytest -m golden          # 只跑对拍
    pytest -m "not golden and not slow"   # PR 快层
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

BASELINE_DIR = Path(__file__).resolve().parent / "baseline"
WINDOW_PATH = BASELINE_DIR / "window.json"
BASELINE_PATH = BASELINE_DIR / "report_baseline.json"
BUDGET_PATH = BASELINE_DIR / "perf_budget.json"

# 内部浮点容差：1e-9 相对误差。依据：float64 的机器精度是 2.2e-16，
# 1.5M 条样本的求和累积误差上界约 n·eps ≈ 3e-10（成对求和更低），
# 1e-9 留了一个数量级余量，足以吸收 BLAS/线程数变化，又足以抓住真正的口径漂移。
REL_TOL = 1e-9

pytestmark = [pytest.mark.golden]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def frozen_report():
    """用冻结窗口构建一次报告（session 级：对拍与预算共用，不重复花 100 秒）。"""
    import os

    os.environ.setdefault("TZ", "Asia/Shanghai")
    from weather_eval.config import load_config
    from weather_eval.evaluate import build_report
    from weather_eval.timeutil import parse_iso

    win = _load(WINDOW_PATH)
    cfg = load_config(None)
    start, end = parse_iso(win["start_dt"]), parse_iso(win["end_dt"])

    t0 = time.perf_counter()
    report = build_report(cfg.station_ids, cfg.models, cfg.eval, start, end,
                          period_label=win["period_label"])
    elapsed = time.perf_counter() - t0

    import resource
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    report["__bench__"] = {"seconds": elapsed, "peak_rss_mb": rss}
    return report


def _num_close(a, b) -> bool:
    """逐位相同优先；不等时用 1e-9 相对容差兜底（内部浮点）。"""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, list) or isinstance(b, list):
        return (isinstance(a, list) and isinstance(b, list) and len(a) == len(b)
                and all(_num_close(x, y) for x, y in zip(a, b)))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        scale = max(abs(a), abs(b), 1.0)
        return abs(a - b) / scale <= REL_TOL
    return a == b


def _board_fingerprint(rows):
    """名次指纹：按基线顺序把 (model, score) 串起来——名次零容差的落点。"""
    return [(r.get("model"), r.get("score")) for r in rows]


def test_baseline_files_exist():
    """基线必须先存在：没有基线的对拍测试是永远为真的假绿灯。"""
    for p in (WINDOW_PATH, BASELINE_PATH, BUDGET_PATH):
        assert p.is_file(), f"缺少基线文件 {p}（先跑 scripts/make_baseline.py）"


def test_window_matches_frozen_days(frozen_report):
    """天数轴必须与冻结基线同轴（否则 CI 会被误判成口径漂移）。"""
    win = _load(WINDOW_PATH)
    meta = frozen_report["meta"]
    assert meta["bootstrap_days"] == win["n_boot_days"], (
        "评估天数轴发生了变化：对拍必须在同一天数轴上进行。"
        "若这是数据自然增长（新增一天），请重跑 scripts/make_baseline.py 重新冻结，"
        "不要放宽这条断言。")


def test_report_matches_frozen_baseline(frozen_report):
    """I8：报告对外展示的每个数字逐位不变，名次零容差。"""
    base = _load(BASELINE_PATH)
    live_boards = frozen_report["leaderboards"]

    assert set(live_boards) == set(base["boards"]), (
        "榜单集合变了：多出/缺失的榜意味着报告结构变化，必须是刻意的决策")

    problems: list[str] = []
    for key, base_rows in base["boards"].items():
        live_rows = live_boards[key]
        if len(live_rows) != len(base_rows):
            problems.append(f"{key}: 行数 {len(live_rows)} != {len(base_rows)}")
            continue
        # 名次零容差：顺序本身就是结论
        if _board_fingerprint(live_rows) != _board_fingerprint(base_rows):
            for i, (lr, br) in enumerate(zip(live_rows, base_rows)):
                if (lr.get("model"), lr.get("score")) != (br.get("model"), br.get("score")):
                    problems.append(
                        f"{key} 第 {i+1} 名不一致：基线 {br.get('model')}={br.get('score')} "
                        f"vs 实测 {lr.get('model')}={lr.get('score')}")
        for i, (lr, br) in enumerate(zip(live_rows, base_rows)):
            for field, bval in br.items():
                if field == "model":
                    continue
                lval = lr.get(field)
                if not _num_close(lval, bval):
                    problems.append(
                        f"{key}[{br.get('model')}].{field}: 基线 {bval} vs 实测 {lval}")

    for m, base_slots in base["scorecard"].items():
        live_slots = frozen_report["scorecard"].get(m) or {}
        for slot, bvals in base_slots.items():
            lvals = live_slots.get(slot) or {}
            for field, bval in bvals.items():
                if not _num_close(lvals.get(field), bval):
                    problems.append(f"scorecard[{m}][{slot}].{field}: "
                                    f"基线 {bval} vs 实测 {lvals.get(field)}")

    assert not problems, "与 Golden Master 基线不一致（前 20 条）：\n" + "\n".join(
        problems[:20])


def test_diagnostics_unchanged(frozen_report):
    """诊断层的数字不受"复用内存数据"影响（TASK-04 的守卫）。

    为什么单独钉这一条：诊断层读的是窗口内的**全部**快照（含被评估排除的残缺与
    未封存快照），而 `build_report` 传给评估的快照口径可能不同。把"不再二次读盘"
    接上去时，最危险的失败不是变慢，而是**顺手缩小了读取范围**导致跨源相关与
    指纹漂移静默改变——这条测试就是那个失败的警报器。
    """
    base = _load(BASELINE_PATH)
    base_diag = (base["meta"] or {}).get("diagnostics") or {}
    live_diag = (frozen_report["meta"] or {}).get("diagnostics") or {}
    assert base_diag.get("available") == live_diag.get("available")
    for key in ("source_correlation", "station_rho", "fingerprint"):
        base_item, live_item = base_diag.get(key), live_diag.get(key)
        if base_item is None and live_item is None:
            continue
        assert (base_item is None) == (live_item is None), f"诊断项 {key} 的出现与否变了"
        if not isinstance(base_item, dict):
            continue
        for field, bv in base_item.items():
            if isinstance(bv, (int, float)) and not isinstance(bv, bool):
                assert _num_close(live_item.get(field), bv), (
                    f"diagnostics.{key}.{field}: 基线 {bv} vs 实测 {live_item.get(field)}")


@pytest.mark.slow
def test_perf_budget(frozen_report):
    """I12：单轮成本有界。预算随优化推进**单向收紧**（见 perf_budget.json）。"""
    budget = _load(BUDGET_PATH)
    bench = frozen_report["__bench__"]
    assert bench["seconds"] <= budget["max_seconds"], (
        f"报告构建 {bench['seconds']:.1f}s 超过预算 {budget['max_seconds']}s")
    assert bench["peak_rss_mb"] <= budget["max_peak_rss_mb"], (
        f"峰值内存 {bench['peak_rss_mb']:.0f} MB 超过预算 {budget['max_peak_rss_mb']} MB")


@pytest.mark.slow
def test_determinism_two_builds():
    """I5：同数据 + 同代码 + 同种子 → 同报告。跑两次比对。

    为什么不比较完整 dict 而只比较对外数字：完整 dict 含 generated_at 之类
    随运行时刻变化的字段，那是**应该**变的。可复现性指的是结论可复现。
    """
    import os

    os.environ.setdefault("TZ", "Asia/Shanghai")
    from weather_eval.config import load_config
    from weather_eval.evaluate import build_report
    from weather_eval.timeutil import parse_iso

    win = _load(WINDOW_PATH)
    cfg = load_config(None)
    start, end = parse_iso(win["start_dt"]), parse_iso(win["end_dt"])

    def fingerprint():
        rep = build_report(cfg.station_ids, cfg.models, cfg.eval, start, end,
                           period_label=win["period_label"])
        return {k: _board_fingerprint(v) for k, v in rep["leaderboards"].items()}

    a, b = fingerprint(), fingerprint()
    assert a == b, "同一份数据两次构建给出了不同的榜单结果（可复现性被破坏）"

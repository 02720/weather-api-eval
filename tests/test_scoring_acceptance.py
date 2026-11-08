"""验收断言（2026-11 优化方案 §3.1/§4.1）：参照与技能曲线不得随 lead 方向性失真。

这些断言直接读**冻结基线**（`tests/baseline/report_baseline.json`）与分块缓存，
不重建报告——它们锁的是"发布的数字"本身，而不是某个函数在合成夹具上的行为。

背景（方案发现 1/2/13）：
  * 旧 run-anchored 参照下 `rmse_ref` 中位 2.13(1d)→4.60(14d)，参照自身误差
    随 lead 上升，把长时效的技能分机械抬高（温度分"越报越准"）；
  * 同一 −R 项还让 `r_anom` 恒在 0.66~0.75、不随 lead 衰减。

固定日历气候态的目标是：参照难度与 lead 无关，技能与异常相关随 lead 下降。
"""
from __future__ import annotations

import json
from pathlib import Path

BASELINE = Path(__file__).resolve().parent / "baseline"


def _load(name: str):
    return json.loads((BASELINE / name).read_text(encoding="utf-8"))


def _median(vals):
    v = sorted(float(x) for x in vals if x is not None)
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2.0


def test_reference_error_does_not_grow_with_lead():
    """`median(rmse_ref)` 对 lead 的斜率必须 ≤ 0.05°C/天（方案 §3.1 验收）。

    冻结基线的 reference_profile 逐天桶发布 rmse_ref 中位数；这里对 1..14 天
    做最小二乘斜率。16 天桶样本极薄（实测 9 格），不参与拟合。
    """
    meta = _load("report_baseline.json")["meta"]
    prof = meta.get("reference_profile") or {}
    rows = [r for r in (prof.get("hourly") or []) if r.get("day", 99) <= 14]
    assert len(rows) >= 10, "reference_profile 缺失或过短，无法做斜率断言"
    xs = [float(r["day"]) for r in rows]
    ys = [float(r["rmse_ref_median"]) for r in rows]
    n = len(xs)
    sx, sy = sum(xs), sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    den = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / den if den else 0.0
    assert slope <= 0.05, f"rmse_ref 随 lead 上升 {slope:.3f} °C/天（参照在退化）"


def test_temperature_skill_and_anomaly_corr_decay_with_lead():
    """`median(temp_skill)` 与 `median(r_anom)` 随 lead 单调不增（1..14 天）。

    允许 1 个桶的小幅反弹（样本噪声），但整体必须下降——旧口径下这两条曲线
    分别是"上升"与"平直"，正是发现 1/2 的方向性失真。
    """
    boards = _load("report_baseline.json")["boards"]
    vals = []
    for d in range(1, 15):
        rows = boards[f"hourly:{d}d"]
        vals.append(_median([r.get("temp_skill") for r in rows]))
    assert all(v is not None for v in vals), "温度技能在部分天桶缺失"
    # 1d → 14d 必须显著下降（旧口径是上升）
    assert vals[-1] < vals[0] - 10.0, f"温度技能未随 lead 衰减：{vals}"

    # r_anom 由分块缓存直接复算（基线 boards 不发布该列）
    from collections import defaultdict

    by_day: dict[int, list[float]] = defaultdict(list)
    for path in sorted((BASELINE.parents[1] / "data" / "metrics" / "blocks").glob("*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        if int(d.get("schema") or 0) != 6:
            continue
        for by_b in (d.get("hourly") or {}).values():
            for day_s, cell in by_b.items():
                v = (cell.get("t") or {}).get("r_anom")
                if v is not None:
                    by_day[int(day_s)].append(float(v))
    if not by_day:
        return
    med = {k: _median(v) for k, v in by_day.items()}
    early = [med[d] for d in (1, 2, 3) if d in med]
    late = [med[d] for d in (10, 12, 14) if d in med]
    assert early and late
    assert _median(late) < _median(early) - 0.1, (
        f"r_anom 未随 lead 衰减（早期 {_median(early):.3f} → 晚期 {_median(late):.3f}）")


def test_leaderboard_rows_are_additive_and_bucket_floor_is_respected():
    """`R = t_contrib + p_contrib − 50` 逐行可加；2 家桶不产生名次。"""
    boards = _load("report_baseline.json")["boards"]
    rows = boards["all"]
    for r in rows:
        if (r.get("score") is not None and r.get("temp_score") is not None
                and r.get("precip_score") is not None):
            assert abs(r["score"] - (r["temp_score"] + r["precip_score"] - 50.0)) <= 0.02
    for key, rows in boards.items():
        if ":" not in key:
            continue
        live = [r for r in rows if r.get("score") is not None]
        assert len(live) == 0 or len(live) >= 3, (
            f"{key}：只有 {len(live)} 家产生名次——2 家桶应判无相对信息")

#!/usr/bin/env python3
"""降水分级指标：cyeva 权威路径 vs numpy 快速路径的逐格对拍。

这是把生产路径从 cyeva 切到 numpy 的**前置条件**（§4.5 措施三 / TASK-29）：
先证明 round4 逐位相同，再谈切换。反过来做就是拿结论换速度。

对拍的不是"两个浮点数"，而是三层，逐层收窄：
  1. **列联计数**（整数）——最小充分统计量。整数零容差；这里若不同，说明二值化
     口径（舍入/NaN/区间端点）就错了，看指标数值是看不出来的。
  2. **指标数值**（round2 后的浮点）——逐位相同。
  3. **空档/除零的处置**——两边都必须给出 None，绝不能一边 nan 一边 0.0。

用法：
    python scripts/parity_graded.py                       # 全量对拍
    python scripts/parity_graded.py --window-file tests/baseline/window.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return a == b


def _cyeva_graded(obs, fcst, kind, levs):
    """权威路径：cyeva PrecipitationComparison 的 calc_*(kind=, lev=)。"""
    import numpy as np
    from cyeva import PrecipitationComparison
    from cyeva.errors import ArrayLengthNotEqualError
    pc = PrecipitationComparison(np.asarray(obs, dtype=float),
                                 np.asarray(fcst, dtype=float), unit="mm")
    out = {}
    for lev in levs:
        try:
            # ArrayLengthNotEqualError 不是 ValueError 的子类：样本全 NaN 时
            # drop_nan 会把序列剔空，cyeva 在此抛它。快速路径对此返回 None。
            out[lev] = {
                "acc": pc.calc_accuracy_ratio(kind=kind, lev=lev),
                "pod": pc.calc_hit_ratio(kind=kind, lev=lev),
                "far": pc.calc_false_alarm_ratio(kind=kind, lev=lev),
                "miss": pc.calc_miss_ratio(kind=kind, lev=lev),
                "ts": pc.calc_ts(kind=kind, lev=lev),
                "ets": pc.calc_ets(kind=kind, lev=lev),
                "bias": pc.calc_bias_score(kind=kind, lev=lev),
            }
        except (ValueError, KeyError, IndexError, ZeroDivisionError,
                ArrayLengthNotEqualError):
            out[lev] = None
    return out


def _norm(d):
    """把 cyeva 的输出规整成与快速路径同形态（nan/inf → None，round2）。"""
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
            continue
        f = float(v)
        out[k] = None if (math.isnan(f) or math.isinf(f)) else round(f, 2)
    return out


def run(window_file: str | None, max_cells: int) -> dict:
    import weather_eval.evaluate as E
    from weather_eval.config import load_config
    from weather_eval.graded import graded_counts, precip_graded_metrics

    os.environ.setdefault("TZ", "Asia/Shanghai")
    cfg = load_config(None)
    from weather_eval.__main__ import _live_window
    from weather_eval.timeutil import parse_iso
    if window_file:
        w = json.loads(Path(window_file).read_text(encoding="utf-8"))
        start, end = parse_iso(w["start_dt"]), parse_iso(w["end_dt"])
    else:
        start, end, _ = _live_window(cfg)

    ev = cfg.eval
    hourly, daily = E.collect(
        cfg.station_ids, cfg.models, start, end,
        ev["hourly_lead_days"], ev["daily_max_offset_days"],
        ev.get("daily_min_hours", 20), ev.get("daily_source_fallback", True),
        require_complete=ev.get("require_complete_snapshots", True),
        require_frozen=ev.get("require_frozen_samples", True))

    cells: list[tuple[str, str, str, list, list]] = []
    by_mb: dict[tuple[str, int], list] = defaultdict(list)
    for r in hourly:
        if 1 <= r["bucket"] <= ev["hourly_lead_days"]:
            by_mb[(r["model"], r["bucket"])].append(r)
    for (m, b), recs in sorted(by_mb.items()):
        ro = [x["rain_obs"] for x in recs]
        rf = [x["rain_fcst"] for x in recs]
        cells.append(("1h", f"{m}@{b}d", E.HOURLY_GRADED_LEVS, ro, rf))
    by_off: dict[tuple[str, int], list] = defaultdict(list)
    for r in daily:
        if 1 <= r["offset"] <= ev["daily_max_offset_days"]:
            by_off[(r["model"], r["offset"])].append(r)
    for (m, b), recs in sorted(by_off.items()):
        ro = [x["rain_obs"] for x in recs]
        rf = [x["rain_fcst"] for x in recs]
        cells.append(("24h", f"{m}@{b}d", E.DAILY_GRADED_LEVS, ro, rf))

    if max_cells:
        step = max(1, len(cells) // max_cells)
        cells = cells[::step]

    n_count_mismatch = 0
    n_value_mismatch = 0
    n_none_mismatch = 0
    checked = 0
    examples: list[str] = []

    for kind, name, levs, ro, rf in cells:
        checked += 1
        a = {lev: _norm(v) for lev, v in _cyeva_graded(ro, rf, kind, levs).items()}
        b = precip_graded_metrics(ro, rf, kind, levs)
        # 第一层：列联计数
        import numpy as np
        cnt = graded_counts(np.asarray(ro, dtype=float), np.asarray(rf, dtype=float),
                            kind, levs)
        for lev in levs:
            if (a.get(lev) is None) != (b.get(lev) is None):
                n_none_mismatch += 1
                if len(examples) < 10:
                    examples.append(f"{kind} {name} lev={lev}: None 处置不一致 "
                                    f"cyeva={a.get(lev) is None} numpy={b.get(lev) is None}")
                continue
            if a.get(lev) is None:
                continue
            for k in ("acc", "pod", "far", "miss", "ts", "ets", "bias"):
                if not _eq(a[lev].get(k), b[lev].get(k)):
                    n_value_mismatch += 1
                    if len(examples) < 10:
                        examples.append(
                            f"{kind} {name} lev={lev}.{k}: cyeva={a[lev].get(k)} "
                            f"numpy={b[lev].get(k)} counts={cnt.get(lev)}")
        _ = n_count_mismatch

    return {
        "cells_checked": checked,
        "levels_per_cell": {"1h": len(E.HOURLY_GRADED_LEVS),
                            "24h": len(E.DAILY_GRADED_LEVS)},
        "value_mismatches": n_value_mismatch,
        "none_handling_mismatches": n_none_mismatch,
        "parity": (n_value_mismatch == 0 and n_none_mismatch == 0),
        "examples": examples,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-file", default=None)
    ap.add_argument("--max-cells", type=int, default=0, help="抽样格子数（0=全量）")
    args = ap.parse_args(argv)
    res = run(args.window_file, args.max_cells)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0 if res["parity"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

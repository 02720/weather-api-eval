#!/usr/bin/env python3
"""temp_metrics 标量六项 / precip_metrics 连续三项：cyeva 权威路径 vs numpy 快速路径的逐格对拍。

这是把生产路径从 cyeva 切到 numpy 的**前置条件**（与 scripts/parity_graded.py
同一纪律）：先证明逐位相同，再谈切换。对拍覆盖三类真实格子（与
`_build_report_unlocked` 的派发同一套构造）：

  * 评分卡三池（temp_24h / temp_72h / temp_all 及对应 precip 连续量）；
  * 逐小时按天桶 1..N（温度全套 + 降水诊断口径 0.1mm，含分级）；
  * 按天按偏移 1..N（最高/最低温度全套）。

对拍的不是"两个浮点数"，而是三层，逐层收窄：
  1. **None 处置**——两边要么都出数、要么都 None，绝不能一边 nan 一边 0.0；
  2. **指标数值**（_r 终端舍入后的浮点）——按 float.hex() 逐位相同；
  3. **整数与结构字段**（n / n_eff / graded 结构）——零容差。

用法：
    python scripts/parity_temp.py                       # 全量对拍
    python scripts/parity_temp.py --window-file tests/baseline/window.json
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


def _bit_eq(a, b) -> bool:
    """对外字段的位级相等：None 恒等、整数恒等、浮点比 float.hex()。"""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return a.hex() == b.hex()
    return a == b


def _diff_fields(a: dict, b: dict, skip: set[str]) -> list[str]:
    out = []
    for k in a:
        if k in skip:
            continue
        va, vb = a.get(k), b.get(k)
        if isinstance(va, dict) and isinstance(vb, dict):
            for lk in set(va) | set(vb):
                if not _bit_eq(va.get(lk), vb.get(lk)):
                    out.append(f"{k}.{lk}")
        elif not _bit_eq(va, vb):
            out.append(k)
    return out


def run(window_file: str | None, max_cells: int) -> dict:
    import weather_eval.evaluate as E
    from weather_eval.config import load_config
    from weather_eval.timeutil import parse_iso

    os.environ.setdefault("TZ", "Asia/Shanghai")
    cfg = load_config(None)
    if window_file:
        w = json.loads(Path(window_file).read_text(encoding="utf-8"))
        start, end = parse_iso(w["start_dt"]), parse_iso(w["end_dt"])
    else:
        from weather_eval.__main__ import _live_window
        start, end, _ = _live_window(cfg)

    ev = cfg.eval
    limits = ev["temp_accuracy_limits"]
    min_sample = ev["min_sample"]
    thr = ev["rain_threshold_mm"]
    hourly, daily = E.collect(
        cfg.station_ids, cfg.models, start, end,
        ev["hourly_lead_days"], ev["daily_max_offset_days"],
        ev.get("daily_min_hours", 20), ev.get("daily_source_fallback", True),
        require_complete=ev.get("require_complete_snapshots", True),
        require_frozen=ev.get("require_frozen_samples", True))

    by_model: dict[str, list] = defaultdict(list)
    for r in hourly:
        by_model[r["model"]].append(r)
    daily_by_model: dict[str, list] = defaultdict(list)
    for r in daily:
        daily_by_model[r["model"]].append(r)

    # 与 _build_report_unlocked 同一套格子构造（dict 路径 n_eff；对拍只比较
    # 后端产出，n_eff/groups 两个 kwargs 对两条路径是同一对象，不影响判定）
    cells: list[tuple[str, tuple, dict]] = []
    for m in cfg.models:
        recs = by_model.get(m, [])
        h24 = [r for r in recs if 1 <= r["lead"] <= 24]
        h72 = [r for r in recs if 1 <= r["lead"] <= 72]
        for tag, pool in (("temp_24h", h24), ("temp_72h", h72),
                          ("temp_all", recs)):
            cells.append((
                f"temp/{m}/{tag}",
                (*E._farrays(pool, ("temp_obs", "temp_fcst")), limits, min_sample),
                {"n_eff": E._n_eff_temp(pool),
                 "groups": E._station_groups(pool, "temp_obs", "temp_fcst")}))
        for tag, pool in (("precip_24h", h24), ("precip_72h", h72),
                          ("precip_all", recs)):
            cells.append((
                f"precip/{m}/{tag}",
                (*E._farrays(pool, ("rain_obs", "rain_fcst")), thr, min_sample),
                {"n_eff": E._n_eff_rain(pool, thr, "valid_iso", "lead")}))
        buckets: dict[int, list] = defaultdict(list)
        for r in recs:
            buckets[r["bucket"]].append(r)
        for b in range(1, ev["hourly_lead_days"] + 1):
            pool = buckets.get(b, [])
            cells.append((
                f"temp/{m}/{b}d",
                (*E._farrays(pool, ("temp_obs", "temp_fcst")), limits, min_sample),
                {"n_eff": E._n_eff_temp(pool),
                 "groups": E._station_groups(pool, "temp_obs", "temp_fcst")}))
            cells.append((
                f"precip/{m}/{b}d",
                (*E._farrays(pool, ("rain_obs", "rain_fcst")), thr, min_sample),
                {"kind": "1h", "graded_levs": E.HOURLY_GRADED_LEVS,
                 "n_eff": E._n_eff_rain(pool, thr, "valid_iso", "lead")}))
    for m in cfg.models:
        by_off: dict[int, list] = defaultdict(list)
        for r in daily_by_model.get(m, []):
            by_off[r["offset"]].append(r)
        for off in range(1, ev["daily_max_offset_days"] + 1):
            sub = by_off.get(off, [])
            for half, ka, kb in (("max", "temp_max_obs", "temp_max_fcst"),
                                 ("min", "temp_min_obs", "temp_min_fcst")):
                cells.append((
                    f"tempd/{m}/{half}{off}d",
                    (*E._farrays(sub, (ka, kb)), limits, min_sample),
                    {"n_eff": E._n_eff_daily_temp(sub),
                     "groups": E._station_groups(sub, ka, kb)}))

    if max_cells:
        step = max(1, len(cells) // max_cells)
        cells = cells[::step]

    checked = 0
    mismatches: list[str] = []
    examples: list[str] = []
    for name, args, kw in cells:
        fn = E.precip_metrics if name.startswith("precip/") else E.temp_metrics
        a = fn(*args, backend="cyeva", **kw)
        b = fn(*args, backend="numpy", **kw)
        checked += 1
        diff = _diff_fields(a, b, skip=set())
        if diff:
            mismatches.append(name)
            if len(examples) < 10:
                examples.append(f"{name}: {diff} cyeva={a} numpy={b}")

    return {
        "cells_checked": checked,
        "mismatch_cells": len(mismatches),
        "parity": not mismatches,
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

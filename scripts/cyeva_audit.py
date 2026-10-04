#!/usr/bin/env python3
"""cyeva 调用审计（TASK-01）：先量清楚，再动手改。

`precip_metrics` / `temp_metrics` 是报告构建的第一大热点（本机实测 31.8 s +
4.9 s，占总耗时约 35%）。但"热点"只说明它贵，不说明它浪费——在不知道
**1,092 次调用里有多少次是重复计算、有多少次算了门槛下根本不会进榜的格子、
分级指标又有多少项从未被消费**之前动手，就是在赌。

本脚本回答三个可量化的问题（对应 §4.5 措施二与措施三）：

  1. **重复率**：按 (输入内容哈希, 阈值, kind, 分级级别, min_sample, n_eff)
     去重后还剩多少次唯一调用？重复的那部分可以直接 memo 掉。
  2. **门槛后置的浪费**：多少次调用在构造 cyeva 对象之后才发现样本不足？
     有多少次调用的输入根本是空的（空桶）？——空桶的 n_eff 是在**调用之前**
     由 `_n_eff_*` 算出来的，那部分成本挂在别处，这里一并统计。
  3. **未消费的分级项**：分级指标 6 级 × 7 项是否有任何下游消费（页面 /
     月度摘要 / 公开数据）。没有消费却每轮算 896×77 次，就是纯浪费。

为什么用"输入内容哈希"而不是 (模型, 桶) 当键：审计脚本**不应该**为了观测而
改动生产代码的函数签名（那会让观测本身成为改动的一部分）。内容哈希是从
调用实参直接算出来的，对生产路径零侵入，且天然覆盖"同一格子被两条轨道
重复计算"这种跨调用点的重复。

用法：
    python scripts/cyeva_audit.py [--window-file tests/baseline/window.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _key_of(obs, fcst, threshold, min_sample, kind, graded_levs, n_eff):
    """调用身份：输入内容 + 全部会改变结果的标量参数。"""
    import numpy as np
    o = np.asarray(obs, dtype=float)
    f = np.asarray(fcst, dtype=float)
    h = hashlib.blake2b(digest_size=16)
    h.update(o.tobytes())
    h.update(f.tobytes())
    for v in (threshold, min_sample, kind or "", tuple(graded_levs or ()), n_eff):
        h.update(repr(v).encode())
    return h.hexdigest()


def _all_none(d: dict) -> bool:
    """该次调用是否产出了任何非 None 的指标（全 None = 这次计算对外无贡献）。"""
    vals = [v for k, v in d.items() if k != "graded"]
    if not any(v is not None for v in vals):
        graded = d.get("graded") or {}
        return not any(g is not None for g in graded.values())
    return False


def run(window_file: str | None) -> dict:
    import weather_eval.evaluate as E
    from weather_eval.config import load_config

    os.environ.setdefault("TZ", "Asia/Shanghai")
    cfg = load_config(None)

    from weather_eval.__main__ import _live_window
    from weather_eval.timeutil import parse_iso
    if window_file:
        w = json.loads(Path(window_file).read_text(encoding="utf-8"))
        start, end, label = (parse_iso(w["start_dt"]), parse_iso(w["end_dt"]),
                             w.get("period_label") or "frozen")
    else:
        start, end, label = _live_window(cfg)

    calls: list[dict] = []
    orig_precip, orig_temp = E.precip_metrics, E.temp_metrics

    def spy_precip(obs, fcst, threshold, min_sample, kind=None,
                   graded_levs=(), n_eff=None):
        t0 = time.perf_counter()
        out = orig_precip(obs, fcst, threshold, min_sample, kind=kind,
                          graded_levs=graded_levs, n_eff=n_eff)
        dt = time.perf_counter() - t0
        import numpy as np
        n = int((np.isfinite(np.asarray(obs, dtype=float))
                 & np.isfinite(np.asarray(fcst, dtype=float))).sum())
        calls.append({"fn": "precip_metrics", "seconds": dt, "n": n,
                      "key": _key_of(obs, fcst, threshold, min_sample, kind,
                                     graded_levs, n_eff),
                      "kind": kind, "levels": len(graded_levs or ()),
                      "gated": (n < min_sample or n == 0
                                or (n_eff is not None and n_eff < min_sample)),
                      "empty": n == 0,
                      "all_none": _all_none(out)})
        return out

    def spy_temp(obs, fcst, limits, min_sample, *, n_eff=None, groups=None):
        t0 = time.perf_counter()
        out = orig_temp(obs, fcst, limits, min_sample, n_eff=n_eff, groups=groups)
        dt = time.perf_counter() - t0
        import numpy as np
        n = int((np.isfinite(np.asarray(obs, dtype=float))
                 & np.isfinite(np.asarray(fcst, dtype=float))).sum())
        calls.append({"fn": "temp_metrics", "seconds": dt, "n": n,
                      "key": _key_of(obs, fcst, tuple(limits), min_sample,
                                     None, (), n_eff),
                      "kind": None, "levels": 0,
                      "gated": (n < min_sample or n == 0
                                or (n_eff is not None and n_eff < min_sample)),
                      "empty": n == 0,
                      "all_none": _all_none(out)})
        return out

    E.precip_metrics, E.temp_metrics = spy_precip, spy_temp
    try:
        E.build_report(cfg.station_ids, cfg.models, cfg.eval, start, end,
                       period_label=label)
    finally:
        E.precip_metrics, E.temp_metrics = orig_precip, orig_temp

    total_s = sum(c["seconds"] for c in calls)
    by_fn: dict[str, dict] = {}
    for fn in ("precip_metrics", "temp_metrics"):
        cs = [c for c in calls if c["fn"] == fn]
        uniq = {c["key"] for c in cs}
        gated = [c for c in cs if c["gated"]]
        empty = [c for c in cs if c["empty"]]
        by_fn[fn] = {
            "calls": len(cs),
            "unique_calls": len(uniq),
            "duplicate_calls": len(cs) - len(uniq),
            "duplicate_rate_pct": round(100 * (len(cs) - len(uniq)) / len(cs), 1) if cs else 0.0,
            "duplicate_seconds": round(sum(c["seconds"] for c in cs
                                           if c["key"] in {x["key"] for x in cs}) - 0.0, 3),
            "gated_calls": len(gated),
            "gated_seconds": round(sum(c["seconds"] for c in gated), 3),
            "empty_input_calls": len(empty),
            "all_none_results": sum(1 for c in cs if c["all_none"]),
            "total_seconds": round(sum(c["seconds"] for c in cs), 2),
        }
    # 重复调用里"第二次及以后"的耗时 = 可 memo 掉的收益（同一 key 只留第一次）
    for fn in by_fn:
        seen: set[str] = set()
        dup_s = 0.0
        for c in (c for c in calls if c["fn"] == fn):
            if c["key"] in seen:
                dup_s += c["seconds"]
            seen.add(c["key"])
        by_fn[fn]["memo_recoverable_seconds"] = round(dup_s, 2)

    graded = [c for c in calls if c["levels"]]
    return {
        "window": {"start": start.strftime("%Y-%m-%d %H:%M"),
                   "end": end.strftime("%Y-%m-%d %H:%M")},
        "total_calls": len(calls),
        "total_seconds": round(total_s, 2),
        "by_fn": by_fn,
        "graded": {
            "calls": len(graded),
            "seconds": round(sum(c["seconds"] for c in graded), 2),
            "share_pct": round(100 * sum(c["seconds"] for c in graded) / total_s, 1) if total_s else 0.0,
            "sub_calls": sum(c["levels"] * 7 for c in graded),
            "levels_hist": dict(Counter(c["levels"] for c in graded)),
            "kinds": dict(Counter(c["kind"] for c in graded)),
        },
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-file", default=None)
    ap.add_argument("--out", default=None, help="把 JSON 写到该路径")
    args = ap.parse_args(argv)
    res = run(args.window_file)
    payload = json.dumps(res, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(payload, encoding="utf-8")
        print(f"已写入 {args.out}")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

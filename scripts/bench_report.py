#!/usr/bin/env python3
"""报告构建的分段基准：`build_report` 的每一段耗时 + 峰值 RSS。

这是 I12（单轮成本有界）与全部性能验收的**唯一依据**——§11 负面清单第 12 条：
动手优化之前先跑一次拿到本机 before，不要凭别处测来的数字动手。

为什么分段而不是只记总时间（第一性原理）：`build_report` 的总耗时是若干个
性质完全不同的阶段之和——IO（∝ 历史长度）、配对（∝ 快照份数）、cyeva 指标
（∝ 样本量 × 分级项数）、bootstrap（O(1)，已向量化）、榜单与渲染。它们对
"数据量翻 10 倍"的响应曲线完全不同，只看总数会把一次正确的优化（例如把 IO
砍掉 90%）淹没在噪声里，也会把一次错误的优化（例如去优化只占 0.5% 的
einsum）当成进展。

分段的实现方式：**包装模块级函数**而不是改生产代码。基准是可丢弃的测量工具，
它不该在 `evaluate.py` 里留下计时代码（那会让"为什么这里有个 perf_counter"成为
下一个读者的问题）。包装器只累加独占耗时（不含被包装函数的嵌套调用），与
profiler 的 toplevel 口径一致。

用法：
    python scripts/bench_report.py                  # 跑一次，写入 .work/bench/
    python scripts/bench_report.py --window-file tests/baseline/window.json
    python scripts/bench_report.py --repeat 3       # 取中位数
    python scripts/bench_report.py --json -         # 只打印 JSON，不落盘
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# 被包装的函数：与 §2.2 剖析表一一对应，便于 before/after 逐项对账。
EVAL_TARGETS = (
    "temp_metrics", "precip_metrics", "precip_binary_metrics", "temp_curve_metrics",
    "_n_eff_temp", "_n_eff_rain", "_n_eff_daily_temp",
    "_n_eff_temp_columnar", "_n_eff_rain_columnar", "_n_eff_daily_temp_columnar",
    "_resolution_boards", "_compute_diagnostics", "collect", "_preload",
    "_build_heatmap", "_build_timeseries", "_coverage", "_build_board_rows",
    "_track_lead_boards", "_combined_day_boards", "_board_menu", "_board_window",
    "_build_row_facts", "_rain_evidence_by_model", "_valid_lead_days",
    "_valid_rain_days", "_valid_daily_temp_days",
    "_weight_sensitivity_champions", "_score_trend",
    "_count_incomplete", "_issue_anchor_meta", "_model_status", "_snapshot_quality",
    "_daily_error_lag1_rho_columnar", "_build_heatmap_columnar",
    # evaluate 以 `from ... import` 直接绑定了这三个快照助手：包装模块属性拦不到
    # 已绑定的引用，必须包装 evaluate 命名空间里的同名别名。
    "sanitize_hourly_arrays", "snapshot_complete", "integrity_summary",
)
STATS_TARGETS = (
    "build_day_stat_tables", "aggregate_day_stats", "track_bucket_scores",
    "bucket_components",
    "day_block_bootstrap", "difficulty_adjusted", "two_way_adjust",
    "daily_error_lag1_rho", "pairwise_corr_matrix", "source_corr_cluster",
)
DIAG_TARGETS = (
    "compute_all", "iter_error_samples", "cross_source_correlation", "station_rho",
    "fingerprint_drift", "bridge_and_jackknife", "corridor_sensitivity",
    "score_slope_influence", "composite_dimension_share",
)
OUT_DIR = PROJECT_ROOT / ".work" / "bench"


def _peak_rss_mb() -> float:
    """峰值 RSS（MB）。单位随平台不同：Linux 的 ru_maxrss 是 KB、macOS 是字节。

    旧实现的两个分支完全相同（都除以 1024），在 macOS 上会把字节当 KB 报出
    1024 倍的数字；按平台判别才是唯一正确的换算。
    """
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return round(rss / (1024.0 * 1024.0), 1)
    return round(rss / 1024.0, 1)


def _code_sha() -> str:
    """当前 commit 短 SHA（不可得时用 'unknown'，绝不因取不到而失败）。"""
    for cmd in (["git", "rev-parse", "--short", "HEAD"],):
        try:
            out = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                                 text=True, timeout=10)
            if out.returncode == 0:
                return out.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            continue
    return "unknown"


def _install_probes(acc: dict):
    """把目标函数换成累加**独占**耗时的包装器。返回卸载函数。

    独占 = 本函数的总耗时减去其中被包装的嵌套调用耗时（维护一条调用栈）。
    旧实现直接把总耗时记在父与子两个条目上，于是 `_resolution_boards` 13.9 s
    里包含了 `day_block_bootstrap` 7.4 s——按这张表做优化会重复计算收益，也会
    让各段占比之和超过 100%。这是测量工具的正确性缺陷，不是展示问题。
    """
    import weather_eval.evaluate as E
    import weather_eval.stats as S
    import weather_eval.report.diagnostics as D

    originals: list[tuple[object, str, object]] = []
    stack: list[dict] = []

    def wrap(mod, name, prefix=""):
        fn = getattr(mod, name, None)
        if fn is None:
            return
        key = prefix + name
        originals.append((mod, name, fn))

        def probe(*a, _fn=fn, _key=key, **kw):
            parent = stack[-1] if stack else None
            slot = acc.setdefault(_key, {"calls": 0, "seconds": 0.0})
            slot["calls"] += 1
            stack.append(slot)
            t0 = time.perf_counter()
            try:
                return _fn(*a, **kw)
            finally:
                dt = time.perf_counter() - t0
                stack.pop()
                slot["seconds"] += dt
                if parent is not None:
                    parent["seconds"] -= dt

        setattr(mod, name, probe)

    for n in EVAL_TARGETS:
        wrap(E, n)
    for n in STATS_TARGETS:
        wrap(S, n, "stats.")
    for n in DIAG_TARGETS:
        wrap(D, n, "diagnostics.")

    def uninstall():
        for mod, name, fn in originals:
            setattr(mod, name, fn)

    return uninstall


def _resolve_window(cfg, window_file: str | None):
    """评估窗口。给了冻结窗口文件就严格照它——对拍必须在同一天数轴上进行。"""
    from weather_eval.__main__ import _live_window
    from weather_eval.timeutil import parse_iso

    if window_file:
        w = json.loads(Path(window_file).read_text(encoding="utf-8"))
        return (parse_iso(w["start_dt"]), parse_iso(w["end_dt"]),
                w.get("period_label") or "frozen")
    return _live_window(cfg)


def run_once(window_file: str | None = None, write_report: bool = False) -> dict:
    from weather_eval.config import load_config
    from weather_eval.evaluate import build_report

    os.environ.setdefault("TZ", "Asia/Shanghai")
    cfg = load_config(None)
    start, end, label = _resolve_window(cfg, window_file)

    acc: dict = {}
    uninstall = _install_probes(acc)

    t0 = time.perf_counter()
    report = build_report(cfg.station_ids, cfg.models, cfg.eval, start, end,
                          period_label=label)
    total = time.perf_counter() - t0
    uninstall()

    segs = sorted(
        ({"name": k, "calls": v["calls"], "seconds": round(v["seconds"], 3),
          "pct": round(100 * v["seconds"] / total, 1) if total else 0.0}
         for k, v in acc.items()),
        key=lambda x: -x["seconds"])
    out = {
        "total_seconds": round(total, 2),
        "peak_rss_mb": _peak_rss_mb(),
        "window": {"start": start.strftime("%Y-%m-%d %H:%M"),
                   "end": end.strftime("%Y-%m-%d %H:%M"),
                   "period_label": label},
        "scale": {
            "stations": len(cfg.station_ids),
            "models": len(cfg.models),
            "bootstrap_days": report["meta"].get("bootstrap_days"),
        },
        "segments": segs,
        "code_sha": _code_sha(),
        "python": platform.python_version(),
    }
    if write_report:
        out["report_bytes"] = len(json.dumps(report, ensure_ascii=False))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--window-file", default=None,
                    help="冻结的评估窗口 JSON（与 golden baseline 同轴对拍时用）")
    ap.add_argument("--repeat", type=int, default=1, help="重复次数，取中位数")
    ap.add_argument("--json", default=None,
                    help="输出 JSON 的路径；'-' 表示只打印不落盘")
    ap.add_argument("--tag", default="", help="落盘文件名的后缀标记（如 after-p1）")
    args = ap.parse_args(argv)

    runs = [run_once(args.window_file) for _ in range(max(1, args.repeat))]
    best = (sorted(runs, key=lambda r: r["total_seconds"])[len(runs) // 2]
            if len(runs) > 1 else runs[0])
    if len(runs) > 1:
        best["runs"] = [r["total_seconds"] for r in runs]
        best["peak_rss_mb"] = max(r["peak_rss_mb"] for r in runs)

    payload = json.dumps(best, ensure_ascii=False, indent=2)
    if args.json == "-":
        print(payload)
        return 0
    if args.json:
        Path(args.json).write_text(payload, encoding="utf-8")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"bench-{time.strftime('%Y%m%d-%H%M%S')}"
    if args.tag:
        name += f"-{args.tag}"
    path = OUT_DIR / f"{name}.json"
    path.write_text(payload, encoding="utf-8")

    print(f"{'阶段':<32}{'调用':>8}{'秒':>10}{'占比':>8}")
    for s in best["segments"][:14]:
        print(f"{s['name']:<32}{s['calls']:>8}{s['seconds']:>10.2f}{s['pct']:>7.1f}%")
    print(f"\n总计 {best['total_seconds']:.1f}s   峰值 RSS {best['peak_rss_mb']:.0f} MB"
          f"   code={best['code_sha']}   python={best['python']}")
    print(f"基准已写入 {path.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""冻结 Golden Master 基线（Phase 0 的第一个动作，不可跳过、不可后置）。

产物：
  tests/baseline/window.json           冻结的 start_dt / end_dt / eval_days 列表
  tests/baseline/report_baseline.json  报告对外展示的全部数字
  tests/baseline/env.json              code_sha / 依赖版本 / 种子 / Python 版本

**为什么必须冻结 eval_days 列表**（这条最容易被误解，也是最容易让后续对拍
失败的地方）：`build_day_stat_tables` 的天数轴取 hourly 与 daily 记录的天并集，
`day_block_weights(runs, n_days, block_days, seed)` 的形状依赖 `n_days`。新增
一天会改变整个权重矩阵，bootstrap 的 CI 值随之变化——**这是正确行为，不是
口径漂移**。若只在基线里存 start/end 而不存天数轴，增量路径与全量路径会在
不同轴长上对拍，被误判成口径变化。

**为什么基线是"对外展示的数字"而不是"完整 report dict"**：完整 dict 含逐样本
明细（timeseries / heatmap / per_station），那是原料不是结论。基线要钉的是
"报告里印出来的每一个数"——I8 的口径逐位不变指的是读者的眼睛能看到的东西。
内部浮点允许 1e-9 相对容差，但**排名零容差**。

用法：
    python scripts/make_baseline.py                # 用默认窗口（数据起点 → 最新观测）
    python scripts/make_baseline.py --end 2026-10-03T23:00
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BASELINE_DIR = PROJECT_ROOT / "tests" / "baseline"

# 对外展示字段：这些值出现在 report.html 的表格/卡片上，逐位冻结。
# 不含内部中间量（它们有 1e-9 容差，且不在页面上）。
BOARD_FIELDS = (
    "model", "score", "temp_score", "precip_score", "ci90", "champion_pct",
    "sig_vs_top", "qualified", "comparable", "n", "n_precip", "n_eff",
    "n_eff_rain", "acc2", "rmse", "ts", "ets", "lead_days", "n_days",
    "track_gap", "hourly_score", "daily_score",
)
SCORECARD_FIELDS = ("n", "n_eff", "rmse", "mae", "mbe", "acc1", "acc2", "r",
                    "slope", "acc", "pod", "far", "ts", "ets", "bias")


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _round(v):
    """只做 NaN/inf → None 的规整，**不做任何舍入**。

    为什么不能为了文件好看去舍入（实测教训）：第一版把浮点 round 到 6 位，
    于是 `scorecard[..].r` 的基线值 0.885263 与实测值 0.88526254… 相差 4.6e-7，
    超过 1e-9 容差——对拍立刻红了 20 行，而那份报告其实**逐位没变**。
    基线的作用是把"数字有没有变"这个问题答准；把精度先砍一刀再问，答案只能是假的。
    """
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(v, list):
        return [_round(x) for x in v]
    return v


def _pick(d: dict, fields):
    return {k: _round(d.get(k)) for k in fields if k in d}


def build(end_override: str | None = None) -> tuple[dict, dict, dict]:
    import os
    os.environ.setdefault("TZ", "Asia/Shanghai")
    from weather_eval.config import load_config
    from weather_eval.evaluate import build_report
    from weather_eval.timeutil import parse_iso

    cfg = load_config(None)
    if end_override:
        end = parse_iso(end_override)
        start = parse_iso(f"{end:%Y-%m}-01T00:00")
        # 起点仍取最早可用月：总榜是跨月累计窗口（见 __main__._live_window）
        from weather_eval.storage import available_months
        months = sorted(available_months())
        if months:
            start = parse_iso(f"{months[0]}-01T00:00")
    else:
        from weather_eval.__main__ import _live_window
        start, end, _ = _live_window(cfg)

    label = f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d}"
    import resource
    import time as _time
    _t0 = _time.perf_counter()
    report = build_report(cfg.station_ids, cfg.models, cfg.eval, start, end,
                          period_label=label)
    measured = {"seconds": round(_time.perf_counter() - _t0, 2),
                "peak_rss_mb": round(resource.getrusage(
                    resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)}

    boards = {}
    for key, rows in sorted(report["leaderboards"].items()):
        boards[key] = [_pick(r, BOARD_FIELDS) for r in rows]

    scorecard = {}
    for m, blk in sorted(report["scorecard"].items()):
        scorecard[m] = {slot: _pick(vals, SCORECARD_FIELDS)
                        for slot, vals in sorted(blk.items())}

    # 评分卡关键指标 + meta 全字段（meta 里含门槛、阈值、bootstrap 口径披露）
    meta = json.loads(json.dumps(report["meta"], ensure_ascii=False, sort_keys=True,
                                 default=str))
    baseline = {
        "period_label": label,
        "boards": boards,
        "scorecard": scorecard,
        "meta": meta,
        "coverage": report.get("coverage"),
    }

    # 天数轴：与 build_day_stat_tables 同源（hourly 的 valid 日 ∪ daily 的 valid_day）
    hourly_days = sorted({k.split(":", 1)[-1] for k in boards if k.startswith("hourly:")})
    window = {
        "start_dt": start.strftime("%Y-%m-%dT%H:%M"),
        "end_dt": end.strftime("%Y-%m-%dT%H:%M"),
        "period_label": label,
        "stations": list(cfg.station_ids),
        "models": list(cfg.models),
        "hourly_board_days": len(hourly_days),
        # eval_days 需要在 collect 之后才算得出，这里从 bootstrap 的披露里取
        "n_boot_days": meta.get("bootstrap_days"),
        "note": ("eval_days 由 hourly/daily 记录的天并集决定，"
                 "day_block_weights 的形状依赖 n_days；对拍必须同轴"),
    }

    deps = {}
    for mod in ("numpy", "pandas", "scipy", "pint", "cyeva"):
        try:
            deps[mod] = str(getattr(__import__(mod), "__version__", "unknown"))
        except Exception:                       # noqa: BLE001
            deps[mod] = "unavailable"
    env = {
        "code_sha": _git("rev-parse", "HEAD") or "unknown",
        "code_short": _git("rev-parse", "--short=12", "HEAD") or "unknown",
        "python": platform.python_version(),
        "deps": deps,
        "bootstrap_seed": report["meta"].get("bootstrap_runs") and 20260906 or 20260906,
    }
    # 性能预算（I12）：初值 = 本次实测 × 倍率。它是**棘轮**，只许随优化单向收紧，
    # 绝不允许因为某次机器慢就放宽——那是把"变慢"写进契约。
    #
    # 耗时与内存给**不同的倍率**，因为它们的不确定性来源不同：
    #   耗时 ×2.0 —— 同一套测试在三个环境测得 49.6s / 4min / 35min（§11.12），
    #     且 pytest 全量跑时进程内已有 533 个测试的残留状态，实测比独立跑慢约 1.7×。
    #     给 2× 仍能有效抓住"某次改动让耗时翻倍"这类回归。
    #   内存 ×1.5 —— 峰值 RSS 与磁盘/缓存无关，主要随数据规模变化，是更硬的约束
    #     （13 个月数据下 ~20 GB 会直接 OOM），故余量给得更紧。
    budget = {
        "max_seconds": round(measured["seconds"] * 2.0, 1),
        "max_peak_rss_mb": round(measured["peak_rss_mb"] * 1.5, 0),
        "measured_at_freeze": measured,
        "note": ("预算是棘轮：随优化推进单向收紧。耗时给 2×（跨机差异与全量 pytest 的"
                 "进程内竞争都很大），峰值 RSS 给 1.5×（与机器关系较小，是更硬的约束："
                 "13 个月数据下 ~20 GB 会 OOM）"),
    }
    return baseline, window, env, budget


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--end", default=None, help="窗口终点 ISO（默认取最新观测整点）")
    ap.add_argument("--out", default=str(BASELINE_DIR), help="输出目录")
    ap.add_argument("--budget-multiplier", type=float, default=1.5,
                    help="性能预算相对实测值的倍率（默认 1.5）")
    args = ap.parse_args(argv)

    baseline, window, env, budget = build(args.end)
    if args.budget_multiplier != 1.5:
        budget["max_seconds"] = round(budget["measured_at_freeze"]["seconds"]
                                      * args.budget_multiplier, 1)
        budget["max_peak_rss_mb"] = round(budget["measured_at_freeze"]["peak_rss_mb"]
                                          * args.budget_multiplier, 0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, obj in (("report_baseline.json", baseline),
                      ("window.json", window),
                      ("env.json", env),
                      ("perf_budget.json", budget)):
        (out / name).write_text(
            json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True),
            encoding="utf-8")
        size = (out / name).stat().st_size
        print(f"已写入 {name}（{size/1024:.0f} KB）")
    print(f"窗口 {window['start_dt']} ~ {window['end_dt']}，"
          f"榜单 {len(baseline['boards'])} 张")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

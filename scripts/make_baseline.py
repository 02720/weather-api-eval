#!/usr/bin/env python3
"""冻结 Golden Master 基线（Phase 0 的第一个动作，不可跳过、不可后置）。

产物：
  tests/baseline/window.json           冻结的 start_dt / end_dt / eval_days 列表
  tests/baseline/report_baseline.json  报告对外展示的全部数字 + 冻结输入的指纹
  tests/baseline/env.json              code_sha / 依赖版本 / 种子 / Python 版本

**基线只能冻结"已沉降"的窗口**（2026-10-04 补，第一性原理）：
Golden Master 的全部价值来自"同输入必然同输出"。而本项目的输入并不都是不变
量——预报快照写一次就不再改（`save_forecast_snapshot` 幂等跳过 + Merkle 根可
验），**观测却是活的**：第三方源修正错报是常态，`obs/cma_data.py` 每轮回看 26
小时并对其中 6 小时强制重抓，`save_obs` 就地改值（旧值挂进 `revisions` 留痕）。

第一版基线把窗口终点取在"最新观测整点"（`_live_window` 的口径），也就是**回改
窗口的正中央**。后果是基线从冻结那一刻起就开始腐烂：下一轮抓取回改窗口内的任
何一个整点，对拍立刻红一片，而红的原因既不在代码里、也不在口径里。实测
（2026-10-04）：4 站各 1 个整点（2026-10-03T08:00）气温被回改约 +1℃，1283 个
展示字段位移、8 处名次互换（显示分打平后顺序翻转）。

因此窗口终点改为 **最新观测 − SETTLE_HOURS**，并对齐到完整自然日：超过回改深度
后输入不再移动，基线才真的具备"重跑必得同一结果"的性质。冻结的输入指纹
（`inputs.obs`）把这件事变成可核验的：对拍变红时先看指纹——指纹变了是**数据
变了**，指纹没变才是**口径变了**，两者的处置方式正好相反。

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
    python scripts/make_baseline.py                # 数据起点 → 已沉降的窗口终点
    python scripts/make_baseline.py --end 2026-10-01T23:00   # 手工指定（未沉降会告警）
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
from datetime import timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

BASELINE_DIR = PROJECT_ROOT / "tests" / "baseline"

# 沉降期：窗口终点至少要比"最新观测"早这么久，窗口内的观测才不再可能被回改。
# 下界是 `obs/cma_data.py` 的 DEFAULT_LOOKBACK_HOURS（26h，每轮回看深度）；
# 48h = 26h + 两轮抓取周期 + 周末/上游故障余量。抓取每天 3 次（7~8h 一轮），
# 48h 意味着窗口内的每个整点都已被 6 轮以上抓取反复确认过。
SETTLE_HOURS = 48

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


def settled_end(newest, settle_hours: int):
    """已沉降的窗口终点（纯函数）：`newest` 往前推 `settle_hours`，再对齐整点日界。

    为什么对齐整点日界（23:00）而不是直接取整点：日榜的口径是"自然日聚合"，
    窗口切在半天中间会让最后一天变成半截样本——那份基线钉住的是一个现实中不存
    在的时间切片，日后没人能复现出同一个"当日"。基线要的是稳定，不是新鲜。

    `newest` 为 None（无观测）时返回 None。纯函数是为了让这条规则能在 PR 快层
    里被秒级覆盖，而不是只能靠 4 分钟的 golden 全量构建来验。
    """
    if newest is None:
        return None
    cutoff = newest - timedelta(hours=settle_hours)
    end = cutoff.replace(hour=23, minute=0, second=0, microsecond=0)
    if end > cutoff:                      # 当天 23:00 还没沉降 → 退到前一天
        end -= timedelta(days=1)
    return end


def build(end_override: str | None = None,
          settle_hours: int = SETTLE_HOURS) -> tuple[dict, dict, dict]:
    import os
    os.environ.setdefault("TZ", "Asia/Shanghai")
    from weather_eval.config import load_config
    from weather_eval.evaluate import build_report
    from weather_eval.obs.cma_data import DEFAULT_LOOKBACK_HOURS
    from weather_eval.provenance import obs_input_digest
    from weather_eval.timeutil import parse_iso

    cfg = load_config(None)
    from weather_eval.provenance import newest_obs_hour
    newest_obs = newest_obs_hour(cfg.station_ids)
    settled_end_dt = settled_end(newest_obs, settle_hours)

    if end_override:
        end = parse_iso(end_override)
        # 起点仍取最早可用月：总榜是跨月累计窗口（见 __main__._live_window）
        from weather_eval.storage import available_months
        months = sorted(available_months())
        start = parse_iso(f"{months[0]}-01T00:00") if months \
            else parse_iso(f"{end:%Y-%m}-01T00:00")
        settled = bool(settled_end_dt is not None and end <= settled_end_dt)
        if not settled:
            print(f"⚠️  警告：--end {end_override} 落在观测回改窗口"
                  f"（最新观测 {newest_obs} − {DEFAULT_LOOKBACK_HOURS}h）之内，"
                  f"已沉降终点为 {settled_end_dt}。\n"
                  f"    这份基线会随下一次观测回改而腐烂，对拍将红。\n"
                  f"    仅在明知代价时使用（例如复现某次历史对拍）。", file=sys.stderr)
    else:
        if settled_end_dt is None:
            raise SystemExit("data/obs 为空：没有可冻结的观测。先跑一次 fetch-obs。")
        if settle_hours < DEFAULT_LOOKBACK_HOURS:
            raise SystemExit(
                f"沉降期 {settle_hours}h 小于观测回看深度 "
                f"{DEFAULT_LOOKBACK_HOURS}h（obs/cma_data.py）——冻结的窗口"
                f"仍可被回改，这份基线从写下那一刻起就是假的。")
        start, end = None, settled_end_dt
        from weather_eval.storage import available_months
        months = sorted(available_months())
        if months:
            start = parse_iso(f"{months[0]}-01T00:00")
        settled = True
    assert start is not None

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
        # 冻结输入的指纹：对拍变红时，先分清是输入动了还是口径动了
        "inputs": {
            "obs": obs_input_digest(cfg.station_ids, start, end),
            "settled": settled,
            "settle_hours": int(settle_hours),
        },
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
        # 沉降声明：窗口内的输入从此不再移动，是"重跑必得同一结果"的前提。
        # tests/test_baseline.py::test_window_is_settled 会拿它对着数据核验，
        # 所以它不是注释，而是一条可失败的断言。
        "settle_hours": int(settle_hours),
        "settled": settled,
        "newest_obs_at_freeze": newest_obs.strftime("%Y-%m-%dT%H:%M") if newest_obs else None,
        "note": ("eval_days 由 hourly/daily 记录的天并集决定，"
                 "day_block_weights 的形状依赖 n_days；对拍必须同轴。"
                 "end_dt 取的是已沉降终点（最新观测 − settle_hours，再对齐整点日界）："
                 "观测会被后续轮次回改，窗口压在回改窗口里，基线从写下起就会腐烂"),
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
    # 棘轮的方向由 _apply_ratchet 守住（main 里最后一步调用），不靠人记得。
    return baseline, window, env, budget


def _apply_ratchet(budget: dict, out_dir: Path) -> dict:
    """重冻结不得让性能预算变松（就地修改并返回 `budget`）。

    为什么需要代码来守：窗口换了（新数据变多、或为了沉降把窗口挪早）都可能让本次
    实测高于上一版，那是"换了更大的考题"而不是"变慢了"。若照抄实测 × 倍率，就等于
    用换题目的机会给自己提额，棘轮从此失效——而棘轮的全部价值就在于它是自动的。
    """
    prev = _load_budget(out_dir)
    if not prev:
        return budget
    clamped = []
    for key in ("max_seconds", "max_peak_rss_mb"):
        old, new = prev.get(key), budget.get(key)
        if isinstance(old, (int, float)) and isinstance(new, (int, float)) and new > old:
            budget[key] = old
            clamped.append(f"{key}: {new} → {old}")
    if clamped:
        budget["note"] += "｜本次重冻结沿用上一版更紧的预算（棘轮只许收紧）"
        print("棘轮：沿用上一版更紧的预算 " + "，".join(clamped), file=sys.stderr)
    return budget


def _load_budget(out_dir: Path) -> dict:
    """读上一版性能预算（不存在则返回 {}）。棘轮比较需要它。"""
    p = Path(out_dir) / "perf_budget.json"
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--end", default=None,
                    help="窗口终点 ISO（默认取**已沉降**终点：最新观测 − 沉降期）")
    ap.add_argument("--settle-hours", type=int, default=SETTLE_HOURS,
                    help=f"沉降期小时数（默认 {SETTLE_HOURS}；不得小于观测回看深度 26h）")
    ap.add_argument("--out", default=str(BASELINE_DIR), help="输出目录")
    ap.add_argument("--budget-multiplier", type=float, default=1.5,
                    help="性能预算相对实测值的倍率（默认 1.5）")
    args = ap.parse_args(argv)

    baseline, window, env, budget = build(args.end, args.settle_hours)
    if args.budget_multiplier != 1.5:
        budget["max_seconds"] = round(budget["measured_at_freeze"]["seconds"]
                                      * args.budget_multiplier, 1)
        budget["max_peak_rss_mb"] = round(budget["measured_at_freeze"]["peak_rss_mb"]
                                          * args.budget_multiplier, 0)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    budget = _apply_ratchet(budget, out)   # 永远最后一步：任何倍率都不得放松棘轮
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

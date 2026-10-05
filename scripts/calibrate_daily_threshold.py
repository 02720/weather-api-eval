"""一次性标定脚本：按天降水二分类阈值扫描（P0-3.3 的数据依据）。

在真实存档上，用与 evaluate.collect 完全相同的配对口径取按天（日累计）样本，
对每个候选阈值计算各源的 TS/ETS/BIAS，观察：
  1) 实测雨日基率（阈值↑基率↓，ETS 的基率校正压力随之变化）；
  2) 各源 ETS 的区分度（max-min 跨度）——跨度大说明该阈值下评分能分辨技巧；
  3) 超报倍率（预报雨日率/实测雨日率）是否回到 1 附近。
结论写进 README 与 evaluate.py docstring。运行：
  PYTHONPATH=src python scripts/calibrate_daily_threshold.py
  PYTHONPATH=src python scripts/calibrate_daily_threshold.py --out docs/calibrate_daily_threshold.md
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict

sys.path.insert(0, "src")

import numpy as np

from weather_eval.config import load_config
from weather_eval.evaluate import collect
from weather_eval.timeutil import now_beijing

THRESHOLDS = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)


def binarize_stats(obs, fcst, thr):
    ob = np.round(np.asarray(obs, dtype=float), 2) >= thr
    fb = np.round(np.asarray(fcst, dtype=float), 2) >= thr
    h = int((ob & fb).sum())
    f = int((~ob & fb).sum())
    m = int((ob & ~fb).sum())
    c = int((~ob & ~fb).sum())
    return h, f, m, c


def ts_ets(h, f, m, c):
    n = h + f + m + c
    if n == 0:
        return None, None
    ts = h / (h + f + m) if (h + f + m) else None
    href = (h + m) * (h + f) / n if n else 0.0
    den = h + f + m - href
    ets = (h - href) / den if den > 0 else None
    return ts, ets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="把本次扫描曲线留档到这个 Markdown 文件")
    args = ap.parse_args()
    say = print
    lines: list[str] = []
    if args.out:
        def say(*a):                      # noqa: D401 —— 打印同时收集，留档用
            line = " ".join(str(x) for x in a)
            lines.append(line)
            print(line)
    cfg = load_config()
    end = now_beijing().replace(minute=0, second=0, microsecond=0)
    start = end.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    hourly, daily = collect(cfg.station_ids, cfg.models, start, end,
                            cfg.eval["hourly_lead_days"], cfg.eval["daily_max_offset_days"],
                            cfg.eval.get("daily_min_hours", 20),
                            bool(cfg.eval.get("daily_source_fallback", True)))

    by_model = defaultdict(lambda: ([], []))
    for r in daily:
        if r["rain_obs"] is not None and r["rain_fcst"] is not None:
            by_model[r["model"]][0].append(r["rain_obs"])
            by_model[r["model"]][1].append(r["rain_fcst"])

    all_obs = np.concatenate([v[0] for v in by_model.values()]) if by_model else np.array([])
    say(f"按天样本（日累计配对）总数: {len(all_obs)}")
    say(f"窗口: {start} ~ {end}")
    say()
    for thr in THRESHOLDS:
        wet = float((np.round(all_obs, 2) >= thr).mean()) * 100
        rows = []
        for m, (o, f) in sorted(by_model.items()):
            h, f_, m_, c = binarize_stats(o, f, thr)
            ts, ets = ts_ets(h, f_, m_, c)
            obs_rate = 100 * (h + m_) / (h + f_ + m_ + c) if (h + f_ + m_ + c) else 0
            fcst_rate = 100 * (h + f_) / (h + f_ + m_ + c) if (h + f_ + m_ + c) else 0
            rows.append((m, ts, ets, obs_rate, fcst_rate))
        ets_vals = [e for _, _, e, _, _ in rows if e is not None]
        span = (max(ets_vals) - min(ets_vals)) if len(ets_vals) >= 2 else 0.0
        say(f"== 阈值 {thr} mm/日：实测雨日率 {wet:.1f}%，ETS 跨度 {span:.3f} ==")
        for m, ts, ets, orate, frate in rows:
            ratio = (frate / orate) if orate else float("inf")
            ets_s = f"{ets:.3f}" if ets is not None else "  —  "
            ts_s = f"{ts:.3f}" if ts is not None else "  —  "
            say(f"  {m:<32s} TS={ts_s} ETS={ets_s} 实测雨日 {orate:5.1f}% 预报雨日 {frate:5.1f}% 超报×{ratio:.2f}")
        say()

    if args.out:
        head = [
            "# 按天降水阈值扫描（留档）",
            "",
            "本文件由 `scripts/calibrate_daily_threshold.py --out docs/calibrate_daily_threshold.md` 生成，",
            "是 README/配置里 `rain_daily_threshold_mm = 1.0` 结论的原始扫描曲线。",
            "**窗口是运行当月的自然月**（数据持续增长，重跑得到的窗口会不同；",
            "结论的引用请以配置注释里的定标窗口为准）。",
            "",
            "```",
        ]
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("\n".join(head + lines + ["```", ""]))
        print(f"已留档 {args.out}")


if __name__ == "__main__":
    main()

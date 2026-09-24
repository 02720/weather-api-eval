"""一次性标定脚本：小时榜降水二分类阈值扫描（2026-09 双轨道重构的数据依据）。

问题：小时榜要回答"预报说某小时下雨，到底下没下"，自然的业务阈值是 ≥0.1mm/h
（国内业务"有降水"的定义）。但在真实存档上，这个口径下的评分测的其实不是技巧：
数值模式普遍每小时产生微量降水（drizzle bias），实测各源报雨频率是实况的 2.7
倍，ETS 全部压在 0.02~0.34 的窄带里，分数主要在给"谁更少毛毛雨"排序。

本脚本在真实存档上扫描候选阈值，观察三件事：
  1) 预报/实况基率比（BIAS 中位数）——回到 1 附近，才说明评分不再被系统性超报主导；
  2) 各源 ETS 的中位数与区分度（max-min 跨度）——跨度大说明该阈值下能分辨技巧；
  3) 换阈值时各家名次是否剧烈洗牌——洗牌剧烈说明这个分数高度依赖口径选择。
结论写进 config.py 的 rain_hourly_threshold_mm 与 README。运行：
  PYTHONPATH=src python scripts/calibrate_hourly_threshold.py
"""
from __future__ import annotations

import statistics as st
import sys
from collections import defaultdict
from datetime import datetime, timedelta

sys.path.insert(0, "src")

import numpy as np

from weather_eval.config import load_config
from weather_eval.evaluate import collect
from weather_eval.timeutil import now_beijing

THRESHOLDS = (0.1, 0.2, 0.5, 1.0, 2.0)


def binarize_stats(obs, fcst, thr):
    ob = np.asarray(obs, dtype=float) >= thr
    fb = np.asarray(fcst, dtype=float) >= thr
    h = int((ob & fb).sum())
    f = int((~ob & fb).sum())
    m = int((ob & ~fb).sum())
    c = int((~ob & ~fb).sum())
    return h, f, m, c


def metrics(h, f, m, c):
    n = h + f + m + c
    if n == 0:
        return {}
    ts = h / (h + f + m) if (h + f + m) else None
    bias = (h + f) / (h + m) if (h + m) else None
    href = (h + m) * (h + f) / n
    ets = (h - href) / ((h + f + m) - href) if (h + f + m - href) else None
    return {"ts": ts, "ets": ets, "bias": bias}


def main():
    cfg = load_config()
    ev = cfg.eval
    end = now_beijing().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=30)
    hourly, _daily = collect(cfg.station_ids, cfg.models, start, end,
                             ev["hourly_lead_days"], ev["daily_max_offset_days"],
                             ev["daily_min_hours"], ev["daily_source_fallback"])
    # 按模型汇齐全部天桶（每家一条完整的小时样本序列）
    flat = defaultdict(lambda: ([], []))
    for r in hourly:
        if (r["bucket"] >= 1 and r["rain_obs"] is not None
                and r["rain_fcst"] is not None):
            o, f = flat[r["model"]]
            o.append(r["rain_obs"])
            f.append(r["rain_fcst"])
    used = [(m, np.asarray(o), np.asarray(f)) for m, (o, f) in flat.items()
            if len(o) >= 200]
    print(f"窗口 {start:%Y-%m-%d} ~ {end:%Y-%m-%d}，参与标定的源 {len(used)} 家")
    print(f"{'thr':>6} {'实况基率%':>9} {'预报基率%':>9} {'BIAS中位':>8} "
          f"{'ETS中位':>8} {'ETS跨度':>8} {'TS中位':>7}")
    rows = {}
    for thr in THRESHOLDS:
        ets_l, ts_l, bias_l, ob_l, fb_l = [], [], [], [], []
        per_model = {}
        for m, o, f in used:
            h, fa, mi, c = binarize_stats(o, f, thr)
            bm = metrics(h, fa, mi, c)
            if not bm or bm["ets"] is None:
                continue
            ets_l.append(bm["ets"])
            ts_l.append(bm["ts"])
            bias_l.append(bm["bias"])
            ob_l.append(100.0 * (o >= thr).mean())
            fb_l.append(100.0 * (f >= thr).mean())
            per_model[m] = bm["ets"]
        rows[thr] = per_model
        print(f"{thr:>6.1f} {st.median(ob_l):>9.2f} {st.median(fb_l):>9.2f} "
              f"{st.median(bias_l):>8.2f} {st.median(ets_l):>8.3f} "
              f"{max(ets_l) - min(ets_l):>8.3f} {st.median(ts_l):>7.3f}")
    # 名次稳定性：阈值变化时各源 ETS 排名的平均绝对变动
    base = rows[THRESHOLDS[0]]
    print("\n各源 ETS 名次随阈值的平均绝对变动（越小越稳健）：")
    for thr in THRESHOLDS[1:]:
        common = set(base) & set(rows[thr])
        if len(common) < 5:
            continue
        order = lambda parent: {m: i for i, m in enumerate(
            sorted(common, key=lambda x: -parent[x]))}
        r0, r1 = order(base), order(rows[thr])
        shift = st.mean(abs(r0[m] - r1[m]) for m in common)
        print(f"  {THRESHOLDS[0]:>4}mm → {thr:>4}mm ：平均名次变动 {shift:.2f} 位"
              f"（参与 {len(common)} 家）")


if __name__ == "__main__":
    main()

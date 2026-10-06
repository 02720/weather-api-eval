"""降水"雨量"维度怎么入分：连续雨量指标归一化口径的离线扫描。

背景：雨量 rmse/mae/mbe 是降水评分轨**唯一一块真正的新信息**（冗余审计显示它们对
晴雨四项的 1−R² ≈ 0.93~0.99，跨源相关 ≤0.2）。但它们的量纲随桶/轨剧烈变化
（逐小时中位跨源 sd 0.07mm、按天 1.35mm，相差约 20 倍），直接套一个全局换算
斜率 λ 会让"按天桶"独占话语权。

本脚本在真实存档上逐格算出候选口径，并按**真正进入分数**的形态评价：
  子分 = clip(换算(口径, λ), 0, 100)，λ 按"桶内跨源 sd = 8"在两轨池化上反解。
比较三件事：
  1) 子分离散度的**跨轨可比性**（小时轨 / 日轨 的子分 sd 之比越接近 1 越好：
     单一 λ 才标得准，否则这项指标只在一条轨上真有话语权）；
  2) 子分离散度的**跨桶稳定性**（P90/P10，越接近 1 越好）；
  3) 与现有晴雨子分的相关性（高 = 白加，低 = 真新信息）。

运行：
  PYTHONPATH=src .venv312/bin/python scripts/audit_amount_norm.py --days 60 \
      [--from-json .work/amount_cells.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import timedelta

sys.path.insert(0, "src")

import numpy as np

MIN_MODELS = 3
SD_TARGET = 8.0


def cell_stats(o: np.ndarray, f: np.ndarray) -> dict:
    m = ~np.isnan(o) & ~np.isnan(f)
    o, f = o[m], f[m]
    n = o.size
    if n == 0:
        return {}
    d = f - o
    mean_o = float(o.mean())
    return {
        "n": n,
        "mean_obs": mean_o,
        "mean_fcst": float(f.mean()),
        "sd_obs": float(o.std()),
        "mae": float(np.abs(d).mean()),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "mbe": float(d.mean()),
        "mae_ref": float(np.abs(o - mean_o).mean()),
    }


def candidates(s: dict) -> dict:
    """每个候选口径 → (值, 换算族)。族与生产换算同式：
    dev: 100 − v×λ · pct: v×λ · log: 100 − |v|×λ（v 已是 log₂ 刻度）"""
    out = {}
    mo, mf, sd = s["mean_obs"], s["mean_fcst"], s["sd_obs"]
    out["mae_raw"] = (s["mae"], "dev")
    out["rmse_raw"] = (s["rmse"], "dev")
    out["|mbe|_raw"] = (abs(s["mbe"]), "dev")
    out["skill_mae"] = ((1 - s["mae"] / s["mae_ref"]) if s["mae_ref"] > 0 else None, "pct")
    out["skill_rmse"] = ((1 - s["rmse"] / sd) if sd > 0 else None, "pct")
    out["nmae"] = ((s["mae"] / mo) if mo > 1e-6 else None, "dev")
    out["nrmse"] = ((s["rmse"] / mo) if mo > 1e-6 else None, "dev")
    out["log_ratio"] = ((math.log2(mf / mo) if (mo > 1e-6 and mf > 1e-6) else None), "log")
    out["mbe_norm"] = ((s["mbe"] / sd) if sd > 0 else None, "dev")
    return out


def conv(family: str, lam: float, v: float) -> float:
    if family == "dev":
        return 100 - v * lam
    if family == "pct":
        return v * lam
    return 100 - abs(v) * lam


def bucket_sds(cells: dict, key: str) -> dict[str, list[float]]:
    """{track: [逐桶跨源 sd]}（原始口径，用于反解 λ）。"""
    by: dict[str, list[float]] = defaultdict(list)
    per: dict[tuple, list[float]] = defaultdict(list)
    for (trk, b, _m), c in cells.items():
        v = c["cand"].get(key)
        if v is None or v[0] is None or not math.isfinite(v[0]):
            continue
        per[(trk, b)].append(v[0])
    for (trk, b), vs in per.items():
        if len(vs) >= MIN_MODELS:
            by[trk].append(float(np.std(vs, ddof=1)))
    return by


def sub_sds(cells: dict, key: str, lam: float) -> dict[str, list[float]]:
    """校准后的**子分**逐桶跨源 sd（这才是真正进入分数的离散度）。"""
    fam = None
    for c in cells.values():
        t = c["cand"].get(key)
        if t and t[0] is not None:
            fam = t[1]
            break
    per: dict[tuple, list[float]] = defaultdict(list)
    for (trk, b, _m), c in cells.items():
        t = c["cand"].get(key)
        if t is None or t[0] is None or not math.isfinite(t[0]):
            continue
        per[(trk, b)].append(min(100.0, max(0.0, conv(fam, lam, t[0]))))
    out: dict[str, list[float]] = defaultdict(list)
    for (trk, b), vs in per.items():
        if len(vs) >= MIN_MODELS:
            out[trk].append(float(np.std(vs, ddof=1)))
    return out


def calibrate(cells: dict, key: str) -> float | None:
    """定点迭代：子分的两轨池化桶内跨源 sd = 8。"""
    fam = None
    for c in cells.values():
        t = c["cand"].get(key)
        if t and t[0] is not None:
            fam = t[1]
            break
    if fam is None:
        return None

    def pooled_sd(lam):
        per: dict[tuple, list[float]] = defaultdict(list)
        for (trk, b, _m), c in cells.items():
            t = c["cand"].get(key)
            if t is None or t[0] is None or not math.isfinite(t[0]):
                continue
            per[(trk, b)].append(min(100.0, max(0.0, conv(fam, lam, t[0]))))
        devs = []
        for vs in per.values():
            if len(vs) >= MIN_MODELS:
                a = np.asarray(vs)
                devs.extend((a - a.mean()).tolist())
        return float(np.std(devs, ddof=1)) if len(devs) > 1 else 0.0

    lam = SD_TARGET / (pooled_sd(1.0) or 1e-9)
    for _ in range(200):
        s = pooled_sd(lam)
        if s < 1e-9:
            break
        if abs(s - SD_TARGET) < 1e-5:
            break
        lam *= SD_TARGET / s
    return lam


def report(cells: dict, precip_sub: dict) -> None:
    print(f"\n{'口径':12s} {'λ*':>9s} {'小时sd':>8s} {'日sd':>8s} {'日/小时':>7s} "
          f"{'P90/P10(池化)':>13s} {'与晴雨子分|ρ|':>12s}")
    keys = list(next(iter(cells.values()))["cand"])
    for key in keys:
        lam = calibrate(cells, key)
        if not lam:
            print(f"{key:12s}  -- 不可用 --")
            continue
        sd = sub_sds(cells, key, lam)
        hs, ds = sd.get("hourly", []), sd.get("daily", [])
        alls = hs + ds
        p10, p90 = np.percentile(alls, [10, 90])
        # 与现有晴雨子分的相关（桶内中心化、池化）
        xs, ys = [], []
        per: dict[tuple, list] = defaultdict(list)
        for (trk, b, m), c in cells.items():
            t = c["cand"].get(key)
            if t is None or t[0] is None or not math.isfinite(t[0]):
                continue
            ps = precip_sub.get((trk, b, m))
            if ps is None:
                continue
            per[(trk, b)].append((min(100.0, max(0.0, conv(t[1], lam, t[0]))), ps))
        for vs in per.values():
            if len(vs) >= MIN_MODELS:
                a = np.asarray([x for x, _ in vs])
                b = np.asarray([y for _, y in vs])
                if a.std() > 0 and b.std() > 0:
                    xs.extend((a - a.mean()).tolist())
                    ys.extend((b - b.mean()).tolist())
        rho = abs(float(np.corrcoef(xs, ys)[0, 1])) if len(xs) > 10 else float("nan")
        print(f"{key:12s} {lam:9.4f} {np.median(hs):8.2f} {np.median(ds):8.2f} "
              f"{(np.median(ds) / np.median(hs) if hs else float('nan')):7.2f} "
              f"{(p90 / p10 if p10 > 0 else float('inf')):13.1f} {rho:12.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--from-json", default=None)
    ap.add_argument("--track-sources", default=".work/track_sources_60d.json")
    ap.add_argument("--out", default=".work/amount_cells.json")
    args = ap.parse_args()

    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        cells = {}
        for trk in ("hourly", "daily"):
            for k, v in (payload.get(trk) or {}).items():
                b, m = k.split("|", 1)
                cells[(trk, b, m)] = {"stats": v["stats"],
                                      "cand": {kk: (vv[0], vv[1]) if isinstance(vv, list) else vv
                                               for kk, vv in v["cand"].items()}}
    else:
        from weather_eval.config import load_config
        from weather_eval.evaluate import collect, _preload
        from weather_eval.timeutil import now_beijing

        cfg = load_config()
        ev = cfg.eval
        end = now_beijing().replace(minute=0, second=0, microsecond=0)
        start = end - timedelta(days=args.days)
        print(f"收集 {start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}…", flush=True)
        obs_maps, snapshots = _preload(cfg.station_ids, cfg.models)
        hourly, daily = collect(cfg.station_ids, cfg.models, start, end,
                                ev["hourly_lead_days"], ev["daily_max_offset_days"],
                                ev["daily_min_hours"], ev["daily_source_fallback"],
                                obs_maps=obs_maps, snapshots=snapshots,
                                require_complete=ev["require_complete_snapshots"],
                                require_frozen=True)
        cells = {}
        for track, recs, bkey, bmax in (
                ("hourly", hourly, "bucket", ev["hourly_lead_days"]),
                ("daily", daily, "offset", ev["daily_max_offset_days"])):
            bym = defaultdict(list)
            for r in recs:
                bym[r["model"]].append(r)
            for m, rs in bym.items():
                byb = defaultdict(list)
                for r in rs:
                    byb[r[bkey]].append(r)
                for b in range(1, bmax + 1):
                    sub = byb.get(b, [])
                    if not sub:
                        continue
                    o = np.asarray([r["rain_obs"] for r in sub], dtype=float)
                    f = np.asarray([r["rain_fcst"] for r in sub], dtype=float)
                    st = cell_stats(o, f)
                    if st.get("n", 0) < ev["min_sample"]:
                        continue
                    cells[(track, f"{b}d", m)] = {"stats": st, "cand": candidates(st)}
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump({t: {f"{c[1]}|{c[2]}": cells[c] for c in cells if c[0] == t}
                       for t in ("hourly", "daily")}, fp, ensure_ascii=False)
        print(f"已写出 {args.out}（{len(cells)} 格）")

    # 现有降水子分（晴雨四项加权）作为"已有信息"的参照
    import weather_eval.evaluate as evm
    evm.apply_score_slopes(None)
    with open(args.track_sources, "r", encoding="utf-8") as f:
        ts = (json.load(f).get("track_sources") or {})
    precip_sub = {}
    for trk in ("hourly", "daily"):
        for m, buckets in (ts.get(trk, {}).get("precip") or {}).items():
            for b, met in (buckets or {}).items():
                s = evm.precip_score(met or {})
                if s is not None:
                    precip_sub[(trk, b, m)] = s
    print(f"\n格子 {len(cells)} · 晴雨子分参照 {len(precip_sub)}"
          f"（λ 目标：两轨池化桶内跨源 sd = {SD_TARGET}）")
    report(cells, precip_sub)


if __name__ == "__main__":
    main()

"""口径变更对拍：**扩展前**（5 温度项 + 4 降水项）vs **扩展后**（7 + 6 项）。

评价"把所有能加的指标都加进来"有没有把名次搅乱：同一批存档、同一批格子，只换
评分表，逐桶比 Spearman / 冠军换人 / 平均名次位移，并给出两版综合分的分布。

同时做一项**可靠性**检查：把 60 天窗口按前后两半各自算一遍新口径的综合分，看
名次能不能复现（新加的雨量维若只是噪声，两半的名次会显著不一致）。

运行：
  PYTHONPATH=src .venv312/bin/python scripts/compare_score_variants.py \
      --from-json .work/track_sources_60d.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys

sys.path.insert(0, "src")

import numpy as np

import weather_eval.evaluate as ev

MIN_MODELS = 4

# 扩展前的评分表（2026-10-05 口径）：权重照抄、换算族照抄
OLD_TEMP = (
    ("acc2", 0.3125, "pct"),
    ("rmse", 0.3125, "dev"),
    ("r", 0.1875, "pct"),
    ("mbe", 0.125, "dev"),
    ("slope", 0.0625, "log"),
)
OLD_PRECIP = (
    ("ets", 0.35 / 0.75, "pct"),
    ("pod", 0.15 / 0.75, "pct"),
    ("far", 0.15 / 0.75, "dev"),
    ("bias", 0.10 / 0.75, "log"),
)


def conv(family: str, key: str, v: float) -> float:
    if family == "pct":
        return v * ev.SCORE_SLOPES[key]
    if family == "dev":
        return 100 - abs(v) * ev.SCORE_SLOPES[key]
    return (100 - abs(math.log2(v)) * ev.SCORE_SLOPES[key]) if v > 0 else 0.0


def dim_score(metrics: dict, parts) -> float | None:
    num = den = 0.0
    for key, w, fam in parts:
        v = (metrics or {}).get(key)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        c = max(0.0, min(100.0, conv(fam, key, v)))
        num += w * c
        den += w
    return num / den if den else None


def temp_of(trk: str, met: dict, parts) -> float | None:
    if trk == "daily":
        hi = dim_score((met or {}).get("max"), parts)
        lo = dim_score((met or {}).get("min"), parts)
        return None if hi is None or lo is None else (hi + lo) / 2
    return dim_score(met, parts)


def composites(ts: dict, temp_parts, precip_parts) -> dict:
    out = {}
    for trk in ("hourly", "daily"):
        tsrc = ts.get(trk, {}).get("temp") or {}
        psrc = ts.get(trk, {}).get("precip") or {}
        models = sorted(set(tsrc) | set(psrc))
        buckets = sorted({b for m in models for b in (tsrc.get(m) or {})}
                         | {b for m in models for b in (psrc.get(m) or {})})
        for b in buckets:
            rows = {}
            for m in models:
                ts_ = temp_of(trk, tsrc.get(m, {}).get(b), temp_parts)
                ps = dim_score(psrc.get(m, {}).get(b), precip_parts)
                if ts_ is not None and ps is not None:
                    rows[m] = round((ts_ + ps) / 2, 2)
            if rows:
                out[f"{trk}:{b}"] = rows
    return out


def spearman(a: list[float], b: list[float]) -> float | None:
    ra = np.argsort(np.argsort([-x for x in a]))
    rb = np.argsort(np.argsort([-x for x in b]))
    if len(a) < 3 or np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(old: dict, new: dict, label: str) -> None:
    rows = []
    for bk in sorted(set(old) & set(new)):
        common = sorted(set(old[bk]) & set(new[bk]))
        if len(common) < MIN_MODELS:
            continue
        ov = [old[bk][m] for m in common]
        nv = [new[bk][m] for m in common]
        rho = spearman(ov, nv)
        oc = max(common, key=lambda m: old[bk][m])
        nc = max(common, key=lambda m: new[bk][m])
        o_ord = {m: i for i, m in enumerate(sorted(common, key=lambda x: -old[bk][x]))}
        n_ord = {m: i for i, m in enumerate(sorted(common, key=lambda x: -new[bk][x]))}
        rows.append({"board": bk, "n": len(common), "rho": rho,
                     "champ_changed": oc != nc, "old_champ": oc, "new_champ": nc,
                     "shift": float(np.mean([abs(o_ord[m] - n_ord[m]) for m in common]))})
    if not rows:
        print("  无可比榜单")
        return
    rhos = [r["rho"] for r in rows if r["rho"] is not None]
    nchg = sum(r["champ_changed"] for r in rows)
    print(f"\n-- {label} --")
    print(f"  榜数 {len(rows)}（≥{MIN_MODELS} 家同台）· 冠军换人 {nchg} · "
          f"Spearman 中位 {np.median(rhos):.3f}（min {min(rhos):.3f}）· "
          f"平均名次位移中位 {np.median([r['shift'] for r in rows]):.2f} 位")
    for r in rows:
        if r["champ_changed"]:
            print(f"    {r['board']:12s} 冠军 {r['old_champ']} → {r['new_champ']}"
                  f"（ρ={r['rho']:.2f}, n={r['n']}）")
    # 最不一致的 5 张榜
    worst = sorted([r for r in rows if r["rho"] is not None],
                   key=lambda r: r["rho"])[:5]
    print("  Spearman 最低的 5 张榜：")
    for r in worst:
        print(f"    {r['board']:12s} ρ={r['rho']:.3f}（n={r['n']}，位移 {r['shift']:.1f} 位）")


def new_parts() -> tuple:
    """把生产评分表（含 5 元组）转成 (key, w, family) 三元组供本脚本复用。"""
    fam = {}
    for k, _w, *_ in ev.TEMP_SCORE_PARTS:
        fam[k] = _family_of(k)
    for k, _w, *_ in ev.PRECIP_SCORE_PARTS:
        fam[k] = _family_of(k)
    return (tuple((k, w, fam[k]) for k, w, *_ in ev.TEMP_SCORE_PARTS),
            tuple((k, w, fam[k]) for k, w, *_ in ev.PRECIP_SCORE_PARTS))


def _family_of(key: str) -> str:
    if key in ("acc1", "acc2", "r", "ets", "pod"):
        return "pct"
    if key in ("rmse", "mae", "mbe", "far", "amt_mae"):
        return "dev"
    return "log"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", required=True)
    args = ap.parse_args()
    ev.apply_score_slopes(None)
    with open(args.from_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ts = payload.get("track_sources") or payload
    print(f"数据：{args.from_json}（窗口 {payload.get('window')}）")

    nt, np_ = new_parts()
    old = composites(ts, OLD_TEMP, OLD_PRECIP)
    new = composites(ts, nt, np_)
    compare(old, new, "扩展前 vs 扩展后（同一批格子、同一批 λ）")

    allv = [v for b in new.values() for v in b.values()]
    allo = [v for b in old.values() for v in b.values()]
    print(f"\n综合分分布：扩展前 {np.median(allo):.1f}"
          f"（P10 {np.percentile(allo, 10):.1f} / P90 {np.percentile(allo, 90):.1f}）"
          f" → 扩展后 {np.median(allv):.1f}"
          f"（P10 {np.percentile(allv, 10):.1f} / P90 {np.percentile(allv, 90):.1f}）"
          f" · 格子 {len(allv)}")


if __name__ == "__main__":
    main()

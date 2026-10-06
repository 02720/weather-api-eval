"""落地后对拍：新口径（8 温度项 + 7 降水项）vs 旧口径（7 + 6）。

与 compare_score_variants.py 同框架，但"新表"直接读 evaluate.py 的现行评分表，
保证对拍的就是生产口径本身。
"""
import json
import math
import sys

sys.path.insert(0, "src")

import numpy as np

import weather_eval.evaluate as ev

d = json.load(open(".work/track_sources_60d_v2.json"))
ts = d["track_sources"]


def cells(track, fam):
    out = {}
    for m, cell in ts[track][fam].items():
        for b, v in cell.items():
            if isinstance(v, dict):
                out.setdefault(b, {})[m] = v
    return out


lam = dict(ev.SCORE_SLOPES)


def conv(fam, key, v):
    if fam == "pct":
        return v * lam[key]
    if fam == "dev":
        return 100 - abs(v) * lam[key]
    return (100 - abs(math.log2(v)) * lam[key]) if v > 0 else 0.0


def dim(met, parts):
    num = den = 0.0
    for key, w, f in parts:
        v = (met or {}).get(key)
        if v is None or not math.isfinite(v):
            continue
        num += w * max(0.0, min(100.0, conv(f, key, v)))
        den += w
    return num / den if den else None


# 现行生产评分表（直接从模块取，对拍的就是上线的口径）
NEW_T = [(k, w, "pct" if "acc" in k or k == "r" else
          ("log" if k == "slope" else "dev")) for k, w, _l, _m, _fn in ev.TEMP_SCORE_PARTS]
NEW_P = [(k, w, "pct" if k in ("ets", "pod", "grade_ets") else
          ("log" if k in ("bias", "amt_bias") else "dev"))
         for k, w, _l, _m, _fn in ev.PRECIP_SCORE_PARTS]
# 旧口径（2026-10-06 第一轮审计后、第二轮之前）
OLD_T = [("acc2", 0.15625, "pct"), ("acc1", 0.15625, "pct"),
         ("rmse", 0.15625, "dev"), ("mae", 0.15625, "dev"),
         ("r", 0.1875, "pct"), ("mbe", 0.125, "dev"), ("slope", 0.0625, "log")]
OLD_P = [("ets", 0.35, "pct"), ("pod", 0.15, "pct"), ("far", 0.15, "dev"),
         ("amt_mae", 0.15, "dev"), ("bias", 0.10, "log"), ("amt_bias", 0.10, "log")]


def boards(tp, pp):
    out = {}
    for track in ("hourly", "daily"):
        tb, pb = cells(track, "temp"), cells(track, "precip")
        for b in sorted(set(tb) | set(pb)):
            sc = {}
            for m in set(tb.get(b, {})) | set(pb.get(b, {})):
                tv, pv = tb.get(b, {}).get(m), pb.get(b, {}).get(m)
                if tv is None or pv is None:
                    continue
                if track == "daily":
                    hi, lo = dim((tv or {}).get("max"), tp), dim((tv or {}).get("min"), tp)
                    ts_ = None if hi is None or lo is None else (hi + lo) / 2
                else:
                    ts_ = dim(tv, tp)
                ps_ = dim(pv, pp)
                if ts_ is None or ps_ is None:
                    continue
                sc[m] = (ts_ + ps_) / 2
            if len(sc) >= 4:
                out[f"{track}:{b}"] = sc
    return out


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


new, old = boards(NEW_T, NEW_P), boards(OLD_T, OLD_P)
rhos, flips, shifts, med = [], 0, [], []
for k in sorted(set(new) & set(old)):
    ms = sorted(set(new[k]) & set(old[k]))
    if len(ms) < 4:
        continue
    r = spearman([old[k][m] for m in ms], [new[k][m] for m in ms])
    if r is not None:
        rhos.append(r)
    o0 = sorted(ms, key=lambda m: -old[k][m])
    o1 = sorted(ms, key=lambda m: -new[k][m])
    flips += int(o0[0] != o1[0])
    r0 = {m: i for i, m in enumerate(o0)}
    r1 = {m: i for i, m in enumerate(o1)}
    shifts.append(np.mean([abs(r0[m] - r1[m]) for m in ms]))
    med.append(np.median([new[k][m] for m in ms]))

print(f"榜数 {len(rhos)} | Spearman 中位 {np.median(rhos):.3f}（min {min(rhos):.3f}）"
      f" | 冠军换人 {flips}/{len(rhos)} | 平均位移中位 {np.median(shifts):.2f} 位"
      f" | 综合分中位 {np.median(med):.2f}")

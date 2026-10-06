"""全指标冗余审计：判定每个"已评测但未入分"的指标能不能进综合分、以什么身份进。

回答三个问题（第一性原理）：
  Q1 恒等 —— 它是不是已有入分指标的**确定性函数**？（是 → 拒收：入分即重复计权）
  Q2 冗余 —— 它与已有入分项在"桶内跨源"空间上的相关性多高？独立信息（1−R²）有多少？
  Q3 可靠性 —— 它的桶内跨源离散度在桶与桶、轨与轨之间稳定吗？（决定单一 λ 能否标定）

数据来源：scripts/capture_track_sources.py 导出的逐桶指标 JSON。

运行：
  PYTHONPATH=src .venv312/bin/python scripts/audit_metric_redundancy.py \
      --from-json .work/track_sources_60d.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys

sys.path.insert(0, "src")

import numpy as np

MIN_MODELS_PER_BUCKET = 3

# 温度：全部已评测指标 / 当前入分项
TEMP_ALL = ("acc1", "acc2", "rmse", "mae", "mbe", "r", "slope", "chi2", "rss")
TEMP_USED = ("acc2", "rmse", "r", "mbe", "slope")
# 降水（评分轨，二分类）：全部已评测 / 当前入分项
PRECIP_ALL = ("acc", "pod", "far", "ts", "ets", "bias")
PRECIP_USED = ("ets", "pod", "far", "bias")
# 降水（连续雨量，诊断轨）：当前全部不入分
AMOUNT_ALL = ("rmse", "mae", "mbe")


# ----------------------------------------------------------------- 数据组织
def temp_cells(ts: dict, key: str) -> dict:
    """{(track,bucket): {model_id: value}}；日轨 max/min 各算半格（与评分同构）。"""
    cells: dict[tuple[str, str], dict[str, float]] = {}
    for trk in ("hourly", "daily"):
        for m, buckets in (ts.get(trk, {}).get("temp") or {}).items():
            for b, met in (buckets or {}).items():
                met = met or {}
                if trk == "daily":
                    cands = ((f"{m}·max", (met.get("max") or {}).get(key)),
                             (f"{m}·min", (met.get("min") or {}).get(key)))
                else:
                    cands = ((m, met.get(key)),)
                for mid, v in cands:
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        cells.setdefault((trk, b), {})[mid] = float(v)
    return {k: v for k, v in cells.items() if len(v) >= MIN_MODELS_PER_BUCKET}


def precip_cells(ts: dict, key: str) -> dict:
    cells: dict[tuple[str, str], dict[str, float]] = {}
    for trk in ("hourly", "daily"):
        for m, buckets in (ts.get(trk, {}).get("precip") or {}).items():
            for b, met in (buckets or {}).items():
                v = (met or {}).get(key)
                if isinstance(v, (int, float)) and math.isfinite(v):
                    cells.setdefault((trk, b), {})[m] = float(v)
    return {k: v for k, v in cells.items() if len(v) >= MIN_MODELS_PER_BUCKET}


def amount_cells(diag: dict, key: str, track: str) -> dict:
    """连续雨量指标（诊断轨与评分轨同一份样本，阈值只影响二分类）。"""
    src = (diag or {}).get("precip_hourly" if track == "hourly" else "precip_daily")
    cells: dict[tuple[str, str], dict[str, float]] = {}
    for m, buckets in (src or {}).items():
        for b, met in (buckets or {}).items():
            v = (met or {}).get(key)
            if isinstance(v, (int, float)) and math.isfinite(v):
                cells.setdefault((track, b), {})[m] = float(v)
    return {k: v for k, v in cells.items() if len(v) >= MIN_MODELS_PER_BUCKET}


def centered(cells: dict) -> tuple[np.ndarray, list]:
    """桶内中心化后池化：返回 (n×1 数组, 桶标签)。"""
    vals, labels = [], []
    for (trk, b), bucket in cells.items():
        ks = sorted(bucket)
        arr = np.asarray([bucket[k] for k in ks], dtype=float)
        vals.extend((arr - arr.mean()).tolist())
        labels.extend([(trk, b)] * len(ks))
    return np.asarray(vals), labels


def centered_matrix(cellmap: dict) -> tuple[np.ndarray, list, list]:
    """多指标对齐：只保留所有指标都非空的格子（同 (格,源) 对齐）。"""
    common = sorted(set.intersection(*[set(c) for c in cellmap.values()]))
    rows, labels, keys = [], [], list(cellmap)
    for cell in common:
        members = sorted(cellmap[keys[0]][cell])
        for mid in members:
            vals = [cellmap[k][cell].get(mid) for k in keys]
            if all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
                rows.append([float(v) for v in vals])
                labels.append((*cell, mid))
    if not rows:
        return np.empty((0, len(keys))), [], keys
    M = np.asarray(rows)
    # 按桶中心化
    out = M.copy()
    for lab in sorted({(l[0], l[1]) for l in labels}):
        idx = [i for i, l in enumerate(labels) if (l[0], l[1]) == lab]
        if len(idx) >= MIN_MODELS_PER_BUCKET:
            out[idx] -= M[idx].mean(axis=0)
        else:
            out[idx] = np.nan
    keep = ~np.isnan(out[:, 0])
    return out[keep], [l for l, k in zip(labels, keep) if k], keys


def corr_of(M: np.ndarray, keys: list) -> np.ndarray:
    C = np.corrcoef(M, rowvar=False)
    return C


def ols_r2(y: np.ndarray, X: np.ndarray) -> float:
    """y 对 X（含截距）的 R²；X 为 n×k。"""
    n, k = X.shape
    if n <= k + 1:
        return float("nan")
    A = np.hstack([np.ones((n, 1)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ----------------------------------------------------------------- Q1 恒等式
def verify_identities(ts: dict, diag: dict) -> None:
    print("\n" + "=" * 78)
    print("Q1 恒等式验证：未入分项是不是已有入分项的确定性函数？")
    print("=" * 78)

    # --- 温度：chi2 == rmse²，rss == n × rmse² ---
    for key, ref, label in (("chi2", "rmse", "χ² = RMSE²"),
                            ("rss", None, "RSS = n × RMSE²")):
        cc = temp_cells(ts, key)
        rc = temp_cells(ts, ref or "rmse")
        nc = temp_cells(ts, "n") if ref is None else None
        errs, rel = [], []
        for cell in sorted(set(cc) & set(rc)):
            if ref is None:
                ncell = (nc or {}).get(cell, {})
                if not ncell:
                    continue
            for mid in sorted(set(cc[cell]) & set(rc[cell])):
                v = cc[cell][mid]
                pred = rc[cell][mid] ** 2
                if ref is None:
                    n = ncell.get(mid)
                    if not n:
                        continue
                    pred *= n
                if pred == 0:
                    continue
                errs.append(abs(v - pred))
                rel.append(abs(v - pred) / pred)
        if rel:
            print(f"  {label:18s} 格子 {len(rel):5d} · 中位相对误差 "
                  f"{np.median(rel):.2e} · P95 {np.percentile(rel, 95):.2e} "
                  f"· 最大 {max(rel):.2e}")
            print(f"    → {'恒等成立（差异仅来自 4 位小数舍入）' if max(rel) < 1e-3 else '不成立，需另行分析'}")

    # --- 降水：BIAS = POD/(1−FAR) ---
    pc = {k: precip_cells(ts, k) for k in ("pod", "far", "bias")}
    rel = []
    for cell in sorted(set(pc["pod"]) & set(pc["far"]) & set(pc["bias"])):
        for mid in sorted(set(pc["pod"][cell]) & set(pc["far"][cell]) & set(pc["bias"][cell])):
            p = pc["pod"][cell][mid] / 100.0
            f = pc["far"][cell][mid] / 100.0
            if f >= 1:
                continue
            pred = p / (1 - f)
            if pred <= 0:
                continue
            rel.append(abs(pc["bias"][cell][mid] - pred) / pred)
    if rel:
        print(f"  {'BIAS = POD/(1−FAR)':18s} 格子 {len(rel):5d} · 中位相对误差 "
              f"{np.median(rel):.2e} · P95 {np.percentile(rel, 95):.2e} · 最大 {max(rel):.2e}")

    # --- 降水：acc / ts 能由 (ETS, POD, FAR, n) 精确重建？ ---
    cells4 = {k: precip_cells(ts, k) for k in ("ets", "pod", "far", "acc", "ts", "n")}
    common = sorted(set.intersection(*[set(c) for c in cells4.values()]))

    def reconstruct(ets, pod, far, n):
        """由 (ETS, POD, FAR, n) 反解列联表 (h, fa, mi, c)；返回 None 表示无解。"""
        p, f = pod / 100.0, far / 100.0
        if not (0 < p <= 1 and 0 <= f < 1):
            return None
        ka = (1 - p) / p + f / (1 - f) + 1   # (d + b + a)/a

        def ets_of(a):
            d = a * (1 - p) / p
            b = a * f / (1 - f)
            c = n - a - b - d
            if min(a, b, d, c) < 0:
                return None
            hr = (a + b) * (a + d) / n
            den = a + b + d - hr
            return (a - hr) / den if den > 0 else None

        lo, hi = 1e-9, n / ka
        flo, fhi = ets_of(lo), ets_of(hi)
        if flo is None or fhi is None:
            return None
        for _ in range(200):
            mid = (lo + hi) / 2
            fm = ets_of(mid)
            if fm is None:
                return None
            if (fm - ets) * (fhi - ets) > 0:
                hi, fhi = mid, fm
            else:
                lo, flo = mid, fm
        a = (lo + hi) / 2
        d = a * (1 - p) / p
        b = a * f / (1 - f)
        c = n - a - b - d
        return a, b, d, c

    ok = err_a = err_t = 0
    rel_a, rel_t = [], []
    for cell in common:
        for mid in sorted(set.intersection(*[set(cells4[k][cell]) for k in cells4])):
            r = reconstruct(cells4["ets"][cell][mid], cells4["pod"][cell][mid],
                            cells4["far"][cell][mid], cells4["n"][cell][mid])
            if r is None:
                continue
            a, b, d, c = r
            n = a + b + d + c
            pred_acc = (a + c) / n * 100
            pred_ts = a / (a + b + d) if (a + b + d) > 0 else None
            ok += 1
            va = cells4["acc"][cell][mid]
            if va:
                rel_a.append(abs(pred_acc - va) / va)
            vt = cells4["ts"][cell][mid]
            if pred_ts and vt:
                rel_t.append(abs(pred_ts - vt) / vt)
            err_a += 1 if not va else 0
            err_t += 1 if not vt else 0
    if rel_a:
        print(f"  {'acc = f(ETS,POD,FAR,n)':18s} 格子 {len(rel_a):5d} · 中位相对误差 "
              f"{np.median(rel_a):.2e} · P95 {np.percentile(rel_a, 95):.2e} · 最大 {max(rel_a):.2e}")
    if rel_t:
        print(f"  {'ts  = f(ETS,POD,FAR,n)':18s} 格子 {len(rel_t):5d} · 中位相对误差 "
              f"{np.median(rel_t):.2e} · P95 {np.percentile(rel_t, 95):.2e} · 最大 {max(rel_t):.2e}")

    # --- 漏报率 = 100 − POD（诊断轨有 miss）---
    ph = (diag or {}).get("precip_hourly") or {}
    rel = []
    for m, buckets in ph.items():
        for b, met in (buckets or {}).items():
            pod, miss = (met or {}).get("pod"), (met or {}).get("miss")
            if isinstance(pod, (int, float)) and isinstance(miss, (int, float)):
                if pod:
                    rel.append(abs(miss - (100 - pod)) / max(pod, 1e-9))
    if rel:
        print(f"  {'漏报率 = 100−POD':18s} 格子 {len(rel):5d} · 中位相对误差 "
              f"{np.median(rel):.2e} · P95 {np.percentile(rel, 95):.2e}")


# ----------------------------------------------------------------- Q2 冗余
def redundancy(ts: dict, diag: dict) -> None:
    print("\n" + "=" * 78)
    print("Q2 冗余度：桶内跨源相关矩阵 + 未入分项对已入分项的增量信息（1−R²）")
    print("=" * 78)

    def report(name: str, cellmap: dict, used: tuple, order: tuple):
        M, labels, keys = centered_matrix(cellmap)
        print(f"\n-- {name}（对齐格子 {len(labels)}；指标 {len(keys)}）--")
        C = np.corrcoef(M, rowvar=False)
        idx = {k: i for i, k in enumerate(keys)}
        hdr = "        " + "".join(f"{k[:6]:>8s}" for k in order)
        print(hdr)
        for k in order:
            row = "".join(f"{C[idx[k], idx[j]]:8.2f}" for j in order)
            mark = " *" if k in used else "  "
            print(f"  {k[:6]:>6s}{row}{mark}")
        print("  （* = 当前入分项）")
        print("\n  未入分项的独立信息（对已入分项做 OLS，1−R² = 未被解释的方差比例）：")
        X = np.column_stack([M[:, idx[k]] for k in used])
        for k in order:
            if k in used:
                continue
            r2 = ols_r2(M[:, idx[k]], X)
            # 与最相关入分项的简单相关
            sims = {u: abs(C[idx[k], idx[u]]) for u in used}
            top = max(sims, key=sims.get)
            print(f"    {k[:6]:>6s}  1−R² = {1 - r2:6.3f}   "
                  f"与已入分项最大 |ρ| = {sims[top]:.3f}（{top}）")

    report("温度", {k: temp_cells(ts, k) for k in TEMP_ALL}, TEMP_USED, TEMP_ALL)
    report("降水·二分类（评分轨）", {k: precip_cells(ts, k) for k in PRECIP_ALL},
           PRECIP_USED, PRECIP_ALL)
    for trk in ("hourly", "daily"):
        cm = {k: amount_cells(diag, k, trk) for k in AMOUNT_ALL}
        cm.update({k: precip_cells(ts, k) for k in PRECIP_USED})
        order = AMOUNT_ALL + PRECIP_USED
        report(f"降水·雨量（{trk} 轨，诊断轨同一份样本）", cm, PRECIP_USED, order)


# ----------------------------------------------------------------- Q3 尺度稳定性
def scale_stability(ts: dict, diag: dict) -> None:
    print("\n" + "=" * 78)
    print("Q3 尺度稳定性：桶内跨源 sd 在桶/轨之间的离散程度（决定单一 λ 能否标定）")
    print("=" * 78)
    print(f"  {'项':10s} {'轨':8s} {'桶数':>5s} {'sd 中位':>9s} {'sd P10':>9s} "
          f"{'sd P90':>9s} {'P90/P10':>8s}")
    rows = []
    for key in TEMP_ALL + ("n",):
        cc = temp_cells(ts, key)
        for trk in ("hourly", "daily"):
            sds = [float(np.std(list(b.values()), ddof=1))
                   for (t, _b), b in cc.items() if t == trk and len(b) >= MIN_MODELS_PER_BUCKET]
            if len(sds) >= 4:
                rows.append((key, trk, sds))
    for key in PRECIP_ALL:
        cc = precip_cells(ts, key)
        for trk in ("hourly", "daily"):
            sds = [float(np.std(list(b.values()), ddof=1))
                   for (t, _b), b in cc.items() if t == trk and len(b) >= MIN_MODELS_PER_BUCKET]
            if len(sds) >= 4:
                rows.append((key, trk, sds))
    for key in AMOUNT_ALL:
        for trk in ("hourly", "daily"):
            cc = amount_cells(diag, key, trk)
            sds = [float(np.std(list(b.values()), ddof=1))
                   for b in cc.values() if len(b) >= MIN_MODELS_PER_BUCKET]
            if len(sds) >= 4:
                rows.append((key + "(雨量)", trk, sds))
    for key, trk, sds in rows:
        p10, p50, p90 = np.percentile(sds, [10, 50, 90])
        ratio = p90 / p10 if p10 > 0 else float("inf")
        print(f"  {key:10s} {trk:8s} {len(sds):5d} {p50:9.4f} {p10:9.4f} {p90:9.4f} "
              f"{ratio:8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", required=True)
    args = ap.parse_args()
    with open(args.from_json, "r", encoding="utf-8") as f:
        payload = json.load(f)
    ts = payload.get("track_sources") or payload
    diag = payload.get("diagnostics") or {}
    models = sorted({m for trk in ("hourly", "daily") for dim in ("temp", "precip")
                     for m in (ts.get(trk, {}).get(dim) or {})})
    print(f"数据：{args.from_json} · {len(models)} 源")
    verify_identities(ts, diag)
    redundancy(ts, diag)
    scale_stability(ts, diag)


if __name__ == "__main__":
    main()

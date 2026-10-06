"""全指标入分（第二轮，2026-10-06）：恒等残差 + 真新维度，能加多少？

上一轮（docs/metric_coverage.md）把每个指标判成 A 真信息 / B 同族冗余 / C 拒收，
温度扩到 7 项、降水扩到 6 项。这一轮回答用户的追问：**还能不能再加？**

三条第一性原理判据（与上一轮同，但这一轮给出"加"的机制而不只是"拒"的理由）：

* **Q1 恒等**：y 是否已有入分项的**解析**函数？若是，y 的独立信息严格为 0，
  加进加权和必然重复计权。但"独立信息为 0"不等于"不能出现在公式里"——
  把它写成 **残差项** r = y − g(入分项)（g 解析、零参数、不依赖参评者），
  则 r 与入分项正交，**任何权重下都不产生重复计权**，而 y 本身照进公式。
  残差若真只是舍入噪声（实测 ~1e-3），它的实际话语权自动 ≈ 0——不是被
  拒收，是被它自己的信息含量判了零权重。这是"全部纳入"与"不失真"的**唯一**
  兼容写法：不是折中，是正交分解（Gram-Schmidt）。
* **Q2 增量**：无解析恒等，但对入分项做 OLS 的 1−R² 仍很高 → 同族，并入族内
  等权，不新开槽位。
* **Q3 可靠性**：子分（截断后）的桶内跨源 sd 在桶与轨之间是否稳定（P90/P10、
  日/小时比）？差 → 先换尺度无关口径，仍差则拒收。

本脚本同时把上一轮**完全没有审视过的一块**纳入视野：分级（graded）雨强指标
（hourly 1~5 级 / daily +1~+6 级，每级 7 项）。它们从来只在明细表露过面。

运行：
  PYTHONPATH=src .venv312/bin/python scripts/audit_full_inclusion.py \
      --from-json .work/track_sources_60d.json \
      --amount-json .work/amount_cells.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys

sys.path.insert(0, "src")

import numpy as np

MIN_MODELS = 4          # 桶内最少同台家数（与 compare_score_variants.py 同）
TARGET_SD = 8.0         # λ 反解目标：子分桶内跨源 sd（与 calibrate_score_slopes 同）
TEMP_IN = ("acc2", "acc1", "rmse", "mae", "r", "mbe", "slope")
PRECIP_IN = ("ets", "pod", "far", "bias", "amt_mae", "amt_bias")
HOURLY_LEVS = ("1", "2", "3", "4", "5")
DAILY_LEVS = ("+1", "+2", "+3", "+4", "+5", "+6")


# ------------------------------------------------------------------ 数据装配
def load(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def temp_subs(track: str, v: dict) -> list[dict]:
    """温度格子 → 该格参与评分的若干"量"（日轨有 max/min 两个）。"""
    if track == "daily":
        return [x for x in (v.get("max"), v.get("min")) if isinstance(x, dict)]
    return [v] if isinstance(v, dict) else []


def cells_by_bucket(ts: dict, track: str, fam: str) -> dict:
    """{桶: {源: 指标字典}}。"""
    src = (ts.get(track) or {}).get(fam) or {}
    out: dict[str, dict] = {}
    for m, cell in src.items():
        for b, v in (cell.items() if isinstance(cell, dict) else []):
            if isinstance(v, dict):
                out.setdefault(b, {})[m] = v
    return out


# ------------------------------------------------------- Q1：解析恒等重建
def solve_contingency(pod: float, far: float, ets: float, n: float):
    """由 (POD, FAR, ETS, n) 反解 2×2 列联表 (h, fa, mi, c)。

    列联表给定 n 只有 3 个自由度，{POD, FAR, ETS} 恰好撑满，故解唯一。
    POD/FAR 定形（mi = h(1−p)/p、fa = h·q/(1−q)），ETS 定大小；ETS 关于 h
    在可行域 [0, n/(1+fa/h+mi/h)] 上单调递减，二分即可。
    """
    if not (n and n > 0 and 0 < pod <= 100 and 0 <= far < 100):
        return None
    p, q = pod / 100.0, far / 100.0
    mi_per_h, fa_per_h = (1 - p) / p, q / (1 - q)
    lo, hi = 1e-9, n / (1 + fa_per_h + mi_per_h)

    def ets_of(h: float) -> float:
        a, bb = h / p, h / (1 - q)
        mi, fa = a - h, bb - h
        href = a * bb / n
        den = h + fa + mi - href
        return (h - href) / den if den > 0 else -1.0

    if ets_of(lo) < ets or ets_of(hi) > ets:
        return None
    for _ in range(200):
        h = (lo + hi) / 2
        if ets_of(h) > ets:
            lo = h
        else:
            hi = h
    h = (lo + hi) / 2
    a, bb = h / p, h / (1 - q)
    mi, fa = a - h, bb - h
    return h, fa, mi, n - h - fa - mi


def q1_identities(ts: dict, dg: dict) -> dict:
    """逐格验证解析恒等式，返回各项的相对残差分布。"""
    temp_res = {"chi2": [], "rss": []}
    for track in ("hourly", "daily"):
        for b, cell in cells_by_bucket(ts, track, "temp").items():
            for s in cell.values():
                for v in temp_subs(track, s):
                    n, rmse = v.get("n"), v.get("rmse")
                    if not n or rmse is None:
                        continue
                    if v.get("chi2") is not None:
                        temp_res["chi2"].append(
                            abs(math.sqrt(v["chi2"]) - rmse) / max(abs(rmse), 1e-9))
                    if v.get("rss") is not None:
                        temp_res["rss"].append(
                            abs(math.sqrt(v["rss"] / n) - rmse) / max(abs(rmse), 1e-9))

    pr = {k: [] for k in ("acc", "ts", "miss", "farate", "bias")}
    for fam in ("precip_hourly", "precip_daily"):
        for b, cell in cells_by_bucket({"x": {fam: dg[fam]}}, "x", fam).items():
            for v in cell.values():
                n, pod, far, ets = v.get("n"), v.get("pod"), v.get("far"), v.get("ets")
                if None in (n, pod, far, ets) or not n or pod <= 0 or far >= 100:
                    continue
                sol = solve_contingency(pod, far, ets, n)
                if not sol:
                    continue
                h, fa, mi, c = sol
                calc = {
                    "acc": 100 * (h + c) / n,
                    "ts": h / (h + fa + mi) if h + fa + mi > 0 else None,
                    "miss": 100 * mi / (h + mi) if h + mi > 0 else None,
                    "farate": 100 * fa / (fa + c) if fa + c > 0 else None,
                    "bias": (h + fa) / (h + mi) if h + mi > 0 else None,
                }
                for k, val in calc.items():
                    obs = v.get(k)
                    if val is None or obs is None:
                        continue
                    pr[k].append(abs(val - obs) / (abs(obs) if abs(obs) > 1e-9 else 1.0))
    return {"temp": temp_res, "precip": pr}


# ------------------------------------------------------- Q2：增量信息 1−R²
def bucket_ols(cells: dict, target, inp=PRECIP_IN, minn: int = MIN_MODELS):
    """桶内中心化后，用入分项 OLS 解释 target，返回 (1−R², max|ρ|, n)。

    target 是指标字典 → float|None 的可调用对象，或指标键名。
    桶内中心化是为了剥离"桶难度"，只留跨源差异——与上一轮审计同口径。
    """
    get = (lambda v: v.get(target)) if isinstance(target, str) else target
    segs = []
    for cell in cells.values():
        rows = []
        for v in cell.values():
            t = get(v)
            if t is None or not math.isfinite(t):
                continue
            if any(v.get(k) is None for k in inp):
                continue
            rows.append(([v[k] for k in inp], t))
        if len(rows) < minn:
            continue
        a = np.array([r[0] for r in rows], float)
        y = np.array([r[1] for r in rows], float)
        segs.append((a - a.mean(0), y - y.mean()))
    if not segs:
        return None
    a = np.vstack([s[0] for s in segs])
    y = np.concatenate([s[1] for s in segs])
    a = np.c_[a, np.ones(len(a))]
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    r2 = 1 - ((y - a @ beta) ** 2).sum() / max((y ** 2).sum(), 1e-12)
    cors = [abs(np.corrcoef(a[:, i], y)[0, 1]) for i in range(len(inp))]
    return 1 - r2, max(cors), len(y)


def bucket_rho(cells: dict, g1, g2, minn: int = MIN_MODELS) -> float:
    """两指标在"桶内跨源"空间上的相关系数。"""
    xs, ys = [], []
    for cell in cells.values():
        pr = [(g1(v), g2(v)) for v in cell.values()]
        pr = [(x, y) for x, y in pr if x is not None and y is not None]
        if len(pr) < minn:
            continue
        a = np.array([p[0] for p in pr], float)
        b = np.array([p[1] for p in pr], float)
        xs.append(a - a.mean())
        ys.append(b - b.mean())
    if not xs:
        return float("nan")
    return float(np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1])


def graded_get(lev: str, key: str = "ets"):
    return lambda v: ((v.get("graded") or {}).get(lev) or {}).get(key)


# ------------------------------------------------------- Q3 / λ 反解
def sub_sd(cells: dict, get, family: str, lam: float) -> dict:
    """给定换算族与 λ，算子分（截断到 [0,100]）的桶内跨源 sd 分布。"""
    devs = []
    for cell in cells.values():
        vals = [get(v) for v in cell.values()]
        vals = [x for x in vals if x is not None and math.isfinite(x)]
        if len(vals) < MIN_MODELS:
            continue
        if family == "pct":
            s = np.array(vals) * lam
        elif family == "dev":
            s = 100 - np.abs(np.array(vals)) * lam
        else:
            s = np.array([(100 - abs(math.log2(x)) * lam) if x > 0 else 0.0
                          for x in vals])
        s = np.clip(s, 0.0, 100.0)
        if np.std(s, ddof=1) > 0:
            devs.append(s - s.mean())
    if not devs:
        return {"sd": float("nan"), "p90_p10": float("nan"), "n_bucket": 0}
    allv = np.concatenate(devs)
    sds = [np.std(d, ddof=1) for d in devs]
    p90, p10 = np.percentile(sds, 90), np.percentile(sds, 10)
    return {"sd": float(np.std(allv, ddof=1)),
            "p90_p10": float(p90 / p10) if p10 > 1e-9 else float("inf"),
            "n_bucket": len(devs)}


def solve_lambda(cells: dict, get, family: str, target: float = TARGET_SD) -> float | None:
    """反解 λ 使子分桶内跨源 sd = target（dev 族 λ 与 sd 成正比，一步反解）。"""
    lo, hi = 1e-3, 1e6
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        sd = sub_sd(cells, get, family, mid)["sd"]
        if not math.isfinite(sd):
            return None
        if sd < target:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ------------------------------------------------------------------ 对拍
def conv(family: str, lam: float, v: float) -> float:
    if family == "pct":
        return v * lam
    if family == "dev":
        return 100 - abs(v) * lam
    return (100 - abs(math.log2(v)) * lam) if v > 0 else 0.0


def dim_score(metrics: dict, parts, adj=()) -> float | None:
    """parts: [(取值器, 权重, 族, λ)]；缺项按剩余权重归一。

    adj: 零中心修正项 [(取值器, 符号, λ)] —— 残差 0 时贡献严格为 0，故它只在
    该指标真的带了入分项之外的信息时才动分数，且方向由符号（越大越好 +1 /
    越小越好 −1）决定。**不能**用 dev 族子分写：dev 族把"残差=0"读成满分
    （100−0×λ=100），会把"完全冗余"误读成"完美"，系统性抬高总分。
    """
    num = den = 0.0
    for get, w, fam, lam in parts:
        v = get(metrics or {})
        if v is None or not math.isfinite(v):
            continue
        c = max(0.0, min(100.0, conv(fam, lam, v)))
        num += w * c
        den += w
    if not den:
        return None
    base = num / den
    for get, sign, lam in adj:
        v = get(metrics or {})
        if v is None or not math.isfinite(v):
            continue
        base += sign * lam * v
    return max(0.0, min(100.0, base))


def composites(ts: dict, tparts, pparts, tadj=(), padj=()) -> dict:
    """逐桶综合分（温度分与降水分各半，缺一即缺）。"""
    out = {}
    for track in ("hourly", "daily"):
        tb = cells_by_bucket(ts, track, "temp")
        pb = cells_by_bucket(ts, track, "precip")
        for b in sorted(set(tb) | set(pb)):
            scores = {}
            for m in set(tb.get(b, {})) | set(pb.get(b, {})):
                tv = tb.get(b, {}).get(m)
                pv = pb.get(b, {}).get(m)
                if tv is None or pv is None:
                    continue
                if track == "daily":
                    hi = dim_score((tv or {}).get("max"), tparts, tadj)
                    lo = dim_score((tv or {}).get("min"), tparts, tadj)
                    ts_ = None if hi is None or lo is None else (hi + lo) / 2
                else:
                    ts_ = dim_score(tv, tparts, tadj)
                ps_ = dim_score(pv, pparts, padj)
                if ts_ is None or ps_ is None:
                    continue
                scores[m] = (ts_ + ps_) / 2
            if len(scores) >= MIN_MODELS:
                out[f"{track}:{b}"] = scores
    return out


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def compare(base: dict, alt: dict) -> dict:
    rhos, shifts, champs, med, deltas = [], [], 0, [], []
    for k in sorted(set(base) & set(alt)):
        b0, b1 = base[k], alt[k]
        ms = sorted(set(b0) & set(b1))
        if len(ms) < MIN_MODELS:
            continue
        s0 = [b0[m] for m in ms]
        s1 = [b1[m] for m in ms]
        deltas.append(float(np.max(np.abs(np.array(s1) - np.array(s0)))))
        r = spearman(s0, s1)
        if r is not None:
            rhos.append(r)
        o0 = sorted(ms, key=lambda m: -b0[m])
        o1 = sorted(ms, key=lambda m: -b1[m])
        champs += int(o0[0] != o1[0])
        rank0 = {m: i for i, m in enumerate(o0)}
        rank1 = {m: i for i, m in enumerate(o1)}
        shifts.append(np.mean([abs(rank0[m] - rank1[m]) for m in ms]))
        med.append(np.median(s1))
    return {"boards": len(rhos),
            "spearman_med": float(np.median(rhos)) if rhos else float("nan"),
            "spearman_min": float(min(rhos)) if rhos else float("nan"),
            "champ_flips": champs,
            "mean_shift": float(np.median(shifts)) if shifts else float("nan"),
            "score_med": float(np.median(med)) if med else float("nan"),
            "max_delta": float(max(deltas)) if deltas else float("nan"),
            "med_delta": float(np.median(deltas)) if deltas else float("nan")}


# ------------------------------------------------------------------ 主流程
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-json", default=".work/track_sources_60d_v2.json")
    ap.add_argument("--amount-json", default=".work/amount_cells.json")
    args = ap.parse_args()

    raw = load(args.from_json)
    ts = raw["track_sources"]
    dg = raw.get("diagnostics") or {}
    window = raw.get("window", {})

    print(f"# 全指标入分审计（第二轮）  窗口 {window.get('start')} ~ {window.get('end')}")
    print(f"# 目标：子分桶内跨源 sd* = {TARGET_SD} 分；桶内 ≥{MIN_MODELS} 家才计")

    # ---- Q1
    q1 = q1_identities(ts, dg)
    print("\n## Q1 解析恒等：残差就是「独立信息」")
    for k, v in q1["temp"].items():
        a = np.array(v)
        print(f"  温度 {k:>5} → RMSE   n={len(a):>4}  中位 {np.median(a):.2e}  "
              f"P95 {np.percentile(a,95):.2e}  max {a.max():.2e}")
    for k, v in q1["precip"].items():
        a = np.array(v)
        if not len(a):
            print(f"  降水 {k:>7}  n=0")
            continue
        print(f"  降水 {k:>7} → (ETS,POD,FAR,n)  n={len(a):>4}  中位 {np.median(a):.2e}  "
              f"P95 {np.percentile(a,95):.2e}  max {a.max():.2e}")

    # ---- Q2 分级
    print("\n## Q2 分级（graded）雨强指标：对晴雨 4 项的增量信息")
    print(f"  {'轨':<7}{'级':>4}{'1−R²':>9}{'max|ρ|':>9}{'n':>6}{'ρ(级ETS,晴雨ETS)':>18}{'级ETS中位':>11}")
    for fam, levs, lab in (("precip_hourly", HOURLY_LEVS, "hourly"),
                           ("precip_daily", DAILY_LEVS, "daily")):
        if fam not in dg:
            continue
        cells = cells_by_bucket({"x": {fam: dg[fam]}}, "x", fam)
        for lev in levs:
            r = bucket_ols(cells, graded_get(lev), inp=("ets", "pod", "far", "bias"))
            if not r:
                print(f"  {lab:<7}{lev:>4}  样本不足")
                continue
            rho = bucket_rho(cells, graded_get(lev), lambda v: v.get("ets"))
            vals = [graded_get(lev)(v) for c in cells.values() for v in c.values()]
            vals = [x for x in vals if x is not None]
            print(f"  {lab:<7}{lev:>4}{r[0]:>9.3f}{r[1]:>9.3f}{r[2]:>6}{rho:>+18.3f}"
                  f"{np.median(vals):>11.3f}")

    # ---- 雨量 RMSE（相对口径）
    print("\n## Q2 雨量 RMSE（相对口径 rmse/Σo）：amt_mae 入分后还剩多少")
    amt = load(args.amount_json)
    amtc: dict[str, dict[str, dict]] = {}
    for tr, cells in amt.items():
        for key, v in cells.items():
            b, m = key.split("|", 1)
            amtc.setdefault(tr, {}).setdefault(b, {})[m] = v["stats"]
    for track in ("hourly", "daily"):
        cells = cells_by_bucket(ts, track, "precip")
        xs, ys = [], []
        for b, cell in cells.items():
            pr = []
            for m, v in cell.items():
                st = amtc.get(track, {}).get(b, {}).get(m)
                if not st or not st.get("mean_obs"):
                    continue
                pr.append((st["rmse"] / st["mean_obs"], st["mae"] / st["mean_obs"]))
            if len(pr) >= MIN_MODELS:
                ar = np.array([p[0] for p in pr]); aa = np.array([p[1] for p in pr])
                xs.append(ar - ar.mean()); ys.append(aa - aa.mean())
        rho = np.corrcoef(np.concatenate(xs), np.concatenate(ys))[0, 1] if xs else float("nan")
        print(f"  {track:<7} ρ(相对雨量 rmse, amt_mae) = {rho:+.3f}   "
              f"（amt_mae 已入分 → rmse 属同族冗余）")

    # ---- 对拍
    print("\n## 对拍：三档变体 vs 当前口径")
    tsl = {}
    for track in ("hourly", "daily"):
        for b, cell in cells_by_bucket(ts, track, "temp").items():
            for v in cell.values():
                for s in temp_subs(track, v):
                    for k in TEMP_IN:
                        tsl.setdefault(k, []).append(s.get(k))
    # λ 沿用 config 冻结值（与生产一致）
    import weather_eval.evaluate as ev
    lam = dict(ev.SCORE_SLOPES)

    def g(k):
        return lambda v: v.get(k)

    base_t = [(g(k), w, fam, lam[k]) for k, w, fam in (
        ("acc2", 0.15625, "pct"), ("acc1", 0.15625, "pct"),
        ("rmse", 0.15625, "dev"), ("mae", 0.15625, "dev"),
        ("r", 0.1875, "pct"), ("mbe", 0.125, "dev"), ("slope", 0.0625, "log"))]
    base_p = [(g(k), w, fam, lam[k]) for k, w, fam in (
        ("ets", 0.35, "pct"), ("pod", 0.15, "pct"), ("far", 0.15, "dev"),
        ("amt_mae", 0.15, "dev"), ("bias", 0.10, "log"), ("amt_bias", 0.10, "log"))]

    base = composites(ts, base_t, base_p)

    print(f"  V0 当前口径（温度 7 项 / 降水 6 项）  综合分中位 "
          f"{np.median([np.median(list(s.values())) for s in base.values()]):.2f}  "
          f"榜数 {len(base)}")

    # V1 残差纳入：恒等项以解析残差形式进公式（χ²/RSS/acc/ts）
    def chi2_res(v):
        x, r = v.get("chi2"), v.get("rmse")
        return None if x is None or r is None else math.sqrt(x) - r
    def rss_res(v):
        x, r, n = v.get("rss"), v.get("rmse"), v.get("n")
        return None if None in (x, r, n) or not n else math.sqrt(x / n) - r
    def _rebuild(v, which):
        n, pod, far, ets = v.get("n"), v.get("pod"), v.get("far"), v.get("ets")
        if None in (n, pod, far, ets) or not n or pod <= 0 or far >= 100:
            return None
        sol = solve_contingency(pod, far, ets, n)
        if not sol:
            return None
        h, fa, mi, c = sol
        if which == "acc":
            return 100 * (h + c) / n
        return h / (h + fa + mi) if h + fa + mi > 0 else None
    def acc_res(v):
        p, o = _rebuild(v, "acc"), v.get("acc")
        return None if p is None or o is None else o - p
    def ts_res(v):
        p, o = _rebuild(v, "ts"), v.get("ts")
        return None if p is None or o is None else o - p
    # V1a：残差当作 0~100 子分（dev 族）进加权和 —— 预期失真，作为反例
    v1a_t = base_t + [(chi2_res, 0.05, "dev", 100.0), (rss_res, 0.05, "dev", 100.0)]
    v1a_p = base_p + [(acc_res, 0.05, "dev", 100.0), (ts_res, 0.05, "dev", 1000.0)]
    v1a = composites(ts, v1a_t, v1a_p)
    r1a = compare(base, v1a)
    print(f"  V1a 残差当子分（dev 族，各 0.05）：Spearman {r1a['spearman_med']:.4f} · "
          f"冠军换人 {r1a['champ_flips']}/{r1a['boards']} · 位移 {r1a['mean_shift']:.3f} 位 · "
          f"综合分中位 {r1a['score_med']:.2f}（较 V0 {r1a['score_med']-50.88:+.2f}）"
          f" ← dev 族把「残差=0」读成满分，失真")

    # V1b：残差作为零中心修正项（残差 0 → 贡献严格 0）
    v1b = composites(ts, base_t, base_p,
                     tadj=[(chi2_res, -1, 20.0), (rss_res, -1, 20.0)],
                     padj=[(acc_res, +1, 0.6), (ts_res, +1, 60.0)])
    r1b = compare(base, v1b)
    print(f"  V1b 残差当零中心修正（r=0→0）：Spearman {r1b['spearman_med']:.4f} · "
          f"冠军换人 {r1b['champ_flips']}/{r1b['boards']} · "
          f"逐格分数偏移 中位 {r1b['med_delta']:.4f} 分 / 最大 {r1b['max_delta']:.4f} 分"
          f" ← 数学上等于恒等 0，写进公式也只是记账")

    # V2 加「空间一致性」代理：|r−r_pooled| 与 |log2(slope/slope_pooled)|
    cells_h = cells_by_bucket(ts, "hourly", "temp")
    cells_d = cells_by_bucket(ts, "daily", "temp")
    def rgap(v):
        a, b = v.get("r"), v.get("r_pooled")
        return None if a is None or b is None else abs(a - b)
    def sgap(v):
        a, b = v.get("slope"), v.get("slope_pooled")
        return None if None in (a, b) or a <= 0 or b <= 0 else abs(math.log2(a / b))
    lam_rg = solve_lambda(cells_h, rgap, "dev") or 1.0
    lam_sg = solve_lambda(cells_h, sgap, "dev") or 1.0
    print(f"     空间一致性代理 λ 反解：|r−r_pooled| λ={lam_rg:.2f}（每差 "
          f"{1/lam_rg:.3f} 扣 1 分）· |log₂(slope/slope_pooled)| λ={lam_sg:.2f}")
    for nm, l1, l2 in (("hourly", lam_rg, lam_sg), ("daily", lam_rg, lam_sg)):
        cb = cells_by_bucket(ts, nm, "temp")
        sub = [s for c in cb.values() for v in c.values() for s in temp_subs(nm, v)]
        cellz = {b: {m: s for m, v in c.items() for s in temp_subs(nm, v)}
                 for b, c in cb.items()}
        print(f"     {nm}: |r−r_pooled| 子分 sd={sub_sd(cellz, rgap, 'dev', l1)['sd']:.2f} "
              f"P90/P10={sub_sd(cellz, rgap, 'dev', l1)['p90_p10']:.1f} | "
              f"slope gap sd={sub_sd(cellz, sgap, 'dev', l2)['sd']:.2f} "
              f"P90/P10={sub_sd(cellz, sgap, 'dev', l2)['p90_p10']:.1f}")

    v2_t = base_t + [(rgap, 0.03125, "dev", lam_rg), (sgap, 0.03125, "dev", lam_sg)]
    v2 = composites(ts, v2_t, base_p)
    r2 = compare(base, v2)
    print(f"  V2 + 空间一致性（代理 |r−r_pooled| / slope gap，合计 0.0625）  "
          f"Spearman {r2['spearman_med']:.4f}（min {r2['spearman_min']:.4f}）· "
          f"冠军换人 {r2['champ_flips']}/{r2['boards']} · 位移 {r2['mean_shift']:.3f} 位 · "
          f"综合分中位 {r2['score_med']:.2f}")

    # V3 加「雨强分辨力」：中等级别 ETS 的合成
    def grade_synth(levs):
        def f(v):
            xs = [((v.get("graded") or {}).get(l) or {}).get("ets") for l in levs]
            xs = [x for x in xs if x is not None]
            return sum(xs) / len(xs) if xs else None
        return f
    gh = grade_synth(("2", "3"))
    gd_ = grade_synth(("+2", "+3"))
    # 评分轨格子没有 graded，从 diagnostics 按 (轨, 桶, 源) 注入
    dgcells_h = cells_by_bucket({"x": {"p": dg.get("precip_hourly", {})}}, "x", "p")
    dgcells_d = cells_by_bucket({"x": {"p": dg.get("precip_daily", {})}}, "x", "p")
    inj_h = {(b, m): v for b, c in dgcells_h.items() for m, v in c.items()}
    inj_d = {(b, m): v for b, c in dgcells_d.items() for m, v in c.items()}
    lam_gh = solve_lambda(dgcells_h, gh, "pct") or 1.0
    lam_gd = solve_lambda(dgcells_d, gd_, "pct") or 1.0
    print(f"     雨强分辨力 λ 反解：hourly 2~3 级 λ={lam_gh:.2f} · daily +2~+3 级 λ={lam_gd:.2f}")
    print(f"     hourly 子分 sd={sub_sd(dgcells_h, gh, 'pct', lam_gh)['sd']:.2f} "
          f"P90/P10={sub_sd(dgcells_h, gh, 'pct', lam_gh)['p90_p10']:.1f} | "
          f"daily sd={sub_sd(dgcells_d, gd_, 'pct', lam_gd)['sd']:.2f} "
          f"P90/P10={sub_sd(dgcells_d, gd_, 'pct', lam_gd)['p90_p10']:.1f}")

    def p_with_grade(track):
        def f(v):
            # v 是评分轨格子；分级值来自 diagnostics 同 (桶,源)
            return v.get("_grade")
        return f
    # 直接构造：把分级值注入降水格子后再合成
    def composites_with_grade(weight: float) -> dict:
        out = {}
        for track, inj, gg, lg in (("hourly", inj_h, gh, lam_gh),
                                   ("daily", inj_d, gd_, lam_gd)):
            tb = cells_by_bucket(ts, track, "temp")
            pb = cells_by_bucket(ts, track, "precip")
            for b in sorted(set(tb) | set(pb)):
                scores = {}
                for m in set(tb.get(b, {})) | set(pb.get(b, {})):
                    tv, pv = tb.get(b, {}).get(m), pb.get(b, {}).get(m)
                    if tv is None or pv is None:
                        continue
                    gv = gg(inj.get((b, m), {}))
                    pv2 = dict(pv)
                    if gv is not None:
                        pv2["grade_ets"] = gv
                    pp = base_p + [(lambda v: v.get("grade_ets"), weight, "pct", lg)]
                    if track == "daily":
                        hi = dim_score((tv or {}).get("max"), base_t)
                        lo = dim_score((tv or {}).get("min"), base_t)
                        ts_ = None if hi is None or lo is None else (hi + lo) / 2
                    else:
                        ts_ = dim_score(tv, base_t)
                    ps_ = dim_score(pv2, pp)
                    if ts_ is None or ps_ is None:
                        continue
                    scores[m] = (ts_ + ps_) / 2
                if len(scores) >= MIN_MODELS:
                    out[f"{track}:{b}"] = scores
        return out

    for w in (0.10, 0.20):
        v3 = composites_with_grade(w)
        r3 = compare(base, v3)
        print(f"  V3 + 雨强分辨力（2~3 / +2~+3 级 ETS 合成，权重 {w:.2f}）  "
              f"Spearman {r3['spearman_med']:.4f}（min {r3['spearman_min']:.4f}）· "
              f"冠军换人 {r3['champ_flips']}/{r3['boards']} · 位移 {r3['mean_shift']:.3f} 位 · "
              f"综合分中位 {r3['score_med']:.2f}")


if __name__ == "__main__":
    main()

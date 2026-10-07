"""一次性标定脚本：评分换算斜率 λ 的反解与话语权核验（2026-10-05，第一性原理审查 P1-1/P1-2）。

问题：一项指标对名次的实际影响力 = 名义权重 × 换算斜率 × 数据离散度，三者耦合、
缺一不可。旧评分表（RMSE×5、MBE×10、BIAS−1 线性……）只声明了前两个，第三个
由数据的偶然分布决定——2026-09 实测：mae 名义权重 10%，实际话语权 2.5%；
bias 名义 10%，实际 27%。README §5 承诺的权重契约没有兑现。

本脚本在真实存档的分桶指标上做三件事：
  1) 反解：为每个子分选一个换算斜率 λ，使该子分"桶内跨源 sd"= sd*（默认 8 分）。
     影响力 = w×sd = w×8 → 实际话语权占比自动 ≈ 名义权重。sd 口径与审查报告
     一致：(track, 天桶) 内 ≥3 家对桶均值中心化、两轨池化、样本 sd（ddof=1）。
     日榜温度维的 max/min 各算半格（与 daily_temp_score 的打分对象同构）。
     log 项（bias/slope）只在"梯度区"（v>0）上标定：v≤0 的退化格在任何合理 λ
     下都截断 0 分、不携带梯度信息，计入只会用悬崖冒充离散度。
  2) 核验：把标定 λ 代回**生产评分函数**（temp_score / precip_score / 
     daily_temp_score，与榜单同一条代码路径），逐轨重算各子分的实际影响力占比
     ——池化应精确 ≈ 名义权重；分轨残差（同一 λ 无法同时对齐两轨的离散度）
     逐项列出，随月报披露。这一残差与"温度:降水的宏观话语权"同源，已被
     macro_weight_range 敏感性与综合分方差分解披露覆盖。
  3) 对拍：新旧口径的分时效名次（Spearman / 冠军换人 / 平均名次位移），
     作为"口径变更前后名次不直接可比"披露的量化附件。

λ 冻结进 config/stations.yaml 的 eval.score_slopes（config.py 的 DEFAULT_EVAL 是
代码内缺省），季度重标定；λ 变更视为评分口径变更，月报注明。方法论与结论留档
docs/score_slopes.md。

数据来源（二选一）：
  --from-json data.json   已捕获的 track_sources（生产管线原地导出，键形如
                          {"hourly": {"temp": {模型: {桶: 指标}}}, "daily": …}）
  --live                  现场跑 build_report（当前配置窗口，约 2~4 分钟）

运行：
  PYTHONPATH=src python scripts/calibrate_score_slopes.py \
      --from-json /tmp/track_sources_2026-09.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import timedelta

sys.path.insert(0, "src")

import numpy as np

SD_TARGET = 8.0            # 每个子分的目标桶内跨源 sd（分）
MIN_MODELS_PER_BUCKET = 3  # 同审查报告：桶内 ≥3 家才参与离散度估计
# 标定验收阈值（2026-10 复核新增）：任一维的子分超过该比例落在 [95,100] 或 [0,5]
# 饱和带上即判定标定失败。λ 标定只等化"桶内跨源离散度"，不等化"水平"——大片
# 格子贴边意味着该维在 0~100 尺上没有可用梯度：要么给所有源发同一个常数、要么
# 给出一片悬崖，名义权重形同虚设。重标或换族后重新体检，通过才允许冻结。
SATURATION_FAIL_SHARE = 0.30

# 温度/降水各项 → (维度轨道源, 换算族, 梯度区判据)。换算族必须与
# evaluate.py 的 _conv_pct/_conv_dev/_conv_log 一一对应（脚本直接调生产函数核验，
# 这里的族标签只用于反解时的 sd(λ) 模型）。
ITEMS = (
    # (key, dim, family, gradient_only)
    ("acc2", "temp", "pct", False),
    ("acc1", "temp", "pct", False),   # 2026-10 全指标审计后入分（命中轮廓族）
    ("rmse", "temp", "dev", False),
    ("mae", "temp", "dev", False),    # 同（误差幅度族）
    ("r", "temp", "pct", False),
    ("mbe", "temp", "dev", False),
    ("slope", "temp", "log", True),
    ("mbe_bdisp", "temp", "dev", False),   # 站间一致性（2026-10-06 入分，空间一致性族）
    ("ets", "precip", "pct", False),
    ("pod", "precip", "pct", False),
    ("far", "precip", "dev", False),
    ("bias", "precip", "log", True),
    ("amt_mae", "precip", "dev", False),   # 雨量量级（相对口径）
    ("amt_bias", "precip", "log", True),   # 雨量总量比（log 对称）
    ("grade_ets", "precip", "pct", False), # 雨强分辨力（2026-10-06 入分）
)


def load_track_sources(args) -> dict:
    """统一成 {track: {dim: {model: {bucket: 指标}}}}。"""
    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        ts = payload.get("track_sources") or payload
        print(f"数据：{args.from_json}"
              f"（{len(ts.get('hourly', {}).get('temp', {}))} 源 /"
              f" {len(ts.get('daily', {}).get('temp', {}))} 源）")
        return ts
    from weather_eval.config import load_config
    from weather_eval.evaluate import build_report
    from weather_eval.timeutil import now_beijing

    cfg = load_config()
    end = now_beijing().replace(minute=0, second=0, microsecond=0)
    start = end - timedelta(days=args.live_days)
    print(f"数据：现场构建 {start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}（约 2~4 分钟）")
    rep = build_report(cfg.station_ids, cfg.models, cfg.eval, start, end,
                       period_label="calibrate")
    return {"hourly": {"temp": rep["temp_hourly"],
                       "precip": rep["precip_hourly_score"]},
            "daily": {"temp": rep["temp_daily"],
                      "precip": rep["precip_score_daily"]}}


def collect_cells(ts: dict, key: str, dim: str) -> dict:
    """{(track, bucket): {model: value}}；日温度维 max/min 各算半格。

    与评分同构：daily_temp_score 就是 max/min 两个量各自的 temp_score 再平均，
    离散度 therefore 必须在"半格"上量——把整格混在一起会稀释日轨的真实摆幅。
    """
    cells: dict[tuple[str, str], dict[str, float]] = {}
    for trk in ("hourly", "daily"):
        for m, buckets in (ts.get(trk, {}).get(dim) or {}).items():
            for b, met in (buckets or {}).items():
                met = met or {}
                if dim == "temp" and trk == "daily":
                    cands = ((f"{m}·max", (met.get("max") or {}).get(key)),
                             (f"{m}·min", (met.get("min") or {}).get(key)))
                else:
                    cands = ((m, met.get(key)),)
                for mid, v in cands:
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        cells.setdefault((trk, b), {})[mid] = float(v)
    return {k: v for k, v in cells.items() if len(v) >= MIN_MODELS_PER_BUCKET}


def _sub_factory(family: str):
    """子分换算 sub(λ)(v)。三族与生产换算同式：
    pct: v×λ · dev: 100−偏差×λ · log: 100−|log₂v|×λ（v≤0 → 0 分，非缺项）"""
    if family == "pct":
        return lambda lam: (lambda v: v * lam)
    if family == "dev":
        return lambda lam: (lambda v: 100 - abs(v) * lam)
    return lambda lam: (lambda v: (100 - abs(math.log2(v)) * lam) if v > 0 else 0.0)


def bucket_centered(cells: dict, conv, gradient_only: bool):
    """桶内中心化后的子分样本池。返回 (devs, n_used, n_degenerate)。"""
    devs: list[float] = []
    n_deg = 0
    for bucket in cells.values():
        use = [(m, v) for m, v in bucket.items()
               if not gradient_only or v > 0]
        n_deg += len(bucket) - len(use)
        if len(use) < MIN_MODELS_PER_BUCKET:
            continue
        sub = np.clip([conv(v) for _, v in use], 0.0, 100.0)
        devs.extend((sub - sub.mean()).tolist())
    return devs, len(devs), n_deg


def sd_of(devs: list[float]) -> float | None:
    return float(np.std(devs, ddof=1)) if len(devs) > 1 else None


def calibrate(cells: dict, family: str, gradient_only: bool,
              target: float) -> tuple[float | None, float | None, int]:
    """定点迭代反解 λ：sd(λ) = target。无截断时 sd ∝ λ，解析解做起点、
    从下方逼近（饱和区 sd(λ) 非单调，二分可能落到错误的单调分支上）。"""
    mk = _sub_factory(family)
    s1 = sd_of(bucket_centered(cells, mk(1.0), gradient_only)[0])
    if not s1 or s1 <= 0:
        return None, None, 0
    lam = target / s1
    n_used = 0
    for _ in range(200):
        devs, n_used, _ = bucket_centered(cells, mk(lam), gradient_only)
        s = sd_of(devs)
        if not s or s < 1e-9:
            break
        if abs(s - target) < 1e-5:
            break
        nl = lam * target / s
        if abs(nl - lam) < 1e-12:
            break
        lam = nl
    return lam, sd_of(bucket_centered(cells, mk(lam), gradient_only)[0]), n_used


# ----------------------------------------------------------------- 新口径核验
def sub_tensors(ts: dict, models: list[str]):
    """新口径（生产函数）下的两轨子分与维度分：
    {track: {dim: {(model, bucket): score}}}。"""
    import weather_eval.evaluate as ev

    ev.apply_score_slopes(None)   # 代码缺省 = 当前冻结的标定值
    out: dict[str, dict[str, dict]] = {"hourly": {"temp": {}, "precip": {}},
                                       "daily": {"temp": {}, "precip": {}}}
    for trk in ("hourly", "daily"):
        for m in models:
            for b, met in (ts.get(trk, {}).get("temp") or {}).get(m, {}).items():
                if trk == "daily":
                    s = ev.daily_temp_score(met)
                else:
                    s = ev.temp_score(met or {})
                if s is not None:
                    out[trk]["temp"][(m, b)] = s
            for b, met in (ts.get(trk, {}).get("precip") or {}).get(m, {}).items():
                s = ev.precip_score(met or {})
                if s is not None:
                    out[trk]["precip"][(m, b)] = s
    return out


def item_sub_score(ts: dict, key: str, dim: str, family: str, trk: str,
                   lam: float) -> dict:
    """单项子分（按生产同式换算+截断）：{(model, bucket): sub}。"""
    mk = _sub_factory(family)
    conv = mk(lam)

    def clamp(v):
        return max(0.0, min(100.0, conv(v)))

    out = {}
    for m, buckets in (ts.get(trk, {}).get(dim) or {}).items():
        for b, met in (buckets or {}).items():
            met = met or {}
            if dim == "temp" and trk == "daily":
                vals = []
                for half in ("max", "min"):
                    v = (met.get(half) or {}).get(key)
                    if isinstance(v, (int, float)) and math.isfinite(v):
                        vals.append(clamp(v))
                if len(vals) == 2:          # 缺一即缺（与 daily_temp_score 同纪律）
                    out[(m, b)] = sum(vals) / 2
            else:
                v = met.get(key)
                if isinstance(v, (int, float)) and math.isfinite(v):
                    out[(m, b)] = clamp(v)
    return out


def influence_shares(cells_scores: dict, weights: dict) -> dict:
    """逐项影响力占比：桶内中心化 sd × 权重 → 归一。cells_scores: {key: {(m,b): sub}}"""
    infl = {}
    for key, subs in cells_scores.items():
        by_bucket: dict[str, list[float]] = {}
        for (m, b), v in subs.items():
            by_bucket.setdefault(b, []).append(v)
        devs = []
        for vals in by_bucket.values():
            if len(vals) < MIN_MODELS_PER_BUCKET:
                continue
            arr = np.asarray(vals)
            devs.extend((arr - arr.mean()).tolist())
        infl[key] = float(np.std(devs, ddof=1)) if len(devs) > 1 else 0.0
    total = sum(weights[k] * s for k, s in infl.items()) or 1.0
    return {k: weights[k] * infl[k] / total for k in infl}, infl


# ----------------------------------------------------------------- 新旧对拍
OLD_TEMP_PARTS = (   # 2026-10-05 前的口径，只用于对拍
    ("acc2", 0.25, lambda v: v),
    ("rmse", 0.25, lambda v: 100 - v * 5),
    ("r", 0.15, lambda v: v * 100),
    ("acc1", 0.10, lambda v: v),
    ("mae", 0.10, lambda v: 100 - v * 5),
    ("mbe", 0.10, lambda v: 100 - abs(v) * 10),
    ("slope", 0.05, lambda v: 100 - abs(v - 1) * 100),
)
OLD_PRECIP_PARTS = (
    ("ets", 0.35, lambda v: v * 100),
    ("ts", 0.25, lambda v: v * 100),
    ("pod", 0.15, lambda v: v),
    ("far", 0.15, lambda v: 100 - v),
    ("bias", 0.10, lambda v: 100 - abs(v - 1) * 100),
)


def _old_score(parts, met: dict) -> float | None:
    num = den = 0.0
    for k, w, fn in parts:
        v = (met or {}).get(k)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        c = max(0.0, min(100.0, fn(v)))
        num += w * c
        den += w
    return round(num / den, 2) if den else None


def old_composites(ts: dict) -> dict:
    """旧口径分时效综合分：{f'{track}:{b}': {model: score}}（两维齐备才算）。"""
    out = {}
    for trk in ("hourly", "daily"):
        temp_src = ts.get(trk, {}).get("temp") or {}
        precip_src = ts.get(trk, {}).get("precip") or {}
        models = set(temp_src) | set(precip_src)
        buckets = sorted({b for m in models for b in (temp_src.get(m) or {})}
                         | {b for m in models for b in (precip_src.get(m) or {})})
        for b in buckets:
            rows = {}
            for m in models:
                t, p = temp_src.get(m, {}).get(b), precip_src.get(m, {}).get(b)
                tsc = _old_score(OLD_TEMP_PARTS, t)
                psc = _old_score(OLD_PRECIP_PARTS, p)
                if tsc is not None and psc is not None:
                    rows[m] = round((tsc + psc) / 2, 2)
            if rows:
                out[f"{trk}:{b}"] = rows
    return out


def new_composites(ts: dict) -> dict:
    import weather_eval.evaluate as ev
    ev.apply_score_slopes(None)
    out = {}
    for trk in ("hourly", "daily"):
        temp_src = ts.get(trk, {}).get("temp") or {}
        precip_src = ts.get(trk, {}).get("precip") or {}
        models = set(temp_src) | set(precip_src)
        buckets = sorted({b for m in models for b in (temp_src.get(m) or {})}
                         | {b for m in models for b in (precip_src.get(m) or {})})
        for b in buckets:
            rows = {}
            for m in models:
                t, p = temp_src.get(m, {}).get(b), precip_src.get(m, {}).get(b)
                tsc = ev.daily_temp_score(t) if trk == "daily" else ev.temp_score(t or {})
                psc = ev.precip_score(p or {})
                if tsc is not None and psc is not None:
                    rows[m] = round((tsc + psc) / 2, 2)
            if rows:
                out[f"{trk}:{b}"] = rows
    return out


def spearman(a: list[float], b: list[float]) -> float | None:
    ra = np.argsort(np.argsort([-x for x in a]))
    rb = np.argsort(np.argsort([-x for x in b]))
    if len(a) < 3 or np.std(ra) == 0 or np.std(rb) == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def compare_boards(old: dict, new: dict) -> list[dict]:
    rows = []
    for bk in sorted(set(old) & set(new), key=lambda x: (x.split(":")[0], int(x.split(":")[1][:-1]))):
        common = set(old[bk]) & set(new[bk])
        if len(common) < 4:
            continue
        ov = [old[bk][m] for m in sorted(common)]
        nv = [new[bk][m] for m in sorted(common)]
        rho = spearman(ov, nv)
        oc = max(old[bk], key=old[bk].get)
        nc = max(new[bk], key=new[bk].get)
        o_order = {m: i for i, m in enumerate(
            sorted(common, key=lambda x: -old[bk][x]))}
        n_order = {m: i for i, m in enumerate(
            sorted(common, key=lambda x: -new[bk][x]))}
        shift = float(np.mean([abs(o_order[m] - n_order[m]) for m in common]))
        rows.append({"board": bk, "n": len(common), "spearman": rho,
                     "champion_changed": oc != nc, "old_champ": oc, "new_champ": nc,
                     "mean_rank_shift": shift})
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-json", help="已捕获的 track_sources JSON")
    src.add_argument("--live", action="store_true", help="现场跑 build_report")
    ap.add_argument("--live-days", type=int, default=30)
    ap.add_argument("--target", type=float, default=SD_TARGET,
                    help=f"每个子分的目标桶内跨源 sd（默认 {SD_TARGET} 分）")
    ap.add_argument("--strict", action="store_true",
                    help="标定验收（§1.5）发现饱和带超标时以非零退出（默认只告警）")
    args = ap.parse_args()

    ts = load_track_sources(args)
    models = sorted({m for trk in ("hourly", "daily")
                     for dim in ("temp", "precip")
                     for m in (ts.get(trk, {}).get(dim) or {})})

    # ---- 1) 反解 λ ----
    print(f"\n=== 1) 反解：目标 sd* = {args.target} 分"
          f"（桶内 ≥{MIN_MODELS_PER_BUCKET} 家中心化、两轨池化、ddof=1） ===")
    print(f"{'项':7s} {'族':4s} {'sd(λ=1)':>8s} {'λ*':>9s} {'每差X扣1分':>11s}"
          f" {'饱和线':>8s} {'梯度格':>6s} {'退化格':>5s}")
    lambdas: dict[str, float] = {}
    for key, dim, family, grad in ITEMS:
        cells = collect_cells(ts, key, dim)
        lam, sd_final, n_used = calibrate(cells, family, grad, args.target)
        if lam is None:
            print(f"{key:7s} {family:4s}  -- 数据不足，保持缺省 --")
            continue
        lambdas[key] = lam
        mk = _sub_factory(family)
        s1 = sd_of(bucket_centered(cells, mk(1.0), grad)[0])
        deg = sum(1 for bu in cells.values() for v in bu.values() if grad and v <= 0)
        if family == "log":
            sat = f"±{2 ** (100.0 / lam):.0f} 倍"
            per = f"{1 / lam:.2f} 倍"
        else:
            sat = ("—" if family == "pct" else f"{100 / lam:.1f}")
            per = f"{1 / lam:.2f}"
        print(f"{key:7s} {family:4s} {s1:8.3f} {lam:9.4f} {per:>9s} {sat:>8s}"
              f" {n_used:6d} {deg:5d}")

    # ---- 1.5) 标定验收：子分水平体检 ----
    # 反解只保证"桶内跨源 sd = 8"，不保证水平可用。用候选 λ 逐项体检子分的
    # 水平分布：>SATURATION_FAIL_SHARE 的格子落在 [95,100] 或 [0,5] 即告警。
    # 默认只告警不失败：log 族指标（slope/amt_bias 类）的"贴上沿"常是数据本性
    # （多数源在该维确实接近完美），换 λ 无从 cure，硬失败会让重标定永远跑不完；
    # 需要把体检当门禁（重标定 CI 化时）加 --strict。体检用**未中心化**的子分
    # ——水平问题恰恰在中心化时被抹掉，那是旧流程看不见它的原因。
    print(f"\n=== 1.5) 标定验收：子分水平体检（>{SATURATION_FAIL_SHARE:.0%} 落"
          f" [95,100]/[0,5]{' 即失败' if args.strict else '（告警）'}） ===")
    print(f"{'项':7s} {'均值':>7s} {'P10':>7s} {'P90':>7s} {'∈[95,100]':>9s} {'∈[0,5]':>7s}")
    sat_fail: list[str] = []
    for key, dim, family, grad in ITEMS:
        cells = collect_cells(ts, key, dim)
        lam = lambdas.get(key)
        if lam is None or not cells:
            continue
        mk = _sub_factory(family)
        subs = [mk(lam)(v) for bu in cells.values()
                for v in bu.values() if not (grad and v <= 0)]
        if len(subs) < 30:
            continue
        a = np.clip(np.asarray(subs, dtype=float), 0.0, 100.0)
        hi = float(np.mean(a >= 95.0))
        lo = float(np.mean(a <= 5.0))
        print(f"{key:7s} {a.mean():7.2f} {np.percentile(a, 10):7.2f} "
              f"{np.percentile(a, 90):7.2f} {hi:9.1%} {lo:7.1%}")
        if hi > SATURATION_FAIL_SHARE or lo > SATURATION_FAIL_SHARE:
            sat_fail.append(key)
    if sat_fail:
        msg = (f"{', '.join(sat_fail)} 的子分超 {SATURATION_FAIL_SHARE:.0%} 落在饱和带"
               "——检查该维是否有可用梯度；log 族贴上沿若属数据本性可豁免，"
               "换族/调锚点需连带重估权重。")
        if args.strict:
            print(f"\n✗ 标定验收失败（--strict）：{msg}")
            sys.exit(1)
        print(f"\n⚠ 标定验收告警：{msg}")
    else:
        print("✓ 标定验收通过：所有维度的饱和占比均在阈值内")

    # ---- 2) 核验（生产评分函数 + 冻结值）----
    print("\n=== 2) 核验：生产评分函数下的逐项影响力占比（应 ≈ 名义权重） ===")
    import weather_eval.evaluate as ev
    ev.apply_score_slopes({k: round(v, 3) for k, v in lambdas.items()})
    lambdas = dict(ev.SCORE_SLOPES)   # 用四舍五入后的值核验（与冻结值一致）
    for trk in ("hourly", "daily"):
        label = "小时轨" if trk == "hourly" else "日轨（max/min 半格）"
        print(f"\n-- {label} --")
        print(f"{'项':7s} {'名义':>6s} {'实际':>6s} {'名义/实际':>8s} {'桶内sd':>7s}")
        for dim, parts, wname in (("temp", ev.TEMP_SCORE_PARTS, "温度"),
                                  ("precip", ev.PRECIP_SCORE_PARTS, "降水")):
            fam = {k: f for k, _d, f, _g in ITEMS}
            cell_scores = {}
            weights = {}
            for k, w, *_ in parts:
                weights[k] = w
                cell_scores[k] = item_sub_score(ts, k, dim, fam[k], trk,
                                                lambdas[k])
            shares, infl = influence_shares(cell_scores, weights)
            for k, _w, *_ in parts:
                print(f"{k:7s} {weights[k]:6.3f} {shares[k]:6.3f}"
                      f" {shares[k] / weights[k]:8.2f} {infl[k]:7.2f}")

    # 综合分的话语权分解（同审查报告 P1-3 的口径）
    print("\n=== 2b) 综合分（温度:降水）方差分解（两维齐备格、桶内中心化） ===")
    dim_scores = sub_tensors(ts, models)
    for trk in ("hourly", "daily"):
        both = set(dim_scores[trk]["temp"]) & set(dim_scores[trk]["precip"])
        if len(both) < 8:
            continue
        by_b: dict[str, list[tuple[float, float]]] = {}
        for pair in both:
            by_b.setdefault(pair[1], []).append(
                (dim_scores[trk]["temp"][pair], dim_scores[trk]["precip"][pair]))
        t_c, p_c = [], []
        for vals in by_b.values():
            if len(vals) < MIN_MODELS_PER_BUCKET:
                continue
            ta = np.asarray([x for x, _ in vals])
            pa = np.asarray([y for _, y in vals])
            t_c.extend((ta - ta.mean()).tolist())
            p_c.extend((pa - pa.mean()).tolist())
        t_c, p_c = np.asarray(t_c), np.asarray(p_c)
        vc = (t_c.var() + p_c.var() + 2 * (t_c * p_c).mean()) / 4
        print(f"  {('小时轨' if trk == 'hourly' else '日轨')}：温度 "
              f"{t_c.var() / 4 / vc * 100:.0f}% · 降水 {p_c.var() / 4 / vc * 100:.0f}%"
              f" · 协方差 {(t_c * p_c).mean() / 2 / vc * 100:.0f}%"
              f"（格子 {len(t_c)}，维度 sd {t_c.std(ddof=1):.1f} vs {p_c.std(ddof=1):.1f}）")

    # ---- 3) 新旧对拍 ----
    print("\n=== 3) 新旧口径对拍（分时效综合分名次；两轨、两维齐备） ===")
    cmp_rows = compare_boards(old_composites(ts), new_composites(ts))
    n_chg = sum(r["champion_changed"] for r in cmp_rows)
    rhos = [r["spearman"] for r in cmp_rows if r["spearman"] is not None]
    print(f"  榜数 {len(cmp_rows)}（≥4 家同台的桶）· 冠军换人 {n_chg}"
          f" · Spearman 中位 {np.median(rhos):.3f}（min {min(rhos):.3f}）"
          f" · 平均名次位移中位 {np.median([r['mean_rank_shift'] for r in cmp_rows]):.2f} 位")
    for r in cmp_rows:
        if r["champion_changed"]:
            print(f"    {r['board']:12s} 冠军 {r['old_champ']} → {r['new_champ']}"
                  f"（ρ={r['spearman']:.2f}, n={r['n']}）")

    # ---- 4) 冻结片段 ----
    print("\n=== 4) 冻结片段（粘进 config/stations.yaml 的 eval 段；"
          "config.py 的 DEFAULT_EVAL 同步） ===")
    print("  score_slopes:")
    vern = {"acc2": "±2°C 命中率每差 %.2f 个百分点扣 1 分",
            "acc1": "±1°C 命中率每差 %.2f 个百分点扣 1 分",
            "rmse": "每差 %.3f °C 扣 1 分（≥%.1f°C 记 0 分）",
            "mae": "每差 %.3f °C 扣 1 分（≥%.1f°C 记 0 分）",
            "r": "r 每差 %.3f 扣 1 分",
            "mbe": "每差 %.3f °C 扣 1 分（≥%.1f°C 记 0 分）",
            "slope": "幅度每偏 %.2f 倍扣 1 分（超/欠对称，±%.0f 倍记 0 分）",
            "ets": "ETS 每差 %.3f 扣 1 分",
            "pod": "命中率每差 %.2f 个百分点扣 1 分",
            "far": "空报率每高 %.2f 个百分点扣 1 分",
            "bias": "报雨频率每偏 %.2f 倍扣 1 分（超/欠报对称，±%.0f 倍记 0 分）",
            "amt_mae": "相对雨量误差每高 %.3f 扣 1 分（≥%.1f 倍记 0 分）",
            "amt_bias": "雨量总量每偏 %.2f 倍扣 1 分（超/欠报对称，±%.0f 倍记 0 分）",
            "mbe_bdisp": "站间偏差离散度每差 %.3f°C 扣 1 分（≥%.1f°C 记 0 分）",
            "grade_ets": "雨强分辨力（中雨/大雨档 ETS 加权）每差 %.3f 扣 1 分"}
    for key, _dim, fam, _g in ITEMS:
        if key not in lambdas:
            continue
        lam = lambdas[key]
        v = vern[key]
        if fam == "log":
            txt = v % (1 / lam, 2 ** (100 / lam))
        elif key in ("rmse", "mae", "mbe", "amt_mae", "mbe_bdisp"):
            txt = v % (1 / lam, 100 / lam)
        else:
            txt = v % (1 / lam)
        print(f"    {key}: {round(lam, 3):<8g} # {txt}")


if __name__ == "__main__":
    main()

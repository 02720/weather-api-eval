"""统计推断层（2026-09-06 新增，对抗式审查 P0-2 / P1-2 / P1-3 的落地）。

此前的评估对"数据没被污染"防御严密，但对"结论是否有意义"没有防御：
排行榜给出精确到 0.01 分的名次，却没有任何不确定性度量。本模块补上三件事：

1. **有效样本量 n_eff**（P1-2）：逐小时气温误差强自相关（实测 lag-1 ρ≈0.64~0.91），
   名义样本量把同一信息重复计数，最大高估 20.5 倍。n_eff = n·(1−ρ₁)/(1+ρ₁)。
2. **站内相关系数的 Fisher-z 合并**（P1-3）：跨站池化的 r 混入"能否复现站间气候
   差异"这一容易得多的任务（实测与站内口径差最多 0.084，在 15% 权重下约合 1.3
   个综合分）。改为各站内部算 r，再按 Fisher-z 变换加权合并。
3. **按天分块 bootstrap + 权重敏感性**（P0-2）：以"天"为重采样块（块长 1 天），
   同时给出综合分的 90% 置信区间、各源成为冠军的频率；权重扰动 ±40% 下冠军
   在不同源之间的轮换分布。块长 1 天同时化解了 P1-2 的自相关问题——同一天内
   高度相关的误差永不跨块拆散。

与 cyeva 的口径一致性（第一性原理：快速路径与全量路径绝不给出两套数字）：
实测 cyeva 类方法在本项目的调用方式下等价于「剔除任一侧为 NaN 的样本对 →
原值计算 → 结果 round(4)」（source_round_digit 装饰器因所有参数经关键字传递而
不生效；result_round_digit(4) 生效）。本模块的快速路径复刻同一口径并补测试。
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np

# n_eff 的最小估计样本：序列短于该值时自相关估计噪声大于信号，直接返回 n
NEFF_MIN_SERIES = 30
# ρ₁ 的截断：ρ→1 时 (1−ρ)/(1+ρ) 发散，截断避免单个高自相关序列炸掉 n_eff
RHO_CLAMP = 0.95
# 站内 r/slope 参与合并的最小站内样本量（与点估计路径一致）
GROUP_MIN_N = 30


# ------------------------------------------------------------------ 有效样本量
def effective_n(err: np.ndarray) -> int:
    """自相关校正后的有效样本量：n_eff = n·(1−ρ₁)/(1+ρ₁)。

    err 是按时间排序的误差（或事件指示）序列。n < NEFF_MIN_SERIES 时不估自相关
    （估计量本身太噪，直接返回 n 不惩罚小样本）；ρ₁ 估计为 NaN（方差为 0 等
    退化情形）同样返回 n。结果截断到 [1, n]。
    """
    e = np.asarray(err, dtype=float)
    e = e[np.isfinite(e)]
    n = int(e.size)
    if n < NEFF_MIN_SERIES + 1 or n <= 2:
        return n
    a, b = e[:-1], e[1:]
    va, vb = a.var(), b.var()
    if va <= 0 or vb <= 0:
        return n
    rho = float(np.cov(a, b)[0, 1] / np.sqrt(va * vb))
    if not np.isfinite(rho):
        return n
    rho = max(-RHO_CLAMP, min(RHO_CLAMP, rho))
    n_eff = int(round(n * (1 - rho) / (1 + rho)))
    return max(1, min(n, n_eff))


def n_eff_from_station_series(series_by_station: dict[str, list[float]]) -> int:
    """各站独立估计 n_eff 后求和（跨站拼接会造出人为的序列跳变，低估 ρ）。"""
    total = 0
    for vals in series_by_station.values():
        total += effective_n(np.asarray(vals, dtype=float))
    return total


# ------------------------------------------------------------------ 站内 r/slope 合并
def fisher_z_combine(rs: list[float], ns: list[int]) -> float | None:
    """站内相关系数按 Fisher-z 变换加权平均：r = tanh(Σ nᵢ·z(rᵢ) / Σ nᵢ)。

    相关系数不可直接平均（有界量，方差异质）；Fisher-z 把 r 映射到近似无界的
    z 尺度后再加权，是跨组合并相关系数的标准做法。rs 中 None（组内退化）跳过。
    """
    num = den = 0.0
    for r, n in zip(rs, ns):
        if r is None or n is None or n <= 0:
            continue
        r = max(-0.999, min(0.999, float(r)))
        num += n * 0.5 * np.log((1 + r) / (1 - r))
        den += n
    if den <= 0:
        return None
    return float(np.tanh(num / den))


def weighted_mean_combine(vals: list[float | None], ns: list[int]) -> float | None:
    """按样本量加权平均（回归斜率的跨站合并：斜率无界，直接 n 加权）。"""
    num = den = 0.0
    for v, n in zip(vals, ns):
        if v is None or n is None or n <= 0:
            continue
        num += n * float(v)
        den += n
    if den <= 0:
        return None
    return num / den


# ------------------------------------------------------------------ 快速指标（与 cyeva 同口径）
def _nan_pair_mask(o: np.ndarray, f: np.ndarray) -> np.ndarray:
    """cyeva drop_nan 的等价掩膜：只剔 NaN、保留 inf（x != x 判定）。"""
    return ~(np.isnan(o) | np.isnan(f))


def _round4(v: float) -> float:
    return round(float(v), 4)


def r_slope_numpy(x: np.ndarray, y: np.ndarray) -> tuple[float | None, float | None]:
    """相关系数与回归斜率（x=预报, y=观测；cyeva 口径 linregress(forecast, obs)）。

    用充分统计量公式，与 scipy.stats.linregress 同一数学定义；NaN 对按 drop_nan
    剔除。退化（零方差）返回 (None, None)。
    """
    m = _nan_pair_mask(x, y)
    if m.sum() < 3:
        return None, None
    xs, ys = x[m], y[m]
    n = xs.size
    sx, sy = xs.sum(), ys.sum()
    sxy = float((xs * ys).sum())
    sxx = float((xs * xs).sum())
    syy = float((ys * ys).sum())
    den_x = n * sxx - sx * sx
    den_y = n * syy - sy * sy
    if den_x <= 0 or den_y <= 0:
        return None, None
    cov = n * sxy - sx * sy
    r = cov / np.sqrt(den_x * den_y)
    slope = cov / den_x
    return _round4(r), _round4(slope)


def temp_core_numpy(o: np.ndarray, f: np.ndarray) -> dict:
    """温度核心指标的 numpy 快速路径（cyeva 同口径：剔 NaN 对、结果 round4）。

    返回 rmse/mae/mbe/acc1/acc2/r/slope（值为 None 或 round4 浮点）与 n。
    用于 1..72h 逐时效曲线（页面只消费 rmse/acc2）与 bootstrap 的统计基础。
    """
    o = np.asarray(o, dtype=float)
    f = np.asarray(f, dtype=float)
    m = _nan_pair_mask(o, f)
    n = int(m.sum())
    out = {"n": n, "rmse": None, "mae": None, "mbe": None,
           "acc1": None, "acc2": None, "r": None, "slope": None}
    if n == 0:
        return out
    ov, fv = o[m], f[m]
    e = fv - ov
    out["rmse"] = _round4(np.sqrt(np.mean(e * e)))
    out["mae"] = _round4(np.mean(np.abs(e)))
    out["mbe"] = _round4(np.mean(e))
    out["acc2"] = _round4(100.0 * float(np.mean(np.abs(ov - fv) <= 2)))
    out["acc1"] = _round4(100.0 * float(np.mean(np.abs(ov - fv) <= 1)))
    r, slope = r_slope_numpy(fv, ov)
    out["r"], out["slope"] = r, slope
    return out


def binary_counts(o: np.ndarray, f: np.ndarray, thr: float) -> tuple[int, int, int, int]:
    """晴雨二分类列联计数 (hits, false_alarms, misses, correct_negatives)。

    与 cyeva calc_threshold_* 类方法同口径：剔除 NaN 对后**原值**与阈值比较
    （cyeva 的 threshold_binarize 不做源舍入；本项目旧手工路径 round2 二值化是
    与类路径不一致的口径分裂，已纠正）。
    """
    o = np.asarray(o, dtype=float)
    f = np.asarray(f, dtype=float)
    m = _nan_pair_mask(o, f)
    ob = o[m] >= thr
    fb = f[m] >= thr
    h = int((ob & fb).sum())
    fa = int((~ob & fb).sum())
    mi = int((ob & ~fb).sum())
    c = int((~ob & ~fb).sum())
    return h, fa, mi, c


def binary_metrics_from_counts(h: int, fa: int, mi: int, c: int) -> dict:
    """列联计数 → 二分类指标（与 cyeva threshold_* 同定义，结果 round4）。

    ts/ets 为 0~1 比值；acc/pod/far 为 0~100 百分数；bias 为比值——
    与既有 precip_metrics 的单位约定一致，评分换算函数依赖这一约定。
    分母为 0 的项返回 NaN（调用方按缺项处理），与 cyeva 的 nan→None 链一致。
    """
    n = h + fa + mi + c
    out = {"hits": h, "false_alarms": fa, "misses": mi, "correct_negatives": c,
           "n": n, "acc": np.nan, "pod": np.nan, "far": np.nan,
           "ts": np.nan, "ets": np.nan, "bias": np.nan}
    if n == 0:
        return out
    out["acc"] = 100.0 * (h + c) / n
    if h + mi:
        out["pod"] = 100.0 * h / (h + mi)
    if h + fa:
        out["far"] = 100.0 * fa / (h + fa)
    if h + fa + mi:
        out["ts"] = h / (h + fa + mi)
    href = (h + mi) * (h + fa) / n
    if h + fa + mi - href > 0:
        out["ets"] = (h - href) / (h + fa + mi - href)
    if h + mi:
        out["bias"] = (h + fa) / (h + mi)
    return {k: (round(v, 4) if isinstance(v, float) and np.isfinite(v) else v)
            for k, v in out.items()}


# ------------------------------------------------------------------ 按天分块 bootstrap
# 温度充分统计量的列含义（逐样本可加，故"天"内可预聚合、重采样时按权重求和）
_TEMP_STATS = ("n", "se2", "ae", "se", "h1", "h2", "sx", "sy", "sxy", "sxx", "syy")
# 降水二分类列联计数的列含义
_RAIN_STATS = ("h", "fa", "mi", "c")


def build_day_stat_tables(
    hourly: list[dict], daily: list[dict], models: list[str],
    n_buckets: int, rain_thr: float,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """把逐小时/按天记录压缩成（天 × 模型 × 桶 × 站）的充分统计量稠密表。

    返回 (days, temp_table, rain_table)：
      temp_table[model, bucket, station, day, 11]（温度 11 项可加统计量）
      rain_table[model, bucket, station, day, 4]（日降水二分类计数，阈值 rain_thr）
    天的全集取两类记录的并集；无数据的天/桶保持全 0，重采样时自然按"无样本"
    处理。桶对齐：温度用逐小时天桶（lead），降水用按天日偏移（offset）——
    两者都是"起报后第 N 天"，与排行榜的天桶语义一致。
    """
    days = sorted({r["valid_iso"][:10] for r in hourly} |
                  {r["valid_day"] for r in daily})
    day_idx = {d: i for i, d in enumerate(days)}
    model_idx = {m: i for i, m in enumerate(models)}
    station_idx: dict[str, int] = {}
    for r in hourly:
        station_idx.setdefault(r["station"], len(station_idx))
    for r in daily:
        station_idx.setdefault(r["station"], len(station_idx))
    n_s = max(1, len(station_idx))
    n_m = max(1, len(models))
    T = np.zeros((n_m, n_buckets, n_s, len(days), len(_TEMP_STATS)), dtype=np.float64)
    R = np.zeros((n_m, n_buckets, n_s, len(days), len(_RAIN_STATS)), dtype=np.float64)

    rows_t, idx_t = [], []
    rows_r, idx_r = [], []
    for r in hourly:
        mi = model_idx.get(r["model"])
        if mi is None or not (1 <= r["bucket"] <= n_buckets):
            continue
        o, f = r["temp_obs"], r["temp_fcst"]
        if o is not None and f is not None:
            e = f - o
            rows_t.append([1.0, e * e, abs(e), e,
                           1.0 if abs(o - f) <= 1 else 0.0,
                           1.0 if abs(o - f) <= 2 else 0.0,
                           f, o, f * o, f * f, o * o])
            idx_t.append((mi, r["bucket"] - 1, station_idx[r["station"]],
                          day_idx[r["valid_iso"][:10]]))
    if rows_t:
        rows_t = np.asarray(rows_t)
        idx_t = np.asarray(idx_t)
        for k in range(len(_TEMP_STATS)):
            np.add.at(T[:, :, :, :, k], tuple(idx_t.T), rows_t[:, k])

    for r in daily:
        mi = model_idx.get(r["model"])
        if mi is None or not (1 <= r["offset"] <= n_buckets):
            continue
        o, f = r["rain_obs"], r["rain_fcst"]
        if o is None or f is None:
            continue
        ob, fb = o >= rain_thr, f >= rain_thr
        row = [1.0 if (ob and fb) else 0.0, 1.0 if (not ob and fb) else 0.0,
               1.0 if (ob and not fb) else 0.0, 1.0 if (not ob and not fb) else 0.0]
        rows_r.append(row)
        idx_r.append((mi, r["offset"] - 1, station_idx[r["station"]],
                      day_idx[r["valid_day"]]))
    if rows_r:
        rows_r = np.asarray(rows_r)
        idx_r = np.asarray(idx_r)
        for k in range(len(_RAIN_STATS)):
            np.add.at(R[:, :, :, :, k], tuple(idx_r.T), rows_r[:, k])

    return days, T, R


def _score_from_parts(values: dict[str, np.ndarray], parts) -> np.ndarray:
    """按评分权重表把指标值数组组合成 0~100 子分；缺项（NaN）按剩余权重归一。

    values: 指标键 → 任意形状数组（NaN = 缺项）。返回同形状（全缺处 NaN）。
    """
    num = den = None
    for key, w, _label, _mp, fn in parts:
        v = values.get(key)
        if v is None:
            continue
        with np.errstate(invalid="ignore"):
            sub = fn(v)
        sub = np.clip(sub, 0.0, 100.0)
        valid = np.isfinite(sub)
        sub = np.where(valid, sub, 0.0)
        num = w * sub if num is None else num + w * sub
        den = w * valid if den is None else den + w * valid
    if num is None:
        return np.full(next(iter(values.values())).shape if values else (), np.nan)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _temp_scores_from_aggregate(A: np.ndarray, temp_parts) -> np.ndarray:
    """聚合温度统计量 (run?, m, b, s, 11) → 站内合并后的温度分 (run?, m, b)。

    r/slope 先站内（n≥GROUP_MIN_N 的站）计算、再 Fisher-z / n 加权合并；
    无站达标时回退池化口径（与点估计的 temp_metrics 行为一致）。
    """
    has_run = A.ndim == 5
    if not has_run:
        A = A[None, ...]
    n_run = A.shape[0]
    n_m, n_b, n_s = A.shape[1], A.shape[2], A.shape[3]
    n = A[..., 0]
    se2, ae, se = A[..., 1], A[..., 2], A[..., 3]
    h1, h2 = A[..., 4], A[..., 5]
    sx, sy, sxy, sxx, syy = A[..., 6], A[..., 7], A[..., 8], A[..., 9], A[..., 10]

    with np.errstate(invalid="ignore", divide="ignore"):
        cov = n * sxy - sx * sy
        den_x = n * sxx - sx * sx
        den_y = n * syy - sy * sy
        r_st = np.where((den_x > 0) & (den_y > 0),
                        cov / np.where((den_x > 0) & (den_y > 0),
                                       np.sqrt(den_x * den_y), 1.0), np.nan)
        slope_st = np.where(den_x > 0, cov / np.where(den_x > 0, den_x, 1.0), np.nan)

    def _combine_station(stat: np.ndarray, how: str) -> np.ndarray:
        """(run, m, b, s) → (run, m, b)：站内有效值加权合并，空则池化回退。"""
        stat = np.where(n >= GROUP_MIN_N, stat, np.nan)
        w = np.where(np.isfinite(stat), n, 0.0)
        wsum = w.sum(axis=-1)
        if how == "fisher":
            z = np.arctanh(np.clip(stat, -0.999, 0.999))
            comb = np.tanh(
                np.nansum(np.where(np.isfinite(z), w * z, 0.0), axis=-1)
                / np.where(wsum > 0, wsum, np.nan))
        else:
            comb = (np.nansum(np.where(np.isfinite(stat), w * stat, 0.0), axis=-1)
                    / np.where(wsum > 0, wsum, np.nan))
        # 无站达标 → 池化回退（把各站统计量直接加总重算）
        N = n.sum(axis=-1)
        if how == "fisher":
            sxT, syT, sxyT = sx.sum(-1), sy.sum(-1), sxy.sum(-1)
            sxxT, syyT = sxx.sum(-1), syy.sum(-1)
            covT = N * sxyT - sxT * syT
            dxT = N * sxxT - sxT * sxT
            dyT = N * syyT - syT * syT
            pooled = np.where((dxT > 0) & (dyT > 0),
                              covT / np.where((dxT > 0) & (dyT > 0),
                                              np.sqrt(dxT * dyT), 1.0), np.nan)
        else:
            sxT, syT, sxyT = sx.sum(-1), sy.sum(-1), sxy.sum(-1)
            sxxT = sxx.sum(-1)
            covT = N * sxyT - sxT * syT
            dxT = N * sxxT - sxT * sxT
            pooled = np.where(dxT > 0, covT / np.where(dxT > 0, dxT, 1.0), np.nan)
        return np.where(wsum > 0, comb, pooled)

    r = _combine_station(r_st, "fisher")
    slope = _combine_station(slope_st, "mean")
    # 非 r/slope 指标按"各站统计量直接加总"池化（等价于合并样本重算）——
    # RMSE 是非线性量，绝不能对站值做加权平均（那是另一口径）
    N = n.sum(axis=-1)
    Nz = np.where(N > 0, N, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled_values = {
            "rmse": np.sqrt(se2.sum(-1) / Nz),
            "mae": ae.sum(-1) / Nz,
            "mbe": se.sum(-1) / Nz,
            "acc1": 100.0 * h1.sum(-1) / Nz,
            "acc2": 100.0 * h2.sum(-1) / Nz,
            "r": r, "slope": slope,
        }
    scores = _score_from_parts(pooled_values, temp_parts)
    return scores if has_run else scores[0]


def _rain_scores_from_aggregate(A: np.ndarray, precip_parts) -> np.ndarray:
    """聚合列联计数 (run?, m, b, s, 4) → 降水分 (run?, m, b)（跨站计数直接相加）。"""
    has_run = A.ndim == 5
    if not has_run:
        A = A[None, ...]
    tot = A.sum(axis=-2)          # 站维求和：二分类计数跨站可加
    h, fa, mi, c = tot[..., 0], tot[..., 1], tot[..., 2], tot[..., 3]
    n = h + fa + mi + c
    with np.errstate(invalid="ignore", divide="ignore"):
        nz = lambda x: np.where(x > 0, x, np.nan)
        acc = 100.0 * (h + c) / np.where(n > 0, n, np.nan)
        pod = 100.0 * h / nz(h + mi)
        far = 100.0 * fa / nz(h + fa)
        ts = h / nz(h + fa + mi)
        bias = (h + fa) / nz(h + mi)
        href = (h + mi) * (h + fa) / np.where(n > 0, n, np.nan)
        ets = (h - href) / ((h + fa + mi) - href)
    values = {"acc": acc, "pod": pod, "far": far, "ts": ts, "ets": ets, "bias": bias}
    scores = _score_from_parts(values, precip_parts)
    return scores if has_run else scores[0]


def day_block_bootstrap(
    hourly: list[dict], daily: list[dict], models: list[str], n_buckets: int,
    rain_thr: float, temp_parts, precip_parts, min_sample: int,
    runs: int = 500, seed: int = 20260906,
    eligible: list[bool] | None = None,
) -> dict[str, dict[str, Any]]:
    """按天分块 bootstrap（块长 1 天）：综合分（天桶 macro 平均）的不确定性。

    返回 {model: {"ci90": [lo, hi] | None, "champion_pct": float,
                  "sig_vs_top": bool | None}}。
    sig_vs_top：与点估计冠军的得分差的 90% 区间是否不含 0（True = 差异显著）。
    同一次重采样同时驱动温度（逐小时）与降水（按天）两条轨道与所有模型，
    因此区间/显著性是**配对**的——模型间比较不受抽样噪声交叉污染。

    eligible：入围冠军竞争的模型（样本量门槛 + 维度齐备）。冠军频率、点估计
    冠军与显著性只在入围者上计算；未入围模型仍给出自己的置信区间供参考。
    """
    if eligible is None:
        eligible = [True] * len(models)
    if not hourly and not daily:
        return {m: {"ci90": None, "champion_pct": 0.0, "sig_vs_top": None}
                for m in models}
    days, T, R = build_day_stat_tables(hourly, daily, models, n_buckets, rain_thr)
    n_d = max(1, len(days))
    rng = np.random.default_rng(seed)
    # 每次重复抽 n_d 天（有放回）→ 每天的出现次数（多重集权重）
    W = np.zeros((runs, n_d))
    for i in range(runs):
        idx = rng.integers(0, n_d, n_d)
        np.add.at(W[i], idx, 1.0)

    # 聚合：At[run, m, b, s, stat] = Σ_d W[run, d]·T[m, b, s, d, stat]
    # （r/slope 需要站级中间量，聚合保留站维，站内合并放在 _temp_scores_from_aggregate）
    At = np.empty((runs,) + T.shape[:3] + (len(_TEMP_STATS),))
    for k in range(len(_TEMP_STATS)):
        At[..., k] = np.einsum("rd,mbsd->rmbs", W, T[:, :, :, :, k])
    Ar = np.empty((runs,) + R.shape[:3] + (len(_RAIN_STATS),))
    for k in range(len(_RAIN_STATS)):
        Ar[..., k] = np.einsum("rd,mbsd->rmbs", W, R[:, :, :, :, k])

    temp_scores = _temp_scores_from_aggregate(At, temp_parts)     # (run, m, b)
    rain_scores = _rain_scores_from_aggregate(Ar, precip_parts)   # (run, m, b)
    # 样本量门槛在重采样里与点估计**逐维同构**：温度/降水各自的样本数非零但
    # < min_sample 时该维缺项（NaN），桶综合分 = 仍在场的维度单独承担——
    # 与点估计 overall_score 的"缺项不计"完全一致，CI 中心才不会系统性偏移。
    n_temp_b = At[..., 0].sum(axis=-1)                            # (run, m, b)
    n_rain_b = Ar[..., 0].sum(axis=-1)
    temp_scores = np.where((n_temp_b > 0) & (n_temp_b < min_sample),
                           np.nan, temp_scores)
    rain_scores = np.where((n_rain_b > 0) & (n_rain_b < min_sample),
                           np.nan, rain_scores)
    # 某桶在天块重采样中一次都没被抽到是合法状态（尤其单天桶）——该桶全 NaN，
    # nanmean 的 "empty slice" RuntimeWarning 属预期，局部抑制。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore"):
            both = np.stack([temp_scores, rain_scores])           # (2, run, m, b)
            bucket = np.nanmean(both, axis=0)                     # (run, m, b)
            macro = np.nanmean(bucket, axis=-1)                   # (run, m)

    out: dict[str, dict[str, Any]] = {}
    point_top = None
    for mi, m in enumerate(models):
        scores = macro[:, mi]
        finite = scores[np.isfinite(scores)]
        ci = None
        if finite.size >= max(5, runs // 10):
            lo, hi = np.percentile(finite, [5, 95])
            ci = [round(float(lo), 2), round(float(hi), 2)]
        out[m] = {"ci90": ci, "champion_pct": 0.0, "sig_vs_top": None}
        if not eligible[mi] or ci is None:
            continue
        mean_score = float(np.nanmean(scores))
        if point_top is None or point_top[1] < mean_score:
            point_top = (mi, mean_score)
    # 冠军频率（逐 run 的 nan-aware 最大；某 run 全模型无分时该 run 不计冠军）。
    # 只有入围者可成为冠军（未入围者不应凭高温度分在频率表上占位）
    elig_arr = np.array(eligible, dtype=bool)
    macro_elig = np.where(elig_arr[None, :], macro, np.nan)
    finite = np.isfinite(macro_elig)
    best = np.where(finite, macro_elig, -np.inf).argmax(axis=1)
    best = np.where(finite.any(axis=1), best, -1)
    counted = int((best >= 0).sum())
    for mi, m in enumerate(models):
        pct = round(100.0 * float((best == mi).sum()) / counted, 1) if counted else 0.0
        out[m]["champion_pct"] = pct
    # 与点估计冠军的显著性：得分差的 90% 区间是否含 0
    if point_top is not None:
        top_i = point_top[0]
        for mi, m in enumerate(models):
            if mi == top_i:
                out[m]["sig_vs_top"] = True
                continue
            if not eligible[mi]:
                continue
            with np.errstate(invalid="ignore"):
                diff = macro[:, top_i] - macro[:, mi]
                d = diff[np.isfinite(diff)]
            if d.size >= max(5, runs // 10):
                lo, hi = np.percentile(d, [5, 95])
                out[m]["sig_vs_top"] = bool(lo > 0 or hi < 0)
    return out


def weight_champion_distribution(
    temp_sub: np.ndarray, precip_sub: np.ndarray, temp_parts, precip_parts,
    runs: int = 500, seed: int = 20260907,
) -> list[dict]:
    """权重敏感性（P0-2.2）：把 13 项权重各扰动 ±40%，统计冠军分布。

    temp_sub / precip_sub：(m, b, k) 的**已换算并截断**的子分张量（NaN=缺项），
    k 顺序与 parts 表一致。权重 w ~ U(0.6, 1.4)×原权重，逐 run 重组
    温度分/降水分 → 桶综合分 → macro 平均 → 冠军。返回
    [{"model": m, "pct": 频率%}, ...]（降序，含 0 频率外的全部模型）。
    """
    n_m, n_b = temp_sub.shape[0], temp_sub.shape[1]
    w_t0 = np.array([p[1] for p in temp_parts])
    w_p0 = np.array([p[1] for p in precip_parts])
    rng = np.random.default_rng(seed)
    wt = w_t0[None, :] * rng.uniform(0.6, 1.4, size=(runs, len(w_t0)))
    wp = w_p0[None, :] * rng.uniform(0.6, 1.4, size=(runs, len(w_p0)))

    mt = ~np.isnan(temp_sub)     # (m, b, k)
    mp = ~np.isnan(precip_sub)
    st = np.where(mt, temp_sub, 0.0)
    sp = np.where(mp, precip_sub, 0.0)

    # num[run, m, b] = Σ_k w[run, k]·sub[m, b, k]；den[run, m, b] = Σ_k w·mask
    num_t = np.einsum("rk,mbk->rmb", wt, st)
    den_t = np.einsum("rk,mbk->rmb", wt, mt.astype(float))
    num_p = np.einsum("rk,mbk->rmb", wp, sp)
    den_p = np.einsum("rk,mbk->rmb", wp, mp.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        t_score = num_t / np.where(den_t > 0, den_t, np.nan)
        p_score = num_p / np.where(den_p > 0, den_p, np.nan)
    both = np.stack([t_score, p_score])
    with warnings.catch_warnings():
        # 某桶在该 run 全模型无分是合法状态（单天桶未被抽到等）
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore"):
            bucket = np.nanmean(both, axis=0)          # (run, m, b)
            macro = np.nanmean(bucket, axis=-1)        # (run, m)
    # 某 run 全模型无分时该 run 不计冠军
    finite = np.isfinite(macro)
    best = np.where(finite, macro, -np.inf).argmax(axis=1)
    best = np.where(finite.any(axis=1), best, -1)
    counts = np.bincount(best[best >= 0], minlength=n_m)
    total = counts.sum()
    # 返回按频率降序的 (模型下标, 频率%) 列表，模型名由调用方映射
    order = np.argsort(-counts)
    return [{"index": int(i), "pct": round(100.0 * float(counts[i]) / total, 1)}
            for i in order if counts[i] > 0]

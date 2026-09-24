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
# 跨站相关 ρ̄ 的最小公共时刻数：少于此值时两两相关的估计噪声大于信号，
# 退回"各站独立"的旧口径（不校正，宁可保守也不过校正）
CROSS_STATION_MIN_OVERLAP = 30

# 天桶难度的双向加法模型（见 two_way_adjust）：一道最少几家同台、一家最少几道
# 才算"能够横向比较"。低于此阈值的单元格提供不了比较信息，会被剔除出劈分设计。
MIN_MODELS_PER_BUCKET = 3
MIN_BUCKETS_PER_MODEL = 2


# ------------------------------------------------------------------ 有效样本量
def pearson_r(a: np.ndarray, b: np.ndarray) -> float | None:
    """皮尔逊相关系数（数值安全版）。

    绝不手写 `np.cov(a,b)/sqrt(a.var()*b.var())`：`np.cov` 用 ddof=1、`ndarray.var()`
    默认 ddof=0，两者混用会把 ρ 系统性放大 n/(n−1) 倍——对完全相关的序列会算出
    1.0208 这种数学上不可能的相关系数（对抗式审查 P0-5），并在小样本上频繁触发
    RHO_CLAMP 截断。`np.corrcoef` 的分子分母同用 ddof=1，是本项目的唯一合法路径。
    退化（长度 < 2、任一侧零方差、非有限）返回 None。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size or a.size < 2:
        return None
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        m = np.isfinite(a) & np.isfinite(b)
        a, b = a[m], b[m]
        if a.size < 2:
            return None
    if a.var() <= 0 or b.var() <= 0:
        return None
    r = float(np.corrcoef(a, b)[0, 1])
    return r if np.isfinite(r) else None


def effective_n(err: np.ndarray) -> int:
    """自相关校正后的有效样本量：n_eff = n·(1−ρ₁)/(1+ρ₁)。

    err 是按时间排序的误差（或事件指示）序列。n < NEFF_MIN_SERIES 时不估自相关
    （估计量本身太噪，直接返回 n 不惩罚小样本）；ρ₁ 估计为 None（方差为 0 等
    退化情形）同样返回 n。结果截断到 [1, n]。
    """
    e = np.asarray(err, dtype=float)
    e = e[np.isfinite(e)]
    n = int(e.size)
    if n < NEFF_MIN_SERIES + 1 or n <= 2:
        return n
    rho = pearson_r(e[:-1], e[1:])
    if rho is None:
        return n
    rho = max(-RHO_CLAMP, min(RHO_CLAMP, rho))
    n_eff = int(round(n * (1 - rho) / (1 + rho)))
    return max(1, min(n, n_eff))


def cross_station_rho(series_by_station: dict[str, list[float]],
                      times_by_station: dict[str, list[str]] | None = None,
                      min_overlap: int = CROSS_STATION_MIN_OVERLAP) -> float | None:
    """站间误差序列的平均两两相关 ρ̄（跨站冗余度）。

    为什么需要它（第一性原理）：`effective_n` 只校正了**时间**维度的自相关，它把
    "各站相互独立"当成了默认事实。但相距几十到几百公里的站点，误差由同一批天气
    系统驱动——实测同一源温度误差的跨站 ρ̄ ≈ 0.23。k 个各含 n_eff 信息的站，合起来
    的信息量只有独立情形的 1/(1+(k−1)ρ̄)（k=4, ρ̄=0.23 → 0.59），直接把各站 n_eff
    相加会**高估约 1.7 倍**（对抗式审查 P0-4）。

    对齐方式：给了 times_by_station 就按**公共时刻**对齐（正确做法，缺测小时自然跳过）；
    否则退回"按位置对齐、截断到最短序列"（调用方未提供时间键时的保守近似，此时若各站
    缺测模式不同，相关会被低估——只会让校正偏保守，不会过校正）。

    返回 None 表示样本不足以估计（< 2 站、公共长度 < min_overlap）。
    """
    ids = list(series_by_station)
    if len(ids) < 2:
        return None
    if times_by_station is not None:
        maps: list[dict[str, float]] = []
        for sid in ids:
            vals = series_by_station[sid]
            times = times_by_station.get(sid)
            if times is None or len(times) != len(vals):
                return None
            maps.append(dict(zip(times, vals)))
        rs: list[float] = []
        for i in range(len(maps)):
            for j in range(i + 1, len(maps)):
                common = maps[i].keys() & maps[j].keys()
                if len(common) < min_overlap:
                    continue
                keys = sorted(common)
                a = np.array([maps[i][k] for k in keys], dtype=float)
                b = np.array([maps[j][k] for k in keys], dtype=float)
                r = pearson_r(a, b)
                if r is not None:
                    rs.append(r)
        if not rs:
            return None
        return float(np.mean(rs))

    lens = [len(series_by_station[sid]) for sid in ids]
    cut = min(lens)
    if cut < min_overlap:
        return None
    rs = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a = np.asarray(series_by_station[ids[i]][:cut], dtype=float)
            b = np.asarray(series_by_station[ids[j]][:cut], dtype=float)
            r = pearson_r(a, b)
            if r is not None:
                rs.append(r)
    if not rs:
        return None
    return float(np.mean(rs))


def n_eff_from_station_series(series_by_station: dict[str, list[float]],
                              times_by_station: dict[str, list[str]] | None = None
                              ) -> int:
    """跨站合并的有效样本量：各站 n_eff 之和 ÷ 跨站冗余因子 1+(k−1)ρ̄。

    站的独立估计仍然逐站做（跨站拼接会造出人为的序列跳变，低估 ρ₁）；但**求和**这
    一步必须扣掉站间相关——否则 4 个相互 ρ̄≈0.23 相关的站会被当成 4 份独立信息，
    n_eff 高估约 1.7 倍，总榜入围门槛 min_board_neff 随之形同虚设（P0-4）。
    ρ̄ 估不出来（站数 < 2、公共样本不足）时不校正，退回旧的"独立求和"。
    """
    total = 0
    for vals in series_by_station.values():
        total += effective_n(np.asarray(vals, dtype=float))
    k = sum(1 for vals in series_by_station.values() if len(vals) > 0)
    if k < 2 or total <= 0:
        return total
    rho_bar = cross_station_rho(series_by_station, times_by_station)
    if rho_bar is None or rho_bar <= 0:
        return total
    rho_bar = min(rho_bar, RHO_CLAMP)
    corrected = total / (1.0 + (k - 1) * rho_bar)
    return max(1, int(round(corrected)))


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


# ------------------------------------------------------------------ 重采样块长
# 重采样块数下限：块数太少时重采样组合退化（n_blk 个块有放回抽 n_blk 个，
# 组合数 = C(2·n_blk−1, n_blk)，n_blk=2 时只有 3 种），区间本身不可信
MIN_BOOTSTRAP_BLOCKS = 6
# 块长上限：天气尺度过程相关时间典型 3~5 天，再长只会把有效样本切得更碎
MAX_BOOTSTRAP_BLOCK_DAYS = 7


def resolve_block_days(n_days: int, block_days: int | None = None,
                       rho: float | None = None) -> int:
    """确定重采样块长（天）。

    block_days 显式给定（≥1）时以它为准，只做"块数下限"的兜底收缩；
    缺省（None/0）时按日尺度误差的自相关估计：AR(1) 的积分时间尺度
    τ = (1+ρ)/(1−ρ)（ρ=0.5 → 3 天，与天气尺度吻合），再截断到
    [1, MAX_BOOTSTRAP_BLOCK_DAYS]。ρ 缺省（None）时取 1。

    最后统一收缩到"块数 ≥ MIN_BOOTSTRAP_BLOCKS"：块长偏短只是让区间偏窄
    （偏差方向已知、可披露），块数不足会让区间本身变成噪声——两害相权取前者。
    """
    if n_days <= 0:
        return 1
    if block_days is None or block_days <= 0:
        if rho is None or not np.isfinite(rho):
            block_days = 1
        else:
            rho = max(-RHO_CLAMP, min(RHO_CLAMP, float(rho)))
            block_days = int(round((1 + rho) / (1 - rho)))
        block_days = max(1, min(MAX_BOOTSTRAP_BLOCK_DAYS, block_days))
    while block_days > 1 and (n_days // block_days) < MIN_BOOTSTRAP_BLOCKS:
        block_days -= 1
    return block_days


def daily_error_lag1_rho(hourly: list[dict]) -> float | None:
    """日尺度温度误差的 lag-1 自相关（各站独立估计后 Fisher-z 合并）。

    用于规划 bootstrap 块长：块长应当覆盖误差的去相关时间。只取"每站每日的
    平均误差"——同一天内多条起报/多个时效是同一天的重复读数，先并成一天一个
    数，否则日内的正相关会把日间相关人为抬高。
    """
    by: dict[str, dict[str, list[float]]] = {}
    for r in hourly:
        o, f = r.get("temp_obs"), r.get("temp_fcst")
        if o is None or f is None:
            continue
        by.setdefault(r["station"], {}).setdefault(r["valid_iso"][:10], []).append(f - o)
    rs: list[float] = []
    ns: list[int] = []
    for _sid, days in by.items():
        items = sorted(days.items())
        if len(items) < 4:
            continue
        series = np.array([float(np.mean(v)) for _d, v in items])
        rho = pearson_r(series[:-1], series[1:])
        if rho is None:
            continue
        rs.append(max(-0.999, min(0.999, rho)))
        ns.append(len(series) - 1)
    if not rs:
        return None
    return fisher_z_combine(rs, ns)


def day_block_weights(runs: int, n_days: int, block_days: int = 1,
                      seed: int = 20260906) -> np.ndarray:
    """生成 (runs, n_days) 的天重采样权重矩阵（每天被抽中的次数）。

    块长 L：把连续 L 天绑成一块整体重采样（尾部不足 L 天并入最后一块，绝不丢
    样本），每次重复抽 ceil(n_days / L) 个块——总权重仍 ≈ n_days，与点估计同
    量级（W≡1 的退化情形严格等于"每天出现一次"）。同一次重采样的权重同时作用
    于所有模型与两条轨道 ⇒ 模型间比较是**配对**的。
    """
    n_d = max(1, int(n_days))
    L = max(1, min(int(block_days or 1), n_d))
    starts = list(range(0, n_d, L))
    n_blk = max(1, len(starts))
    rng = np.random.default_rng(seed)
    W = np.zeros((max(1, int(runs)), n_d))
    for i in range(W.shape[0]):
        picked = rng.integers(0, n_blk, n_blk)
        for bi in picked:
            lo = starts[bi]
            np.add.at(W[i], np.arange(lo, min(lo + L, n_d)), 1.0)
    return W


# ------------------------------------------------------------------ 按天分块 bootstrap
# 温度充分统计量的列含义（逐样本可加，故"天"内可预聚合、重采样时按权重求和）
_TEMP_STATS = ("n", "se2", "ae", "se", "h1", "h2", "sx", "sy", "sxy", "sxx", "syy")
# 降水二分类列联计数的列含义
_RAIN_STATS = ("h", "fa", "mi", "c")

# ------------------------------------------------- 按时间分辨率拆分的证据表（2026-09 重构）
# 重构前只有两张表：温度走逐小时、降水走日累计，两者 nanmean 成一个"综合分"。
# 于是"某时刻报得准不准"与"这天的最高/最低/总雨量报得准不准"混在同一个数字里——
# 一个日内相位偏、但日极值很准的源，和一个恰好相反的源，可能拿到同一个分数。
# 现在按分辨率拆成两张互不相通的证据表：
#   hourly —— 逐小时温度 + 逐小时晴雨（该小时够不够得上"在下雨"）
#   daily  —— 日最高/最低温度 + 日累计降水（这一天的总量与极值）
# 两条轨道各自闭环成一个"桶综合分"，再由总榜跨分辨率联合。
TABLE_KEYS = ("temp_hourly", "rain_hourly",
              "temp_daily_max", "temp_daily_min", "rain_daily")


def eval_days(hourly: list[dict], daily: list[dict]) -> list[str]:
    """评估窗口内的自然日全集（逐小时有效时刻所在日 ∪ 按天有效日）。

    单独导出是为了让调用方在**不建稠密表**的前提下知道块长规划所需的天数。
    """
    return sorted({r["valid_iso"][:10] for r in hourly}
                  | {r["valid_day"] for r in daily})


def _temp_stat_row(o: float, f: float) -> list[float]:
    """单个温度样本 → 11 项可加统计量（小时量值与日最高/最低共用同一套）。"""
    e = f - o
    return [1.0, e * e, abs(e), e,
            1.0 if abs(e) <= 1 else 0.0,
            1.0 if abs(e) <= 2 else 0.0,
            f, o, f * o, f * f, o * o]


def _rain_stat_row(o: float, f: float, thr: float) -> list[float]:
    """单个降水样本 → 4 项二分类列联计数（h / fa / mi / c）。"""
    ob, fb = o >= thr, f >= thr
    return [1.0 if (ob and fb) else 0.0, 1.0 if (not ob and fb) else 0.0,
            1.0 if (ob and not fb) else 0.0, 1.0 if (not ob and not fb) else 0.0]


def build_day_stat_tables(
    hourly: list[dict], daily: list[dict], models: list[str],
    n_buckets_hourly: int, n_buckets_daily: int,
    rain_thr_daily: float, rain_thr_hourly: float,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """把逐小时/按天记录压成（模型 × 桶 × 站 × 天）的可加充分统计量稠密表。

    返回 (days, tables)，tables 的形状：
      temp_hourly    (m, b, s, d, 11)  逐小时温度（小时榜 · 温度维）
      rain_hourly    (m, b, s, d, 4)   逐小时晴雨，阈值 rain_thr_hourly（小时榜 · 降水维）
      temp_daily_max (m, b, s, d, 11)  日最高温（日榜 · 温度维之一）
      temp_daily_min (m, b, s, d, 11)  日最低温（日榜 · 温度维之一）
      rain_daily     (m, b, s, d, 4)   日累计晴雨，阈值 rain_thr_daily（日榜 · 降水维）

    两条井水不犯河水的证据链是这次重构的核心：同一批存档，按"时刻"答一次
    （小时榜）、按"自然日"答一次（日榜），谁也不替谁说话。天数取两类记录的
    并集；无数据的天/桶保持全 0，重采样时自然按"无样本"处理。

    桶对齐：两条轨道的天桶都是"起报日之后的第 N 个自然日"（小时 extends 用
    bucket、日离用 offset），与排行榜的天桶语义一致——两榜的横轴是同一条刻度，
    总榜才能把它们当成同一批"难度"来劈。
    """
    days = eval_days(hourly, daily)
    day_idx = {d: i for i, d in enumerate(days)}
    model_idx = {m: i for i, m in enumerate(models)}
    station_idx: dict[str, int] = {}
    for r in hourly:
        station_idx.setdefault(r["station"], len(station_idx))
    for r in daily:
        station_idx.setdefault(r["station"], len(station_idx))
    n_s = max(1, len(station_idx))
    n_m = max(1, len(models))
    # 两条轨道的桶数**各用各的**（hourly_lead_days 与 daily_max_offset_days 语义
    # 不同，配置允许不等——此前共用一个 n_buckets=max(H,D)，H≠D 时列数与劈分
    # 设计的 H+D 宽度对不上，bootstrap 直接崩）。小时表 H 列、日表 D 列，
    # track_bucket_scores 拼接后正好是 H+D 列。
    shape_t_h = (n_m, n_buckets_hourly, n_s, len(days), len(_TEMP_STATS))
    shape_r_h = (n_m, n_buckets_hourly, n_s, len(days), len(_RAIN_STATS))
    shape_t_d = (n_m, n_buckets_daily, n_s, len(days), len(_TEMP_STATS))
    shape_r_d = (n_m, n_buckets_daily, n_s, len(days), len(_RAIN_STATS))
    tables = {
        "temp_hourly": np.zeros(shape_t_h, dtype=np.float64),
        "rain_hourly": np.zeros(shape_r_h, dtype=np.float64),
        "temp_daily_max": np.zeros(shape_t_d, dtype=np.float64),
        "temp_daily_min": np.zeros(shape_t_d, dtype=np.float64),
        "rain_daily": np.zeros(shape_r_d, dtype=np.float64),
    }
    # 每条记录 → (目标表名, 索引, 统计量行)：先收集再一次性 scatter-add
    jobs = {k: ([], []) for k in tables}
    for r in hourly:
        mi = model_idx.get(r["model"])
        if mi is None or not (1 <= r["bucket"] <= n_buckets_hourly):
            continue
        key = (mi, r["bucket"] - 1, station_idx[r["station"]],
               day_idx[r["valid_iso"][:10]])
        o, f = r["temp_obs"], r["temp_fcst"]
        if o is not None and f is not None:
            rows, idx = jobs["temp_hourly"]
            rows.append(_temp_stat_row(o, f))
            idx.append(key)
        o, f = r["rain_obs"], r["rain_fcst"]
        if o is not None and f is not None:
            rows, idx = jobs["rain_hourly"]
            rows.append(_rain_stat_row(o, f, rain_thr_hourly))
            idx.append(key)
    for r in daily:
        mi = model_idx.get(r["model"])
        if mi is None or not (1 <= r["offset"] <= n_buckets_daily):
            continue
        key = (mi, r["offset"] - 1, station_idx[r["station"]],
               day_idx[r["valid_day"]])
        for tk, ka, kb in (("temp_daily_max", "temp_max_obs", "temp_max_fcst"),
                           ("temp_daily_min", "temp_min_obs", "temp_min_fcst")):
            o, f = r.get(ka), r.get(kb)
            if o is not None and f is not None:
                rows, idx = jobs[tk]
                rows.append(_temp_stat_row(o, f))
                idx.append(key)
        o, f = r["rain_obs"], r["rain_fcst"]
        if o is not None and f is not None:
            rows, idx = jobs["rain_daily"]
            rows.append(_rain_stat_row(o, f, rain_thr_daily))
            idx.append(key)
    for name, (rows, idx) in jobs.items():
        if not rows:
            continue
        rows_a = np.asarray(rows)
        idx_a = np.asarray(idx)
        stat_len = rows_a.shape[1]
        for k in range(stat_len):
            np.add.at(tables[name][:, :, :, :, k], tuple(idx_a.T), rows_a[:, k])
    return days, tables


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
            # 平方和在大样本下会丢精度，dxT·dyT 可能算出微小负数（数学上非负）。
            # 直接开方会刷 RuntimeWarning 并产出 NaN——先把负数夹到 0，再由
            # 外层 dxT>0 & dyT>0 的判定决定取用（本就只取正定情形）。
            ok = (dxT > 0) & (dyT > 0)
            pooled = np.where(ok, covT / np.where(ok, np.sqrt(np.maximum(dxT * dyT, 0.0)), 1.0),
                              np.nan)
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
    hourly: list[dict], daily: list[dict], models: list[str],
    n_buckets_hourly: int, n_buckets_daily: int,
    rain_thr_daily: float, rain_thr_hourly: float, temp_parts, precip_parts,
    min_sample: int,
    runs: int = 500, seed: int = 20260906,
    eligible: list[bool] | None = None,
    block_days: int = 1,
    alpha: float = 0.10,
    top_model: str | None = None,
    boards: dict[str, dict] | None = None,
    temp_point_valid: np.ndarray | None = None,
    rain_point_valid: np.ndarray | None = None,
    bucket_valid: np.ndarray | None = None,
    adj_row: np.ndarray | None = None,
    adj_col: np.ndarray | None = None,
    adj_w: np.ndarray | None = None,
    adj_ridge: float = 0.0,
) -> dict[str, dict[str, Any]]:
    """按天分块 bootstrap：各榜单那个"难度对齐综合分"的不确定性。

    返回 {榜单名: {model: {"ci90": [lo, hi] | None, "champion_pct": float,
                  "sig_vs_top": bool | None}}}；未传 boards 时只有一个键 "all"。

    所有榜单共享**同一批重采样**：三张榜（总榜 / 小时榜 / 日榜）若各抽各的，
    同一季天气在这张榜上被抽重、在那张榜上没被抽重的情形会同时发生，于是
    "总榜 A 略胜 B"与"小时榜 A 反输 B"这类跨榜差异里混进纯抽样噪声。
    共享重采样后两家在任意一处的先后都源自同一份天气局面，比较是可配对的。

    boards: 榜名 → 该榜的设计参数（未给的子项回退到外层同名全局参数）：
      columns        该榜取合成表的哪些列（小时榜取前 b 列、日榜取后 b 列、
                     总榜取全部 2b 列；None = 全部）
      adj_row/adj_col 该榜自己的双向劈分设计（m,）、（该榜列数,）布尔
      adj_w          该榜的格子权重（m, 该榜列数）
      eligible       该榜入围冠军竞争的模型
      top_model      该榜点估计冠军（显著性/冠军频率的参照必须是戴冠那个源）
      *_point_valid  该榜列布局下的点估计缺项掩码

    每张榜的置信区间都用它**自己那张设计**归总（P0-2：给 A 数字配 B 数字的
    区间是直接误导），于是 bootstrap 与点估计回答的是同一个估计量。
    """
    if eligible is None:
        eligible = [True] * len(models)
    if not hourly and not daily:
        empty = {m: {"ci90": None, "champion_pct": 0.0, "sig_vs_top": None}
                 for m in models}
        if not boards:
            return {"all": empty}
        return {name: {m: dict(v) for m, v in empty.items()} for name in boards}
    days, tables = build_day_stat_tables(hourly, daily, models,
                                         n_buckets_hourly, n_buckets_daily,
                                         rain_thr_daily, rain_thr_hourly)
    W = day_block_weights(runs, len(days), block_days, seed=seed)
    if not boards:
        macro = macro_scores_from_weights(
            W, tables, temp_parts, precip_parts, min_sample,
            temp_point_valid=temp_point_valid, rain_point_valid=rain_point_valid,
            bucket_valid=bucket_valid, adj_row=adj_row, adj_col=adj_col,
            adj_w=adj_w, adj_ridge=adj_ridge)
        return {"all": _summarize_bootstrap(macro, models, eligible, alpha=alpha,
                                            top_model=top_model)}
    # 每张榜用自己的设计把**同一批重采样**归总成行分；表只建一次、只聚合一次
    agg = {k: aggregate_day_stats(W, v) for k, v in tables.items()}
    out: dict[str, dict[str, Any]] = {}
    for name, spec in boards.items():
        cols = spec.get("columns")
        b_row = spec.get("adj_row", adj_row)
        b_col = spec.get("adj_col", adj_col)
        # adj_w 已是**该榜自己那份**（形状与该榜列数一致），不要再按 cols 切一次：
        # 日榜的 cols 是 16..31，而它自己的权重矩阵本来就只有 16 列
        b_w = spec.get("adj_w", adj_w)
        # 缺项掩码：温度/降水的"这格有没有结论"是数据的属性，三榜共享；
        # bucket_valid 则是"这格进没进点估计的设计"，各榜用自己的 cell_valid
        # （见 evaluate._resolution_boards）——两者必须逐格对齐，CI 中心才不漂。
        buckets = track_bucket_scores(
            agg, temp_parts, precip_parts, min_sample,
            temp_point_valid=temp_point_valid, rain_point_valid=rain_point_valid,
            bucket_valid=spec.get("bucket_valid", bucket_valid))
        if cols is not None:
            buckets = buckets[:, :, cols]
        if b_row is None or b_col is None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                macro = np.nanmean(buckets, axis=-1)
        else:
            macro = difficulty_adjusted(buckets, np.asarray(b_row, dtype=bool),
                                        np.asarray(b_col, dtype=bool),
                                        weights=b_w, ridge=adj_ridge)
        out[name] = _summarize_bootstrap(
            macro, models, spec.get("eligible", eligible), alpha=alpha,
            top_model=spec.get("top_model", top_model))
    return out


# ------------------------------------------------- 天桶难度的双向加法劈分（P0-4）
def design_mask(V: np.ndarray,
                min_col: int = MIN_MODELS_PER_BUCKET,
                min_row: int = MIN_BUCKETS_PER_MODEL,
                rounds: int = 6,
                min_col_frac: float = 0.0,
                segment_sizes: tuple[int, ...] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """按"行/列最少有效数"互剪观测掩膜 (m, b)，直到不再变化。

    只有横向可比的格子能留在设计里：一个天桶若只有 1 家覆盖，它的"桶难度"就与
    那一家的技巧完全混叠（两个效应分不开），留着等于给那家发一张白卷；同理一家
    只在 1 个桶有分，也谈不上"跨时效的能力"。互剪是迭代的——剔除行会把某些列
    降到阈值以下，反之亦然。

    min_col_frac：**相对门槛**——列的家数还必须 ≥ ceil(min_col_frac × 最大桶家数)。
    绝对门槛 `min_col` 挡不住"长尾桶"：第 16 桶只有 7 家时，绝对门槛 3 会让它进来，
    而它那 7 个格子的样本薄到"桶难度"本身就是噪声，却要参与**所有**源的行分计算
    （对抗式审查 P1-4）。0.5 表示"家数不足最热闹那一半的桶不进主设计"。
    0（默认）关闭相对门槛，保持旧行为（既有测试与对照口径依赖它）。

    segment_sizes：**赛段门槛**（为总榜而设）。把列按顺序切成若干段
    （如 [b, b] = 小时榜段 + 日榜段），行必须在**每个仍有列留存的赛段**里至少
    有一个格子。这条约束是"综合"二字的最低要求：一家只在小时分辨率上被验证过、
    日分辨率一个桶都没有，它的"综合分"其实就是它的小时分；让它与两条轨道都被
    验证过的源同榜竞争，等于让"没被考的科目自动满分"。注意与 min_row 不同：
    min_row 管"总共要有几个桶"，赛段门槛管"每个分辨率都要有"。

    返回 (row_keep (m,), col_keep (b,)) 布尔数组；全空时两个都是全 False。
    """
    V = np.asarray(V, dtype=bool)
    if V.ndim != 2:
        raise ValueError("design_mask 需要二维 (m, b) 掩膜")
    eff_min_col = int(min_col)
    if min_col_frac and min_col_frac > 0:
        per_col = V.sum(axis=0)
        if per_col.size:
            eff_min_col = max(eff_min_col,
                              int(np.ceil(min_col_frac * float(per_col.max()))))
    # 赛段的列下标切片：整段被剔除的赛段（该分辨率本榜无数据）不作要求
    segs: list[np.ndarray] = []
    if segment_sizes:
        off = 0
        for size in segment_sizes:
            segs.append(np.arange(off, off + int(size)))
            off += int(size)
    row = np.ones(V.shape[0], dtype=bool)
    col = np.ones(V.shape[1], dtype=bool)
    for _ in range(rounds):
        cnt_col = (V & row[:, None]).sum(axis=0)
        new_col = cnt_col >= eff_min_col
        cnt_row = (V & new_col[None, :]).sum(axis=1)
        new_row = cnt_row >= min_row
        for sidxs in segs:
            keep = new_col[sidxs]
            if not keep.any():
                continue        # 该赛段整段被剔除（本榜无此分辨率的数据）→ 不要求
            new_row = new_row & (V[:, sidxs][:, keep].sum(axis=1) >= 1)
        if np.array_equal(new_col, col) and np.array_equal(new_row, row):
            return new_row, new_col
        row, col = new_row, new_col
    return row, col


def largest_component_mask(row_keep: np.ndarray, col_keep: np.ndarray,
                           V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """把劈分设计收缩到最大的连通分量（行列构成的二分图）。

    二分图不连通时，各分量各有自己的加法常数——分量之间的 α 差异不可识别，
    把它们并进一张榜等于宣称了无从得知的结论。与其静默出错，这里保留最大
    分量（其余的源记"样本积累中"），并把举动交给调用方记录在案。
    """
    V = np.asarray(V, dtype=bool) & row_keep[:, None] & col_keep[None, :]
    n_row, n_col = V.shape
    if n_row == 0 or n_col == 0:
        return np.zeros(n_row, dtype=bool), np.zeros(n_col, dtype=bool)
    n = n_row + n_col
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i in range(n_row):
        for j in np.flatnonzero(V[i]):
            union(i, n_row + int(j))
    labels = np.array([find(i) for i in range(n)])
    counts: dict[int, int] = {}
    for lab in labels:
        counts[int(lab)] = counts.get(int(lab), 0) + 1
    # 以涉及的（行+列）节点数最多的分量为最大分量
    best = max(counts, key=lambda k: counts[k])
    return labels[:n_row] == best, labels[n_row:] == best


def two_way_fit(S: np.ndarray, valid: np.ndarray,
                weights: np.ndarray | None = None,
                ridge: float = 0.0,
                max_iter: int = 300, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """对 (r, m, b) 的分数张量做行/列效应的加权交替最小二乘（WLS/ALS）。

    模型 S[m, b] = α_m + β_b + 噪声；valid 为观测掩膜。Gauss–Seidel 迭代到收敛；
    每行/每列的估计都只用自己的观测格（缺一格不影响相邻行列）。

    weights（P0-1，本模块最重要的正确性开关）：每个格子的**信息量权重**，缺省
    None 表示等权（旧行为，W≡1）。等权拟合把"降水样本 5 条"与"降水样本 92 条"当成
    同样可信的观测——本项目的格子样本量跨三个数量级（温度 788~2,360、降水 5~92），
    当权重与样本量无关时，"桶难度"会被最薄的那几个格子带偏，而这些列效应又会等量
    地打进**每一行**的名次。按样本量加权是这个问题的最小修正：让信息多的格子说话。

    ridge（P1-4）：列效应的经验贝叶斯式收缩——分母加上 λ 后，家数少的桶的列效应
    被拉向 0（"这一档难度未知，先按平均难度算"），而不是靠 7 个薄格子给出一个
    会污染全榜的极端值。0（默认）= 不收缩。

    等权且不收缩时，本函数与旧实现逐位相同（既有回归测试锁定）。
    """
    V = np.asarray(valid, dtype=bool)
    X = np.where(V, np.asarray(S, dtype=float), 0.0)
    if weights is None:
        W = V.astype(np.float64)
    else:
        W = np.where(V, np.asarray(weights, dtype=float), 0.0)
        W = np.where(np.isfinite(W) & (W > 0), W, 0.0)
    cnt_b = W.sum(axis=2)            # (r, m) 每行权重和
    cnt_m = W.sum(axis=1)            # (r, b) 每列权重和
    lam = max(0.0, float(ridge))
    alpha = np.zeros(X.shape[:2])
    beta = np.zeros((X.shape[0], X.shape[2]))
    for _ in range(max_iter):
        # 行效应：以权重求和后按权重和（+λ）归一
        num_a = np.where(V, W * (X - beta[:, None, :]), 0.0).sum(axis=2)
        den_a = cnt_b + lam
        new_alpha = np.where(den_a > 0, num_a / np.where(den_a > 0, den_a, 1.0), 0.0)
        num_b = np.where(V, W * (X - new_alpha[:, :, None]), 0.0).sum(axis=1)
        den_b = cnt_m + lam
        new_beta = np.where(den_b > 0, num_b / np.where(den_b > 0, den_b, 1.0), 0.0)
        delta = max(float(np.max(np.abs(new_alpha - alpha)) if new_alpha.size else 0.0),
                    float(np.max(np.abs(new_beta - beta)) if new_beta.size else 0.0))
        alpha, beta = new_alpha, new_beta
        if np.isfinite(delta) and delta < tol:
            break
    return alpha, beta


def _fit_parts(S3: np.ndarray, row_keep: np.ndarray, col_keep: np.ndarray,
               max_iter: int, tol: float,
               weights: np.ndarray | None = None,
               ridge: float = 0.0,
               valid: np.ndarray | None = None):
    """(r, m, b) 劈分的核心：返回 (mu (r,), alpha (r,m), beta (r,b), V)；无可用格子时 None。

    weights：与 S3 同形状的 (r, m, b) 权重（见 two_way_fit）。
    valid：**显式的格子掩膜**，给定即以此为准（不再由"该矩阵自身是否有限"决定）。
    这是 P1-3 的落点：排行榜的派生列（acc2 / RMSE / TS / ETS）必须与综合分**逐格
    同集**——否则某源某桶因降水缺测而没有综合分、却因为 acc2 有值而继续参与"难度
    对齐 ±2°C"的估计，页面上的两列就来自两批不同的格子，"同一张设计"只是半真话。
    """
    S3 = np.asarray(S3, dtype=float)
    if valid is not None:
        V = np.asarray(valid, dtype=bool) & row_keep[None, :, None] & col_keep[None, None, :]
    else:
        V = np.isfinite(S3) & row_keep[None, :, None] & col_keep[None, None, :]
    if not V.any():
        return None
    W = None
    if weights is not None:
        W = np.asarray(weights, dtype=float)
    alpha, beta = two_way_fit(S3, V, weights=W, ridge=ridge, max_iter=max_iter, tol=tol)
    # 归一化：让保留列的平均难度为 0，此时 α 就是"在平均难度上这家值多少分"
    n_col = max(int(col_keep.sum()), 1)
    shift = beta[:, col_keep].sum(axis=1) / n_col
    alpha = alpha + shift[:, None]
    beta = beta - shift[:, None]
    resid = np.where(V, S3 - alpha[:, :, None] - beta[:, None, :], 0.0)
    cnt = V.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = np.where(cnt > 0, resid.sum(axis=(1, 2)) / np.where(cnt > 0, cnt, 1.0), np.nan)
    return mu, alpha, beta, V


def difficulty_adjusted(S3: np.ndarray, row_keep: np.ndarray, col_keep: np.ndarray,
                        max_iter: int = 300, tol: float = 1e-8,
                        weights: np.ndarray | None = None,
                        ridge: float = 0.0,
                        valid: np.ndarray | None = None) -> np.ndarray:
    """把 (r, m, b) 的桶劈分张量 → (r, m) 的难度对齐行分（NaN = 无法比较）。

    落在 row_keep/col_keep 之外的格子不参与劈分。每一趟重采样都用**同一张设计**
    与**同一组权重**，这样点估计与 bootstrap 回答的是同一个估计量，置信区间的中心
    才不会漂。weights / valid 的语义见 _fit_parts。
    """
    S3 = np.asarray(S3, dtype=float)
    if S3.ndim != 3:
        raise ValueError("difficulty_adjusted 需要 (r, m, b) 三维数组")
    row_keep = np.asarray(row_keep, dtype=bool)
    col_keep = np.asarray(col_keep, dtype=bool)
    if row_keep.shape[0] != S3.shape[1] or col_keep.shape[0] != S3.shape[2]:
        raise ValueError("row_keep / col_keep 的维度与分数张量不匹配")
    parts = _fit_parts(S3, row_keep, col_keep, max_iter, tol,
                       weights=weights, ridge=ridge, valid=valid)
    if parts is None:
        return np.full(np.asarray(S3).shape[:2], np.nan)
    mu, alpha, _beta, _V = parts
    return np.where(row_keep[None, :], mu[:, None] + alpha, np.nan)


def two_way_adjust(S2: np.ndarray,
                   min_col: int = MIN_MODELS_PER_BUCKET,
                   min_row: int = MIN_BUCKETS_PER_MODEL,
                   max_iter: int = 300, tol: float = 1e-8,
                   weights: np.ndarray | None = None,
                   min_col_frac: float = 0.0,
                   ridge: float = 0.0,
                   min_cell_weight: float = 0.0,
                   segment_sizes: tuple[int, ...] | None = None) -> dict:
    """把 (m, b) 的分数矩阵劈成"行的技巧"与"列的难度"，返回难度对齐后的行分。

    为什么要这一步（第一性原理）：预报难度随时效单调上升，而各家能预报的天数
    长短不一——直接把自己覆盖到的天桶平均起来，短覆盖的源天然吃到简单的桶，
    "覆盖越短排名越高"；只看共同覆盖窗口（取交集）又是把大多数数据扔掉（实测
    26 源同榜时窗口只剩 4 天）。双向加法模型是这个问题的正解：把每个格子看成

        S(m, b) = μ + 技巧_m + 难度_b + 噪声

    用所有格子联合估计技巧与难度，再回答"如果每家都被验证在同一批难度上，谁排
    前面"。既不用扔数据，也不用假设各源覆盖一致。

    weights / min_cell_weight / min_col_frac / ridge（对抗式审查 P0-1、P1-4）：
    等权拟合会把"5 条样本的格子"与"2,360 条样本的格子"一视同仁，而本项目的格子
    样本量跨三个数量级——名次因此是噪声排序的产物。weights 按格子信息量加权；
    min_cell_weight 把权重低于门槛的格子**直接剔出设计**（不靠 min_sample=5 放行）；
    min_col_frac 剔除"家数不足最热闹桶一半"的长尾桶；ridge 对列效应做收缩。
    四者都缺省关闭，等权路径与旧实现逐位相同（回归测试锁定）。

    segment_sizes：列被切成若干"赛段"（如总榜的 [b_hourly, b_daily]）时，要求
    行在每个非空赛段都有格子——见 design_mask。缺省 None 即不设赛段门槛。

    代价要说清楚：这是**加法假设**——若某家在短时效特别强、长时效特别弱（存在
    源 × 时效的交互），"对齐"后的单一数字表达不了这种差异，跨覆盖范围的比较仍
    应以分时效榜为准。交互项有多大由 variance_decomposition 量化并进披露。

    返回 dict：
      scores      (m,) 难度对齐行分（NaN = 该行无法参与横向比较）
      row_effects (m,) 行效应 α（= scores − μ；μ 为全场残差均值，通常 ≈0）
      col_effects (b,) 各天桶难度相对"平均难度"的偏离（NaN=未入设计）
      mu          全场平均难度下的参考水平
      row_keep / col_keep  入设计的行/列
      cell_valid  (m, b) 真正进入拟合的格子掩膜（P1-3 的"同一张设计"凭据）
      n_components 二分图连通分量数（>1 时只保留最大分量）
      dropped_thin_cells 因权重门槛被剔除的格子数（披露用）
      effective_min_col 实际生效的列家数门槛（含相对门槛）
    """
    S2 = np.asarray(S2, dtype=float)
    if S2.ndim != 2:
        raise ValueError("two_way_adjust 需要二维 (m, b) 分数矩阵")
    W0 = None
    if weights is not None:
        W0 = np.asarray(weights, dtype=float)
        if W0.shape != S2.shape:
            raise ValueError("weights 必须与分数矩阵同形状")
    V0 = np.isfinite(S2)
    dropped_thin = 0
    if W0 is not None and min_cell_weight > 0:
        thin = V0 & ~(np.isfinite(W0) & (W0 >= min_cell_weight))
        dropped_thin = int(thin.sum())
        V0 = V0 & ~thin
    row_keep, col_keep = design_mask(V0, min_col=min_col, min_row=min_row,
                                     min_col_frac=min_col_frac,
                                     segment_sizes=segment_sizes)
    comp_rows, comp_cols = largest_component_mask(row_keep, col_keep, V0)
    n_components = 1
    if not (np.array_equal(comp_rows, row_keep) and np.array_equal(comp_cols, col_keep)):
        # 不连通：各分量各有自己的加法常数，分量间的差异不可识别——只留最大
        # 分量，其余暂不外比（调用方应把这件事写进披露信息）
        row_keep, col_keep = comp_rows, comp_cols
        n_components = 2
    V = V0 & row_keep[:, None] & col_keep[None, :]
    if not V.any():
        return {"scores": np.full(S2.shape[0], np.nan),
                "row_effects": np.full(S2.shape[0], np.nan),
                "col_effects": np.full(S2.shape[1], np.nan),
                "mu": None, "row_keep": row_keep, "col_keep": col_keep,
                "n_components": n_components, "cell_valid": V,
                "dropped_thin_cells": dropped_thin,
                "effective_min_col": int(min_col)}
    W3 = W0[None, ...] if W0 is not None else None
    parts = _fit_parts(S2[None, ...], row_keep, col_keep, max_iter, tol,
                       weights=W3, ridge=ridge)
    mu, alpha, beta, V3 = parts
    scores = np.where(row_keep, mu[0] + alpha[0], np.nan)
    return {"scores": scores, "row_effects": np.where(row_keep, alpha[0], np.nan),
            "col_effects": np.where(col_keep, beta[0], np.nan),
            "mu": float(mu[0]) if np.isfinite(mu[0]) else None,
            "row_keep": row_keep, "col_keep": col_keep,
            "n_components": n_components, "cell_valid": V3[0],
            "dropped_thin_cells": dropped_thin,
            "effective_min_col": int(min_col)}


# ------------------------------------------------ 加法假设的量化与名次稳定性
def variance_decomposition(S: np.ndarray, row_keep: np.ndarray,
                           col_keep: np.ndarray,
                           weights: np.ndarray | None = None) -> dict:
    """把 (m, b) 分数矩阵的总方差拆成 源间 / 时效 / 残差（交互+噪声）。

    总榜的"单一数字"能不能表达"谁家预报最准"，取决于源×时效**交互项**有多大：
    交互大 ⇒ 一个源在 1 天最强、在 10 天最弱，把它压成一个数字就是在抹平真实差异。
    README 承认了这个代价，却从未量化它；本函数把它变成可核对的比例（P0-2）。

    用拟合出的加性模型做平方和分解（在入设计的格子上）：
      SS_row = Σ(α_m)²   SS_col = Σ(β_b)²   SS_resid = Σ r²
      SS_total = Σ(S − S̄)²
    残差份额 = SS_resid / SS_total，即"加法模型解释不了的那部分"。

    返回 dict：total / row / col / residual（方差）与 row_share / col_share /
    residual_share（比例，和≈1）、n_cells。
    """
    S = np.asarray(S, dtype=float)
    V = np.isfinite(S) & np.asarray(row_keep, dtype=bool)[:, None] \
        & np.asarray(col_keep, dtype=bool)[None, :]
    n = int(V.sum())
    if n == 0:
        return {"total": None, "row": None, "col": None, "residual": None,
                "row_share": None, "col_share": None, "residual_share": None,
                "n_cells": 0}
    w = None
    if weights is not None:
        w = np.where(V, np.asarray(weights, dtype=float), 0.0)
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    # 必须用**未平移**的原始行/列效应：_fit_parts 会把 alpha 加上"保留列的平均
    # 难度"（好让 μ+α 直接就是分数），平移后的 alpha 含 μ≈60，平方和比总平方和
    # 还大——份额会算出 49.8 这种不可能的数（曾把这条披露变成噪声）。
    alpha_raw, beta_raw = two_way_fit(S[None, ...], V[None, ...],
                                      weights=(w[None, ...] if w is not None else None),
                                      max_iter=300, tol=1e-10)
    if alpha_raw.size == 0:
        return {"total": None, "row": None, "col": None, "residual": None,
                "row_share": None, "col_share": None, "residual_share": None,
                "n_cells": n}
    vals = S[V]
    ww = w[V] if w is not None else np.ones_like(vals)
    ww = np.where(ww > 0, ww, 0.0)
    if ww.sum() <= 0:
        ww = np.ones_like(vals)
    grand = float(np.sum(ww * vals) / np.sum(ww))
    # 行/列效应按**拟合所用的同一组权重**中心化：加权 ALS 解出的是 Σw·α=0、
    # Σw·β=0（逐格权重），只有按同一个 w 加权中心化，平方和分解 Σw(S−S̄)² =
    # SS_row + SS_col + SS_resid 才是恒等式。此前按无权均值中心化（权重全 1 时
    # 两者恰好重合，测试看不出），权重非均匀时三项之和会偏离 1（实测偏差 1e-4）。
    row_w = np.where(np.asarray(row_keep, dtype=bool),
                     w.sum(axis=1) if w is not None else V.sum(axis=1), 0.0)
    col_w = np.where(np.asarray(col_keep, dtype=bool),
                     w.sum(axis=0) if w is not None else V.sum(axis=0), 0.0)
    a0 = np.where(np.asarray(row_keep, dtype=bool), alpha_raw[0], 0.0)
    b0 = np.where(np.asarray(col_keep, dtype=bool), beta_raw[0], 0.0)
    a0 = np.where(np.isfinite(a0), a0, 0.0)
    b0 = np.where(np.isfinite(b0), b0, 0.0)
    a0 = a0 - (float(np.sum(row_w * a0)) / max(float(np.sum(row_w)), 1e-12))
    b0 = b0 - (float(np.sum(col_w * b0)) / max(float(np.sum(col_w)), 1e-12))

    def _ss(resid_grid: np.ndarray) -> float:
        r = np.where(V, resid_grid, 0.0)
        return float(np.sum(ww * (r[V] ** 2)))

    # 总平方和必须是**离均差**的平方和 Σw(S−S̄)²；写成 Σw·S̄² 会把 S̄≈57 的整块
    # 常数塞进总方差（实测把 total 抬高 200 倍），residual = total − row − col 随之
    # 被 max(0,·) 压到 0，于是"交互项占比"永远显示成一个小得离谱的数。
    ss_row = _ss(np.broadcast_to(a0[:, None], S.shape))
    ss_col = _ss(np.broadcast_to(b0[None, :], S.shape))
    ss_total = _ss(S - grand)
    ss_resid = max(0.0, ss_total - ss_row - ss_col)
    denom = ss_total if ss_total > 0 else None
    share = (lambda x: round(x / denom, 4) if denom else None)
    return {
        "total": round(ss_total / max(int(V.sum()), 1), 4),
        "row": round(ss_row / max(int(V.sum()), 1), 4),
        "col": round(ss_col / max(int(V.sum()), 1), 4),
        "residual": round(ss_resid / max(int(V.sum()), 1), 4),
        "row_share": share(ss_row),
        "col_share": share(ss_col),
        "residual_share": share(ss_resid),
        "n_cells": int(V.sum()),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman 秩相关（无 scipy 依赖；并列值用平均秩）。"""
    if a.size != b.size or a.size < 3:
        return None
    ra, rb = _rankdata(a), _rankdata(b)
    return pearson_r(ra, rb)


def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(1, x.size + 1, dtype=float)
    # 并列值取平均秩
    sx = x[order]
    i = 0
    while i < sx.size:
        j = i
        while j + 1 < sx.size and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    return ranks


def bucket_rank_stability(S: np.ndarray, row_keep: np.ndarray,
                          col_keep: np.ndarray, min_common: int = 5) -> dict:
    """跨天桶的名次一致性（Spearman）：总榜的单一数字是否配得上"名次"这个词。

    逐桶把难度扣掉（score_b(m) = S[m,b] − β_b），再算两两桶的 Spearman。实测
    桶1 vs 桶7 只有 0.19——意味着"短时效第 3 名"与"长时效第 3 名"往往不是同一家。
    这个数字此前一个都没有，读者无从知道总榜的适用边界（P0-2）。
    """
    S = np.asarray(S, dtype=float)
    row_keep = np.asarray(row_keep, dtype=bool)
    col_keep = np.asarray(col_keep, dtype=bool)
    parts = _fit_parts(S[None, ...], row_keep, col_keep, 300, 1e-10)
    if parts is None:
        return {"pairs": [], "min_pair": None, "adjacent_mean": None}
    _mu, _alpha, beta, V = parts
    b3 = beta[0]
    kept = [b for b in range(S.shape[1]) if col_keep[b]]
    pairs = []
    for i, bi in enumerate(kept):
        for bj in kept[i + 1:]:
            Vb = V[0][:, bi] & V[0][:, bj]
            if int(Vb.sum()) < min_common:
                continue
            a = S[Vb, bi] - (b3[bi] if np.isfinite(b3[bi]) else 0.0)
            c = S[Vb, bj] - (b3[bj] if np.isfinite(b3[bj]) else 0.0)
            rho = _spearman(a, c)
            if rho is not None:
                pairs.append({"a": bi + 1, "b": bj + 1, "rho": round(float(rho), 3),
                              "n": int(Vb.sum())})
    if not pairs:
        return {"pairs": [], "min_pair": None, "adjacent_mean": None}
    rhos = [p["rho"] for p in pairs]
    adjacent = [p["rho"] for p in pairs if p["b"] - p["a"] == 1]
    return {
        "pairs": pairs,
        "min_pair": {"a": min(pairs, key=lambda p: p["rho"])["a"],
                     "b": min(pairs, key=lambda p: p["rho"])["b"],
                     "rho": min(rhos)},
        "adjacent_mean": round(float(np.mean(adjacent)), 3) if adjacent else None,
        "overall_mean": round(float(np.mean(rhos)), 3),
    }


def rank_sensitivity(equal_scores: np.ndarray, weighted_scores: np.ndarray) -> dict:
    """等权 vs 加权两套名次的差异（P0-1 的披露面）。

    榜单不能只给"加权后的名次"就完事——读者有权知道名次对加权方式的敏感度。
    返回 Spearman 与变动最大的若干源（按名次差绝对值），以及"前十换了几家"。
    """
    a = np.asarray(equal_scores, dtype=float)
    b = np.asarray(weighted_scores, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    n = int(m.sum())
    if n < 3:
        return {"spearman": None, "n": n, "movers": [], "top10_changed": None}
    rho = _spearman(a[m], b[m])
    # 名次（1 = 最好）；只对同时有效的源排名，未入围者不参与
    idx = np.flatnonzero(m)
    order_a = idx[np.argsort(-a[m])]
    order_b = idx[np.argsort(-b[m])]
    rank_a = {int(k): i + 1 for i, k in enumerate(order_a)}
    rank_b = {int(k): i + 1 for i, k in enumerate(order_b)}
    movers = sorted(({"index": int(k), "rank_equal": rank_a[k],
                      "rank_weighted": rank_b[k], "delta": rank_b[k] - rank_a[k]}
                     for k in idx), key=lambda d: -abs(d["delta"]))
    top_a = {int(k) for k in order_a[:10]}
    top_b = {int(k) for k in order_b[:10]}
    return {"spearman": round(float(rho), 3) if rho is not None else None,
            "n": n, "movers": movers[:8],
            "top10_changed": len(top_a ^ top_b) // 2}


def holm_bonferroni(pvals: list[float], alpha: float = 0.10) -> list[bool]:
    """Holm–Bonferroni 逐步校正：控制族错误率（FWER）的通用做法。

    26 个源同榜、每个都与第一名比一次，α=0.1 下至少一次假阳性的概率约 93%——
    不经校正的"† 与第一名无显著差异"标记基本是噪声。Holm 把第 k 小的 p 值与
    α/(m−k+1) 比较，一旦不显著则此后全部判不显著（逐步降级、单调）。

    返回与输入等长的显著/不显著布尔列表（True = 显著）。
    """
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (pvals[i] is None, pvals[i]))
    out = [False] * m
    still = True
    for k, i in enumerate(order):
        p = pvals[i]
        if p is None:
            still = False
        elif still and p <= alpha / (m - k):
            out[i] = True
        else:
            still = False
    return out


def _summarize_bootstrap(macro: np.ndarray, models: list[str],
                         eligible: list[bool], alpha: float = 0.10,
                         top_model: str | None = None) -> dict[str, dict[str, Any]]:
    """(runs, m) 的 macro 分 → 每模型的 CI90 / 冠军频率 / 与冠军的显著性。

    sig_vs_top 经 Holm–Bonferroni 逐步校正（P1-3）：与第一名比较是"一族检验"，
    逐对 α=0.1 不做校正在 26 源同榜下假阳性率约 93%。原始 p 值在 p_vs_top 里
    一并给出，未校正的判定在 sig_vs_top_raw 里保留——校正前后都可见，读者能
    自己判断结论对校正有多敏感。

    top_model：与"榜单上戴冠的那个源"比较（由调用方按点估计名次传入）。显著性
    的参照必须是读者看到的第一名——bootstrap 分布均值最高的源偶尔会与点估计
    冠军分叉，那时 † 标记会指向一个并非冠军的源。缺省时退回分布均值最高者。
    """
    out: dict[str, dict[str, Any]] = {}
    point_top = None
    if top_model is not None and top_model in models:
        ti = models.index(top_model)
        if eligible[ti]:
            point_top = (ti, float(np.nanmean(macro[:, ti])))
    for mi, m in enumerate(models):
        scores = macro[:, mi]
        finite = scores[np.isfinite(scores)]
        ci = None
        if finite.size >= max(5, macro.shape[0] // 10):
            lo, hi = np.percentile(finite, [5, 95])
            ci = [round(float(lo), 2), round(float(hi), 2)]
        out[m] = {"ci90": ci, "champion_pct": 0.0, "sig_vs_top": None,
                  "sig_vs_top_raw": None, "p_vs_top": None}
        if not eligible[mi] or ci is None:
            continue
        if point_top is not None:
            continue
        mean_score = float(np.nanmean(scores))
        if point_top is None or point_top[1] < mean_score:
            point_top = (mi, mean_score)
    # 冠军频率（逐 run 的 nan-aware 最大；某 run 全模型无分时该 run 不计冠军）。
    # 只有入围者可成为冠军（未入围者不应凭高温度分在频率表上占位）
    elig_arr = np.array(eligible, dtype=bool)
    macro_elig = np.where(elig_arr[None, :], macro, np.nan)
    finite = np.isfinite(macro_elig)
    # 与权重敏感性同样的平局处理：先取整到 1e-6 分再比大小，避免浮点噪声
    # （1e-14 级）决定"谁在这一轮是冠军"
    best = np.where(finite, np.round(macro_elig, 6), -np.inf).argmax(axis=1)
    best = np.where(finite.any(axis=1), best, -1)
    counted = int((best >= 0).sum())
    for mi, m in enumerate(models):
        pct = round(100.0 * float((best == mi).sum()) / counted, 1) if counted else 0.0
        out[m]["champion_pct"] = pct
    # 与冠军的显著性：先用配对 bootstrap 分布求双侧 p 值，再对"整族比较"做
    # Holm–Bonferroni 校正
    if point_top is not None:
        top_i = point_top[0]
        idxs: list[int] = []
        pvals: list[float | None] = []
        for mi, m in enumerate(models):
            if mi == top_i:
                out[m]["sig_vs_top"] = True
                out[m]["sig_vs_top_raw"] = True
                out[m]["p_vs_top"] = 0.0
                continue
            if not eligible[mi]:
                continue
            idxs.append(mi)
            with np.errstate(invalid="ignore"):
                diff = macro[:, top_i] - macro[:, mi]
                d = diff[np.isfinite(diff)]
            if d.size < max(5, macro.shape[0] // 10):
                pvals.append(None)
                continue
            # 双侧 p：分布落在 0 另一侧的比例 ×2（配对重采样：同一 run 比同一 run）
            pvals.append(min(1.0, 2.0 * min(float((d <= 0).mean()),
                                            float((d >= 0).mean()))))
        for mi, p, sig in zip(idxs, pvals, holm_bonferroni(pvals, alpha=alpha)):
            if p is None:
                continue
            out[models[mi]]["p_vs_top"] = round(p, 4)
            out[models[mi]]["sig_vs_top_raw"] = bool(p <= alpha)
            out[models[mi]]["sig_vs_top"] = bool(sig)
    return out


def aggregate_day_stats(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(runs, n_days) 权重 × (m, b, s, d, k) 充分统计量表 → (runs, m, b, s, k)。

    温度/降水两条轨道共用；k 维是各自的可加统计量（温度 11 项 / 降水 4 项）。
    """
    out = np.empty(W.shape[:1] + X.shape[:3] + (X.shape[-1],))
    for k in range(X.shape[-1]):
        out[..., k] = np.einsum("rd,mbsd->rmbs", W, X[:, :, :, :, k])
    return out


def _nanmean_stack(arrs: list[np.ndarray]) -> np.ndarray:
    """沿新轴做忽略 NaN 的均值；全缺处为 NaN（不刷 RuntimeWarning）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(np.stack(arrs), axis=0)


def track_bucket_scores(tables: dict[str, np.ndarray],
                        temp_parts, precip_parts, min_sample: int,
                        temp_point_valid: np.ndarray | None = None,
                        rain_point_valid: np.ndarray | None = None,
                        bucket_valid: np.ndarray | None = None,
                        ) -> np.ndarray:
    """证据表 (run?, m, b, s, k) → 跨分辨率拼接的桶综合分 (run?, m, 2·b)。

    列布局是这次重构的关键约定：**前 b 列是小时榜的天桶、后 b 列是日榜的天桶**
    （第 k 列与第 b+k 列指同一个"提前第 k 天"，只是时间分辨率不同）。拼成一条
    横轴后，(天桶 × 分辨率) 就是"难度"的一个自然笛卡尔积，总榜直接把这
    2b 列塞进同一个加法模型——每小时/每日各自的难度由各自的列效应吸收，
    行效应则是"跨两种分辨率、所有被验证过的难度"的综合技巧。

    三处缺项口径与点估计逐格对齐：
    · 样本非零但 < min_sample 的维度在该次重采样里为缺；
    · 点估计判缺的格子恒为缺（点估计还看 n_eff，重采样只能算 n，两者互补）；
    · 日榜温度维要求日最高与日最低**两个量都有结论**——日预报承诺的是这两个
      数，只凭其中一个给分等于把半个证据当整个用（源可以靠容易的那一半刷分）。
    """
    t_h = _temp_scores_from_aggregate(tables["temp_hourly"], temp_parts)
    r_h = _rain_scores_from_aggregate(tables["rain_hourly"], precip_parts)
    t_max = _temp_scores_from_aggregate(tables["temp_daily_max"], temp_parts)
    t_min = _temp_scores_from_aggregate(tables["temp_daily_min"], temp_parts)
    r_d = _rain_scores_from_aggregate(tables["rain_daily"], precip_parts)
    # 各维的有效成对样本数：沿站维求和后是 (run?, m, b)，与各自的计分函数内部
    # 用的 N 同口径（_temp_scores_from_aggregate 里 N = n.sum(-1)）。
    # 日最高/日最低**各自用自己的样本数**判定：两边都薄时"相加凑够样本"会把两个
    # 都不足门槛的量伪装成一个够样本的维度。
    n_th = tables["temp_hourly"][..., 0].sum(axis=-1)
    n_rh = tables["rain_hourly"].sum(axis=(-2, -1))
    n_tmax = tables["temp_daily_max"][..., 0].sum(axis=-1)
    n_tmin = tables["temp_daily_min"][..., 0].sum(axis=-1)
    n_rd = tables["rain_daily"].sum(axis=(-2, -1))

    def _thin(scores, counts):
        return np.where((counts > 0) & (counts < min_sample), np.nan, scores)

    t_h, r_h = _thin(t_h, n_th), _thin(r_h, n_rh)
    t_max, t_min = _thin(t_max, n_tmax), _thin(t_min, n_tmin)
    r_d = _thin(r_d, n_rd)
    # 日温度维：最高与最低必须两两齐全
    daily_t = np.where(np.isfinite(t_max) & np.isfinite(t_min),
                       _nanmean_stack([t_max, t_min]), np.nan)

    has_run = t_h.ndim == 3
    if not has_run:
        t_h, r_h, daily_t, r_d = t_h[None], r_h[None], daily_t[None], r_d[None]
    b = t_h.shape[-1]
    # 逐 track 施加掩码后再拼接：跨整条 2b 横轴做布尔索引会丢掉"这一半是哪一条
    # 轨道"的信息（过去把 station-wise 求和忘在同一层，吃亏的就是这类位置耦合）
    tv_h, rv_h = np.isfinite(t_h), np.isfinite(r_h)
    tv_d, rv_d = np.isfinite(daily_t), np.isfinite(r_d)
    if temp_point_valid is not None:
        pv = np.asarray(temp_point_valid, dtype=bool)
        tv_h, tv_d = tv_h & pv[:, :b], tv_d & pv[:, b:]
    if rain_point_valid is not None:
        pv = np.asarray(rain_point_valid, dtype=bool)
        rv_h, rv_d = rv_h & pv[:, :b], rv_d & pv[:, b:]
    if bucket_valid is not None:
        bv = np.asarray(bucket_valid, dtype=bool)
        valid_h, valid_d = bv[:, :b], bv[:, b:]
    else:
        valid_h, valid_d = tv_h & rv_h, tv_d & rv_d
    h_scores = np.where(
        valid_h,
        _nanmean_stack([np.where(tv_h, t_h, np.nan),
                        np.where(rv_h, r_h, np.nan)]), np.nan)
    d_scores = np.where(
        valid_d,
        _nanmean_stack([np.where(tv_d, daily_t, np.nan),
                        np.where(rv_d, r_d, np.nan)]), np.nan)
    # 必须用 concatenate(axis=-1) 而不是 hstack：hstack 对 ndim≥2 的数组沿
    # **axis 1** 拼接，而这里的倒数第二维是"模型"，天桶才是最后一维——用 hstack
    # 会把两条轨道沿着模型维拼起来，形状变成 (run, 2m, b) 还不报错。
    scores = np.concatenate([h_scores, d_scores], axis=-1)
    return scores if has_run else scores[0]


def macro_scores_from_weights(
    W: np.ndarray, tables: dict[str, np.ndarray],
    temp_parts, precip_parts, min_sample: int,
    temp_point_valid: np.ndarray | None = None,
    rain_point_valid: np.ndarray | None = None,
    bucket_valid: np.ndarray | None = None,
    adj_row: np.ndarray | None = None,
    adj_col: np.ndarray | None = None,
    adj_w: np.ndarray | None = None,
    adj_ridge: float = 0.0,
) -> np.ndarray:
    """(runs, n_days) 天权重 → 每次重采样的总榜综合分 (runs, n_models)。

    归总方式必须与榜单上那个数字**完全同构**：点估计用双向加法模型把天桶难度
    劈掉后再取行分（two_way_adjust），这里就用同一张设计走同一个函数。给 A 数字
    配 B 数字的置信区间是直接误导读者（P0-2 的教训）。难度对齐让这条不变量更值得
    机器校验——因为此时"归总"不再只是"某几列取个均值"这么直观，肉眼对不上。

    tables：build_day_stat_tables 出的五张证据表（未聚合，(m, b, s, d, k)）；
    本函数按 W 先把天维加权聚合成 (run, m, b, s, k)，再交给 track_bucket_scores。

    temp_point_valid / rain_point_valid / bucket_valid：(m, 2·b) 布尔，列布局同
    track_bucket_scores（前 b 列小时榜、后 b 列日榜）。点估计在哪个格子有结论，
    bootstrap 就在哪个格子有结论——两条门槛必须逐格对齐，CI 中心才不漂。

    单独成函数是为了让"退化 bootstrap"可测：W 全置 1 时等价于不做重采样，
    返回的行分必须精确等于点估计的总榜综合分。这条不变量一次性兜住所有
    "bootstrap 与点估计口径漂移"类缺陷（2026-09-13 P0-1 的教训：当时唯一根因
    是降水的样本量字段取错，注释里写了意图却没有机器校验）。
    """
    agg = {k: aggregate_day_stats(W, v) for k, v in tables.items()}
    bucket_scores = track_bucket_scores(
        agg, temp_parts, precip_parts, min_sample,
        temp_point_valid=temp_point_valid, rain_point_valid=rain_point_valid,
        bucket_valid=bucket_valid)
    if adj_row is None or adj_col is None:
        # 未给设计 → 旧 macro 口径（各桶等权平均），不去除天桶难度
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(bucket_scores, axis=-1)
    # 权重与设计**必须与点估计逐格同源**：否则每次重采样回答的是另一个估计量，
    # 置信区间的中心会系统性偏离榜单上那个数字（P0-1 与 2026-09-13 P0-1 同一条教训）
    return difficulty_adjusted(bucket_scores, np.asarray(adj_row, dtype=bool),
                               np.asarray(adj_col, dtype=bool),
                               weights=(None if adj_w is None else np.asarray(adj_w, dtype=float)),
                               ridge=adj_ridge)


def weight_champion_distribution(
    temp_sub: np.ndarray, precip_sub: np.ndarray, temp_parts, precip_parts,
    runs: int = 500, seed: int = 20260907,
    adj_row: np.ndarray | None = None,
    adj_col: np.ndarray | None = None,
    macro_weight_range: tuple[float, float] = (0.30, 0.70),
    adj_w: np.ndarray | None = None,
    adj_ridge: float = 0.0,
) -> list[dict]:
    """权重敏感性（P0-2.2）：把各项权重各扰动 ±40%，统计冠军分布。

    temp_sub / precip_sub：(m, b, k) 的**已换算并截断**的子分张量（NaN=缺项），
    k 顺序与 parts 表一致。权重 w ~ U(0.6, 1.4)×原权重，逐 run 重组
    温度分/降水分 → 桶综合分 → 难度对齐行分 → 冠军。返回
    [{"model": m, "pct": 频率%}, ...]（降序，含 0 频率外的全部模型）。

    macro_weight_range（P1-2，本函数此前最大的漏洞）：温度:降水的**宏观权重**
    也要扰动。旧实现把两维各自按扰动权重归一化后**恒定按 50:50 平均**——于是
    全榜最有争议、最影响名次的那个参数压根不在敏感性分析范围内。为什么它特别
    要紧：温度分均值 79.5（极差 57~93）、降水分均值 33.5（极差 0~69），两维
    均值差 46 分、标准差差 1.4 倍，50:50 与 70:30 给出的名次完全不同。默认
    U(0.30, 0.70) 覆盖"任一模态最多占七成"的合理分歧区间。

    adj_row / adj_col / adj_w：与总榜名次同尺——权重敏感性回答的是"名次对权重有多
    敏感"，若用另一把尺子归总，答的就是另一个冠军。故这里也走同一步难度对齐
    （同一张设计、同一组格子权重）。
    """
    n_m, n_b = temp_sub.shape[0], temp_sub.shape[1]
    w_t0 = np.array([p[1] for p in temp_parts])
    w_p0 = np.array([p[1] for p in precip_parts])
    rng = np.random.default_rng(seed)
    wt = w_t0[None, :] * rng.uniform(0.6, 1.4, size=(runs, len(w_t0)))
    wp = w_p0[None, :] * rng.uniform(0.6, 1.4, size=(runs, len(w_p0)))
    # 宏观权重：温度占综合分的比例，逐 run 独立抽取
    alpha_lo, alpha_hi = macro_weight_range
    w_macro = rng.uniform(alpha_lo, alpha_hi, size=(runs, 1, 1))

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
    # 宏观权重下的缺项加权平均：nums/总权重，而不是 nanmean（后者恒为 50:50）
    v_t = np.isfinite(t_score)
    v_p = np.isfinite(p_score)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(invalid="ignore", divide="ignore"):
            tot_w = w_macro * v_t + (1.0 - w_macro) * v_p
            num = w_macro * np.where(v_t, t_score, 0.0) \
                + (1.0 - w_macro) * np.where(v_p, p_score, 0.0)
            bucket = np.where(tot_w > 0, num / np.where(tot_w > 0, tot_w, 1.0), np.nan)
    if adj_row is None or adj_col is None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            macro = np.nanmean(bucket, axis=-1)        # (run, m)
    else:
        macro = difficulty_adjusted(bucket, np.asarray(adj_row, dtype=bool),
                                    np.asarray(adj_col, dtype=bool),
                                    weights=(None if adj_w is None
                                             else np.asarray(adj_w, dtype=float)),
                                    ridge=adj_ridge)
    # 某 run 全模型无分时该 run 不计冠军。
    # **先按 1e-6 分取整再取最大**：数值上完全平局的两家会因浮点结合律差出
    # 1e-14（实测 0.5·100+0.5·0 与 0.5·0+0.5·100 不等），argmax 于是被噪声
    # 决定——冠军频率会被凭空摊薄（实测平局用例出现 84/16 而不是 100/0）。
    # 1e-6 分远小于任何真实差异，取整只吃掉浮点噪声。
    finite = np.isfinite(macro)
    best = np.where(finite, np.round(macro, 6), -np.inf).argmax(axis=1)
    best = np.where(finite.any(axis=1), best, -1)
    counts = np.bincount(best[best >= 0], minlength=n_m)
    total = counts.sum()
    # 返回按频率降序的 (模型下标, 频率%) 列表，模型名由调用方映射
    order = np.argsort(-counts)
    return [{"index": int(i), "pct": round(100.0 * float(counts[i]) / total, 1)}
            for i in order if counts[i] > 0]

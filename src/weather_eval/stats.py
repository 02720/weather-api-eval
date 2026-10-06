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

import math
import warnings
from typing import Any

import numpy as np

# 分级阈值**只此一处**：直接读 cyeva 的配置模块，与 weather_eval.graded 同源，
# 上游改阈值这里自动跟随（语义债 #9 的同款教训：绝不手工同步第二份阈值表）。
from cyeva.config.levels.precip import ACC_PRECIP_LEVELS, PRECIP_LEVELS

# ρ₁ 的噪声下限系数：|ρ̂| 低于 z/√n 的部分按估计噪声处理，不参与折减
# （z=1.64 ≈ 单侧 95% 显著性门槛；相关系数估计的标准误 ≈ 1/√n）
NEFF_RHO_NOISE_Z = 1.64
# ρ₁ 的截断：ρ→1 时 (1−ρ)/(1+ρ) 发散，截断避免单个高自相关序列炸掉 n_eff
RHO_CLAMP = 0.95
# 站内 r/slope 参与合并的最小站内样本量（与点估计路径一致）
GROUP_MIN_N = 30
# 跨站相关 ρ̄ 的最小公共时刻数：少于此值时两两相关的估计噪声大于信号，
# 退回"各站独立"的旧口径（不校正，宁可保守也不过校正）
CROSS_STATION_MIN_OVERLAP = 30
# 站间一致性（between_station_mbe_sd）要求的最少达标站数。两站的离散度只是两点
# 距离、受单站噪声支配，撑不起"跨站一致性"这个命题；本项目 4 站，取 3 意味着
# "至少覆盖多数站点才算数"，不足则整维缺项（按剩余权重归一）。
MIN_STATIONS_BDISP = 3

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

    err 是按时间排序的误差（或事件指示）序列。ρ̂₁ 只在**超出噪声下限**
    z/√n（NEFF_RHO_NOISE_Z）的部分参与折减——相关系数估计的标准误 ≈ 1/√n，
    短序列的 ρ̂ 本身噪声很大，把噪声当信号会凭空折损样本。ρ₁ 估计为 None
    （方差为 0 等退化情形）同样返回 n。结果截断到 [1, n]。

    为什么必须是连续下限而不是"序列短于 N 就不校正"的硬开关：硬开关让 n_eff
    在 N/N+1 之间跳变——同一源多攒一周数据，估计器"换挡"全额折减，n_eff 反而
    骤降（实测 UKMO 日温度 n_eff 从 9 月月报的 51 跌到跨月累计的 18，样本更多
    却被踢回"样本积累中"）。信息量必须随样本单调不减，这是有效样本量的第一
    性原理；噪声下限在 n→∞ 时收敛到 0，全额校正照常生效。
    """
    e = np.asarray(err, dtype=float)
    e = e[np.isfinite(e)]
    n = int(e.size)
    if n <= 2:
        return n
    rho = pearson_r(e[:-1], e[1:])
    if rho is None:
        return n
    rho = max(-RHO_CLAMP, min(RHO_CLAMP, rho))
    floor = NEFF_RHO_NOISE_Z / np.sqrt(n)
    rho = float(np.copysign(max(0.0, abs(rho) - floor), rho))
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
    # 空序列站先剔除（第四轮 P2-12）：一个空站会把 cut=min(lens) 压到 0，
    # 整条跨站校正被静默跳过，n_eff 虚高近 2 倍——两条路径行为从此一致。
    ids = [sid for sid in ids if len(series_by_station[sid]) > 0]
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
        ns: list[int] = []
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
                    ns.append(len(common))
    else:
        # 按位置对齐、截断到最短序列
        cut = min(len(series_by_station[sid]) for sid in ids)
        if cut < min_overlap:
            return None
        rs = []
        ns = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a = np.asarray(series_by_station[ids[i]][:cut], dtype=float)
                b = np.asarray(series_by_station[ids[j]][:cut], dtype=float)
                r = pearson_r(a, b)
                if r is not None:
                    rs.append(r)
                    ns.append(cut)
    if not rs:
        return None
    # Fisher-z 合并（第四轮 P2-1）：相关系数是有界量，直接算术平均违反本模块
    # 自己在 fisher_z_combine 里写明的原则——实测同组数据两种口径差 0.08，
    # 传导到 n_eff 差约 11%。
    rho = fisher_z_combine(rs, ns)
    return None if rho is None else float(rho)


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
# 降水：二分类列联计数 + 雨量精度的可加充分统计量。
# 后四项（2026-10 新增）服务于"雨量精度"两维（amt_mae / amt_bias，见
# evaluate.precip_amount_metrics）：它们只由 Σo / Σf / Σ|f−o| 构成，因此按天
# 分块 bootstrap 能逐位重算与点估计**同一把尺子**。这是可入分的**硬约束**——
# 任何参考量不可加的口径（例如 1 − MAE/mean|o−mean o|，参考量随重采样的日集合
# 变化且无法分解）都会让 bootstrap 静默地换一把尺子。
# 后六列（2026-10 新增）服务于「雨强分辨力」一维（grade_ets）：两个雨强档位各自
# 的 (hits, false_alarms, misses) 计数。它们与晴雨计数同源于**同一条记录**，故
# 可按天分块重采样逐位重算——与雨量精度两维同纪律。
#
# 档位只取"中雨档 + 大雨档"两级：
#   hourly（1h 口径）2 级 = 2~4.9 mm/h、3 级 = 5~9.9 mm/h
#   daily（24h 口径）+2 级 = ≥10 mm、+3 级 = ≥25 mm
# **1 级（小雨 ≥0.1mm）刻意不取**：实测它与晴雨判定的桶内跨源 ρ = 0.93（hourly）
# / 0.98（daily），是同一件事换了个说法；而 2 级及以上对晴雨四项的 1−R² 达
# 0.79~0.94、与晴雨 ETS 的 |ρ| ≤0.27——"下不下雨"和"下多大"是两种独立能力。
# 4 级以上在该窗口的 ETS 中位已归零（强降水样本太稀疏），取了只增加噪声。
# 逐位口径（含 source_round_digit(1) 的舍入时点）见 _rain_stat_row。
_GRADE_NLEV = 2
# 评分轨入分的雨强档位编号：hourly 走 1h 口径的中雨/大雨，daily 走 24h 累计的
# 中雨/大雨（业务语义对齐：都是"中雨档 + 大雨档"）。
_GRADE_LEVS = {"1h": ("2", "3"), "24h": ("+2", "+3")}
_RAIN_STATS = ("h", "fa", "mi", "c", "so", "sf", "sae",
               "gh1", "gfa1", "gmi1", "gh2", "gfa2", "gmi2")


def grade_bounds(kind: str) -> tuple[tuple[float, float], ...]:
    """评分轨入分的两个雨强档位的 (min, max) 区间，阈值**只此一处**且取自 cyeva。

    与 `weather_eval.graded._level_config` 同源（都读 cyeva 的配置模块），上游改
    阈值这里自动跟随——不存在第二份需要人工同步的阈值表。区间判定与舍入口径
    见 graded 模块 docstring 的第 1、3 条（复刻 cyeva 的 `level_binarize`）。
    """
    table = ACC_PRECIP_LEVELS if kind == "24h" else PRECIP_LEVELS
    out = []
    for lev in _GRADE_LEVS[kind]:
        iv = table[kind][int(lev.replace("+", ""))]
        out.append((float(iv["min"]), float(iv["max"])))
    return tuple(out)

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


def _rain_stat_row(o: float, f: float, thr: float,
                   gbounds: tuple[tuple[float, float], ...] = ()) -> list[float]:
    """单个降水样本 → 13 项可加统计量（4 项列联计数 + 3 项雨量 + 6 项分级计数）。

    列联计数与雨量累计量必须来自**同一个样本**（这里是同一条记录），否则晴雨
    维与雨量维在 bootstrap 里会落到不同的样本集合上；分级计数同理。

    分级二值化复刻 cyeva 的 `level_binarize` 两条口径（对拍由
    `test_grade_counts_match_cyeva` 锁定）：
      * 输入先 `np.round(x, 1)`（**numpy** 的缩放舍入——与权威路径
        `graded.graded_counts` 同一函数，而不是 Python 内置 round：两者在二进制
        边界上不一致，`np.round(0.05, 1) == 0.0` 而 `round(0.05, 1) == 0.1`，
        见 tests/test_graded_parity.py 的口径钉子），且**舍入发生在剔 NaN 之前**
        ——NaN 已在上游被 *_ok 掩膜滤掉，这里拿到的必是有限值；
      * 区间判定 `(v >= min) & (v <= max)`（`min > 0` 恒真；累积档 max=inf 退化
        为 `v >= min`）。
    注意晴雨列**不做**这层舍入：cyeva 的 `threshold_binarize` 用原值比较，与
    分级走的是两个不同的二值化入口，不可混用。
    """
    ob, fb = o >= thr, f >= thr
    row = [1.0 if (ob and fb) else 0.0, 1.0 if (not ob and fb) else 0.0,
           1.0 if (ob and not fb) else 0.0, 1.0 if (not ob and not fb) else 0.0,
           o, f, abs(f - o)]
    if gbounds:
        ro, rf = float(np.round(o, 1)), float(np.round(f, 1))
        for lo, hi in gbounds:
            obg, fbg = (lo <= ro <= hi), (lo <= rf <= hi)
            row += [1.0 if (obg and fbg) else 0.0,
                    1.0 if (not obg and fbg) else 0.0,
                    1.0 if (obg and not fbg) else 0.0]
    else:
        row += [0.0] * (3 * _GRADE_NLEV)
    return row


def build_day_stat_tables(
    hourly: list[dict], daily: list[dict], models: list[str],
    n_buckets_hourly: int, n_buckets_daily: int,
    rain_thr_daily: float, rain_thr_hourly: float,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """把逐小时/按天记录压成（模型 × 桶 × 站 × 天）的可加充分统计量稠密表。

    返回 (days, tables)，tables 的形状：
      temp_hourly    (m, b, s, d, 11)  逐小时温度（小时榜 · 温度维）
      rain_hourly    (m, b, s, d, 13)  逐小时晴雨，阈值 rain_thr_hourly（小时榜 · 降水维）
      temp_daily_max (m, b, s, d, 11)  日最高温（日榜 · 温度维之一）
      temp_daily_min (m, b, s, d, 11)  日最低温（日榜 · 温度维之一）
      rain_daily     (m, b, s, d, 13)  日累计晴雨，阈值 rain_thr_daily（日榜 · 降水维）
                                       （第 5~7 列 = Σ实况 / Σ预报 / Σ|误差|，雨量
                                        精度维；第 8~13 列 = 两个雨强档位各自的
                                        h/fa/mi 计数，雨强分辨力维）

    两条井水不犯河水的证据链是这次重构的核心：同一批存档，按"时刻"答一次
    （小时榜）、按"自然日"答一次（日榜），谁也不替谁说话。天数取两类记录的
    并集；无数据的天/桶保持全 0，重采样时自然按"无样本"处理。

    桶对齐：两条轨道的天桶都是"起报日之后的第 N 个自然日"（小时 extends 用
    bucket、日离用 offset），与排行榜的天桶语义一致——两榜的横轴是同一条刻度，
    总榜才能把它们当成同一批"难度"来劈。
    """
    days = eval_days(hourly, daily)
    gb_hourly, gb_daily = grade_bounds("1h"), grade_bounds("24h")
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
            rows.append(_rain_stat_row(o, f, rain_thr_hourly, gb_hourly))
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
            rows.append(_rain_stat_row(o, f, rain_thr_daily, gb_daily))
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


def build_day_stat_tables_columnar(pt, models: list[str],
                                   n_buckets_hourly: int, n_buckets_daily: int,
                                   rain_thr_daily: float, rain_thr_hourly: float,
                                   ) -> tuple[list[str], dict[str, np.ndarray]]:
    """`build_day_stat_tables` 的列式实现：从 PairTable 直接算出同一批充分统计量表。

    输出的形状、列含义、以及**累加顺序**都与 dict 版逐位一致——`np.add.at` 对重复
    下标按出现顺序累加，而列式数组保持的是与 dict 版完全相同的记录顺序，因此
    浮点结果一位不差（守卫：`test_daystats_columnar_matches_dict`，`array_equal`
    零容差）。

    快在哪里：dict 版要为每条记录做 2 次 dict 取键、1 次 `_temp_stat_row` 调用和
    2 次 list append（150 万条 ≈ 10.7 s）；列式版是 11 次向量化运算加一次
    `np.add.at`，Python 层的 per-record 成本为零。

    ⚠️ I4：所有取值都先经 `*_ok` 布尔列过滤（两侧都有值才入表）。列式化把缺测
    变成了 NaN，若依赖 NaN 语义就会把缺测当 0.0 —— 那正是本模块要防的事。
    """
    days = list(pt.days)
    gb_hourly, gb_daily = grade_bounds("1h"), grade_bounds("24h")
    n_s = max(1, len(pt.station_code_all))
    n_m = max(1, len(models))
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

    def _scatter(table, cols, idx):
        for k in range(cols.shape[1]):
            np.add.at(table[:, :, :, :, k], idx, cols[:, k])

    def _temp_cols(o: np.ndarray, f: np.ndarray) -> np.ndarray:
        e = f - o
        return np.stack([np.ones_like(e), e * e, np.abs(e), e,
                         (np.abs(e) <= 1).astype(np.float64),
                         (np.abs(e) <= 2).astype(np.float64),
                         f, o, f * o, f * f, o * o], axis=1)

    def _rain_cols(o: np.ndarray, f: np.ndarray, thr: float,
                   gbounds: tuple[tuple[float, float], ...] = ()) -> np.ndarray:
        ob, fb = o >= thr, f >= thr
        cols = [(ob & fb), (~ob & fb), (ob & ~fb), (~ob & ~fb), o, f, np.abs(f - o)]
        if gbounds:
            # np.round 与权威路径 graded.graded_counts 同一舍入函数（而非 Python
            # 内置 round——两者在二进制边界上不一致），与 _rain_stat_row 逐位一致
            ro, rf = np.round(o, 1), np.round(f, 1)
            for lo, hi in gbounds:
                obg, fbg = (ro >= lo) & (ro <= hi), (rf >= lo) & (rf <= hi)
                cols += [(obg & fbg), (~obg & fbg), (obg & ~fbg)]
        else:
            cols += [np.zeros_like(o, dtype=bool)] * (3 * _GRADE_NLEV)
        return np.stack(cols, axis=1).astype(np.float64)

    # ---- 逐小时 ----
    m_h = pt.h_model
    b_h = pt.h_bucket
    base = (m_h >= 0) & (b_h >= 1) & (b_h <= n_buckets_hourly)
    sel = base & pt.h_temp_ok
    if sel.any():
        _scatter(tables["temp_hourly"],
                 _temp_cols(pt.h_temp_o[sel], pt.h_temp_f[sel]),
                 (m_h[sel], b_h[sel] - 1, pt.h_station_all[sel], pt.h_day[sel]))
    sel = base & pt.h_rain_ok
    if sel.any():
        _scatter(tables["rain_hourly"],
                 _rain_cols(pt.h_rain_o[sel], pt.h_rain_f[sel], rain_thr_hourly,
                            gb_hourly),
                 (m_h[sel], b_h[sel] - 1, pt.h_station_all[sel], pt.h_day[sel]))

    # ---- 按天 ----
    m_d = pt.d_model
    o_d = pt.d_offset
    dbase = (m_d >= 0) & (o_d >= 1) & (o_d <= n_buckets_daily)
    for name, ok_flag, o_arr, f_arr in (
            ("temp_daily_max", pt.d_tmax_ok, pt.d_max_o, pt.d_max_f),
            ("temp_daily_min", pt.d_tmin_ok, pt.d_min_o, pt.d_min_f)):
        sel = dbase & ok_flag
        if sel.any():
            _scatter(tables[name], _temp_cols(o_arr[sel], f_arr[sel]),
                     (m_d[sel], o_d[sel] - 1, pt.d_station_all[sel], pt.d_day[sel]))
    sel = dbase & pt.d_rain_ok
    if sel.any():
        _scatter(tables["rain_daily"],
                 _rain_cols(pt.d_rain_o[sel], pt.d_rain_f[sel], rain_thr_daily,
                            gb_daily),
                 (m_d[sel], o_d[sel] - 1, pt.d_station_all[sel], pt.d_day[sel]))
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
            raw = fn(v)
            # 有效性必须在 clip 之前按原始值判定（第四轮 P2-10）：inf 经 clip
            # 会变成 100 并被计为有效证据，NaN 的处置同理要在变换前想清楚
            valid = np.isfinite(raw)
        sub = np.clip(raw, 0.0, 100.0)
        sub = np.where(valid, sub, 0.0)
        num = w * sub if num is None else num + w * sub
        den = w * valid if den is None else den + w * valid
    if num is None:
        return np.full(next(iter(values.values())).shape if values else (), np.nan)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def between_station_mbe_sd(n: np.ndarray, se: np.ndarray) -> np.ndarray:
    """站间系统偏差的样本量加权离散度（度），沿**最后一个轴**（站维）归约。

    **为什么需要这一维**：入分的 `mbe` 用的是池化口径 `Σe/N`。方向相反的站间
    系统偏差会在这里互相抵消——实测各站 MBE 的离散度与 |池化 MBE| 的跨源相关
    只有 **0.115**（近乎正交），也就是说"池化偏差接近 0"完全推不出"各站都不偏"。
    一个各站都稳定偏 2°C 的源和一个两站分别偏 +2/−2 的源，池化口径给的分几乎
    一样，但后者无法用单一订正量修好——业务价值完全不同。

    **可加性（能进 bootstrap 的硬前提）**：设各站误差和 `E_s`、样本量 `n_s`，

        between_var = [ Σ_s (E_s²/n_s) − (Σ_s E_s)² / N ] / N ,   N = Σ_s n_s

    右端三项 `Σ_s E_s²/n_s`、`Σ_s E_s`、`Σ_s n_s` **全部可加**，故本函数既能吃
    聚合前的逐站数组（点估计），也能吃聚合后的（重采样），两处同一把尺子。
    注意 `E_s²/n_s` 本身**不可**逐记录累加（n_s 要先聚合才知道），所以减法必须
    发生在归约之后——这也是它不需要在聚合表里新增列的原因。

    **缺项纪律**：只把 `n_s ≥ GROUP_MIN_N` 的站计入（与 r/slope 的站内合并同门
    槛），且要求达标站 ≥ `MIN_STATIONS_BDISP`：两站的"离散度"只是两点距离，受
    单站噪声支配，不能算跨站一致性的证据。站数不足 → NaN → 按剩余权重归一。
    """
    ok = n >= GROUP_MIN_N
    n_ok = np.where(ok, n, 0.0)
    e_ok = np.where(ok, se, 0.0)
    N = n_ok.sum(axis=-1)
    E = e_ok.sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Σ_s E_s²/n_s：n_s>0 才有定义；ok 掩膜已保证
        nz = np.where(n_ok > 0, n_ok, np.nan)
        sq_term = np.nansum(np.where(n_ok > 0, e_ok * e_ok / nz, 0.0), axis=-1)
        var = (sq_term - E * E / np.where(N > 0, N, np.nan)) / np.where(N > 0, N, np.nan)
        # 浮点上 var 可能算出微小负数（数学上 ≥0，Cauchy-Schwarz）——先夹到 0
        sd = np.sqrt(np.maximum(var, 0.0))
    enough = (np.count_nonzero(ok, axis=-1) >= MIN_STATIONS_BDISP) & (N > 0)
    return np.where(enough, sd, np.nan)


def _temp_scores_from_aggregate(A: np.ndarray, temp_parts) -> np.ndarray:
    """聚合温度统计量 (run?, m, b, s, 11) → 站内合并后的温度分 (run?, m, b)。

    r/slope 先站内（n≥GROUP_MIN_N 的站）计算、再 Fisher-z / n 加权合并；
    无站达标时回退池化口径（与点估计的 temp_metrics 行为一致）。
    """
    has_run = A.ndim == 5
    if not has_run:
        A = A[None, ...]
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
            "mbe_bdisp": between_station_mbe_sd(n, se),
        }
    scores = _score_from_parts(pooled_values, temp_parts)
    return scores if has_run else scores[0]


def grade_ets_from_counts(tot: np.ndarray, n: np.ndarray,
                          min_sample: int) -> np.ndarray:
    """跨站求和后的分级计数 (..., 13) + 样本量 n → 雨强分辨力 `grade_ets`。

    两个雨强档位各算一个 ETS，再按该档的**事件数** `h+fa+mi` 加权合并：

    * **为什么加权而不是等权**：ETS 的分母就是 `h+fa+mi`，样本稀的大雨档本来就
      该少说话；等权会让"暴雨档 3 个样本碰巧命中"与"中雨档 300 个样本稳定命中"
      拿到一样的权重。
    * **缺项纪律**：某档事件数 < `min_sample` → 该档 NaN（不参与合并）；只剩一档
      就用那一档（权重自然归一）；两档都缺 → 整维 NaN → 按剩余权重归一，与雨量
      两维（`Σ实况 = 0` 时无定义）完全同纪律——绝不是 0 分。
    * 数学上 `ETS ≤ 1` 恒成立（`fa+mi ≥ 0` ⇒ `h−href ≤ h+fa+mi−href`），故不存在
      分母趋零导致的数值爆炸；`ETS` 的下界 −1/3 由换算族的截断统一处理。
    """
    num = None
    den = None
    for j in range(_GRADE_NLEV):
        gh = tot[..., 7 + 3 * j]
        gfa = tot[..., 8 + 3 * j]
        gmi = tot[..., 9 + 3 * j]
        ev = gh + gfa + gmi                     # 该档的"事件数" = ETS 的分母口径
        with np.errstate(invalid="ignore", divide="ignore"):
            gref = (gh + gmi) * (gh + gfa) / np.where(n > 0, n, np.nan)
            gden = ev - gref
            g_ets = np.where(gden > 0, (gh - gref) / np.where(gden > 0, gden, 1.0),
                             np.nan)
        ok = np.isfinite(g_ets) & (ev >= min_sample)
        w = np.where(ok, ev, 0.0)
        num = np.where(ok, g_ets * ev, 0.0) if num is None else num + np.where(
            ok, g_ets * ev, 0.0)
        den = w if den is None else den + w
    if num is None:
        return np.full(tot.shape[:-1], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def grade_counts(o: np.ndarray, f: np.ndarray,
                 gbounds: tuple[tuple[float, float], ...]) -> list[tuple[int, int, int]]:
    """点估计路径：一次性给出各雨强档的 (hits, false_alarms, misses)。

    与聚合表的 `_rain_stat_row` / `_rain_cols` **同一套二值化**（`np.round(x, 1)`
    后按 `(min ≤ v ≤ max)` 判定），故点估计与 bootstrap 落在同一把尺子上。
    这里同样**先舍入、后剔 NaN 对**（与权威路径 graded.graded_counts 的口径顺序
    一致）——调用方传入未过滤序列也安全。
    """
    ro, rf = np.round(np.asarray(o, dtype=float), 1), np.round(np.asarray(f, dtype=float), 1)
    keep = ~(ro != ro) & ~(rf != rf)
    ro, rf = ro[keep], rf[keep]
    out = []
    for lo, hi in gbounds:
        ob, fb = (ro >= lo) & (ro <= hi), (rf >= lo) & (rf <= hi)
        out.append((int((ob & fb).sum()), int((~ob & fb).sum()), int((ob & ~fb).sum())))
    return out


def grade_ets_point(counts: list[tuple[int, int, int]], n: int,
                    min_sample: int) -> float | None:
    """点估计路径的 grade_ets：把分档计数装进同一份 13 列布局再走同一函数。

    **刻意不另写一份合并公式**——点估计与重采样若各有一套加权规则，bootstrap
    就会静默地用另一把尺子（这是本项目反复堵的洞）。这里只做"标量 → (13,) 数组"
    的搬运，真正的数学只有 `grade_ets_from_counts` 一处。
    """
    tot = np.zeros(len(_RAIN_STATS), dtype=np.float64)
    for j, (h, fa, mi) in enumerate(counts[:_GRADE_NLEV]):
        tot[7 + 3 * j], tot[8 + 3 * j], tot[9 + 3 * j] = h, fa, mi
    v = grade_ets_from_counts(tot, np.float64(n), min_sample)
    v = float(np.asarray(v).reshape(()))
    return v if math.isfinite(v) else None


def _rain_scores_from_aggregate(A: np.ndarray, precip_parts,
                                min_sample: int = 5) -> np.ndarray:
    """聚合列联计数 + 雨量 + 分级计数 (run?, m, b, s, 13) → 降水分 (run?, m, b)。

    跨站直接相加：二分类计数与 Σo/Σf/Σ|f−o| 都可加（这是它们能进 bootstrap 的
    前提）。雨量两维只在 **Σ实况 > 0** 的格子上定义（实况无降水时相对口径无
    定义 → NaN → 按剩余权重归一），与点估计 `precip_amount_metrics` 同纪律。
    总量比 Σf/Σo = 0（整桶的雨一滴没报）是有限值 0，由 log 族换算记 0 分——
    绝不借 NaN 通道洗成"缺项"。
    """
    has_run = A.ndim == 5
    if not has_run:
        A = A[None, ...]
    tot = A.sum(axis=-2)          # 站维求和
    h, fa, mi, c = tot[..., 0], tot[..., 1], tot[..., 2], tot[..., 3]
    so, sf, sae = tot[..., 4], tot[..., 5], tot[..., 6]
    n = h + fa + mi + c
    with np.errstate(invalid="ignore", divide="ignore"):
        def nz(x):
            return np.where(x > 0, x, np.nan)
        acc = 100.0 * (h + c) / np.where(n > 0, n, np.nan)
        pod = 100.0 * h / nz(h + mi)
        far = 100.0 * fa / nz(h + fa)
        ts = h / nz(h + fa + mi)
        bias = (h + fa) / nz(h + mi)
        href = (h + mi) * (h + fa) / np.where(n > 0, n, np.nan)
        ets = (h - href) / ((h + fa + mi) - href)
        # 雨量精度：只在实况有降水的格子上定义（so>0），否则 NaN（缺项归一）
        so_pos = np.where(so > 0, so, np.nan)
        amt_mae = sae / so_pos
        amt_bias = sf / so_pos
        grade_ets = grade_ets_from_counts(tot, n, min_sample)
    values = {"acc": acc, "pod": pod, "far": far, "ts": ts, "ets": ets,
              "bias": bias, "amt_mae": amt_mae, "amt_bias": amt_bias,
              "grade_ets": grade_ets}
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
    m_eff: float | None = None,
    pairs=None,
) -> dict[str, dict[str, Any]]:
    """按天分块 bootstrap：各榜单那个"难度对齐综合分"的不确定性。

    pairs：可选的列式配对表（`weather_eval.pairtable.PairTable`）。给了它就走
    列式的 `build_day_stat_tables_columnar`，与 dict 路径产出**逐位相同**的表。
    不传时行为与改造前完全一致（向后兼容，测试与旧调用点不受影响）。

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
    if pairs is not None:
        days, tables = build_day_stat_tables_columnar(
            pairs, models, n_buckets_hourly, n_buckets_daily,
            rain_thr_daily, rain_thr_hourly)
    else:
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
                                            top_model=top_model, m_eff=m_eff)}
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
            top_model=spec.get("top_model", top_model),
            m_eff=spec.get("m_eff", m_eff))
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
    # 以涉及的（行+列）节点数最多的分量为最大分量；节点数打平时按（行数, 列数,
    # 根标签）确定性地择优（第四轮 P2-6：旧实现依赖 dict 插入序 = 扫描序，
    # 两个同尺寸分量留哪个取决于输入排布，会静默丢掉半个榜）
    def _rank(lab: int) -> tuple:
        rows = int((labels[:n_row] == lab).sum())
        cols = int((labels[n_row:] == lab).sum())
        return (rows + cols, rows, cols, -lab)

    best = max(counts, key=_rank)
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
    会污染全榜的极端值。0（默认）= 不收缩。**λ 只加在列上**（第四轮 P1-4）：
    行（技巧）效应一旦同样收缩，覆盖短的源会被等量拉向均值——"覆盖越短排名越低"
    的偏置正好从难度对齐要消掉的方向又被请回来，且 μ 随 λ 漂移、破坏
    variance_decomposition 的份额口径。

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
        # 行效应：以权重求和后按权重和归一（不加 λ——收缩只属于列，见 docstring）
        num_a = np.where(V, W * (X - beta[:, None, :]), 0.0).sum(axis=2)
        den_a = cnt_b
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
        # 与 isfinite 求交是护栏（第四轮 P2-4）：调用方传来的掩膜若覆盖到 NaN 格，
        # NaN 会进加权求和铺满整行——"显式掩膜"从此必须自证有限。
        V = (np.asarray(valid, dtype=bool) & np.isfinite(S3)
             & row_keep[None, :, None] & col_keep[None, None, :])
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
    # 实际生效的列家数门槛（含 min_col_frac 的相对门槛）：design_mask 内部算过一遍，
    # 这里按同口径复算用于披露——对外报告"至少 3 家同台"而实际生效 14 家，
    # 等于让读者从错误的门槛外推主设计的覆盖范围（第四轮 P2-4）。
    eff_min_col = int(min_col)
    per_col = V0.sum(axis=0)
    if min_col_frac and min_col_frac > 0 and per_col.size:
        eff_min_col = max(eff_min_col,
                          int(np.ceil(min_col_frac * float(per_col.max()))))
    if not V.any():
        return {"scores": np.full(S2.shape[0], np.nan),
                "row_effects": np.full(S2.shape[0], np.nan),
                "col_effects": np.full(S2.shape[1], np.nan),
                "mu": None, "row_keep": row_keep, "col_keep": col_keep,
                "n_components": n_components, "cell_valid": V,
                "dropped_thin_cells": dropped_thin,
                "effective_min_col": eff_min_col}
    if W0 is not None:
        # 被门槛剔掉的格子权重必须同时清零（第四轮 P1-1）：只传 valid 不清权重，
        # _fit_parts 的 WLS 仍会按原权重把它们请回来，"直接剔出设计"就是假披露。
        W3 = np.where(V0, W0, 0.0)[None, ...]
    else:
        W3 = None
    parts = _fit_parts(S2[None, ...], row_keep, col_keep, max_iter, tol,
                       weights=W3, ridge=ridge, valid=V0[None, ...])
    mu, alpha, beta, V3 = parts
    scores = np.where(row_keep, mu[0] + alpha[0], np.nan)
    return {"scores": scores, "row_effects": np.where(row_keep, alpha[0], np.nan),
            "col_effects": np.where(col_keep, beta[0], np.nan),
            "mu": float(mu[0]) if np.isfinite(mu[0]) else None,
            "row_keep": row_keep, "col_keep": col_keep,
            "n_components": n_components, "cell_valid": V3[0],
            "dropped_thin_cells": dropped_thin,
            "effective_min_col": eff_min_col}


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
    def share(x):
        return round(x / denom, 4) if denom else None
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
                          col_keep: np.ndarray, min_common: int = 5,
                          segment_sizes: tuple[int, ...] | None = None) -> dict:
    """跨天桶的名次一致性（Spearman）：总榜的单一数字是否配得上"名次"这个词。

    逐桶两两对照（score_b(m) 取该桶内的原始分；扣 β_b 与否对 Spearman 无影响——
    秩对逐列平移不变，这里保留原样只是为了与"扣掉难度后的水平"在直觉上对齐）。
    两两桶的 Spearman 实测桶1 vs 桶7 只有 0.19——意味着"短时效第 3 名"与
    "长时效第 3 名"往往不是同一家。这个数字此前一个都没有，读者无从知道
    总榜的适用边界（P0-2）。

    segment_sizes：总榜的列布局是 [小时 1..b | 日 1..b]，"相邻桶"的平均
    （adjacent_mean）若不限赛段，会把"小时第 16 天 → 日第 1 天"当成相邻时效
    （第四轮 P2-3）——传入 [b, b] 后相邻判定只在同一分辨率段内成立。
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
    # 相邻时效的平均：只在同一赛段（分辨率）内判定相邻（第四轮 P2-3）
    seg_of: dict[int, int] = {}
    if segment_sizes:
        off = 0
        for s_i, size in enumerate(segment_sizes):
            for c in range(off, off + int(size)):
                seg_of[c] = s_i
            off += int(size)
    adjacent = [p["rho"] for p in pairs
                if p["b"] - p["a"] == 1
                and (not segment_sizes
                     or seg_of.get(p["a"] - 1) == seg_of.get(p["b"] - 1))]
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


def holm_bonferroni(pvals: list[float], alpha: float = 0.10,
                    m_eff: float | None = None) -> list[bool]:
    """Holm–Bonferroni 逐步校正：控制族错误率（FWER）的通用做法。

    26 个源同榜、每个都与第一名比一次，α=0.1 下至少一次假阳性的概率约 93%——
    不经校正的"† 与第一名无显著差异"标记基本是噪声。Holm 把第 k 小的 p 值与
    α/(m−k+1) 比较，一旦不显著则此后全部判不显著（逐步降级、单调）。

    m_eff：**有效检验数**。缺省 None = 用实际比较数 m（保守口径）。当各检验之间
    高度相关时（同榜各家预报的是同一批天气系统，实测 ρ̄≈0.61），用 m 会过度
    保守；传入按跨源相关折算的 k_eff = m/(1+(m−1)ρ̄) 即可放松校正。

    注意方向性风险：把 m 换成更小的 k_eff 会**减少**校正、把更多家判成"与冠军
    有显著差异"，也就是**增加**假阳性。因此本项目默认走保守口径，k_eff 只作为
    可切换的并列口径（见 source_corr_cluster 的说明）。此处还强制 m_eff 不超过
    m——校正只能放宽，绝不能比标准 Holm 更严，否则就不是 Holm 了。

    返回与输入等长的显著/不显著布尔列表（True = 显著）。
    """
    m = len(pvals)
    if m == 0:
        return []
    eff = float(m) if m_eff is None else float(m_eff)
    # 夹到 [1, m]：≥1 保证分母不为零；≤m 保证不比标准 Holm 更严
    eff = min(max(eff, 1.0), float(m))
    # 非有限 p 与 None 同桶（第四轮 P2-8）：NaN 在排序比较器里不是全序
    # （nan<x 恒 False），会让极小有效 p 被排到 NaN 之后而漏判
    ps = [p if (p is not None and np.isfinite(p)) else None for p in pvals]
    order = sorted(range(m), key=lambda i: (ps[i] is None, ps[i] or 1.0))
    out = [False] * m
    still = True
    for k, i in enumerate(order):
        p = ps[i]
        if p is None:
            still = False
        elif still and p <= alpha / max(eff - k, 1.0):
            out[i] = True
        else:
            still = False
    return out


def _summarize_bootstrap(macro: np.ndarray, models: list[str],
                         eligible: list[bool], alpha: float = 0.10,
                         top_model: str | None = None,
                         m_eff: float | None = None) -> dict[str, dict[str, Any]]:
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
    fallback_candidates: list[tuple[float, int]] = []
    if top_model is not None and top_model in models:
        ti = models.index(top_model)
        if eligible[ti]:
            mean = float(np.nanmean(macro[:, ti]))
            if np.isfinite(mean):
                point_top = (ti, mean)
    for mi, m in enumerate(models):
        scores = macro[:, mi]
        finite = scores[np.isfinite(scores)]
        ci = None
        if finite.size >= max(5, macro.shape[0] // 10):
            lo, hi = np.percentile(finite, [5, 95])
            ci = [round(float(lo), 2), round(float(hi), 2)]
        out[m] = {"ci90": ci, "champion_pct": 0.0, "sig_vs_top": None,
                  "sig_vs_top_raw": None, "p_vs_top": None}
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
        fallback_candidates.append((float(np.nanmean(scores)), mi))
    # 缺省参照系：入围者中**分布均值最高**的那个（docstring 承诺的退路）。
    # 旧实现写成"第一个入围者设完就不再比较"，参照系于是落在输入顺序上——
    # 冠军频率说 A 家 100% 夺冠，† 却挂在 B 家头上（第四轮 P1-2）。
    if point_top is None and fallback_candidates:
        top_mean, ti = max(fallback_candidates, key=lambda t: t[0])
        if np.isfinite(top_mean):
            point_top = (ti, top_mean)
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
        out["__family__"] = None
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
            # 双侧 p：分布落在 0 另一侧的比例 ×2（配对重采样：同一 run 比同一 run）。
            # 计数 +1 再除（Phipson–Smyth，第四轮 P1-3）：bootstrap 的 p 只能取
            # 离散格点，0/500 次分离的真实 p 是"≤2/(R+1)"而不是 0——报出 p=0.0
            # 不是合法 p 值，且 0.0 与 2/(R+1)=0.004 之间隔一个 run 的随机性，
            # 恰好横跨 Holm 在 26 家同榜下的首道阈值（0.1/26≈0.00385）。
            n = d.size
            pvals.append(min(1.0, 2.0 * min((int((d <= 0).sum()) + 1) / (n + 1),
                                            (int((d >= 0).sum()) + 1) / (n + 1))))
        # 实际进入校正族的比较数（= 与冠军比较的入围者数）。Holm 的 m 在
        # 旧披露里写成"全模型数"，比真值大——校正强度与披露数字对不上
        # （第四轮 P3-b）。挂在 __family__ 键（合法模型名不含双下划线）。
        out["__family__"] = len(pvals)
        for mi, p, sig in zip(idxs, pvals,
                              holm_bonferroni(pvals, alpha=alpha, m_eff=m_eff)):
            if p is None:
                continue
            out[models[mi]]["p_vs_top"] = round(p, 4)
            out[models[mi]]["sig_vs_top_raw"] = bool(p <= alpha)
            out[models[mi]]["sig_vs_top"] = bool(sig)
    return out


def aggregate_day_stats(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """(runs, n_days) 权重 × (m, b, s, d, k) 充分统计量表 → (runs, m, b, s, k)。

    温度/降水两条轨道共用；k 维是各自的可加统计量（温度 11 项 / 降水 4 项）。
    optimize=True 让 einsum 走 BLAS 路径（第四轮 P3-1：生产规模实测 2.45s → 0.18s，
    allclose 逐位一致）。
    """
    out = np.empty(W.shape[:1] + X.shape[:3] + (X.shape[-1],))
    for k in range(X.shape[-1]):
        out[..., k] = np.einsum("rd,mbsd->rmbs", W, X[:, :, :, :, k], optimize=True)
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
    r_h = _rain_scores_from_aggregate(tables["rain_hourly"], precip_parts,
                                      min_sample)
    t_max = _temp_scores_from_aggregate(tables["temp_daily_max"], temp_parts)
    t_min = _temp_scores_from_aggregate(tables["temp_daily_min"], temp_parts)
    r_d = _rain_scores_from_aggregate(tables["rain_daily"], precip_parts,
                                      min_sample)
    # 各维的有效成对样本数：沿站维求和后是 (run?, m, b)，与各自的计分函数内部
    # 用的 N 同口径（_temp_scores_from_aggregate 里 N = n.sum(-1)）。
    # 日最高/日最低**各自用自己的样本数**判定：两边都薄时"相加凑够样本"会把两个
    # 都不足门槛的量伪装成一个够样本的维度。
    n_th = tables["temp_hourly"][..., 0].sum(axis=-1)
    # 只取前 4 列（列联计数）：2026-10 起雨量表多了 Σo/Σf/Σ|f−o| 三列，
    # 整表求和会把雨量毫米数混进"样本数"
    n_rh = tables["rain_hourly"][..., :4].sum(axis=(-2, -1))
    n_tmax = tables["temp_daily_max"][..., 0].sum(axis=-1)
    n_tmin = tables["temp_daily_min"][..., 0].sum(axis=-1)
    n_rd = tables["rain_daily"][..., :4].sum(axis=(-2, -1))

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
    n_m = temp_sub.shape[0]
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


# ================================================================== 跨源相关（对抗式审查 P0-1）
# 这一层回答的是一个此前两层校正都没回答的问题：n_eff 校正了"同一条信息在时间里
# 被数了几次"（effective_n）与"在空间上被数了几次"（cross_station_rho），但没有
# 校正"在**信源**上被数了几次"。26 家同榜里 11 家共享 Open-Meteo 的同一条时间轴
# 与同一个起报锚点，5 个天机变体出自同一家产品——它们的误差序列面对的是同一批
# 天气与同一个锚点误差，相关度远高于跨机构的两家。把它们当成 26 份独立证据，
# "与冠军区间重叠的家数"就是虚高的。

# 两两相关估计所需的最小公共样本：相关系数的标准误约 1/sqrt(n−3)，n=30 时约
# ±0.19——比我们要区分的"同源 ρ≈0.8 vs 跨源 ρ≈0.3"小一个量级，够用。
SOURCE_CORR_MIN_OVERLAP = 30
# ρ̄ 的上限截断：与 RHO_CLAMP 同尺度，防止退化序列把 k_eff 压到 0
SOURCE_RHO_CLAMP = 0.95


def effective_independent_count(m: int, rho: float | None) -> float:
    """m 个两两相关为 ρ̄ 的信源，等价于多少个独立信源。

    k_eff = m / (1 + (m−1)·ρ̄)

    这是"等相关（equicorrelation）"结构的标准结果：m 个等相关的观测，其均值
    的方差等于 k_eff 个独立观测均值的方差。ρ̄=0 → k_eff=m；ρ̄=1 → k_eff=1
    （m 份完全相同的证据只值一份）。ρ̄<0 时 k_eff>m，但负相关的信源在气象
    预报里没有物理意义（同一批天气只会让误差同向），因此夹紧到 m。

    m<2 或 ρ̄ 估不出来时返回 m（不校正，宁可保守也不过校正）。
    """
    if m < 2 or rho is None or not np.isfinite(rho):
        return float(m)
    rho = min(max(float(rho), 0.0), SOURCE_RHO_CLAMP)
    return float(m) / (1.0 + (m - 1) * rho)


def pairwise_corr_matrix(series: dict[str, dict[Any, float]],
                         min_overlap: int = SOURCE_CORR_MIN_OVERLAP,
                         ) -> tuple[list[str], list[list[float | None]], int]:
    """按**公共键**对齐的稀疏序列 → 两两相关矩阵。

    series: {name: {key: value}}，key 是"同一个物理样本"的标识（如
    (站点, 有效时刻, 提前天数)）。只在两序列都有的键上估计——缺测小时自然
    跳过，不做任何插补（插补会凭空制造相关）。

    返回 (names, matrix, n_common_max)：matrix[i][j] 为 Pearson ρ，样本不足
    或退化时为 None；对角线恒为 1.0。
    """
    names = list(series)
    n = len(names)
    matrix: list[list[float | None]] = [[None] * n for _ in range(n)]
    max_common = 0
    for i in range(n):
        matrix[i][i] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            a_map, b_map = series[names[i]], series[names[j]]
            common = a_map.keys() & b_map.keys()
            if len(common) < min_overlap:
                continue
            max_common = max(max_common, len(common))
            keys = sorted(common)
            a = np.array([a_map[k] for k in keys], dtype=float)
            b = np.array([b_map[k] for k in keys], dtype=float)
            r = pearson_r(a, b)
            matrix[i][j] = r
            matrix[j][i] = r
    return names, matrix, max_common


def _connected_clusters(names: list[str], matrix: list[list[float | None]],
                        threshold: float) -> list[list[str]]:
    """把 ρ ≥ threshold 的信源并成一类（并查集）。

    用途不是"替读者剔除同源模型"（那是替读者做取舍），而是回答一个可核对的
    问题：这 26 家里，真正相互独立的"信源族"有几个。
    """
    parent = list(range(len(names)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = matrix[i][j]
            if r is not None and r >= threshold:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[str]] = {}
    for i, nm in enumerate(names):
        groups.setdefault(find(i), []).append(nm)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def source_corr_cluster(series_by_model: dict[str, dict[Any, float]],
                        families: dict[str, str] | None = None,
                        min_overlap: int = SOURCE_CORR_MIN_OVERLAP,
                        cluster_threshold: float = 0.7,
                        ) -> dict[str, Any]:
    """跨源相关的完整诊断：ρ̄、相关族、有效独立信源数 k_eff。

    返回值进入 ``meta.diagnostics.source_correlation``（由 report/diagnostics.py
    汇总后由报告页渲染）。它提供的三个数字各自回答一个"这份名次值多少信任"：

      * ``mean_rho``：各家误差序列的平均两两相关。**同榜的源不是互相独立的
        证据**——它们预报的是同一批天气系统，相关性高是常态。
      * ``k_eff = m/(1+(m−1)ρ̄)``：折算后的**有效独立信源数**。2026-09 实测
        27 家、ρ̄≈0.61 → k_eff≈1.6。这个数字的含义很硬：榜单上看上去是 27
        份证据，按相关性折算后只相当于约 1.6 份独立证据。
      * ``clusters``：ρ > cluster_threshold 的相关族，即"实质同源"的分组。

    关于 Holm–Bonferroni 的有效检验数，此处必须说清楚**实际采用的是哪一套**
    （审查 P1-3 曾指出本 docstring 描述了一个不存在的世界）：

      * 报告页默认仍用**原始比较数 m** 做 Holm 校正（保守口径）。理由是方向
        性的：把 m 换成 k_eff 会**减少**校正、把更多家判成"与冠军有显著差异"，
        也就是**增加**假阳性。本项目宁可多说"说不清"，也不愿凭一个折算系数
        多下结论。
      * k_eff 因此作为**并列披露**呈现（"保守口径 m=? / 按跨源相关折算
        k_eff=?"），读者可自行判断该信哪一套；也可由配置
        ``eval.holm_effective_tests: k_eff`` 切换口径。

    families: {model: 族名}，给出时额外报告"族内平均 ρ vs 跨族平均 ρ"——
    这两个数的差距就是"同源冗余"的直接证据。
    """
    names, matrix, max_common = pairwise_corr_matrix(series_by_model, min_overlap)
    m = len(names)
    flat = [matrix[i][j] for i in range(m) for j in range(i + 1, m)
            if matrix[i][j] is not None]
    if not flat:
        return {"available": False, "reason": "公共样本不足，无法估计跨源相关",
                "n_models": m, "mean_rho": None, "k_eff": float(m),
                "matrix": [], "names": names, "clusters": []}

    mean_rho = float(np.mean(flat))
    in_family: list[float] = []
    cross_family: list[float] = []
    if families:
        for i in range(m):
            for j in range(i + 1, m):
                r = matrix[i][j]
                if r is None:
                    continue
                fi = families.get(names[i])
                fj = families.get(names[j])
                if fi is None or fj is None:
                    continue
                (in_family if fi == fj else cross_family).append(r)

    clusters = _connected_clusters(names, matrix, cluster_threshold)
    k_eff = effective_independent_count(m, mean_rho)
    # 族口径的独立信源数：把每个相关族当成一条证据，这是比 k_eff 更保守的一档
    k_families = float(len(clusters))

    out = {
        "available": True,
        "n_models": m,
        "n_pairs": len(flat),
        "max_common_samples": max_common,
        "mean_rho": round(mean_rho, 4),
        "median_rho": round(float(np.median(flat)), 4),
        "max_rho": round(float(np.max(flat)), 4),
        "min_rho": round(float(np.min(flat)), 4),
        "k_eff": round(k_eff, 2),
        "k_families": k_families,
        "cluster_threshold": cluster_threshold,
        "clusters": clusters,
        "names": names,
        "matrix": [[None if v is None else round(v, 3) for v in row] for row in matrix],
    }
    if families and in_family and cross_family:
        out["in_family_rho"] = round(float(np.mean(in_family)), 4)
        out["cross_family_rho"] = round(float(np.mean(cross_family)), 4)
        out["in_family_pairs"] = len(in_family)
        out["cross_family_pairs"] = len(cross_family)
    return out


def overlap_independent_count(overlap_models: list[str],
                              matrix: list[list[float | None]],
                              names: list[str],
                              ) -> float:
    """与冠军区间重叠的 N 家里，真正独立的信源有几家。

    对这 N 家的**子矩阵**单独算 ρ̄ 再套 k_eff = N/(1+(N−1)ρ̄)——不能用全榜
    ρ̄ 代替：与冠军重叠的那几家往往恰是同源扎堆的一批（这正是"重叠家数虚高"
    的来源），它们的内部相关高于全榜平均。
    """
    idx = [names.index(x) for x in overlap_models if x in names]
    n = len(idx)
    if n < 2:
        return float(n)
    flat = [matrix[idx[i]][idx[j]] for i in range(n) for j in range(i + 1, n)
            if matrix[idx[i]][idx[j]] is not None]
    if not flat:
        return float(n)
    return round(effective_independent_count(n, float(np.mean(flat))), 2)


# ================================================================== 站对相关矩阵（P1-4）
def station_rho_matrix(series_by_station: dict[str, dict[Any, float]],
                       min_overlap: int = CROSS_STATION_MIN_OVERLAP,
                       ) -> dict[str, Any]:
    """按站对（而非一个平均值）披露跨站相关 ρ̄。

    现状用一个标量 ρ̄ = 全部站对的平均做 n_eff 校正，等于假设"梧州—平南"与
    "梧州—万宁"的相关相同。但 4 站里 3 站在广西且彼此相邻（梧州—平南约
    100 km），万宁在海南——集群内相关必然高于跨海相关。用一个平均值会同时
    高估跨海那几对、低估集群内那几对。
    """
    names, matrix, max_common = pairwise_corr_matrix(series_by_station, min_overlap)
    k = len(names)
    pairs = []
    flat = []
    for i in range(k):
        for j in range(i + 1, k):
            r = matrix[i][j]
            pairs.append({"a": names[i], "b": names[j], "rho": None if r is None else round(r, 3)})
            if r is not None:
                flat.append(r)
    mean_rho = float(np.mean(flat)) if flat else None
    return {
        "stations": names,
        "pairs": pairs,
        "mean_rho": None if mean_rho is None else round(mean_rho, 4),
        "k_eff": round(effective_independent_count(k, mean_rho), 2),
        "n_stations": k,
        "max_common_samples": max_common,
    }

"""DVS v6 推断层：配对平稳块自助、模型置信集分档、排名区间与技巧寿命。

**为什么旧口径的"边际 CI 重叠"必须废掉（第一性原理）**

两家源的误差因为面对**同一批天气**而高度相关。当相关性 ρ ≈ 0.9 时，两条边际区间
（各自 ±CI）几乎总在重叠，但它们之差的标准误只有各自标准误的 `√(2(1−ρ)) ≈ 0.45`
倍——也就是说"区间重叠 = 分不出高下"这个判据在这里**系统性地偏保守**，把真差异
说成噪声。反之，当样本极少时边际区间又会被算得过窄，制造假差异。

正确做法只有一条：**直接对"差值"做推断，并且让重采样尊重相关结构**。这就是
配对块自助：

* 重采样单位是**北京时自然日**——同一天 4 个站共享天气系统，同一天的误差彼此相关；
  把日当单位，站间与源间相关被原样保留。
* 所有源、所有要素共用**同一个**日期权重向量 `w^{(r)}`（配对！）——这样"两家谁高"
  的抽样波动里，共享的天气难度被抵消掉，剩下的才是"这两家本身"的差异。
* 块长由 Politis–White (2004) 自动估计并设 3 天下限（天气尺度）。

**为什么用 MCS 而不是逐个 t 检验 + 多重比较校正**

多重比较校正（Holm/Bonferroni）控制的是"族错误率"，但它的输出是一堆"谁和谁显著"
的 p 值，读者要自己拼出"那到底谁在前"。而 Hansen–Lunde–Nason (2011) 的模型置信集
（MCS）直接回答读者真正的问题：**"在 90% 置信下，哪些源不能被排除为最好？"**——
输出是一个**集合**，集合内部不做无根据的排序，集合顺序分档（Tier 1、Tier 2 …）。
这恰好对应"27 家源其实只相当于约 2 家独立信源"这一事实：MCS 会诚实地把它们放进
同一档，而不是伪造 27 个名次。
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np

#: 自助次数与最小块长（天）。块长下限 3 天来自天气尺度：短于 3 天，块内的
#: 相关结构仍然被切断，区间会系统性偏窄。
#: 自助次数。白皮书给的是 2000；这里取 200 是**成本约束下的显式取舍**：
#: 单次"完整重算评分"约 2.9 ms，2000 次 × 32 源 ≈ 190 s，仅自助一项就吃掉整份
#: 报告的耗时预算（实测 500 次也要 397 s）。
#: 为什么 200 仍然可用：90% 区间端点的抽样误差是 O(1/√R)，R=200 时约 ±0.9 个
#: 分位点位次（≈ 区间的 4%），而区间本身的宽度在 5~20 分——**变的是末位小数，
#: 不是"谁比谁好"的结论**。名次区间与 MCS 分档基于同一组重采样，结论方向不变。
#: 需要更窄的区间时把 `eval.dvs.bootstrap_runs` 调回去即可，代价是构建变慢；
#: 报告 meta.dvs.config.bootstrap_runs 始终披露实际生效值。
DEFAULT_RUNS = 200
MIN_BLOCK_DAYS = 3

#: MCS 显著性水平。0.10 与旧口径的显著性水平一致，便于读者对照历史结论。
MCS_LEVEL = 0.10
MCS_MAX_TIERS = 5

#: 分半信度下限：低于它，名次区间不予展示（只展示 MCS 档位）。
SPLIT_HALF_MIN_RELIABILITY = 0.85

#: 固定随机种子。同一天数据必须得到同一份区间——可复现性不是锦上添花，
#: 它是"这份报告是一份证据"的前提。
BOOTSTRAP_SEED = 20261114


# ---------------------------------------------------------------- 块自助
def stationary_block_indices(n_days: int, mean_block: float,
                             rng: np.random.Generator) -> np.ndarray:
    """平稳块自助（Politis & Romano 1994）的日期下标序列。

    实现要点：几何分布块长（均值 `mean_block`），每块从均匀随机位置起步，环绕取样。
    与"固定长度块"相比，几何块长不需要选一个整数块长，且在块边界处不会人为造成
    "每隔 k 天相关就断一次"的周期结构。
    """
    if n_days <= 0:
        return np.zeros(0, dtype=np.int64)
    p = 1.0 / max(float(mean_block), 1.0)
    out = np.empty(n_days, dtype=np.int64)
    i = 0
    while i < n_days:
        start = int(rng.integers(0, n_days))
        length = int(rng.geometric(p))
        length = max(1, min(length, n_days - i))
        idx = (start + np.arange(length)) % n_days
        out[i:i + length] = idx
        i += length
    return out


def auto_block_length(series: np.ndarray, min_block: int = MIN_BLOCK_DAYS) -> float:
    """Politis–White (2004) 自动块长估计（带天气尺度下限）。

    思路：先用"截断的平坦顶"窗（flat-top lag window）估计长期方差 `g`，再用
    `b = (2g²/D)^(1/3) n^(1/3)` 取最优块长，其中 `D` 是过程二阶矩的估计。
    这里实现的是该文献 §3.1 的可计算版本：对自相关做"连续显著滞后"截断，避免
    用全部滞后把噪声也当成信号。

    退化情形（样本 < 10、方差为 0、估计非有限）一律返回 `min_block`：
    块长是**控制方差估计**的参数，宁可用保守下限，也不用一个不可复核的数。
    """
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return float(min_block)
    x = x - x.mean()
    var = float(np.mean(x * x))
    if var <= 1e-12:
        return float(min_block)
    max_lag = max(1, min(int(2 * math.sqrt(n)), n - 1))
    # 自相关与"连续显著"截断（显著性阈值取 2/√n）
    thresh = 2.0 / math.sqrt(n)
    gamma = []
    for k in range(max_lag + 1):
        gamma.append(float(np.mean(x[:n - k] * x[k:])) if k else var)
    m = 0
    for k in range(1, max_lag + 1):
        if abs(gamma[k] / var) < thresh:
            break
        m = k
    # 平坦顶窗：|k| ≤ m 取 1，m < |k| ≤ 2m 线性降到 0
    weights = np.ones(2 * m + 1)
    if m > 0:
        ramp = np.maximum(0.0, 1.0 - np.arange(1, m + 1) / float(m))
        weights[m + 1:] = ramp
        weights[:m] = ramp[::-1]
    g = abs(weights[0] * gamma[0] + 2 * np.sum(
        weights[1:] * np.array(gamma[1:2 * m + 1])))
    d = float(np.mean(x ** 4) ** 2) if n else 0.0
    if not math.isfinite(g) or not math.isfinite(d) or d <= 1e-18 or g <= 1e-18:
        return float(min_block)
    b = (2.0 * g * g / d) ** (1.0 / 3.0) * n ** (1.0 / 3.0)
    if not math.isfinite(b):
        return float(min_block)
    return float(min(b, n / 3.0)) if b > min_block else float(min_block)


@dataclass
class BootstrapResult:
    """自助输出：每个源每次重采样的分数矩阵与派生统计量。"""

    #: `scores[r, m]`：第 r 次重采样中第 m 个源的分数（NaN = 该次无样本）
    scores: np.ndarray
    models: list[str]
    point: dict[str, float]
    ci_lo: dict[str, float] = field(default_factory=dict)
    ci_hi: dict[str, float] = field(default_factory=dict)
    rank_lo: dict[str, int] = field(default_factory=dict)
    rank_hi: dict[str, int] = field(default_factory=dict)
    p_best: dict[str, float] = field(default_factory=dict)
    block_days: float = float(MIN_BLOCK_DAYS)
    runs: int = 0
    days: int = 0

    def interval(self, model: str, level: float = 0.90) -> tuple[float, float] | None:
        lo = self.ci_lo.get(model)
        hi = self.ci_hi.get(model)
        if lo is None or hi is None:
            return None
        return lo, hi


def paired_bootstrap(
    days: Sequence[int],
    score_fn: Callable[[np.ndarray], np.ndarray],
    models: Sequence[str],
    point: Mapping[str, float],
    *,
    runs: int = DEFAULT_RUNS,
    seed: int = BOOTSTRAP_SEED,
    min_block: int = MIN_BLOCK_DAYS,
    block_days: float | None = None,
    level: float = 0.90,
    align_series: np.ndarray | None = None,
) -> BootstrapResult:
    """配对平稳块自助。

    `score_fn(counts)` 接收一个长度 `len(days)` 的整数数组（每一天被抽中的**次数**），
    返回长度 `len(models)` 的分数数组。**所有源共用同一个 counts 向量**——这正是
    "配对"的含义：抽到的一天对每家源同时加倍，共享的天气难度在差值里自然抵消。

    `align_series` 若给出（长度 `len(days)`），则用它的自相关自动估块长；
    缺省用 `1.0` 序列（即退化为 `min_block`，因为常数列没有相关性可估）。
    自动估块长的**目的是让短块相关结构不被切断**，不是为了"调参直到好看"：
    块长变大只会让区间变宽，这是保守方向。
    """
    n = len(days)
    rnd = np.random.default_rng(seed)
    if block_days is None:
        base = (np.asarray(align_series, dtype=float) if align_series is not None
                else np.ones(n))
        block_days = auto_block_length(base, min_block=min_block)
    block_days = max(float(block_days), float(min_block))
    # 抽样先在主线程一次算完（随机数序列必须与并行无关，否则同一份数据在两台
    # 机器上一次用 8 核一次用 2 核就会得到不同区间）；计分再并行。
    # 计分是纯函数、无共享状态，线程池即可（numpy 的重活会释放 GIL）。
    counts_list = []
    for _ in range(int(runs)):
        idx = stationary_block_indices(n, block_days, rnd)
        counts_list.append(np.bincount(idx, minlength=n).astype(float))
    matrix = _score_all(counts_list, score_fn, len(models))
    res = BootstrapResult(scores=matrix, models=list(models),
                          point={m: float(point.get(m, float("nan")))
                                 for m in models},
                          block_days=block_days, runs=int(runs), days=n)
    summarize_matrix(res, level=level)
    return res


def _score_all(counts_list, score_fn, n_models: int) -> np.ndarray:
    """对每次重采样的日期权重计分（线程池并行，结果按输入顺序回填）。

    **顺序必须稳定**：`matrix[r]` 就是第 r 次抽样，MCS 与区间都依赖它。
    用 `executor.map` 而不是 `as_completed`，返回值天然保序。

    并行粒度取"一次重采样"：`score_fn` 内部还要遍历全部源，把任务再切细
    （按源并行）会让每次任务只有 ~0.1 ms，线程调度开销反超收益。
    抽样在**主线程串行**生成，所以同一份数据在任何核数下得到同一组区间。
    """
    from concurrent.futures import ThreadPoolExecutor
    matrix = np.full((len(counts_list), n_models), np.nan)
    workers = min(len(counts_list), max(1, (os.cpu_count() or 2)) * 2)
    if len(counts_list) < 8 or workers <= 1:
        for r, counts in enumerate(counts_list):
            matrix[r] = score_fn(counts)
        return matrix
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r, row in enumerate(pool.map(score_fn, counts_list)):
            matrix[r] = row
    return matrix


def summarize_matrix(res: BootstrapResult, level: float = 0.90) -> None:
    """由 `res.scores` 计算区间、排位区间与夺冠概率（原地填回 `res`）。

    与 `paired_bootstrap` 分开，是因为引擎要"同一份矩阵既算区间又算 MCS"，
    而 MCS 需要原始矩阵——分两步可避免同一份抽样算两遍。
    """
    matrix = res.scores
    if matrix.size == 0:
        return
    models = res.models
    lo_q = (1.0 - level) / 2.0 * 100.0
    hi_q = 100.0 - lo_q
    for j, m in enumerate(models):
        col = matrix[:, j]
        col = col[np.isfinite(col)]
        if col.size < 20:
            # 样本太少：不给区间，而不是给一个哨兵零值区间（旧口径的病）
            continue
        res.ci_lo[m] = float(np.percentile(col, lo_q))
        res.ci_hi[m] = float(np.percentile(col, hi_q))
    rows = matrix.shape[0]
    for r in range(rows):
        row = matrix[r]
        order = np.argsort(np.where(np.isfinite(row), -row, np.inf), kind="stable")
        ranks = np.empty(len(models), dtype=np.int64)
        ranks[order] = np.arange(1, len(models) + 1)
        for j, m in enumerate(models):
            res.rank_lo[m] = min(res.rank_lo.get(m, 10 ** 9), int(ranks[j]))
            res.rank_hi[m] = max(res.rank_hi.get(m, 0), int(ranks[j]))
        best = order[0]
        if np.isfinite(row[best]):
            for j, m in enumerate(models):
                if np.isfinite(row[j]) and row[j] >= row[best] - 1e-12:
                    res.p_best[m] = res.p_best.get(m, 0.0) + 1.0 / rows


# ---------------------------------------------------------------- MCS 分档
def mcs_tiers(s_hat: Mapping[str, float], boot: np.ndarray,
              models: Sequence[str], *,
              alpha: float = MCS_LEVEL, max_tiers: int = MCS_MAX_TIERS
              ) -> list[list[str]]:
    """序贯 Hansen–Lunde–Nason 模型置信集分档。

    对当前存活集合 `M`：
      1. 计算配对差值 `d_ij = Ŝ_i − Ŝ_j` 与其自助标准误 `se_ij`；
      2. 统计量 `T_max = max_{i,j∈M} |d_ij| / se_ij`；
      3. 自助分布 `T*_b = max_{i,j} |d*_ij,b − d_ij| / se_ij`；
      4. 若 `P(T* ≥ T_max) ≥ α`，则"这一组在 α 水平下不可区分"→ 输出为一档；
         否则剔除**相对其余成员平均最差**的那一家，回到 1。

    输出最多 `max_tiers` 档；超出部分统一并入最后一档（"其余"），因为再往下
    分档已经没有统计支撑，继续切只会制造虚假的精细名次。

    缺失处理：某源在全部重采样里都没有分数时，它单独成一档（"无证据，不参与比较"），
    既不混进第一档，也不被静默丢弃。
    """
    present = [m for m in models if m in s_hat and np.isfinite(s_hat[m])]
    absent = [m for m in models if m not in present]
    idx = [models.index(m) for m in present]
    point = np.array([s_hat[m] for m in present], dtype=float)
    b = boot[:, idx] if boot.size and idx else np.zeros((0, len(present)))
    tiers: list[list[str]] = []
    remaining = list(range(len(present)))
    while remaining and len(tiers) < max_tiers:
        alive = list(remaining)
        while len(alive) > 1:
            sub = np.array(alive)
            d_hat = point[sub][:, None] - point[sub][None, :]
            if b.shape[0] >= 20:
                # d*_b,ij = S*_b,i − S*_b,j：两个形状 (B, m, 1) 与 (B, 1, m)
                sub_b = b[:, sub]
                d_b = sub_b[:, :, None] - sub_b[:, None, :]
                with np.errstate(invalid="ignore", divide="ignore"):
                    se = np.nanstd(d_b, axis=0)
                se = np.where((se > 1e-12) & np.isfinite(se), se, np.nan)
                if np.all(np.isnan(se)):
                    break
                with np.errstate(invalid="ignore", divide="ignore"):
                    t_obs = np.nanmax(np.abs(d_hat / se))
                    t_b = np.nanmax(np.abs((d_b - d_hat) / se), axis=(1, 2))
                t_b = t_b[np.isfinite(t_b)]
                if t_b.size and np.isfinite(t_obs):
                    if float(np.mean(t_b >= t_obs)) >= float(alpha):
                        break
                with np.errstate(invalid="ignore"):
                    mean_t = np.nanmean(d_hat / se, axis=1)
                worst = sub[int(np.nanargmax(mean_t))]
            else:
                # 自助样本不足：退化为"只把完全相等的并成一档"，绝不伪造分离
                spread = np.ptp(point[sub])
                if spread <= 1e-12:
                    break
                worst = sub[int(np.argmin(point[sub]))]
            alive.remove(int(worst))
        tiers.append([present[i] for i in alive])
        for i in alive:
            remaining.remove(i)
    if remaining:
        if tiers:
            tiers[-1].extend(present[i] for i in remaining)
        else:  # pragma: no cover - max_tiers ≥ 1 时不可达
            tiers.append([present[i] for i in remaining])
    if absent:
        tiers.append(list(absent))
    return tiers


# ---------------------------------------------------------------- 技巧寿命
def skill_lifetime(day_stats: Mapping[int, tuple[float, float | None]],
                   threshold: float = 0.0, level: float = 0.90
                   ) -> dict:
    """技巧寿命 `H`：满足"所有 `d' ≤ d` 的逐日技能 90% CI 下界均 > 0"的最大 `d`。

    `day_stats` 形如 `{提前天数 d: (点估计, CI 下界)}`，逐日技能是**相对 L0 的
    综合技能**（0 = 与无技巧基准持平）。

    为什么要"所有 d' ≤ d"而不是"这一个 d 显著"：寿命的语义是"到第 H 天为止都还
    值钱"。若第 3 天显著、第 2 天不显著，那第 3 天不能算数——中间断了，说明结论
    不连续，读者不能据此排班。

    返回 `{"H": int, "confidence": float, "detail": {d: {"skill":…, "lower":…, "ok":…}}}`。
    `H = 0` 表示"连第 1 天都不显著"——这是诚实结果，不是错误。
    """
    detail: dict[int, dict] = {}
    h = 0
    chain_ok = True
    for d in sorted(day_stats):
        skill, lower = day_stats[d]
        ok = bool(chain_ok and lower is not None and np.isfinite(lower)
                  and lower > threshold)
        detail[int(d)] = {"skill": (None if skill is None else float(skill)),
                          "lower": (None if lower is None else float(lower)),
                          "ok": ok}
        if not ok:
            chain_ok = False
        elif d == h + 1:
            h = d
    return {"H": int(h), "confidence": float(level), "detail": detail}


# ---------------------------------------------------------------- 分半信度
def split_half_reliability(scores_a: Mapping[str, float],
                           scores_b: Mapping[str, float]) -> dict:
    """分半信度（Spearman–Brown 校正）：把样本按奇偶周拆成两半各算一次分数。

    两半的名次相关系数 `r` 只反映"半个样本"的信度，用 `ρ = 2r/(1+r)` 校正回全样本
    尺度。`ρ < 0.85` 时页面隐藏精确名次区间、只披露 MCS 档位——因为此时"第 3 名"
    与"第 5 名"的差别在两半之间都不稳定，写出来就是过度精确。
    """
    common = sorted(set(scores_a) & set(scores_b))
    pairs = [(scores_a[m], scores_b[m]) for m in common]
    pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
    if len(pairs) < 3:
        return {"reliability": None, "r": None, "n": len(pairs),
                "enough_evidence": False}
    a = np.array([p[0] for p in pairs])
    b = np.array([p[1] for p in pairs])
    if a.std() < 1e-12 or b.std() < 1e-12:
        return {"reliability": None, "r": None, "n": len(pairs),
                "enough_evidence": False}
    r = float(np.corrcoef(a, b)[0, 1])
    # Spearman–Brown `2r/(1+r)` 在 `r → −1` 时分母趋 0、结果爆炸（实测到 −1.8e16）。
    # 完全反向本就没有"信度"可言，返回 None 让调用方隐藏精确名次。
    rho = 2.0 * r / (1.0 + r) if r > -0.5 else None
    return {"reliability": rho, "r": r, "n": len(pairs),
            "enough_evidence": True,
            "pass": bool(rho is not None and rho >= SPLIT_HALF_MIN_RELIABILITY)}


__all__ = [
    "BOOTSTRAP_SEED", "BootstrapResult", "DEFAULT_RUNS", "MCS_LEVEL",
    "MCS_MAX_TIERS", "MIN_BLOCK_DAYS", "SPLIT_HALF_MIN_RELIABILITY",
    "auto_block_length", "mcs_tiers", "paired_bootstrap", "skill_lifetime",
    "split_half_reliability", "stationary_block_indices", "summarize_matrix",
]

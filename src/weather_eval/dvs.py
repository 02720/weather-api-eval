"""DVS v6 度量层：增量决策价值分（Decision-Value Score）的核心算式。

本模块是**纯数学层**：输入是"充分统计量"，输出是分数与分解项。它不知道站、源、
天气，也不知道数据库——所有数据接入在 `dvs_engine.py`，所有不确定性推断在
`dvs_uncertainty.py`。这样切分的目的只有一个：每一条公式都能被单独喂进一个手算
小例子来对拍（见 `tests/test_dvs_axioms.py`）。

──────────────────────────────────────────────────────────────────────────────
第一性原理：什么叫"这家预报值多少钱"？

用户不是要"误差最小"，用户是要**做对决策**。那么"一家源值多少"就有了唯一定义：
在同一个决策问题上，用它做决策比用**免费可得的最佳朴素决策**少损失多少期望代价。

这一定义要求三件事，缺一不可：

1. **0 点必须是"无技巧基准"，不是"同行平均"**。用同行均值当 0 点，分数就随同台
   对手增删而变（旧口径的桶内 z 分），它测的是相对名次，不是价值。
2. **100 点必须是完美预报**，且从 0 到 100 的路径必须是 score 的**期望**损失线性
   映射——否则"差 10 分"在两段区间意义不同，读者无法把分差换算成钱。
3. **评分规则必须 proper**：诚实报出自己相信的分布，期望损失最小。任何"永远报干"
   或"永远报雨"能拿正分的规则，都在教模型作弊，样本一多就必然被利用。

具体实现分两维：

**温度**：`MAE` 技能与 `RMSE` 技能的显式加权和。
    `SS_MAE = 1 − Σ|e| / Σ|e⁰|`，`SS_RMSE = 1 − √Σe² / √Σ(e⁰)²`
    `T_x = max(−1, 0.6·SS_MAE + 0.4·SS_RMSE)`

  * MAE 是确定性预报下 CRPS 的退化形式（对点预报，CRPS ≡ MAE），也是将来接入集合
    /分位数预报的**同一数学接口**；RMSE 在平方意义下对大误差递增惩罚。
  * 两者按 `0.6 : 0.4` 混合是对"损失函数形状"的**显式声明**（线性代价 vs 大错厌恶），
    不是从数据方差反解出来的权重。
  * 分母是**基准自身的误差总和**，所以 `T = 0` 精确等于"和基准一样差"，
    `T = 1` 精确等于"完美"，不需要任何标定斜率。
  * `−1` 只是极端崩溃保护（需披露触发率）：`SS = −1` 意味着误差是基准的两倍。

**降水**：多尺度多阈值 **Richardson (2000) 相对经济价值** `V̄⁺`，对先验代价-损失比
`Beta(2,5)` 积分。为什么不直接用 ETS：
  * ETS 不是 proper 规则，在低基率下天然奖励超报（旧口径因此不得不外挂频率阻尼
    `d(B)` 补丁）；
  * 用户面对的从来不是"ETS 多少"，而是"要不要为这场雨做防护"——一个**二元决策**，
    代价 `C`、灾损 `L`、比值 `α = C/L ∈ (0,1)`。价值泛函就是这个决策的期望损失节约。
  * `max(V, 0)` 的内层截断是**理性用户**的行为：若某家的预报在他自己的 α 上带来
    负价值，他会退回气候决策（不买）。于是"恒报雨"与"恒报不雨"的 `V̄⁺ ≡ 0` 是
    **定理**而不是补丁。

**稀缺事件噪声抑制**：`λ_ev = n_ev / (n_ev + n0)`，`n_ev` 是观测实际发生该级事件的
独立自然日数。它是共轭先验下的收缩因子（等价于给"技巧"加 `n0` 个"不技巧"的虚拟
事件日），连续压低重阈值的有限样本正偏置，而**不是**硬性门限——证据少只让该项
退回 0（= 无技巧），不会把整个格子删掉。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------- 声明常量
#: 温度损失形状：线性代价（MAE）与大错厌恶（RMSE）的显式权衡。
LOSS_SHAPE = (0.6, 0.4)

#: 温度技能的极端崩溃保护下限。`SS = −1` ⟺ 误差是基准的两倍。
#: 触发率必须披露（`skill_floor_rate`），否则一个"保护"会悄悄变成"封顶"。
SKILL_FLOOR = -1.0

#: 降水代价-损失比先验 `α ~ Beta(a, b)` 与其离散积分格点数。
#: `Beta(2,5)` 的均值 2/7 ≈ 0.29、`P(α ≤ 0.5) ≈ 0.89`：偏重"日常低成本防护"决策
#: （带伞、遮盖、调度），这正是天气 API 的主要用户群。
COST_LOSS_PRIOR = (2.0, 5.0)
COST_LOSS_GRID = 20

#: 3h / 24h 累计两组事件阈值及其显式权重。
RAIN_EVENTS: dict[str, tuple[tuple[float, float], ...]] = {
    "r3": ((1.0, 0.6), (5.0, 0.4)),
    "r24": ((1.0, 0.5), (10.0, 0.3), (25.0, 0.2)),
}

#: 尺度合成权重：3h（短时强降水/出行）与 24h（日总量/排涝）各半。
SCALE_WEIGHTS = {"r3": 0.5, "r24": 0.5}

#: 五段时效需求权重与温度:降水宏观价值权重。
BAND_WEIGHTS = (0.30, 0.25, 0.25, 0.12, 0.08)
ALPHA_V = 0.5

#: 稀疏事件收缩的虚拟事件日数（共轭先验强度）。
RAIN_SHRINK_N0_DAYS = 15

#: 温度维内部结构：`T_b = 0.5·temp_h + 0.25·tmax + 0.25·tmin`。
TEMP_TARGET_WEIGHTS = {"temp_h": 0.5, "tmax": 0.25, "tmin": 0.25}

PRECIP_TARGETS = ("r3_ge1", "r3_ge5", "r24_ge1", "r24_ge10", "r24_ge25")
TEMP_TARGETS = ("temp_h", "tmax", "tmin")


def alpha_grid(a: float = COST_LOSS_PRIOR[0], b: float = COST_LOSS_PRIOR[1],
               n: int = COST_LOSS_GRID) -> tuple[np.ndarray, np.ndarray]:
    """`Beta(a, b)` 在 `n` 个中点上的离散积分权重（归一化到和为 1）。

    用中点规则而不是端点：端点 0 与 1 对应"零代价"与"必损"两个退化决策，两者都不
    是真实用户会做的选择，给它们权重只会让积分在边界上被定义域噪声污染。
    """
    mid = (np.arange(int(n)) + 0.5) / float(n)
    # Beta pdf 的核 x^(a-1)(1-x)^(b-1)；常数 B(a,b) 在归一化时抵消
    w = np.power(mid, a - 1.0) * np.power(1.0 - mid, b - 1.0)
    total = w.sum()
    if not math.isfinite(total) or total <= 0:
        return mid, np.full(int(n), 1.0 / n)
    return mid, w / total


ALPHAS, ALPHA_W = alpha_grid()


# ---------------------------------------------------------------- 温度技能
@dataclass
class TempAggregate:
    """温度充分统计量（全部**可加**：跨日、跨站、跨桶直接相加即可）。

    `sum_f` / `sum_o` / `sum_f_sq` / `sum_o_sq` / `sum_fo` 五项只服务 Murphy 零权重
    诊断，不参与任何分数——它们是"差在哪里"的分解原料，不是另一套权重。
    """

    n: float = 0.0
    sum_abs_e: float = 0.0
    sum_sq_e: float = 0.0
    sum_abs_e0: float = 0.0
    sum_sq_e0: float = 0.0
    sum_f: float = 0.0
    sum_o: float = 0.0
    sum_f_sq: float = 0.0
    sum_o_sq: float = 0.0
    sum_fo: float = 0.0

    def __add__(self, other: "TempAggregate") -> "TempAggregate":
        return TempAggregate(
            n=self.n + other.n,
            sum_abs_e=self.sum_abs_e + other.sum_abs_e,
            sum_sq_e=self.sum_sq_e + other.sum_sq_e,
            sum_abs_e0=self.sum_abs_e0 + other.sum_abs_e0,
            sum_sq_e0=self.sum_sq_e0 + other.sum_sq_e0,
            sum_f=self.sum_f + other.sum_f,
            sum_o=self.sum_o + other.sum_o,
            sum_f_sq=self.sum_f_sq + other.sum_f_sq,
            sum_o_sq=self.sum_o_sq + other.sum_o_sq,
            sum_fo=self.sum_fo + other.sum_fo,
        )

    def scale(self, k: float) -> "TempAggregate":
        """按权重缩放（bootstrap 的日权重乘子）。"""
        return TempAggregate(**{f: getattr(self, f) * k
                                for f in self.__dataclass_fields__})


def temp_aggregate(obs: np.ndarray, fcst: np.ndarray, base: np.ndarray,
                   weights: np.ndarray | None = None) -> TempAggregate:
    """由配对样本构造温度充分统计量。

    `obs` / `fcst` / `base` 等长；`weights` 可选（bootstrap 的日权重）。
    含 NaN 的行必须由调用方先行剔除——本函数不做掩膜，以避免"静默丢样本"。
    """
    if obs.size == 0:
        return TempAggregate()
    w = np.ones(obs.size) if weights is None else np.asarray(weights, dtype=float)
    e = fcst - obs
    e0 = base - obs
    return TempAggregate(
        n=float(w.sum()),
        sum_abs_e=float(np.sum(w * np.abs(e))),
        sum_sq_e=float(np.sum(w * e * e)),
        sum_abs_e0=float(np.sum(w * np.abs(e0))),
        sum_sq_e0=float(np.sum(w * e0 * e0)),
        sum_f=float(np.sum(w * fcst)),
        sum_o=float(np.sum(w * obs)),
        sum_f_sq=float(np.sum(w * fcst * fcst)),
        sum_o_sq=float(np.sum(w * obs * obs)),
        sum_fo=float(np.sum(w * fcst * obs)),
    )


def temp_skill(agg: TempAggregate, *,
               loss_shape: tuple[float, float] = LOSS_SHAPE,
               floor: float = SKILL_FLOOR) -> float | None:
    """温度技能 `T = max(floor, w_mae·SS_MAE + w_rmse·SS_RMSE)`。

    返回 `None` 当且仅当**没有可比的分母**（n = 0 或基准误差恒为 0）：
    基准完美时"不如基准"没有意义，此时给任何数都是伪造。

    分母为 0 的两种情形要分开：
      * `Σ|e⁰| = 0` 且 `Σe⁰² = 0`：基准完美（现实里只出现在构造用例中）→ 无定义；
      * 只有其中一项为 0：该项技能无定义，按**另一项**归一（不失一般性）。
    """
    if agg.n <= 0:
        return None
    parts: list[float] = []
    weights: list[float] = []
    w_mae, w_rmse = loss_shape
    if agg.sum_abs_e0 > 1e-12 and w_mae > 0:
        parts.append(1.0 - agg.sum_abs_e / agg.sum_abs_e0)
        weights.append(w_mae)
    if agg.sum_sq_e0 > 1e-18 and w_rmse > 0:
        parts.append(1.0 - math.sqrt(max(agg.sum_sq_e, 0.0))
                     / math.sqrt(agg.sum_sq_e0))
        weights.append(w_rmse)
    if not parts:
        return None
    total = sum(weights)
    raw = sum(p * w for p, w in zip(parts, weights)) / total
    return float(max(floor, raw))


def temp_skill_parts(agg: TempAggregate) -> dict[str, float | None]:
    """`SS_MAE` / `SS_RMSE` / `SS_MSE` 三个分解值（披露用，权重仍是 0.6/0.4）。

    `SS_MSE = 1 − Σe²/Σe⁰²` 与 `SS_RMSE` 严格单调等价，但它是 Murphy 分解的
    自然宿主（`MSESS = r² − (r − s_f/s_o)² − ((f̄−ō)/s_o)²`），所以要一并发布。
    """
    out: dict[str, float | None] = {"ss_mae": None, "ss_rmse": None,
                                    "ss_mse": None}
    if agg.n <= 0:
        return out
    if agg.sum_abs_e0 > 1e-12:
        out["ss_mae"] = float(1.0 - agg.sum_abs_e / agg.sum_abs_e0)
    if agg.sum_sq_e0 > 1e-18:
        out["ss_rmse"] = float(1.0 - math.sqrt(max(agg.sum_sq_e, 0.0))
                               / math.sqrt(agg.sum_sq_e0))
        out["ss_mse"] = float(1.0 - agg.sum_sq_e / agg.sum_sq_e0)
    return out


def murphy_decomposition(agg: TempAggregate) -> dict[str, float | None]:
    """Murphy (1988) 均方误差技能三重分解（**零权重诊断**）。

    `MSESS = r_fo² − (r_fo − s_f/s_o)² − ((f̄ − ō)/s_o)²`

    三项分别回答三个不同的问题：
      * `phase`（`r²`）：相位/形态对不对——异常场是否同向起伏（满分 1）；
      * `cond_bias`（条件偏差）：振幅对不对——预报振幅与实况振幅是否匹配；
      * `uncond_bias`（系统偏差）：平均值对不对——是否整体偏暖/偏冷。

    只发布、不计分：这三项是 `MSE` 的**恒等分解**，把它们当独立权重就是拿同一个
    误差计多遍（旧口径 80% 温度权重共线测量误差尺度，正是这个病）。
    """
    if agg.n <= 0:
        return {"phase": None, "cond_bias": None, "uncond_bias": None,
                "mean_fcst": None, "mean_obs": None}
    n = agg.n
    mf = agg.sum_f / n
    mo = agg.sum_o / n
    vf = max(agg.sum_f_sq / n - mf * mf, 0.0)
    vo = max(agg.sum_o_sq / n - mo * mo, 0.0)
    cov = agg.sum_fo / n - mf * mo
    if vo <= 1e-18 or vf <= 1e-18:
        return {"phase": 0.0, "cond_bias": None, "uncond_bias": None,
                "mean_fcst": mf, "mean_obs": mo}
    r = float(np.clip(cov / math.sqrt(vf * vo), -1.0, 1.0))
    s_ratio = math.sqrt(vf / vo)
    return {
        "phase": r * r,
        "cond_bias": (r - s_ratio) ** 2,
        "uncond_bias": ((mf - mo) ** 2) / vo,
        "mean_fcst": mf,
        "mean_obs": mo,
        "corr": r,
        "amplitude_ratio": s_ratio,
    }


# ---------------------------------------------------------------- 降水经济价值
@dataclass
class RainCounts:
    """一个（尺度 × 阈值 × 段）格子的分站列联表（整型，可加）。

    `a` 命中、`b` 空报、`c` 漏报、`d` 正确否定；外加 `n_ev_days`（观测实际发生该级
    事件的**独立自然日数**，用于稀疏收缩）与 `sum_f_mm` / `sum_o_mm`（明细诊断的
    雨量比，不入分）。
    """

    a: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    b: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    c: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    d: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    n_ev_days: int = 0
    sum_f_mm: float = 0.0
    sum_o_mm: float = 0.0

    def __add__(self, other: "RainCounts") -> "RainCounts":
        n = max(self.a.size, other.a.size)

        def pad(x: np.ndarray) -> np.ndarray:
            if x.size == n:
                return x
            out = np.zeros(n, dtype=np.int64)
            out[:x.size] = x
            return out

        return RainCounts(a=pad(self.a) + pad(other.a), b=pad(self.b) + pad(other.b),
                          c=pad(self.c) + pad(other.c), d=pad(self.d) + pad(other.d),
                          n_ev_days=self.n_ev_days + other.n_ev_days,
                          sum_f_mm=self.sum_f_mm + other.sum_f_mm,
                          sum_o_mm=self.sum_o_mm + other.sum_o_mm)

    @property
    def total(self) -> np.ndarray:
        return self.a + self.b + self.c + self.d


def rain_counts_from_arrays(obs_mm: np.ndarray, fcst_mm: np.ndarray,
                            station: np.ndarray, valid_day: np.ndarray,
                            threshold: float) -> RainCounts:
    """由配对样本累计分站列联表。

    `n_ev_days` 的去重口径：**观测**发生过该级事件的独立自然日数（跨站合并）。
    用观测而不是预报，是因为收缩要压的是"这场雨真没真下过"这一证据强度；用预报
    会把"报了 30 天雨但从没下过"误当成充足证据。
    """
    if obs_mm.size == 0:
        return RainCounts()
    o = obs_mm >= threshold
    f = fcst_mm >= threshold
    stations = np.unique(station)
    a = np.zeros(stations.size, dtype=np.int64)
    b = np.zeros(stations.size, dtype=np.int64)
    c = np.zeros(stations.size, dtype=np.int64)
    d = np.zeros(stations.size, dtype=np.int64)
    for i, sid in enumerate(stations):
        m = station == sid
        a[i] = int(np.sum(m & f & o))
        b[i] = int(np.sum(m & f & ~o))
        c[i] = int(np.sum(m & ~f & o))
        d[i] = int(np.sum(m & ~f & ~o))
    n_ev = int(np.unique(valid_day[o]).size) if o.any() else 0
    return RainCounts(a=a, b=b, c=c, d=d, n_ev_days=n_ev,
                      sum_f_mm=float(fcst_mm.sum()), sum_o_mm=float(obs_mm.sum()))


#: 气候基率的贝叶斯先验强度（以"虚拟站点"计的样本量）。
#: 站点自有样本只有一季时，`s_k` 的抽样噪声可达 ±0.1，而气候决策的期望损失
#: `min(α, s_k)` 对它是一阶敏感的；用先验把"小样本的站点基率"向"全站合并基率"
#: 收缩，收缩后的值才是**一个真实用户作决策时实际持有的信息**。
RAIN_RATE_PRIOR_N = 96.0


def adoptable_rate(station_counts: "RainCounts",
                   prior_rate: float | None = None,
                   prior_n: float = RAIN_RATE_PRIOR_N) -> np.ndarray:
    """用户**实际可采纳**的气候基率 `s̃_k`（分站）。

    `s̃_k = (a_k + c_k + n_prior · s̄) / (n_k + n_prior)`

    其中 `s̄` 是全部站合并的事件基率。两条理由，缺一不可：

    1. **样本**：站点自有观测只有一季，`s_k = (a+c)/n` 的抽样噪声在 0.01~0.1 量级，
       而气候决策的期望损失对它一阶敏感。用户不会拿一个 ±0.1 的估计去排涝。
    2. **信息**：`s̄` 是同一片区域的可得信息（同纬度同气候带），用户当然会用。
       若把它排除，`max(V, 0)` 的零点会落在比任何真实决策者都更笨的基准上，
       完美预报拿不到 1，量表顶点就名不副实。

    收缩后的 `s̃_k` 仍的分站保留（跨站气候差异不被抹平），只把**样本噪声**压掉。
    """
    n = station_counts.total.astype(float)
    keep = n > 0
    if not keep.any():
        return np.zeros(0)
    wet = (station_counts.a[keep] + station_counts.c[keep]).astype(float)
    nk = n[keep]
    empirical = wet / np.maximum(nk, 1.0)
    # 先验率只用**其余**站点估计（留一），单站时没有"其余站点"，因此不收缩。
    # 这一条保证 `adoptable_rate` 在单站上**精确退化**为实测基率 `(a+c)/n`：
    # 收缩只在"有外部信息可借"时才发生，而不是在没信息时把估计拉向自身（那样
    # 分母 n+prior_n 会把率永久压低一半，峰值的 `V̄⁺` 就永远到不了 1）。
    if nk.size == 1 or prior_rate is not None:
        pool = (float((wet.sum() - wet) / max(nk.sum() - nk, 1.0))
                if nk.size > 1 else 0.0)
        base = np.full(nk.size, float(prior_rate)) if prior_rate is not None \
            else pool
    else:
        base = (wet.sum() - wet) / np.maximum(nk.sum() - nk, 1.0)
    pn = max(float(prior_n), 0.0)
    if nk.size == 1 or pn <= 0:
        return empirical
    return (wet + pn * base) / (nk + pn)


def economic_value(counts: RainCounts, alphas: np.ndarray = ALPHAS,
                   weights: np.ndarray = ALPHA_W,
                   n0_days: float = RAIN_SHRINK_N0_DAYS,
                   climatology_rate: float | None = None,
                   cost_loss_prior: tuple[float, float] = COST_LOSS_PRIOR,
                   cost_loss_grid: int = COST_LOSS_GRID) -> dict:
    """Richardson (2000) 经济价值 `V̄⁺` 与稀疏收缩后的降水分 `P`。

    分站分层（`k` 遍历站点）：
        `E_fcst(α) = Σ_k [α(a_k + b_k) + c_k] / N`
        `E_clim(α) = Σ_k [n_k · min(α, s_k)] / N`
        `E_perf(α) = Σ_k α(a_k + c_k) / N`
        `V(α) = (E_clim − E_fcst) / (E_clim − E_perf)`

    **为什么必须分站算 `E_clim`**：不分站时，模型可以靠"知道梧州比博白湿"白拿技巧分
    ——它只是在复述气候，不是在预报天气。分站后每个站的无信息决策只能用它自己的
    基率 `s_k`，跨站的气候差异不再能兑换成技巧。

    返回：
      * `v_plus`：`Σ_j w_j · max(V(α_j), 0)`（未收缩的原始价值）；
      * `shrink`：`n_ev / (n_ev + n0)`；
      * `score`：`shrink · v_plus`（入分的 `P`）。

    三条定理（`tests/test_dvs_axioms.py` 逐条对拍）：
      * 完美预报 ⇒ `V(α) ≡ 1` ⇒ `V̄⁺ = 1`；
      * 恒报雨 / 恒不报雨 ⇒ `V(α) ≤ 0` ⇒ `max(·,0)` ⇒ `V̄⁺ = 0`；
      * 与基率同分布的随机预报 ⇒ `E[V̄⁺] = 0`（只有采样噪声带来的正偏置，
        由 `shrink` 压制）。
    """
    if (tuple(float(x) for x in cost_loss_prior) != tuple(float(x) for x in COST_LOSS_PRIOR)
            or int(cost_loss_grid) != int(COST_LOSS_GRID)):
        # 配置改了先验/格点数就必须用配置算：默认实参在 import 时已绑定模块级
        # ALPHAS，光把它们写进 meta.config 而不进公式等于仪表盘说谎。
        alphas, weights = alpha_grid(float(cost_loss_prior[0]),
                                     float(cost_loss_prior[1]), int(cost_loss_grid))
    n = counts.total.astype(float)
    keep = n > 0
    out = {"v_plus": 0.0, "score": 0.0, "shrink": 0.0,
           "n_ev_days": int(counts.n_ev_days), "n": float(n.sum()),
           "value_by_alpha": {}}
    if not keep.any():
        return out
    # d（正确否定）由 a/b/c/n 唯一决定，不需要单独参与运算；保留在结构里供读者
    # 核对恒等式 a + b + c + d = n_k。
    a = counts.a[keep].astype(float)
    b = counts.b[keep].astype(float)
    c = counts.c[keep].astype(float)
    nk = n[keep]
    big_n = float(nk.sum())
    # 用户可采纳的基率：站点样本 + 全站合并先验（`climatology_rate` 若给出则用它作先验）
    s = adoptable_rate(counts, prior_rate=climatology_rate)
    v_bar = 0.0
    curve: dict[str, float] = {}
    for al, w in zip(alphas, weights):
        e_f = float(np.sum(al * (a + b) + c)) / big_n
        e_c = float(np.sum(nk * np.minimum(al, s))) / big_n
        e_p = float(np.sum(al * (a + c))) / big_n
        denom = e_c - e_p
        v = (e_c - e_f) / denom if denom > 1e-12 else 0.0
        v_pos = max(float(v), 0.0)
        v_bar += float(w) * v_pos
        curve[f"{al:.3f}"] = float(v)
    shrink = (counts.n_ev_days / (counts.n_ev_days + float(n0_days))
              if counts.n_ev_days > 0 else 0.0)
    out.update(v_plus=float(v_bar), score=float(shrink * v_bar),
               shrink=float(shrink), value_by_alpha=curve)
    return out


def rain_target_score(counts_by_event: Mapping[str, RainCounts],
                      events: Mapping[str, Sequence[tuple[float, float]]]
                      = RAIN_EVENTS,
                      *, n0_days: float = RAIN_SHRINK_N0_DAYS,
                      cost_loss_prior: tuple[float, float] = COST_LOSS_PRIOR,
                      cost_loss_grid: int = COST_LOSS_GRID) -> dict:
    """同一尺度下多个阈值的加权合成（如 3h：`≥1mm` 0.6 + `≥5mm` 0.4）。

    **缺项处理**：某阈值没有样本时**不是**按剩余权重归一（那会让唯一的稀疏重阈值
    独占话语权——旧口径"最小存活权重"病），而是按原权重算 0：没有证据 = 无技巧。
    `coverage` 披露实际有样本的权重占比，读者据此知道这个尺度分有多少是"真测到的"。
    """
    per_event: dict[str, dict] = {}
    num = 0.0
    covered = 0.0
    n_scales = 0
    for name, specs in events.items():
        total_w = sum(w for _thr, w in specs) or 1.0
        ev_num = 0.0
        ev_cov = 0.0
        for thr, w in specs:
            key = f"{name}_{_thr_key(thr)}"
            c = counts_by_event.get(key)
            if c is None or c.total.sum() == 0:
                per_event[key] = {"score": 0.0, "v_plus": 0.0, "shrink": 0.0,
                                  "n": 0.0, "threshold_mm": thr, "weight": w}
                continue
            r = economic_value(c, n0_days=n0_days,
                               cost_loss_prior=cost_loss_prior,
                               cost_loss_grid=cost_loss_grid)
            per_event[key] = {**{k: r[k] for k in
                                 ("score", "v_plus", "shrink", "n", "n_ev_days")},
                              "threshold_mm": thr, "weight": w}
            ev_num += w * r["score"]
            ev_cov += w
        per_event[name] = {"score": ev_num / total_w,
                           "coverage": ev_cov / total_w}
        num += ev_num / total_w
        covered += ev_cov / total_w
        n_scales += 1
    n_scales = max(n_scales, 1)
    return {"score": num / n_scales, "coverage": covered / n_scales,
            "events": per_event}


def _thr_key(thr: float) -> str:
    return f"ge{int(thr)}" if float(thr).is_integer() else f"ge{thr}"


def precip_score(r3: dict | None, r24: dict | None,
                 scale_weights: Mapping[str, float] = SCALE_WEIGHTS) -> dict:
    """尺度合成：`P = w_r3 · P_r3 + w_r24 · P_r24`。

    两个尺度代表截然不同的生活/作业决策（"这几小时要不要出门"vs"今天要不要排涝"），
    因此按**等价值**加权，而不是按数据方差加权——冗余相关性由统计层（按日聚类的
    块自助）自动吸收，不需要在点估计里做方差匹配。
    """
    parts = {"r3": r3 or {"score": 0.0, "coverage": 0.0},
             "r24": r24 or {"score": 0.0, "coverage": 0.0}}
    total_w = sum(scale_weights.values()) or 1.0
    score = sum(float(scale_weights.get(k, 0.0)) * float(v.get("score") or 0.0)
                for k, v in parts.items()) / total_w
    coverage = sum(float(scale_weights.get(k, 0.0)) * float(v.get("coverage") or 0.0)
                   for k, v in parts.items()) / total_w
    return {"score": float(score), "coverage": float(coverage), "scales": parts}


# ---------------------------------------------------------------- 合成
def temp_score(targets: Mapping[str, float | None],
               weights: Mapping[str, float] = TEMP_TARGET_WEIGHTS) -> float | None:
    """段内温度维：`T_b = Σ_x w_x · T_x / Σ w_x`（缺项按剩余权重归一并披露覆盖）。

    为什么要归一：若某源不提供日产品，`tmax/tmin` 缺失不该让它的 `T_b` 变成 0
    （那是把"不提供"罚成"最差"），也不该让 `temp_h` 独占（那是把一项当全部）。
    覆盖比例由调用方另行披露（`coverage`），主分的"能力外记 0"在**段级**做，不在
    项级做——段的缺失是"这家没有这个时效能力"，项的缺失是"这个产品它不单独提供"。
    """
    num = 0.0
    den = 0.0
    for k, w in weights.items():
        v = targets.get(k)
        if v is None:
            continue
        num += float(w) * float(v)
        den += float(w)
    if den <= 0:
        return None
    return float(num / den)


def target_coverage(targets: Mapping[str, float | None],
                    weights: Mapping[str, float] = TEMP_TARGET_WEIGHTS) -> float:
    den = sum(float(w) for w in weights.values()) or 1.0
    num = sum(float(w) for k, w in weights.items() if targets.get(k) is not None)
    return num / den


def segment_score(temp: float | None, precip: float | None,
                  alpha_v: float = ALPHA_V) -> float:
    """段分 `S_b = α_V·T_b + (1−α_V)·P_b`。

    **能力外记 0**：`None`（该源声明不提供该要素/该时效）按 0 处理，即"用它替代
    基准（L0）"——这正是"能力外没有增量价值"的定义。故障缺测**不走这条路**：它
    既不进分子也不进分母，而是计入可用率 `avail`（见 `availability`）。
    """
    t = 0.0 if temp is None else float(temp)
    p = 0.0 if precip is None else float(precip)
    return float(alpha_v) * t + (1.0 - float(alpha_v)) * p


@dataclass
class Profile:
    """一套业务画像：宏观权重 + 段权重 + 降水先验/阈值偏好。"""

    key: str
    label: str
    alpha_v: float
    band_weights: tuple[float, ...]
    cost_loss_prior: tuple[float, float] = COST_LOSS_PRIOR
    rain_events: Mapping[str, Sequence[tuple[float, float]]] = None  # type: ignore[assignment]
    note: str = ""

    def __post_init__(self) -> None:
        if self.rain_events is None:
            object.__setattr__(self, "rain_events", RAIN_EVENTS)


PROFILES: dict[str, Profile] = {
    "general": Profile(
        key="general", label="通用（默认）", alpha_v=0.50,
        band_weights=(0.30, 0.25, 0.25, 0.12, 0.08),
        note="温度:降水 = 50:50，Beta(2,5)，默认阈值权重",
    ),
    "outdoor": Profile(
        key="outdoor", label="出行/户外", alpha_v=0.30,
        band_weights=(0.45, 0.30, 0.20, 0.04, 0.01),
        cost_loss_prior=(2.0, 4.0),
        note="偏重短期晴雨；更看重 1–3 天",
    ),
    "agriculture": Profile(
        key="agriculture", label="农业/防汛", alpha_v=0.30,
        band_weights=(0.15, 0.20, 0.35, 0.20, 0.10),
        cost_loss_prior=(2.0, 3.0),
        rain_events={"r3": ((1.0, 0.6), (5.0, 0.4)),
                     "r24": ((1.0, 0.3), (10.0, 0.4), (25.0, 0.3))},
        note="24h 重阈值权重上调；Beta(2,3) 偏重高保护代价决策",
    ),
    "energy": Profile(
        key="energy", label="能源/暖通", alpha_v=0.80,
        band_weights=(0.35, 0.30, 0.25, 0.08, 0.02),
        note="偏重高低温精度；温度:降水 = 80:20",
    ),
}


def composite(segments: Mapping[int, float],
              band_weights: Sequence[float] = BAND_WEIGHTS) -> float:
    """`DVS = 100 · Σ_b w_b · S_b`（段权重和为 1 时严格落在 −100..100 的刻度上）。"""
    total = 0.0
    for i, w in enumerate(band_weights, 1):
        total += float(w) * float(segments.get(i, 0.0))
    return 100.0 * total


def composite_from_parts(temp_bands: Mapping[int, float | None],
                         precip_bands: Mapping[int, float | None],
                         alpha_v: float = ALPHA_V,
                         band_weights: Sequence[float] = BAND_WEIGHTS) -> dict:
    """由 `5×2` 个段值直接合成（前端滑块用的也是这一条公式）。"""
    seg = {}
    for b in range(1, len(band_weights) + 1):
        seg[b] = segment_score(temp_bands.get(b), precip_bands.get(b), alpha_v)
    return {
        "score": float(composite(seg, band_weights)),
        "S_b": [float(seg[b]) for b in sorted(seg)],
        "T_b": [None if (tv := temp_bands.get(b)) is None else float(tv)
                for b in range(1, len(band_weights) + 1)],
        "P_b": [None if (pv := precip_bands.get(b)) is None else float(pv)
                for b in range(1, len(band_weights) + 1)],
    }


def coverage_score(temp_bands: Mapping[int, float | None],
                   precip_bands: Mapping[int, float | None],
                   alpha_v: float = ALPHA_V,
                   band_weights: Sequence[float] = BAND_WEIGHTS) -> dict:
    """`S|覆盖`：只在**有样本**的段内重新归一权重的平均表现。

    它回答"在其能力/证据覆盖的时效内，这家有多准"，与主分回答的"考虑能力边界的
    决策价值"是两件事。主榜排名只认主分；`S|覆盖` 用于公平识别"短临专家"。
    """
    num = 0.0
    den = 0.0
    bands = 0
    for i, w in enumerate(band_weights, 1):
        t = temp_bands.get(i)
        p = precip_bands.get(i)
        if t is None and p is None:
            continue
        num += float(w) * segment_score(t, p, alpha_v)
        den += float(w)
        bands += 1
    if den <= 0:
        return {"score": None, "bands": 0, "weight": 0.0}
    return {"score": float(100.0 * num / den), "bands": bands, "weight": den}


def availability(samples: Mapping[str, Mapping[str, float | int]]) -> float:
    """可用率：声明提供（`declared`）的时刻里，实际拿到并配对成功的比例。

    分子分母都数**决策时刻**而不是"样本行"：一个批次抓 16 天与抓 3 天，前者对
    可用率的贡献不应该更大——可用率问的是"我叫它的时候它在不在"。
    """
    ok = 0.0
    total = 0.0
    for _target, row in samples.items():
        ok += float(row.get("delivered", 0.0))
        total += float(row.get("expected", 0.0))
    if total <= 0:
        return 0.0
    return float(min(max(ok / total, 0.0), 1.0))


def delivered_score(dvs: float | None, avail: float) -> float | None:
    """`DVS_del = avail · DVS`：面向业务接入方的"考虑掉线与残缺后的实际收益"。"""
    if dvs is None:
        return None
    return float(avail) * float(dvs)


def l1_align(dvs_m_window: float | None, dvs_l1_window: float | None,
             dvs_l1_full: float | None) -> float | None:
    """短历史新源的**无同台依赖**时期校准。

        `DVS_aligned(m) = DVS(L1; W) + [DVS(m; D_m) − DVS(L1; D_m)]`

    `L1 = ecmwf_ifs025` 是永久免费的固定锚点，它经历了与新源完全相同的那几周冷锋或
    干旱，所以两者的**同日配对差**把"季节/天气难度"约掉了。整个过程不接触任何其他
    竞争源，因此公理 A1（增删任意其他源，本源分数逐位不变）毫发无损——这是它相对
    "双向加法难度对齐 + 经验贝叶斯收缩"的根本区别：后者必须看同台全部选手。
    """
    if dvs_l1_full is None:
        return None
    if dvs_m_window is None or dvs_l1_window is None:
        return dvs_l1_full
    return float(dvs_l1_full) + (float(dvs_m_window) - float(dvs_l1_window))


__all__ = [
    "ALPHAS", "ALPHA_V", "ALPHA_W", "BAND_WEIGHTS", "COST_LOSS_GRID",
    "COST_LOSS_PRIOR", "LOSS_SHAPE", "PRECIP_TARGETS", "PROFILES",
    "RAIN_EVENTS", "RAIN_SHRINK_N0_DAYS", "SCALE_WEIGHTS", "SKILL_FLOOR",
    "TEMP_TARGETS", "TEMP_TARGET_WEIGHTS", "Profile", "RainCounts",
    "TempAggregate", "alpha_grid", "availability", "composite",
    "composite_from_parts", "coverage_score", "delivered_score",
    "economic_value", "l1_align", "murphy_decomposition", "precip_score",
    "rain_counts_from_arrays", "rain_target_score", "segment_score",
    "target_coverage", "temp_aggregate", "temp_score", "temp_skill",
    "temp_skill_parts",
]

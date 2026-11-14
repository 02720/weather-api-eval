"""DVS v6 基准层：L0 因果物理基准（分数 0 点）与静态气候表。

**L0 是什么、为什么是它（第一性原理）**

分数的语义是"这家源替我消除了多少决策损失"，那必须有一个**代价为零的对照**：
如果我什么都不买，我自己能拿到什么样的预报？L0 就是这个问题的答案，它必须同时
满足三条约束——

1. **因果可得**：L0 只能使用决策时刻 `t_run` 之前已经入库的观测。任何用到
   `t_run` 之后信息的基准都会把"未来的天气"塞进零点，让所有源集体虚高。
2. **有界**：L0 不能在长时效上发散。气候态 + 阻尼持续性在 `ℓ → ∞` 时收敛到气候
   态，不会像线性外推那样越推越离谱（旧 v5 的参照 RMSE 随 lead 从 2.13 涨到
   4.60°C，正是这个病）。
3. **物理可解释**：每个加项都有明确的天文/地理/气象含义，读者能用页面公布的数值
   手算复现。

于是：

    F⁰ᵀ(s, t) = clim(s, doy ± w, h) + δ_s(t_run) + exp(−ℓ/τ_s) · a₀(t_run)

* `clim`：ERA5-Land 1991–2020 同期同钟点气候均值。仓库里没有运行期联网依赖，
  静态表由 `scripts/build_era5_climatology.py` 一次性生成并随仓库固化。**表不存在
  时**退回站点自有观测的因果平滑气候态（`source="obs_climatology"`），并在报告里
  如实披露当前用的是哪一个——绝不假装有 ERA5 表。
* `δ_s`：`t_run − δ_obs` 之前 `offset_days`（30）个自然日内 `(obs − clim)` 的均值。
  它吸收的是**静态**偏差（站点与格点的海拔差、城市热岛、下垫面），不是"今天暖多少"。
* `a₀`：`t_run − δ_obs` 之前最近 3 小时观测相对 `clim + δ_s` 的平均异常。
* `τ_s`：分站（可选分季）的异常自相关衰减常数。`ℓ' → 0` 时 `exp(0) = 1`，基准退化
  为**持续性**（最朴素但很难被击败的本地预报）；`ℓ' → ∞` 时有界收敛到气候态。

**为什么日极值不能由逐小时气候态取 max/min**：日最高温出现的**钟点**本身在摆动，
先算逐小时气候平均再取日 max，等于先抹平峰值时刻再取峰值——Jensen 不等式给出
系统性低估。所以日极值必须直接对**日最高/日最低序列**建气候表，再叠加日极值偏移
与按日阻尼的持续性。

**降水没有"人造降雨序列"**：降水是零膨胀量，捏一条"气候态小雨"当基准只会制造一个
没人会照着做的稻草人。降水的 0 点直接定义在代价-损失框架里：**该站历史基率下的
最优无信息决策**（见 `dvs.economic_value`）。本模块只为它提供"气候基率"这一输入。
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

#: 气候态的日历平滑半窗（天，静态表生成时使用；也作为观测气候态的因果窗）。
#: `doy ± 7` 抹平天气尺度的抽样噪声，同时保留季内变化：再窄则样本仍受单次寒潮
#: 污染，再宽则季内趋势被抹平。
DEFAULT_CLIM_SMOOTH_DAYS = 7

#: 站点静态偏移的因果窗（自然日）。30 天足够抹掉一次寒潮/回暖，又足够短到能跟上
#: 季节推进（秋分后月均温每 30 天下降 5~8°C，用 90 天会把 9 月的 δ 带到 11 月）。
DEFAULT_OFFSET_DAYS = 30

#: 最近异常窗（小时）：`t_run − δ_obs` 之前最近这几个小时的观测。
DEFAULT_RECENT_HOURS = 3

#: 持续性衰减常数 τ 的拟合上下界（小时）。下限 6h 防止把一小时尺度噪声当信号；
#: 上限 240h 防止某个不稳定站把持续性拉成"永不消失的异常"。
TAU_BOUNDS = (6.0, 240.0)

#: τ 的声明兜底值：72h ≈ 3 天，是天气系统的时间尺度量级。用它是**声明**而非猜测，
#: 报告会披露每个站实际用的是"拟合值"还是这个兜底值。
DEFAULT_TAU_HOURS = 72.0

#: 相对湿度/降水判定阈值：国内业务"有降水"= 0.1mm。
RAIN_OCCURRENCE_MM = 0.1


def persistence_decay(lead_hours: np.ndarray, tau_hours: float) -> np.ndarray:
    """`exp(−ℓ/τ)`：异常保留比例。

    两个端点是有意设计的：`ℓ → 0` 时 = 1（完全持续性，基准 = 最近观测实况），
    `ℓ → ∞` 时 = 0（有界收敛到气候态）。中间是单参数单调过渡——只声明一个时间
    常数，读者"为什么第 5 天基准只保留 20% 异常"能一步算术复现。
    """
    ell = np.maximum(np.asarray(lead_hours, dtype=float), 0.0)
    tau = max(float(tau_hours), 1e-6)
    return np.exp(-ell / tau)


# ---------------------------------------------------------------- 日序工具
def doy_of(days: np.ndarray) -> np.ndarray:
    """自然日序号（`date.toordinal()` 尺度）→ 年内日序 1..366（闰年顺延）。"""
    arr = np.asarray(days, dtype=np.int64)
    if arr.size == 0:
        return arr.astype(np.int64)
    uniq, inv = np.unique(arr, return_inverse=True)
    doy = np.array([date.fromordinal(int(d)).timetuple().tm_yday for d in uniq],
                   dtype=np.int64)
    return doy[inv.ravel()]


def doy_of_iso(days_iso: Sequence[str]) -> np.ndarray:
    """ISO 日期串 → 年内日序。"""
    return np.array([date.fromisoformat(s[:10]).timetuple().tm_yday
                     for s in days_iso], dtype=np.int64)


# ---------------------------------------------------------------- 静态气候表
@dataclass
class StationClimatology:
    """单站气候表（小时表 + 日极值表 + 降水基率表）。"""

    station: str
    #: `hourly[doy-1, hour]` = 温度气候均值，形状 (366, 24)
    hourly: np.ndarray
    #: `tmax[doy-1]` / `tmin[doy-1]` = 日极值气候均值，形状 (366,)
    tmax: np.ndarray
    tmin: np.ndarray
    #: `rain_prob[hour, doy-1]` = 该钟点有降水（≥0.1mm）的气候概率，形状 (24, 366)
    rain_prob: np.ndarray | None = None
    tau_hours: float = DEFAULT_TAU_HOURS
    #: 0=DJF, 1=MAM, 2=JJA, 3=SON
    tau_by_season: dict[int, float] = field(default_factory=dict)
    #: "era5land_1991_2020" / "obs_climatology"
    source: str = "unknown"
    tau_fitted: bool = True

    def seasonal_tau(self, month: int) -> float:
        season = _season_of_month(month)
        return float(self.tau_by_season.get(season, self.tau_hours))

    def tau_for_days(self, days: np.ndarray) -> np.ndarray:
        """按有效日取该日所属季节的 τ（无分季值则用全年值）。"""
        arr = np.asarray(days, dtype=np.int64)
        if not self.tau_by_season:
            return np.full(arr.shape, self.tau_hours, dtype=float)
        uniq, inv = np.unique(arr, return_inverse=True)
        taus = np.array([self.seasonal_tau(date.fromordinal(int(d)).month)
                         for d in uniq], dtype=float)
        return taus[inv.ravel()]


def _season_of_month(month: int) -> int:
    return {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1,
            6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}.get(int(month), 0)


class ERA5LandClimatology:
    """多站静态气候表容器（随仓库固化的 JSON，一次生成、零运行期联网）。

    文件格式（`data/climatology/era5land_1991_2020.json`）::

        {"schema": 1, "baseline": "1991-2020", "source": "era5land",
         "stations": {"wuzhou": {"hourly": [[..24..] x 366],
                                 "tmax": [...366...], "tmin": [...366...],
                                 "rain_prob": [[...366...] x 24] | null,
                                 "tau_hours": 68.4,
                                 "tau_by_season": {"0": 80.1, ...}}}}

    表不存在/不可解析时**不报错**：`available` 为 False，调用方退回观测气候态并披露。
    """

    def __init__(self, stations: Mapping[str, StationClimatology],
                 baseline: str = "unknown", source: str = "unknown"):
        self._stations = dict(stations)
        self.baseline = baseline
        self.source = source

    @property
    def available(self) -> bool:
        return bool(self._stations)

    def get(self, station: str) -> StationClimatology | None:
        return self._stations.get(station)

    @classmethod
    def load(cls, path: str | Path) -> "ERA5LandClimatology":
        p = Path(path)
        if not p.is_file():
            return cls({}, baseline="missing", source=f"missing:{p.name}")
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls({}, baseline="unreadable", source=f"unreadable:{p.name}")
        out: dict[str, StationClimatology] = {}
        for sid, node in (raw.get("stations") or {}).items():
            if not isinstance(node, dict):
                continue
            try:
                hourly = np.asarray(node["hourly"], dtype=float)
                tmax = np.asarray(node["tmax"], dtype=float)
                tmin = np.asarray(node["tmin"], dtype=float)
            except (KeyError, TypeError, ValueError):
                continue
            # 形状校验：一张形状错位的表会让所有分数静默错位，宁可当作没有表
            if (hourly.shape != (366, 24) or tmax.shape != (366,)
                    or tmin.shape != (366,)):
                continue
            rp = None
            if node.get("rain_prob") is not None:
                cand = np.asarray(node["rain_prob"], dtype=float)
                if cand.shape == (24, 366):
                    rp = cand
            out[sid] = StationClimatology(
                station=sid, hourly=hourly, tmax=tmax, tmin=tmin, rain_prob=rp,
                tau_hours=float(node.get("tau_hours") or DEFAULT_TAU_HOURS),
                tau_by_season={int(k): float(v) for k, v in
                               (node.get("tau_by_season") or {}).items()},
                source=str(raw.get("source") or "era5land"),
                tau_fitted=bool(node.get("tau_hours")),
            )
        return cls(out, baseline=str(raw.get("baseline") or "unknown"),
                   source=str(raw.get("source") or "era5land"))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "schema": 1, "baseline": self.baseline, "source": self.source,
            "stations": {
                sid: {
                    "hourly": np.round(c.hourly, 3).tolist(),
                    "tmax": np.round(c.tmax, 3).tolist(),
                    "tmin": np.round(c.tmin, 3).tolist(),
                    "rain_prob": (np.round(c.rain_prob, 4).tolist()
                                  if c.rain_prob is not None else None),
                    "tau_hours": round(float(c.tau_hours), 2),
                    "tau_by_season": {str(k): round(float(v), 2)
                                      for k, v in c.tau_by_season.items()},
                } for sid, c in self._stations.items()},
        }, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        return p


# ---------------------------------------------------------------- 观测气候态
def build_obs_climatology(station: str,
                          obs_pairs: Sequence[tuple[str, float | None, float | None]],
                          *, smooth_days: int = DEFAULT_CLIM_SMOOTH_DAYS,
                          ) -> StationClimatology | None:
    """用**站点自有观测**构造过渡期气候表（ERA5 静态表缺失时的降级路径）。

    **因果**：`doy` 的气候值只用该日**之前**的观测（逐日递推），绝不让基准偷看它
    要预测的那一天——与旧 v5 的 `fixed_trailing` 同一哲学。

    **代价必须说清楚**：站点自有样本只有一季（实测 82 天），"同期同钟点"平均不过
    1~2 个样本，气候表本身就是噪声。所以本表的定位是**把分数跑起来**，
    报告同时披露 `climatology_source="obs_climatology"`，绝不冒称 ERA5 30 年气候。

    实现是逐日递推（每天把当天 24 个钟点并入累计和），不是"每个 doy 扫一遍样本"：
    后者是 O(366 × 年数 × 24)，在长历史上是几十万次 numpy 小数组操作。
    """
    by_day: dict[int, dict[int, float]] = defaultdict(dict)
    rain_by_day: dict[int, dict[int, float]] = defaultdict(dict)
    for ts, temp, rain in obs_pairs:
        if not isinstance(ts, str) or len(ts) < 13:
            continue
        try:
            d = date.fromisoformat(ts[:10]).toordinal()
        except ValueError:
            continue
        hour = int(ts[11:13])
        if temp is not None and isinstance(temp, (int, float)) and math.isfinite(temp):
            by_day[d][hour] = float(temp)
        if rain is not None and isinstance(rain, (int, float)) and math.isfinite(rain):
            rain_by_day[d][hour] = float(rain)
    if not by_day:
        return None
    # 逐日递推累计（严格因果：先写当天的气候值，再把当天并入累计）
    sum_h = np.zeros(24)
    cnt_h = np.zeros(24)
    rain_cnt = np.zeros(24)
    rain_tot = np.zeros(24)
    hourly = np.full((366, 24), np.nan)
    rain_prob = np.full((24, 366), np.nan)
    tmax = np.full(366, np.nan)
    tmin = np.full(366, np.nan)
    sum_max = 0.0
    sum_min = 0.0
    cnt_day = 0
    for d in sorted(by_day):
        doy = date.fromordinal(d).timetuple().tm_yday
        with np.errstate(invalid="ignore", divide="ignore"):
            hourly[doy - 1] = np.where(cnt_h > 0, sum_h / np.maximum(cnt_h, 1),
                                       np.nan)
            rain_prob[:, doy - 1] = np.where(
                rain_cnt > 0, rain_tot / np.maximum(rain_cnt, 1), np.nan)
        if cnt_day:
            tmax[doy - 1] = sum_max / cnt_day
            tmin[doy - 1] = sum_min / cnt_day
        vals = by_day[d]
        for hour, v in vals.items():
            sum_h[hour] += v
            cnt_h[hour] += 1
        rvals = rain_by_day.get(d) or {}
        for hour, r in rvals.items():
            rain_cnt[hour] += 1
            rain_tot[hour] += 1.0 if r >= RAIN_OCCURRENCE_MM else 0.0
        if vals:
            sum_max += max(vals.values())
            sum_min += min(vals.values())
            cnt_day += 1
    # 兜底：整列缺样本的钟点用该钟点的全局均值；日极值缺样本用已有均值
    col_mean = np.nanmean(np.where(np.isfinite(hourly), hourly, np.nan), axis=0)
    for hour in range(24):
        col = hourly[:, hour]
        bad = ~np.isfinite(col)
        if bad.any() and np.isfinite(col_mean[hour]):
            col[bad] = col_mean[hour]
    hourly = np.where(np.isfinite(hourly), hourly, np.nanmean(hourly))
    tmax = np.where(np.isfinite(tmax), tmax, np.nanmean(tmax))
    tmin = np.where(np.isfinite(tmin), tmin, np.nanmean(tmin))
    # 降水基率缺样本 → 0（"没有证据说明会下雨"是无信息决策的诚实输入）
    rain_prob = np.where(np.isfinite(rain_prob), rain_prob, 0.0)
    series = [(d * 1440 + h, v) for d, vals in sorted(by_day.items())
              for h, v in sorted(vals.items())]
    tau = fit_persistence_tau(series)
    return StationClimatology(
        station=station, hourly=hourly, tmax=tmax, tmin=tmin,
        rain_prob=rain_prob,
        tau_hours=tau if tau is not None else DEFAULT_TAU_HOURS,
        source="obs_climatology", tau_fitted=tau is not None,
    )


def fit_persistence_tau(series: Sequence[tuple[int, float]],
                        bounds: tuple[float, float] = TAU_BOUNDS,
                        ) -> float | None:
    """由（分钟序号, 温度）观测序列拟合异常自相关衰减时间常数 τ。

    **方法**：先去季节循环（按钟点做因果平滑均值）得到异常 `a_t`；再按滞后
    `lag`（6..120h）算 `ρ(lag) = corr(a_t, a_{t+lag})`；最后对 `log ρ ≈ −lag/τ`
    过原点做最小二乘解 τ。

    为什么用指数闭式而不是 AR(1) 逐点递推：`exp(−ℓ/τ)` 是**单参数**的，读者一眼
    能算出"第 N 天还保留多少"，也无从在长时效上悄悄漂移。

    返回 `None` 表示样本不足或拟合不可靠（有效滞后点 < 4）——调用方退回
    `DEFAULT_TAU_HOURS` 并披露"该站 τ 为声明兜底值"，绝不假装拟合过。
    """
    rows = [(int(m), float(v)) for m, v in series
            if v is not None and math.isfinite(float(v))]
    if len(rows) < 240:
        return None
    rows.sort()
    minute = np.array([r[0] for r in rows], dtype=np.int64)
    vals = np.array([r[1] for r in rows], dtype=float)
    anom = vals - _causal_seasonal(vals, minute)
    ok = np.isfinite(anom)
    if ok.sum() < 240:
        return None
    xs: list[float] = []
    ys: list[float] = []
    n = len(anom)
    for lag in range(6, 121, 6):
        if lag >= n // 2:
            break
        a = anom[:-lag][ok[:-lag] & ok[lag:]]
        b = anom[lag:][ok[:-lag] & ok[lag:]]
        if a.size < 100 or a.std() < 1e-6 or b.std() < 1e-6:
            continue
        rho = float(np.corrcoef(a, b)[0, 1])
        if not math.isfinite(rho) or rho <= 0.02:
            continue
        xs.append(float(lag))
        ys.append(math.log(rho))
    if len(xs) < 4:
        return None
    x = np.asarray(xs)
    y = np.asarray(ys)
    slope = float(np.sum(x * y) / np.sum(x * x))
    if slope >= 0:
        return None
    tau = -1.0 / slope
    return float(min(max(tau, bounds[0]), bounds[1]))


def _causal_seasonal(vals: np.ndarray, minute: np.ndarray,
                     window_days: int = 15) -> np.ndarray:
    """逐点因果季节基线：只用该点**之前** `window_days` 天内的同钟点样本。

    纯 numpy 的滑动窗口（每个钟点一次 `cumsum` + 两次 `searchsorted`），复杂度
    O(n log n)，不是逐点重扫。样本不足（< 3 个）时退回该钟点全局均值。
    """
    hour = (minute % 1440) // 60
    day = minute // 1440
    out = np.full(len(vals), np.nan)
    for h in range(24):
        idx = np.flatnonzero(hour == h)
        if idx.size == 0:
            continue
        order = np.argsort(day[idx], kind="stable")
        idx = idx[order]
        d = day[idx]
        v = vals[idx]
        finite = np.isfinite(v)
        cs = np.concatenate(([0.0], np.cumsum(np.where(finite, v, 0.0))))
        cc = np.concatenate(([0.0], np.cumsum(finite.astype(np.int64))))
        lo = np.searchsorted(d, d - window_days, side="left")
        hi = np.searchsorted(d, d, side="left")     # 严格因果：不含当天
        cnt = cc[hi] - cc[lo]
        tot = cs[hi] - cs[lo]
        mean = np.where(cnt >= 3, tot / np.maximum(cnt, 1), np.nanmean(v[finite])
                        if finite.any() else np.nan)
        out[idx] = mean
    return out


# ---------------------------------------------------------------- L0 基准
@dataclass
class L0Baseline:
    """单站、单决策时刻的 L0 因果物理基准。

    `hourly_temp` / `daily_temp` / `rain_prob` 都是**纯函数**：给定有效时刻（或有效
    日）与时效，返回基准值。基准与预报源完全无关（公理 A1），也与榜上其他源无关。
    """

    station: str
    climatology: StationClimatology
    offset_temp: float = 0.0
    offset_tmax: float = 0.0
    offset_tmin: float = 0.0
    a0_temp: float = 0.0
    a0_tmax: float = 0.0
    a0_tmin: float = 0.0
    tau_hours: float = DEFAULT_TAU_HOURS
    observed_offset_days: int = 0
    climatology_source: str = "unknown"
    tau_fitted: bool = True

    def hourly_temp(self, valid_minute: np.ndarray,
                    lead_hours: np.ndarray) -> np.ndarray:
        """`clim(doy, h) + δ + exp(−ℓ/τ)·a₀`（向量化）。"""
        vm = np.asarray(valid_minute, dtype=np.int64)
        lead = np.asarray(lead_hours, dtype=float)
        doy = doy_of(vm // 1440)
        hour = (vm % 1440) // 60
        clim = _lookup_hourly(self.climatology.hourly, doy, hour)
        tau = self.climatology.tau_for_days(vm // 1440) if self.climatology.tau_by_season \
            else np.full(vm.shape, self.tau_hours)
        decay = np.exp(-lead / np.maximum(tau, 1e-6))
        return clim + self.offset_temp + decay * self.a0_temp

    def daily_temp(self, valid_day: np.ndarray, day_index: np.ndarray,
                   which: str) -> np.ndarray:
        """日最高/最低基准：`clim_ext(doy) + δ_ext + exp(−k·24/τ)·a₀_ext`。

        `day_index` 是**日序号 k**（不是小时 lead）：日极值的兑现时刻是那一天本身，
        用 `k·24` 换算成等效小时，保证 `exp(−ℓ/τ)` 与"提前几天"同一刻度。
        """
        vd = np.asarray(valid_day, dtype=np.int64)
        k = np.asarray(day_index, dtype=float) * 24.0
        doy = doy_of(vd)
        tau = self.climatology.tau_for_days(vd) if self.climatology.tau_by_season \
            else np.full(vd.shape, self.tau_hours)
        decay = np.exp(-k / np.maximum(tau, 1e-6))
        if which == "max":
            return _lookup_daily(self.climatology.tmax, doy) + self.offset_tmax \
                + decay * self.a0_tmax
        return _lookup_daily(self.climatology.tmin, doy) + self.offset_tmin \
            + decay * self.a0_tmin

    def rain_prob(self, valid_minute: np.ndarray) -> np.ndarray:
        """该时刻的气候降水基率（代价-损失框架的输入，不是一条人造降雨序列）。"""
        rp = self.climatology.rain_prob
        if rp is None:
            return np.full(len(valid_minute), np.nan)
        vm = np.asarray(valid_minute, dtype=np.int64)
        doy = doy_of(vm // 1440)
        hour = (vm % 1440) // 60
        return rp[hour, np.clip(doy - 1, 0, 365)]


def _lookup_hourly(table: np.ndarray, doy: np.ndarray, hour: np.ndarray) -> np.ndarray:
    d = np.clip(np.asarray(doy, dtype=np.int64) - 1, 0, table.shape[0] - 1)
    h = np.clip(np.asarray(hour, dtype=np.int64), 0, table.shape[1] - 1)
    return table[d, h]


def _lookup_daily(table: np.ndarray, doy: np.ndarray) -> np.ndarray:
    d = np.clip(np.asarray(doy, dtype=np.int64) - 1, 0, table.shape[0] - 1)
    return table[d]


def build_l0_baseline(station: str, climatology: StationClimatology | None,
                      obs: Sequence[tuple[int, float | None]],
                      t_run_minute: int, *,
                      offset_days: int = DEFAULT_OFFSET_DAYS,
                      recent_hours: int = DEFAULT_RECENT_HOURS,
                      obs_latency_h: int = 3) -> L0Baseline | None:
    """在决策时刻 `t_run` 上确定 L0 的三个加项（δ、a₀、分季 τ）。

    `obs` 是 `(分钟序号, 温度)` 的**时间升序**序列。因果边界严格取
    `t_cut = t_run − obs_latency_h`：观测链入库有滞后，用更近的"观测"等于假装
    决策时已经知道它。所有加项都只用 `≤ t_cut` 的数据。

    观测不足（< 24 小时）时退回"纯气候态"（δ = a₀ = 0），并如实记录
    `observed_offset_days=0`——不假装拟合过。
    """
    if climatology is None:
        return None
    t_cut = int(t_run_minute) - int(obs_latency_h) * 60
    lo = t_cut - int(offset_days) * 1440
    win = [(m, float(t)) for m, t in obs
           if t is not None and lo < m <= t_cut and math.isfinite(float(t))]
    base = dict(station=station, climatology=climatology,
                tau_hours=climatology.tau_hours,
                climatology_source=climatology.source,
                tau_fitted=climatology.tau_fitted)
    if len(win) < 24:
        return L0Baseline(**base)
    mins = np.array([m for m, _t in win], dtype=np.int64)
    vals = np.array([t for _m, t in win])
    doy = doy_of(mins // 1440)
    hour = (mins % 1440) // 60
    clim = _lookup_hourly(climatology.hourly, doy, hour)
    resid = vals - clim
    ok = np.isfinite(resid)
    if not ok.any():
        return L0Baseline(**base)
    delta = float(np.mean(resid[ok]))
    # 最近 recent_hours 小时的异常（相对 clim + δ）
    recent = [(m, v) for m, v in win if m > t_cut - recent_hours * 60]
    if recent:
        rm = np.array([m for m, _ in recent], dtype=np.int64)
        rv = np.array([v for _, v in recent])
        rdoy = doy_of(rm // 1440)
        rhour = (rm % 1440) // 60
        a0 = float(np.mean(rv - (_lookup_hourly(climatology.hourly, rdoy, rhour)
                                 + delta)))
    else:
        a0 = 0.0
    # 日极值偏移：先按自然日取日最高/最低，再与日极值气候表比
    by_day: dict[int, list[float]] = defaultdict(list)
    for m, v in zip(mins, vals):
        by_day[int(m) // 1440].append(v)
    dev_max: list[float] = []
    dev_min: list[float] = []
    full_days = {d: v for d, v in by_day.items() if len(v) >= 12}
    if full_days:
        dd = doy_of(np.array(sorted(full_days)))
        for i, d in enumerate(sorted(full_days)):
            vs = full_days[d]
            dev_max.append(max(vs) - float(climatology.tmax[dd[i] - 1]))
            dev_min.append(min(vs) - float(climatology.tmin[dd[i] - 1]))
    off_max = float(np.mean(dev_max)) if dev_max else delta
    off_min = float(np.mean(dev_min)) if dev_min else delta
    # 日极值的最近异常：最近一个完整日的日极值相对 (clim_ext + δ_ext)
    a0_max = a0_min = 0.0
    if full_days:
        last = max(full_days)
        vs = full_days[last]
        dd = doy_of(np.array([last]))
        a0_max = float(max(vs) - (float(climatology.tmax[dd[0] - 1]) + off_max))
        a0_min = float(min(vs) - (float(climatology.tmin[dd[0] - 1]) + off_min))
    return L0Baseline(**base, offset_temp=delta, offset_tmax=off_max,
                      offset_tmin=off_min, a0_temp=a0, a0_tmax=a0_max,
                      a0_tmin=a0_min, observed_offset_days=len(full_days))


# ---------------------------------------------------------------- 站点预载
@dataclass
class StationSeries:
    """单站观测的时间升序序列（分钟序号 + 温度 + 降水）。"""

    minute: np.ndarray
    temp: np.ndarray
    rain: np.ndarray
    obs_map: dict = field(default_factory=dict)

    @classmethod
    def from_map(cls, obs_map: Mapping[str, dict]) -> "StationSeries":
        keys = sorted(k for k in obs_map if isinstance(k, str) and len(k) >= 16)
        n = len(keys)
        minute = np.zeros(n, dtype=np.int64)
        temp = np.full(n, np.nan)
        rain = np.full(n, np.nan)
        for i, k in enumerate(keys):
            minute[i] = _minute_of_iso(k)
            rec = obs_map[k]
            t = rec.get("temp")
            r = rec.get("rain")
            if isinstance(t, (int, float)) and math.isfinite(t):
                temp[i] = float(t)
            if isinstance(r, (int, float)) and math.isfinite(r):
                rain[i] = float(r)
        return cls(minute=minute, temp=temp, rain=rain, obs_map=dict(obs_map))

    def pairs(self) -> list[tuple[str, float | None, float | None]]:
        return [(k, (None if not np.isfinite(self.temp[i]) else float(self.temp[i])),
                 (None if not np.isfinite(self.rain[i]) else float(self.rain[i])))
                for i, k in enumerate(sorted(self.obs_map))]

    def temp_series(self) -> list[tuple[int, float | None]]:
        return [(int(m), (None if not np.isfinite(t) else float(t)))
                for m, t in zip(self.minute, self.temp)]


def _minute_of_iso(s: str) -> int:
    d = datetime.fromisoformat(s[:16])
    return (d.toordinal() - 730120) * 1440 + d.hour * 60 + d.minute


__all__ = [
    "DEFAULT_TAU_HOURS", "DEFAULT_CLIM_SMOOTH_DAYS", "DEFAULT_OFFSET_DAYS",
    "DEFAULT_RECENT_HOURS", "RAIN_OCCURRENCE_MM", "TAU_BOUNDS",
    "ERA5LandClimatology", "L0Baseline", "StationClimatology", "StationSeries",
    "build_l0_baseline", "build_obs_climatology", "doy_of", "doy_of_iso",
    "fit_persistence_tau", "persistence_decay",
]

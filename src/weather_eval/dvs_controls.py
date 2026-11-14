"""DVS v6 对照源：给每一个分数标定"它在真实误差尺度上意味着什么"。

**为什么必须内建对照源（第一性原理）**

任何 0–100 的量表都能排出一个名次，但名次本身不携带"这一分有多难"的信息。读者看到
某源 42 分，唯一能判断它值不值的是：**比它高一档的那些分，对应的物理误差到底差多少**。
旧口径靠"标定斜率 λ + 展示增益"把分数拉开，读者反而看不见尺度——那是拿可读性换掉了
可解释性。

DVS v6 的做法是：每次构建都自动生成一组**已知答案的合成源**，让它们与真实源走完全
相同的管道、算完全相同的分数，然后以灰色参考行直接列在榜上。于是：

* `L0（=0）`：与无技巧基准持平——量表零点，自动由管道本身产生；
* `Oracle（=100）`：完美预报——量表顶点，同时是"完美预报得分必须为 100"的公理测试；
* `Oracle ± σ(1/2/3°C)`：给观测叠加已知幅度的高斯噪声，读者立刻看到"0.6°C 的误差
  值多少分"；
* `滞后 3h`：用 `t−3h` 的实况当"预报"，即"什么都不做只把实况往后推"；
* `全报雨 / 全不报雨 / 随机基率`：三个**博弈者**。前两个的降水维必须恰好为 0
  （properness 的定理），第三个的 `V̄+` 底座应趋近 0——若它在真实样本上显著为正，
  说明评分规则可以被"知道基率"套利，需要立刻审计而不是继续发榜。

这些源**不参与排名、不参与 MCS、不写进正式榜位**：它们是标尺，不是选手。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dvs_protocol import HourlyFacts

#: 对照源的键前缀。榜面渲染据此把它们放进"灰色参考行"区块。
CONTROL_PREFIX = "ctl:"

#: Oracle 噪声档位（°C）。三档覆盖"优秀/可用/勉强"三个量级，是读者最需要的换算尺。
ORACLE_SIGMAS = (1.0, 2.0, 3.0)

#: 滞后对照的小时数：实况链入库滞后 P95 ≈ 3h，"滞后 3h 的实况"是免费可得的最强
#: 朴素替代品之一（比持续性略差，因为它还丢了最近 3 小时）。
LAG_HOURS = 3


@dataclass(frozen=True)
class ControlSource:
    """一个对照源的定义：名字 + 它对每个观测值给出的"预报"。"""

    key: str
    label: str
    note: str
    kind: str          # "oracle" | "constant" | "random" | "lag" | "noise"


CONTROLS: tuple[ControlSource, ...] = (
    ControlSource("l0", "L0 无技巧基准", "气候态 + 阻尼持续性（分数零点，恒为 0）",
                  "constant"),
    ControlSource("oracle", "Oracle 完美预报", "直接给出实况（分数顶点，恒为 100）",
                  "oracle"),
    ControlSource("oracle_s1", "Oracle + σ1°C", "实况叠加 1°C 高斯噪声", "noise"),
    ControlSource("oracle_s2", "Oracle + σ2°C", "实况叠加 2°C 高斯噪声", "noise"),
    ControlSource("oracle_s3", "Oracle + σ3°C", "实况叠加 3°C 高斯噪声", "noise"),
    ControlSource("lag3h", "滞后 3h 实况", "用 3 小时前的实况当预报", "lag"),
    ControlSource("always_rain", "全报雨", "每个格点都报有雨（降水维必须为 0）",
                  "constant"),
    ControlSource("never_rain", "全不报雨", "每个格点都报无雨（降水维必须为 0）",
                  "constant"),
    ControlSource("random_rate", "随机基率", "按站点基率抛硬币（V̄⁺ 底座应趋近 0）",
                  "random"),
)

CONTROL_BY_KEY: dict[str, ControlSource] = {c.key: c for c in CONTROLS}


def control_name(key: str) -> str:
    return f"{CONTROL_PREFIX}{key}"


def is_control(model: str) -> bool:
    return str(model).startswith(CONTROL_PREFIX)


def _sigma_of(key: str) -> float:
    return {"oracle_s1": 1.0, "oracle_s2": 2.0, "oracle_s3": 3.0}.get(key, 0.0)


def build_control_facts(hourly: HourlyFacts, *,
                        seed: int = 20261114,
                        controls: Sequence[ControlSource] = CONTROLS) -> HourlyFacts:
    """由**真实配对事实**生成全部对照源的逐小时事实。

    做法：对同一批 `(站, 有效时刻, 批次)` 行复制一份，只把 `temp_fcst` / `rain_fcst`
    换成该对照源的"预报"。这样对照源与真实源**共用同一批观测、同一批时段、同一套
    协议护栏**——分数差异只能来自预报本身，不能来自样本不同。

    随机源用固定种子：同一天的数据必须得到同一份对照分数（可复现性）。
    """
    rng = np.random.default_rng(seed)
    chunks: list[HourlyFacts] = []
    base_ok_t = hourly.temp_ok
    base_ok_r = hourly.rain_ok
    for i, ctl in enumerate(controls):
        if ctl.key == "l0":
            continue        # L0 的分数恒为 0，由管道自身给出（基准误差 = 自身）
        n = hourly.n
        temp_fcst = np.array(hourly.temp_fcst, copy=True)
        rain_fcst = np.array(hourly.rain_fcst, copy=True)
        if ctl.kind == "oracle":
            temp_fcst = hourly.temp_obs.copy()
            rain_fcst = hourly.rain_obs.copy()
        elif ctl.kind == "noise":
            sigma = _sigma_of(ctl.key)
            temp_fcst = hourly.temp_obs + rng.normal(0.0, sigma, n)
            rain_fcst = np.maximum(hourly.rain_obs + rng.normal(0.0, sigma * 0.5, n),
                                   0.0)
        elif ctl.kind == "lag":
            temp_fcst, rain_fcst = _shift_by_hours(hourly, LAG_HOURS)
        elif ctl.kind == "constant":
            if ctl.key == "always_rain":
                rain_fcst = np.full(n, 50.0)
            else:
                rain_fcst = np.zeros(n)
            # 温度维对"全报雨/全不报雨"没有定义，用 L0 基准自身的值无从得知，
            # 因此温度侧保持原值但**标记为无效**——对照源只在它有语义的维度上计分。
            temp_fcst = np.full(n, np.nan)
        elif ctl.kind == "random":
            rain_fcst = (rng.random(n) < _station_rate(hourly)).astype(float) * 50.0
            temp_fcst = np.full(n, np.nan)
        else:  # pragma: no cover - 未知类型由 CONTROLS 穷举保证
            continue
        chunks.append(HourlyFacts(
            run_minute=hourly.run_minute, run_day=hourly.run_day,
            station=hourly.station,
            model=np.full(n, 100 + i, dtype=np.int32),
            valid_day=hourly.valid_day, valid_minute=hourly.valid_minute,
            hour_bjt=hourly.hour_bjt, day_index=hourly.day_index,
            band=hourly.band, lead_hours=hourly.lead_hours,
            temp_obs=hourly.temp_obs, temp_fcst=temp_fcst,
            rain_obs=hourly.rain_obs, rain_fcst=rain_fcst,
            temp_ok=base_ok_t & np.isfinite(temp_fcst),
            rain_ok=base_ok_r & np.isfinite(rain_fcst),
        ))
    if not chunks:
        return hourly.take(np.zeros(hourly.n, dtype=bool))
    out = chunks[0]
    for c in chunks[1:]:
        out = out + c
    return out


def control_model_codes(controls: Sequence[ControlSource] = CONTROLS,
                        offset: int = 100) -> dict[str, int]:
    """对照源 → 模型编码（与 `build_control_facts` 的 100+i 一一对应）。"""
    return {control_name(c.key): offset + i for i, c in enumerate(controls)}


def _shift_by_hours(hourly: HourlyFacts, hours: int) -> tuple[np.ndarray, np.ndarray]:
    """用同一站 `hours` 小时之前的观测值当"预报"。

    严格按 `(站, 有效时刻)` 平移：跨站不借值，缺失保持缺失（绝不插值）。
    """
    key = hourly.station.astype(np.int64) * 10_000_000 + hourly.valid_minute
    offset = key + int(hours) * 60
    lookup = dict(zip(key.tolist(), range(key.size)))
    pos = np.array([lookup.get(int(k), -1) for k in offset.tolist()], dtype=np.int64)
    out_t = np.full(hourly.n, np.nan)
    out_r = np.full(hourly.n, np.nan)
    hit = pos >= 0
    out_t[hit] = hourly.temp_obs[pos[hit]]
    out_r[hit] = hourly.rain_obs[pos[hit]]
    return out_t, out_r


def _station_rate(hourly: HourlyFacts) -> np.ndarray:
    """每行的"该站该级事件基率"（用**观测**估计，与气候决策的真实信息集一致）。

    这是"随机基率"对照源的定义：它只知道"这个站常不常下雨"，不知道今天下不下。
    如果它能拿到显著正分，说明评分规则把"复述气候"误当成了"预报技巧"。
    """
    rate = np.full(hourly.n, 0.0)
    ok = hourly.rain_ok
    if not ok.any():
        return rate
    for sid in np.unique(hourly.station[ok]):
        m = ok & (hourly.station == sid)
        obs = hourly.rain_obs[m]
        wet = float(np.mean(obs >= 1.0)) if obs.size else 0.0
        rate[m] = wet
    return rate


def worst_case_bounds() -> dict:
    """对照源的**已知答案**：用来在 CI 里做自动化验收（见 `tests/test_dvs_axioms.py`）。

    这些不是"参考实现"，而是定理的数值形式：
      * L0 的温度技能恒为 0、降水价值恒为 0；
      * Oracle 的温度技能恒为 1、降水价值恒为 1；
      * 全报雨/全不报雨 的降水价值恒为 0；
      * 随机基率的降水价值数学期望为 0（容忍有限样本正偏置 < 0.015）。
    """
    return {
        control_name("l0"): {"temp": 0.0, "precip": 0.0},
        control_name("oracle"): {"temp": 1.0, "precip": 1.0},
        control_name("always_rain"): {"precip": 0.0},
        control_name("never_rain"): {"precip": 0.0},
        control_name("random_rate"): {"precip_max": 0.015},
    }


__all__ = [
    "CONTROL_BY_KEY", "CONTROL_PREFIX", "CONTROLS", "ControlSource", "LAG_HOURS",
    "ORACLE_SIGMAS", "build_control_facts", "control_model_codes", "control_name",
    "is_control", "worst_case_bounds",
]

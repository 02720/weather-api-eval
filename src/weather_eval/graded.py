"""降水分级指标的 numpy 快速路径（口径以 cyeva 为准，此处只换实现）。

**为什么值得为它单开一个模块**（实测，见 docs/cyeva-call-audit.md）：
分级指标占全部 cyeva 调用的 **70%**（26.65 s / 38.06 s），896 次调用展开成
34,496 次 cyeva 子调用——每个 (级别, 指标) 组合都要把整条序列重新二值化一遍，
再用 `collections.Counter` 逐元素数一遍列联表。同一份样本被扫了 77 遍。

而它的数学非常简单：二值化只有一条区间比较，列联表只有 4 个计数。可加性意味着
**一次向量化扫描就能同时给出全部级别的全部计数**，把 77 次全序列扫描压成
2 × 级别数 次比较。

**为什么不改 cyeva 而在这里另写一份**：cyeva 是指标口径的**权威实现**，保留它
才能让"快速路径与权威口径同值"成为一条可被测试检验的命题（与项目既有的
`temp_curve_metrics` / `stats.temp_core_numpy` 完全同一模式，由
`test_numpy_fast_paths_match_cyeva` 锁定）。本模块从不改写 cyeva，只复刻它，
并且**从 cyeva 自己的配置里读分级阈值**（`cyeva.config.levels.precip`）——阈值
若在上游变化，这里自动跟随，不存在第二份需要手工同步的阈值表。

**复刻的口径细节**（每一条都来自 cyeva 源码，且被对拍测试逐位锁定）：
  1. `source_round_digit`：输入先 `np.round(x, 1)` 再比较（银行家舍入），
     且**舍入发生在剔 NaN 之前**。
     ⚠️ 这里是全套口径里最容易踩错的一处：`calc_precip_*_indicators` 上的装饰器
     写作 `@source_round_digit(2)`，而该装饰器的签名是
     `source_round_digit(series_num=2, digit_num=1)`——那个 `2` 是**要舍入的前几个
     位置参数**，不是小数位数。真正的位数是默认值 **1**。（`level_binarize` 上
     写的是 `@source_round_digit(digit_num=2)`，但它拿到的已经是 1 位小数了，
     再舍入到第 2 位不改变结果。）第一次实现按 2 位写，对拍立刻抓出 200 处
     不一致——这正是"先写对拍脚本"存在的意义。
  2. `drop_nan` 只剔 NaN（`x != x`），**保留 inf**：inf 会被 `>=` 判为"有雨"。
  3. `level_binarize` 的区间判定：`min > 0` 恒真，故为 `(v >= min) & (v <= max)`。
     累积档（`+N`）的 max 是 inf，退化为 `v >= min`。
  4. `"1h"` 没有第 6 档：lev "6"/"+6" 被映射到 "5"（cyeva 的兼容分支）。
  5. 列联计数用 Python int（`Counter(...)[True]`），除零即 `ZeroDivisionError`
     → 由 `fix_zero_division` 折算为 NaN。这里用等价的"分母为 0 → NaN"判定，
     绝不用 numpy 的 inf（`0/0` 的 numpy 语义）糊过去——两者对外都是 None，
     但只有前者的 NaN 会走到 `_r()` 的 None 分支，口径必须显式一致。
  6. `result_round_digit(2)`：结果 `round(v, 2)`（Python 内置 round，同样是
     银行家舍入）。注意 cyeva 的 `if result:` 对 `0.0` 直接返回不取整，而
     `round(0.0, 2) == 0.0`，等价。
"""
from __future__ import annotations

import math

import numpy as np

# 分级阈值**只此一处**，且直接取自 cyeva 的配置模块：上游改了阈值，这里跟着改，
# 不存在第二份需要人工同步的表（语义债 #9 的同款教训）。
from cyeva.config.levels.precip import ACC_PRECIP_LEVELS, PRECIP_LEVELS

GRADED_KEYS = ("acc", "pod", "far", "miss", "ts", "ets", "bias")


def _level_config(kind: str, lev: str) -> tuple[float, float]:
    """(min, max)：按 cyeva 的规则取该档的区间。"""
    kind = kind.lower()
    # cyeva 的兼容分支：1h 没有第 6 档，传 "6"/"+6" 一律降级到第 5 档
    if kind == "1h" and lev in ("6", "+6"):
        lev = lev.replace("6", "5")
    table = ACC_PRECIP_LEVELS if lev.startswith("+") else PRECIP_LEVELS
    lev_id = int(lev.replace("+", ""))
    interval = table[kind][lev_id]
    return float(interval["min"]), float(interval["max"])


def _div(a: float, b: float) -> float:
    """cyeva 的 `fix_zero_division`：分母为 0 时是 ZeroDivisionError → NaN。"""
    return math.nan if b == 0 else a / b


def _nan_round(v: float) -> float:
    """cyeva 的 `result_round_digit(2)`：0.0 原样返回（等价于 round 后不变）。"""
    if v == 0.0:
        return 0.0
    return round(v, 2)


def graded_counts(obs: np.ndarray, fcst: np.ndarray, kind: str,
                  levels: tuple[str, ...]) -> dict[str, tuple[int, int, int, int, int]]:
    """一次扫描给出所有级别的列联计数 (hits, misses, false_alarms, correct_rejects, total)。

    返回的是整数计数——**指标口径的最小充分统计量**。把它单独导出，是为了让
    对拍测试能直接比对"cyeva 的 Counter 与 numpy 的 count_nonzero 是否数出同一张
    表"，而不是只比对最终浮点（浮点相同可能是两处错误互相抵消）。
    """
    # 口径 1：先舍入到 1 位小数（见模块 docstring 的踩坑记录），且舍入先于剔 NaN
    o = np.round(np.asarray(obs, dtype=float), 1)
    f = np.round(np.asarray(fcst, dtype=float), 1)
    # 口径 2：只剔 NaN（x != x 判定），保留 inf
    keep = ~(o != o) & ~(f != f)
    o, f = o[keep], f[keep]
    out: dict[str, tuple[int, int, int, int, int]] = {}
    for lev in levels:
        mn, mx = _level_config(kind, lev)
        # 口径 3：min > 0 恒真 ⇒ (v >= min) & (v <= max)
        ob = (o >= mn) & (o <= mx)
        fb = (f >= mn) & (f <= mx)
        h = int(np.count_nonzero(ob & fb))
        mi = int(np.count_nonzero(ob & ~fb))
        fa = int(np.count_nonzero(~ob & fb))
        c = int(np.count_nonzero(~ob & ~fb))
        out[lev] = (h, mi, fa, c, h + mi + fa + c)
    return out


def graded_metrics_from_counts(h: int, mi: int, fa: int, c: int,
                               total: int) -> dict[str, float]:
    """列联计数 → 7 项分级指标（公式与 cyeva.core.statistic 逐条对应）。"""
    href = _div((h + mi) * (h + fa), total) if total else math.nan
    den_ets = (h + fa + mi - href) if not math.isnan(href) else math.nan
    return {
        "acc": _nan_round(_div(h + c, total) * 100.0),
        "pod": _nan_round(_div(h, h + mi) * 100.0),
        "far": _nan_round(_div(fa, h + fa) * 100.0),
        "miss": _nan_round(_div(mi, h + mi) * 100.0),
        "ts": _nan_round(_div(h, h + fa + mi)),
        "ets": _nan_round(_div(h - href, den_ets)) if not math.isnan(den_ets) else math.nan,
        "bias": _nan_round(_div(h + fa, h + mi)),
    }


def precip_graded_metrics(obs, fcst, kind: str,
                          levels: tuple[str, ...]) -> dict[str, dict[str, float] | None]:
    """分级指标：`{级别: {acc, pod, far, miss, ts, ets, bias}}`，NaN 处记 None。

    与 `PrecipitationComparison.calc_*(kind=, lev=)` 同口径；对拍脚本
    `scripts/parity_graded.py` 在真实数据上逐格比对，测试
    `tests/test_graded_parity.py` 把这条不变量钉在 CI 里。
    """
    counts = graded_counts(obs, fcst, kind, levels)
    out: dict[str, dict[str, float] | None] = {}
    for lev, (h, mi, fa, c, total) in counts.items():
        m = graded_metrics_from_counts(h, mi, fa, c, total)
        out[lev] = ({k: (None if (v is None or math.isnan(v) or math.isinf(v)) else v)
                     for k, v in m.items()}
                    if total > 0 else None)
    return out

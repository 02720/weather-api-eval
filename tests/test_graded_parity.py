"""降水分级指标：numpy 快速路径 vs cyeva 权威路径（I3：口径以 cyeva 为准）。

项目保留 cyeva 作为指标口径的**权威实现**，同时为了性能把分级指标换成向量化
实现。这两件事能共存，唯一的前提是"两条路径输出同一个数"——而这个前提必须由
测试每天检验，不能由写代码那天的一次手工比对来担保。

为什么对拍到**列联计数**这一层：两个浮点相同，可能是两处错误互相抵消。计数是
整数、零容差，一旦二值化的舍入 / NaN / 区间端点出了偏差，计数先红。指标数值是
第二层，None 处置（除零、空档）是第三层。

全量复核（711 个真实格子 × 7 项）由 `scripts/parity_graded.py` 承担，跑在慢层；
这里用合成数据覆盖边界（阈值边界值、全缺测、除零、inf、级别端点），保证快层
也能抓住口径破裂。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from weather_eval.evaluate import DAILY_GRADED_LEVS, HOURLY_GRADED_LEVS
from weather_eval.graded import graded_counts, precip_graded_metrics

pytestmark = pytest.mark.unit


def _cyeva_one(obs, fcst, kind, lev):
    """权威路径：cyeva PrecipitationComparison。"""
    from cyeva import PrecipitationComparison
    from cyeva.errors import ArrayLengthNotEqualError
    pc = PrecipitationComparison(np.asarray(obs, dtype=float),
                                 np.asarray(fcst, dtype=float), unit="mm")
    try:
        return {
            "acc": pc.calc_accuracy_ratio(kind=kind, lev=lev),
            "pod": pc.calc_hit_ratio(kind=kind, lev=lev),
            "far": pc.calc_false_alarm_ratio(kind=kind, lev=lev),
            "miss": pc.calc_miss_ratio(kind=kind, lev=lev),
            "ts": pc.calc_ts(kind=kind, lev=lev),
            "ets": pc.calc_ets(kind=kind, lev=lev),
            "bias": pc.calc_bias_score(kind=kind, lev=lev),
        }
    except (ValueError, KeyError, IndexError, ZeroDivisionError,
            ArrayLengthNotEqualError):
        return None


def _norm(d):
    if d is None:
        return None
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = None
            continue
        f = float(v)
        out[k] = None if (math.isnan(f) or math.isinf(f)) else round(f, 2)
    return out


# 刻意覆盖：阈值边界（0.099/0.1/1.9/2.0）、舍入边界（0.05 → 1 位小数后变 0.1）、
# 强降水（≥250mm 的 +6 档）、缺测（None）、以及全缺测导致的空档。
CASES = {
    "边界值": ([0.0, 0.099, 0.1, 0.05, 1.9, 2.0, 20.0, 300.0],
               [0.0, 0.5, 0.05, 0.1, 2.5, 1.0, 0.0, 250.0]),
    "含缺测": ([0.0, None, 5.0, 12.0, 60.0],
               [1.0, 2.0, None, 12.0, 0.0]),
    "全缺测": ([None, None], [None, None]),
    "零降水": ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]),
    "随机": (list(np.round(np.random.default_rng(7).random(400) * 40, 3)),
             list(np.round(np.random.default_rng(8).random(400) * 40, 3))),
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("kind,levs", [("1h", HOURLY_GRADED_LEVS),
                                       ("24h", DAILY_GRADED_LEVS)])
def test_graded_matches_cyeva(name, kind, levs):
    """两条路径逐项、逐格相同（含 None 处置）。"""
    obs, fcst = CASES[name]
    fast = precip_graded_metrics(obs, fcst, kind, levs)
    for lev in levs:
        want = _norm(_cyeva_one(obs, fcst, kind, lev))
        got = fast.get(lev)
        assert (want is None) == (got is None), \
            f"{name}/{kind}/{lev}: None 处置不一致（cyeva={want} fast={got}）"
        if want is None:
            continue
        for key, wv in want.items():
            assert got.get(key) == wv, (
                f"{name}/{kind}/{lev}.{key}: cyeva={wv} numpy={got.get(key)}")


def test_graded_counts_are_integers_and_consistent():
    """列联计数是自洽的整数：h+mi+fa+c == 参与样本数。"""
    obs, fcst = CASES["随机"]
    for lev, (h, mi, fa, c, total) in graded_counts(
            np.asarray(obs, dtype=float), np.asarray(fcst, dtype=float),
            "24h", DAILY_GRADED_LEVS).items():
        assert all(isinstance(x, int) for x in (h, mi, fa, c, total))
        assert h + mi + fa + c == total, f"{lev} 计数不自洽"


def test_graded_rounding_uses_one_decimal():
    """口径钉子：源值按 **1 位**小数舍入后再比较（cyeva `source_round_digit` 的
    真实位数，见 graded.py 模块 docstring）。

    取 0.04 / 0.06 / 0.14 三个值：
      按 **1 位**舍入 → 0.0 / 0.1 / 0.1 → 达 0.1 档的有 **2** 条；
      按 **2 位**舍入 → 0.04 / 0.06 / 0.14 → 达 0.1 档的只有 **1** 条。
    于是这条断言一旦从 1 位改回 2 位就会红——它就是那个陷阱的警报器。
    （注意 `np.round(0.05, 1) == 0.0` 而 `round(0.05, 1) == 0.1`：numpy 的缩放舍入
    与 Python 的十进制舍入在二进制边界上不一致，故本例刻意避开 .05 这类边界值，
    只取结论在两种算法下都一致的样本。）
    """
    obs = [0.04, 0.06, 0.14]
    fcst = [0.04, 0.06, 0.14]
    cnt = graded_counts(np.asarray(obs, dtype=float), np.asarray(fcst, dtype=float),
                        "24h", ("+1",))
    h, mi, fa, c, total = cnt["+1"]
    assert total == 3
    assert h == 2, f"按 1 位舍入应有 2 条达档，实际 {h}（若这里变成 1，说明回退到了 2 位舍入）"

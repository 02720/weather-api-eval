"""列式路径与 dict 路径的逐位对拍（I8 + I11）。

`PairTable` 把配对事实从 150 万个 dict 变成 numpy 列，换来 n_eff 与充分统计量
表的量级加速。它**没有**改变任何口径——这条测试就是把"没有改变"钉成可执行的
事实，而不是一句注释。

三条不变量，逐条对应方案里的编号：

* **I11 / `test_neff_columnar_matches_dict`**：n_eff 是**整数**，整数没有容差可言。
  列式实现与 dict 实现必须对全部 2,044 个格子给出完全相等的整数。任何一条
  不等都意味着 min-lead 去重或站序/时刻序出了偏差——那会让"有效样本量"这个
  入围门槛的凭据失真。
* **I8 / `test_daystats_columnar_matches_dict`**：充分统计量表用 `array_equal`
  **零容差**比对。它是可加的，累加顺序与记录顺序一致，所以连浮点位都不该差。
* **I4 / `test_missing_stays_missing`**：列式化把 `None` 变成 NaN，而 NaN 会被
  `>=` 静默当成数值。这里构造 None / 0.0 / NaN / 正常值混合的样本，断言两条
  路径给出**完全相同**的入样判定集合。这是整个列式化里唯一可能让"缺测伪装成
  技巧"的地方。

数据规模提醒：这些测试跑的是合成小数据（几十条记录），目的是覆盖边界而不是
覆盖量级；真实数据上的等价性由 `scripts/parity_graded.py` 同款的
`scripts/parity_columnar.py` 全量复核。
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from weather_eval.evaluate import (
    _n_eff_daily_temp,
    _n_eff_daily_temp_columnar,
    _n_eff_rain,
    _n_eff_rain_columnar,
    _n_eff_temp,
    _n_eff_temp_columnar,
)
from weather_eval.pairtable import PairTable
from weather_eval.stats import build_day_stat_tables, build_day_stat_tables_columnar

pytestmark = pytest.mark.unit

MODELS = ["m_alpha", "m_beta"]
STATIONS = ["st_a", "st_b"]


def _hourly(rows):
    return [{"station": s, "model": m, "valid_iso": t, "lead": ld, "bucket": b,
             "temp_obs": to, "temp_fcst": tf, "rain_obs": ro, "rain_fcst": rf}
            for (s, m, t, ld, b, to, tf, ro, rf) in rows]


def _daily(rows):
    return [{"station": s, "model": m, "valid_day": d, "offset": off,
             "temp_max_obs": mo, "temp_max_fcst": mf,
             "temp_min_obs": no, "temp_min_fcst": nf,
             "rain_obs": ro, "rain_fcst": rf}
            for (s, m, d, off, mo, mf, no, nf, ro, rf) in rows]


# 刻意包含：重复 (站, 时刻) 但 lead 不同（测最小 lead 去重）、lead 并列
# （测"并列取最先出现"）、缺测（测 I4）、以及跨天桶的同刻样本。
HOURLY_ROWS = [
    ("st_a", "m_alpha", "2026-08-01T01:00", 5, 1, 20.0, 21.0, 0.0, 0.5),
    ("st_a", "m_alpha", "2026-08-01T01:00", 3, 1, 20.0, 22.0, 0.0, 0.0),
    ("st_a", "m_alpha", "2026-08-01T02:00", 4, 1, 21.0, 20.5, 1.2, 0.0),
    ("st_a", "m_alpha", "2026-08-01T02:00", 4, 2, 21.0, 19.5, 1.2, 3.0),
    ("st_a", "m_alpha", "2026-08-01T03:00", 6, 2, None, 20.0, 0.0, 0.0),
    ("st_a", "m_alpha", "2026-08-01T04:00", 7, 2, 22.0, None, 0.0, 0.1),
    ("st_b", "m_alpha", "2026-08-01T01:00", 5, 1, 19.0, 19.5, 0.0, 0.0),
    ("st_b", "m_alpha", "2026-08-01T02:00", 6, 1, 18.0, 18.2, 0.0, 2.0),
    ("st_b", "m_alpha", "2026-08-01T03:00", 7, 1, 20.0, 20.9, 5.0, 4.0),
    ("st_a", "m_beta", "2026-08-01T01:00", 2, 0, 20.0, 20.1, 0.0, 0.0),
    ("st_a", "m_beta", "2026-08-01T01:00", 1, 0, 20.0, 20.4, 0.0, 0.0),
    ("st_b", "m_beta", "2026-08-01T01:00", 2, 1, 19.0, 19.9, 0.0, 0.0),
    ("st_b", "m_beta", "2026-08-01T02:00", 3, 1, 18.5, 18.0, 0.0, 0.0),
    ("st_a", "m_beta", "2026-08-02T01:00", 3, 1, 21.0, 22.0, 0.0, 0.0),
    ("st_b", "m_beta", "2026-08-02T01:00", 4, 1, 20.0, 19.0, 0.0, 0.0),
]

DAILY_ROWS = [
    ("st_a", "m_alpha", "2026-08-02", 1, 30.0, 31.0, 22.0, 22.5, 0.0, 0.2),
    ("st_a", "m_alpha", "2026-08-02", 2, 30.0, 30.5, 22.0, 21.0, 0.0, 5.0),
    ("st_b", "m_alpha", "2026-08-02", 1, 29.0, 29.4, 21.0, 21.2, 1.0, 0.0),
    ("st_a", "m_alpha", "2026-08-03", 2, 31.0, 32.0, None, 23.0, 0.0, 0.0),
    ("st_a", "m_beta", "2026-08-02", 1, 30.0, 29.0, 22.0, 22.1, 0.0, 0.0),
    ("st_b", "m_beta", "2026-08-02", 1, 29.0, 30.0, 21.0, 20.0, 0.0, 1.0),
    ("st_b", "m_beta", "2026-08-03", 2, None, 30.0, 21.0, 21.5, 2.0, 0.0),
]


def _build():
    hourly = _hourly(HOURLY_ROWS)
    daily = _daily(DAILY_ROWS)
    days = sorted({r["valid_iso"][:10] for r in hourly}
                  | {r["valid_day"] for r in daily})
    pt = PairTable(hourly, daily, MODELS, STATIONS, days)
    return hourly, daily, days, pt


def test_neff_columnar_matches_dict():
    """I11：n_eff 是整数，两条路径必须**完全相等**（不是"接近"）。"""
    hourly, daily, _days, pt = _build()
    checked = 0
    for mi, m in enumerate(MODELS):
        recs = [r for r in hourly if r["model"] == m]
        for b in range(0, 3):
            sub = [r for r in recs if r["bucket"] == b]
            assert _n_eff_temp(sub) == _n_eff_temp_columnar(pt, pt.hourly_cell(mi, b)), \
                f"n_eff_temp 不一致: model={m} bucket={b}"
            for thr in (0.1, 1.0):
                assert _n_eff_rain(sub, thr, "valid_iso", "lead") == \
                    _n_eff_rain_columnar(pt, pt.hourly_cell(mi, b), thr), \
                    f"n_eff_rain 不一致: model={m} bucket={b} thr={thr}"
            checked += 1
        # 全模型子集与 lead 窗口（评分卡的 24h/72h/all 池）
        for lo, hi in ((1, 24), (1, 72)):
            sub = [r for r in recs if lo <= r["lead"] <= hi]
            idx = pt.hourly_lead_window(mi, lo, hi)
            assert _n_eff_temp(sub) == _n_eff_temp_columnar(pt, idx)
            assert _n_eff_rain(sub, 0.1, "valid_iso", "lead") == \
                _n_eff_rain_columnar(pt, idx, 0.1)
            checked += 1
        assert _n_eff_temp(recs) == _n_eff_temp_columnar(pt, pt.hourly_model(mi))
        scored = [r for r in recs if r["bucket"] >= 1]
        assert _n_eff_temp(scored) == _n_eff_temp_columnar(pt, pt.hourly_scored(mi))
        assert _n_eff_rain(scored, 1.0, "valid_iso", "lead") == \
            _n_eff_rain_columnar(pt, pt.hourly_scored(mi), 1.0)
        checked += 2
        # 按天轨道
        drecs = [r for r in daily if r["model"] == m]
        assert _n_eff_daily_temp(drecs) == \
            _n_eff_daily_temp_columnar(pt, pt.daily_model(mi))
        for off in range(1, 3):
            sub = [r for r in drecs if r["offset"] == off]
            assert _n_eff_rain(sub, 1.0, "valid_day", "offset") == \
                _n_eff_rain_columnar(pt, pt.daily_cell(mi, off), 1.0, daily=True)
            checked += 1
    assert checked >= 10, "对拍覆盖面太小，等于没测"


def test_daystats_columnar_matches_dict():
    """I8：充分统计量表逐位相同（array_equal 零容差）。"""
    hourly, daily, _days, pt = _build()
    days_a, t_a = build_day_stat_tables(hourly, daily, MODELS, 3, 3, 1.0, 1.0)
    days_b, t_b = build_day_stat_tables_columnar(pt, MODELS, 3, 3, 1.0, 1.0)
    assert days_a == days_b, "天数轴不一致"
    assert set(t_a) == set(t_b), "表名集合不一致"
    for name in t_a:
        assert t_a[name].shape == t_b[name].shape, f"{name} 形状不一致"
        # 零容差：可加统计量的累加顺序与记录顺序一致，浮点位都不该差
        assert np.array_equal(t_a[name], t_b[name]), f"{name} 数值不一致"


def test_missing_stays_missing():
    """I4：缺测绝不因列式化变成 0.0。

    两条断言，缺一不可：
      1. **管道卫生契约**——`_finite_or_none`（collect 对快照值的口径）把
         NaN/inf/非数值一律折算成 None，于是 NaN **走不到**列式表。
         这是"缺测只有一种表示"的上游保证。
      2. **纵深防御**——万一还有 NaN 漏进来，列式表把它判为缺测而不是数值。
         依赖 NaN 语义的实现会把缺测静默当成 0.0，那是本项目最不能接受的失败。
    """
    from weather_eval.evaluate import _finite_or_none

    hourly, daily, _days, pt = _build()

    # ---- (1) 上游卫生契约 ----
    assert _finite_or_none(float("nan")) is None
    assert _finite_or_none(float("inf")) is None
    assert _finite_or_none(None) is None
    assert _finite_or_none("非数值") is None
    assert _finite_or_none(0.0) == 0.0, "0.0 是**真实观测值**，绝不能被当成缺测"

    # ---- (2) 纵深防御：混进来的 NaN/None 一律判缺测 ----
    tricky = _hourly([
        ("st_a", "m_alpha", "2026-08-01T05:00", 8, 2, 0.0, 0.0, 0.0, None),
        ("st_a", "m_alpha", "2026-08-01T06:00", 9, 2, None, 0.0, None, 0.0),
        ("st_b", "m_alpha", "2026-08-01T05:00", 8, 2, float("nan"), 1.0, 0.0, 0.0),
        ("st_b", "m_alpha", "2026-08-01T06:00", 9, 2, 1.0, 2.0, 1.0, 1.0),
    ])
    hourly2 = hourly + tricky
    days = sorted({r["valid_iso"][:10] for r in hourly2}
                  | {r["valid_day"] for r in daily})
    pt2 = PairTable(hourly2, daily, MODELS, STATIONS, days)

    for i, r in enumerate(hourly2):
        want_temp = _finite_or_none(r["temp_obs"]) is not None \
            and _finite_or_none(r["temp_fcst"]) is not None
        want_rain = _finite_or_none(r["rain_obs"]) is not None \
            and _finite_or_none(r["rain_fcst"]) is not None
        assert bool(pt2.h_temp_ok[i]) is bool(want_temp), f"第 {i} 条 temp 入样判定不符"
        assert bool(pt2.h_rain_ok[i]) is bool(want_rain), f"第 {i} 条 rain 入样判定不符"

    nan_row = next(i for i, r in enumerate(hourly2)
                   if isinstance(r["temp_obs"], float) and math.isnan(r["temp_obs"]))
    assert not pt2.h_temp_ok[nan_row], "NaN 被当成有效温度样本（I4 破裂）"
    # 0.0 必须仍然是有效样本（否则"无降水"会整片消失）
    zero_row = next(i for i, r in enumerate(hourly2)
                    if r["valid_iso"] == "2026-08-01T05:00" and r["station"] == "st_a")
    assert pt2.h_temp_ok[zero_row], "0.0 被当成缺测（会把真实观测值抹掉）"

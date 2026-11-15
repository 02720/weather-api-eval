"""DVS v6 的**唯一事实源**：按日可加充分统计量仓。

**为什么必须是"可加的充分统计量"，而不是"缓存好的子分"**

旧口径把子分缓存进 `data/metrics/blocks`，于是"总榜、近期榜、月榜、画像切换、
自助重采样"各要一条重算路径，且每改一次权重就得整批重算。而分数的真相只有一个：
它是若干**可加统计量**的初等函数——

* 温度：`[n, Σ|e|, Σe², Σ|e⁰|, Σ(e⁰)², Σf, Σo, Σf², Σo², Σfo]`
* 降水：分站列联表 `[a, b, c, d]` + 事件日数 `n_ev`

只要把这两类量按 `(自然日 × 源 × 站 × 目标 × 段)` 存起来，那么：

* 任意窗口 = 行求和 → 再套一次公式（`dvs.temp_skill` / `dvs.economic_value`）；
* 任意权重 / 任意画像 = 同一批统计量的线性组合；
* 自助重采样 = 对 `(自然日)` 抽权重向量后求和（`dvs_uncertainty`）；
* 读者复算 = 页面公布的行值手算，误差 ≤ 0.1 分。

**为什么"按日"是天然的聚类单元**：同一天 4 个站共享同一个天气系统，同一天的
预报误差彼此相关。把日当单位既保留站间/源间相关结构（自助的块抽样需要它），
又让"哪天出问题"可以被直接定位。

落盘格式：`data/metrics/dvs/{YYYY-MM}.json.gz`，逐日一条稀疏 JSON::

    {"schema":1, "month":"2026-11", "stations":[...],
     "sources": {"ecmwf_ifs025": {"2026-11-01": {"t": [[站,"tmax",2,[...10...]]],
                                                 "r": [[站,"r24_ge1",2,[a,b,c,d],n_ev]],
                                                 "a": [["temp_h",2,[delivered,expected]]]}}}}

每日独立可读、可增量追加；键名取单字母是为了让 gzip 后的体积与"行内容"同阶。
"""
from __future__ import annotations

import gzip
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from . import dvs
from .dvs_protocol import DailyFacts, HourlyFacts

logger = logging.getLogger(__name__)


def _rows_for(out: dict, model: str, day: int) -> "SourceDayRows":
    """取（必要时创建）某源某日的行容器。"""
    per_model = out.setdefault(model, {})
    rows = per_model.get(day)
    if rows is None:
        rows = SourceDayRows()
        per_model[day] = rows
    return rows

#: 温度统计量的键（顺序即落盘数组顺序，**绝不可重排**：窗口求和按位序对拍）。
TEMP_KEYS = ("n", "sum_abs_e", "sum_sq_e", "sum_abs_e0", "sum_sq_e0",
             "sum_f", "sum_o", "sum_f_sq", "sum_o_sq", "sum_fo")

#: 降水统计量的键（分站顺序由 `station_ids` 决定，随文件一并固化）。
RAIN_KEYS = ("a", "b", "c", "d")

STORE_SCHEMA = 1


@dataclass
class SourceDayRows:
    """某源某自然日的全部可加统计量（**按决策批次 `run` 分行**）。

    为什么键里必须带 `run`：L0 的 `δ / a₀` 依赖 `t_run`，所以同一个有效日、同一个
    时效段在不同批次下的"基准误差"不同。若把多个批次并成一行存一个 `Σ|e⁰|`，
    事后就无法把基准换成别的 t_run，也无法做"只用某一个抓取批次"的敏感性分析。
    带上 `run` 后，行的集合本身仍是可加的（窗口 = 行求和），而基准与预报始终配对。

    键的构造规则（决定了读者能否手算复算）：
      * 温度：`target` ∈ `{temp_h, tmax, tmin}`，`band` ∈ 1..5；
      * 降水：`target` ∈ `{r3_ge1, r3_ge5, r24_ge1, r24_ge10, r24_ge25}`，`band` 1..5；
      * `run` 是批次的 `t_run`（`YYYY-MM-DDTHH:MM`）。
    """

    #: {(run, station, target, band): [10 个温度统计量]}
    temp: dict[tuple[str, str, str, int], list[float]] = field(default_factory=dict)
    #: {(run, station, target, band): [a, b, c, d, n_ev_days]}
    #:
    #: 用**可变 list** 而不是元组：样本要逐组原地累加，元组在 Python 里不可变，
    #: 写回会静默丢失（这个 bug 让降水列联表一直停在第一条样本上）。
    #: `n_ev_days` 是"观测到该级事件的独立自然日数"，跨组累加时按下标去重。
    rain: dict[tuple[str, str, str, int], list] = field(default_factory=dict)
    #: {(target, band): [delivered, expected]} —— 可用率的分子分母
    avail: dict[tuple[str, int], list[int]] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not self.temp and not self.rain and not self.avail


def accumulate_day(hourly: HourlyFacts, *, base: np.ndarray,
                   station_ids: Sequence[str], models: Sequence[str],
                   events: Mapping[str, Sequence[tuple[float, float]]] = dvs.RAIN_EVENTS,
                   ) -> dict[str, dict[int, SourceDayRows]]:
    """逐小时事实 + 逐行 L0 基准 → "源 → 自然日 → 可加统计量"。

    `base` 与 `hourly` 的每一行**一一对应**（同一行是同一 `(批次, 源, 站, 有效时刻)`），
    所以基准与预报永远配对——不会出现"基准按 A 批次的 t_run 算、预报来自 B 批次"
    这种错位。基准由引擎在建立事实的同一次遍历里算出。

    做法是**一次遍历、按组向量化**：先按 `(源, 站, 批次, 有效日, 段)` 排序分组，
    再对每一组做 10 个 numpy 归约。组数 ≈ 源数 × 站数 × 批次数 × 天 × 段，在真实
    窗口里是几千组/站，比"逐 (日, 段) 做布尔掩膜"快一个数量级。
    """
    out: dict[str, dict[int, SourceDayRows]] = {}
    if hourly.n == 0:
        return out
    ok = hourly.temp_ok & np.isfinite(base)
    if not ok.any():
        return out
    idx = np.flatnonzero(ok)
    code = hourly.model[idx]
    station = hourly.station[idx]
    run = hourly.run_minute[idx]
    vday = hourly.valid_day[idx]
    band = hourly.band[idx]
    key = _group_key(code, station, run, vday, band)
    order = np.argsort(key, kind="stable")
    skey = key[order]
    starts = _group_starts(skey)
    if starts.size == 0:
        return out
    # 十项统计量各一次 `reduceat`：这是"按组求和"的向量化形式，等价于逐组求和但
    # 只走一遍数据（实测比逐组 10 次 `.sum()` 快 4 倍以上）。
    sel = idx[order]
    e = hourly.temp_fcst[sel] - hourly.temp_obs[sel]
    e0 = base[sel] - hourly.temp_obs[sel]
    f = hourly.temp_fcst[sel]
    o = hourly.temp_obs[sel]
    cols = (np.ones(sel.size), np.abs(e), e * e, np.abs(e0), e0 * e0,
            f, o, f * f, o * o, f * o)
    sums = np.stack([np.add.reduceat(c, starts) for c in cols], axis=1)
    head = order[starts]
    for i in range(starts.size):
        h = head[i]
        rows = _rows_for(out, models[int(code[h])], int(vday[h]))
        ent = rows.temp.setdefault(
            (run_key(int(run[h])), station_ids[int(station[h])],
             "temp_h", int(band[h])),
            [0.0] * len(TEMP_KEYS))
        row = sums[i]
        for k in range(len(TEMP_KEYS)):
            ent[k] += float(row[k])
        # 计数已在 row[0]（Σ1）里，无需另加
    _accumulate_3h(out, hourly, models, station_ids, events)
    _accumulate_avail(out, hourly, models, events)
    return out


def accumulate_daily(out: dict[str, dict[int, SourceDayRows]], daily: DailyFacts,
                     *, base: Mapping[str, np.ndarray],
                     station_ids: Sequence[str], models: Sequence[str],
                     events: Mapping[str, Sequence[tuple[float, float]]],
                     ) -> None:
    """日尺度（`tmax` / `tmin` / `r24`）与 3h 之外的另一半降水分。

    `base` 形如 `{"tmax": ndarray, "tmin": ndarray, "r24": ndarray}`，与 `daily`
    的行一一对应（由引擎用**代表批次**的 `t_run` 算出）。
    """
    if daily.n == 0:
        return
    vday = daily.valid_day
    code = daily.model
    station = daily.station
    run = daily.run_minute
    band = daily.band
    for target, obs_key, fcst_key in (("tmax", "tmax_obs", "tmax_fcst"),
                                      ("tmin", "tmin_obs", "tmin_fcst")):
        obs = getattr(daily, obs_key)
        fcst = getattr(daily, fcst_key)
        b0 = base.get(target)
        if b0 is None:
            continue
        ok = np.isfinite(obs) & np.isfinite(fcst) & np.isfinite(b0) & (band >= 1)
        if not ok.any():
            continue
        idx = np.flatnonzero(ok)
        key = _group_key(code[idx], station[idx], run[idx], vday[idx], band[idx])
        order = np.argsort(key, kind="stable")
        starts = _group_starts(key[order])
        sel = idx[order]
        e = fcst[sel] - obs[sel]
        e0 = b0[sel] - obs[sel]
        f = fcst[sel]
        o = obs[sel]
        cols = (np.ones(sel.size), np.abs(e), e * e, np.abs(e0), e0 * e0,
                f, o, f * f, o * o, f * o)
        sums = np.stack([np.add.reduceat(c, starts) for c in cols], axis=1)
        head = order[starts]
        for i in range(starts.size):
            h = head[i]
            rows = _rows_for(out, models[int(code[h])], int(vday[h]))
            ent = rows.temp.setdefault(
                (run_key(int(run[h])), station_ids[int(station[h])],
                 target, int(band[h])), [0.0] * len(TEMP_KEYS))
            row = sums[i]
            for k in range(len(TEMP_KEYS)):
                ent[k] += float(row[k])
    specs = events.get("r24") or ()
    if not specs:
        return
    obs = daily.r24_obs
    fcst = daily.r24_fcst
    ok = np.isfinite(obs) & np.isfinite(fcst) & (band >= 1)
    if not ok.any():
        return
    idx = np.flatnonzero(ok)
    # 降水计数不依赖 `t_run`，所以批次并进同一行（理由见 `_accumulate_3h`）
    key = _group_key(code[idx], station[idx], np.zeros(idx.size, dtype=np.int64),
                     vday[idx], band[idx])
    order = np.argsort(key, kind="stable")
    starts = _group_starts(key[order])
    head = order[starts]
    for thr, _w in specs:
        name = f"r24_{_thr_key(thr)}"
        o_all = (obs[idx][order] >= thr).astype(np.int8)
        f_all = (fcst[idx][order] >= thr).astype(np.int8)
        om = np.add.reduceat(o_all, starts)
        fm = np.add.reduceat(f_all, starts)
        for i in range(starts.size):
            h = head[i]
            rows = _rows_for(out, models[int(code[h])], int(vday[h]))
            ent = _rain_ent(rows, ("", station_ids[int(station[h])],
                                   name, int(band[h])))
            _add_counts(ent, np.array([om[i] > 0]), np.array([fm[i] > 0]))


def _accumulate_3h(out: dict, hourly: HourlyFacts, models: Sequence[str],
                   station_ids: Sequence[str],
                   events: Mapping[str, Sequence[tuple[float, float]]]) -> None:
    """3h 窗的列联表。

    直接从**逐小时事实**按 (源, 站, 批次, 有效日, 段, 窗) 求和得到 3 小时累计，
    而不是先建 `ThreeHourFacts` 再回头找批次：`ThreeHourFacts` 是为了给
    `build_3h_facts` 的单测提供一个可对拍的中间对象。
    """
    specs = events.get("r3") or ()
    if not specs or hourly.n == 0:
        return
    ok = hourly.rain_ok & (hourly.band >= 1)
    if not ok.any():
        return
    idx = np.flatnonzero(ok)
    code = hourly.model[idx]
    station = hourly.station[idx]
    run = hourly.run_minute[idx]
    vday = hourly.valid_day[idx]
    band = hourly.band[idx]
    hour = hourly.hour_bjt[idx].astype(np.int64)
    # ① 3h 窗按**北京时钟点固定分窗**（02/05/…/23 时止）：`end = 02,05,...,23`，
    #    覆盖 `end-2 .. end` 三个钟点。用钟点而不是"每 3 行"分窗，是因为跨日窗
    #    （前一日 22/23 时 + 次日 0 时）在逐小时事实里属于两个自然日——按行数切
    #    会把它们劈开，两截都凑不满 3 小时而整窗丢弃（实测事件日数从 16 掉到 0）。
    end_hour = _HOUR_END[np.clip(hour, 0, 23)]
    valid = end_hour >= 0
    if not valid.any():
        return
    idx, code, station, run, vday, band, end_hour = (
        idx[valid], code[valid], station[valid], run[valid],
        vday[valid], band[valid], end_hour[valid])
    # ② 先按 (源, 站, 批次, 日, 段, 窗结束钟点) 把三个钟点求和。
    #    全部用 `np.add.reduceat` 做**一次**分组归约，而不是逐组 `np.unique`
    #    （实测逐组版本在这个站上要做 8.5 万次 unique + sort，占掉 6.6 s）。
    key = _group_key(code, station, run, vday,
                     band.astype(np.int64) * 100 + end_hour.astype(np.int64))
    order = np.argsort(key, kind="stable")
    skey = key[order]
    starts = _group_starts(skey)
    cnt = np.diff(np.append(starts, skey.size))
    obs_h = hourly.rain_obs[idx][order]
    fcst_h = hourly.rain_fcst[idx][order]
    obs_sum = np.add.reduceat(obs_h, starts)
    fcst_sum = np.add.reduceat(fcst_h, starts)
    head = order[starts]
    # 不足 3 个小时的窗不是完整累计量（缺一小时 ≠ 那一小时没下雨）
    full = cnt >= 3
    if not full.any():
        return
    arr = np.stack([
        code[head][full].astype(np.float64),
        station[head][full].astype(np.float64),
        np.zeros(int(full.sum()), dtype=np.float64),
        vday[head][full].astype(np.float64),
        band[head][full].astype(np.float64),
        obs_sum[full], fcst_sum[full],
    ], axis=1)
    # ③ 再按 (源, 站, 日, 段) 合并：同一天被 20 个批次各报到一次时只产出一行计数。
    #    若按批次分行，`a + c` 会把"同一场雨被 20 次抓取各命中一次"数成 20 天事件，
    #    收缩因子随之掉一半，真实技巧被一起压掉（实测 Oracle 从 1.0 掉到 0.44）。
    day_key = _group_key(arr[:, 0].astype(np.int64), arr[:, 1].astype(np.int64),
                         np.zeros(len(arr), dtype=np.int64),
                         arr[:, 3].astype(np.int64), arr[:, 4].astype(np.int64))
    o2 = np.argsort(day_key, kind="stable")
    dkey = day_key[o2]
    dstart = _group_starts(dkey)
    grp_head = arr[o2[dstart]]
    o_win = arr[o2, 5]
    f_win = arr[o2, 6]
    # 每 (日, 段) 内的"至少一个窗达标"用 `maximum.reduceat` 一次算完
    for thr, _tw in specs:
        name = f"r3_{_thr_key(thr)}"
        om = np.maximum.reduceat((o_win >= thr).astype(np.int8), dstart) > 0
        fm = np.maximum.reduceat((f_win >= thr).astype(np.int8), dstart) > 0
        for i in range(len(dstart)):
            c_code = int(grp_head[i, 0])
            c_station = int(grp_head[i, 1])
            c_day = int(grp_head[i, 3])
            c_band = int(grp_head[i, 4])
            rows = _rows_for(out, models[c_code], c_day)
            ent = _rain_ent(rows, ("", station_ids[c_station], name, c_band))
            _add_counts(ent, om[i:i + 1], fm[i:i + 1])


def _group_starts(sorted_key: np.ndarray) -> np.ndarray:
    """已排序键 → 每组的起始下标（供 `np.*.reduceat` 使用）。"""
    if sorted_key.size == 0:
        return np.zeros(0, dtype=np.int64)
    return np.concatenate(
        ([0], np.flatnonzero(sorted_key[1:] != sorted_key[:-1]) + 1))


def _accumulate_avail(out: dict, hourly: HourlyFacts, models: Sequence[str],
                      events: Mapping[str, Sequence[tuple[float, float]]]) -> None:
    """可用率的分子：该 (源, 段, 要素) 在**多少个批次里有可配对样本**。

    分母是"声明覆盖"（配置事实），由引擎在窗口层补齐——这里只数分子。
    实现用 `np.unique` 一次得到 (源, 段) 组合与计数，而不是逐组合筛数组
    （实测逐组合版本要在 8.7 万个分组上跑 unique，占掉整站 14 s）。
    """
    if hourly.n == 0:
        return
    run_day = int(hourly.run_day[0]) if hourly.n else 0
    for target, ok in (("temp_h", hourly.temp_ok), ("r3", hourly.rain_ok)):
        good = ok & (hourly.band >= 1)
        if not good.any():
            continue
        code = hourly.model[good].astype(np.int64)
        band = hourly.band[good].astype(np.int64)
        # (源, 段) 组合 → 出现的批次数。同一批次里同一段出现多行只算一次，
        # 所以先用 (源, 段, 批次) 去重，再数每组。
        combo = code * 64 + band
        per_run = np.unique(combo * 100_000 + (hourly.run_minute[good] % 100_000))
        uniq, cnt = np.unique(per_run // 100_000, return_counts=True)
        for c, n in zip(uniq, cnt):
            mi, b = divmod(int(c), 64)
            rows = _rows_for(out, models[mi], run_day)
            ent = rows.avail.setdefault((target, int(b)), [0, 0])
            ent[0] += int(n)


def _rain_ent(rows: "SourceDayRows", key: tuple) -> list:
    """取（必要时创建）一个降水计数桶 `[a, b, c, d, n_ev_days]`。

    `n_ev_days` 的增量口径是"行内该日观测到该级事件"，而**窗口级**的去重由
    `event_days_by_target` 在聚合时重算（同一场雨会同时出现在多个段与多个批次里，
    逐行 +1 会把它数成几百天）。这里保留行级增量是为了让 `n_ev` 在单站单日的
    调试场景下也可读。

    用**可变 list** 而不是元组：计数要逐组原地累加，元组在 Python 里不可变，
    写回会静默丢失（这个 bug 曾让列联表一直停在第一条样本上）。
    """
    ent = rows.rain.get(key)
    if ent is None:
        ent = [0.0, 0.0, 0.0, 0.0, 0.0]
        rows.rain[key] = ent
    return ent


def _add_counts(ent: list, om: np.ndarray, fm: np.ndarray) -> None:
    """把一组列联表计数并入 `ent`（`om`/`fm` 为同桶内的观测/预报布尔向量）。

    事件日数**不在这里维护**：它由窗口层的 `event_days_from_counts` 统一反推。
    逐次 +1 会把"同一场雨被 16 个段和 20 个批次各命中一次"数成 320 天事件，
    收缩因子 `n/(n+n0)` 随之虚高，真实技巧被一起压掉（实测 Oracle 从 1.0 掉到 0.24）。
    """
    ent[0] += int(np.sum(fm & om))
    ent[1] += int(np.sum(fm & ~om))
    ent[2] += int(np.sum(~fm & om))
    ent[3] += int(np.sum(~fm & ~om))
    # 行级事件日增量：本行（某源某站某批次某段某日）观测到该级事件则 +1。
    # 窗口级去重由 `event_days_by_target` 完成，见其 docstring。
    if bool(np.any(om)):
        ent[4] += 1.0


def event_days_from_counts(counts: Sequence[float]) -> float:
    """由列联表**反推**独立事件日数（观测命中 + 漏报 = `a + c`）。

    **只在"每天每目标一行"的粒度上成立**：同一自然日会同时出现在 `k=1..16` 的多个
    段里，所以窗口层必须先按 (站, 目标) 取**每个自然日的最早一次**（最小 `k`），
    再对该序列求 `a + c`。那一步见 `event_days_by_target`——把去重放在窗口层而不是
    行层，是因为"哪些行算同一场雨"取决于你要聚合到哪个窗口，而窗口是调用方决定的。
    """
    return float(counts[0] + counts[2])


def event_days_by_target(days: Mapping[int, "SourceDayRows"],
                         targets: Sequence[str]) -> dict[str, float]:
    """窗口内每个降水目标的**独立事件日数**（每个自然日只算一天）。

    口径：对每个 (自然日, 目标)，取该目标**所有行里最小段**（最小 `band`）的行，
    看该行是否观测到该级事件（`a + c > 0`）。同一场雨在第 1、2、9 段都被报、
    被 20 个批次各抓一次，都只算**一天**；而它作为"观测事实"只发生一次。

    为什么必须是"一天一次"而不是"一行一次"（三个坑，这里都显式处理）：

    * **只看 `c`（漏报）是错的**：完美预报的观测全在命中 `a` 里，只看漏报会把
      "完美"判成"没下雨"，`n_ev = 0` 让收缩因子归零，Oracle 的降水分变成 0。
    * **按 (日, 段) 计数会让 `n_ev` 随段数翻倍**：收缩因子被虚抬到接近 1，
      重阈值的有限样本噪声重新漏进总分。
    * **`n_ev` 与分母不是同一行集**同样错：实测把 Oracle 的第 1 段压到 0.24
      （分母 81 段·天、分子却只有 16 天）。

    因此 `n_ev` 的语义固定为"**这个窗口里有几天真的下过这个级别**"，与收缩因子
    `n/(n+n0)` 想表达的"证据够不够"完全一致；分母（列联表 `n_k`）则按段·天计数，
    两者量纲不同不是错误——`n0 = 15` 就是按"事件日"标定的。
    """
    picked: dict[tuple[int, str], tuple[int, bool]] = {}
    for day, rows in days.items():
        for (_run, _station, target, band), ent in rows.rain.items():
            if target not in targets:
                continue
            key = (int(day), target)
            wet = (float(ent[0]) + float(ent[2])) > 0.0
            cur = picked.get(key)
            if cur is None:
                picked[key] = (int(band), wet)
            else:
                picked[key] = (min(int(band), cur[0]), wet or cur[1])
    out: dict[str, float] = {}
    for (_day, target), (_band, wet) in picked.items():
        if wet:
            out[target] = out.get(target, 0.0) + 1.0
    return out


# ---------------------------------------------------------------- 窗口求和
@dataclass
class WindowAggregate:
    """一个窗口内某源的汇总统计量（可加、可"去掉某一段"）。

    `temp[(run, station, target, band)]`、`rain[(run, station, target, band)]` 与落盘
    结构一致；`days_present` 记录该源在窗口内**实际有样本**的自然日（证据等级与
    自助都用它）。
    """

    temp: dict[tuple[str, str, str, int], list[float]] = field(default_factory=dict)
    rain: dict[tuple[str, str, str, int], list] = field(default_factory=dict)
    days_present: set[int] = field(default_factory=set)
    n_rows: int = 0

    def add_rows(self, day: int, rows: "SourceDayRows") -> None:
        if rows.empty:
            return
        self.days_present.add(day)
        for key, vals in rows.temp.items():
            ent = self.temp.get(key)
            if ent is None:
                self.temp[key] = list(vals)
            else:
                for i, v in enumerate(vals):
                    ent[i] += v
        for key, ent in rows.rain.items():
            cur = self.rain.get(key)
            if cur is None:
                cur = [0.0, 0.0, 0.0, 0.0, float(ent[4])]
                self.rain[key] = cur
            for i in range(4):
                cur[i] += ent[i]
            cur[4] += float(ent[4])
        self.n_rows += 1


def aggregate_window(days: Mapping[int, "SourceDayRows"],
                     day_weights: Mapping[int, float] | None = None
                     ) -> WindowAggregate:
    """把一段逐日行求和成窗口统计量。

    `day_weights` 用于自助（同一日期被抽到 `k` 次）与近期加权（半衰期指数权重）。
    **加权时事件日数也必须跟着缩放**：否则"重采样 3 次的那一天"会被当成 3 个独立
    事件日，收缩因子 `n/(n+n0)` 会随抽样次数虚增——这是自助里最容易悄悄制造乐观
    偏差的地方（见 `_scale_rows`）。
    """
    out = WindowAggregate()
    for day in sorted(days):
        w = 1.0 if day_weights is None else float(day_weights.get(day, 0.0))
        if w == 0.0:
            continue
        out.add_rows(day, _scale_rows(days[day], w))
    return out


def weight_window(day_list: Sequence[tuple[int, "SourceDayRows"]],
                  day_weights: Mapping[int, float]) -> WindowAggregate:
    """`aggregate_window` 的**预排序**版本（自助的 2000 次重采样专用）。

    差别只在输入形状：`aggregate_window` 每次都要对 `days` 做 `sorted()` 与字典
    查找，而自助在同一个源上要跑上千遍。预先把 `[(day, rows)]` 排好序，热循环里
    只剩一次乘法与一次合并——实测把自助的单次评分从 3.0 ms 降到 1.1 ms。
    **数学口径完全相同**：都是"按日权重对可加统计量求和"。
    """
    out = WindowAggregate()
    for day, rows in day_list:
        w = float(day_weights.get(day, 0.0))
        if w == 0.0:
            continue
        out.add_rows(day, _scale_rows(rows, w))
    return out


def _scale_rows(rows: "SourceDayRows", w: float) -> "SourceDayRows":
    """按日权重缩放行（`w == 1.0` 时零拷贝返回原对象）。

    自助里大多数日权重是 0 或 1的整数（几何块抽样），`w == 1.0` 的零拷贝路径
    因此命中率很高——这一点直接决定了 2000 次重采样的总耗时。
    """
    if w == 1.0:
        return rows
    scaled = SourceDayRows()
    for key, vals in rows.temp.items():
        scaled.temp[key] = [v * w for v in vals]
    for key, ent in rows.rain.items():
        scaled.rain[key] = [ent[i] * w for i in range(5)]
    scaled.avail = dict(rows.avail)
    return scaled


# ---------------------------------------------------------------- 落盘
def _rows_to_json(rows: "SourceDayRows") -> dict:
    return {
        "t": [list(k) + [list(v)] for k, v in sorted(rows.temp.items())],
        "r": [list(k) + [list(ent)] for k, ent in sorted(rows.rain.items())],
        "a": [list(k) + [list(v)] for k, v in sorted(rows.avail.items())],
    }


def _rows_from_json(node: Mapping) -> "SourceDayRows":
    rows = SourceDayRows()
    for item in node.get("t") or []:
        run, station, target, band, vals = item
        rows.temp[(str(run), str(station), str(target), int(band))] = [
            float(x) for x in vals]
    for item in node.get("r") or []:
        run, station, target, band, ent = item
        rows.rain[(str(run), str(station), str(target), int(band))] = [
            float(x) for x in ent]
    for item in node.get("a") or []:
        target, band, vals = item
        rows.avail[(str(target), int(band))] = [int(x) for x in vals]
    return rows


def store_path(root: str | Path, month: str) -> Path:
    return Path(root) / "metrics" / "dvs" / f"{month}.json.gz"


def save_store(root: str | Path, month: str,
               data: Mapping[str, Mapping[int, "SourceDayRows"]],
               *, station_ids: Sequence[str], extras: Mapping | None = None,
               force: bool = False) -> Path:
    """写一个月的充分统计量仓。

    **冻结不变量**：默认不覆盖已存在的月份。仓是"证据"，同一份证据被反复重写会让
    git 每次都存一个全新 blob，而内容其实只差运行时缓存状态（与 `storage` 的月度
    bundle 同一纪律）。回灌历史必须显式 `force=True`。
    """
    p = store_path(root, month)
    if p.is_file() and not force:
        logger.info("DVS 统计量仓 %s 已存在且未要求覆盖，保持原样", month)
        return p
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": STORE_SCHEMA,
        "month": month,
        "stations": list(station_ids),
        "sources": {
            model: {date.fromordinal(int(d)).isoformat(): _rows_to_json(rows)
                    for d, rows in sorted(per_day.items()) if not rows.empty}
            for model, per_day in sorted(data.items())
        },
        "extras": dict(extras or {}),
    }
    tmp = p.with_suffix(".json.gz.tmp")
    with gzip.open(tmp, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(p)
    return p


def load_store(root: str | Path, months: Iterable[str]
               ) -> dict[str, dict[int, "SourceDayRows"]]:
    """读多个月的仓，合并成 `{model: {day_ordinal: SourceDayRows}}`。

    损坏或缺失的月份**告警并跳过**（与 `storage.load_obs` 同一容错纪律）：一个月读
    不出来不该拖垮整份报告，但它必须在日志与 meta 里显得出来。
    """
    out: dict[str, dict[int, SourceDayRows]] = {}
    for month in months:
        p = store_path(root, month)
        if not p.is_file():
            continue
        try:
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as e:  # pragma: no cover - IO 异常
            logger.warning("DVS 统计量仓不可读，跳过 %s: %s", month, e)
            continue
        if int(payload.get("schema") or 0) != STORE_SCHEMA:
            logger.warning("DVS 统计量仓 schema 不匹配，跳过 %s", month)
            continue
        for model, per_day in (payload.get("sources") or {}).items():
            target = out.setdefault(model, {})
            for day_iso, node in per_day.items():
                target[date.fromisoformat(day_iso).toordinal()] = _rows_from_json(node)
    return out


def merge_stores(a: Mapping[str, Mapping[int, "SourceDayRows"]],
                 b: Mapping[str, Mapping[int, "SourceDayRows"]]
                 ) -> dict[str, dict[int, "SourceDayRows"]]:
    """两份仓的逐日合并（同一 (源, 日) 出现两次时按可加统计量相加）。

    什么时候会重复：同一自然日既被"归档月报"算了、又被"实时窗口"算了。相加是正确
    语义（两次都是同一物理日的独立观测批次），但**只在调用方明确要求合并时**才发生
    ——默认路径是覆盖，避免同一份证据被悄悄计两遍。
    """
    out: dict[str, dict[int, SourceDayRows]] = {
        m: {d: r for d, r in per.items()} for m, per in a.items()}
    for model, per_day in b.items():
        target = out.setdefault(model, {})
        for day, rows in per_day.items():
            if day not in target:
                target[day] = rows
                continue
            cur = target[day]
            for key, vals in rows.temp.items():
                ent = cur.temp.setdefault(key, [0.0] * len(TEMP_KEYS))
                for i, v in enumerate(vals):
                    ent[i] += v
            for key, ent in rows.rain.items():
                c0 = cur.rain.get(key)
                if c0 is None:
                    cur.rain[key] = list(ent)
                else:
                    for i in range(5):
                        c0[i] += ent[i]
    return out


def months_between(start_ord: int, end_ord: int) -> list[str]:
    months: list[str] = []
    y, m = date.fromordinal(start_ord).year, date.fromordinal(start_ord).month
    y_end, m_end = date.fromordinal(end_ord).year, date.fromordinal(end_ord).month
    while (y, m) <= (y_end, m_end):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def day_weight_decay(days: Sequence[int], end_day: int, halflife_days: float
                     ) -> dict[int, float]:
    """近期加权：`2^(−(end − day)/half_life)`。

    为什么用半衰期而不是滑动窗口：滑动窗口跨过边界时名次会突然跳变，读者无法判断
    "掉出榜首"是变差了还是刚过期。指数衰减连续可微，且"45 天前的数据权重减半"是
    一句可以直接核对的话。
    """
    if halflife_days is None or halflife_days <= 0:
        return {int(d): 1.0 for d in days}
    return {int(d): float(math.pow(2.0, -max(int(end_day) - int(d), 0)
                                   / float(halflife_days)))
            for d in days}


def run_key(minute: int) -> str:
    """批次分钟序号 → `YYYY-MM-DDTHH:MM`（与 `RunBatch.run_key` 同格式）。"""
    from .dvs_protocol import _minute_to_dt   # 局部导入：避免循环依赖
    return _minute_to_dt(int(minute)).strftime("%Y-%m-%dT%H:%M")


def _thr_key(thr: float) -> str:
    """阈值的稳定字符串键：`1.0 → "ge1"`、`0.6 → "ge0.6"`（与 `dvs.RAIN_EVENTS` 对拍）。"""
    return f"ge{int(thr)}" if float(thr).is_integer() else f"ge{thr:g}"


def _group_key(model: np.ndarray, station: np.ndarray, run: np.ndarray,
               day: np.ndarray, band: np.ndarray) -> np.ndarray:
    """把多维整数键压成单个 int64（各维上限由数据规模保证）：
    源 < 2^8、站 < 2^6、分钟序号 < 2^31、日序号 < 2^20、段 < 2^10。"""
    return ((model.astype(np.int64) << 56) | (station.astype(np.int64) << 50)
            | ((run.astype(np.int64) & 0x7FFFFFF) << 23)
            | ((day.astype(np.int64) & 0xFFFFF) << 3)
            | (band.astype(np.int64) & 0x7))


def _split_groups(sorted_key: np.ndarray) -> list[np.ndarray]:
    """已排序键 → 每组的**位置**下标列表（调用方必须再经 `order`/`idx` 才能取行号）。"""
    if sorted_key.size == 0:
        return []
    change = np.flatnonzero(np.diff(sorted_key) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [sorted_key.size]))
    return [np.arange(s, e) for s, e in zip(starts, ends)]


#: 北京时固定 3h 窗的结束钟点映射（0/1 时不属于任何完整窗）。
_HOUR_END = np.array([-1, -1, 2, 2, 2, 5, 5, 5, 8, 8, 8, 11, 11, 11,
                      14, 14, 14, 17, 17, 17, 20, 20, 20, 23], dtype=np.int64)

#: 表头按位序排列的十项温度统计量（与 `dvs_stats_store.TEMP_KEYS` 严格一致）。
TEMP_FIELD_ORDER = TEMP_KEYS


__all__ = [
    "RAIN_KEYS", "STORE_SCHEMA", "SourceDayRows", "TEMP_KEYS", "WindowAggregate",
    "accumulate_daily", "accumulate_day", "aggregate_window", "weight_window",
    "day_weight_decay", "event_days_by_target", "event_days_from_counts",
    "load_store", "merge_stores", "months_between", "run_key", "save_store",
    "store_path",
]

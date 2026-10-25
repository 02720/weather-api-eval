"""配对事实的列式表（TASK-07 的列式化，但**不动 collect 的输出契约**）。

`collect()` 返回 1,500,000 个 dict，而下游有 2,044 次调用要把它们重新扫一遍
（n_eff 用到 9.7 s）、还有一次要压成充分统计量表（10.7 s）。dict 在这里是最贵
的表示：每个字段一次哈希查找，每次扫描一遍全部记录。列式表把同一份配对事实
存成 numpy 数组，于是：
  * 一次扫描变成一次向量化掩膜；
  * 每个 (模型, 天桶) 格子是一段**下标**而不是一份新的 list，零拷贝；
  * 充分统计量表可以直接用 `np.add.at` 从列算出来，不再逐条 append。

**I4 是这里的第一约束，也是最容易踩坏的一条**：列式化天然会把缺测变成 NaN，
而 NaN 会被 `>=` 之类的比较静默当成数值。所以本模块显式保存 `*_ok` 布尔列——
**判定只看 `_ok`，NaN 只是"没有值的格子填了个占位"**。任何下游读取都必须先查
`_ok` 列再取值，绝不依赖 NaN 语义（守卫见 `tests/test_missing_stays_missing`）。

本模块**不替换** `collect()` 的返回值：`hourly` / `daily` 的 dict 列表仍然是
报告其余部分的输入（覆盖面统计、热力图、榜单行计数都在用它）。列式表是为热点
路径服务的**并列的、可重建的**视图——删掉它，一切照旧。
"""
from __future__ import annotations

import numpy as np

from .timeutil import minute_index


def _names_in_code_order(code: dict[str, int]) -> list[str]:
    """编码 → 站名列表（按编码升序，即首次出现顺序）。"""
    return [s for s, _ in sorted(code.items(), key=lambda kv: kv[1])]


class PairTable:
    """逐小时与按天配对事实的列式视图。

    下标约定：所有 *_code 列都是**首次出现顺序**编码的整数（不是字典），
    这是为了能用 `np.lexsort` 复现原实现里 `defaultdict` 的插入顺序——
    顺序一旦不同，n_eff 的求和顺序就会变（I8 的浮点容差保护不了取整边界）。
    """

    def __init__(self, hourly: list[dict], daily: list[dict],
                 models: list[str], stations: list[str], days: list[str]):
        self.models = list(models)
        self.stations = list(stations)
        self.days = list(days)
        self.model_code = {m: i for i, m in enumerate(models)}
        # 站编码**逐轨道单独编号**，且都按该轨道内的首次出现顺序。
        # 为什么不能共用一套编码：旧实现里 `by` 是 defaultdict，站序 = 该轨道内
        # 首次出现的顺序。逐小时轨道与按天轨道的首次出现顺序可能不同（某站只有
        # 按天样本时尤其明显），而 `n_eff_from_station_series` 按 dict 顺序做浮点
        # 求和——站序变了，n_eff 的取整边界就可能翻。共用编码会在这里埋雷。
        self.station_code_h: dict[str, int] = {}
        for r in hourly:
            self.station_code_h.setdefault(r["station"], len(self.station_code_h))
        self.station_code_d: dict[str, int] = {}
        for r in daily:
            self.station_code_d.setdefault(r["station"], len(self.station_code_d))
        self.station_names_h = _names_in_code_order(self.station_code_h)
        self.station_names_d = _names_in_code_order(self.station_code_d)
        # 与 stats.build_day_stat_tables **完全同一套**站索引（先 hourly 后 daily）：
        # 充分统计量表的站维宽度与列号由它决定，两套编码一旦错位，整张表就是错的
        station_code_all: dict[str, int] = {}
        for r in hourly:
            station_code_all.setdefault(r["station"], len(station_code_all))
        for r in daily:
            station_code_all.setdefault(r["station"], len(station_code_all))
        self.station_code_all = station_code_all
        self.station_names_all = _names_in_code_order(station_code_all)
        self._h2all = np.array([station_code_all.get(s, 0)
                                for s in self.station_names_h], dtype=np.int64)
        self._d2all = np.array([station_code_all.get(s, 0)
                                for s in self.station_names_d], dtype=np.int64)
        day_code = {d: i for i, d in enumerate(days)}

        self._build_hourly(hourly, day_code)
        self._build_daily(daily, day_code)

    # ------------------------------------------------------------------ 逐小时
    def _build_hourly(self, hourly, day_code):
        sc = self.station_code_h
        mc = self.model_code
        # 列式化用列表推导逐列生成，再由 numpy 一次性转数组：逐元素 `arr[i] =`
        # 的 14×N 次 Python 下标写入是构建期最大的纯解释器开销（实测 5.7 s）。
        # 列表推导仍是 O(N) Python 级，但每个元素只做一次 dict 取值，随后交给
        # C 层转换——列之间的 dtype 校验由 numpy 完成，不会引入第二套语义。
        self.h_model = np.array([mc.get(r["model"], -1) for r in hourly], dtype=np.int32)
        self.h_bucket = np.array([r["bucket"] for r in hourly], dtype=np.int16)
        self.h_station = np.array([sc[r["station"]] for r in hourly], dtype=np.int16)
        self.h_day = np.array([day_code.get(r["valid_iso"][:10], -1)
                               for r in hourly], dtype=np.int32)
        self.h_lead = np.array([r["lead"] for r in hourly], dtype=np.int32)
        # 时间键：分钟序号（int32）代替定长 U19 字符串列。旧列 2.75M × 76 B
        # ≈ 209 MB，而所有消费者只需要"同一时刻相等 + 按时间排序"这两个性质；
        # 定长 ISO 的字典序与分钟序完全一致（timeutil.minute_index 的等价性已
        # 单独论证），故整数键逐位等价。时间轴若非定长 ISO（minute_index 返回
        # None），整列退回字符串——绝不用一个猜测的整数键替代真实时刻。
        minutes = [minute_index(r["valid_iso"]) for r in hourly]
        if None in minutes:
            self.h_time = np.array([r["valid_iso"] for r in hourly], dtype="U19")
        else:
            self.h_time = np.array(minutes, dtype=np.int32)
        self.h_temp_o = np.array([np.nan if r["temp_obs"] is None else r["temp_obs"]
                                  for r in hourly], dtype=np.float64)
        self.h_temp_f = np.array([np.nan if r["temp_fcst"] is None else r["temp_fcst"]
                                  for r in hourly], dtype=np.float64)
        self.h_rain_o = np.array([np.nan if r["rain_obs"] is None else r["rain_obs"]
                                  for r in hourly], dtype=np.float64)
        self.h_rain_f = np.array([np.nan if r["rain_fcst"] is None else r["rain_fcst"]
                                  for r in hourly], dtype=np.float64)
        # 入样判定**向量化**且不依赖 NaN 语义：两侧都必须是**有限**值。
        # 未写过的格子是 NaN，写了 NaN/inf 的也算缺测——于是 NaN 在这里只有一个
        # 含义（"没有值"），永远不会被下游当成 0.0 参与比较（I4）。
        self.h_temp_ok = np.isfinite(self.h_temp_o) & np.isfinite(self.h_temp_f)
        self.h_rain_ok = np.isfinite(self.h_rain_o) & np.isfinite(self.h_rain_f)
        # 站维统一编码（供充分统计量表使用）：与 stats 的 station_idx 同一套
        self.h_station_all = self._h2all[self.h_station]
        # (模型, 天桶) 格子的下标段：一次排序，之后每个格子零拷贝取用
        key = self.h_model.astype(np.int64) * 1000 + self.h_bucket
        self._h_order = np.argsort(key, kind="stable")
        sorted_key = key[self._h_order]
        self._h_key_sorted = sorted_key

    def _h_cell_bounds(self, model_idx: int, bucket: int):
        target = model_idx * 1000 + bucket
        lo = int(np.searchsorted(self._h_key_sorted, target, side="left"))
        hi = int(np.searchsorted(self._h_key_sorted, target, side="right"))
        return lo, hi

    def hourly_cell(self, model_idx: int, bucket: int) -> np.ndarray:
        """该 (模型, 天桶) 格子在列式表里的下标（已按原序稳定排好）。"""
        lo, hi = self._h_cell_bounds(model_idx, bucket)
        return self._h_order[lo:hi]

    def hourly_model(self, model_idx: int) -> np.ndarray:
        """该模型的全部逐小时下标。"""
        lo = int(np.searchsorted(self._h_key_sorted, model_idx * 1000, side="left"))
        hi = int(np.searchsorted(self._h_key_sorted, (model_idx + 1) * 1000, side="left"))
        return self._h_order[lo:hi]

    def hourly_scored(self, model_idx: int) -> np.ndarray:
        """该模型**进天桶榜**的下标（bucket ≥ 1；bucket=0 是起报当日哨兵，不入榜）。"""
        idx = self.hourly_model(model_idx)
        return idx[self.h_bucket[idx] >= 1]

    def hourly_lead_window(self, model_idx: int, lo_h: int, hi_h: int) -> np.ndarray:
        """该模型在 lead ∈ [lo_h, hi_h] 内的下标（评分卡的 24h/72h 池）。"""
        idx = self.hourly_model(model_idx)
        lead = self.h_lead[idx]
        return idx[(lead >= lo_h) & (lead <= hi_h)]

    # ------------------------------------------------------------------ 按天
    def _build_daily(self, daily, day_code):
        sc = self.station_code_d
        mc = self.model_code
        # 与逐小时同款：列式列表推导 + 一次转换。日轨的时间键直接用天编码
        # （`days` 由 eval_days 按字符串升序给出，编码升序 == 自然日升序），
        # 旧的 U10 字符串列因此完全不需要存在。
        self.d_model = np.array([mc.get(r["model"], -1) for r in daily], dtype=np.int32)
        self.d_offset = np.array([r["offset"] for r in daily], dtype=np.int16)
        self.d_station = np.array([sc[r["station"]] for r in daily], dtype=np.int16)
        self.d_day = np.array([day_code.get(r["valid_day"], -1)
                               for r in daily], dtype=np.int32)
        self.d_time = self.d_day
        self.d_max_o = np.array([np.nan if r["temp_max_obs"] is None else r["temp_max_obs"]
                                 for r in daily], dtype=np.float64)
        self.d_max_f = np.array([np.nan if r["temp_max_fcst"] is None else r["temp_max_fcst"]
                                 for r in daily], dtype=np.float64)
        self.d_min_o = np.array([np.nan if r["temp_min_obs"] is None else r["temp_min_obs"]
                                 for r in daily], dtype=np.float64)
        self.d_min_f = np.array([np.nan if r["temp_min_fcst"] is None else r["temp_min_fcst"]
                                 for r in daily], dtype=np.float64)
        self.d_rain_o = np.array([np.nan if r["rain_obs"] is None else r["rain_obs"]
                                  for r in daily], dtype=np.float64)
        self.d_rain_f = np.array([np.nan if r["rain_fcst"] is None else r["rain_fcst"]
                                  for r in daily], dtype=np.float64)
        # 日温度维的两个量**各自**记录是否成对：旧实现是"两个量里至少一个成对
        # 就取这一条，值取成对的那些量的平均"。混合成一个 bool 会把"只有最高成对"
        # 的样本丢掉，n_eff 立刻对不上。
        self.d_tmax_ok = np.isfinite(self.d_max_o) & np.isfinite(self.d_max_f)
        self.d_tmin_ok = np.isfinite(self.d_min_o) & np.isfinite(self.d_min_f)
        self.d_rain_ok = np.isfinite(self.d_rain_o) & np.isfinite(self.d_rain_f)
        self.d_station_all = self._d2all[self.d_station]
        key = self.d_model.astype(np.int64) * 1000 + self.d_offset
        self._d_order = np.argsort(key, kind="stable")
        self._d_key_sorted = key[self._d_order]

    def daily_cell(self, model_idx: int, offset: int) -> np.ndarray:
        target = model_idx * 1000 + offset
        lo = int(np.searchsorted(self._d_key_sorted, target, side="left"))
        hi = int(np.searchsorted(self._d_key_sorted, target, side="right"))
        return self._d_order[lo:hi]

    def daily_model(self, model_idx: int) -> np.ndarray:
        lo = int(np.searchsorted(self._d_key_sorted, model_idx * 1000, side="left"))
        hi = int(np.searchsorted(self._d_key_sorted, (model_idx + 1) * 1000, side="left"))
        return self._d_order[lo:hi]


def min_lead_select(station: np.ndarray, time_key: np.ndarray, lead: np.ndarray,
                    valid: np.ndarray):
    """按 (站, 时刻) 取 lead 最小的那条记录，返回 `(sel, sel_st, keep)`。

    sel：选中记录在**过滤后数组**里的下标（已按"站序 → 时刻升序"排好）；
    sel_st：选中记录的站编码；keep：`valid` 的下标（调用方据此取自己的值列）。

    time_key 可以是整数分钟序号（PairTable 的常规路径）或定长 ISO 字符串
    （非标准时间轴的退化路径）。两种键的排序次序一致——定长 ISO 的字典序
    == 时刻序，分钟序 == 时刻序——所以下面的去重与排序逻辑对两者完全相同。

    与旧 dict 实现逐位对齐的四条语义（差一条 n_eff 就会漂）：
      * 只保留 `valid` 为真的记录（旧代码 `if o is None or f is None: continue`）；
      * 同一 (站, 时刻) 留 lead 最小者，**并列时取最先出现的那条**——旧代码用的是
        `if cur is None or lead < cur[0]`（严格小于），故用**稳定排序**保留原序；
      * 每个站内部的时刻按时间升序（旧代码 `sorted(d.items())`）；
      * 站序按首次出现（旧代码 `defaultdict` 的插入顺序）——由编码顺序保证。

    无任何有效记录时返回 `(None, None, None)`（旧代码 `if not by: return None`）。
    """
    if valid.size == 0 or not valid.any():
        return None, None, None
    keep = np.flatnonzero(valid)
    st = station[keep]
    ld = lead[keep]
    ts = time_key[keep]
    # 时刻编码：np.unique 按 time_key 升序给出编码，于是 code 顺序 == sorted() 顺序
    _u, tcode = np.unique(ts, return_inverse=True)
    # lexsort：末位是主键，稳定排序保留上一轮顺序
    #   → (站, 时刻) 组内按 lead 升序；lead 并列时保留原始出现顺序
    order = np.lexsort((ld, tcode.astype(np.int64), st.astype(np.int64)))
    s_sorted = st[order]
    t_sorted = tcode[order]
    new = np.ones(order.size, dtype=bool)
    if order.size > 1:
        new[1:] = (s_sorted[1:] != s_sorted[:-1]) | (t_sorted[1:] != t_sorted[:-1])
    sel = order[np.flatnonzero(new)]
    return sel, st[sel], keep


def grouped_by_station(sel_st: np.ndarray, values, station_names: list[str]) -> dict:
    """按站切段成 {站名: 值列表}（站序 = 编码升序 = 首次出现顺序）。"""
    out: dict[str, list] = {}
    for code in np.unique(sel_st):
        rows = sel_st == code
        name = station_names[int(code)]
        if isinstance(values, np.ndarray) and values.dtype.kind in ("U", "S"):
            out[name] = [str(v) for v in values[rows]]
        else:
            out[name] = values[rows].tolist() if isinstance(values, np.ndarray) \
                else [values[i] for i in np.flatnonzero(rows)]
    return out

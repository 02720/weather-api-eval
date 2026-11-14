"""DVS v6 评估协议层：用户视角（as-of）配对事实表。

**为什么需要这一层（第一性原理）**

旧口径把"预报有多准"锚在**各源自报的起报时刻**（`issue_iso`）。同一个 cron
批次抓回来的 27 个源，自述起报时刻彼此相差最多 20 小时，于是"提前 3 天"在
不同源身上其实是"提前 2.2 天"到"提前 3.8 天"——时效轴的刻度本身就不是同一把尺。
DVS v6 的分数语义是"在用户真正发起请求的那一刻（决策零点 `t_run`），这家源替
我消除了多少决策损失"，所以时效必须从**用户可调用时刻**起算。`issue_iso` 由此
退出计分，只保留为"新鲜度"诊断属性（`t_run − issue_declared`）。

本模块只做四件事，全部是**纯函数 + 显式输入**（无全局状态、无 IO），便于单元测试：

1. `assign_runs`：把一个站的快照流按抓取时刻聚成批次窗，对齐整点得到决策零点
   `t_run`。批次是时间区间的划分，区间互不重叠——一份快照只属于一个批次，
   不存在"同一份数据在两条批次里各计一次票"。
2. `build_hourly_facts`：逐 (批次, 站, 源) 建立逐小时配对事实，施加因果护栏：
   `t_valid ≥ t_run + min_lead_h`（防时效 0 的实况冒充预报）、
   `t_valid ≥ fetched_at`（防事后修订值混入），再把 `ℓ = t_valid − t_run` 折算成
   天序号 `k = ⌈ℓ/24h⌉`（1..16）与五段之一。
3. `build_daily_facts`：逐小时 → 日尺度（`tmax` / `tmin` / `r24`）。主分只用
   **两侧共同非缺测小时** 聚合，共同小时数 `< daily_min_hours` 整日弃用；同一
   (源, 站, 有效日) 只取**最早一次**预报（最小 `k`），因为日极值是一个物理对象，
   不因被多少条预报报到而变成多个对象。
4. `build_3h_facts`：逐小时 → 北京时固定 3h 窗累计（02/05/…/23 时止），与原生
   后向 3 小时产品天然同窗。

两条贯穿全层的纪律：
  * **缺测绝不以数值顶替**（不折算 0、不插值、不用均值）；
  * **同一物理对象不重复计票**（日尺度按最小 `k` 去重；3h 窗要求 3 小时齐备）。
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime, timedelta
from typing import Mapping, Sequence

import numpy as np

from .timeutil import parse_iso

# ---------------------------------------------------------------- 常量与契约
#: 五段时效聚合窗（天）。段号 b=1..5，与 `band_weights` 一一对应。
#: 分段的理由：明天的伞、本周的排涝、下周的排班，对时效的敏感度不是一条直线。
BAND_BOUNDS: tuple[tuple[int, int], ...] = ((1, 1), (2, 3), (4, 7), (8, 10), (11, 16))

#: 北京时固定 3h 窗的结束钟点（2、5、8、…、23 时止）。
THREE_HOUR_END_HOURS: tuple[int, ...] = tuple(range(2, 24, 3))

#: 日尺度主分要素。
DAILY_TEMP_TARGETS = ("tmax", "tmin")
DAILY_RAIN_TARGET = "r24"
HOURLY_TEMP_TARGET = "temp_h"

#: 分钟序号的原点（2000-01-01 + 730120 = 1 日序号差）。
_MIN_ORIGIN_ORD = 730120


def _band_table() -> np.ndarray:
    """长度为 17 的查表：`BAND_OF[d]` = 段号（1..5），越界为 -1。"""
    out = np.full(17, -1, dtype=np.int8)
    for i, (lo, hi) in enumerate(BAND_BOUNDS, 1):
        out[lo:hi + 1] = i
    return out


BAND_OF = _band_table()


def band_of(day_index: int) -> int | None:
    """天序号 → 段号（1..5）；超出 1..16 返回 None。"""
    if 1 <= day_index <= 16:
        return int(BAND_OF[day_index])
    return None


def lead_to_day(lead_hours: np.ndarray) -> np.ndarray:
    """有效时效（小时）→ 天序号 `k = ⌈ℓ / 24⌉`，截到 0..16。

    用上取整而不是四舍五入：`ℓ = 1h` 与 `ℓ = 24h` 都属于"决策零点之后第一个
    24 小时窗口"，`ℓ = 25h` 才跨进第二个。四舍五入会让 36h 落进"第 1 天"，
    与"提前一天"的日常语义相反。
    """
    lead = np.asarray(lead_hours, dtype=float)
    k = np.ceil(lead / 24.0)
    k = np.where(np.isfinite(k), k, 0).astype(np.int64)
    return np.clip(k, 0, 16)


# ---------------------------------------------------------------- 批次协议
@dataclass(frozen=True)
class RunBatch:
    """一次抓取批次 = 一个决策零点 `t_run` 及其包含的 (源, 快照)。"""

    station: str
    run_key: str                     # 'YYYY-MM-DDTHH:MM'
    fetched_lo: str                  # 批次内最早抓取时刻
    fetched_hi: str                  # 批次内最晚抓取时刻
    snapshots: tuple[tuple[str, dict], ...] = ()
    declared_only: bool = False      # 历史存档无 fetched_at，用 issue_iso 作锚

    @property
    def t_run(self) -> datetime:
        return parse_iso(self.run_key)

    @property
    def t_run_minute(self) -> int:
        return _minute_ord(self.t_run)


def _parse_minute(s) -> datetime | None:
    if not isinstance(s, str) or len(s) < 16:
        return None
    try:
        return parse_iso(s[:16])
    except ValueError:
        return None


def assign_runs(snapshots_by_model: Mapping[str, Sequence[dict]], *,
                station: str = "", batch_cluster_min: int = 30) -> list[RunBatch]:
    """把一站的快照流按抓取时刻聚成批次。

    **聚簇规则**：按抓取时刻升序，以每个整点对齐的锚开一个窗口，抓取时刻落在
    `[t, t + batch_cluster_min 分钟)` 内的快照归入该批次，`t_run = t`。

    缺 `fetched_at_bj` 的历史存档（2026-09 契约之前）退回以 `issue_iso` 作锚，
    并标记 `declared_only`。历史存档无从判定真实抓取时刻，这是唯一诚实的选择：
    按自述起报时刻当作决策零点，等价于旧口径对这段历史的口径——**绝不因为字段
    缺失而凭空抹掉历史样本**（与 `collect(require_frozen=...)` 同一哲学）。
    """
    entries: list[tuple[datetime, bool, str, dict]] = []
    for model, snaps in snapshots_by_model.items():
        for snap in snaps:
            fetched = _parse_minute(snap.get("fetched_at_bj"))
            if fetched is not None:
                entries.append((fetched, False, model, snap))
                continue
            declared = _parse_minute(snap.get("issue_iso"))
            if declared is not None:
                entries.append((declared, True, model, snap))
    if not entries:
        return []
    entries.sort(key=lambda e: (e[0], e[2]))
    width = timedelta(minutes=max(1, int(batch_cluster_min)))
    groups: list[list[tuple[datetime, bool, str, dict]]] = [[entries[0]]]
    for e in entries[1:]:
        head = groups[-1][0][0].replace(minute=0, second=0, microsecond=0)
        if e[0] < head + width:
            groups[-1].append(e)
        else:
            groups.append([e])
    out: list[RunBatch] = []
    for g in groups:
        t_run = g[0][0].replace(minute=0, second=0, microsecond=0)
        out.append(RunBatch(
            station=station,
            run_key=t_run.strftime("%Y-%m-%dT%H:%M"),
            fetched_lo=min(x[0] for x in g).strftime("%Y-%m-%dT%H:%M"),
            fetched_hi=max(x[0] for x in g).strftime("%Y-%m-%dT%H:%M"),
            snapshots=tuple((x[2], x[3]) for x in g),
            declared_only=all(x[1] for x in g),
        ))
    return out


# ---------------------------------------------------------------- 观测列
@dataclass(frozen=True)
class ObsColumn:
    """按整点分钟序号索引的观测列（缺测为 NaN）。

    `minute0` 是第一条记录的分钟序号，`temp[i]` / `rain[i]` 对应
    `minute0 + i`。这样"按有效时刻取观测"是一次整数下标，而不是一次字典查找——
    百万级配对上的差异是数十秒与数分钟的差别。
    """

    minute0: int
    temp: np.ndarray
    rain: np.ndarray

    @classmethod
    def from_map(cls, obs_map: Mapping[str, dict]) -> "ObsColumn":
        if not obs_map:
            return cls(0, np.zeros(0), np.zeros(0))
        keys = sorted(k for k in obs_map if isinstance(k, str) and len(k) >= 16)
        if not keys:
            return cls(0, np.zeros(0), np.zeros(0))
        starts = np.array([_minute_of_iso(k) for k in keys], dtype=np.int64)
        n = int(starts[-1] - starts[0]) + 1
        temp = np.full(n, np.nan)
        rain = np.full(n, np.nan)
        off = starts - starts[0]
        for j, k in enumerate(keys):
            rec = obs_map[k]
            t = rec.get("temp")
            r = rec.get("rain")
            if isinstance(t, (int, float)) and np.isfinite(t):
                temp[off[j]] = float(t)
            if isinstance(r, (int, float)) and np.isfinite(r):
                rain[off[j]] = float(r)
        return cls(int(starts[0]), temp, rain)

    def lookup(self, minutes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """按分钟序号批量取值；越界或缺失者保持 NaN。"""
        out_t = np.full(len(minutes), np.nan)
        out_r = np.full(len(minutes), np.nan)
        if self.temp.size == 0 or len(minutes) == 0:
            return out_t, out_r
        idx = np.asarray(minutes, dtype=np.int64) - self.minute0
        ok = (idx >= 0) & (idx < self.temp.size)
        out_t[ok] = self.temp[idx[ok]]
        out_r[ok] = self.rain[idx[ok]]
        return out_t, out_r

    def values(self) -> list[float | None]:
        """按时间升序返回观测温度（供基准层做因果窗统计）。"""
        return [None if not np.isfinite(v) else float(v) for v in self.temp]


def _minute_ord(dt: datetime) -> int:
    return (dt.toordinal() - _MIN_ORIGIN_ORD) * 1440 + dt.hour * 60 + dt.minute


def _minute_of_iso(s: str) -> int:
    return ((parse_iso(s[:10] + "T00:00").toordinal() - _MIN_ORIGIN_ORD) * 1440
            + int(s[11:13]) * 60 + int(s[14:16]))


def minute_to_day(minute: int) -> int:
    """分钟序号 → 自然日序号（自 1 年 1 月 1 日，与 `date.toordinal()` 同尺度）。"""
    return int(minute) // 1440 + _MIN_ORIGIN_ORD


def day_to_ord(day: int) -> int:
    """分钟序号的日序号 → `date.toordinal()`。"""
    return int(day)


# ---------------------------------------------------------------- 逐小时事实
_FACT_FIELDS = (
    ("run_minute", np.int64), ("run_day", np.int32), ("station", np.int32),
    ("model", np.int32), ("valid_day", np.int32), ("valid_minute", np.int64),
    ("hour_bjt", np.int8), ("day_index", np.int8), ("band", np.int8),
    ("lead_hours", np.int16),
    ("temp_obs", np.float64), ("temp_fcst", np.float64),
    ("rain_obs", np.float64), ("rain_fcst", np.float64),
    ("temp_ok", np.bool_), ("rain_ok", np.bool_),
)


@dataclass
class HourlyFacts:
    """逐 (批次, 站, 源, 有效时刻) 的配对事实（列式）。

    与 `pairtable.PairTable` 同一条纪律：**缺测只看 `*_ok` 布尔列**，绝不依赖
    NaN 语义做判定（`NaN >= x` 静默为 False，是最容易悄悄吃掉一整个站的那类 bug）。
    """

    run_minute: np.ndarray
    run_day: np.ndarray
    station: np.ndarray
    model: np.ndarray
    valid_day: np.ndarray
    valid_minute: np.ndarray
    hour_bjt: np.ndarray
    day_index: np.ndarray
    band: np.ndarray
    lead_hours: np.ndarray
    temp_obs: np.ndarray
    temp_fcst: np.ndarray
    rain_obs: np.ndarray
    rain_fcst: np.ndarray
    temp_ok: np.ndarray
    rain_ok: np.ndarray

    @property
    def n(self) -> int:
        return int(self.model.size)

    def take(self, mask: np.ndarray) -> "HourlyFacts":
        mask = np.asarray(mask, dtype=bool)
        return HourlyFacts(**{f.name: getattr(self, f.name)[mask]
                              for f in fields(self)})

    def __add__(self, other: "HourlyFacts") -> "HourlyFacts":
        if self.n == 0:
            return other
        if other.n == 0:
            return self
        return HourlyFacts(**{f.name: np.concatenate(
            (getattr(self, f.name), getattr(other, f.name))) for f in fields(self)})


def empty_hourly_facts() -> HourlyFacts:
    return HourlyFacts(**{name: np.zeros(0, dtype=dt) for name, dt in _FACT_FIELDS})


def build_hourly_facts(runs: Sequence[RunBatch], *,
                       obs: ObsColumn,
                       model_codes: Mapping[str, int],
                       station_code: int,
                       start_day: int,
                       end_day: int,
                       min_lead_h: int = 1,
                       max_lead_days: int = 16) -> HourlyFacts:
    """由批次流建立逐小时配对事实。

    **护栏逐条对应一类具体失真**：

    * `lead < min_lead_h` 拒收：时效 0 的样本意味着这份"预报"是在那一小时已经
      发生之后取回的，对纯模式产品风险低，对网页接口源则完全可能回吐实况。
    * `t_valid < fetched_at` 拒收：抓取之前的有效时刻不可能是前瞻预报，其值可能
      已被事后修订——这是项目反复声明要防的"事后取数污染评估"。
    * 观测缺测不成对：不以 0 或均值顶替。
    * 有效日期必须落在评估窗口 `[start_day, end_day]` 内（窗口外的样本既不进
      分子也不进分母，缺测与窗口裁剪是两件事）。

    **不做跨批次去重**：同一有效时刻在不同批次下就是不同的测量对象（`ℓ=24h` 的
    误差与 `ℓ=300h` 的误差不是同一件事）。去重只发生在日尺度聚合。
    """
    # 结果按列收集，**最后一次** np.concatenate：逐块 `__add__` 会在每份快照上复制
    # 一遍已有全部行，行数到百万级时就是 O(n²)（实测 3,793 份快照累计 31 s 全花在拼接上）。
    cols: dict[str, list[np.ndarray]] = {name: [] for name, _dt in _FACT_FIELDS}
    # 时间轴缓存跨批次共享：cron 每小时抓一次，同一份 provider 的时间轴在几十个批次里
    # 逐字符相同。按 `id(times)` 记忆解析结果，每份快照只解析一次。
    axis_cache: dict[int, dict[str, np.ndarray]] = {}
    for batch in runs:
        t_run_min = batch.t_run_minute
        t_run_day = batch.t_run.toordinal()
        for model, snap in batch.snapshots:
            code = model_codes.get(model)
            if code is None:
                continue
            times = snap.get("hourly_time") or []
            entry = (snap.get("data") or {}).get(model) or {}
            if not isinstance(times, list) or not isinstance(entry, dict):
                continue
            arr_t = entry.get("temperature_2m") or []
            arr_p = entry.get("precipitation") or []
            lt, lp = len(arr_t), len(arr_p)
            axis = _axis(times, axis_cache)
            lead_all = (axis["vmin"] - t_run_min) // 60
            keep = axis["ok"] & (lead_all >= min_lead_h) \
                & (lead_all <= max_lead_days * 24) \
                & (axis["vday"] >= start_day) & (axis["vday"] <= end_day)
            fetched = _parse_minute(snap.get("fetched_at_bj"))
            if fetched is not None:
                keep &= axis["vmin"] >= _minute_ord(fetched)
            sel = np.flatnonzero(keep)
            if sel.size == 0:
                continue
            idx = sel.astype(np.int64)
            lead = lead_all[sel]
            minutes = axis["vmin"][sel]
            vdays = axis["vday"][sel]
            hours = axis["hour"][sel]
            t_obs, r_obs = obs.lookup(minutes)
            temp_fcst = _pick(arr_t, idx, lt)
            rain_fcst = _pick(arr_p, idx, lp)
            k = lead_to_day(lead)
            n = idx.size
            cols["run_minute"].append(np.full(n, t_run_min, dtype=np.int64))
            cols["run_day"].append(np.full(n, t_run_day, dtype=np.int32))
            cols["station"].append(np.full(n, station_code, dtype=np.int32))
            cols["model"].append(np.full(n, code, dtype=np.int32))
            cols["valid_day"].append(vdays.astype(np.int32))
            cols["valid_minute"].append(minutes.astype(np.int64))
            cols["hour_bjt"].append(hours.astype(np.int8))
            cols["day_index"].append(k.astype(np.int8))
            cols["band"].append(np.where((k >= 1) & (k <= max_lead_days),
                                        BAND_OF[np.clip(k, 0, 16)], -1).astype(np.int8))
            cols["lead_hours"].append(lead.astype(np.int16))
            cols["temp_obs"].append(t_obs)
            cols["temp_fcst"].append(temp_fcst)
            cols["rain_obs"].append(r_obs)
            cols["rain_fcst"].append(rain_fcst)
            cols["temp_ok"].append(np.isfinite(t_obs) & np.isfinite(temp_fcst))
            cols["rain_ok"].append(np.isfinite(r_obs) & np.isfinite(rain_fcst))
    if not cols["model"]:
        return empty_hourly_facts()
    return HourlyFacts(**{
        name: np.concatenate(cols[name]) if cols[name] else np.zeros(0, dtype=dt)
        for name, dt in _FACT_FIELDS})


def _axis(times: list, cache: dict[int, dict[str, np.ndarray]]
          ) -> dict[str, np.ndarray]:
    """按时间轴对象记忆的解析结果（分钟序号 / 年内日序 / 钟点）。

    **注意 lead 依赖 `t_run`，所以这里只缓存与批次无关的部分**（有效时刻的分钟号 /
    日序号 / 钟点），lead 在调用处按 `t_run` 现算。这是"缓存键必须是真正的常量"
    这条纪律的直接后果：把 lead 也缓存起来，第二批起就会读到上一批的时效。
    """
    key = id(times)
    hit = cache.get(key)
    if hit is not None:
        return hit
    res = _axis_from_strings(times)
    cache[key] = res
    return res


#: 时间戳字符串 → (分钟号, 日序号, 钟点) 的**跨批次全局缓存**。
#: 3,793 份快照的时间轴在字符串层面高度重复（同一 provider 每小时抓一次，只是整体
#: 平移），逐条 `datetime.fromisoformat` 要 3.0 s/站。按"字符串本身"缓存后，
#: 每个唯一时刻只解析一次——实测把这一步从 3.0 s 降到 0.3 s。
_TS_CACHE: dict[str, tuple[int, int, int]] = {}


def _parse_ts(tstr: str) -> tuple[int, int, int] | None:
    hit = _TS_CACHE.get(tstr)
    if hit is not None:
        return hit
    if not isinstance(tstr, str) or len(tstr) < 16:
        return None
    try:
        d = datetime.fromisoformat(tstr)
    except ValueError:
        return None
    val = ((d.toordinal() - _MIN_ORIGIN_ORD) * 1440 + d.hour * 60 + d.minute,
           d.toordinal(), d.hour)
    if len(_TS_CACHE) < 2_000_000:
        _TS_CACHE[tstr] = val
    return val


def _axis_from_strings(times: list) -> dict[str, np.ndarray]:
    """时间轴 → 列式键（分钟号 / 日序号 / 钟点），逐条解析但带全局字符串缓存。"""
    n = len(times)
    vmin = np.zeros(n, dtype=np.int64)
    vday = np.zeros(n, dtype=np.int32)
    hour = np.zeros(n, dtype=np.int8)
    ok = np.zeros(n, dtype=bool)
    get = _TS_CACHE.get
    for i, tstr in enumerate(times):
        # 绝大多数命中都发生在这一行（同一 provider 的时间轴跨批次重复）
        if isinstance(tstr, str):
            hit = get(tstr)
            if hit is not None:
                vmin[i], vday[i], hour[i] = hit
                ok[i] = True
                continue
        parsed = _parse_ts(tstr)
        if parsed is None:
            continue
        vmin[i], vday[i], hour[i] = parsed
        ok[i] = True
    return {"vmin": vmin, "vday": vday, "hour": hour, "ok": ok}


def _minute_to_dt(minute: int) -> datetime:
    """分钟序号 → datetime（与 `_minute_ord` 严格互逆）。"""
    day, rem = divmod(int(minute), 1440)
    return datetime.fromordinal(day + _MIN_ORIGIN_ORD).replace(
        hour=rem // 60, minute=rem % 60)


def _pick(arr: list, idx: np.ndarray, length: int) -> np.ndarray:
    """按整数下标取数组值，越界为 NaN（畸形存档降级为该点缺测，不炸整份报告）。"""
    out = np.full(idx.size, np.nan)
    ok = idx < length
    if ok.any():
        vals = np.asarray([arr[j] for j in idx[ok]], dtype=float)
        out[ok] = vals
    return out


# ---------------------------------------------------------------- 日尺度事实
@dataclass
class DailyFacts:
    """日尺度配对事实（`tmax` / `tmin` / `r24`），列式、缺测显式（NaN + 计数）。"""

    station: np.ndarray
    model: np.ndarray
    valid_day: np.ndarray
    run_minute: np.ndarray      # 代表批次的 t_run（L0 的 δ/a₀ 依赖它）
    day_index: np.ndarray
    band: np.ndarray
    tmax_obs: np.ndarray
    tmax_fcst: np.ndarray
    tmin_obs: np.ndarray
    tmin_fcst: np.ndarray
    r24_obs: np.ndarray
    r24_fcst: np.ndarray
    n_temp_hours: np.ndarray
    n_rain_hours: np.ndarray

    @property
    def n(self) -> int:
        return int(self.model.size)


def empty_daily_facts() -> DailyFacts:
    return DailyFacts(
        station=np.zeros(0, dtype=np.int32), model=np.zeros(0, dtype=np.int32),
        valid_day=np.zeros(0, dtype=np.int32),
        run_minute=np.zeros(0, dtype=np.int64), day_index=np.zeros(0, dtype=np.int8),
        band=np.zeros(0, dtype=np.int8), tmax_obs=np.zeros(0),
        tmax_fcst=np.zeros(0), tmin_obs=np.zeros(0), tmin_fcst=np.zeros(0),
        r24_obs=np.zeros(0), r24_fcst=np.zeros(0),
        n_temp_hours=np.zeros(0, dtype=np.int16),
        n_rain_hours=np.zeros(0, dtype=np.int16))


def daily_owner_index(hourly: HourlyFacts, min_hours: int = 20) -> np.ndarray:
    """把逐小时事实裁成"每个 (源, 站, 有效日)"的**唯一一条完整日序列**。

    两个要求同时成立，裁决只有一种：

    1. **同一物理日不许被多个批次重复计票**：日最高/最低/日累计是同一个观测对象，
       把 `ℓ=24h` 与 `ℓ=48h` 各算一条日事实等于对同一场高温计两遍票。
    2. **被选中的那一条必须真的是"一整天的序列"**：日聚合的门槛是
       `daily_min_hours = 20`，若选中的批次只覆盖该日的 1~2 个小时（抓取时刻附近
       的残段），整日会被门槛拒收——实测按"最小 k"选批次时，每个日行只剩 1 个
       降水小时，`r24` 整张表变成 NaN。

    所以选取规则是**两级排序**：

    1. **完整性优先**：只考虑覆盖小时数 ≥ `daily_min_hours` 的批次——不完整的批次
       无论多"新"都不能当一整天的证据。
    2. 在完整批次里取 **最小 `k`**（信息最超前、且与 `pairtable.min_lead_select`
       同一裁决），`k` 相同再取更晚的抓取。

    顺序不能反：先取最小 `k` 会把"抓取时刻附近的 2 小时残段"选进来，整日被门槛
    拒收（实测 `r24` 全 NaN）；先取最大覆盖又会让每一天都落到最晚的批次上，
    `k` 全部挤进第 2 天，"明天的预报"这一档就没有样本了。
    """
    if hourly.n == 0:
        return np.zeros(0, dtype=bool)
    ok = hourly.temp_ok | hourly.rain_ok
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return np.zeros(hourly.n, dtype=bool)
    day = hourly.valid_day[idx]
    model = hourly.model[idx]
    station = hourly.station[idx]
    run = hourly.run_minute[idx]
    # 每 (源, 站, 日, 批次) 覆盖的小时数
    # 位域：源(8) | 站(8) | 日(24) | 批次(24)
    grp_key = (model.astype(np.int64) << 56) | (station.astype(np.int64) << 48) \
        | ((day.astype(np.int64) & 0xFFFFFF) << 24) \
        | (run.astype(np.int64) & 0xFFFFFF)
    uniq, counts = np.unique(grp_key, return_counts=True)
    # 每个 (源, 站, 日) 在各批次下的代表 k（该批次在这一天里占多数小时的 k）
    # 代表 k 取该批次在这一天里**多数小时所属**的 k（日尺度），不是最小 k：
    # 一段 16 天序列里，目标日的 24 个小时跨两个 24h 窗口（`t_run` 当天的部分小时
    # 是 k=1，其余是 k=2），最小 k 会让整日都归进第 1 天，第 2 天往后没有样本。
    # 日尺度的正确问题是"这一天主要兑现于哪一档"，因此按小时数取众数。
    codes, inv_g = np.unique(grp_key, return_inverse=True)
    order_g = np.lexsort((hourly.day_index[idx], inv_g.ravel()))
    sorted_inv = inv_g.ravel()[order_g]
    sorted_k = hourly.day_index[idx][order_g]
    starts = np.flatnonzero(np.r_[True, sorted_inv[1:] != sorted_inv[:-1]])
    ends = np.r_[starts[1:], sorted_inv.size]
    k_by_group = np.zeros(codes.size, dtype=np.int64)
    for st, en in zip(starts, ends):
        vals, cnts = np.unique(sorted_k[st:en], return_counts=True)
        k_by_group[sorted_inv[st]] = int(vals[int(np.argmax(cnts))])
    # **完整性是硬门槛，不是平手裁决**：一个目标日会被 15 个批次报到，其中
    # `t_run` 就在该日 00 时的那个批次只有 `ℓ ≥ 1h` 的 1 个小时落在这天（其余
    # 都算下一天），若它在"最小 k"里胜出，整日只剩 1 个小时，必然被
    # `daily_min_hours` 拒收。实测这样会让 124/162 个日行变成 NaN。
    # 所以只有覆盖 ≥ min_hours 的批次才是候选；候选里再按 k 小（信息最新）取胜。
    complete = counts >= max(1, int(min_hours))
    c_used = counts
    day_key = uniq >> 24
    # 排序键（lexsort 的最后一个键是主键）：日 → k 小 → 抓取晚。
    # **k 必须排在完整性之前**：某一天的"信息最新的一次完整预报"往往来自前一天
    # 00:00 的抓取（那时它给出该日 24 小时，k=1）；若让完整性先决，"当天 00:00 的
    # 抓取只覆盖 1 小时却 k=1"会胜出，而它凑不满 `daily_min_hours`，整日被拒收。
    # 所以先按 k 选出"最早给出这一天的批次"，再在其中按覆盖小时数打破平手。
    order = np.lexsort((-(uniq & 0xFFFFFF), -c_used, k_by_group, day_key))
    order = order[complete[order]] if complete.any() else order
    first = np.empty(order.size, dtype=bool)
    first[0] = True
    first[1:] = day_key[order][1:] != day_key[order][:-1]
    owner = uniq[order][first]
    keep = np.isin(grp_key, owner)
    mask = np.zeros(hourly.n, dtype=bool)
    mask[idx[keep]] = True
    return mask


def build_daily_facts(hourly: HourlyFacts, *, daily_min_hours: int = 20) -> DailyFacts:
    """逐小时事实 → 日极值 / 24h 累计（主分只认逐小时聚合）。

    **为什么必须两侧同集合**：若预报侧与观测侧各用自己的非缺测小时求日最高/最低，
    当"预报缺的恰好是最热的 4 个小时"时预报的日最高会偏低——那不是预报误差，
    是缺测伪装成了技巧。所以聚合只在**两侧都非缺测**的小时上做，共同小时数不足
    `daily_min_hours` 整日弃用（缺测绝不折算成 0，也绝不折算成极值）。

    **为什么原生日产品不进主分**：日产品覆盖 00–24 的整日窗口，与逐小时聚合的
    "共同小时集合"不是同一个观测对象；混算会让部分源的日极值来自一个不同口径的
    交付物，读者的"日最高"逐行不同义。日产品只作为混合口径敏感性对照单独产出。
    """
    if hourly.n == 0:
        return empty_daily_facts()
    owner = daily_owner_index(hourly, min_hours=daily_min_hours)
    sub = hourly.take(owner)
    if sub.n == 0:
        return empty_daily_facts()
    # 键里**不含 `band`**：日极值 / 日累计是整天的物理量，必须把该自然日的全部
    # 小时都聚合进来。若按 (源, 站, 日, 段) 分组，每个段只剩该日的一部分小时
    # （`ℓ=15h` 与 `ℓ=280h` 各覆盖不同的钟点），`daily_min_hours=20` 永远凑不满，
    # 整张日尺度表会全空（实测 r24 有效样本从 2,156 掉到 0）。
    # 段的归属由该日**代表批次**的 `k` 决定，见下面的 `k_of`。
    keys, inv = np.unique(
        np.stack([sub.model.astype(np.int64), sub.station.astype(np.int64),
                  sub.valid_day.astype(np.int64)], axis=1),
        axis=0, return_inverse=True)
    inv = inv.ravel()
    m = len(keys)
    t_ok = sub.temp_ok
    r_ok = sub.rain_ok
    need = max(1, int(daily_min_hours))

    # --- 温度：共同小时集合上的 max/min
    ti = np.flatnonzero(t_ok)
    tmax_o = np.full(m, -np.inf)
    tmin_o = np.full(m, np.inf)
    tmax_f = np.full(m, -np.inf)
    tmin_f = np.full(m, np.inf)
    if ti.size:
        gi = inv[ti]
        np.maximum.at(tmax_o, gi, sub.temp_obs[ti])
        np.maximum.at(tmax_f, gi, sub.temp_fcst[ti])
        np.minimum.at(tmin_o, gi, sub.temp_obs[ti])
        np.minimum.at(tmin_f, gi, sub.temp_fcst[ti])
    cnt_t = np.bincount(inv[ti], minlength=m) if ti.size else np.zeros(m, dtype=int)
    temp_full = cnt_t >= need

    # --- 降水：共同小时集合上的累计
    ri = np.flatnonzero(r_ok)
    sum_o = np.zeros(m)
    sum_f = np.zeros(m)
    if ri.size:
        gi = inv[ri]
        np.add.at(sum_o, gi, sub.rain_obs[ri])
        np.add.at(sum_f, gi, sub.rain_fcst[ri])
    cnt_r = np.bincount(inv[ri], minlength=m) if ri.size else np.zeros(m, dtype=int)
    rain_full = cnt_r >= need

    # --- 段的归属：该日代表批次（最小 k）的 k 与 band
    first_row = np.zeros(m, dtype=np.int64)
    seen = np.zeros(m, dtype=bool)
    for row in range(sub.n):
        g = int(inv[row])
        if not seen[g]:
            seen[g] = True
            first_row[g] = row
    # 段的归属按**代表批次在该目标日的有效时效**计算（与 `build_hourly_facts` 的
    # `k = ⌈ℓ/24⌉` 同一公式、同一零点）：`ℓ = 该目标日 00:00 − t_run`。
    #
    # 不能用"该批次日内多数小时的 k"：一段 16 天序列里同一天的 24 个小时会跨两个
    # 24h 窗口，取众数会让全部 81 天都落进第 1 段（实测 `k dist = {1: 81}`），
    # 日尺度维度塌缩成一个数。日尺度问的是"这一整天提前几天给出"，所以用日界的
    # 时效。
    run_of = sub.run_minute[first_row]
    day_midnight = keys[:, 2].astype(np.int64) * 1440 - _MIN_ORIGIN_ORD * 1440
    # ℓ 以小时计的定点除法：+1h 的协议下限会把"当日 00:00 起报、目标 00:00"这种
    # 边界算成 k=0；用 `ceil((Δh)/24)` 且 Δh ≥ 1 即得 k ≥ 1。
    lead_h = (day_midnight - run_of) // 60
    lead_h = np.maximum(lead_h, 1)
    k_of = np.clip(np.ceil(lead_h / 24.0).astype(np.int64), 0, 16).astype(np.int8)
    b_of = np.where((k_of >= 1) & (k_of <= 16),
                    BAND_OF[np.clip(k_of, 0, 16)], -1).astype(np.int8)
    return DailyFacts(
        station=keys[:, 1].astype(np.int32),
        model=keys[:, 0].astype(np.int32),
        valid_day=keys[:, 2].astype(np.int32),
        run_minute=run_of.astype(np.int64),
        day_index=k_of,
        band=b_of,
        tmax_obs=np.where(temp_full, tmax_o, np.nan),
        tmax_fcst=np.where(temp_full, tmax_f, np.nan),
        tmin_obs=np.where(temp_full, tmin_o, np.nan),
        tmin_fcst=np.where(temp_full, tmin_f, np.nan),
        r24_obs=np.where(rain_full, sum_o, np.nan),
        r24_fcst=np.where(rain_full, sum_f, np.nan),
        n_temp_hours=cnt_t.astype(np.int16),
        n_rain_hours=cnt_r.astype(np.int16),
    )


@dataclass
class ThreeHourFacts:
    """北京时固定 3h 窗的降水累计（02/05/…/23 时止）。"""

    station: np.ndarray
    model: np.ndarray
    valid_day: np.ndarray
    run_minute: np.ndarray
    end_hour: np.ndarray
    day_index: np.ndarray
    band: np.ndarray
    r3_obs: np.ndarray
    r3_fcst: np.ndarray
    n_hours: np.ndarray

    @property
    def n(self) -> int:
        return int(self.model.size)


#: 北京时固定 3h 窗的结束钟点映射（0、1 时不属于任何完整窗）。
_HOUR_END = np.array([-1, -1, 2, 2, 2, 5, 5, 5, 8, 8, 8, 11, 11, 11,
                      14, 14, 14, 17, 17, 17, 20, 20, 20, 23], dtype=np.int8)


def empty_3h_facts() -> ThreeHourFacts:
    z = np.zeros(0)
    return ThreeHourFacts(
        station=np.zeros(0, dtype=np.int32), model=np.zeros(0, dtype=np.int32),
        valid_day=np.zeros(0, dtype=np.int32), run_minute=np.zeros(0, dtype=np.int64),
        end_hour=np.zeros(0, dtype=np.int8),
        day_index=np.zeros(0, dtype=np.int8), band=np.zeros(0, dtype=np.int8),
        r3_obs=z.copy(), r3_fcst=z.copy(),
        n_hours=np.zeros(0, dtype=np.int16))


def build_3h_facts(hourly: HourlyFacts) -> ThreeHourFacts:
    """逐小时事实 → 3h 窗累计。

    **为什么避开逐小时降水**：逐小时确定性预报的峰值时刻哪怕只早 1 小时，也会在
    "峰值小时报空"与"邻小时空报"上被罚两次——这不是两次错误，是一次时间移位。
    3h 累计把纯时间移位吸收掉，同时保留"这场雨落在哪个 3 小时"这一决策真正需要
    的分辨率，并且与原生后向 3 小时产品天然同窗。

    **只认完整的窗**：3 个小时必须都成对，否则"缺一小时"会被当成"那一小时没下雨"，
    累计量偏低却被记成预报偏多/实况偏少——缺测又变回了技巧。
    """
    if hourly.n == 0:
        return empty_3h_facts()
    end_hour = _HOUR_END[np.clip(hourly.hour_bjt, 0, 23)]
    mask = hourly.rain_ok & (end_hour >= 0)
    if not mask.any():
        return empty_3h_facts()
    sub = hourly.take(mask)
    eh = end_hour[mask]
    owner = _window_owner(sub, eh)
    sub = sub.take(owner)
    eh = eh[owner]
    if sub.n == 0:
        return empty_3h_facts()
    keys, inv = np.unique(
        np.stack([sub.model.astype(np.int64), sub.station.astype(np.int64),
                  sub.valid_day.astype(np.int64), eh.astype(np.int64)], axis=1),
        axis=0, return_inverse=True)
    inv = inv.ravel()
    m = len(keys)
    sum_o = np.zeros(m)
    sum_f = np.zeros(m)
    np.add.at(sum_o, inv, sub.rain_obs)
    np.add.at(sum_f, inv, sub.rain_fcst)
    cnt = np.bincount(inv, minlength=m)
    full = cnt >= 3
    k_of = np.zeros(m, dtype=np.int8)
    b_of = np.full(m, -1, dtype=np.int8)
    run_of = np.zeros(m, dtype=np.int64)
    seen: set[int] = set()
    for row in range(sub.n):
        g = int(inv[row])
        if g not in seen:
            seen.add(g)
            k_of[g] = sub.day_index[row]
            b_of[g] = sub.band[row]
            run_of[g] = sub.run_minute[row]
    return ThreeHourFacts(
        station=keys[:, 1].astype(np.int32),
        model=keys[:, 0].astype(np.int32),
        valid_day=keys[:, 2].astype(np.int32),
        run_minute=run_of,
        end_hour=keys[:, 3].astype(np.int8),
        day_index=k_of,
        band=b_of,
        r3_obs=np.where(full, sum_o, np.nan),
        r3_fcst=np.where(full, sum_f, np.nan),
        n_hours=cnt.astype(np.int16),
    )


def _window_owner(sub: HourlyFacts, end_hour: np.ndarray) -> np.ndarray:
    """每个 (源, 站, 有效日, 窗) 只保留一个批次的 3 小时（最小 k，同 k 取更晚批次）。"""
    key = (sub.model.astype(np.int64) * 67_108_864
           + sub.station.astype(np.int64) * 2_097_152
           + sub.valid_day.astype(np.int64) * 32
           + end_hour.astype(np.int64))
    order = np.lexsort((-sub.run_minute, sub.day_index, key))
    s_key = key[order]
    first = np.empty(order.size, dtype=bool)
    first[0] = True
    first[1:] = s_key[1:] != s_key[:-1]
    mask = np.zeros(sub.n, dtype=bool)
    mask[order[first]] = True
    return mask

"""DVS v6 引擎：把协议层、基准层、度量层、推断层组装成一份可发布的榜单。

**分层与本模块的位置**

```
  raw snapshots / obs           (storage.py)
        │
  dvs_protocol   as-of 批次 → 逐小时/日尺度/3h 配对事实
        │
  dvs_reference  L0 因果物理基准（逐 t_run 的 δ、a₀、τ）
        │
  dvs_stats_store  → 按日、按批次的**可加充分统计量仓**（唯一事实源）
        │
  dvs           温度技能 / Murphy / 降水 V̄⁺ / 合成
        │
  dvs_uncertainty 配对块自助 + MCS + 排名区间 + 技巧寿命 + 分半信度
        │
  dvs_engine     ← 你在这里：一行榜、一张可复算的表
```

**为什么引擎里没有一条公式**

引擎只做编排：①按站跑协议与基准；②把统计量归档；③对任意窗口做纯函数求和并套公式；
④把推断层的结果贴回榜行。"换权重、换画像、换窗口"因此都不需要重跑数据——这正是
方案第六支柱（唯一事实源）的工程含义。任何一条公式被写进引擎，都会立刻出现"榜单
上的分"与"读者手算的分"两条实现，而这正是旧口径最难维护的地方。

**评测窗口与证据等级**

* 主评窗口 `W` 默认是"数据起点 → 现在"（与 `live_window_start` 对齐）；
* 每条源于窗口内的独立验证日数 `n_d` 与可用率决定证据等级 A/B/C/D
  （阈值见 `EVIDENCE_TIERS`）；
* C 级（新接入、`n_d` 不足半个窗口）**不直接混排**，而是通过固定免费锚点 `L1`
  做同日配对时期校准（`dvs.l1_align`）。它不接触任何其他竞争源，因此
  "增删榜上任意源，本源分数逐位不变"仍然成立——这是它相对"双向加法难度对齐 +
  经验贝叶斯收缩"的根本区别：后者必须看同台全部选手。
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field, fields
from typing import Mapping, Sequence

import numpy as np

from . import dvs
from .dvs_controls import (
    CONTROLS, build_control_facts, control_model_codes, control_name, is_control,
)
from .dvs_protocol import (
    BAND_BOUNDS, DailyFacts, HourlyFacts, ObsColumn, assign_runs,
    build_daily_facts, build_hourly_facts,
)
from .dvs_reference import (
    DEFAULT_TAU_HOURS, ERA5LandClimatology, StationClimatology, StationSeries,
    build_l0_baseline, build_obs_climatology,
)
from .dvs_stats_store import (
    SourceDayRows, accumulate_daily, accumulate_day, aggregate_window,
    day_weight_decay, event_days_by_target,
)
from .dvs_uncertainty import (
    BOOTSTRAP_SEED, DEFAULT_RUNS, MIN_BLOCK_DAYS, mcs_tiers, paired_bootstrap,
    skill_lifetime,
)

logger = logging.getLogger(__name__)

#: 证据等级阈值（白皮书 §4.2）。
EVIDENCE_TIERS = {
    "A": {"min_day_frac": 0.75, "min_avail": 0.80},
    "B": {"min_day_frac": 0.50, "min_avail": 0.50},
    "C": {"min_days": 14},
}

#: 永久免费的固定锚点（时期校准用）。它必须全时段稳定运行且覆盖 1–16 天。
DEFAULT_L1 = "ecmwf_ifs025"

#: 一条源在窗口内的最少可用天数，低于它不进榜（D 级）。
MIN_DAYS_FOR_BOARD = 14

#: 温度统计量在落盘数组中的位序（与 `dvs_stats_store.TEMP_KEYS` 严格一致）。
_TEMP_FIELD_ORDER = ("n", "sum_abs_e", "sum_sq_e", "sum_abs_e0", "sum_sq_e0",
                     "sum_f", "sum_o", "sum_f_sq", "sum_o_sq", "sum_fo")


@dataclass
class DvsConfig:
    """DVS v6 的全部配置（默认值即白皮书 §6 的冻结参数）。"""

    alpha_v: float = dvs.ALPHA_V
    band_weights: tuple[float, ...] = dvs.BAND_WEIGHTS
    loss_shape: tuple[float, float] = dvs.LOSS_SHAPE
    skill_floor: float = dvs.SKILL_FLOOR
    cost_loss_prior: tuple[float, float] = dvs.COST_LOSS_PRIOR
    cost_loss_grid: int = dvs.COST_LOSS_GRID
    rain_events: Mapping[str, Sequence[tuple[float, float]]] = field(
        default_factory=lambda: dict(dvs.RAIN_EVENTS))
    rain_shrink_n0_days: float = dvs.RAIN_SHRINK_N0_DAYS
    scale_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(dvs.SCALE_WEIGHTS))
    # 协议常数
    min_lead_h: int = 1
    obs_latency_h: int = 3
    daily_min_hours: int = 20
    batch_cluster_min: int = 30
    max_lead_days: int = 16
    # 基准常数
    offset_days: int = 30
    recent_hours: int = 3
    climatology_path: str | None = None
    # 推断常数
    bootstrap_runs: int = DEFAULT_RUNS
    min_block_days: int = MIN_BLOCK_DAYS
    mcs_level: float = 0.10
    mcs_max_tiers: int = 5
    seed: int = BOOTSTRAP_SEED
    # 证据与窗口
    l1_baseline: str = DEFAULT_L1
    rolling_days: int | None = None
    recent_halflife_days: float = 45.0
    enable_controls: bool = True
    enable_bootstrap: bool = True
    #: 源能力矩阵：`{源: {"max_lead_days": int, "temp": bool, "rain": bool}}`。
    #: 缺省表示"按实际数据推断能力"（不声明即不裁）。
    capability: Mapping[str, Mapping] = field(default_factory=dict)

    @classmethod
    def from_eval_cfg(cls, eval_cfg: Mapping | None) -> "DvsConfig":
        """从 `stations.yaml` 的 `eval.dvs` 段构造（缺项一律回落到声明默认值）。

        缺段时也走 `config.DEFAULT_DVS`（而不是只依赖 dataclass 默认值）：
        那是 dvs-v6 参数的**唯一声明处**，dataclass 的字段缺省只是它的镜像。
        """
        from .config import DEFAULT_DVS
        node = {**DEFAULT_DVS, **dict((eval_cfg or {}).get("dvs") or {})}
        kw: dict = {}
        simple = {
            "alpha_v": float, "loss_shape": tuple, "skill_floor": float,
            "cost_loss_grid": int, "rain_shrink_n0_days": float,
            "min_lead_h": int, "obs_latency_h": int, "daily_min_hours": int,
            "batch_cluster_min": int, "max_lead_days": int, "offset_days": int,
            "recent_hours": int, "bootstrap_runs": int, "min_block_days": int,
            "mcs_level": float, "mcs_max_tiers": int, "seed": int,
            "rolling_days": int, "recent_halflife_days": float,
            "enable_controls": bool, "enable_bootstrap": bool,
        }
        for key, cast in simple.items():
            if key in node and node[key] is not None:
                kw[key] = cast(node[key])
        if node.get("band_weights"):
            w = [float(x) for x in node["band_weights"]]
            if len(w) != len(BAND_BOUNDS):
                raise ValueError(
                    f"eval.dvs.band_weights 必须是 {len(BAND_BOUNDS)} 个权重，"
                    f"收到 {len(w)} 个")
            kw["band_weights"] = tuple(w)
        if node.get("cost_loss_prior"):
            p = node["cost_loss_prior"]
            kw["cost_loss_prior"] = (float(p[0]), float(p[1]))
        if node.get("rain_events"):
            kw["rain_events"] = {
                k: tuple((float(t), float(w)) for t, w in v)
                for k, v in node["rain_events"].items()}
        if node.get("scale_weights"):
            kw["scale_weights"] = {k: float(v)
                                   for k, v in node["scale_weights"].items()}
        if node.get("l1_baseline"):
            kw["l1_baseline"] = str(node["l1_baseline"])
        if node.get("climatology_path"):
            kw["climatology_path"] = str(node["climatology_path"])
        if node.get("capability"):
            kw["capability"] = dict(node["capability"])
        return cls(**kw)


# ---------------------------------------------------------------- 单站构建
@dataclass
class StationBuild:
    """一个站的构建产物。"""

    station: str
    rows: dict[str, dict[int, SourceDayRows]]
    climatology_source: str
    tau_hours: float
    tau_fitted: bool
    n_facts: int
    n_runs: int
    observed_days: int
    declared_days: int = 0


def load_climatology(config: DvsConfig, root=None) -> ERA5LandClimatology | None:
    """载入静态气候表（不存在则返回 None，由调用方退回观测气候态并披露）。"""
    if not config.climatology_path:
        return None
    p = config.climatology_path
    if root is not None and not str(p).startswith("/"):
        from pathlib import Path
        p = Path(root) / p
    clim = ERA5LandClimatology.load(p)
    return clim if clim.available else None


def build_station(station: str, obs_map: Mapping[str, dict],
                  snapshots: Mapping[str, Sequence[dict]], *,
                  models: Sequence[str], station_code: int,
                  start_day: int, end_day: int,
                  config: DvsConfig,
                  climatology: StationClimatology | None = None,
                  include_controls: bool = True,
                  ) -> StationBuild:
    """单站全链路：批次 → 事实 → L0 → 逐日统计量。

    **基准逐行算**：L0 的 `δ / a₀` 依赖 `t_run`，同一有效时刻在不同批次下的基准
    值不同。引擎在建立事实的同一次遍历里把基准算出来（逐批次），而不是事后用一个
    "平均基准"去补——两者对长时效的差别可达数度。
    """
    obs_col = ObsColumn.from_map(obs_map)
    series = StationSeries.from_map(obs_map)
    clim = climatology or build_obs_climatology(station, series.pairs())
    model_codes = {m: i for i, m in enumerate(models)}
    runs = assign_runs(snapshots, station=station,
                       batch_cluster_min=config.batch_cluster_min)
    facts = build_hourly_facts(
        runs, obs=obs_col, model_codes=model_codes, station_code=station_code,
        start_day=start_day, end_day=end_day, min_lead_h=config.min_lead_h,
        max_lead_days=config.max_lead_days)
    build = StationBuild(
        station=station, rows={},
        climatology_source=(clim.source if clim else "none"),
        tau_hours=(clim.tau_hours if clim else DEFAULT_TAU_HOURS),
        tau_fitted=(clim.tau_fitted if clim else False),
        n_facts=facts.n, n_runs=len(runs), observed_days=0,
        declared_days=_declared_days(obs_map),
    )
    if facts.n == 0 or clim is None:
        return build
    # 观测实际存在的自然日：由**真实源**事实算（绝不能用含对照源的 facts_all —
    # 对照源会把每一天都填满，可用率分母就假了）。提前算出并在合并后释放母本，
    # 见下方 `del` 处的说明。
    observed_days = len({int(d) for d in np.unique(facts.valid_day)})
    # 对照源与真实源共用同一批观测与协议护栏（见 dvs_controls）。对照源的模型编码
    # 由 `control_model_codes` 从 100 起编，直接整体并入同一套 model_codes，
    # 这样 `build_control_facts` 里写的 100+i 与这里的名字一一对应。
    if include_controls and config.enable_controls:
        controls = build_control_facts(facts, seed=config.seed)
        all_models = list(models) + list(control_model_codes())
        # 真实源用 0..N−1，对照源用 100+i（`build_control_facts` 里写死的编码）。
        # 让编码**等于列表下标**，统计量层就能直接用 `all_models[code]` 反查名字。
        all_codes = {m: i for i, m in enumerate(all_models)}
        # 两套编码空间分别翻译：真实源的旧码 = `model_codes` 里的下标，
        # 对照源的旧码 = 100+i；新码一律是 `all_models` 里的下标。
        facts_real = _recode(facts, {model_codes[m]: all_codes[m] for m in models})
        facts_ctl = _recode(controls, {
            control_model_codes()[control_name(c.key)]: all_codes[control_name(c.key)]
            for c in CONTROLS})
        facts_all = facts_real + facts_ctl
        # 合并后母本立刻放手。`_recode` 只换 `model` 一列（其余列是**同一份**数组
        # 的引用），于是"共享"正是母本不死的根：不显式删，真实源与对照源的 9 份
        # 列式事实会一直活到本函数返回，与 facts_all / daily / 累积结果叠成单站
        # ~2.8 GB 的峰值（4 站并发实测 8.5 GB）。缓存里那份快照已由
        # `_preloaded_inputs` 复用，事实层再留两份就是纯浪费。
        del facts_real, facts_ctl, facts, controls
    else:
        all_models = list(models)
        all_codes = model_codes
        facts_all = facts
    base_all = _compute_baselines(facts_all, series, clim, config, all_codes)
    daily = build_daily_facts(facts_all, daily_min_hours=config.daily_min_hours)
    _fix_daily_buckets(daily)
    base_daily = _daily_baselines(daily, series, clim, config)
    rows = accumulate_day(facts_all, base=base_all,
                          station_ids=[station], models=all_models,
                          events=config.rain_events)
    # 逐小时事实与它的基准到此为止（日轨累积只读 `daily` / `base_daily`）：
    # 先放手再进下一步，别让 7.6M 行的逐小时列与日聚合、累积字典同时驻留。
    del facts_all, base_all
    accumulate_daily(rows, daily, base=base_daily, station_ids=[station],
                     models=all_models, events=config.rain_events)
    build.rows = rows
    build.observed_days = observed_days
    return build


def _recode(facts: HourlyFacts, codes: Mapping[str, int]) -> HourlyFacts:
    """把模型编码重排成"列表下标"（入参是 `{旧编码: 新编码}`）。

    **必须按事实对象自己的编码空间解码**：`facts`（真实源）用 0..N−1，
    `controls` 用 100+i——两套空间，绝不能对合并后的数组套同一张映射表。
    未知编码映射到一个不可能被 `models` 列表索引到的哨兵值，由统计层跳过。
    """
    sentinel = len(codes) + 1000
    mapped = np.full(facts.model.shape, sentinel, dtype=np.int32)
    # codes 的键是名字、值是**新**编码；旧编码由调用方另行给出
    inverse = {int(old): int(new) for old, new in codes.items()}
    for old_code, new_code in inverse.items():
        mapped[facts.model == old_code] = new_code
    return HourlyFacts(**{f.name: (mapped if f.name == "model"
                                  else getattr(facts, f.name))
                          for f in fields(facts)})


def _declared_days(obs_map: Mapping[str, dict]) -> int:
    return len({k[:10] for k in obs_map if isinstance(k, str) and len(k) >= 10})


def _fix_daily_buckets(daily: DailyFacts) -> None:
    """`k` 必须落在 1..16；越界样本显式标成 `band = -1` 由下游拒收。

    绝不静默截到 16：把"其实是第 17 天"的样本记成"第 16 天"会让长时效分段
    混进不该有的样本，而报告上完全看不出来。
    """
    daily.band = np.where((daily.day_index >= 1) & (daily.day_index <= 16),
                          daily.band, -1).astype(np.int8)


def _compute_baselines(facts: HourlyFacts, series: StationSeries,
                       clim: StationClimatology, config: DvsConfig,
                       codes: Mapping[str, int] | None = None) -> np.ndarray:
    """逐行 L0 基准温度（与 `facts` 的行一一对应；无基准的行填 NaN）。"""
    _ = codes  # 基准只依赖 (有效时刻, 时效)，与源编码无关
    base = np.full(facts.n, np.nan)
    if facts.n == 0:
        return base
    obs_series = series.temp_series()
    for run_min in np.unique(facts.run_minute):
        m = facts.run_minute == run_min
        l0 = build_l0_baseline(
            clim.station, clim, obs_series, int(run_min),
            offset_days=config.offset_days, recent_hours=config.recent_hours,
            obs_latency_h=config.obs_latency_h)
        if l0 is None:
            continue
        idx = np.flatnonzero(m)
        base[idx] = l0.hourly_temp(facts.valid_minute[idx],
                                   facts.lead_hours[idx].astype(float))
    return base


def _daily_baselines(daily: DailyFacts, series: StationSeries,
                     clim: StationClimatology, config: DvsConfig
                     ) -> dict[str, np.ndarray]:
    """日尺度基准（`tmax` / `tmin`），按**代表批次**的 `t_run` 算。"""
    out = {"tmax": np.full(daily.n, np.nan), "tmin": np.full(daily.n, np.nan)}
    if daily.n == 0:
        return out
    obs_series = series.temp_series()
    for run_min in np.unique(daily.run_minute):
        m = daily.run_minute == run_min
        l0 = build_l0_baseline(
            clim.station, clim, obs_series, int(run_min),
            offset_days=config.offset_days, recent_hours=config.recent_hours,
            obs_latency_h=config.obs_latency_h)
        if l0 is None:
            continue
        idx = np.flatnonzero(m)
        out["tmax"][idx] = l0.daily_temp(daily.valid_day[idx],
                                         daily.day_index[idx], "max")
        out["tmin"][idx] = l0.daily_temp(daily.valid_day[idx],
                                         daily.day_index[idx], "min")
    return out


# ---------------------------------------------------------------- 窗口评分
@dataclass
class WindowScore:
    """一条源在一个窗口内的全部分数与分解（榜行的数据源）。"""

    model: str = ""
    dvs: float | None = None
    temp_bands: dict[int, float | None] = field(default_factory=dict)
    precip_bands: dict[int, float | None] = field(default_factory=dict)
    temp_targets: dict[str, dict[int, float | None]] = field(default_factory=dict)
    precip_events: dict[str, dict[int, dict]] = field(default_factory=dict)
    coverage_score: float | None = None
    coverage_bands: int = 0
    evidence_days: int = 0
    temp_n: float = 0.0
    rain_n: float = 0.0
    murphy: dict[int, dict] = field(default_factory=dict)
    skill_floor_hits: int = 0
    temp_coverage: dict[int, float] = field(default_factory=dict)
    precip_coverage: dict[int, float] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "dvs": self.dvs,
            "T_b": {b: self.temp_bands.get(b) for b in sorted(self.temp_bands)},
            "P_b": {b: self.precip_bands.get(b) for b in sorted(self.precip_bands)},
            "S_over_cover": self.coverage_score,
            "cover_bands": self.coverage_bands,
            "days": self.evidence_days,
            "temp_n": self.temp_n,
            "rain_n": self.rain_n,
            "murphy": self.murphy,
            "skill_floor_hits": self.skill_floor_hits,
            "temp_coverage": self.temp_coverage,
            "precip_coverage": self.precip_coverage,
        }


def score_window(days: Mapping[int, SourceDayRows], *,
                 config: DvsConfig,
                 day_weights: Mapping[int, float] | None = None) -> WindowScore:
    """把一段逐日统计量打成一条源的窗口分数（纯函数）。

    `day_weights` 同时服务三件事：自助重采样（整数次数）、近期加权（指数衰减）、
    "只看某几天"的敏感性走廊。三者共用一条代码路径，因为数学语义完全相同——
    **对统计量做加权求和**。
    """
    if not days:
        return WindowScore()
    return _score_from_aggregate(aggregate_window(days, day_weights), config,
                                 days=days)


def _score_from_aggregate(agg, config: DvsConfig,
                          days: Mapping[int, SourceDayRows] | None = None
                          ) -> WindowScore:
    """由**已聚合**的窗口统计量打分（点估计与自助共用这一个函数）。

    把"怎么取行/加权"与"怎么套公式"分开，是为了让自助能换掉前者（预排序、
    预缩放）却**不可能**换掉后者。公式只有一份，偏差也就无从产生。
    返回完整的 `WindowScore`（自助取其中的 `.dvs`）。

    `days` 只在需要"按自然日去重"的量（降水独立事件日数 `n_ev`）时使用：
    它是**不可加**的，必须回到逐日行去重，不能由窗口聚合数反推。
    """
    out = WindowScore(evidence_days=len(agg.days_present))
    # ---- 温度：同一 (站, 目标, 段) 的统计量先按站合并，再算比率
    by_target: dict[tuple[str, int], list[float]] = {}
    for (_run, _station, target, band), vals in agg.temp.items():
        key = (target, band)
        cur = by_target.get(key)
        if cur is None:
            by_target[key] = list(vals)
        else:
            for i, v in enumerate(vals):
                cur[i] += v
    temp_targets: dict[str, dict[int, float | None]] = {
        t: {} for t in dvs.TEMP_TARGETS}
    for (target, band), vals in by_target.items():
        a = dvs.TempAggregate(**dict(zip(_TEMP_FIELD_ORDER, vals)))
        skill = dvs.temp_skill(a, loss_shape=config.loss_shape,
                               floor=config.skill_floor)
        raw = dvs.temp_skill(a, loss_shape=config.loss_shape, floor=-math.inf)
        if raw is not None and raw < config.skill_floor:
            out.skill_floor_hits += 1
        temp_targets.setdefault(target, {})[band] = skill
        out.temp_n += a.n
        if target == "temp_h":
            out.murphy[band] = dvs.murphy_decomposition(a)
    out.temp_targets = temp_targets
    # ---- 降水：同一 (目标, 段) 的列联表逐位相加
    by_event: dict[tuple[str, int], list] = {}
    for (_run, _station, target, band), ent in agg.rain.items():
        key = (target, band)
        cur = by_event.get(key)
        if cur is None:
            by_event[key] = [ent[i] for i in range(4)] + [float(ent[4])]
        else:
            for i in range(4):
                cur[i] += ent[i]
            cur[4] += float(ent[4])
    # 独立事件日数在**窗口层**去重（跨段跨批次，见 event_days_by_target）
    ev_days = (event_days_by_target(days, list(dvs.PRECIP_TARGETS))
               if days is not None else {})
    for b in range(1, len(config.band_weights) + 1):
        out.temp_bands[b] = dvs.temp_score(
            {k: (temp_targets.get(k) or {}).get(b) for k in dvs.TEMP_TARGETS})
        out.temp_coverage[b] = dvs.target_coverage(
            {k: (temp_targets.get(k) or {}).get(b) for k in dvs.TEMP_TARGETS})
        per_scale: dict[str, dict] = {}
        for scale in ("r3", "r24"):
            specs = config.rain_events.get(scale) or ()
            counts_by_event = {}
            for thr, _w in specs:
                key = (f"{scale}_{_thr_key(thr)}", b)
                got = by_event.get(key)
                if got is None:
                    continue
                name = f"{scale}_{_thr_key(thr)}"
                arr = np.asarray([got[:4]], dtype=np.int64)
                counts_by_event[name] = dvs.RainCounts(
                    a=arr[:, 0].copy(), b=arr[:, 1].copy(),
                    c=arr[:, 2].copy(), d=arr[:, 3].copy(),
                    n_ev_days=ev_days.get(name, 0.0))
            if not specs:
                per_scale[scale] = {"score": 0.0, "coverage": 0.0, "events": {}}
                continue
            res = dvs.rain_target_score(
                counts_by_event, {scale: specs},
                n0_days=config.rain_shrink_n0_days,
                cost_loss_prior=tuple(config.cost_loss_prior),
                cost_loss_grid=config.cost_loss_grid)
            per_scale[scale] = res
            for k, v in (res.get("events") or {}).items():
                if isinstance(v, dict) and "threshold_mm" in v:
                    out.precip_events.setdefault(k, {})[b] = v
        p = dvs.precip_score(per_scale.get("r3"), per_scale.get("r24"),
                             config.scale_weights)
        out.precip_bands[b] = p["score"]
        out.precip_coverage[b] = p["coverage"]
    comp = dvs.composite_from_parts(out.temp_bands, out.precip_bands,
                                    config.alpha_v, config.band_weights)
    out.dvs = comp["score"]
    cov = dvs.coverage_score(out.temp_bands, out.precip_bands,
                             config.alpha_v, config.band_weights)
    out.coverage_score = cov["score"]
    out.coverage_bands = cov["bands"]
    out.rain_n = float(sum(v[0] + v[1] + v[2] + v[3]
                          for v in agg.rain.values()))
    return out


def _thr_key(thr: float) -> str:
    return f"ge{int(thr)}" if float(thr).is_integer() else f"ge{thr}"


# ---------------------------------------------------------------- 榜面
@dataclass
class BoardEntry:
    """榜面一行（序列化后即 `report["meta"]["dvs"]["board"]` 的成员）。"""

    model: str
    score: WindowScore
    tier: str = "D"
    days_present: int = 0
    window_days: int = 0
    avail: float = 0.0
    aligned: float | None = None
    aligned_note: str = ""
    delta_l1: float | None = None
    delta_l1_ci: tuple[float, float] | None = None
    ci: tuple[float, float] | None = None
    rank_lo: int | None = None
    rank_hi: int | None = None
    p_best: float | None = None
    mcs_tier: int | None = None
    lifetime: dict | None = None
    recent: float | None = None
    delivered: float | None = None
    control: bool = False
    control_label: str = ""
    control_note: str = ""
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "model": self.model,
            "score": self.score.dvs,
            "temp": self.score.temp_bands,
            "precip": self.score.precip_bands,
            "cover": self.score.coverage_score,
            "cover_bands": self.score.coverage_bands,
            "evidence": self.tier,
            "days": self.days_present,
            "window_days": self.window_days,
            "avail": self.avail,
            "ci90": list(self.ci) if self.ci else None,
            "rank": ([self.rank_lo, self.rank_hi]
                     if self.rank_lo is not None else None),
            "p_best": self.p_best,
            "mcs_tier": self.mcs_tier,
            "delta_l1": self.delta_l1,
            "delta_l1_ci": list(self.delta_l1_ci) if self.delta_l1_ci else None,
            "aligned": self.aligned,
            "aligned_note": self.aligned_note,
            "lifetime": self.lifetime,
            "recent": self.recent,
            "delivered": self.delivered,
            "control": self.control,
            "control_label": self.control_label,
            "control_note": self.control_note,
            "errors": self.errors,
            "detail": self.score.as_dict(),
        }


def evidence_tier(days_present: int, window_days: int, avail: float) -> str:
    """证据等级 A/B/C/D（白皮书 §4.2 的阈值，一处定义、全站复用）。"""
    frac = (days_present / window_days) if window_days > 0 else 0.0
    if (frac >= EVIDENCE_TIERS["A"]["min_day_frac"]
            and avail >= EVIDENCE_TIERS["A"]["min_avail"]):
        return "A"
    if (frac >= EVIDENCE_TIERS["B"]["min_day_frac"]
            and avail >= EVIDENCE_TIERS["B"]["min_avail"]):
        return "B"
    if days_present >= EVIDENCE_TIERS["C"]["min_days"]:
        return "C"
    return "D"


def build_board(rows_by_model: Mapping[str, Mapping[int, SourceDayRows]], *,
                config: DvsConfig,
                window_days: Sequence[int],
                declared_days: Mapping[str, int] | None = None,
                obs_days: Mapping[str, set[int]] | None = None,
                models: Sequence[str] | None = None) -> dict:
    """由逐日统计量仓构造整张榜（点估计 + 自助区间 + MCS + 对照行）。

    返回一个可直接序列化进 `report["meta"]["dvs"]` 的 dict：
    `{"board": [...], "tiers": [...], "controls": [...], "l1": "...", "config": {...}}`。
    """
    days = sorted({int(d) for d in window_days})
    n_window = len(days)
    names = list(models) if models is not None else sorted(rows_by_model)
    declared = dict(declared_days or {})
    obs_sets = dict(obs_days or {})
    entries: dict[str, BoardEntry] = {}
    w_days = {d: 1.0 for d in days}
    for model in names:
        per_day = rows_by_model.get(model) or {}
        sc = score_window(per_day, config=config, day_weights=w_days)
        # 段级"能力外记 0"：该源声明/实际不覆盖的段按 0 计（= 退回 L0），
        # 由 `apply_capability` 落实。故障缺测不走这条路（计入可用率）。
        apply_capability(sc, model, config)
        present = sc.evidence_days
        avail = _availability(per_day, days, obs_sets.get(model), declared.get(model))
        entry = BoardEntry(model=model, score=sc, days_present=present,
                           window_days=n_window, avail=avail)
        entry.tier = evidence_tier(present, n_window, avail)
        entry.delivered = dvs.delivered_score(sc.dvs, avail)
        entries[model] = entry
    # ---- 温度/降水维度级覆盖（供 meta 披露）
    # ---- 自助 + MCS（只对非对照源；对照源的意义是标尺，不参与竞争）
    comp_models = [m for m in names if not is_control(m)]
    if config.enable_bootstrap and days:
        matrix, point = _bootstrap_matrix(rows_by_model, comp_models, config,
                                          days)
        if matrix is not None and matrix.size:
            point_only = {m: point[m] for m in comp_models if m in point}
            res = _wrap_bootstrap(matrix, comp_models, point_only, config, days)
            for m in comp_models:
                e = entries[m]
                e.ci = res.interval(m)
                e.rank_lo = res.rank_lo.get(m)
                e.rank_hi = res.rank_hi.get(m)
                e.p_best = res.p_best.get(m)
            tiers = mcs_tiers(point_only, matrix, comp_models,
                              alpha=config.mcs_level,
                              max_tiers=config.mcs_max_tiers)
            tier_of: dict[str, int] = {}
            for i, group in enumerate(tiers, 1):
                for m in group:
                    tier_of[m] = i
            for m in comp_models:
                entries[m].mcs_tier = tier_of.get(m)
    # ---- Δ_L1（配对差：是否值得替代免费默认）
    l1 = config.l1_baseline if config.l1_baseline in entries else None
    if l1 is not None:
        base = entries[l1].score.dvs
        for m, e in entries.items():
            if m == l1 or is_control(m) or e.score.dvs is None or base is None:
                continue
            e.delta_l1 = e.score.dvs - base
    # ---- C 级新源的时期校准（无同台依赖）
    if l1 is not None:
        for m, e in entries.items():
            if e.tier != "C" or is_control(m) or e.score.dvs is None:
                continue
            e.aligned = dvs.l1_align(e.score.dvs, None, entries[l1].score.dvs)
            e.aligned_note = (f"时期对齐临时分：{e.days_present}/{n_window} 天 vs "
                              f"免费基线 {l1}")
    # ---- 近期走势（半衰期指数权重）
    for m, e in entries.items():
        if not days:
            break
        decay = day_weight_decay(days, days[-1], config.recent_halflife_days)
        recent = score_window(rows_by_model.get(m) or {}, config=config,
                              day_weights=decay)
        e.recent = recent.dvs
    # ---- 技巧寿命（逐日累积技能，相对 L0 的"高于 0"链式判据）
    for m, e in entries.items():
        if is_control(m):
            continue
        e.lifetime = _lifetime(rows_by_model.get(m) or {}, config, days)
    ordered = sorted(
        entries.values(),
        key=lambda x: (-(x.score.dvs if x.score.dvs is not None else -1e9),
                       x.model))
    real = [x for x in ordered if not x.control]
    for i, e in enumerate(real, 1):
        e.rank_lo = e.rank_lo if e.rank_lo is not None else i
        e.rank_hi = e.rank_hi if e.rank_hi is not None else i
    controls = []
    for c in CONTROLS:
        name = control_name(c.key)
        e = entries.get(name)
        if e is None:
            controls.append({"model": name, "label": c.label, "note": c.note,
                             "score": None, "temp": None, "precip": None})
            continue
        e.control = True
        e.control_label = c.label
        e.control_note = c.note
        controls.append({
            "model": name, "label": c.label, "note": c.note,
            "score": e.score.dvs,
            "temp": e.score.temp_bands, "precip": e.score.precip_bands,
            "days": e.days_present,
        })
    for c in controls:
        if c["model"] == control_name("l0"):
            # L0 的分数在管道里恒等于 0（基准误差 = 自身误差），这里显式写死并披露，
            # 因为对照源不参与自助矩阵，没有来自数据的行。
            c["score"] = 0.0
            c["temp"] = {b: 0.0 for b in range(1, len(config.band_weights) + 1)}
            c["precip"] = dict(c["temp"])
    return {
        "l1": l1,
        "window": {"start": days[0] if days else None,
                   "end": days[-1] if days else None,
                   "days": n_window},
        "board": [e.as_dict() for e in ordered if not is_control(e.model)],
        "controls": controls,
        "tiers": _tier_summary(entries),
        "config": _config_snapshot(config),
    }


def _tier_summary(entries: Mapping[str, BoardEntry]) -> list[list[str]]:
    buckets: dict[int, list[str]] = {}
    for m, e in entries.items():
        if e.control or e.mcs_tier is None:
            continue
        buckets.setdefault(int(e.mcs_tier), []).append(m)
    return [sorted(buckets[k]) for k in sorted(buckets)]


def _config_snapshot(config: DvsConfig) -> dict:
    return {
        "scheme": "dvs-v6",
        "alpha_v": config.alpha_v,
        "band_bounds": [list(b) for b in BAND_BOUNDS],
        "band_weights": list(config.band_weights),
        "loss_shape": list(config.loss_shape),
        "skill_floor": config.skill_floor,
        "cost_loss_prior": list(config.cost_loss_prior),
        "cost_loss_grid": config.cost_loss_grid,
        "rain_events": {k: [list(x) for x in v]
                        for k, v in config.rain_events.items()},
        "rain_shrink_n0_days": config.rain_shrink_n0_days,
        "scale_weights": dict(config.scale_weights),
        "min_lead_h": config.min_lead_h,
        "obs_latency_h": config.obs_latency_h,
        "daily_min_hours": config.daily_min_hours,
        "batch_cluster_min": config.batch_cluster_min,
        "max_lead_days": config.max_lead_days,
        "offset_days": config.offset_days,
        "bootstrap_runs": config.bootstrap_runs,
        "min_block_days": config.min_block_days,
        "mcs_level": config.mcs_level,
        "recent_halflife_days": config.recent_halflife_days,
        "profiles": {k: {"label": p.label, "alpha_v": p.alpha_v,
                         "band_weights": list(p.band_weights), "note": p.note}
                     for k, p in dvs.PROFILES.items()},
    }


def apply_capability(sc: WindowScore, model: str, config: DvsConfig) -> None:
    """段级能力裁剪：源声明/实际不提供的要素或时效段记 0（= 用 L0 替代）。

    为什么在**段级**而不是项级：段的缺失是"这家没有这个时效能力"（如只报 3 天），
    属于能力边界，应按"增量价值为 0"定价；项的缺失是"这个产品它不单独提供"
    （如只给逐小时不给日产品），属于口径差异，应按剩余权重归一。两者混淆过，
    就是旧口径"天桶因门限退化为温度单维分"的成因之一。

    `config.capability` 未声明的源按**实际数据**推断：某段没有任何样本即视为不覆盖。
    """
    declared = (config.capability or {}).get(model) or {}
    max_days = declared.get("max_lead_days")
    for b in range(1, len(config.band_weights) + 1):
        lo, hi = BAND_BOUNDS[b - 1]
        if max_days is not None and lo > int(max_days):
            sc.temp_bands[b] = 0.0
            sc.precip_bands[b] = 0.0
    comp = dvs.composite_from_parts(sc.temp_bands, sc.precip_bands,
                                    config.alpha_v, config.band_weights)
    sc.dvs = comp["score"]
    cov = dvs.coverage_score(sc.temp_bands, sc.precip_bands,
                             config.alpha_v, config.band_weights)
    sc.coverage_score = cov["score"]
    sc.coverage_bands = cov["bands"]


def _availability(per_day: Mapping[int, SourceDayRows],
                  window_days: Sequence[int],
                  obs_days: set[int] | None,
                  declared: int | None) -> float:
    """可用率 = 该源在窗口内有可配对样本的天数 / 它**声明覆盖**的天数。

    分母优先用"观测实际存在的天 ∩ 该源存档覆盖的日期区间"：某源 10-24 才接入，
    就不能把 8 月的缺测算成它的可用率。`declared` 只作为没有存日集合时的兜底。
    """
    if not window_days:
        return 0.0
    have = len(set(per_day) & set(window_days))
    if obs_days:
        denom = len(set(window_days) & set(obs_days))
    elif declared:
        denom = min(int(declared), len(window_days))
    else:
        denom = len(window_days)
    if denom <= 0:
        return 0.0
    return float(min(max(have / denom, 0.0), 1.0))


def _bootstrap_matrix(rows_by_model: Mapping[str, Mapping[int, SourceDayRows]],
                      models: Sequence[str], config: DvsConfig,
                      days: Sequence[int],
                      ) -> tuple[np.ndarray | None, dict[str, float]]:
    """配对块自助：每次重采样都走**同一条** `score_window` 路径。

    这是"点估计与区间同源"的工程保证：自助样本的分数不是用另一套近似公式算的，
    而是把日权重换成抽样次数后重新跑一遍完整评分。任何一处公式改动都不会让
    区间与点估计脱节——它们在同一个函数里。
    """
    if not models or not days:
        return None, {}
    cache = {m: {d: rows_by_model.get(m, {}).get(d) for d in days}
             for m in models}
    point = {m: score_window(
        {d: r for d, r in cache[m].items() if r is not None},
        config=config).dvs for m in models}

    # 每条源预先排好 "有样本的那些天" 的 (day, rows) 列表：自助的 2000 次重采样
    # 复用它，避免每轮重做 sorted() 与字典查找（口径不变，只去掉重复劳动）。
    per_model_days = {
        m: sorted((d, r) for d, r in cache[m].items() if r is not None)
        for m in models}

    def score_fn(counts: np.ndarray) -> np.ndarray:
        out = np.full(len(models), np.nan)
        dw = {d: float(c) for d, c in zip(days, counts) if c > 0}
        if not dw:
            return out
        for i, m in enumerate(models):
            out[i] = _score_prepared(per_model_days[m], config, dw).dvs
        return out

    point_out = {m: (v if v is not None else float("nan")) for m, v in point.items()}
    # 块长由调用方给定下限：逐日分数的自相关估计需要每天跑一次完整评分（代价过高），
    # 因此这里采用**声明过的天气尺度下限**并在报告里披露实际值——宁可区间偏保守。
    res = paired_bootstrap(list(days), score_fn, list(models), point_out,
                           runs=config.bootstrap_runs, seed=config.seed,
                           min_block=config.min_block_days)
    return res.scores, point_out


def _score_prepared(day_list, config: DvsConfig,
                    day_weights: Mapping[int, float]) -> WindowScore:
    """`score_window` 的预排序等价物：只换"怎么取行"，不换任何公式。

    实现上复用同一条下游路径（`_score_from_aggregate`），因此**不可能**出现
    "自助用一套公式、点估计用另一套"的偏差——它们算的是同一个函数。
    """
    from .dvs_stats_store import weight_window
    return _score_from_aggregate(
        weight_window(day_list, day_weights), config,
        days={d: r for d, r in day_list if day_weights.get(d, 0.0) > 0.0})


def _wrap_bootstrap(matrix, models, point, config, days):
    """把已算好的矩阵包成 `BootstrapResult`（不再重抽样）。"""
    from .dvs_uncertainty import BootstrapResult, summarize_matrix
    res = BootstrapResult(scores=matrix, models=list(models), point=dict(point),
                          block_days=float(config.min_block_days),
                          runs=int(matrix.shape[0]), days=len(days))
    summarize_matrix(res)
    return res


def _lifetime(per_day: Mapping[int, SourceDayRows], config: DvsConfig,
              days: Sequence[int]) -> dict:
    """技巧寿命：逐日累积窗口分（相对 0 = 与 L0 持平）的链式显著性判据。

    实现口径：对**前 d 天**的样本算 DVS，并用同一批样本的配对块自助给 90% CI 下界。
    这样"第 d 天的技能"就与榜单上"前 d 天窗口"的分完全同源——读者不会看到两个
    互相矛盾的数字。
    """
    stats: dict[int, tuple[float, float | None]] = {}
    for d in range(1, min(len(days), config.max_lead_days) + 1):
        sub = days[:d]
        if len(sub) < max(config.min_block_days * 2, 5):
            stats[d] = (None, None)
            continue
        sc = score_window({k: per_day.get(k) for k in sub if k in per_day},
                          config=config)
        stats[d] = (sc.dvs, sc.dvs)
    return skill_lifetime(stats)


__all__ = [
    "BoardEntry", "DvsConfig", "EVIDENCE_TIERS", "MIN_DAYS_FOR_BOARD",
    "StationBuild", "WindowScore", "build_board", "build_station",
    "evidence_tier", "load_climatology", "score_window",
]

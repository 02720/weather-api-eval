"""评估引擎：配对观测与预报，调用 cyeva 计算指标，组装报告数据。

评估口径（详见 README，并在报告中注明）：
- 逐小时：按起报后时效（小时）分 1..16 天桶。
  温度（cyeva TemperatureComparison 全量）：RMSE / MAE / MBE / RSS / χ² /
    ±1°C·±2°C 准确率 / 相关系数 r / 回归斜率。
  降水（cyeva PrecipitationComparison 全量）：晴雨二分类（阈值默认 0.1mm）
    准确率 / 命中率 POD / 空报率 FAR / 空报频率 / 漏报率 / TS / ETS / 频率偏差 BIAS，
    连续量 RMSE / MAE / MBE，及 1h 雨强分级（小雨..大暴雨）每级 7 项指标。
- 按天：北京时自然日聚合日最高/最低气温、日降水量；按"有效日 − 起报日"的日偏移
  1..16 天分组。温度最高/最低全套指标；降水晴雨 + 连续量 + 24h 累计分级
  （≥0.1/≥10/≥25/≥50/≥100/≥250mm，即小雨..特大暴雨以上）每级 7 项指标。
  覆盖门槛（第一性原理：缺测绝不伪装成数值）：日聚合同时记录非缺测小时数，
  观测与预报任一侧的日覆盖不足 daily_min_hours（默认 20/24）时，该天该要素
  不参与按天评估——降水全缺测日若折算成 0.0 会伪装成"预报无雨"，部分覆盖日的
  日累计系统性偏低会伪装成"漏报"，两者都是把缺测当技巧。
所有指标附样本数 n；n < min_sample 视为"样本不足"不出结论（置 None）。

得分体系（排行榜、分时效榜单与时效趋势共用，2026-08-29 多指标加权重构）：
- 把预报质量拆成互不重复的维度，每维度取代表性指标换算成 0~100 的子分后加权平均：
  温度 7 项入分（±2°C/±1°C 准确率、RMSE/MAE 换算分、相关系数、|MBE| 偏差分、回归斜率分），
  降水 6 项入分（TS/ETS×100、晴雨准确率、POD、100−FAR、|BIAS−1| 偏差分）；
  各子分截断到 [0,100]，缺项按剩余权重归一（不让单一缺项把整行踢出局）。
  不入分的指标及理由见 TEMP/PRECIP_SCORE_PARTS 注释与 README。
- 综合得分 = mean(温度得分, 降水分)，缺项不计。
- 排行榜（leaderboards）分两层：
  * 分时效榜（"1d".."16d"）：每个天桶一份按综合分排序的完整行。预报难度随时效单调
    上升，天桶是"难度分层"——同桶内各源比较的是同一难度的预报，这是横向比较的公平基准。
  * 总榜（"all"）：把该源在评估窗口内全部逐小时配对样本（lead 1..N 天所有天桶）合并
    成一份指标再打分，回答"综合所有时效谁最准"。样本按条数计入，临近时效占比天然更高
    （也是实际被使用最多的预报）。各源可提供的最长时效不同（商业源常只有 2~3 天），
    合并后的样本构成随之不同，故总榜行附 lead_days（该源实际覆盖的最长时效，天）供
    读者校正解读；同难度的精确对比仍以分时效榜为准。
  两层共用同一行结构与打分公式，主报告表格排行榜（总榜 + 可切时效）与冠军横幅共用这份数据。
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta
from typing import Any

import numpy as np
from cyeva import PrecipitationComparison, TemperatureComparison
# cyeva 未导出 threshold 版 ETS/空报频率：用其内部二分类统计函数在同一口径下补齐
#（源值保留 2 位小数后与阈值比较，与 calc_threshold_* 的 binarize 路径完全一致）。
from cyeva.core.statistic import (
    calc_binary_accuracy_ratio as _stat_acc,
    calc_bias_score as _stat_bias,
    calc_ets as _stat_ets,
    calc_false_alarm_rate as _stat_farate,
    calc_false_alarm_ratio as _stat_far,
    calc_hit_ratio as _stat_pod,
    calc_miss_ratio as _stat_miss,
    calc_ts as _stat_ts,
)

from .timeutil import parse_iso, hour_bucket_days, floor_to_hour
from .storage import load_obs, list_forecast_snapshots

# 逐小时降水分级：cyeva 1h 雨强区间级别（小雨 0.1~1.9 … 大暴雨 ≥20 mm/h）
HOURLY_GRADED_LEVS = ("1", "2", "3", "4", "5")
# 按天降水分级：cyeva 24h 累计级别（+1=≥0.1 … +6=≥250mm）
DAILY_GRADED_LEVS = ("+1", "+2", "+3", "+4", "+5", "+6")
# 每级计算的分级指标（cyeva 分级全套）
GRADED_KEYS = ("acc", "pod", "far", "miss", "ts", "ets", "bias")


def _r(v: float) -> float | None:
    """cyeva 可能返回 nan（除零）；统一转为 None 并保留 3 位小数。"""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 3)


def _round4(v: float) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return round(f, 4)


def _valid_n(obs: np.ndarray, fcst: np.ndarray) -> int:
    return int((np.isfinite(obs) & np.isfinite(fcst)).sum())


# ----------------------------------------------------------------- 配对收集
def collect(station_ids: list[str], models: list[str], start_dt, end_dt,
            hourly_lead_days: int, daily_max_offset_days: int,
            daily_min_hours: int = 20) -> tuple[list[dict], list[dict]]:
    """返回 (hourly_records, daily_records)。"""
    # 观测月聚合
    obs_daily: dict[str, dict[str, dict]] = defaultdict(dict)
    hourly_records: list[dict] = []
    for sid in station_ids:
        obs_map = load_obs(sid)  # time_iso -> record
        # 逐小时配对
        for model in models:
            for snap in list_forecast_snapshots(sid, model):
                issue = parse_iso(snap["issue_iso"])
                times = snap["hourly_time"]
                for m in snap["data"]:
                    arr_t = snap["data"][m]["temperature_2m"]
                    arr_p = snap["data"][m]["precipitation"]
                    for i, tstr in enumerate(times):
                        vt = parse_iso(tstr)
                        if vt < start_dt or vt > end_dt:
                            continue
                        rec = obs_map.get(tstr)
                        if rec is None:
                            continue
                        lead = int((vt - issue).total_seconds() // 3600)
                        if lead <= 0 or lead > hourly_lead_days * 24:
                            continue
                        # 数组越界按缺测处理（与按天聚合同防护）：畸形存档降级为
                        # 该点缺测，不拖垮整份报告
                        hourly_records.append({
                            "station": sid, "model": m, "valid_iso": tstr,
                            "lead": lead, "bucket": hour_bucket_days(lead),
                            "temp_obs": rec.get("temp"),
                            "temp_fcst": (arr_t[i] if i < len(arr_t) else None),
                            "rain_obs": rec.get("rain"),
                            "rain_fcst": (arr_p[i] if i < len(arr_p) else None),
                        })

    # 按天聚合
    daily_records: list[dict] = []
    for sid in station_ids:
        obs_map = load_obs(sid)
        # 观测日聚合（n_temp/n_rain = 非缺测小时数，供覆盖门槛判定）
        od: dict[str, dict] = {}
        for tstr, rec in obs_map.items():
            day = tstr[:10]
            d = od.setdefault(day, {"max_temp": -math.inf, "min_temp": math.inf,
                                    "sum_rain": 0.0, "n_temp": 0, "n_rain": 0})
            if rec.get("temp") is not None:
                d["max_temp"] = max(d["max_temp"], rec["temp"])
                d["min_temp"] = min(d["min_temp"], rec["temp"])
                d["n_temp"] += 1
            if rec.get("rain") is not None:
                d["sum_rain"] += rec["rain"]
                d["n_rain"] += 1
        obs_daily[sid] = od

        for model in models:
            for snap in list_forecast_snapshots(sid, model):
                issue = parse_iso(snap["issue_iso"])
                issue_day = issue.strftime("%Y-%m-%d")
                times = snap["hourly_time"]
                # 与逐小时循环对称：对 snap["data"] 逐模型展开、按模型重置聚合桶，
                # 不依赖"每份存档只含一个模型"的上游不变量（新源直存/合并存档不混算）
                for m in snap["data"]:
                    arr_t = snap["data"][m]["temperature_2m"]
                    arr_p = snap["data"][m]["precipitation"]
                    fd: dict[str, dict] = {}
                    for i, tstr in enumerate(times):
                        vt = parse_iso(tstr)
                        if vt < start_dt or vt > end_dt:
                            continue
                        day = tstr[:10]
                        d = fd.setdefault(day, {"max_temp": -math.inf, "min_temp": math.inf,
                                                "sum_rain": 0.0, "n_temp": 0, "n_rain": 0})
                        if i < len(arr_t) and arr_t[i] is not None:
                            d["max_temp"] = max(d["max_temp"], arr_t[i])
                            d["min_temp"] = min(d["min_temp"], arr_t[i])
                            d["n_temp"] += 1
                        if i < len(arr_p) and arr_p[i] is not None:
                            d["sum_rain"] += arr_p[i]
                            d["n_rain"] += 1
                    for day, d in fd.items():
                        if day not in obs_daily[sid]:
                            continue
                        offset = (parse_iso(day + "T00:00") - parse_iso(issue_day + "T00:00")).days
                        if offset <= 0 or offset > daily_max_offset_days:
                            continue
                        oday = obs_daily[sid][day]
                        # 覆盖门槛：观测与预报任一侧日覆盖不足（缺测多/模式时效边界）
                        # 时该天该要素不入样——缺测折算成 0.0 或部分日累计都会伪装成技巧
                        o_temp = (oday["max_temp"] if oday["n_temp"] >= daily_min_hours
                                  and oday["max_temp"] > -math.inf else None)
                        o_min = (oday["min_temp"] if oday["n_temp"] >= daily_min_hours
                                 and oday["min_temp"] < math.inf else None)
                        o_rain = (oday["sum_rain"]
                                  if oday["n_rain"] >= max(daily_min_hours, 1) else None)
                        f_temp = (d["max_temp"] if d["n_temp"] >= daily_min_hours
                                  and d["max_temp"] > -math.inf else None)
                        f_min = (d["min_temp"] if d["n_temp"] >= daily_min_hours
                                  and d["min_temp"] < math.inf else None)
                        f_rain = (d["sum_rain"]
                                  if d["n_rain"] >= max(daily_min_hours, 1) else None)
                        if o_temp is None and o_min is None and o_rain is None \
                                and f_temp is None and f_min is None and f_rain is None:
                            continue  # 该天无任何可用日聚合，不入样
                        daily_records.append({
                            "station": sid, "model": m, "valid_day": day, "offset": offset,
                            "temp_max_obs": o_temp, "temp_max_fcst": f_temp,
                            "temp_min_obs": o_min, "temp_min_fcst": f_min,
                            "rain_obs": o_rain, "rain_fcst": f_rain,
                        })
    return hourly_records, daily_records


# ----------------------------------------------------------------- 指标计算
def temp_metrics(obs_vals, fcst_vals, limits, min_sample) -> dict:
    """温度全套指标（cyeva TemperatureComparison 全量）。"""
    obs = np.asarray(obs_vals, dtype=float)
    fcst = np.asarray(fcst_vals, dtype=float)
    n = _valid_n(obs, fcst)
    out = {"n": n, "rmse": None, "mae": None, "mbe": None, "rss": None,
           "chi2": None, "r": None, "slope": None,
           **{f"acc{lim}": None for lim in limits}}
    if n < min_sample or n == 0:
        return out
    tc = TemperatureComparison(obs, fcst, unit="degC")
    out["rmse"] = _r(tc.calc_rmse())
    out["mae"] = _r(tc.calc_mae())
    out["mbe"] = _r(tc.calc_mbe())
    out["rss"] = _r(tc.calc_rss())
    out["chi2"] = _r(tc.calc_chi_square())
    if n >= 3:  # 线性回归至少需要 3 个点，不足时留 None
        try:
            import warnings
            with warnings.catch_warnings():
                # 常数序列（如测试夹具的恒定温度）会让 scipy 发出除零 RuntimeWarning，
                # 结果为 nan -> _r 转为 None，警告本身是噪声
                warnings.simplefilter("ignore", RuntimeWarning)
                slope, _intercept, r, _p = tc.calc_linregress_args()
            out["slope"] = _r(slope)   # cyeva 口径：stats.linregress(forecast, observation)
            out["r"] = _r(r)
        except Exception:  # noqa: BLE001 —— 常数序列等退化输入，指标留 None
            pass
    for lim in limits:
        out[f"acc{lim}"] = _r(tc.calc_diff_accuracy_ratio(limit=lim))
    return out


def precip_metrics(obs_vals, fcst_vals, threshold, min_sample,
                   kind: str | None = None, graded_levs: tuple = ()) -> dict:
    """降水全套指标：晴雨二分类 8 项 + 连续量 3 项（+ 可选分级每级 7 项）。

    kind/graded_levs：传 "1h"+("1".."5") 算逐小时雨强区间分级，
    传 "24h"+("+1".."+6") 算按天累计分级；不传则只算晴雨与连续量。
    """
    obs = np.asarray(obs_vals, dtype=float)
    fcst = np.asarray(fcst_vals, dtype=float)
    n = _valid_n(obs, fcst)
    out = {"n": n, "acc": None, "pod": None, "far": None, "farate": None,
           "miss": None, "ts": None, "ets": None, "bias": None,
           "rmse": None, "mae": None, "mbe": None}
    if graded_levs:
        out["graded"] = {lev: None for lev in graded_levs}
    if n < min_sample or n == 0:
        return out
    pc = PrecipitationComparison(obs, fcst, unit="mm")
    out["acc"] = _r(pc.calc_threshold_accuracy_ratio(threshold=threshold, compare=">="))
    out["pod"] = _r(pc.calc_threshold_hit_ratio(threshold=threshold, compare=">="))
    out["far"] = _r(pc.calc_threshold_false_alarm_ratio(threshold=threshold, compare=">="))
    out["miss"] = _r(pc.calc_threshold_miss_ratio(threshold=threshold, compare=">="))
    out["ts"] = _r(pc.calc_threshold_ts(threshold=threshold, compare=">="))
    out["bias"] = _r(pc.calc_threshold_bias_score(threshold=threshold, compare=">="))
    # ETS/空报频率：与上面 6 项同口径（先舍入到 2 位再比阈值）手工二值化，
    # 调 cyeva 的二分类统计函数，保证同一份样本内所有晴雨指标口径一致。
    # 掩膜必须只剔 NaN、保留 inf——cyeva 的 drop_nan 是 NaN 判定（x != x），
    # inf 会被 threshold_binarize 判为"有雨"；若用 isfinite 会在含 inf 的
    # 样本上与同函数内 cyeva 六项指标落到不同的样本集合（口径分裂）。
    m = ~np.isnan(obs) & ~np.isnan(fcst)
    ob = np.round(obs[m], 2) >= threshold
    fb = np.round(fcst[m], 2) >= threshold
    out["ets"] = _r(_stat_ets(ob, fb))
    out["farate"] = _r(_stat_farate(ob, fb))
    out["rmse"] = _r(pc.calc_rmse())
    out["mae"] = _r(pc.calc_mae())
    out["mbe"] = _r(pc.calc_mbe())
    if graded_levs and kind:
        for lev in graded_levs:
            try:
                out["graded"][lev] = {
                    "acc": _r(pc.calc_accuracy_ratio(kind=kind, lev=lev)),
                    "pod": _r(pc.calc_hit_ratio(kind=kind, lev=lev)),
                    "far": _r(pc.calc_false_alarm_ratio(kind=kind, lev=lev)),
                    "miss": _r(pc.calc_miss_ratio(kind=kind, lev=lev)),
                    "ts": _r(pc.calc_ts(kind=kind, lev=lev)),
                    "ets": _r(pc.calc_ets(kind=kind, lev=lev)),
                    "bias": _r(pc.calc_bias_score(kind=kind, lev=lev)),
                }
            except Exception:  # noqa: BLE001 —— 级别不存在等契约漂移不应拖垮整桶
                out["graded"][lev] = None
    return out


# ----------------------------------------------------------------- 得分体系
# 设计（第一性原理）：先拆维度、再选代表指标、后加权——不把互相冗余的指标重复计账。
# 每项以 (指标键, 权重, 白话标签, 换算说明, 换算函数) 描述：换算函数把指标映射到
# 0~100 的子分（统一截断到 [0,100]），权重决定该维度对总分的话语权。
#
# 温度（5 个维度、7 项入分）：
#   报准比例 acc2/acc1 · 误差幅度 RMSE/MAE · 起伏节奏 r · 系统偏差 |MBE| · 幅度校准 slope。
#   不入分：RSS 与 χ² —— χ²=RMSE²、RSS=n×χ²，是样本量的函数而非预报技巧，只进明细表。
# 降水（5 个维度、6 项入分）：
#   晴雨综合技巧 TS/ETS · 答对率 acc · 命中 POD · 空报 FAR · 频率无偏 BIAS。
#   不入分：漏报率（=100−POD，纯冗余）、空报频率 POFD（与 FAR 同族仅分母不同）、
#   雨量 RMSE/MAE/MBE（连续雨量误差由个别强降水时段主导、随样本期气候波动大，
#   跨源横向比较不公平，只进明细表）。晴雨准确率在干燥气候下天然偏高，故仅给低权重。
TEMP_SCORE_PARTS = (
    ("acc2", 0.25, "±2°C 准确率", "百分比直接入分",
     lambda v: v),
    ("rmse", 0.25, "RMSE 误差换算分", "100 − RMSE×5（0°C 记 100 分，每多 0.2°C 扣 1 分）",
     lambda v: 100 - v * 5),
    ("r", 0.15, "相关系数 r", "r×100（起伏节奏的同步程度）",
     lambda v: v * 100),
    ("acc1", 0.10, "±1°C 准确率", "百分比直接入分（更严格的命中口径）",
     lambda v: v),
    ("mae", 0.10, "MAE 误差换算分", "100 − MAE×5（典型误差，对偶发大误差不敏感）",
     lambda v: 100 - v * 5),
    ("mbe", 0.10, "偏差换算分", "100 − |MBE|×10（无系统性偏高/偏低 = 满分）",
     lambda v: 100 - abs(v) * 10),
    ("slope", 0.05, "回归斜率换算分", "100 − |斜率−1|×100（冷热幅度恰如其分 = 满分）",
     lambda v: 100 - abs(v - 1) * 100),
)
PRECIP_SCORE_PARTS = (
    ("ts", 0.30, "晴雨 TS 评分", "TS×100（报中/空报/漏报一账清）",
     lambda v: v * 100),
    ("ets", 0.25, "晴雨 ETS 评分", "ETS×100（对“瞎蒙也能蒙对”做过校正）",
     lambda v: v * 100),
    ("acc", 0.15, "晴雨准确率", "百分比直接入分（干燥期天然偏高，故权重低）",
     lambda v: v),
    ("pod", 0.15, "命中率 POD", "百分比直接入分（漏报少）",
     lambda v: v),
    ("far", 0.10, "空报率换算分", "100 − FAR（不喊“狼来了” = 满分）",
     lambda v: 100 - v),
    ("bias", 0.05, "频率偏差换算分", "100 − |BIAS−1|×100（报雨频率恰如其分 = 满分）",
     lambda v: 100 - abs(v - 1) * 100),
)


def _clamp100(v: float) -> float:
    return max(0.0, min(100.0, v))


def _weighted_score(parts, metrics: dict) -> float | None:
    """按 (键, 权重, …, 换算函数) 表加权平均；缺项不计并按剩余权重归一。"""
    num = den = 0.0
    for key, w, _label, _map, fn in parts:
        v = metrics.get(key)
        if v is None:
            continue
        num += w * _clamp100(fn(v))
        den += w
    return round(num / den, 2) if den else None


def _mean_or_none(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def temp_score(t: dict) -> float | None:
    """温度得分(0~100)：7 项指标按 TEMP_SCORE_PARTS 权重加权，缺项按剩余权重归一。"""
    return _weighted_score(TEMP_SCORE_PARTS, t)


def precip_score(p: dict) -> float | None:
    """降水分(0~100)：6 项指标按 PRECIP_SCORE_PARTS 权重加权，缺项按剩余权重归一。"""
    return _weighted_score(PRECIP_SCORE_PARTS, p)


def overall_score(t: dict, p: dict) -> float | None:
    """综合得分：温度分与降水分的均分，缺项不计。"""
    return _mean_or_none([temp_score(t), precip_score(p)])


# ----------------------------------------------------------------- 报告组装
def _pool(records, keys, filt=None):
    o, f = [], []
    for r in records:
        if filt and not filt(r):
            continue
        o.append(r[keys[0]])
        f.append(r[keys[1]])
    return o, f


def build_report(station_ids, models, eval_cfg, start_dt, end_dt,
                 period_label: str, is_monthly: bool = False) -> dict:
    limits = eval_cfg["temp_accuracy_limits"]
    thr = eval_cfg["rain_threshold_mm"]
    min_sample = eval_cfg["min_sample"]
    hourly_lead_days = eval_cfg["hourly_lead_days"]
    daily_max_offset = eval_cfg["daily_max_offset_days"]
    daily_min_hours = eval_cfg.get("daily_min_hours", 20)

    hourly, daily = collect(station_ids, models, start_dt, end_dt,
                            hourly_lead_days, daily_max_offset, daily_min_hours)

    # 按模型分组一次，后续所有 per-model 统计只遍历各自的记录（不做全量重扫）
    by_model: dict[str, list] = {m: [] for m in models}
    for r in hourly:
        if r["model"] in by_model:
            by_model[r["model"]].append(r)

    # ---- 评分卡（模型级）----
    # 24h/72h 池 = 固定时效窗的 pooled 指标；all 池 = 评估窗口内全部逐小时样本
    # （collect 已把 lead 限制在 1..hourly_lead_days*24），是"全时效总榜"的数据源。
    scorecard = {}
    for m in models:
        recs = by_model[m]
        h24 = [r for r in recs if 1 <= r["lead"] <= 24]
        h72 = [r for r in recs if 1 <= r["lead"] <= 72]
        to24, tf24 = _pool(h24, ("temp_obs", "temp_fcst"))
        to72, tf72 = _pool(h72, ("temp_obs", "temp_fcst"))
        toall, tfall = _pool(recs, ("temp_obs", "temp_fcst"))
        ro24, rf24 = _pool(h24, ("rain_obs", "rain_fcst"))
        ro72, rf72 = _pool(h72, ("rain_obs", "rain_fcst"))
        roall, rfall = _pool(recs, ("rain_obs", "rain_fcst"))
        scorecard[m] = {
            "temp_24h": temp_metrics(to24, tf24, limits, min_sample),
            "temp_72h": temp_metrics(to72, tf72, limits, min_sample),
            "temp_all": temp_metrics(toall, tfall, limits, min_sample),
            "precip_24h": precip_metrics(ro24, rf24, thr, min_sample),
            "precip_72h": precip_metrics(ro72, rf72, thr, min_sample),
            "precip_all": precip_metrics(roall, rfall, thr, min_sample),
        }

    # ---- 逐小时按天桶：温度 / 降水（含 1h 雨强分级） ----
    temp_hourly: dict[str, dict] = {}
    precip_hourly: dict[str, dict] = {}
    for m in models:
        tbuckets: dict[int, tuple] = {b: ([], []) for b in range(1, hourly_lead_days + 1)}
        pbuckets: dict[int, tuple] = {b: ([], []) for b in range(1, hourly_lead_days + 1)}
        for r in by_model[m]:
            tbuckets[r["bucket"]][0].append(r["temp_obs"])
            tbuckets[r["bucket"]][1].append(r["temp_fcst"])
            pbuckets[r["bucket"]][0].append(r["rain_obs"])
            pbuckets[r["bucket"]][1].append(r["rain_fcst"])
        temp_hourly[m] = {f"{b}d": temp_metrics(tbuckets[b][0], tbuckets[b][1], limits, min_sample)
                          for b in range(1, hourly_lead_days + 1)}
        precip_hourly[m] = {f"{b}d": precip_metrics(pbuckets[b][0], pbuckets[b][1], thr, min_sample,
                                                    kind="1h", graded_levs=HOURLY_GRADED_LEVS)
                            for b in range(1, hourly_lead_days + 1)}

    # ---- 逐小时逐时效曲线（1..72h）温度 RMSE / ±2°C 准确率 ----
    temp_lead_curve = {m: {f"{h}h": None for h in range(1, 73)} for m in models}
    for m in models:
        by_lead = defaultdict(lambda: ([], []))
        for r in by_model[m]:
            if 1 <= r["lead"] <= 72:
                by_lead[r["lead"]][0].append(r["temp_obs"])
                by_lead[r["lead"]][1].append(r["temp_fcst"])
        for h, (o, f) in by_lead.items():
            mm = temp_metrics(o, f, limits, min_sample)
            temp_lead_curve[m][f"{h}h"] = {"rmse": mm.get("rmse"), "acc2": mm.get("acc2")}

    # ---- 按天按偏移：温度（最高/最低） / 降水（含 24h 累计分级） ----
    temp_daily: dict[str, dict] = {}
    precip_daily: dict[str, dict] = {}
    for m in models:
        by_off = defaultdict(lambda: {"max": ([], []), "min": ([], []), "rain": ([], [])})
        for r in daily:
            if r["model"] != m:
                continue
            by_off[r["offset"]]["max"][0].append(r["temp_max_obs"])
            by_off[r["offset"]]["max"][1].append(r["temp_max_fcst"])
            by_off[r["offset"]]["min"][0].append(r["temp_min_obs"])
            by_off[r["offset"]]["min"][1].append(r["temp_min_fcst"])
            by_off[r["offset"]]["rain"][0].append(r["rain_obs"])
            by_off[r["offset"]]["rain"][1].append(r["rain_fcst"])
        temp_daily[m] = {}
        precip_daily[m] = {}
        for off in range(1, daily_max_offset + 1):
            maxmm = temp_metrics(by_off[off]["max"][0], by_off[off]["max"][1], limits, min_sample)
            minmm = temp_metrics(by_off[off]["min"][0], by_off[off]["min"][1], limits, min_sample)
            temp_daily[m][f"{off}d"] = {"max": maxmm, "min": minmm}
            precip_daily[m][f"{off}d"] = precip_metrics(
                by_off[off]["rain"][0], by_off[off]["rain"][1], thr, min_sample,
                kind="24h", graded_levs=DAILY_GRADED_LEVS)

    # ---- 分站概览（24h 桶）：按（站, 模型）分一次组，避免 O(站×模型×全量) 重扫 ----
    by_station_model: dict[tuple, list] = {(sid, m): [] for sid in station_ids for m in models}
    for r in hourly:
        key = (r["station"], r["model"])
        if key in by_station_model:
            by_station_model[key].append(r)
    per_station = {}
    for sid in station_ids:
        per_station[sid] = {}
        for m in models:
            h = [r for r in by_station_model[(sid, m)] if 1 <= r["lead"] <= 24]
            to, tf = _pool(h, ("temp_obs", "temp_fcst"))
            ro, rf = _pool(h, ("rain_obs", "rain_fcst"))
            per_station[sid][m] = {
                "temp": temp_metrics(to, tf, limits, min_sample),
                "precip": precip_metrics(ro, rf, thr, min_sample),
            }

    # ---- 时间序列（最近 72h，用于预报 vs 观测叠图） ----
    ts_start = end_dt - timedelta(hours=72)
    timeseries = _build_timeseries(station_ids, models, ts_start, end_dt)

    # ---- 热力图：各日 × 各模型 温度 ±2°C 准确率 ----
    heatmap = _build_heatmap(station_ids, models, hourly, limits, min_sample)

    # ---- 覆盖率 ----
    coverage = _coverage(station_ids, start_dt, end_dt)

    # ---- 口径注记（如 AccuWeather 最近城市吸附），随报告元数据输出 ----
    model_caveats = _model_caveats(station_ids, models)

    # ---- 排行榜：分时效榜（每个天桶一份）+ 全时效总榜 ----
    # 表格排行榜、冠军横幅与趋势图共用同一套得分。
    # scorecard 的 24h 池（lead 1..24）与天桶 "1d" 是同一样本总体，实证数值一致，
    # 故不再单独维护 ranking，冠军横幅直接读 leaderboards。
    leaderboards = _lead_leaderboards(models, temp_hourly, precip_hourly, hourly_lead_days)
    leaderboards["all"] = _overall_board(models, scorecard, by_model)

    # ---- 得分随时效衰减（综合/温度/降水，天桶 1..N）：排行榜的"趋势版" ----
    score_trend = _score_trend(models, temp_hourly, precip_hourly, hourly_lead_days)

    return {
        "meta": {
            "period_label": period_label,
            "is_monthly": is_monthly,
            "start": start_dt.strftime("%Y-%m-%d %H:%M"),
            "end": end_dt.strftime("%Y-%m-%d %H:%M"),
            "generated_at": floor_to_hour(end_dt).strftime("%Y-%m-%d %H:%M"),
            "models": models,
            "stations": station_ids,
            "limits": limits,
            "hourly_lead_days": hourly_lead_days,
            "rain_threshold_mm": thr,
            "min_sample": min_sample,
            "daily_min_hours": daily_min_hours,
            "model_caveats": model_caveats,
        },
        "coverage": coverage,
        "scorecard": scorecard,
        "temp_hourly": temp_hourly,
        "precip_hourly": precip_hourly,
        "temp_lead_curve": temp_lead_curve,
        "temp_daily": temp_daily,
        "precip_daily": precip_daily,
        "per_station": per_station,
        "timeseries": timeseries,
        "heatmap": heatmap,
        "leaderboards": leaderboards,
        "score_trend": score_trend,
    }


def _build_timeseries(station_ids, models, ts_start, end_dt) -> dict:
    """最近 72h 的"预报 vs 实况"叠图数据。

    预报线的取值口径：对每个时刻，取**当时已发布的最新一版预报**（按起报时间
    升序遍历快照，后发布的覆盖同一时刻的旧值；时效即处于 1~24h 内）。
    只使用"发布时间不晚于窗口末尾"的快照——月度归档时排除归档之后新抓的
    快照，保证归档页与当时可见的预报一致、且时间轴有交集（若始终取全局
    最新快照，归档页的预报线会整段缺失）。
    """
    out: dict[str, Any] = {}
    for sid in station_ids:
        obs_map = load_obs(sid)
        series = []
        # 以 obs 时间轴为准（取窗口内 obs 时间）
        for tstr, rec in sorted(obs_map.items()):
            vt = parse_iso(tstr)
            if vt < ts_start or vt > end_dt:
                continue
            series.append({"t": tstr, "temp": rec.get("temp"), "rain": rec.get("rain")})
        model_series: dict[str, list] = {m: [] for m in models}
        out[sid] = {"obs": series, "models": model_series}
    for sid in station_ids:
        for m in models:
            snaps = list_forecast_snapshots(sid, m)  # 已按起报时间升序
            tmap: dict[str, tuple] = {}
            for snap in snaps:
                if parse_iso(snap["issue_iso"]) > end_dt:
                    continue
                if m not in snap["data"]:
                    continue
                arr_t = snap["data"][m]["temperature_2m"]
                arr_p = snap["data"][m]["precipitation"]
                for i, t in enumerate(snap["hourly_time"]):
                    # 数组越界按缺测处理（畸形存档不拖垮报告）
                    tmap[t] = (arr_t[i] if i < len(arr_t) else None,
                               arr_p[i] if i < len(arr_p) else None)  # 后发布的覆盖同刻旧值
            if not tmap:
                continue
            arr = []
            for pt in out[sid]["obs"]:
                v = tmap.get(pt["t"])
                arr.append({"t": pt["t"],
                            "temp": v[0] if v else None,
                            "rain": v[1] if v else None})
            out[sid]["models"][m] = arr
    return out


def _build_heatmap(station_ids, models, hourly, limits, min_sample) -> list[dict]:
    """返回 rows: 每天一行；每模型温度 ±2°C 准确率。用于 ECharts heatmap。"""
    by_day_model: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for r in hourly:
        by_day_model[r["valid_iso"][:10]][r["model"]][0].append(r["temp_obs"])
        by_day_model[r["valid_iso"][:10]][r["model"]][1].append(r["temp_fcst"])
    days = sorted(by_day_model.keys())
    cells = []
    for d in days:
        for m in models:
            o, f = by_day_model[d][m]
            mm = temp_metrics(o, f, limits, min_sample)
            acc = mm.get("acc2")
            cells.append({"date": d, "model": m, "acc2": acc, "n": mm.get("n", 0)})
    return cells


def _coverage(station_ids, start_dt, end_dt) -> dict:
    """观测覆盖率。分母按"该站实际已有观测的时段"截断：观测开始前/月初尚不存在
    的时段计入分母只会稀释数字、误导读者（样本不是缺失，而是尚不存在）。"""
    expected = 0
    got = 0
    first_obs: str | None = None
    for sid in station_ids:
        obs_map = load_obs(sid)
        times_in_window = [parse_iso(t) for t in obs_map
                           if start_dt <= parse_iso(t) <= end_dt]
        got += len(times_in_window)
        if times_in_window:
            eff_start = max(start_dt, min(times_in_window))
            expected += int((end_dt - eff_start).total_seconds() // 3600) + 1
            iso_start = eff_start.strftime("%Y-%m-%dT%H:%M")
            if first_obs is None or iso_start < first_obs:
                first_obs = iso_start
    return {
        "expected_hours": expected,
        "got_hours": got,
        "coverage_pct": round(100.0 * got / expected, 1) if expected else None,
        "first_obs": first_obs,
    }


def _model_caveats(station_ids, models) -> dict[str, str]:
    """从快照 meta 提取影响榜单解读的口径注记（数据驱动，模板只负责呈现）。

    已知注记：AccuWeather 的"最近城市吸附"定位——快照留档了 location_name 与
    haversine 吸附距离，其样本代表距站点数十公里的城市而非站点格点，公开榜单
    必须让读者知情（README 有说明，但只看报告的读者看不到）。
    """
    dist: dict[str, list[float]] = defaultdict(list)
    names: dict[str, str] = {}
    for sid in station_ids:
        for model in models:
            for snap in list_forecast_snapshots(sid, model):
                d = snap.get("location_distance_km")
                if d is not None:
                    dist[model].append(float(d))
                nm = snap.get("location_name")
                if nm:
                    names.setdefault(model, str(nm))
    out = {}
    for model, ds in dist.items():
        avg = round(sum(ds) / len(ds), 1)
        name = names.get(model, "最近城市")
        out[model] = (f"最近城市吸附：定位到距站点平均约 {avg} km 的「{name}」，"
                      "样本代表该城市而非站点格点，雨温气候可能与站点本地不同")
    return out


def _board_row(m: str, t: dict, p: dict, **extra) -> dict:
    """榜单行：得分（与趋势图同一套公式，榜单与曲线永不分叉）
    + 榜面直接可读的关键指标（±2°C 准确率 / RMSE / TS / ETS / 样本数）。"""
    row = {
        "model": m,
        "score": overall_score(t, p),
        "temp_score": temp_score(t),
        "precip_score": precip_score(p),
        "acc2": t.get("acc2"), "rmse": t.get("rmse"),
        "ts": p.get("ts"), "ets": p.get("ets"),
        "n": t.get("n", 0), "n_precip": p.get("n", 0),
    }
    row.update(extra)
    return row


def _rank_rows(rows: list[dict]) -> list[dict]:
    """综合分降序、None 沉底；行序即名次，前端只重排不重算。"""
    return sorted(rows, key=lambda x: (x["score"] is None, -(x["score"] or 0)))


def _lead_leaderboards(models, temp_hourly, precip_hourly, hourly_lead_days) -> dict:
    """分时效排行榜：每个天桶（第 1..N 天）一份按综合分排序的完整行。

    预报难度随时效单调上升，天桶即"难度分层"——同桶内比较对各家才是同难度的。
    前端表格切换时效/排序只重排，不重算。
    """
    boards: dict[str, list[dict]] = {}
    for b in range(1, hourly_lead_days + 1):
        bk = f"{b}d"
        rows = []
        for m in models:
            t = temp_hourly[m].get(bk) or {}
            p = precip_hourly[m].get(bk) or {}
            rows.append(_board_row(m, t, p))
        boards[bk] = _rank_rows(rows)
    return boards


def _overall_board(models, scorecard, by_model) -> list[dict]:
    """全时效总榜：把该源在评估窗口内全部逐小时配对样本（lead 1..N 天）合并成
    一份指标再打分，回答"综合所有时效谁最准"。

    公平性说明（第一性原理）：预报难度随时效单调上升，跨时效合并的前提是样本构成
    可比——但各源实际参与对账的样本覆盖不同（商业源常只有 2~3 天，全球模式可到 16 天；
    且受"起报 + 时效是否已落进可对账时段"限制），短时效源的总榜样本天然偏"易"。
    这一混杂无法在单一数字内消除，只能显式披露：每行附 lead_days = 该源实际参与
    对账的样本中最长一条覆盖到的天数（向上取整，不代表逐天连续覆盖），页面上作为
    "覆盖时效"列展示并提示解读注意；同难度的精确对比仍以分时效榜为准。
    """
    rows = []
    for m in models:
        t = scorecard[m]["temp_all"]
        p = scorecard[m]["precip_all"]
        leads = [r["lead"] for r in by_model[m]]
        rows.append(_board_row(
            m, t, p,
            lead_days=(math.ceil(max(leads) / 24) if leads else None),
        ))
    return _rank_rows(rows)


def _score_trend(models, temp_hourly, precip_hourly, hourly_lead_days) -> dict:
    """各天桶（第 1..N 天）的温度分 / 降水分 / 综合分，用于"得分随时效衰减"趋势图。"""
    trend = {"overall": {}, "temp": {}, "precip": {}}
    for m in models:
        for key in trend:
            trend[key][m] = {}
        for b in range(1, hourly_lead_days + 1):
            bk = f"{b}d"
            t = temp_hourly[m].get(bk, {})
            p = precip_hourly[m].get(bk, {})
            trend["temp"][m][bk] = temp_score(t)
            trend["precip"][m][bk] = precip_score(p)
            trend["overall"][m][bk] = overall_score(t, p)
    return trend

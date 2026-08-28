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
所有指标附样本数 n；n < min_sample 视为"样本不足"不出结论（置 None）。

得分体系（排行榜与时效趋势共用）：
- 温度得分(0~100) = mean(±2°C 准确率, 100−RMSE×5)，缺项不计；
- 降水分(0~100) = mean(晴雨TS×100, 晴雨准确率)，缺项不计；
- 综合得分 = mean(温度得分, 降水分)，缺项不计（与旧四项均分公式在数据齐全时等值）。
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

from .timeutil import parse_iso, ymd, hour_bucket_days, floor_to_hour
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
            hourly_lead_days: int, daily_max_offset_days: int) -> tuple[list[dict], list[dict]]:
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
                        hourly_records.append({
                            "station": sid, "model": m, "valid_iso": tstr,
                            "lead": lead, "bucket": hour_bucket_days(lead),
                            "temp_obs": rec.get("temp"), "temp_fcst": arr_t[i],
                            "rain_obs": rec.get("rain"), "rain_fcst": arr_p[i],
                        })

    # 按天聚合
    daily_records: list[dict] = []
    for sid in station_ids:
        obs_map = load_obs(sid)
        # 观测日聚合
        od: dict[str, dict] = {}
        for tstr, rec in obs_map.items():
            day = tstr[:10]
            d = od.setdefault(day, {"max_temp": -math.inf, "min_temp": math.inf,
                                    "sum_rain": 0.0, "n": 0})
            if rec.get("temp") is not None:
                d["max_temp"] = max(d["max_temp"], rec["temp"])
                d["min_temp"] = min(d["min_temp"], rec["temp"])
            if rec.get("rain") is not None:
                d["sum_rain"] += rec["rain"]
            d["n"] += 1
        obs_daily[sid] = od

        for model in models:
            for snap in list_forecast_snapshots(sid, model):
                issue = parse_iso(snap["issue_iso"])
                issue_day = issue.strftime("%Y-%m-%d")
                times = snap["hourly_time"]
                # 该快照该模型逐日聚合
                fd: dict[str, dict] = {}
                for m in snap["data"]:
                    arr_t = snap["data"][m]["temperature_2m"]
                    arr_p = snap["data"][m]["precipitation"]
                    for i, tstr in enumerate(times):
                        vt = parse_iso(tstr)
                        if vt < start_dt or vt > end_dt:
                            continue
                        day = tstr[:10]
                        d = fd.setdefault(day, {"max_temp": -math.inf, "min_temp": math.inf,
                                                "sum_rain": 0.0, "n": 0})
                        if arr_t[i] is not None:
                            d["max_temp"] = max(d["max_temp"], arr_t[i])
                            d["min_temp"] = min(d["min_temp"], arr_t[i])
                        if arr_p[i] is not None:
                            d["sum_rain"] += arr_p[i]
                        d["n"] += 1
                for day, d in fd.items():
                    if day not in obs_daily[sid]:
                        continue
                    offset = (parse_iso(day + "T00:00") - parse_iso(issue_day + "T00:00")).days
                    if offset <= 0 or offset > daily_max_offset_days:
                        continue
                    oday = obs_daily[sid][day]
                    daily_records.append({
                        "station": sid, "model": m, "valid_day": day, "offset": offset,
                        "temp_max_obs": (oday["max_temp"] if oday["max_temp"] > -math.inf else None),
                        "temp_max_fcst": (d["max_temp"] if d["max_temp"] > -math.inf else None),
                        "temp_min_obs": (oday["min_temp"] if oday["min_temp"] < math.inf else None),
                        "temp_min_fcst": (d["min_temp"] if d["min_temp"] < math.inf else None),
                        "rain_obs": oday["sum_rain"], "rain_fcst": d["sum_rain"],
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
def _mean_or_none(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 2) if vals else None


def temp_score(t: dict) -> float | None:
    """温度得分(0~100)：±2°C 准确率 与 误差换算分(100−RMSE×5) 的均分，缺项不计。"""
    parts = []
    if t.get("acc2") is not None:
        parts.append(t["acc2"])
    if t.get("rmse") is not None:
        parts.append(max(0.0, 100 - t["rmse"] * 5))
    return _mean_or_none(parts)


def precip_score(p: dict) -> float | None:
    """降水分(0~100)：晴雨TS×100 与 晴雨准确率 的均分，缺项不计。"""
    parts = []
    if p.get("ts") is not None:
        parts.append(p["ts"] * 100)
    if p.get("acc") is not None:
        parts.append(p["acc"])
    return _mean_or_none(parts)


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

    hourly, daily = collect(station_ids, models, start_dt, end_dt,
                            hourly_lead_days, daily_max_offset)

    # ---- 评分卡（模型级，覆盖全窗口）----
    scorecard = {}
    for m in models:
        h24 = [r for r in hourly if r["model"] == m and 1 <= r["lead"] <= 24]
        h72 = [r for r in hourly if r["model"] == m and 1 <= r["lead"] <= 72]
        to24, tf24 = _pool(h24, ("temp_obs", "temp_fcst"))
        to72, tf72 = _pool(h72, ("temp_obs", "temp_fcst"))
        ro24, rf24 = _pool(h24, ("rain_obs", "rain_fcst"))
        ro72, rf72 = _pool(h72, ("rain_obs", "rain_fcst"))
        scorecard[m] = {
            "temp_24h": temp_metrics(to24, tf24, limits, min_sample),
            "temp_72h": temp_metrics(to72, tf72, limits, min_sample),
            "precip_24h": precip_metrics(ro24, rf24, thr, min_sample),
            "precip_72h": precip_metrics(ro72, rf72, thr, min_sample),
        }

    # ---- 逐小时按天桶：温度 / 降水（含 1h 雨强分级） ----
    temp_hourly: dict[str, dict] = {}
    precip_hourly: dict[str, dict] = {}
    for m in models:
        tbuckets: dict[int, tuple] = {b: ([], []) for b in range(1, hourly_lead_days + 1)}
        pbuckets: dict[int, tuple] = {b: ([], []) for b in range(1, hourly_lead_days + 1)}
        for r in hourly:
            if r["model"] != m:
                continue
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
        for r in hourly:
            if r["model"] == m and 1 <= r["lead"] <= 72:
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

    # ---- 分站概览（24h 桶） ----
    per_station = {}
    for sid in station_ids:
        per_station[sid] = {}
        for m in models:
            h = [r for r in hourly if r["station"] == sid and r["model"] == m and 1 <= r["lead"] <= 24]
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

    # ---- 综合排名：主报告与月度归档都展示"冠军榜"，故无条件计算 ----
    ranking = _ranking(scorecard)

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
            "rain_threshold_mm": thr,
            "min_sample": min_sample,
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
        "ranking": ranking,
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
                    tmap[t] = (arr_t[i], arr_p[i])  # 后发布的覆盖同刻旧值
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
    total_hours = int((end_dt - start_dt).total_seconds() // 3600) + 1
    expected = total_hours * len(station_ids)
    got = 0
    for sid in station_ids:
        obs_map = load_obs(sid)
        for tstr in obs_map:
            vt = parse_iso(tstr)
            if start_dt <= vt <= end_dt:
                got += 1
    return {
        "expected_hours": expected,
        "got_hours": got,
        "coverage_pct": round(100.0 * got / expected, 1) if expected else None,
    }


def _ranking(scorecard: dict) -> list[dict]:
    """主榜按"提前 1 天（24h）"口径打分；温度分/降水分给出综合分的分解。"""
    rows = []
    for m, sc in scorecard.items():
        t = sc["temp_24h"]
        p = sc["precip_24h"]
        rows.append({
            "model": m,
            "score": overall_score(t, p),
            "temp_score": temp_score(t),
            "precip_score": precip_score(p),
            "temp_acc2_24h": t.get("acc2"),
            "precip_ts_24h": p.get("ts"),
        })
    rows.sort(key=lambda x: (x["score"] is None, -(x["score"] or 0)))
    return rows


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

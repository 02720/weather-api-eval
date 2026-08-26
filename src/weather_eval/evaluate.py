"""评估引擎：配对观测与预报，调用 cyeva 计算指标，组装报告数据。

评估口径（详见 README，并在报告中注明）：
- 逐小时：按起报后时效（小时）分 1..16 天桶；温度 RMSE/MAE/MBE/±1°C/±2°C 准确率，
  降水 0.1mm 晴雨二分类（准确率/命中率 POD/空报率 FAR/漏报率/TS/BIAS）。
- 按天：北京时自然日聚合日最高/最低气温、日降水量；按"有效日 - 起报日"的日偏移 1..16 天分组。
  温度最高/最低 ±2°C 准确率、MAE、RMSE；降水晴雨二分类 + 分级 TS（≥0.1/≥10/≥25mm）。
所有指标附样本数 n；n < min_sample 视为"样本不足"不出结论（置 None）。
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import timedelta
from typing import Any

import numpy as np
from cyeva import PrecipitationComparison, TemperatureComparison

from .timeutil import parse_iso, ymd, hour_bucket_days, floor_to_hour
from .storage import load_obs, list_forecast_snapshots


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
    obs = np.asarray(obs_vals, dtype=float)
    fcst = np.asarray(fcst_vals, dtype=float)
    n = _valid_n(obs, fcst)
    out = {"n": n, "rmse": None, "mae": None, "mbe": None,
           **{f"acc{lim}": None for lim in limits}}
    if n < min_sample:
        return out
    tc = TemperatureComparison(obs, fcst, unit="degC")
    out["rmse"] = _r(tc.calc_rmse())
    out["mae"] = _r(tc.calc_mae())
    out["mbe"] = _r(tc.calc_mbe())
    for lim in limits:
        out[f"acc{lim}"] = _r(tc.calc_diff_accuracy_ratio(limit=lim))
    return out


def precip_binary_metrics(obs_vals, fcst_vals, threshold, min_sample) -> dict:
    obs = np.asarray(obs_vals, dtype=float)
    fcst = np.asarray(fcst_vals, dtype=float)
    n = _valid_n(obs, fcst)
    out = {"n": n, "acc": None, "pod": None, "far": None, "miss": None, "ts": None, "bias": None}
    if n < min_sample:
        return out
    pc = PrecipitationComparison(obs, fcst, unit="mm")
    out["acc"] = _r(pc.calc_threshold_accuracy_ratio(threshold=threshold, compare=">="))
    out["pod"] = _r(pc.calc_threshold_hit_ratio(threshold=threshold, compare=">="))
    out["far"] = _r(pc.calc_threshold_false_alarm_ratio(threshold=threshold, compare=">="))
    out["miss"] = _r(pc.calc_threshold_miss_ratio(threshold=threshold, compare=">="))
    out["ts"] = _r(pc.calc_threshold_ts(threshold=threshold, compare=">="))
    out["bias"] = _r(pc.calc_threshold_bias_score(threshold=threshold, compare=">="))
    return out


def precip_graded_ts(obs_vals, fcst_vals, kind, levels, min_sample) -> dict:
    obs = np.asarray(obs_vals, dtype=float)
    fcst = np.asarray(fcst_vals, dtype=float)
    n = _valid_n(obs, fcst)
    out = {"n": n, **{f"ts{lv}": None for lv in levels}}
    if n < min_sample:
        return out
    pc = PrecipitationComparison(obs, fcst, unit="mm")
    for lv in levels:
        out[f"ts{lv}"] = _r(pc.calc_ts(kind=kind, lev=lv))
    return out


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
    graded_levels = ["+1", "+2", "+3"]

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
            "precip_24h": precip_binary_metrics(ro24, rf24, thr, min_sample),
            "precip_72h": precip_binary_metrics(ro72, rf72, thr, min_sample),
        }

    # ---- 逐小时按天桶：温度 ----
    temp_hourly = {}
    for m in models:
        buckets = {b: ([], []) for b in range(1, hourly_lead_days + 1)}
        for r in hourly:
            if r["model"] != m:
                continue
            buckets[r["bucket"]][0].append(r["temp_obs"])
            buckets[r["bucket"]][1].append(r["temp_fcst"])
        temp_hourly[m] = {f"{b}d": temp_metrics(buckets[b][0], buckets[b][1], limits, min_sample)
                          for b in range(1, hourly_lead_days + 1)}

    # ---- 逐小时按天桶：降水 ----
    precip_hourly = {}
    for m in models:
        buckets = {b: ([], []) for b in range(1, hourly_lead_days + 1)}
        for r in hourly:
            if r["model"] != m:
                continue
            buckets[r["bucket"]][0].append(r["rain_obs"])
            buckets[r["bucket"]][1].append(r["rain_fcst"])
        precip_hourly[m] = {f"{b}d": precip_binary_metrics(buckets[b][0], buckets[b][1], thr, min_sample)
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

    # ---- 按天按偏移：温度（最高/最低） ----
    temp_daily = {}
    for m in models:
        by_off = defaultdict(lambda: {"max": ([], []), "min": ([], [])})
        for r in daily:
            if r["model"] != m:
                continue
            by_off[r["offset"]]["max"][0].append(r["temp_max_obs"])
            by_off[r["offset"]]["max"][1].append(r["temp_max_fcst"])
            by_off[r["offset"]]["min"][0].append(r["temp_min_obs"])
            by_off[r["offset"]]["min"][1].append(r["temp_min_fcst"])
        temp_daily[m] = {}
        for off in range(1, daily_max_offset + 1):
            maxmm = temp_metrics(by_off[off]["max"][0], by_off[off]["max"][1], limits, min_sample)
            minmm = temp_metrics(by_off[off]["min"][0], by_off[off]["min"][1], limits, min_sample)
            temp_daily[m][f"{off}d"] = {"max": maxmm, "min": minmm}

    # ---- 按天按偏移：降水（二分类 + 分级 TS） ----
    precip_daily = {}
    for m in models:
        by_off = defaultdict(lambda: ([], []))
        for r in daily:
            if r["model"] != m:
                continue
            by_off[r["offset"]][0].append(r["rain_obs"])
            by_off[r["offset"]][1].append(r["rain_fcst"])
        precip_daily[m] = {}
        for off in range(1, daily_max_offset + 1):
            o, f = by_off[off]
            precip_daily[m][f"{off}d"] = {
                "binary": precip_binary_metrics(o, f, thr, min_sample),
                "graded": precip_graded_ts(o, f, "24h", graded_levels, min_sample),
            }

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
                "precip": precip_binary_metrics(ro, rf, thr, min_sample),
            }

    # ---- 时间序列（最近 72h，用于预报 vs 观测叠图） ----
    ts_start = end_dt - timedelta(hours=72)
    timeseries = _build_timeseries(station_ids, models, ts_start, end_dt)

    # ---- 热力图：各日 × 各模型 温度 ±2°C 准确率 ----
    heatmap = _build_heatmap(station_ids, models, hourly, limits, min_sample)

    # ---- 覆盖率 ----
    coverage = _coverage(station_ids, start_dt, end_dt)

    # ---- 月度排名（可选） ----
    ranking = None
    if is_monthly:
        ranking = _ranking(scorecard)

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
    }


def _build_timeseries(station_ids, models, ts_start, end_dt) -> dict:
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
        # 每个模型：从该站最近一次快照取对应时间点的预报
        model_series: dict[str, list] = {m: [] for m in models}
        out[sid] = {"obs": series, "models": model_series}
    # 仅当存在快照时填预报（保持简单：用各模型最近快照按时间匹配）
    for sid in station_ids:
        for m in models:
            snaps = list_forecast_snapshots(sid, m)
            if not snaps:
                continue
            latest = snaps[-1]
            tmap = {t: (latest["data"][m]["temperature_2m"][i],
                       latest["data"][m]["precipitation"][i])
                    for i, t in enumerate(latest["hourly_time"])}
            arr = []
            for pt in out[sid]["obs"]:
                if pt["t"] in tmap:
                    arr.append({"t": pt["t"], "temp": tmap[pt["t"]][0], "rain": tmap[pt["t"]][1]})
                else:
                    arr.append({"t": pt["t"], "temp": None, "rain": None})
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
    rows = []
    for m, sc in scorecard.items():
        t = sc["temp_24h"]
        p = sc["precip_24h"]
        # 综合得分（越高越好）：温度±2°C准确率 + 降水TS，缺项不计
        score = 0.0
        parts = 0
        if t.get("acc2") is not None:
            score += t["acc2"]; parts += 1
        if t.get("rmse") is not None:
            score += max(0.0, 100 - t["rmse"] * 5); parts += 1
        if p.get("ts") is not None:
            score += p["ts"] * 100; parts += 1
        if p.get("acc") is not None:
            score += p["acc"]; parts += 1
        rows.append({"model": m, "score": round(score / parts, 2) if parts else None,
                     "temp_acc2_24h": t.get("acc2"), "precip_ts_24h": p.get("ts")})
    rows.sort(key=lambda x: (x["score"] is None, -(x["score"] or 0)))
    return rows

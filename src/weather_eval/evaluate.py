"""评估引擎：配对观测与预报，调用 cyeva 计算指标，组装报告数据。

评估口径（详见 README，并在报告中注明）：
- 逐小时：按起报后时效（小时）分 1..16 天桶。
  温度（cyeva TemperatureComparison 全量）：RMSE / MAE / MBE / RSS / χ² /
    ±1°C·±2°C 准确率 / 相关系数 r / 回归斜率。
  降水（cyeva PrecipitationComparison 全量）：晴雨二分类（阈值默认 0.1mm）
    准确率 / 命中率 POD / 空报率 FAR / 空报频率 / 漏报率 / TS / ETS / 频率偏差 BIAS，
    连续量 RMSE / MAE / MBE，及 1h 雨强分级（小雨..大暴雨）每级 7 项指标。
  **逐小时降水自 2026-09 起为诊断轨道，不再进综合分**（见下"评分轨道"）。
- 按天：北京时自然日聚合日最高/最低气温、日降水量；按"有效日 − 起报日"的日偏移
  1..16 天分组。温度最高/最低全套指标；降水晴雨 + 连续量 + 24h 累计分级
  （≥0.1/≥10/≥25/≥50/≥100/≥250mm，即小雨..特大暴雨以上）每级 7 项指标。
  覆盖门槛（第一性原理：缺测绝不伪装成数值）：日聚合同时记录非缺测小时数，
  观测与预报任一侧的日覆盖不足 daily_min_hours（默认 20/24）时，该天该要素
  不参与按天评估——降水全缺测日若折算成 0.0 会伪装成"预报无雨"，部分覆盖日的
  日累计系统性偏低会伪装成"漏报"，两者都是把缺测当技巧。

评分轨道（2026-09-06 重构，对抗式审查 P0-1/P0-3 的落地）：
- **降水入分轨道 = 按天 24h 累计、阈值 rain_daily_threshold_mm（默认 1mm）**。
  逐小时 ≥0.1mm 口径下数值模式普遍每小时产生微量降水（drizzle bias），实测
  超报 2.7~15.3 倍、全模式 ETS≤0.054、降水分被压在无区分度的窄带里；24h 累计
  让毛毛雨在求和中自然抵消，1mm/日（业务"有效降水日"）阈值下 ETS 上限恢复到
  0.25（标定扫描见 scripts/calibrate_daily_threshold.py，结论在 README 留档）。
  逐小时晴雨指标保留为诊断视图（明细表），不进综合分。
- **总榜 = 各天桶得分的等权平均（macro-average）**，不再把全部时效样本倒进
  同一个池子。池化总榜把"时效构成差异"直接混进结论：实测 25 源中 14 个在
  同难度对照下名次变动 ≥5。macro 化后每源先在自己的每个可用天桶各得一分、
  再跨桶平均，时效构成从"改变结论"降级为"影响可用桶数"（配合 n_eff 门槛与
  覆盖时效列披露）。分时效榜（同难度基准）与总榜共用同一套桶得分。
- **不确定性入榜**：按天分块 bootstrap（块长 1 天）给出综合分 90% 置信区间、
  冠军频率、与第一名的显著性；权重 ±40% 扰动的冠军分布进报告（P0-2）。
  实现见 stats.py（块重采样同时化解逐小时误差自相关，即 P1-2）。

逐日预报补位（2026-09 新增，daily_source_fallback 开关控制，默认开）：
  多数 API 的逐日预报比逐小时预报覆盖得更远（逐小时常止于 5~10 天，逐日可到
  15 天），逐小时一断供，按天评估就跟着断在第一段时效上。快照可另带一个可选的
  "逐日预报"块（契约见 forecast/base.py），此时按天评估走**双轨**：
  - 该日逐小时覆盖达标 → 用逐小时聚合（与历史存档同一口径，旧数字不变）；
  - 该日逐小时覆盖不足或根本没有逐小时数据 → 回退源自带的日产品值，该条样本
    记 temp_src/rain_src="daily"，与 "hourly" 分开计数并在报告中披露。
  边界（第一性原理，不可越界）：
  * **绝不由日产品反推逐小时序列**——插值/均摊是凭空造出日内变化，会让逐小时
    指标与排行榜出现根本不存在的样本。补位只发生在按天轨道。
  * 观测侧的覆盖门槛不因补位放松：实况当天不足 daily_min_hours 照样不入样。
  * 日产品的日界窗口与"逐小时求和"并非严格同一窗口（相差约 1 小时边界），
    且日最高/最低是源自己的估计量。这是补位口径的固有差异，只能披露不能消除。

统计推断层（2026-09-06 新增，P1-1/P1-2/P1-3 的落地，实现见 stats.py）：
- n 列同时披露 n_eff（有效样本量）：逐小时气温误差强自相关（lag-1 ρ 实测
  0.64~0.91），名义 n 把同一信息重复计数（最大高估 20.5×）。min_sample 判定
  与总榜入围门槛均以 n_eff 为准。
- 相关系数 r 与回归斜率按**站内计算后合并**（r 用 Fisher-z 加权、斜率按样本量
  加权）：跨站池化的 r 混入"复现站间气候差异"这一容易得多的任务，实测与站内
  口径差最多 0.084（约合 1.3 个综合分）。池化值保留在 r_pooled/slope_pooled
  并列披露。
- 「覆盖时效」按各指标**实际参与计算的样本**（两侧值同时非缺测）分别计算
  （temp/rain 分列）：collect 只要该时刻有观测就生成 record，序列尾部的 null
  值曾让该列按"序列长度"虚报（P1-1）。

所有指标附样本数 n；n 或 n_eff < min_sample 视为"样本不足"不出结论（置 None）。

得分体系（排行榜、分时效榜单与时效趋势共用）：
- 把预报质量拆成互不重复的维度，每维度取代表性指标换算成 0~100 的子分后加权平均：
  温度 7 项入分（±2°C/±1°C 准确率、RMSE/MAE 换算分、相关系数、|MBE| 偏差分、回归斜率分），
  降水 5 项入分（ETS（首位）、TS、POD、100−FAR、|BIAS−1| 偏差分）。
  降水中不再入分：晴雨准确率 acc——它主要由气候基率决定（无雨日占 70%+ 全
  答无雨即得高分），不是技巧；ETS 本身已做过基率校正。POD 权重低于
  FAR+BIAS 之和，评分不再奖励"多报占便宜"（旧权重下超报 15 倍的源反而
  高于克制源）。
  各子分截断到 [0,100]，缺项按剩余权重归一（不让单一缺项把整行踢出局）。
  不入分的指标及理由见 TEMP/PRECIP_SCORE_PARTS 注释与 README。
- 综合得分 = mean(温度得分, 降水分)，缺项不计。
- 排行榜（leaderboards）分两层：
  * 分时效榜（"1d".."16d"）：每个天桶一份按综合分排序的完整行。预报难度随时效单调
    上升，天桶是"难度分层"——同桶内各源比较的是同一难度的预报，这是横向比较的公平基准。
    桶内温度分来自逐小时轨道，降水分来自按天累计轨道（该桶 = 起报后第 N 天）。
  * 总榜（"all"）：各天桶综合分的等权平均（macro），行附 90% 置信区间、
    冠军频率、与第一名显著性、n_eff 门槛（min_board_neff）达标标记
    （未达标 = 样本积累中，不参与冠军竞争）与覆盖时效（按各指标实际有效样本）。
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

from . import stats as _stats
from .stats import (
    GROUP_MIN_N,
    binary_counts,
    binary_metrics_from_counts,
    day_block_bootstrap,
    effective_n,
    r_slope_numpy,
    temp_core_numpy,
    weight_champion_distribution,
)
from .timeutil import parse_iso, hour_bucket_days, floor_to_hour
from .storage import load_obs, list_forecast_snapshots

# 逐小时降水分级：cyeva 1h 雨强区间级别（小雨 0.1~1.9 … 大暴雨 ≥20 mm/h）
HOURLY_GRADED_LEVS = ("1", "2", "3", "4", "5")
# 按天降水分级：cyeva 24h 累计级别（+1=≥0.1 … +6=≥250mm）
DAILY_GRADED_LEVS = ("+1", "+2", "+3", "+4", "+5", "+6")
# 每级计算的分级指标（cyeva 分级全套）
GRADED_KEYS = ("acc", "pod", "far", "miss", "ts", "ets", "bias")

# bootstrap 固定种子：同一天数据必须得到同一份置信区间（可复现性）
BOOTSTRAP_SEED = 20260906


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


def _finite_or_none(v: Any) -> float | None:
    """把快照里的值规整为有限浮点数；None/NaN/inf/非数值一律归 None。

    日产品块来自外部 JSON（可能混入 null、字符串、NaN），入库前统一在这里过一遍，
    绝不把缺测伪装成 0.0、也绝不让 inf 流进指标。
    """
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


# ----------------------------------------------------------------- 配对收集
def daily_block_of(snap: dict, model: str) -> dict[str, dict]:
    """抽取快照自带的逐日预报块 → {自然日: {"temp_max","temp_min","precipitation"}}。

    块缺省/结构异常一律返回空 dict——历史存档无此块，行为完全回退到逐小时聚合
    （新字段是纯增量，绝不改变已有结论）。数组越界按缺测处理，与逐小时路径同款
    防护：畸形存档只降级为"该日无日产品"，不拖垮整份报告。
    """
    times = snap.get("daily_time")
    block = snap.get("daily")
    if not isinstance(times, list) or not isinstance(block, dict):
        return {}
    entry = block.get(model)
    if not isinstance(entry, dict):
        return {}
    out: dict[str, dict] = {}
    for i, day in enumerate(times):
        if not isinstance(day, str) or len(day) < 10:
            continue
        out[day[:10]] = {
            k: _at(entry, k, i)
            for k in ("temp_max", "temp_min", "precipitation")
        }
    return out


def _at(entry: dict, key: str, i: int) -> float | None:
    arr = entry.get(key)
    return _finite_or_none(arr[i]) if isinstance(arr, list) and i < len(arr) else None


def collect(station_ids: list[str], models: list[str], start_dt, end_dt,
            hourly_lead_days: int, daily_max_offset_days: int,
            daily_min_hours: int = 20,
            daily_source_fallback: bool = True, *,
            obs_maps: dict[str, dict] | None = None,
            snapshots: dict[tuple[str, str], list[dict]] | None = None
            ) -> tuple[list[dict], list[dict]]:
    """返回 (hourly_records, daily_records)。

    daily_source_fallback：允许用快照自带的逐日预报块为按天评估补位（默认开）。
    关掉后按天轨道与补位前完全一致，用于口径对照/回归。

    obs_maps / snapshots：调用方预加载的观测与快照缓存（build_report 做一次
    IO 全量预载后传入，避免同一批文件被下游各节重复读 4 遍）；缺省时自行加载
    （保持独立调用 collect 的兼容性）。
    """
    # 观测月聚合
    obs_daily: dict[str, dict[str, dict]] = defaultdict(dict)
    hourly_records: list[dict] = []
    for sid in station_ids:
        obs_map = obs_maps[sid] if obs_maps is not None and sid in obs_maps \
            else load_obs(sid)
        # 逐小时配对
        for model in models:
            for snap in (snapshots[(sid, model)] if snapshots is not None
                         and (sid, model) in snapshots
                         else list_forecast_snapshots(sid, model)):
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
    start_day, end_day = start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")
    for sid in station_ids:
        obs_map = obs_maps[sid] if obs_maps is not None and sid in obs_maps \
            else load_obs(sid)
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
            for snap in (snapshots[(sid, model)] if snapshots is not None
                         and (sid, model) in snapshots
                         else list_forecast_snapshots(sid, model)):
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
                    # 源自带的逐日预报块（可缺省）：与逐小时聚合按自然日合并成双轨
                    dblock = daily_block_of(snap, m) if daily_source_fallback else {}
                    for day in sorted(set(fd) | set(dblock)):
                        if day not in obs_daily[sid]:
                            continue
                        # 日块按自然日判窗口（逐小时侧已按整点过滤过，两者语义一致）
                        if day < start_day or day > end_day:
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
                        # ---- 预报侧双轨：逐小时聚合优先，覆盖不足才用源自带日产品 ----
                        d = fd.get(day)
                        h_temp_ok = bool(d and d["n_temp"] >= daily_min_hours
                                         and d["max_temp"] > -math.inf
                                         and d["min_temp"] < math.inf)
                        h_rain_ok = bool(d and d["n_rain"] >= max(daily_min_hours, 1))
                        db = dblock.get(day) or {}
                        if h_temp_ok:
                            f_temp, f_min, temp_src = d["max_temp"], d["min_temp"], "hourly"
                        else:
                            f_temp, f_min = db.get("temp_max"), db.get("temp_min")
                            temp_src = "daily" if (f_temp is not None or f_min is not None) else None
                        if h_rain_ok:
                            f_rain, rain_src = d["sum_rain"], "hourly"
                        else:
                            f_rain = db.get("precipitation")
                            rain_src = "daily" if f_rain is not None else None
                        if o_temp is None and o_min is None and o_rain is None \
                                and f_temp is None and f_min is None and f_rain is None:
                            continue  # 该天无任何可用日聚合，不入样
                        daily_records.append({
                            "station": sid, "model": m, "valid_day": day, "offset": offset,
                            "temp_max_obs": o_temp, "temp_max_fcst": f_temp,
                            "temp_min_obs": o_min, "temp_min_fcst": f_min,
                            "rain_obs": o_rain, "rain_fcst": f_rain,
                            # 来源留档（"hourly"=逐小时聚合 / "daily"=源自带日产品 /
                            # None=该要素无值）—— 补位绝不静默，报告按此披露构成
                            "temp_src": temp_src, "rain_src": rain_src,
                        })
    return hourly_records, daily_records


def daily_source_mix(models: list[str], daily: list[dict]) -> dict[str, dict[str, dict]]:
    """统计按天样本的来源构成：逐小时聚合 vs 源自带日产品（分温度/降水、分日偏移）。

    跨源横向比较按天指标时，"这条样本是逐小时聚合出来的，还是源自己的日产品"
    是口径差异——同一日偏移桶内可能两种来源混在一家或多家身上。这份统计让差异
    可见（报告披露），而不是让读者把口径不同的数字当成同难度对比。

    计数与指标同源：只统计**真正配成对**的样本（观测侧与预报侧同时有值）。
    否则"观测缺测被门槛剔除"的记录也会被算成一条日产品样本，披露出来的构成
    与桶里实际的 n 对不上——披露必须与被披露的东西一一对齐。
    """
    counts: dict[str, dict[str, dict[str, int]]] = {m: {} for m in models}
    for r in daily:
        m = r.get("model")
        if m not in counts:
            continue
        paired = {
            "temp": ((r.get("temp_max_obs") is not None and r.get("temp_max_fcst") is not None)
                     or (r.get("temp_min_obs") is not None and r.get("temp_min_fcst") is not None)),
            "rain": (r.get("rain_obs") is not None and r.get("rain_fcst") is not None),
        }
        for key, field in (("temp", "temp_src"), ("rain", "rain_src")):
            src = r.get(field)
            if not paired[key] or src not in ("hourly", "daily"):
                continue
            slot = counts[m].setdefault(f"{r['offset']}d", {})
            cell = slot.setdefault(key, {"hourly": 0, "daily": 0})
            cell[src] += 1
    # 空桶不输出：报告里"没有样本"就是没有，不必给一个全 0 的结构让人费解
    return {m: slots for m, slots in counts.items() if slots}


# ----------------------------------------------------------------- 指标计算
def temp_metrics(obs_vals, fcst_vals, limits, min_sample, *,
                 n_eff: int | None = None,
                 groups: list | None = None) -> dict:
    """温度全套指标（cyeva TemperatureComparison 全量）。

    n_eff：有效样本量（考虑误差自相关，见 stats.effective_n）。n 或 n_eff 低于
    min_sample 视为样本不足——名义 n 达标但 n_eff 不达标同样不出结论（P1-2）。

    groups：按站分组的 [(obs, fcst), ...]（P1-3）。相关系数 r 与回归斜率在
    各站内部计算（站内样本 ≥ GROUP_MIN_N 才参与），r 按 Fisher-z 加权合并、
    斜率按样本量加权合并——跨站池化的 r 混入"复现站间气候差异"这一容易得多的
    任务，不是对"站内起伏同步程度"的度量。池化口径保留在 r_pooled/slope_pooled
    并列披露。单组/缺省时直接对池化样本计算（r_pooled 即 r）。
    """
    obs = np.asarray(obs_vals, dtype=float)
    fcst = np.asarray(fcst_vals, dtype=float)
    n = _valid_n(obs, fcst)
    out = {"n": n, "n_eff": n_eff,
           "rmse": None, "mae": None, "mbe": None, "rss": None,
           "chi2": None, "r": None, "slope": None,
           "r_pooled": None, "slope_pooled": None,
           **{f"acc{lim}": None for lim in limits}}
    if n < min_sample or n == 0 or (n_eff is not None and n_eff < min_sample):
        return out
    tc = TemperatureComparison(obs, fcst, unit="degC")
    out["rmse"] = _r(tc.calc_rmse())
    out["mae"] = _r(tc.calc_mae())
    out["mbe"] = _r(tc.calc_mbe())
    out["rss"] = _r(tc.calc_rss())
    out["chi2"] = _r(tc.calc_chi_square())
    for lim in limits:
        out[f"acc{lim}"] = _r(tc.calc_diff_accuracy_ratio(limit=lim))
    # r/slope：numpy 充分统计量公式（与 scipy.linregress 同定义，cyeva 亦委托之，
    # 一致性由测试锁定）。池化与站内合并两条口径都算，缺项回退池化。
    r_pooled, slope_pooled = r_slope_numpy(fcst, obs)   # x=预报, y=观测（cyeva 口径）
    out["r_pooled"] = r_pooled
    out["slope_pooled"] = slope_pooled
    r, slope = r_pooled, slope_pooled
    if groups and len(groups) >= 1:
        rs, ns, ss = [], [], []
        for go, gf in groups:
            goa = np.asarray(go, dtype=float)
            gfa = np.asarray(gf, dtype=float)
            gn = _valid_n(goa, gfa)
            if gn < GROUP_MIN_N:
                continue
            rg, sg = r_slope_numpy(gfa, goa)
            rs.append(rg)
            ns.append(gn)
            ss.append(sg)
        r_c = _stats.fisher_z_combine(rs, ns)
        s_c = _stats.weighted_mean_combine(ss, ns)
        r = r_c if r_c is not None else r_pooled
        slope = s_c if s_c is not None else slope_pooled
    out["r"] = r
    out["slope"] = slope
    return out


def temp_curve_metrics(obs_vals, fcst_vals, min_sample) -> dict:
    """1..72h 逐时效曲线专用轻量路径：只算页面消费的 rmse 与 ±2°C 准确率。

    曲线此前对每个 (模型, 时效) 跑一遍全量 cyeva（含 linregress/RSS/χ²），
    4,200 次调用里约 1,800 次只用到 2 个字段（P3-3）。numpy 快速路径与 cyeva
    全量路径同口径：剔 NaN 对、结果 round4（cyeva result_round_digit）、再过
    与全量路径相同的 _r（round3）终端舍入——报告里的数字不允许
    "曲线一套、表格一套"（一致性由测试锁定）。"""
    o = np.asarray(obs_vals, dtype=float)
    f = np.asarray(fcst_vals, dtype=float)
    n = _valid_n(o, f)
    if n < min_sample:
        return {"n": n, "rmse": None, "acc2": None}
    mm = temp_core_numpy(o, f)
    return {"n": n, "rmse": _r(mm.get("rmse")), "acc2": _r(mm.get("acc2"))}


def precip_metrics(obs_vals, fcst_vals, threshold, min_sample,
                   kind: str | None = None, graded_levs: tuple = (),
                   n_eff: int | None = None) -> dict:
    """降水全套指标：晴雨二分类 8 项 + 连续量 3 项（+ 可选分级每级 7 项）。

    kind/graded_levs：传 "1h"+("1".."5") 算逐小时雨强区间分级，
    传 "24h"+("+1".."+6") 算按天累计分级；不传则只算晴雨与连续量。
    """
    obs = np.asarray(obs_vals, dtype=float)
    fcst = np.asarray(fcst_vals, dtype=float)
    n = _valid_n(obs, fcst)
    out = {"n": n, "n_eff": n_eff,
           "acc": None, "pod": None, "far": None, "farate": None,
           "miss": None, "ts": None, "ets": None, "bias": None,
           "rmse": None, "mae": None, "mbe": None}
    if graded_levs:
        out["graded"] = {lev: None for lev in graded_levs}
    if n < min_sample or n == 0 or (n_eff is not None and n_eff < min_sample):
        return out
    pc = PrecipitationComparison(obs, fcst, unit="mm")
    out["acc"] = _r(pc.calc_threshold_accuracy_ratio(threshold=threshold, compare=">="))
    out["pod"] = _r(pc.calc_threshold_hit_ratio(threshold=threshold, compare=">="))
    out["far"] = _r(pc.calc_threshold_false_alarm_ratio(threshold=threshold, compare=">="))
    out["miss"] = _r(pc.calc_threshold_miss_ratio(threshold=threshold, compare=">="))
    out["ts"] = _r(pc.calc_threshold_ts(threshold=threshold, compare=">="))
    out["bias"] = _r(pc.calc_threshold_bias_score(threshold=threshold, compare=">="))
    # ETS/空报频率：与上面 6 项同口径手工二值化，调 cyeva 的二分类统计函数，
    # 保证同一份样本内所有晴雨指标口径一致。掩膜必须只剔 NaN、保留 inf——
    # cyeva 的 drop_nan 是 NaN 判定（x != x），inf 会被 threshold_binarize 判为
    # "有雨"；若用 isfinite 会在含 inf 的样本上与同函数内 cyeva 六项指标落到
    # 不同的样本集合（口径分裂）。
    # 二值化用**原值**比较（cyeva threshold_binarize 不做源舍入；本项目实际
    # 调用全部经关键字传参，source_round_digit 装饰器不生效）——旧实现的
    # round2 二值化会在 [thr−0.005, thr) 边界值上与 cyeva 类路径分裂。
    m = ~np.isnan(obs) & ~np.isnan(fcst)
    ob = obs[m] >= threshold
    fb = fcst[m] >= threshold
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


def precip_binary_metrics(obs_vals, fcst_vals, threshold, min_sample,
                          n_eff: int | None = None) -> dict:
    """评分轨道专用的按天晴雨二分类指标（阈值 rain_daily_threshold_mm）。

    纯 numpy 列联计数（stats.binary_counts / binary_metrics_from_counts），与
    cyeva calc_threshold_* 同口径（剔 NaN 对、原值比阈值、结果 round4，一致性
    由测试锁定）。相对 cyeva 全量构造省掉 pint 单位换算开销——评分轨道需要对
    每个天桶 × 每个模型各算一遍，cyeva 全量在该量级下构成构建热点。
    ts/ets 为 0~1 比值，acc/pod/far 为百分数，bias 为比值（与评分换算函数约定一致）。
    """
    obs = np.asarray(obs_vals, dtype=float)
    fcst = np.asarray(fcst_vals, dtype=float)
    m = ~np.isnan(obs) & ~np.isnan(fcst)
    n = int(m.sum())
    out = {"n": n, "n_eff": n_eff, "acc": None, "pod": None, "far": None,
           "ts": None, "ets": None, "bias": None}
    if n < min_sample or n == 0 or (n_eff is not None and n_eff < min_sample):
        return out
    h, fa, mi, c = binary_counts(obs[m], fcst[m], threshold)
    bm = binary_metrics_from_counts(h, fa, mi, c)
    for k in ("acc", "pod", "far", "ts", "ets", "bias"):
        out[k] = _r(bm[k])
    return out


# ----------------------------------------------------------------- 得分体系
# 设计（第一性原理）：先拆维度、再选代表指标、后加权——不把互相冗余的指标重复计账。
# 每项以 (指标键, 权重, 白话标签, 换算说明, 换算函数) 描述：换算函数把指标映射到
# 0~100 的子分（统一截断到 [0,100]），权重决定该维度对总分的话语权。
#
# 温度（5 个维度、7 项入分）：
#   报准比例 acc2/acc1 · 误差幅度 RMSE/MAE · 起伏节奏 r · 系统偏差 |MBE| · 幅度校准 slope。
#   不入分：RSS 与 χ² —— χ²=RMSE²、RSS=n×χ²，是样本量的函数而非预报技巧，只进明细表。
# 降水（4 个维度、5 项入分；2026-09-06 重构）：
#   晴雨综合技巧 ETS（首位）/TS · 命中 POD · 空报 FAR · 频率无偏 BIAS。
#   不入分：晴雨准确率 acc——主要由气候基率决定（无雨日 70%+，全答无雨即得高分），
#   ETS 已含基率校正，acc 只进明细表；漏报率（=100−POD，纯冗余）、空报频率 POFD
#   （与 FAR 同族仅分母不同）、雨量 RMSE/MAE/MBE（连续雨量误差由个别强降水时段
#   主导、随样本期气候波动大，跨源横向比较不公平，只进明细表）。
#   POD 权重 (0.15) 低于 FAR+BIAS (0.15+0.10=0.25)：评分不奖励"多报占便宜"
#   （旧权重下超报 15 倍的源反而高于克制源）。
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
    ("ets", 0.35, "晴雨 ETS 评分", "ETS×100（对“瞎蒙也能蒙对”做过校正，技巧首位）",
     lambda v: v * 100),
    ("ts", 0.25, "晴雨 TS 评分", "TS×100（报中/空报/漏报一账清）",
     lambda v: v * 100),
    ("pod", 0.15, "命中率 POD", "百分比直接入分（漏报少）",
     lambda v: v),
    ("far", 0.15, "空报率换算分", "100 − FAR（不喊“狼来了” = 满分）",
     lambda v: 100 - v),
    ("bias", 0.10, "频率偏差换算分", "100 − |BIAS−1|×100（报雨频率恰如其分 = 满分）",
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
    """降水分(0~100)：5 项指标按 PRECIP_SCORE_PARTS 权重加权，缺项按剩余权重归一。"""
    return _weighted_score(PRECIP_SCORE_PARTS, p)


def overall_score(t: dict, p: dict) -> float | None:
    """综合得分：温度分与降水分的均分，缺项不计。"""
    return _mean_or_none([temp_score(t), precip_score(p)])


# ----------------------------------------------------------------- 有效样本量
def _n_eff_temp(recs: list[dict]) -> int | None:
    """温度误差序列的有效样本量（按站独立估计后求和）。

    同一有效时刻会被多个起报覆盖（评估把每个 (起报, 有效时刻) 记为一条样本，
    这是指标口径）；自相关必须沿"时刻"单序列估计，故同一有效时刻只保留
    最新一版起报（最小 lead）的误差——同一预报值重复出现会人为抬高 ρ₁。
    """
    by: dict[str, dict[str, tuple[int, float]]] = defaultdict(dict)
    for r in recs:
        o, f = r["temp_obs"], r["temp_fcst"]
        if o is None or f is None:
            continue
        cur = by[r["station"]].get(r["valid_iso"])
        if cur is None or r["lead"] < cur[0]:
            by[r["station"]][r["valid_iso"]] = (r["lead"], f - o)
    if not by:
        return None
    series = {sid: [err for _vt, (_lead, err) in sorted(d.items())]
              for sid, d in by.items()}
    return _stats.n_eff_from_station_series(series)


def _n_eff_rain(recs: list[dict], thr: float, time_key: str,
                lead_key: str) -> int | None:
    """降水"报错"序列的有效样本量：晴雨判定不一致的 0/1 指示序列（按站）。"""
    by: dict[str, dict[str, tuple[int, float]]] = defaultdict(dict)
    for r in recs:
        o, f = r["rain_obs"], r["rain_fcst"]
        if o is None or f is None:
            continue
        cur = by[r["station"]].get(r[time_key])
        if cur is None or r[lead_key] < cur[0]:
            by[r["station"]][r[time_key]] = (
                r[lead_key], 1.0 if ((o >= thr) != (f >= thr)) else 0.0)
    if not by:
        return None
    series = {sid: [err for _t, (_lead, err) in sorted(d.items())]
              for sid, d in by.items()}
    return _stats.n_eff_from_station_series(series)


def _station_groups(recs: list[dict], ka: str, kb: str) -> list[tuple[list, list]]:
    """按站分组取 (obs, fcst) 值对列表（站序稳定），供 temp_metrics 的站内合并。"""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by[r["station"]].append(r)
    return [([r[ka] for r in rs], [r[kb] for r in rs]) for _sid, rs in sorted(by.items())]


# ----------------------------------------------------------------- 报告组装
def _pool(records, keys, filt=None):
    o, f = [], []
    for r in records:
        if filt and not filt(r):
            continue
        o.append(r[keys[0]])
        f.append(r[keys[1]])
    return o, f


def _preload(station_ids: list[str], models: list[str]) -> tuple[dict, dict]:
    """一次 IO 读全所有观测与快照，下游各节共用（P3-2：同批文件此前被读 4 遍）。"""
    obs_maps = {sid: load_obs(sid) for sid in station_ids}
    snapshots = {(sid, m): list_forecast_snapshots(sid, m)
                 for sid in station_ids for m in models}
    return obs_maps, snapshots


def build_report(station_ids, models, eval_cfg, start_dt, end_dt,
                 period_label: str, is_monthly: bool = False) -> dict:
    limits = eval_cfg["temp_accuracy_limits"]
    thr = eval_cfg["rain_threshold_mm"]
    thr_daily = eval_cfg.get("rain_daily_threshold_mm", 1.0)
    min_sample = eval_cfg["min_sample"]
    min_board_neff = eval_cfg.get("min_board_neff", 30)
    bootstrap_runs = int(eval_cfg.get("bootstrap_runs", 500))
    sensitivity_runs = int(eval_cfg.get("sensitivity_runs", 500))
    hourly_lead_days = eval_cfg["hourly_lead_days"]
    daily_max_offset = eval_cfg["daily_max_offset_days"]
    daily_min_hours = eval_cfg.get("daily_min_hours", 20)
    daily_source_fallback = bool(eval_cfg.get("daily_source_fallback", True))

    obs_maps, snapshots = _preload(station_ids, models)
    hourly, daily = collect(station_ids, models, start_dt, end_dt,
                            hourly_lead_days, daily_max_offset, daily_min_hours,
                            daily_source_fallback,
                            obs_maps=obs_maps, snapshots=snapshots)

    # 按模型分组一次，后续所有 per-model 统计只遍历各自的记录（不做全量重扫）
    by_model: dict[str, list] = {m: [] for m in models}
    for r in hourly:
        if r["model"] in by_model:
            by_model[r["model"]].append(r)
    daily_by_model: dict[str, list] = {m: [] for m in models}
    for r in daily:
        if r["model"] in daily_by_model:
            daily_by_model[r["model"]].append(r)

    # ---- 评分卡（模型级）----
    # 24h/72h 池 = 固定时效窗的 pooled 指标；all 池 = 评估窗口内全部逐小时样本。
    # 评分卡是明细表数据源（诊断），综合分不再从这里出（见总榜 macro 化）。
    scorecard = {}
    for m in models:
        recs = by_model[m]
        h24 = [r for r in recs if 1 <= r["lead"] <= 24]
        h72 = [r for r in recs if 1 <= r["lead"] <= 72]
        g24 = _station_groups(h24, "temp_obs", "temp_fcst")
        g72 = _station_groups(h72, "temp_obs", "temp_fcst")
        gall = _station_groups(recs, "temp_obs", "temp_fcst")
        neff_all = _n_eff_temp(recs)
        scorecard[m] = {
            "temp_24h": temp_metrics(*_pool(h24, ("temp_obs", "temp_fcst")),
                                     limits, min_sample,
                                     n_eff=_n_eff_temp(h24), groups=g24),
            "temp_72h": temp_metrics(*_pool(h72, ("temp_obs", "temp_fcst")),
                                     limits, min_sample,
                                     n_eff=_n_eff_temp(h72), groups=g72),
            "temp_all": temp_metrics(*_pool(recs, ("temp_obs", "temp_fcst")),
                                     limits, min_sample,
                                     n_eff=neff_all, groups=gall),
            "precip_24h": precip_metrics(*_pool(h24, ("rain_obs", "rain_fcst")),
                                         thr, min_sample,
                                         n_eff=_n_eff_rain(h24, thr, "valid_iso", "lead")),
            "precip_72h": precip_metrics(*_pool(h72, ("rain_obs", "rain_fcst")),
                                         thr, min_sample,
                                         n_eff=_n_eff_rain(h72, thr, "valid_iso", "lead")),
            "precip_all": precip_metrics(*_pool(recs, ("rain_obs", "rain_fcst")),
                                         thr, min_sample,
                                         n_eff=_n_eff_rain(recs, thr, "valid_iso", "lead")),
        }

    # ---- 逐小时按天桶：温度（入分）/ 逐小时降水（诊断，含 1h 雨强分级） ----
    temp_hourly: dict[str, dict] = {}
    precip_hourly: dict[str, dict] = {}
    for m in models:
        by_bucket: dict[int, list] = defaultdict(list)
        for r in by_model[m]:
            by_bucket[r["bucket"]].append(r)
        temp_hourly[m] = {}
        precip_hourly[m] = {}
        for b in range(1, hourly_lead_days + 1):
            recs = by_bucket.get(b, [])
            to, tf = _pool(recs, ("temp_obs", "temp_fcst"))
            ro, rf = _pool(recs, ("rain_obs", "rain_fcst"))
            temp_hourly[m][f"{b}d"] = temp_metrics(
                to, tf, limits, min_sample,
                n_eff=_n_eff_temp(recs),
                groups=_station_groups(recs, "temp_obs", "temp_fcst"))
            precip_hourly[m][f"{b}d"] = precip_metrics(
                ro, rf, thr, min_sample, kind="1h", graded_levs=HOURLY_GRADED_LEVS,
                n_eff=_n_eff_rain(recs, thr, "valid_iso", "lead"))

    # ---- 降水的评分轨道：按天累计（24h）+ rain_daily_threshold_mm 阈值 ----
    # 每模型每日偏移一份二分类指标（P0-3），入分的同时披露 acc 供明细。
    precip_score_daily: dict[str, dict] = {}
    for m in models:
        by_off: dict[int, list] = defaultdict(list)
        for r in daily_by_model[m]:
            by_off[r["offset"]].append(r)
        precip_score_daily[m] = {}
        for off in range(1, hourly_lead_days + 1):
            recs = by_off.get(off, [])
            ro, rf = _pool(recs, ("rain_obs", "rain_fcst"))
            precip_score_daily[m][f"{off}d"] = precip_binary_metrics(
                ro, rf, thr_daily, min_sample,
                n_eff=_n_eff_rain(recs, thr_daily, "valid_day", "offset"))

    # ---- 逐小时逐时效曲线（1..72h）温度 RMSE / ±2°C 准确率（轻量路径）----
    temp_lead_curve: dict[str, dict] = {m: {} for m in models}
    for m in models:
        by_lead = defaultdict(lambda: ([], []))
        for r in by_model[m]:
            if 1 <= r["lead"] <= 72:
                by_lead[r["lead"]][0].append(r["temp_obs"])
                by_lead[r["lead"]][1].append(r["temp_fcst"])
        for h, (o, f) in by_lead.items():
            mm = temp_curve_metrics(o, f, min_sample)
            # 稀疏存储：无样本的时效不输出（P3-4；页面把缺键当断点）
            if mm.get("rmse") is not None or mm.get("acc2") is not None:
                temp_lead_curve[m][f"{h}h"] = {"rmse": mm.get("rmse"),
                                               "acc2": mm.get("acc2")}

    # ---- 按天按偏移：温度（最高/最低） / 降水（诊断口径 0.1mm，含 24h 累计分级） ----
    temp_daily: dict[str, dict] = {}
    precip_daily: dict[str, dict] = {}
    for m in models:
        by_off = defaultdict(lambda: {"max": ([], []), "min": ([], []), "rain": ([], [])})
        for r in daily_by_model[m]:
            by_off[r["offset"]]["max"][0].append(r["temp_max_obs"])
            by_off[r["offset"]]["max"][1].append(r["temp_max_fcst"])
            by_off[r["offset"]]["min"][0].append(r["temp_min_obs"])
            by_off[r["offset"]]["min"][1].append(r["temp_min_fcst"])
            by_off[r["offset"]]["rain"][0].append(r["rain_obs"])
            by_off[r["offset"]]["rain"][1].append(r["rain_fcst"])
        temp_daily[m] = {}
        precip_daily[m] = {}
        for off in range(1, daily_max_offset + 1):
            sub = [r for r in daily_by_model[m] if r["offset"] == off]
            # 按天温度样本量小（多数站 n<30 不参与站内合并，回退池化），仍按站分组
            gmax = _station_groups(sub, "temp_max_obs", "temp_max_fcst")
            gmin = _station_groups(sub, "temp_min_obs", "temp_min_fcst")
            maxmm = temp_metrics(by_off[off]["max"][0], by_off[off]["max"][1],
                                 limits, min_sample, groups=gmax)
            minmm = temp_metrics(by_off[off]["min"][0], by_off[off]["min"][1],
                                 limits, min_sample, groups=gmin)
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
    timeseries = _build_timeseries(station_ids, models, ts_start, end_dt,
                                   obs_maps=obs_maps, snapshots=snapshots)

    # ---- 热力图：各日 × 各模型 温度 ±2°C 准确率 ----
    heatmap = _build_heatmap(models, hourly, limits, min_sample)

    # ---- 覆盖率 ----
    coverage = _coverage(station_ids, start_dt, end_dt, obs_maps=obs_maps)

    # ---- 口径注记（如 AccuWeather 最近城市吸附），随报告元数据输出 ----
    model_caveats = _model_caveats(station_ids, models, snapshots=snapshots)

    # ---- 排行榜：分时效榜（每个天桶一份）+ 总榜（天桶 macro 平均 + 不确定性）----
    # 表格排行榜、冠军横幅与趋势图共用同一套桶得分。
    leaderboards = _lead_leaderboards(models, temp_hourly, precip_score_daily,
                                      hourly_lead_days)
    leaderboards["all"] = _overall_board(
        models, temp_hourly, precip_score_daily, hourly_lead_days,
        by_model, daily_by_model,
        n_eff_all={m: _n_eff_temp(by_model[m]) for m in models},
        min_sample=min_sample, min_board_neff=min_board_neff,
        thr_daily=thr_daily, bootstrap_runs=bootstrap_runs)
    qualified_models = {r["model"] for r in leaderboards["all"] if r.get("qualified")}
    weight_sensitivity = {
        "runs": sensitivity_runs,
        "champions": _weight_sensitivity_champions(
            models, temp_hourly, precip_score_daily, hourly_lead_days,
            sensitivity_runs, qualified_models),
    }

    # ---- 得分随时效衰减（综合/温度/降水，天桶 1..N）：排行榜的"趋势版" ----
    score_trend = _score_trend(models, temp_hourly, precip_score_daily,
                               hourly_lead_days)

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
            "rain_daily_threshold_mm": thr_daily,
            "min_sample": min_sample,
            "min_board_neff": min_board_neff,
            "bootstrap_runs": bootstrap_runs,
            "daily_min_hours": daily_min_hours,
            "daily_source_fallback": daily_source_fallback,
            "model_caveats": model_caveats,
            # 不确定性披露：冠军在不同权重方案下的分布（P0-2.2）
            "weight_sensitivity": weight_sensitivity,
        },
        "coverage": coverage,
        "scorecard": scorecard,
        "temp_hourly": temp_hourly,
        "precip_hourly": precip_hourly,
        "temp_lead_curve": temp_lead_curve,
        "temp_daily": temp_daily,
        "precip_daily": precip_daily,
        # 降水的评分轨道指标（按天累计 + 1mm 阈值），榜单与明细共用
        "precip_score_daily": precip_score_daily,
        # 按天样本的来源构成（逐小时聚合 / 源自带日产品补位），分模型分日偏移
        "daily_source_mix": (daily_source_mix(models, daily)
                             if daily_source_fallback else {}),
        "per_station": per_station,
        "timeseries": timeseries,
        "heatmap": heatmap,
        "leaderboards": leaderboards,
        "score_trend": score_trend,
    }


def _weight_sensitivity_champions(models, temp_hourly, precip_score_daily,
                                  hourly_lead_days, runs,
                                  qualified: set[str]) -> list[dict]:
    """权重 ±40% 扰动下的冠军分布（P0-2.2）：桶得分重组冠军计 500 次。

    指标值不随权重变化，只需把**已换算并截断**的子分张量按扰动权重重组——
    向量化实现（stats.weight_champion_distribution），成本可忽略。
    qualified：总榜入围者（n_eff 门槛 + 维度齐备）；敏感性回答的是
    "入围者之间的冠军之争对权重有多敏感"，未入围者不在轮换范围内。
    """
    _eligible_of = lambda m: m in qualified
    def tensor(metric_dicts, parts, eligible):
        T = np.full((len(models), hourly_lead_days, len(parts)), np.nan)
        for mi, m in enumerate(models):
            if not eligible[mi]:
                continue      # 未入围者不参与冠军竞争（即使扰动权重也不会轮到它）
            for b in range(1, hourly_lead_days + 1):
                md = metric_dicts[m].get(f"{b}d")
                if not md:
                    continue
                for ki, (key, _w, _l, _mp, fn) in enumerate(parts):
                    v = md.get(key)
                    if v is None:
                        continue
                    T[mi, b - 1, ki] = _clamp100(fn(v))
        return T

    eligible = [_eligible_of(m) for m in models]
    t_sub = tensor(temp_hourly, TEMP_SCORE_PARTS, eligible)
    p_sub = tensor(precip_score_daily, PRECIP_SCORE_PARTS, eligible)
    dist = weight_champion_distribution(t_sub, p_sub, TEMP_SCORE_PARTS,
                                        PRECIP_SCORE_PARTS, runs=runs)
    return [{"model": models[d["index"]], "pct": d["pct"]} for d in dist]


def _build_timeseries(station_ids, models, ts_start, end_dt, *,
                      obs_maps=None, snapshots=None) -> dict:
    """最近 72h 的"预报 vs 实况"叠图数据。

    预报线的取值口径：对每个时刻，取**当时已发布的最新一版预报**（按起报时间
    升序遍历快照，后发布的覆盖同一时刻的旧值；时效即处于 1~24h 内）。
    只使用"发布时间不晚于窗口末尾"的快照——月度归档时排除归档之后新抓的
    快照，保证归档页与当时可见的预报一致、且时间轴有交集（若始终取全局
    最新快照，归档页的预报线会整段缺失）。
    """
    def r2(v):
        return None if v is None else round(float(v), 2)

    out: dict[str, Any] = {}
    for sid in station_ids:
        obs_map = obs_maps[sid] if obs_maps is not None and sid in obs_maps \
            else load_obs(sid)
        series = []
        # 以 obs 时间轴为准（取窗口内 obs 时间）
        for tstr, rec in sorted(obs_map.items()):
            vt = parse_iso(tstr)
            if vt < ts_start or vt > end_dt:
                continue
            series.append({"t": tstr, "temp": r2(rec.get("temp")),
                           "rain": r2(rec.get("rain"))})
        model_series: dict[str, list] = {m: [] for m in models}
        out[sid] = {"obs": series, "models": model_series}
    for sid in station_ids:
        for m in models:
            snaps = (snapshots[(sid, m)] if snapshots is not None
                     and (sid, m) in snapshots
                     else list_forecast_snapshots(sid, m))
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
                            "temp": r2(v[0]) if v else None,
                            "rain": r2(v[1]) if v else None})
            out[sid]["models"][m] = arr
    return out


def _build_heatmap(models, hourly, limits, min_sample) -> list[dict]:
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
            mm = temp_curve_metrics(o, f, min_sample)
            acc = mm.get("acc2")
            cells.append({"date": d, "model": m, "acc2": acc, "n": mm.get("n", 0)})
    return cells


def _coverage(station_ids, start_dt, end_dt, *, obs_maps=None) -> dict:
    """观测覆盖率。分母按"该站实际已有观测的时段"截断：观测开始前/月初尚不存在
    的时段计入分母只会稀释数字、误导读者（样本不是缺失，而是尚不存在）。
    last_obs = 全站最新观测时刻：观测源静默停摆（页面数据滞后）时这是读者
    可见的直接信号（P2-1 的报告侧配套）。"""
    expected = 0
    got = 0
    first_obs: str | None = None
    last_obs: str | None = None
    for sid in station_ids:
        obs_map = obs_maps[sid] if obs_maps is not None and sid in obs_maps \
            else load_obs(sid)
        times_in_window = [parse_iso(t) for t in obs_map
                           if start_dt <= parse_iso(t) <= end_dt]
        got += len(times_in_window)
        if obs_map:
            latest = max(obs_map)
            if last_obs is None or latest > last_obs:
                last_obs = latest
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
        "last_obs": last_obs,
    }


def _model_caveats(station_ids, models, *, snapshots=None) -> dict[str, str]:
    """从快照 meta 提取影响榜单解读的口径注记（数据驱动，模板只负责呈现）。

    已知注记：AccuWeather / MSN（中国天气网）等"最近城市吸附"定位——快照留档了
    location_name 与 haversine 吸附距离，其样本代表距站点数公里的城市而非站点
    格点，公开榜单必须让读者知情（README 有说明，但只看报告的读者看不到）。
    吸附距离是全站均值，而各站吸附到的城市名不同——只取第一站的名字会让人误以为
    所有站都定位到同一城市，故多站时列全（名称集合天然去重）。
    """
    dist: dict[str, list[float]] = defaultdict(list)
    names: dict[str, set[str]] = defaultdict(set)
    for sid in station_ids:
        for model in models:
            for snap in (snapshots[(sid, model)] if snapshots is not None
                         and (sid, model) in snapshots
                         else list_forecast_snapshots(sid, model)):
                d = snap.get("location_distance_km")
                if d is not None:
                    dist[model].append(float(d))
                nm = snap.get("location_name")
                if nm:
                    names[model].add(str(nm))
    out = {}
    for model, ds in dist.items():
        avg = round(sum(ds) / len(ds), 1)
        got = sorted(names.get(model, ()))
        who = f"「{got[0]}」" if len(got) == 1 else \
            (f"各站最近城市（{'、'.join(got)}）" if got else "最近城市")
        out[model] = (f"最近城市吸附：定位到距站点平均约 {avg} km 的{who}，"
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
    """综合分降序、None 沉底；未达 n_eff 门槛的源（qualified=False）排在其后、
    仍按分数序——榜单语义是"达标者先竞技，未达标者列样本积累中"（P0-2.3）。"""
    def key(r):
        q = r.get("qualified", True)
        s = r["score"]
        return (s is None, not q, -(s or 0))
    return sorted(rows, key=key)


def _lead_leaderboards(models, temp_hourly, precip_score_daily, hourly_lead_days) -> dict:
    """分时效排行榜：每个天桶（第 1..N 天）一份按综合分排序的完整行。

    预报难度随时效单调上升，天桶即"难度分层"——同桶内比较对各家才是同难度的。
    桶内温度分来自逐小时轨道；降水分来自按天累计轨道（阈值 rain_daily_threshold_mm，
    该桶 = 起报后第 N 天）。前端表格切换时效/排序只重排，不重算。
    """
    boards: dict[str, list[dict]] = {}
    for b in range(1, hourly_lead_days + 1):
        bk = f"{b}d"
        rows = []
        for m in models:
            t = temp_hourly[m].get(bk) or {}
            p = precip_score_daily[m].get(bk) or {}
            rows.append(_board_row(
                m, t, p,
                # 维度齐备门槛与总榜同构：温度或降水缺一个维度的"综合分"与
                # 两维齐备的综合分不可比（缺项不计会让单维分虚高），降为
                # 未达标行——分数保留在行内，名次让给齐备者
                qualified=(overall_score(t, p) is not None
                           and temp_score(t) is not None
                           and precip_score(p) is not None)))
        boards[bk] = _rank_rows(rows)
    return boards


def _bucket_display_mean(metric_dicts: dict, key: str, hourly_lead_days: int):
    """各天桶某指标值的等权平均（总榜行展示用；与 macro 得分同一平均口径）。"""
    vals = []
    for b in range(1, hourly_lead_days + 1):
        md = metric_dicts.get(f"{b}d") or {}
        v = md.get(key)
        if v is not None:
            vals.append(v)
    return round(sum(vals) / len(vals), 3) if vals else None


def _overall_board(models, temp_hourly, precip_score_daily, hourly_lead_days,
                   by_model, daily_by_model, n_eff_all, min_sample, min_board_neff,
                   thr_daily, bootstrap_runs) -> list[dict]:
    """全时效总榜：各天桶综合分的**等权平均**（macro-average）。

    公平性（第一性原理，P0-1）：预报难度随时效单调上升，"综合谁最准"只有两种
    不作弊的答法——限定共同覆盖窗口（扔掉大半数据），或让每个源先在自己的每个
    可用天桶各得一分、再跨桶平均。采用后者：时效构成差异从"改变结论"降级为
    "影响各源的可用桶数"；桶内难度对齐由天桶划分保证，跨桶平均不引入
    "临近时效样本天然更多"的池化偏置。旧池化口径的问题（实测 25 源中 14 个
    在同难度对照下名次变动 ≥5）见对抗式审查报告 P0-1。

    不确定性（P0-2）：每行附 90% 置信区间（按天分块 bootstrap）、冠军频率、
    与第一名显著性；n_eff < min_board_neff 的源 qualified=False，排在其后
    显示"样本积累中"，不参与冠军竞争（156 条样本争冠军的教训）。
    """
    rows = []
    for m in models:
        bucket_overalls, bucket_temps, bucket_precips = [], [], []
        n_buckets = 0
        for b in range(1, hourly_lead_days + 1):
            t = temp_hourly[m].get(f"{b}d") or {}
            p = precip_score_daily[m].get(f"{b}d") or {}
            ov = overall_score(t, p)
            if ov is not None:
                n_buckets += 1
            bucket_overalls.append(ov)
            bucket_temps.append(temp_score(t))
            bucket_precips.append(precip_score(p))
        score = _mean_or_none(bucket_overalls)
        t_score = _mean_or_none(bucket_temps)
        p_score = _mean_or_none(bucket_precips)
        neff = n_eff_all.get(m)
        # 覆盖时效（P1-1）：按各指标实际参与计算的样本（两侧值同时非缺测）计算
        lead_days = _valid_lead_days(by_model[m], "temp_obs", "temp_fcst")
        rain_days = _valid_rain_days(daily_by_model[m])
        n_temp = sum(1 for r in by_model[m]
                     if r["temp_obs"] is not None and r["temp_fcst"] is not None)
        n_rain = sum(1 for r in daily_by_model[m]
                     if r["rain_obs"] is not None and r["rain_fcst"] is not None)
        row = _board_row(
            m,
            {"acc2": _bucket_display_mean(temp_hourly[m], "acc2", hourly_lead_days),
             "rmse": _bucket_display_mean(temp_hourly[m], "rmse", hourly_lead_days),
             "n": n_temp},
            {"ts": _bucket_display_mean(precip_score_daily[m], "ts", hourly_lead_days),
             "ets": _bucket_display_mean(precip_score_daily[m], "ets", hourly_lead_days),
             "n": n_rain},
            score=score, temp_score=t_score, precip_score=p_score,
            n=n_temp, n_precip=n_rain, n_eff=neff,
            # 达标三条件（P0-2.3）：n_eff 门槛 + 温度/降水两个维度都有分。
            # "综合"承诺的是两维各半——只有温度维有分的源拿温度分与别人的
            # (温度+降水)/2 比不是同口径（实测 MSN 无按天降水样本却凭温度分
            # 坐上 95.9 的"综合冠军"），维度不齐 = 综合结论未到位 = 样本积累中。
            qualified=(score is not None and neff is not None and neff >= min_board_neff
                       and t_score is not None and p_score is not None),
            lead_days=lead_days, rain_days=rain_days, n_buckets=n_buckets,
        )
        rows.append(row)
    # bootstrap 的冠军频率/显著性只作用于入围者（行内 qualified 已含 n_eff
    # 门槛与维度齐备两个条件）；置信区间全员提供
    bootstrap = day_block_bootstrap(
        [r for m in models for r in by_model[m]],
        [r for m in models for r in daily_by_model[m]],
        models, hourly_lead_days, thr_daily,
        TEMP_SCORE_PARTS, PRECIP_SCORE_PARTS, min_sample,
        runs=max(100, bootstrap_runs), seed=BOOTSTRAP_SEED,
        eligible=[r.get("qualified", False) for r in rows])
    for r in rows:
        b = bootstrap.get(r["model"], {})
        r["ci90"] = b.get("ci90")
        r["champion_pct"] = b.get("champion_pct")
        r["sig_vs_top"] = b.get("sig_vs_top")
    return _rank_rows(rows)


def _valid_lead_days(recs: list[dict], ka: str, kb: str) -> int | None:
    """温度轨道覆盖时效：实际参与指标计算的样本（两侧值同时非缺测）的最长 lead，
    向上取整到天。collect 对"有观测但值为 null"的时刻也生成 record——旧实现
    用全量 records 的 max(lead) 会把序列尾部的 null 段虚报成覆盖（P1-1）。"""
    leads = [r["lead"] for r in recs if r[ka] is not None and r[kb] is not None]
    return math.ceil(max(leads) / 24) if leads else None


def _valid_rain_days(recs: list[dict]) -> int | None:
    """降水轨道覆盖时效：按天样本里两侧同时非缺测的最长日偏移。"""
    offs = [r["offset"] for r in recs
            if r["rain_obs"] is not None and r["rain_fcst"] is not None]
    return max(offs) if offs else None


def _score_trend(models, temp_hourly, precip_score_daily, hourly_lead_days) -> dict:
    """各天桶（第 1..N 天）的温度分 / 降水分 / 综合分，用于"得分随时效衰减"趋势图。

    与分时效榜共用同一套桶得分（温度=逐小时轨道、降水=按天累计轨道）——
    榜单与趋势图永不分叉。
    """
    trend = {"overall": {}, "temp": {}, "precip": {}}
    for m in models:
        for key in trend:
            trend[key][m] = {}
        for b in range(1, hourly_lead_days + 1):
            bk = f"{b}d"
            t = temp_hourly[m].get(bk, {})
            p = precip_score_daily[m].get(bk, {})
            trend["temp"][m][bk] = temp_score(t)
            trend["precip"][m][bk] = precip_score(p)
            trend["overall"][m][bk] = overall_score(t, p)
    return trend

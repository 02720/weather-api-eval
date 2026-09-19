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
- **总榜 = 难度对齐后的期望分**（2026-09-18 重构，取代"共同窗口/全窗口"双轨）：
  总榜要回答的问题只有一个——"谁家预报最准"。但各家能预报的天数长短不一，而
  预报难度随时效单调上升：把自己覆盖到的天桶直接平均，短覆盖的源白拿简单桶的
  分；只取所有源共同覆盖的交集，又要把九成样本扔掉（实测 26 源同榜时窗口只剩
  4 天）。正解是**双向加法劈分**（核心实现见 stats.two_way_adjust）：把每个
  （源, 天桶）格子看成 S(m,b) = 技巧_m + 难度_b + 噪声，用全部格子联合估计
  行效应与列效应，再回答"若各家都被验证在同一批难度上谁排前面"。不扔数据、
  不要求覆盖一致，也从根上消掉了"每多覆盖 1 个天桶平均扣 0.74 分"的偏置。
  **矩阵只收温度与降水两维齐备的桶**——单维分不是综合分（此前 MSN 的 8 个桶
  里有 2 个缺降水维，却按纯温度分 88 分进了平均）；同时要求每个格子至少
  MIN_MODELS_PER_BUCKET 家同台（同台家数不足时该档被剔除；全体只有 2 家时阈值
  自动降到 2；只剩 1 家则无从比较，退回各源可用桶的等权平均），因为"这一档有
  多难"与"这一家有多强"在少于 2 家的档上根本分不开。
  代价要说清楚：这是**加法假设**，源的相对强弱若随天桶系统性变化（源×时效交互），
  总榜的单一数字表达不了——这类差异见分时效榜。
- **不确定性入榜**：按天分块 bootstrap 给出名次依据分数的 90% 置信区间、
  冠军频率、与第一名的显著性（经 Holm–Bonferroni 多重比较校正）；权重 ±40%
  扰动的冠军分布进报告，且同样走同一张劈分设计。实现见 stats.py。
  **块长不再是固定 1 天**：由日尺度误差的去相关时间决定（ρ≈0.46 → 2~3 天），
  受"块数 ≥ 6"约束——块长 1 天只捕获日内相关，实测把置信区间算窄 40%~90%（P1-1）。

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
  * 总榜（"all"）：**难度对齐综合分**——用双向加法模型把"天桶难度"与"源技巧"
    劈开后给出的期望分（详见上方"评分轨道"），回答"谁家预报最准"这一个问
    题。行附 90% 置信区间、冠军频率、与第一名显著性、n_eff 门槛
    （min_board_neff）达标标记（未达标 = 样本积累中，不参与冠军竞争）与覆盖
    时效（按各指标实际有效样本）。
  两层共用同一行结构与打分公式，主报告表格排行榜（总榜 + 可切时效）与冠军横幅共用这份数据。
"""
from __future__ import annotations

import logging
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
    difficulty_adjusted,
    effective_n,
    r_slope_numpy,
    temp_core_numpy,
    two_way_adjust,
    weight_champion_distribution,
)
from .timeutil import parse_iso, hour_bucket_days, floor_to_hour
from .snapshot_meta import (ISSUE_SOURCE_LABELS, SUSPECT_ISSUE_SOURCES,
                             integrity_summary, snapshot_complete)
from .storage import load_obs, list_forecast_snapshots

# 逐小时降水分级：cyeva 1h 雨强区间级别（小雨 0.1~1.9 … 大暴雨 ≥20 mm/h）
HOURLY_GRADED_LEVS = ("1", "2", "3", "4", "5")
# 按天降水分级：cyeva 24h 累计级别（+1=≥0.1 … +6=≥250mm）
DAILY_GRADED_LEVS = ("+1", "+2", "+3", "+4", "+5", "+6")
# 每级计算的分级指标（cyeva 分级全套）
GRADED_KEYS = ("acc", "pod", "far", "miss", "ts", "ets", "bias")

# bootstrap 固定种子：同一天数据必须得到同一份置信区间（可复现性）
BOOTSTRAP_SEED = 20260906

logger = logging.getLogger(__name__)


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


def _snaps_for(sid: str, model: str, snapshots: dict | None,
               require_complete: bool = True) -> list[dict]:
    """取该 (站, 源) 的快照列表，并按完整性门槛过滤。"""
    snaps = (snapshots[(sid, model)] if snapshots is not None
             and (sid, model) in snapshots
             else list_forecast_snapshots(sid, model))
    if not require_complete:
        return snaps
    return [s for s in snaps if snapshot_complete(s)]


def collect(station_ids: list[str], models: list[str], start_dt, end_dt,
            hourly_lead_days: int, daily_max_offset_days: int,
            daily_min_hours: int = 20,
            daily_source_fallback: bool = True, *,
            obs_maps: dict[str, dict] | None = None,
            snapshots: dict[tuple[str, str], list[dict]] | None = None,
            require_complete: bool = True
            ) -> tuple[list[dict], list[dict]]:
    """返回 (hourly_records, daily_records)。

    daily_source_fallback：允许用快照自带的逐日预报块为按天评估补位（默认开）。
    关掉后按天轨道与补位前完全一致，用于口径对照/回归。

    require_complete：排除契约标了 complete=false 的残缺快照（P0-6/6.4）。残缺快照
    的"实际时效/完整性"藏在日志里、外壳看起来却是正常的，让它们进榜等于把"这家
    少抓了一半分片"混进"这家预报得准不准"。旧存档无该字段 → 按完整处理。

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
            for snap in _snaps_for(sid, model, snapshots, require_complete):
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
                        # ---- 天桶按**日历时窗**对齐（对抗式审查 P0-3）----
                        # 旧口径用 (lead−1)//24+1 分桶：这是"起报后的第几个滚动 24h"，
                        # 与降水侧（有效日 − 起报日）不是同一个时间窗。后果是同一个
                        # "提前 1 天"里，Open-Meteo 的温度分 96% 来自**起报当天**、
                        # 风清的 72.5% 来自起报次日，而降水分对所有源都严格是"起报
                        # 次日全天"——两维日历构成不同却平均成同一个数字。
                        # 现在温度也用"有效时刻所属自然日 − 起报自然日"：天桶 N 对
                        # 温度与降水都指"起报日之后第 N 个自然日"，两维真正同窗。
                        #
                        # bucket=0（起报当日）是一条**哨兵**：它不进任何天桶榜
                        # （build_day_stat_tables 与各天桶循环都只取 1..N），但记录
                        # 仍然保留，供"固定时效窗"诊断口径（scorecard 的 24h/72h 池、
                        # 逐时效曲线）使用——那是另一套合法且已披露的口径，不该被
                        # 天桶对齐连带删掉，否则等于用一次口径修正把诊断视图掏空。
                        days_off = (vt.date() - issue.date()).days
                        if days_off < 0 or days_off > hourly_lead_days:
                            continue
                        rec = obs_map.get(tstr)
                        if rec is None:
                            continue
                        # 数组越界按缺测处理（与按天聚合同防护）：畸形存档降级为
                        # 该点缺测，不拖垮整份报告
                        hourly_records.append({
                            "station": sid, "model": m, "valid_iso": tstr,
                            "issue_iso": snap["issue_iso"],
                            "lead": lead, "bucket": days_off,
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
            for snap in _snaps_for(sid, model, snapshots, require_complete):
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
                            "issue_iso": snap["issue_iso"],
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
            # 收窄到具体异常类型（P2-11）：宽泛 except 会把真正的编程错误
            # （NameError/TypeError）当成"该级别不存在"静默吞掉，只剩一个 None
            except (ValueError, KeyError, IndexError, ZeroDivisionError) as exc:
                logger.debug("分级指标 %s/%s 计算失败：%s", kind, lev, exc)
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
    """温度误差序列的有效样本量（按站独立估计后求和，再扣掉跨站相关）。

    同一有效时刻会被多个起报覆盖（评估把每个 (起报, 有效时刻) 记为一条样本，
    这是指标口径）；自相关必须沿"时刻"单序列估计，故同一有效时刻只保留
    最新一版起报（最小 lead）的误差——同一预报值重复出现会人为抬高 ρ₁。

    同时把各站的**时刻键**一并传入：`n_eff_from_station_series` 要靠公共时刻
    估计跨站相关 ρ̄，再按 1+(k−1)ρ̄ 折算。只做时间维自相关校正（旧口径）会把
    4 个由同一批天气系统驱动的站当成 4 份独立信息，n_eff 高估约 1.7 倍（P0-4）。
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
    times = {sid: [vt for vt, _x in sorted(d.items())] for sid, d in by.items()}
    return _stats.n_eff_from_station_series(series, times)


def _n_eff_rain(recs: list[dict], thr: float, time_key: str,
                lead_key: str) -> int | None:
    """降水"报错"序列的有效样本量：晴雨判定不一致的 0/1 指示序列（按站）。

    与温度侧同样做跨站相关校正（P0-4）——相邻站"今天下没下雨"的判定高度同步，
    各站 n_eff 直接相加同样会高估。
    """
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
    times = {sid: [t for t, _x in sorted(d.items())] for sid, d in by.items()}
    return _stats.n_eff_from_station_series(series, times)


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
    # 降水维的入围门槛单独设：n_eff_rain 数的是"独立雨日判定"（≈ 站×天），
    # 与温度侧的"独立误差样本"不是一个量级，套用同一个 30 会把只差一点的源
    # 一刀切掉（实测 MSN 28 个雨日判定 vs 全员 48 个）。默认 20 ≈ 5 天×4 站。
    min_board_neff_rain = int(eval_cfg.get("min_board_neff_rain", 20))
    bootstrap_runs = int(eval_cfg.get("bootstrap_runs", 500))
    sensitivity_runs = int(eval_cfg.get("sensitivity_runs", 500))
    # bootstrap 块长（天）：0/缺省 = 按日尺度误差自相关自动定（见 stats.resolve_block_days）
    bootstrap_block_days = int(eval_cfg.get("bootstrap_block_days", 0) or 0)
    hourly_lead_days = eval_cfg["hourly_lead_days"]
    daily_max_offset = eval_cfg["daily_max_offset_days"]
    daily_min_hours = eval_cfg.get("daily_min_hours", 20)
    daily_source_fallback = bool(eval_cfg.get("daily_source_fallback", True))
    # ---- 对抗式审查 P0-1 / P0-2 / P1-2 / P1-4 / P1-5 的口径开关 ----
    cell_weighting = str(eval_cfg.get("board_cell_weighting", "neff"))
    min_cell_neff = float(eval_cfg.get("min_cell_neff", 15))
    min_col_frac = float(eval_cfg.get("board_min_col_frac", 0.5))
    ridge = float(eval_cfg.get("board_ridge", 0.0))
    long_tail = bool(eval_cfg.get("board_long_tail_board", True))
    macro_w = tuple(eval_cfg.get("macro_weight_range", (0.30, 0.70)))
    require_complete = bool(eval_cfg.get("require_complete_snapshots", True))

    obs_maps, snapshots = _preload(station_ids, models)
    excluded_incomplete = _count_incomplete(snapshots)
    hourly, daily = collect(station_ids, models, start_dt, end_dt,
                            hourly_lead_days, daily_max_offset, daily_min_hours,
                            daily_source_fallback,
                            obs_maps=obs_maps, snapshots=snapshots,
                            require_complete=require_complete)

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
        # 按天轨道的桶上限是 daily_max_offset_days，不是逐小时的 hourly_lead_days——
        # 两者当前同为 16，但语义不同（逐小时时效上限 vs 按天日偏移上限），
        # 混用会让"按天能评到哪一天"被逐小时配置悄悄截断（P2-10）
        for off in range(1, daily_max_offset + 1):
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
    # 起报锚点语义与争议口径（P0-6）：每源的 issue_source + 争议项
    model_issue = _issue_anchor_meta(models, snapshots, require_complete=require_complete)
    # ---- 数据充分性披露（P2-1/P2-2/P2-3）----
    model_status = _model_status(models, snapshots, station_ids)
    snapshot_quality = _snapshot_quality(models, snapshots)

    # ---- 排行榜：分时效榜（每个天桶一份）+ 总榜（难度对齐 + 不确定性）----
    # 表格排行榜、冠军横幅与趋势图共用同一套桶得分。
    # 块长（P1-1）：块长 1 天只捕获日内相关，块间相关被当成独立 ⇒ CI 偏窄。
    # 由日尺度温度误差的 lag-1 自相关推出去相关时间，再受"块数下限"约束。
    n_boot_days = len(_stats.eval_days(hourly, daily))
    boot_rho = _stats.daily_error_lag1_rho(hourly)
    block_days = _stats.resolve_block_days(n_boot_days, bootstrap_block_days, boot_rho)
    leaderboards = _lead_leaderboards(models, temp_hourly, precip_score_daily,
                                      hourly_lead_days)
    difficulty_window: dict = {}
    leaderboards["all"] = _overall_board(
        models, temp_hourly, precip_score_daily, hourly_lead_days,
        by_model, daily_by_model,
        n_eff_all={m: _n_eff_temp([r for r in by_model[m]
                                   if r.get("bucket", 0) >= 1]) for m in models},
        min_sample=min_sample, min_board_neff=min_board_neff,
        thr_daily=thr_daily, bootstrap_runs=bootstrap_runs,
        block_days=block_days, min_board_neff_rain=min_board_neff_rain,
        window_out=difficulty_window,
        cell_weighting=cell_weighting, min_cell_neff=min_cell_neff,
        min_col_frac=min_col_frac, ridge=ridge, long_tail_board=long_tail,
        macro_weight_range=macro_w, model_issue=model_issue)
    # 日历跨度（日历维度）：入围源各自的验证日数差异有多大
    qdays = [r["n_days"] for r in leaderboards["all"]
             if r.get("qualified") and r.get("n_days")]
    calendar_span = {
        "min_days": min(qdays) if qdays else 0,
        "max_days": max(qdays) if qdays else 0,
    }

    qualified_models = {r["model"] for r in leaderboards["all"] if r.get("qualified")}
    # 权重敏感性：与总榜同走一张劈分设计——回答的必须是"同一个估计量"的稳定性。
    # 宏观权重（温度:降水）也纳入扰动（P1-2）。
    weight_sensitivity = {
        "runs": sensitivity_runs,
        "macro_weight_range": [float(macro_w[0]), float(macro_w[1])],
        "champions": _weight_sensitivity_champions(
            models, temp_hourly, precip_score_daily, hourly_lead_days,
            sensitivity_runs, qualified_models,
            adj_row=np.array(difficulty_window.get("row_keep") or [], dtype=bool),
            adj_col=np.array(difficulty_window.get("col_keep") or [], dtype=bool),
            adj_w=_board_weight_matrix(models, temp_hourly, precip_score_daily,
                                       hourly_lead_days, cell_weighting),
            adj_ridge=ridge, macro_weight_range=macro_w),
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
            "min_board_neff_rain": min_board_neff_rain,
            "bootstrap_runs": bootstrap_runs,
            # 不确定性口径披露：块长与块数决定 CI 的宽窄（块长越短越偏窄）
            "bootstrap_block_days": block_days,
            "bootstrap_blocks": (n_boot_days // block_days) if n_boot_days else 0,
            "bootstrap_days": n_boot_days,
            "bootstrap_daily_rho": (round(boot_rho, 3) if boot_rho is not None else None),
            # 天桶难度的劈分设计与各桶难度值：总榜公平性的可核对凭据
            "difficulty_window": difficulty_window,
            # 日历跨度：入围源各自被验证的自然日数区间（差异大 = 时期效应风险）
            "calendar_span": calendar_span,
            "daily_min_hours": daily_min_hours,
            "daily_source_fallback": daily_source_fallback,
            "model_caveats": model_caveats,
            # 数据充分性披露（P2-1/P2-2/P2-3）
            "model_status": model_status,
            "snapshot_quality": snapshot_quality,
            # ---- 可审计性（§7.1）：本轮存档的哈希链根与抓取时刻范围 ----
            # 任何人拿仓库里的快照文件重算一遍，就能验证"这批预报确实在实况之前
            # 封存、且事后未被修改"——把项目最大的护城河从自律升级为可验证。
            "integrity": integrity_summary(
                [s for lst in snapshots.values() for s in lst]),
            # 完整性门槛排除的快照数（P0-6）：残缺快照不再与完整快照同权进榜
            "excluded_incomplete": excluded_incomplete,
            "require_complete_snapshots": require_complete,
            # ---- 起报锚点语义（P0-6）：每源 declaration + 争议理由 ----
            "issue_anchors": model_issue,
            # 总榜拟合口径（P0-1/P1-4）：加权方式、门槛、收缩强度
            "board_cell_weighting": cell_weighting,
            "min_cell_neff": min_cell_neff,
            "board_min_col_frac": min_col_frac,
            "board_ridge": ridge,
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


def _board_weight_matrix(models, temp_hourly, precip_score_daily,
                         hourly_lead_days: int, weighting: str) -> np.ndarray | None:
    """总榜格子权重矩阵（供总榜与权重敏感性共用同一把尺子）。"""
    if weighting != "neff":
        return None
    ne_t = _cell_neff_matrix(models, temp_hourly, precip_score_daily,
                             hourly_lead_days, "temp")
    ne_r = _cell_neff_matrix(models, temp_hourly, precip_score_daily,
                             hourly_lead_days, "rain")
    return _board_cell_weights(ne_t, ne_r, weighting)[0]


def _count_incomplete(snapshots: dict) -> dict[str, int]:
    """按源统计被完整性门槛排除的快照数（披露用，P0-6）。"""
    out: dict[str, int] = defaultdict(int)
    for (_sid, m), snaps in (snapshots or {}).items():
        for s in snaps:
            if not snapshot_complete(s):
                out[m] += 1
    return dict(out)


# 起报锚点语义（P0-6 / §6.1）：五种互不等价的物理语义，却在同一张榜上比较。
# 词汇表本体在 snapshot_meta（契约层），此处只做引用，避免两处词表漂移。
_SUSPECT_ISSUE_SOURCES = SUSPECT_ISSUE_SOURCES


def _issue_anchor_meta(models: list[str], snapshots: dict,
                       require_complete: bool = True) -> dict[str, dict]:
    """每源的起报锚点语义与争议口径清单（P0-6）。

    快照的 issue_source 由各 provider 按契约声明（见 forecast/base.py）。缺失
    （历史存档）一律记 "unknown" 而不是猜一个——"不知道"本身就是要披露的事实。

    争议项（disputed）判定：**只认正面证据**，不把"历史存档没这个字段"当成争议。
    旧存档（2026-09 之前）没有 issue_source，若一律记 "unknown" 并判争议，则全榜
    27 家会被同时标红——标记一旦对所有人都亮，就等于没有标记。
      1. 该源**显式声明**的锚点语义属于"≈抓取时刻"或"未知"（suspect 集合）；
      2. 快照存在残缺（complete=false）或携带 missing_shards；
      3. 存在空间吸附（location_distance_km）——样本代表城市而非站点；
      4. 温度被量化（quantized_temp）——整数摄氏度自带 ±0.5°C 量化误差。
    只声明了轴首点/模式轮次的源不算争议（它们只是口径不同，不是被评在更容易的
    样本上）。未声明的情形单独进 issue_source_note，由页面作为"可审计性进度"披露。
    """
    out: dict[str, dict] = {}
    for m in models:
        sources: dict[str, int] = defaultdict(int)
        reasons: list[str] = []
        quantized = False
        partial = 0
        adsorption_km: list[float] = []
        declared = 0
        total_seen = 0
        for (sid, smodel), snaps in (snapshots or {}).items():
            if smodel != m:
                continue
            for s in snaps:
                total_seen += 1
                if require_complete and not snapshot_complete(s):
                    partial += 1
                    continue
                # "声明过"= provider 真的给出了锚点语义（见 snapshot_meta 的
                # issue_source_declared）。历史存档没有它——这正是"锚点语义未知"
                # 与"provider 明确说不知道"的区别所在。
                if s.get("issue_source_declared"):
                    declared += 1
                    sources[str(s.get("issue_source") or "unknown")] += 1
                if s.get("quantized_temp"):
                    quantized = True
                if s.get("missing_shards"):
                    partial += 1
                d = s.get("location_distance_km")
                if d is not None:
                    try:
                        adsorption_km.append(float(d))
                    except (TypeError, ValueError):
                        pass
        if not sources:
            out[m] = {"issue_source": "unknown",
                      "issue_source_label": ISSUE_SOURCE_LABELS["unknown"],
                      "issue_source_undeclared": total_seen,
                      "issue_source_note": ("历史存档早于元数据契约，起报锚点语义未声明"
                                            if declared == 0 else None),
                      "disputed": False, "disputed_reasons": []}
            continue
        # 主导语义 = 出现最多的那一种（多语义混合时如实列出）
        dom = max(sources, key=lambda k: sources[k])
        mixed = len(sources) > 1
        if dom in _SUSPECT_ISSUE_SOURCES:
            reasons.append(f"起报锚点语义为「{ISSUE_SOURCE_LABELS[dom]}」，"
                           "声明时效可能系统性长于实际时效")
        if partial:
            reasons.append(f"{partial} 份快照残缺（分片未取全）")
        if adsorption_km:
            avg = sum(adsorption_km) / len(adsorption_km)
            reasons.append(f"最近城市吸附（平均 {avg:.1f} km，代表城市而非站点）")
        if quantized:
            reasons.append("温度被量化到整数摄氏度（自带 ±0.5°C 量化误差）")
        if mixed:
            reasons.append("同一源混用多种起报锚点语义：" + "、".join(
                f"{ISSUE_SOURCE_LABELS.get(k, k)}×{v}" for k, v in sorted(sources.items())))
        out[m] = {
            "issue_source": dom if not mixed else "mixed",
            "issue_source_label": ISSUE_SOURCE_LABELS.get(dom, dom),
            "issue_source_counts": {k: int(v) for k, v in sorted(sources.items())},
            "issue_source_declared": declared,
            # 未声明份数：可审计性覆盖率的直接指标（随归档自然递减，不判争议）
            "issue_source_undeclared": max(0, total_seen - declared),
            "issue_source_note": ("全部快照早于元数据契约，起报锚点语义未声明"
                                  if declared == 0 else
                                  ("仍有 %d 份快照未声明锚点语义"
                                   % (total_seen - declared) if total_seen > declared
                                   else None)),
            "disputed": bool(reasons),
            "disputed_reasons": reasons,
        }
    return out


def _weight_sensitivity_champions(models, temp_hourly, precip_score_daily,
                                  hourly_lead_days, runs,
                                  qualified: set[str],
                                  adj_row: np.ndarray | None = None,
                                  adj_col: np.ndarray | None = None,
                                  adj_w: np.ndarray | None = None,
                                  adj_ridge: float = 0.0,
                                  macro_weight_range: tuple = (0.30, 0.70)) -> list[dict]:
    """权重 ±40% 扰动下的冠军分布（P0-2.2）：桶得分重组冠军计 500 次。

    指标值不随权重变化，只需把**已换算并截断**的子分张量按扰动权重重组——
    向量化实现（stats.weight_champion_distribution），成本可忽略。
    qualified：总榜入围者（n_eff 门槛 + 维度齐备）；敏感性回答的是
    "入围者之间的冠军之争对权重有多敏感"，未入围者不在轮换范围内。
    macro_weight_range：温度:降水宏观权重的扰动区间（P1-2）——旧实现恒为 50:50，
    等于把最有争议的那个权重排除在敏感性分析之外。
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
                                        PRECIP_SCORE_PARTS, runs=runs,
                                        adj_row=adj_row, adj_col=adj_col,
                                        adj_w=adj_w, adj_ridge=adj_ridge,
                                        macro_weight_range=tuple(macro_weight_range))
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


def _rank_rows(rows: list[dict], keys: tuple[str, ...] = ("score",)) -> list[dict]:
    """综合分降序、None 沉底；未达 n_eff 门槛的源（qualified=False）排在其后、
    仍按分数序——榜单语义是"达标者先竞技，未达标者列样本积累中"（P0-2.3）。

    keys：依次降级的排序键——取第一个非 None 的键决定名次，全 None 者沉底。
    """
    def key(r):
        q = r.get("qualified", True)
        for k in keys:
            s = r.get(k)
            if s is not None:
                return (0, not q, -s)
        return (1, not q, 0.0)
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


def _metric_matrix(models: list[str], hourly_lead_days: int, value_of) -> np.ndarray:
    """(m, b) 指标矩阵：value_of(model, "Nd") → 数值，None 一律记 NaN。"""
    S = np.full((len(models), hourly_lead_days), np.nan)
    for i, m in enumerate(models):
        for b in range(1, hourly_lead_days + 1):
            v = value_of(m, f"{b}d")
            if v is not None:
                S[i, b - 1] = float(v)
    return S


def _composite_cells(temp_hourly, precip_score_daily, m, bk):
    """该（模型, 天桶）的 (综合分, 温度分, 降水分)；缺任一维则全 None——不进总榜。"""
    t = temp_hourly[m].get(bk) or {}
    p = precip_score_daily[m].get(bk) or {}
    ts, ps = temp_score(t), precip_score(p)
    if ts is None or ps is None:
        return (None, None, None)
    return (overall_score(t, p), ts, ps)


def _fin(x, digits: int = 3):
    """NaN/inf/None → None（在页面上显示 —），有限值按位数取整。"""
    if x is None:
        return None
    x = float(x)
    return round(x, digits) if np.isfinite(x) else None


def _cell_neff_matrix(models, temp_hourly, precip_score_daily,
                      hourly_lead_days, kind: str) -> np.ndarray:
    """(m, b) 的格子有效样本量矩阵（kind="temp"|"rain"）。

    阈值/加权的依据必须是**有效**样本量而不是名义 n：同一批天气系统在 4 个站上
    制造的是同一份信息（跨站 ρ̄≈0.23），名义 n 会把 1 份信息数成 4 份（P0-4）。
    """
    M = np.full((len(models), hourly_lead_days), np.nan)
    src = temp_hourly if kind == "temp" else precip_score_daily
    for i, m in enumerate(models):
        for b in range(1, hourly_lead_days + 1):
            md = src[m].get(f"{b}d") or {}
            v = md.get("n_eff")
            if v is None:
                v = md.get("n")
            if v is not None:
                M[i, b - 1] = float(v)
    return M


def _board_cell_weights(neff_t: np.ndarray, neff_r: np.ndarray,
                        weighting: str,
                        min_cell_neff: float = 0.0) -> tuple[np.ndarray | None, float]:
    """格子权重矩阵（P0-1）：√(两维有效样本量的较小者)，以及**权重单位下的门槛**。

    为什么取"两维的较小者"：综合分承诺温度/降水各半，一格能不能说话取决于**更薄的
    那一维**——温度有 2,360 个有效样本、降水只有 5 个时，这个格子的综合分实质上
    是"5 个样本的降水分 + 大量温度分"的混合物。
    为什么取平方根而不是原始样本量：格子样本量跨三个数量级（5~2,360），直接用
    原始量会让最厚的格子垄断整个拟合（权重占比 >90%），√ 把它压到 8 倍量级，
    既保留"信息多的格子更可信"的方向，又不至于让长尾格子集体失声。

    返回值第二项是"把 min_cell_neff 换算到权重单位"后的门槛——**必须换算**：
    门槛与权重量纲不同时（拿 √n_eff 去比 n_eff 门槛），门槛会要么形同虚设、要么
    把所有格子一次清空，而两种失败都表现为"设计被降级"，不报错。
    返回 (None, 0.0) 表示等权口径（旧行为，用于对照披露）。
    """
    if weighting != "neff":
        return None, 0.0
    with np.errstate(invalid="ignore"):
        n = np.minimum(np.asarray(neff_t, dtype=float), np.asarray(neff_r, dtype=float))
    n = np.where(np.isfinite(n) & (n > 0), n, 0.0)
    return np.sqrt(n), float(np.sqrt(max(0.0, float(min_cell_neff))))


def _segment_board(S: np.ndarray, W: np.ndarray | None, cols: list[int],
                   min_cell_neff: float, min_col_frac: float = 0.0) -> dict[int, float]:
    """在给定列子集上单独跑一次双向劈分，返回 {模型下标: 难度对齐行分}。

    用于"技巧剖面"（短/中/长时效各一段，P0-2 建议 2）：加法模型把源×时效交互塞进
    残差，一个数字表达不了"短时效强、长时效弱"；分段各自对齐后，读者看到的是一张
    剖面而不是一个被平均掉的数。
    """
    if not cols:
        return {}
    sub = S[:, cols]
    w = W[:, cols] if W is not None else None
    n_with = int(np.isfinite(sub).any(axis=1).sum())
    if n_with < 2:
        return {i: (float(sub[i, np.isfinite(sub[i])].mean())
                    if np.isfinite(sub[i]).any() else np.nan) for i in range(sub.shape[0])}
    min_col = max(2, min(_stats.MIN_MODELS_PER_BUCKET, n_with))
    adj = two_way_adjust(sub, min_col=min_col, min_row=1, weights=w,
                         min_cell_weight=(min_cell_neff if w is not None else 0.0),
                         min_col_frac=min_col_frac)
    return {i: float(adj["scores"][i]) for i in range(sub.shape[0])}


def _overall_board(models, temp_hourly, precip_score_daily, hourly_lead_days,
                   by_model, daily_by_model, n_eff_all, min_sample, min_board_neff,
                   thr_daily, bootstrap_runs, block_days: int = 1,
                   min_board_neff_rain: int = 20,
                   window_out: dict | None = None,
                   cell_weighting: str = "neff", min_cell_neff: float = 15.0,
                   min_col_frac: float = 0.5, ridge: float = 0.0,
                   long_tail_board: bool = True,
                   macro_weight_range: tuple[float, float] = (0.30, 0.70),
                   model_issue: dict[str, dict] | None = None) -> list[dict]:
    """总榜：**难度对齐综合分**——各家覆盖天数不同时唯一公平的答法。

    公平性（第一性原理）：预报难度随时效单调上升，而各家能预报的天数长短不一。
    把自己覆盖到的天桶直接平均，"只报近几天"的源白拿简单桶的分（实测全窗口分
    与覆盖桶数 r = −0.54、每多覆盖 1 个天桶平均扣 0.74 分）；只取所有源共同覆盖
    的交集，26 源同榜时窗口只剩 4 天——九成样本被扔掉。本实现走第三条路：把每个
    （源, 天桶）格子看成

        S(m, b) = 技巧_m + 难度_b + 噪声

    用**全部**格子联合估计行效应与列效应（双向加法模型 / 交替最小二乘，见
    stats.two_way_adjust），再回答"若各家都被验证在同一批难度上，谁排前面"。
    既不扔数据，也不要求各家覆盖一致。

    榜单所有数字列（综合/温度/降水/±2°C/RMSE/TS/ETS）走**同一张行列设计**做
    难度对齐——读者按任何一列排序，看到的都是同一批格子上的同类比较。

    不确定性：每行附 90% 置信区间（按天分块 bootstrap）、冠军频率、与第一名
    显著性；n_eff < min_board_neff 的源 qualified=False，排在其后显示"样本
    积累中"，不参与冠军竞争（156 条样本争冠军的教训）。
    """
    rows: list[dict] = []
    # ---- 1. 组装"综合分 × 天桶"矩阵，劈分出行列设计与天桶难度 ----
    # 两维齐备的桶才进矩阵。旧实现把"只有温度分"的单维桶也算进平均——那不是综合
    # 分，是"温度分换个名字"（实测 MSN 的 8 个桶里有 2 个缺降水维，却按纯温度分
    # 88 分进了平均）。综合分承诺温度/降水各半，缺一维的桶不可用。
    S = _metric_matrix(
        models, hourly_lead_days,
        lambda m, bk: _composite_cells(temp_hourly, precip_score_daily, m, bk)[0])
    # 格子有效样本量（加权与门槛的依据，P0-1/P0-4）
    NE_t = _cell_neff_matrix(models, temp_hourly, precip_score_daily,
                             hourly_lead_days, "temp")
    NE_r = _cell_neff_matrix(models, temp_hourly, precip_score_daily,
                             hourly_lead_days, "rain")
    W_cell, cell_min_w = _board_cell_weights(NE_t, NE_r, cell_weighting,
                                             min_cell_neff)
    # 同台门槛随"实际有几家"自适应：只有 2 家时要求 ≥3 家同台会把两家一起踢出局
    # （2 家同桶已经足够分离行效应与列效应）；低于 2 家则无从比较，走下面的降级。
    n_with_data = int(np.isfinite(S).any(axis=1).sum())
    min_col = max(2, min(_stats.MIN_MODELS_PER_BUCKET, n_with_data)) if n_with_data else 1
    adj = two_way_adjust(S, min_col=min_col, min_row=1, weights=W_cell,
                         min_cell_weight=cell_min_w,
                         min_col_frac=min_col_frac, ridge=ridge)
    row_keep, col_keep = adj["row_keep"], adj["col_keep"]
    # 门槛把设计清空 = "门槛与当前数据规模不匹配"，不是"没有可比较的源"。
    # 此时放宽门槛，但**如实标记**（gate_relaxed 进披露），绝不让页面看起来像
    # 用了门槛却其实没有——静默放宽等于把 P0-1 的修复又退回去。
    gate_relaxed = False
    if cell_min_w > 0 and not (row_keep.any() and col_keep.any()):
        adj = two_way_adjust(S, min_col=min_col, min_row=1, weights=W_cell,
                             min_cell_weight=0.0,
                             min_col_frac=min_col_frac, ridge=ridge)
        row_keep, col_keep = adj["row_keep"], adj["col_keep"]
        gate_relaxed = True
    degraded = not (row_keep.any() and col_keep.any())
    if degraded:
        # 降级：可用格子凑不出一张能比较的表（典型是只接了一个源）。此时"难度"
        # 无从估计，各源自家可用天桶的等权平均就是唯一诚实的数字——劈分在此
        # 退化情形下的解正好等于它（单行的行效应 = 该行的桶平均分）。
        row_keep = np.isfinite(S).any(axis=1)
        col_keep = np.isfinite(S).any(axis=0)
        adj = two_way_adjust(S, min_col=1, min_row=1, weights=W_cell,
                             min_cell_weight=0.0,
                             min_col_frac=0.0, ridge=ridge)
    # 唯一的格子掩膜：综合分与所有派生列**必须**逐格同集（P1-3）
    cell_valid = np.asarray(adj["cell_valid"], dtype=bool)
    scores = np.asarray(adj["scores"], dtype=float)
    # 等权对照（同一张设计、同一批格子，只把权重换成 1）——用于披露"名次对加权
    # 方式有多敏感"。这不是"另一个榜单"，而是同一估计量的一个反事实。
    scores_equal = difficulty_adjusted(S[None, ...], row_keep, col_keep,
                                       weights=None, valid=cell_valid[None, ...],
                                       ridge=ridge)[0]
    sens = _stats.rank_sensitivity(scores_equal, scores)
    # 名次变动者带上模型名：stats 层不认识模型名，页面/Tab 需要直接可读
    for mv in sens.get("movers", []):
        mv["model"] = models[mv["index"]]
    if cell_weighting == "neff":
        sens["primary"] = "weighted"
        sens["spearman_label"] = "按样本量加权 vs 等权"
    else:
        sens["primary"] = "equal"
        sens["spearman_label"] = "等权 vs 按样本量加权"

    def _aligned(matrix: np.ndarray) -> np.ndarray:
        """用总榜同一张设计（同一批格子、同一组权重）归总成"难度对齐后的一个数"。

        valid=cell_valid 是 P1-3 的核心：派生列（±2°C / RMSE / TS / ETS）也要
        逐格同集，否则某源某桶"因降水缺而没有综合分"却"因 acc2 有值而参与难度
        对齐"，页面上的两列就来自两批不同的格子——"所有数字列走同一张设计"
        此前只是 row/col 掩码层面的半真话。
        """
        return difficulty_adjusted(matrix[None, ...], row_keep, col_keep,
                                   weights=(W_cell[None, ...] if W_cell is not None else None),
                                   ridge=ridge, valid=cell_valid[None, ...])[0]

    matrices = {
        "temp_score": _metric_matrix(
            models, hourly_lead_days,
            lambda m, bk: _composite_cells(temp_hourly, precip_score_daily, m, bk)[1]),
        "precip_score": _metric_matrix(
            models, hourly_lead_days,
            lambda m, bk: _composite_cells(temp_hourly, precip_score_daily, m, bk)[2]),
        "acc2": _metric_matrix(models, hourly_lead_days,
                               lambda m, bk: (temp_hourly[m].get(bk) or {}).get("acc2")),
        "rmse": _metric_matrix(models, hourly_lead_days,
                               lambda m, bk: (temp_hourly[m].get(bk) or {}).get("rmse")),
        "ts": _metric_matrix(models, hourly_lead_days,
                             lambda m, bk: (precip_score_daily[m].get(bk) or {}).get("ts")),
        "ets": _metric_matrix(models, hourly_lead_days,
                              lambda m, bk: (precip_score_daily[m].get(bk) or {}).get("ets")),
    }
    aligned = {k: _aligned(v) for k, v in matrices.items()}
    # 技巧剖面（P0-2）：短 1-3d / 中 4-7d / 长 8d+，各段独立对齐
    segments = {"short": [b - 1 for b in range(1, min(3, hourly_lead_days) + 1)],
                "mid": [b - 1 for b in range(4, min(7, hourly_lead_days) + 1)],
                "long": [b - 1 for b in range(8, hourly_lead_days + 1)]}
    profile = {}
    for name, cols in segments.items():
        if not cols:
            continue
        seg = _segment_board(S, W_cell, cols, cell_min_w)
        profile[name] = {
            "buckets": [c + 1 for c in cols],
            "scores": [(_fin(seg[i]) if (i in seg and seg[i] is not None
                                         and np.isfinite(seg[i])) else None)
                       for i in range(len(models))],
        }
    rank_in_segment = {}
    for name, seg in profile.items():
        order = sorted((i for i in range(len(models))
                        if seg["scores"][i] is not None),
                       key=lambda i: -seg["scores"][i])
        rank_in_segment[name] = {i: r + 1 for r, i in enumerate(order)}

    # ---- 2. 逐行：难度对齐后的各项数值 + 样本充分性披露 ----
    for i, m in enumerate(models):
        bs = [b for b in range(1, hourly_lead_days + 1) if np.isfinite(S[i, b - 1])]
        # 本行的样本/天数/时效只统计**进天桶评估**的样本（bucket ≥ 1）。
        # bucket=0 是起报当日（P0-3 的哨兵），它不进任何天桶，因此也不该被算进
        # "这一行有多少证据"——否则总榜行会用一批没参与打分的样本虚报样本量。
        # 全 lead 的配对数另存 n_all_leads 供明细口径核对。
        scored = [r for r in by_model[m] if r.get("bucket", 0) >= 1]
        # 降水维的有效样本量（各桶 n_eff 之和）：温度侧早有 n_eff 门槛，降水侧
        # 此前完全没设总榜级门槛——一个只攒到十几个独立雨日样本、却靠温度分达标
        # 的源也能上榜（实测 MSN 各桶降水的 n_eff 只有 16~28，低于温度侧门槛 30，
        # 却在综合分里占了半壁江山）。两个维度必须同门槛。
        neff_rain = _n_eff_rain(daily_by_model[m], thr_daily, "valid_day", "offset")
        neff = n_eff_all.get(m)
        score, t_score, p_score = _fin(scores[i], 2), _fin(aligned["temp_score"][i], 2), \
            _fin(aligned["precip_score"][i], 2)
        # 覆盖时效（P1-1）：按各指标实际参与计算的样本（两侧值同时非缺测）计算
        lead_days = _valid_lead_days(scored, "temp_obs", "temp_fcst")
        rain_days = _valid_rain_days(daily_by_model[m])
        n_temp = sum(1 for r in scored
                     if r["temp_obs"] is not None and r["temp_fcst"] is not None)
        n_all_leads = sum(1 for r in by_model[m]
                          if r["temp_obs"] is not None and r["temp_fcst"] is not None)
        n_rain = sum(1 for r in daily_by_model[m]
                     if r["rain_obs"] is not None and r["rain_fcst"] is not None)
        # 验证日数（日历维度）：样本量相同也可能只覆盖几天。新接入的源只在自己
        # 入榜之后的日子上被验证——若那几天的天气形势偏难/偏易，分数里就混进了
        # "时期效应"而非技巧（实测同一批源换到 8 天重算，分数变动最高 7.3 分）。
        # 这不是能靠门槛挡住的，只能显式披露。
        days_temp = {r["valid_iso"][:10] for r in scored
                     if r["temp_obs"] is not None and r["temp_fcst"] is not None}
        days_rain = {r["valid_day"] for r in daily_by_model[m]
                     if r["rain_obs"] is not None and r["rain_fcst"] is not None}
        row = _board_row(
            m,
            {"acc2": _fin(aligned["acc2"][i]), "rmse": _fin(aligned["rmse"][i], 3),
             "n": n_temp},
            {"ts": _fin(aligned["ts"][i], 3), "ets": _fin(aligned["ets"][i], 3),
             "n": n_rain},
            score=score, temp_score=t_score, precip_score=p_score,
            n=n_temp, n_precip=n_rain, n_eff=neff, n_eff_rain=neff_rain,
            # 全 lead 的配对数（含起报当日）：明细口径核对用，不参与名次
            n_all_leads=n_all_leads,
            # 达标条件（P0-2.3）：n_eff 门槛 + 温度/降水两个维度都有分 + 进得了
            # 劈分设计。"综合"承诺的是两维各半——只有温度维有分的源拿温度分与
            # 别人的 (温度+降水)/2 比不是同口径（实测 MSN 无按天降水样本却凭温度
            # 分坐上 95.9 的"综合冠军"），维度不齐 = 综合结论未到位 = 样本积累中。
            qualified=(score is not None and neff is not None and neff >= min_board_neff
                       and neff_rain is not None and neff_rain >= min_board_neff_rain
                       and t_score is not None and p_score is not None
                       and bool(row_keep[i])),
            lead_days=lead_days, rain_days=rain_days, n_buckets=len(bs),
            n_days=len(days_temp | days_rain), n_days_rain=len(days_rain),
            # 起报轮次数（P2-1）：1 个起报的 56 条样本与 17 个起报的 15,992 条，
            # 可信度不是一个量级——只披露覆盖天数会把它俩显示成同一档
            n_issues=len({r.get("issue_iso") for r in by_model[m] if r.get("issue_iso")}),
            # 未进入劈分设计的源（可用天桶太少/所在的桶同台家数不足）无法横向比较
            comparable=bool(row_keep[i]),
            # 技巧剖面名次（P0-2）：同一家在不同时效段的名次可以差很多，
            # 单一数字抹不掉这个事实，只能披露
            profile_rank={k: rank_in_segment.get(k, {}).get(i)
                          for k in ("short", "mid", "long")},
            # 起报锚点语义与争议口径（P0-6）：把"这家被评在什么口径上"写在行上
            **((model_issue or {}).get(m) or {}),
        )
        rows.append(row)
    # ---- 3. 把"难度对齐"这件事本身做成可读的披露信息 ----
    if window_out is not None:
        cols = adj["col_effects"]
        kept = [b for b in range(1, hourly_lead_days + 1) if col_keep[b - 1]]
        dropped_cols = [b for b in range(1, hourly_lead_days + 1)
                        if not col_keep[b - 1]
                        and bool(np.isfinite(S[:, b - 1]).any())]
        # 加法假设的量化（P0-2）：交互项占多少方差、跨桶名次一致性多高——
        # 这两个数字此前一个都没有，读者无从知道总榜的适用边界
        vdec = _stats.variance_decomposition(S, row_keep, col_keep, W_cell)
        stability = _stats.bucket_rank_stability(S, row_keep, col_keep)
        win_valid = W_cell[cell_valid] if W_cell is not None else None
        window_out.update({
            "buckets": kept,
            "days": max(kept) if kept else 0,
            "models": int(row_keep.sum()),
            "qualified": sum(1 for r in rows if r.get("qualified")),
            "n_components": adj["n_components"],
            # 降级标记：True = 可用格子凑不出可比较的设计，名次退回各源桶平均分
            "degraded": degraded,
            # 各天桶难度相对"窗口平均难度"的偏离（分）：−6.2 表示该桶全场平均分
            # 比窗口平均低 6.2 分——这正是"覆盖越长分数越低"的那部分，已被扣掉
            "difficulty": {b: (round(float(cols[b - 1]), 2)
                               if np.isfinite(cols[b - 1]) else None)
                           for b in range(1, hourly_lead_days + 1)},
            # 各桶真正参与比较的家数（只计入设计内的源）：不足阈值的桶不进比较
            "coverage": {b: int((np.isfinite(S[:, b - 1]) & row_keep).sum())
                         for b in range(1, hourly_lead_days + 1)},
            "min_models_per_bucket": int(_stats.MIN_MODELS_PER_BUCKET),
            # 设计本身（行=源 / 列=天桶 是否进入横向比较）：权重敏感性等派生
            # 分析必须用同一张设计，否则答的是另一个估计量
            "row_keep": [bool(v) for v in row_keep],
            # ---- P0-1 / P1-4 的口径披露 ----
            "cell_weighting": cell_weighting,
            "min_cell_neff": float(min_cell_neff),
            "min_col_frac": float(min_col_frac),
            "ridge": float(ridge),
            "dropped_thin_cells": int(adj.get("dropped_thin_cells", 0)),
            # 真正进入拟合的格子掩膜（P1-3 的"同一张设计"凭据，供读者/回归核对）
            "cell_valid": [[bool(v) for v in row] for row in np.asarray(cell_valid)],
            # 门槛是否因"与数据规模不匹配"被放宽（True = 页面上的名次没有真正
            # 用上下限门槛，读者必须知道）
            "gate_relaxed": bool(gate_relaxed),
            "excluded_long_tail_buckets": dropped_cols,
            "cell_weights": (None if win_valid is None or win_valid.size == 0 else {
                "min": round(float(np.min(win_valid)), 2),
                "median": round(float(np.median(win_valid)), 2),
                "max": round(float(np.max(win_valid)), 2),
            }),
            # ---- P0-2 的口径披露：加法假设解释不了多少 ----
            "variance": vdec,
            "rank_stability": stability,
            # ---- P0-1 的披露：名次对加权方式的敏感度 ----
            "rank_sensitivity": sens,
            "equal_weight_scores": {m: _fin(scores_equal[i], 2)
                                    for i, m in enumerate(models)},
            # ---- 技巧剖面（短/中/长时效各自的名次）----
            "profile": {k: {"buckets": v["buckets"],
                            "scores": {m: v["scores"][i] for i, m in enumerate(models)}}
                        for k, v in profile.items()},
            "col_keep": [bool(v) for v in col_keep],
        })
        # 长尾参考榜：被主设计剔除的桶单独成榜，绝不与主流桶混在同一张设计里
        # （P1-4 / 建议 3）。它的样本薄、列效应不可靠，所以只是"参考值"。
        if long_tail_board and dropped_cols:
            tail_cols = [b - 1 for b in dropped_cols]
            tail = _segment_board(S, W_cell, tail_cols, cell_min_w)
            window_out["long_tail"] = {
                "buckets": dropped_cols,
                "note": "样本量与同台家数均不足，仅供追踪趋势，不参与总榜名次",
                "scores": {m: (_fin(tail[i], 2)
                               if (i in tail and tail[i] is not None
                                   and np.isfinite(tail[i])) else None)
                           for i, m in enumerate(models)},
                "coverage": {b: int((np.isfinite(S[:, b - 1]) & row_keep).sum())
                             for b in dropped_cols},
            }
    # 显著性/冠军频率的参照必须是**榜单上戴冠那个源**（点估计名次），
    # 而不是 bootstrap 分布均值最高的源——两者偶尔会分叉，那时 † 标记就会
    # 指向一个并非冠军的源，读者完全无法察觉
    point_champ = next((r["model"] for r in _rank_rows(rows, keys=("score",))
                        if r.get("qualified") and r.get("score") is not None), None)
    if point_champ is not None:
        row_top = next((r for r in rows if r["model"] == point_champ), None)
        if row_top is not None:
            row_top["is_top"] = True
    # bootstrap 的冠军频率/显著性只作用于入围者（行内 qualified 已含 n_eff
    # 门槛与维度齐备两个条件）；置信区间全员提供。
    # 点估计的缺项口径以掩码注入（P0-1）：温度/降水各自的 (m, b) 是否有结论。
    # 缺这一层时 bootstrap 只能按重采样样本数判门槛，与点估计的 n_eff 门槛
    # 不同构，CI 中心会系统性偏离点估计。
    temp_point_valid = np.array(
        [[temp_score(temp_hourly[m].get(f"{b}d") or {}) is not None
          for b in range(1, hourly_lead_days + 1)] for m in models], dtype=bool)
    rain_point_valid = np.array(
        [[precip_score(precip_score_daily[m].get(f"{b}d") or {}) is not None
          for b in range(1, hourly_lead_days + 1)] for m in models], dtype=bool)
    # CI / 冠军频率 / 显著性必须对应**榜单上那个分数**（P0-2：不能给 A 数字配 B
    # 数字的区间）——故 bootstrap 走同一张劈分设计、同一个归总函数。
    bootstrap = day_block_bootstrap(
        [r for m in models for r in by_model[m]],
        [r for m in models for r in daily_by_model[m]],
        models, hourly_lead_days, thr_daily,
        TEMP_SCORE_PARTS, PRECIP_SCORE_PARTS, min_sample,
        runs=max(100, bootstrap_runs), seed=BOOTSTRAP_SEED,
        eligible=[r.get("qualified", False) for r in rows],
        temp_point_valid=temp_point_valid, rain_point_valid=rain_point_valid,
        bucket_valid=(temp_point_valid & rain_point_valid),
        block_days=block_days, adj_row=row_keep, adj_col=col_keep,
        # 加权与收缩必须与点估计**逐格同源**：给 A 数字配 B 数字的区间是直接误导
        adj_w=W_cell, adj_ridge=ridge,
        top_model=point_champ)
    for r in rows:
        b = bootstrap.get(r["model"], {})
        r["ci90"] = b.get("ci90") if r.get("score") is not None else None
        r["champion_pct"] = b.get("champion_pct")
        r["sig_vs_top"] = b.get("sig_vs_top")
        r["sig_vs_top_raw"] = b.get("sig_vs_top_raw")
        r["p_vs_top"] = b.get("p_vs_top")
    return _rank_rows(rows, keys=("score",))


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


def _model_status(models: list[str], snapshots: dict,
                  station_ids: list[str]) -> dict[str, str]:
    """各源的数据接入状态（P2-3）："ok" 有快照；"no_data" 已配置但从未产出数据。

    读者看到榜单末尾一个 0 分的源，无法区分"它报得差"和"它压根没数据"
    （fuxi_det 需 FUXI_DATA_TOKEN，未配置时就是后者）——状态必须显式标注。
    """
    out = {}
    for m in models:
        have = any(snapshots.get((sid, m)) for sid in station_ids)
        out[m] = "ok" if have else "no_data"
    return out


def _snapshot_quality(models: list[str], snapshots: dict) -> dict[str, dict]:
    """各源快照的序列完整性（P2-2）：残缺快照计数与中位序列长度。

    抓取中断/接口限流会留下"只有几十个小时"的残缺快照（实测 MSN 25~240 条
    剧烈波动）。它们仍会入库参与评估，虽然只影响名义样本量（n_eff 已处理），
    但"这家的数据有多少是残的"应当可见——否则读者会把样本量当成可靠性。
    判定：序列长度 < 该源中位长度 70% 记为残缺。
    """
    out: dict[str, dict] = {}
    for m in models:
        lens = []
        for (_sid, smodel), snaps in snapshots.items():
            if smodel != m:
                continue
            for snap in snaps:
                times = snap.get("hourly_time")
                if isinstance(times, list):
                    lens.append(len(times))
        if not lens:
            continue
        ordered = sorted(lens)
        median = ordered[len(ordered) // 2]
        truncated = sum(1 for L in lens if L < 0.7 * median)
        out[m] = {"snapshots": len(lens), "median_len": median,
                  "min_len": ordered[0], "truncated": truncated}
        if truncated:
            logger.warning("源 %s 有 %d/%d 份残缺快照（序列长度 < 中位 %d 的 70%%，最短 %d）",
                           m, truncated, len(lens), median, ordered[0])
    return out


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

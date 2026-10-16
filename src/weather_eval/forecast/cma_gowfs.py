"""中国气象数据网（data.cma.cn）全国智能网格预报快照器。

数据源：中国气象数据网 `data.cma.cn` 的**公开格点预报接口**（游客态可用、无需凭据；
接口契约来自 2026-10-11 线上实测，18 个经纬点 × 两个端点逐点核对，扫描脚本
`scripts/calibrate_cma_gowfs.py` 可复现）。站点以**经纬度**寻址（非站号——与
`cma_public` / `obs/cma_data` 的 WMO 站号寻址是两套寻址体系，勿混）。

  GET https://data.cma.cn/rest/gowfs/hour?lat=23.4783&lon=111.304
      Header: User-Agent（见契约第 1 条，必需）
  响应:
  {"hourTitle":"数据来源于全国智能网格预报","station":"",
   "detail":[                       # 96 点：前 24 点逐小时、其后逐 3 小时
     {"time":"2026-10-11 09:00",     # 有效时刻（北京时，见契约第 3 条）
      "temperature":"24.4",          # 字符串型气温 ℃
      "total_rain":"0.0",            # 降水 mm（分段口径，见契约第 5/6 条）
      "humidity":66.0,"visibility":9.12,"air_pressure":9999.0,
      "total_cloud":0.0,"weather_icon":"0","weather_info":"晴",
      "wind_direction":83.66,"wind_speed":0.91,
      "timelimit":0.0,"dealWith":false},
     ...],
   }

  GET https://data.cma.cn/rest/gowfs/day?lat=...&lon=...       # 逐日产品
  响应:
  {"dayTitle":"","station":"","publishTime":"",
   "detail":[
     {"date":"2026-10-11",           # 北京时自然日
      "pt":"2026-10-11 08:00",       # 产品发布时间（起报锚点，见契约第 7 条）
      "day":  {"weather":{"info":"晴","img":"0","temperature":"33.0"},"wind":{...}},
      "night":{"weather":{"info":"晴","img":"0","temperature":"22.0"},"wind":{...}},
      "precipitation":""},           # 恒为空串（见契约第 9 条）
     ... 共 30 天 ...]}

─────────────────────────────────────────────────────────────────────
关键契约（2026-10-11 线上实测；非官方契约，改版即可能失效，改动前必须复核）
─────────────────────────────────────────────────────────────────────
1. ── User-Agent 有白名单，空 UA 直接被拒 ──
   实测（同一 URL、同一时刻）：
     `weather-api-eval/0.1 (+https://github.com/)`（仓库统一 UA）→ 200
     `curl/8.5.0` → 200 ；`Mozilla/5.0` → 200 ；空 UA → **403**
     `python-requests/2.34.2` → **连接被服务端直接断开**（非 HTTP 响应）
   即服务端按已知爬虫 UA 名单拒绝，本源沿用仓库统一 UA（实测在名单外）。403 属
   确定性失败（4xx），`request_with_retries` 立即熔断、不烧退避。
2. ── 经纬度参数非法时返回 HTTP 417（不是 4xx 业务码）──
   实测 `lat=999` / `lon=999` / `lat=abc` / 缺参数 → **HTTP 417 + HTML 错误页**；
   合法但域外坐标（0,0）→ HTTP 200 + **空 body**。两者都是确定性失败：前者按 4xx
   熔断，后者在 JSON 解析处转为响亮失败。重试不改变结果。
3. ── 时间语义：接口原生就是**北京时**（无时区标注）──
   `time` 形如 `2026-10-11 09:00`。判据是**日变化相位**：18 个经纬点实测温度峰值
   恒在 14~15 时、谷值恒在 05~06 时（如梧州 10-11 的 05/14 时为 21.1/32.4℃），
   按北京时解释恰为"午后最高、清晨最低"；按 UTC 解释则变成北京时 22 时最高、
   13 时最低，与物理不符。格式解析**严格匹配** `YYYY-MM-DD HH:MM`：接口若改回
   带偏移的 ISO 串，这里必须响亮失败——时区错 8 小时不会报任何错，只会让该源
   全部样本静默归零（评估按北京时整点精确配对，错则样本数为 0）。
4. ── 网格是**分段**的：前 24 点逐小时，其后严格 3 小时 ──
   实测 18 个经纬点全部 96 点：相邻间隔恒为 `1h × 23` + `3h × 71`，即
     · 原生逐小时段：`t_first .. t_first+23h`（24 点，lead 1..24h）
     · 逐 3 小时段：`t_first+23h .. t_first+239h`（73 点，lead 24..240h）
   两段共用交接点 `t_first+23h`（该点的两种解释都覆盖 (t−3h, t]，见契约第 6 条）。
   总视野 240h = 10 天，覆盖「提前 N 天」天桶 1..10。
   **分段点是本源的第一个静默陷阱**：若把整条序列当成同一种分辨率处理，
   （a）温度会把 3h 台阶当逐小时直接用（日极值系统性失真），或
   （b）降水会把 3h 累计当 1h 累计直接入库（雨量放大 3 倍）。
   实现按契约第 6 条分开处置，`precip_1h_until` 留档每一份的实际交接点。
5. ── 降水口径**分段**（本契约最关键的一条，扫描脚本可复现）──
   近场（lead 1..24h，逐小时段）：`total_rain` 是 **1 小时累计**，窗口 (t−1h, t]。
   远场（lead 24..240h，逐 3 小时段）：`total_rain` 是 **3 小时累计**，
   窗口 (t−3h, t]。
   判定证据（`scripts/calibrate_cma_gowfs.py` 可复现，样本 = 18 个经纬点 × 96 点）：
     a) **近场 = 1h 累计（决定性）**：块内恒定性检验。把原生逐小时段按 3h 相位
        切成 3 小时块，若"每个点是 3h 累计、按小时重复展示"，块内 3 个值必须
        **逐位相同**。2026-10-11 运行：126 块中 112 块全零（无雨，无鉴别力），
        余下 14 个含雨块 **13 块有变化**（仅 1 块偶然相同）；块内相邻小时差 |Δ|
        中位 0.030mm、p90 0.220mm、最大 0.49mm——恰是 1 小时降水量的量级。
        （块计数随实时雨区漂移，同一天另一次运行得 113/13/12；**结论不变**：
        含雨块恒定比例 ≪ 1。）同时能观测到同一场雨的完整逐小时起落（如郑州
        10-12 00:00→06:00 的 0.28/0.48/0.49/0.48/0.25/0.0），即每个点各自描述
        自己那一个小时。
     b) **远场 = 3h 后向累计（决定性）**：与 `/day` 端点的定性等级对拍。12h 窗"白天
        08–20 时 / 夜间 20–08 时"的总量 = 恰好铺满该窗的 4 个累计点之和。按 CMA
        12h 等级标准（小雨 0.1~4.9 / 中雨 5.0~14.9 / 大雨 15.0~29.9 / 暴雨 ≥30mm）
        判级：后向口径 (t−3h, t] 命中 **77/102（75%）**，前向口径 [t, t+3h) 命中
        58/102（57%）；分等级看 小雨 86% vs 64%、中雨 60% vs 40%。**大雨档 n=3 偏小
        且结论相反（33% vs 67%），如实留档**，待满月样本后按 −1/0/+1h 平移标定法复核
        （同彩云/和风/AccuWeather/cma_public 的处置原则）。
        即远场点值就是 3h 累计本身，逐小时入库前必须按 1/3 摊到小时。
     c) 与同机构产品的一致性旁证：本接口与 `cma_public`（weather.cma.cn）的 3h
        网格**同相位**（02/05/08/11/14/17/20/23），且 CMA-NDFS 等 3h 产品在既有
        存档中已确认为后向累计（见 ew4all.py 契约）——同一机构的网格产品族沿用
        同一套窗口约定。
   入库处置（两段分别走 `resample.py` 的两个函数，缺一不可）：
     · 近场：原生 1h 累计**直接透传**，与观测 `rain@t`=(t−1h, t] 的配对口径同字。
     · 远场：`spread_accumulation(3)` —— 平铺子集无损摊薄，数学上等价于按窗口
       均摊 /3，是唯一保证"任意完整跨度求和 == 原始累计总量"的确定性展开
       （与风乌 tp6h、EW4ALL 的 3h/6h 族、`cma_public` 同法，共用一个实现）。
   两项代价写入已知限制：远场短时强降水被摊到 3 小时（对 1mm/h 晴雨阈值偏保守）；
   远近场交接处（lead 24h）两种口径并置，`precip_1h_until` 逐份留档。
6. ── 交接点的处理（避免重复计数）──
   两段共用时刻 `t_seam = t_first + 23h`。近场的 24 个 1h 值已完整覆盖
   `[t_first, t_seam]`；若把远场第一个 3h 点 `t_seam+3h` 的窗口 (t_seam, t_seam+3h]
   与近场并置，起报小时轴上的每个小时仍恰属一个窗口（实现只把 `h > t_seam` 的
   小时交给远场平铺），因此总降水严格守恒、无重复计入、无缺口。
   哨兵：若远场相邻间隔不是 3 小时（缺一个采样点），`tile_endpoints` 会告警，
   缺口覆盖小时为缺测（None）——绝不借用邻窗值。
7. ── 起报锚点：逐日端点自述的 `pt`；结构恒等式兜底 ──
   `/day` 响应每个条目都带 `pt`（产品发布时间）。实测 18/18 个经纬点满足恒等式

       pt == 首个逐小时有效时刻 − 1h == 末个逐小时有效时刻 − 240h

   （三个量在全部 18 点上逐位相等），故 `pt` 就是本产品的起报时刻。这是接口
   自述量之间的恒等式，不依赖任何硬编码偏移，产品换循环时自动跟随。
   兜底梯子（每步写入 `issue_source_detail` / `issue_notes` 供审计）：
     a) `day_pt`：读 `/day` 的 `pt`（首选，接口自述）。
     b) `axis_minus_1h`：`首个逐小时时刻 − 1h` —— 由第 4/7 条的结构恒等式推断，
        与 `day_pt` 给出**同一个物理量**（同一次起报），故同样声明 `model_run`；
        两者相差超过 24h 时告警（产品循环可能已变）。
     c) `axis_start`：退回首个逐小时时刻（lead 从 0 起算）——仅当 pt 不可解析
        且结构恒等式不成立（首点非整点/间隔异常）时使用。
   逐日端点抓取失败**不拖垮**逐小时快照：此时降级为不带逐日块的快照（WARNING
   可见），锚点按 (b) 推断——**lead 分布与 (a) 完全一致**，不会因为逐日端点
   抖动而让同一批样本落进不同的天桶。
8. ── 逐日产品：只接温度，且日最低要**移一天**取用 ──
   `/day` 的 `day`/`night` 是**白天窗（08:00–20:00）与夜间窗（20:00–次日 08:00）**
   的要素，不是北京时自然日的极值（与 `cma_public` 姊妹端点同类的日界陷阱）。
     · `day.weather.temperature` ≈ 当日白天窗最高 → `temp_max(D)`。
     · `night.weather.temperature` ≈ **当日夜间窗**最低。自然日 D 的最低气温出现在
       D 日凌晨 05~07 时，落在 **D−1 的夜间窗**里。实测 162 个样本对拍：
       `nightT(D) − 自然日最低(D)` 中位 |Δ| = 1.07℃、均值 −0.17℃，而
       `nightT(D−1) − 自然日最低(D)` 中位 |Δ| = 0.68℃、均值 +0.003℃（近乎无偏）。
       故 `temp_min(D) = nightT(D−1)`；D−1 无夜间条目时该日 `temp_min` 记缺测
       （绝不挪用 D 自己的夜间值——那会系统性把日最低报高约 1℃）。
     · 与本源自有逐小时产品存在**系统差**：`dayT − 逐小时日最高` 均值 +0.99℃
       （中位 |Δ| 0.78℃、p90 3.42℃）。这不是接入缺陷而是两个产品在日极端处的
       固有差异（类比中科星图 `tem_max` 与自家逐小时聚合 ±1℃ 的产品差，
       见 README §9）。由于逐小时段（lead ≤ 24h）覆盖完整，按天评估在 lead ≤ 10 天
       全部走逐小时聚合，本块只在**逐小时覆盖不足**时才启用（`daily_source_fallback`），
       即只贡献 lead 11..16 天的日温度样本；`temp_src="daily"` 的样本在报告
       "口径注记"中单独披露。
     · 首日（offset 0）没有 `temp_min`（D−1 不在响应里），如实记缺测。
9. ── 逐日降水：恒为空串，拒绝接入（留档）──
   实测 18 个经纬点 × 30 天 = 540 个样本，`precipitation` 字段**全部是空串 `""`**。
   按 base.py 契约，不得由天气现象码（`info` 的"小雨/中雨/大雨"）折算毫米——那是
   凭空造值。故 `precipitation` 给全 null 数组（与中科星图逐日块同款处置）。
10. ── 视野 30 天但只评 16 天 ──
    `/day` 返回起报日起 30 天，`/hour` 只到 10 天。按 base.py 契约，评测范围由
    `eval.daily_max_offset_days`（默认 16）划定，超范围的日产品由**写路径统一截除**
    （`snapshot_meta.truncate_daily_block`）——provider 不自行截断，也不必为多出来
    的天申请评测语义。故本源实际被评测的日偏移是 1..10（逐小时聚合兜底）+ 11..16
    （源自带日产品补位，仅温度）。
11. ── 不做格点回显 ──
    两个响应都不含吸附后的格点经纬度（`station` 恒为空串），`grid_lat/lon` 记请求
    坐标；底层为全国智能网格预报（5km/1km 级格点插值到请求坐标，具体分辨率未由
    接口披露），代表点误差计入已知限制。
12. ── 其他字段的缺测哨兵 ──
     `air_pressure` / `visibility` 在部分时刻回填 `9999.0`（同字段族哨兵）。
     `total_rain` 实测 1728 点最大 15.07mm、未出现哨兵，但保留 `≥999` 即缺测的
     防御性哨兵（与 `cma_public` 的 999.9 同口径，见契约第 12 条同源处置）；
     温度同理设 ±100℃ 物理上界（实测 −17.81~35.25℃，未触发）。
     `timelimit` 恒 0.0、`dealWith` 恒 false——未参与评估，仅在 docstring 留档。

已知风险：无官方契约的公开接口，页面/接口改版即可能失效（失败只影响本源，不阻断
其他源）。失效形态有四种，全部在实现里转为响亮失败，绝不落空快照：HTTP 417（参数
非法）、HTTP 200 + 空 body（域外坐标）、`detail` 缺失/为空、`time` 字段格式漂移。
"""

from __future__ import annotations
import logging
import re
from datetime import datetime, timedelta
from typing import Any

import requests

from .base import ForecastProvider
from .http import (DEFAULT_UA, DEFAULT_TIMEOUT as HTTP_DEFAULT_TIMEOUT,
                   TimeBudget, request_with_retries)
from .resample import floor_hour, interpolate_hourly, spread_accumulation

logger = logging.getLogger(__name__)

BASE_URL = "https://data.cma.cn/rest/gowfs"
HOUR_PATH = "/hour"
DAY_PATH = "/day"

# 仓库统一 UA：实测不在该站的爬虫 UA 名单内（名单拒绝空 UA 与 python-requests，
# 见 docstring 契约第 1 条）。若日后被误伤，只改这一处。
HEADERS = {"User-Agent": DEFAULT_UA,
           "Accept": "application/json, text/plain, */*",
           "Referer": "https://data.cma.cn/"}

SOURCE = "cma_gowfs"
MODEL_NAME = "cma_gowfs_v1"

# 网格结构（docstring 契约第 4/6 条）：前 24 点逐小时，其后严格 3 小时
NATIVE_1H_POINTS = 24
FAR_STEP_HOURS = 3
HORIZON_HOURS = 240
# 实测原生逐小时段长度应为 24 点；明显短于此即视为上游截断（健康哨兵用）
_MIN_NATIVE_1H_POINTS = 20

# 降水缺测哨兵：同字段族在 air_pressure/visibility 上实测为 9999.0，故对 total_rain
# 保留 ≥999 即缺测的防御性哨兵（docstring 契约第 12 条；实测 total_rain 最大 15.07mm）
PRECIP_SENTINEL_MIN = 999.0
# 地面气温的物理上界（实测 −17.81~35.25℃，未触发，防御性哨兵）
TEMP_ABS_MAX = 100.0

# 严格格式：接口原生北京时、无时区标注；形态一变即失败（docstring 契约第 3 条）
_RE_TIME = re.compile(r"^(\d{4})-(\d{1,2})-(\d{1,2}) (\d{1,2}):(\d{2})$")


class GowfsPayloadError(RuntimeError):
    """响应外壳/结构/字段异常（契约漂移、坐标域外、参数非法）——确定性失败，重试无意义。"""


def parse_dt_bj(s: Any) -> datetime:
    """接口时间串 `YYYY-MM-DD HH:MM` → 北京时 naive datetime。

    严格匹配形态并拒绝任何带偏移的串（docstring 契约第 3 条）：把 UTC 串当北京时读
    会整体错 8 小时且不报任何错，只会让该源样本静默归零，故宁可响亮失败。
    """
    m = _RE_TIME.match(str(s).strip()) if s is not None else None
    if not m:
        raise GowfsPayloadError(
            f"气象数据网时间串格式异常（契约应为 'YYYY-MM-DD HH:MM' 北京时）: {s!r}")
    year, month, day, hour, minute = (int(g) for g in m.groups())
    return datetime(year, month, day, hour, minute)


def _num(v: Any) -> float | None:
    """数值归一：非法/bool/NaN/inf → None，绝不把缺测伪装成 0。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _temp(v: Any, stats: dict[str, int]) -> float | None:
    """温度归一：超出 ±100℃ 物理上界视作占位/契约漂移（docstring 契约第 12 条）。"""
    f = _num(v)
    if f is None:
        return None
    if abs(f) > TEMP_ABS_MAX:
        stats["temp_implausible"] = stats.get("temp_implausible", 0) + 1
        return None
    return f


def _precip(v: Any, stats: dict[str, int]) -> float | None:
    """降水归一：非物理值（≥999 的占位、负值）一律置 None 并计数。

    实测本源 total_rain 未出现哨兵（1728 点最大 15.07mm），但同字段族在
    air_pressure/visibility 上回填 9999.0，故保留防御性哨兵；负降水物理上不存在。
    """
    f = _num(v)
    if f is None:
        return None
    if f >= PRECIP_SENTINEL_MIN:
        stats["sentinel"] = stats.get("sentinel", 0) + 1
        return None
    if f < 0:
        stats["negative"] = stats.get("negative", 0) + 1
        return None
    return f


def extract_hour_points(payload: Any) -> tuple[list[tuple[datetime, dict]], dict[str, Any]]:
    """校验外壳并展开 `detail` → [(北京时整点, 条目)]（升序、按时刻去重）。

    失效形态（docstring 契约第 2 条）：HTTP 200 + 空 body、`detail` 缺失/非列表/
    为空、条目非对象——全部转为 `GowfsPayloadError`（响亮失败，绝不建空快照）。
    """
    if not isinstance(payload, dict):
        raise GowfsPayloadError(f"响应不是 JSON 对象: {str(payload)[:200]!r}")
    data = payload.get("detail")
    if not isinstance(data, list) or not data:
        raise GowfsPayloadError(
            "响应 detail 为空或非列表（坐标可能在预报域外，或接口契约已变化）: "
            f"{str(data)[:120]!r}")
    points: dict[datetime, dict] = {}
    conflicts = 0
    bad_time = 0
    non_hour = 0
    for entry in data:
        if not isinstance(entry, dict):
            bad_time += 1
            continue
        try:
            t = parse_dt_bj(entry.get("time"))
        except GowfsPayloadError:
            bad_time += 1
            continue
        if t.minute or t.second:
            non_hour += 1
        prev = points.get(t)
        if prev is not None:
            # 同一时刻数值不一致 = 响应内混了两个数据版本，必须可见
            if (prev.get("temperature"), prev.get("total_rain")) != \
                    (entry.get("temperature"), entry.get("total_rain")):
                conflicts += 1
            continue
        points[t] = entry
    meta = {"conflicts": conflicts, "bad_time": bad_time, "non_hour": non_hour}
    return sorted(points.items()), meta


def native_1h_prefix_len(points: list[tuple[datetime, dict]]) -> int:
    """原生逐小时前缀的长度（点）：从首点起相邻间隔严格 1h 的连续段。

    这就是 docstring 契约第 4/6 条的分段点来源，不依赖任何硬编码偏移——产品若把
    原生逐小时段从 24 点改成 48 点，这里自动跟随，无需改代码。
    """
    if not points:
        return 0
    n = 1
    for (t0, _), (t1, _) in zip(points, points[1:]):
        if (t1 - t0) == timedelta(hours=1):
            n += 1
        else:
            break
    return n


def build_hourly_series(points: list[tuple[datetime, dict]], n_native: int
                        ) -> tuple[list[datetime], list[float | None], list[float | None],
                                   dict[str, int]]:
    """按分段口径构造逐小时轴与两列要素（docstring 契约第 5/6 条）。

    返回 (axis, temps, precips, stats)。轴为首末采样之间的逐小时整点（含两端）。
    温度走 `interpolate_hourly`（线性插值；既有整点采样原样保留，端点缺测停插值）；
    降水分段处置：`h <= t_seam` 用原生 1h 值直接透传，`h > t_seam` 走
    `spread_accumulation(3)` 后向累计均摊。
    """
    stats: dict[str, int] = {}
    # 轴与插值产物必须同地板规则：`interpolate_hourly` 的键是整点（非整点采样会
    # 被地板），而本源的时间正则接受任意分钟（实测契约恒为整点，但契约漂移时会
    # 出现 :30）。直接用原始首末时刻累加 ⇒ `temps.get(h)` 对每个 h 落空 ⇒
    # **整列温度静默变缺测**。故轴/交接点比较/近场键一律用地板后的时刻，
    # 远场（平铺相位由原始采样定义）仍用原始时刻，两侧各自自洽（第五轮 P1-2）。
    first, last = points[0][0], points[-1][0]
    axis_first, axis_last = floor_hour(first), floor_hour(last)
    # 原生逐小时前缀至少 2 点才成立"前段是逐小时"这一判断；不足时按**全 3h 网格**
    # 处理（降级路径——绝不把 3h 累计误当 1h 累计透传）
    if n_native < 2:
        n_native = 0
    if n_native:
        # 交接点按**地板后的时刻**与轴对称轻柔比较（见上方同地板规则的注释）
        t_seam = floor_hour(points[n_native - 1][0])
        near = [(floor_hour(t), _precip(e.get("total_rain"), stats))
                for t, e in points[:n_native]]
        # 远场以交接点为相位锚（tile_endpoints 以首个采样为相位）；交接点留在远场
        # 样本里是安全的——同一时刻的两种解释都覆盖 (t−3h, t]，而该小时由近场接管
        far = [(t, _precip(e.get("total_rain"), stats))
               for t, e in points[n_native - 1:]]
        n_after = int((axis_last - t_seam).total_seconds() // 3600)
        hours_after = [t_seam + timedelta(hours=i) for i in range(1, n_after + 1)]
    else:
        t_seam = None
        near = []
        far = [(t, _precip(e.get("total_rain"), stats)) for t, e in points]
        n_axis = int((axis_last - axis_first).total_seconds() // 3600)
        hours_after = [axis_first + timedelta(hours=i) for i in range(n_axis + 1)]

    temp_samples = [(t, _temp(e.get("temperature"), stats)) for t, e in points]
    temps = dict(interpolate_hourly(temp_samples))

    far_col = spread_accumulation(far, hours_after, FAR_STEP_HOURS) \
        if hours_after else []
    far_map = dict(zip(hours_after, far_col))
    near_map = dict(near)
    axis_n = int((axis_last - axis_first).total_seconds() // 3600)
    axis = [axis_first + timedelta(hours=i) for i in range(axis_n + 1)]
    temp_col = [temps.get(h) for h in axis]
    # 逐小时降水：交接点及以前用原生 1h 值透传，其后用 3h 后向累计均摊
    precip_col = [near_map[h] if (t_seam is not None and h <= t_seam)
                  else far_map.get(h) for h in axis]
    return axis, temp_col, precip_col, stats


def extract_day_publish(payload: Any) -> tuple[datetime | None, set[str]]:
    """逐日响应里的产品发布时间 `pt`（北京时）→ (pt, 原始串集合)。

    `pt` 在整份响应内应只有一个取值（docstring 契约第 7 条）；多于一个即说明响应
    跨了两个发布版本，此时不采用、交由调用方按结构恒等式兜底。
    """
    if not isinstance(payload, dict):
        return None, set()
    data = payload.get("detail")
    if not isinstance(data, list):
        return None, set()
    raws: set[str] = set()
    parsed: set[datetime] = set()
    for entry in data:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("pt")
        if raw is None or str(raw).strip() == "":
            continue
        raws.add(str(raw))
        try:
            parsed.add(parse_dt_bj(raw))
        except GowfsPayloadError:
            continue
    if len(parsed) == 1:
        return next(iter(parsed)), raws
    return None, raws


def daily_precip_from_hourly(axis: list[datetime],
                             precips: list[float | None]) -> dict[str, float | None]:
    """逐小时降水 → 北京时自然日 24h 累计降水（mm），{YYYY-MM-DD: 值 或 None}。

    为什么这样做（docstring 契约第 9 条）：`/day` 的 `precipitation` 字段实测**恒为空串**
    （18 点 × 30 天 = 540 个样本无一例外），即日端点不提供日降水量。但本源的逐小时
    序列是完整的 1h 分辨率（近场原生、远场均摊），按自然日逐小时**求和**即得该日
    总量——这是对同一份源数据做无损聚合，不是从天气现象码折算 mm（后者是凭空造值，
    本项目在星图 / cma_public 上均已明令拒绝）。

    三条边界（否则日降水量会被静默高估或凭空出现）：
      * **必须 24 小时齐全**：当天不是恰好 24 个整点 → 该日记 None（缺测），
        绝不拿"部分小时的和"当全天量（那会把缺测伪装成小雨）。
      * **任一小时缺测即整日缺测**：求和里含 None 则该日 None——部分和不是全天量。
      * **超出逐小时视野的自然日记 None**：`/hour` 只到 +240h（天桶 1..10），
        第 11~30 天没有逐小时来源，累加无从谈起，如实留空。
    """
    buckets: dict[str, dict[int, float | None]] = {}
    for h, v in zip(axis, precips):
        buckets.setdefault(h.date().isoformat(), {})[h.hour] = v
    out: dict[str, float | None] = {}
    for day, hours in buckets.items():
        if sorted(hours) != list(range(24)):
            out[day] = None          # 覆盖不全：不得拿部分小时的和冒充全天
            continue
        vals = [hours[k] for k in range(24)]
        out[day] = None if any(v is None for v in vals) else round(sum(vals), 4)
    return out


def build_daily_block(payload: Any,
                       daily_precip: dict[str, float | None] | None = None,
                       ) -> dict[str, Any] | None:
    """逐日响应 → {time: [YYYY-MM-DD...], data: {MODEL: {temp_max/min/precipitation}}}。
    只接温度 + 由逐小时聚合的日降水（docstring 契约第 8/9 条）：
      temp_max(D) = day.weather.temperature        （当日白天窗 08–20 时最高）
                    最低气温落在 D 日凌晨，属 D−1 的夜间窗；实测中位 |Δ| 从 1.07℃
                    降到 0.68℃、均值 +0.17℃ → +0.003℃，见契约第 8 条对拍数据）
      precipitation(D) = 该日 24 个整点降水之和（由 daily_precip_from_hourly 聚合）
                    日端点自身**不提供**日降水量（字段恒空串）；此处的 mm 全部来自
                    本源逐小时序列的无损求和，不是由天气现象码折算。覆盖不全/含缺测/
                    超出 +240h 视野的日子一律 None（见该函数的三条边界）。
    D−1 无夜间条目时该日 temp_min 记缺测——绝不挪用 D 自己的夜间值。
    """
    if not isinstance(payload, dict):
        return None
    data = payload.get("detail")
    if not isinstance(data, list) or not data:
        return None
    day_t: dict[str, float | None] = {}
    night_t: dict[str, float | None] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        raw = entry.get("date")
        if not isinstance(raw, str) or len(raw) < 10:
            continue
        try:
            day_key = datetime.strptime(raw[:10], "%Y-%m-%d").date().isoformat()
        except ValueError:
            continue
        for key, slot in (("day", day_t), ("night", night_t)):
            blk = entry.get(key)
            if not isinstance(blk, dict):
                slot.setdefault(day_key, None)
                continue
            weather = blk.get("weather")
            v = weather.get("temperature") if isinstance(weather, dict) else None
            if day_key not in slot:
                slot[day_key] = _num(v)
    times = sorted(day_t)
    if not times:
        return None
    stats: dict[str, int] = {}
    temp_max = [_temp(day_t.get(d), stats) for d in times]
    # 日最低移一天取用：D 的 temp_min 来自 D−1 的 night（docstring 契约第 8 条）
    temp_min = []
    for d in times:
        try:
            prev = (datetime.strptime(d, "%Y-%m-%d")
                    - timedelta(days=1)).date().isoformat()
        except ValueError:
            prev = None
        temp_min.append(_temp(night_t.get(prev), stats) if prev else None)
    # 日降水来自逐小时聚合（daily_precip 为 None 时——如逐日端点不可用——全 null）
    precip_col_daily = [None if daily_precip is None else daily_precip.get(d)
                        for d in times]

    return {"time": times,
            "data": {MODEL_NAME: {"temp_max": temp_max, "temp_min": temp_min,
                                  "precipitation": precip_col_daily}}}
# provider 自有词表 → 契约枚举（见 snapshot_meta.VALID_ISSUE_SOURCES）：
#   day_pt         = 读逐日端点自述的产品发布时间      → 真实模式轮次口径
#   axis_minus_1h  = 由结构恒等式推断的同一物理量       → 真实模式轮次口径
#   axis_start     = 时间轴首点                        → 时间轴首点口径
_ISSUE_SOURCE_MAP = {
    "day_pt": "model_run",
    "axis_minus_1h": "model_run",
    "axis_start": "axis_start",
}


def derive_issue(points: list[tuple[datetime, dict]], n_native: int,
                 pt: datetime | None, notes: list[str]) -> tuple[datetime, str]:
    """解析起报锚点，候选顺序见 docstring 契约第 7 条。

    返回 (issue, issue_source_detail)。三种候选中前两种给出**同一个物理量**
    （实测 18/18 站 pt == 首点−1h == 末点−240h），故都声明 `model_run`；只有
    结构恒等式也不成立时才退回时间轴首点（`axis_start`）。
    """
    first_valid = points[0][0]
    last_valid = points[-1][0]
    # 结构恒等式推断值：原生逐小时段的 lead 应为 1..24h（docstring 契约第 4 条）
    axis_minus_1h = first_valid - timedelta(hours=1)
    if pt is not None:
        if pt == axis_minus_1h:
            return pt, "day_pt"
        delta_h = (pt - axis_minus_1h).total_seconds() / 3600.0
        notes.append(
            f"逐日端点 pt={pt:%Y-%m-%dT%H:%M} 与逐小时轴首点−1h="
            f"{axis_minus_1h:%Y-%m-%dT%H:%M} 相差 {delta_h:+.0f}h（实测应恒为 0，"
            "产品循环节奏可能已变化）")
        # 相差超过 24h（早或晚）即判两端点不同源：入库的是逐小时轴，故以轴为准。
        # 绝不信 pt —— 用一个跨天的错锚点会把全部 lead 平移，"提前 N 天"天桶整体错位。
        if abs(delta_h) > 24:
            notes.append("pt 与逐小时轴的结构恒等式相差超过 24h，判为两个端点不同源，"
                         "改用轴结构恒等式推断")
            return axis_minus_1h, "axis_minus_1h"
        if pt > first_valid:
            # 会产生负 lead（评估侧静默丢弃），不可用
            notes.append(
                f"pt={pt:%Y-%m-%dT%H:%M} 晚于首个有效时刻 {first_valid:%Y-%m-%dT%H:%M}"
                "（会产生负 lead），不采用，改用轴结构恒等式推断")
            return axis_minus_1h, "axis_minus_1h"
        return pt, "day_pt"
    # 无可用 pt：结构恒等式兜底
    if n_native == NATIVE_1H_POINTS:
        notes.append("逐日端点未给出可用 pt，按轴结构恒等式（首点−1h）推断起报时刻")
        return axis_minus_1h, "axis_minus_1h"
    horizon = last_valid - timedelta(hours=HORIZON_HOURS)
    notes.append(
        f"逐日端点未给出可用 pt、原生逐小时段为 {n_native} 点（预期 {NATIVE_1H_POINTS}），"
        "结构恒等式不成立；退回末点−240h 与首点较晚者")
    return max(first_valid, horizon), "axis_start"


def _fmt(dt: datetime) -> str:
    """审计用的时刻表示（北京时墙钟）。"""
    return dt.strftime("%Y-%m-%dT%H:%M")



class CmaGowfsProvider(ForecastProvider):
    """中国气象数据网智能网格预报快照器：无需凭据，单模型快照 dict。"""

    def __init__(self, timeout: int | tuple = HTTP_DEFAULT_TIMEOUT, retries: int = 3,
                 session: requests.Session | None = None):
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self._budget = TimeBudget()  # 单源单轮总预算（第四轮 P1-7）

    # ------------------------------------------------------------------ 对外
    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> dict:
        # 预算是**每站各自**的额度：本实例被 CLI 在所有站点间复用，不重置会让
        # 第 1 站耗尽预算后，后续站全部零重试。
        self._budget.reset()
        hour_payload = self._request(HOUR_PATH, station)
        points, meta = extract_hour_points(hour_payload)
        if not points:
            raise RuntimeError(
                f"气象数据网站点 {station.id} 解析出 0 个有效时刻，拒绝入库空快照"
                "（空快照会被同 issue 幂等键永久锁死）")
        n_native = native_1h_prefix_len(points)
        axis, temp_col, precip_col, stats = build_hourly_series(points, n_native)

        # 逐日端点：只用于取 pt 与逐日温度块；失败**绝不拖垮**逐小时快照
        notes: list[str] = []
        if meta["conflicts"]:
            notes.append(f"{meta['conflicts']} 个重复时刻的数值不一致（疑似混版本）")
        if meta["bad_time"]:
            notes.append(f"{meta['bad_time']} 条记录的 time 无法解析")
        if meta["non_hour"]:
            notes.append(f"{meta['non_hour']} 个时刻不是整点（接口契约应恒为整点）")

        day_payload, day_error = self._try_request_day(station)
        pt, pt_raws = extract_day_publish(day_payload) if day_payload is not None \
            else (None, set())
        if day_payload is None:
            notes.append(f"逐日端点不可用（{day_error}），本次快照不带逐日预报块")
        elif len(pt_raws) > 1:
            notes.append(f"逐日响应内出现 {len(pt_raws)} 个不同的 pt（疑似跨版本）")
        issue, issue_detail = derive_issue(points, n_native, pt, notes)

        # 空快照不得落盘：任一侧有值就仍有存档价值（另一侧按缺测）
        if all(v is None for v in temp_col) and all(v is None for v in precip_col):
            raise RuntimeError(
                f"气象数据网站点 {station.id} 起报 {_fmt(issue)} 的温度与降水均无有效值，"
                "拒绝入库空快照")

        # 日降水由逐小时聚合（/day 端点不提供降水量，字段恒空串）
        daily_precip = daily_precip_from_hourly(axis, precip_col)
        daily_block = (build_daily_block(day_payload, daily_precip)
                       if day_payload is not None else None)
        self._warn_health(station, points, n_native, axis, temp_col, precip_col,
                          stats, notes, issue)

        snapshot = {
            "issue_iso": issue.strftime("%Y-%m-%dT%H:%M"),
            "station_id": station.id,
            "source": SOURCE,
            "models": [MODEL_NAME],
            # 接口不回显吸附格点（docstring 契约第 11 条）；请求坐标即代表点
            "grid_lat": float(station.lat),
            "grid_lon": float(station.lon),
            "elevation": None,
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            # 起报锚点来源与推导说明。细节留在 issue_source_detail，契约字段用统一枚举
            "issue_source": _ISSUE_SOURCE_MAP[issue_detail],
            "issue_source_detail": issue_detail,
            "issue_raw": (next(iter(pt_raws), None) or
                          issue.strftime("%Y-%m-%dT%H:%M")),
            "issue_notes": notes,
            # 原生分辨率：远场主导步长（3h）。近场前 24h 为原生 1h，
            # 由 precip_1h_until 留档交接点（同 EW4ALL/CMA-NDFS 的处置）
            "resolution_hours": FAR_STEP_HOURS,
            "precip_unit": "mm",
            # 远场为 3h 后向累计；近场为原生 1h 累计（见 docstring 契约第 5 条）
            "precip_accum_window_hours": FAR_STEP_HOURS,
            "grid_step_hours": FAR_STEP_HOURS,
            "precip_interval_hours": FAR_STEP_HOURS,
            "expansion": (
                "temperature: linear-interp-1h; "
                "precipitation: native 1h passthrough up to seam, then "
                "3h back-accumulation spread over (t-3h, t]"),
            "precip_1h_until": (
                (points[n_native - 1][0].strftime("%Y-%m-%dT%H:00"))
                if 2 <= n_native else None),
            "native_1h_points": n_native,
            "horizon_hours": HORIZON_HOURS,
            "missing_sentinel_precip": stats.get("sentinel", 0),
            "negative_precip": stats.get("negative", 0),
            "implausible_temp": stats.get("temp_implausible", 0),
            "hourly_time": [h.strftime("%Y-%m-%dT%H:%M") for h in axis],
            "data": {MODEL_NAME: {
                "temperature_2m": temp_col,
                "precipitation": precip_col,
            }},
        }
        if daily_block:
            snapshot["daily_time"] = daily_block["time"]
            snapshot["daily"] = daily_block["data"]
        logger.info(
            "站点 %s 已抓取气象数据网智能网格预报起报 %s（源=%s，%d 个逐小时点，"
            "原生 1h 段 %d 点，逐日产品 %d 天）", station.id, snapshot["issue_iso"],
            issue_detail, len(axis), n_native,
            len(daily_block["time"]) if daily_block else 0)
        return snapshot

    # ------------------------------------------------------------------ 内部
    def _request(self, path: str, station: Any) -> Any:
        """逐小时端点请求（退避/熔断统一走共享助手）。"""
        return request_with_retries(
            self.session, BASE_URL + path, params=self._params(station),
            headers=HEADERS, timeout=self.timeout, retries=self.retries,
            source="气象数据网", classify=self._classify, budget=self._budget)

    def _try_request_day(self, station: Any) -> tuple[Any, str]:
        """逐日端点请求；任何失败只返回 (None, 原因)，绝不向上抛。

        逐日是延长线而非主干（base.py 契约）：它挂了不能拖垮已经抓到的逐小时快照。
        这里不烧额外退避——失败一次就放弃延长线，把时间留给下一个站点/源。
        """
        try:
            return self._request(DAY_PATH, station), ""
        except Exception as e:  # noqa: BLE001  逐日块失败绝不拖垮整份快照
            return None, str(e)

    @staticmethod
    def _params(station: Any) -> dict[str, Any]:
        return {"lat": station.lat, "lon": station.lon}

    @staticmethod
    def _classify(resp: Any) -> tuple[str, Any]:
        status = getattr(resp, "status_code", None)
        if status == 200:
            try:
                return "return", resp.json()
            except Exception as e:  # noqa: BLE001  200 但非 JSON 属确定性失败
                raise GowfsPayloadError(f"气象数据网响应非 JSON: {e}") from e
        digest = ""
        try:
            digest = (resp.text or "")[:200]
        except Exception:  # noqa: BLE001
            pass
        if status == 417:
            # 实测坐标参数非法（lat/lon 超范围、非数值、缺参数）返回 417 + HTML 页
            return "fatal", GowfsPayloadError(
                f"气象数据网拒绝请求 HTTP 417（经纬度参数非法，如超范围或非数值）"
                f"body={digest!r}")
        if isinstance(status, int) and 400 <= status < 500 and status != 429:
            # 403 实测为 UA 反爬名单拒绝（docstring 契约第 1 条）
            return "fatal", GowfsPayloadError(
                f"气象数据网拒绝请求 HTTP {status}（User-Agent 可能落入反爬名单；"
                f"实测空 UA 与 python-requests 被拒）body={digest!r}")
        return "retry", f"HTTP {status} body={digest!r}"

    @staticmethod
    def _warn_health(station: Any, points: list[tuple[datetime, dict]], n_native: int,
                     axis: list[datetime], temps: list, precips: list,
                     stats: dict[str, int], notes: list[str], issue: datetime) -> None:
        """健康哨兵：把静默退化变成可见日志（本源失效形态多为"少了一部分数据"）。"""
        for note in notes:
            logger.warning("气象数据网站点 %s：%s", station.id, note)
        if stats.get("sentinel"):
            logger.warning(
                "气象数据网站点 %s 有 %d 个时刻的降水是占位哨兵值（≥%.0f），已按缺测处理",
                station.id, stats["sentinel"], PRECIP_SENTINEL_MIN)
        if stats.get("negative"):
            logger.warning(
                "气象数据网站点 %s 有 %d 个时刻降水为负值（非物理），已按缺测处理",
                station.id, stats["negative"])
        if stats.get("temp_implausible"):
            logger.warning(
                "气象数据网站点 %s 有 %d 个温度超出物理上界（±%.0f℃），已按缺测处理，"
                "契约可能已变化", station.id, stats["temp_implausible"], TEMP_ABS_MAX)
        if all(v is None for v in temps):
            logger.warning("气象数据网站点 %s 插值后温度全缺测，契约可能已变化", station.id)
        if all(v is None for v in precips):
            logger.warning(
                "气象数据网站点 %s 降水全缺测（分段展开后无任何有效值）", station.id)
        if n_native and n_native < _MIN_NATIVE_1H_POINTS:
            logger.warning(
                "气象数据网站点 %s 原生逐小时段仅 %d 点（预期 %d），近场可能被上游截断",
                station.id, n_native, NATIVE_1H_POINTS)
        if len(axis) < 24:
            logger.warning(
                "气象数据网站点 %s 逐小时轴仅 %d 点，时效可能被严重截断",
                station.id, len(axis))
        max_lead = int((axis[-1] - issue).total_seconds() // 3600)
        if max_lead < HORIZON_HOURS:
            logger.warning(
                "气象数据网站点 %s 最大时效仅 %dh（预期 %dh/10 天），长时效可能被截断",
                station.id, max_lead, HORIZON_HOURS)
        far_gaps = {(b[0] - a[0]).total_seconds() for a, b in
                    zip(points[n_native:], points[n_native + 1:])}
        if far_gaps and far_gaps != {FAR_STEP_HOURS * 3600}:
            logger.warning(
                "气象数据网站点 %s 远场采样间隔异常 %s（预期恒为 %dh），部分小时会按缺测处理",
                station.id, sorted(int(g / 3600) for g in far_gaps), FAR_STEP_HOURS)

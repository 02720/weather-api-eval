"""MSN 天气（msn.cn/zh-cn/weather）逐小时预报快照器。

数据源：MSN 中国天气页的**服务端渲染（SSR）内嵌状态**，实际预报由中国天气网
（weather.com.cn）提供，页脚 `provider` 字段即为其署名（实测 `{"name":"中国天气网"}`）。

  GET https://www.msn.cn/zh-cn/weather/forecast/in-{lat},{lon}
      ?loc={base64url({"x":"<lon>","y":"<lat>"})}&weadegreetype=C&day={1..10}
  解析响应 HTML 中的 <script id="redux-data" type="application/json">
    → WeatherData["_@STATE@_"]["forecast"][i]["hourly"]

─────────────────────────────────────────────────────────────────────
关键契约（2026-09-05 线上实测验证；非官方契约，改版即可能失效，改动前必须复核）
─────────────────────────────────────────────────────────────────────
1. 无独立 JSON 接口：必须抓 HTML 页面再抽取 `redux-data` 脚本块。游客态可用，
   无需任何 Token/鉴权头。抽取失败即视为契约漂移并明确报错——解析不到状态就
   返回空快照会让"抓取成功"假象误导整个排行榜。
2. `loc` 参数决定实际定位：值为 URL-safe base64 的 `{"x":"<经度>","y":"<纬度>"}`
   —— **x 是经度、y 是纬度**（与直觉相反，写反会静默抓到另一个半球/省份）。
   路径段 `in-{lat},{lon}` 只是路由展示，服务端以 loc 为准；二者必须同源构造，
   否则 URL 与取数点不一致、且难以察觉。
3. `weadegreetype=C` 强制摄氏度。省略时存在返回华氏的风险——把华氏当摄氏入库
   是最危险的静默错误之一，故该参数固定携带，不做"单位自适应"。
4. **`day` 参数（本源最关键的约束）**：SSR 一次只渲染**一天**的逐小时。
   - `day=1`（或省略）→ 滚动 24h 窗口：forecast[0] 当前小时→23:00（部分）
     ＋ forecast[1] 00:00→当前小时（部分）；
   - `day=2..10` → 对应 forecast[1..9] 的完整 24 个整点；
   - `day>10` → 服务端静默回退为滚动窗口（实测 day=11 ≡ day=1），绝不可靠 day
     做越界扩展。
   因此**一份快照需要 10 次请求**，按整点去重合并后约 225 点（≈9.3 天）。
   各分片来自同一数据版本时可安全合并（实测 day=1,2,3 三次请求去重后恰为
   9+24+24=57 点，逐点一致），故合并策略是安全的。
5. **起报时刻**：接口不回显 NWP 模式起报轮次，只有 `lastUpdated`（数据更新时间，
   服务端缓存级——实测连续 3 次请求、以及间隔 75s 的两次请求均返回同一值，
   说明它是数据时间戳而非渲染时刻）。故

       issue := floor_hour(lastUpdated)

   语义与彩云（时间戳锚定请求时刻并下取整整点）一致：表示"我们观测到该预报的
   时刻"，符合本项目"预报先存档、事后与观测配对"的第一性原理。
   ⚠ 版本漂移：10 次请求跨越数秒，若期间 `lastUpdated` 推进，后面的分片属于
   **另一个数据版本**。核心不变量：**一份快照内绝不跨版本**——数值与 lead 必须
   同源。检出漂移时舍弃已收集的旧版本分片、以新版本**整体重抓**（限
   MAX_VERSION_RESTARTS=1 次；新版本刚上线、数秒内再次滚动的概率极低，实测
   版本约每 10 分钟推进一次而 10 个分片总耗时约 4 秒）。重抓仍漂移则中止并
   保残缺快照（起报快照错过即无法追补，残缺胜过零样本，但残缺必须可见）。
   绝不采取"保留旧分片 + 补新分片"——那正是跨版本混合。
6. **字段映射（易错点）**：
   - 温度 := `hourly[].temperature`（int，℃）。
   - 降水 := `hourly[].rainAmount`（mm）。
   - ❌ **绝不可使用 `precipitation` 字段**：它是**降水概率百分数**的字符串
     （实测 "10"/"14"/"17"），却与 `precipitationHeightUnit: "毫米"` 同现——
     单位标注具有误导性。误用它会把"17%"当成"17mm"入库，单条即可摧毁降水评分。
7. **降水口径自检（本源独有的强校验）**：`raAccu` 是**按自然日重置**的累计降水
   量，与 `rainAmount` 互为严格约束。实测（万宁 2026-09-07）：
       rainAmount 0.32 → raAccu 0.32（该日首个降水小时，累计重置为自身）
       rainAmount 0.36 → raAccu 0.68（0.32+0.36）
       rainAmount 0.29 → raAccu 0.97（0.68+0.29）
   这同时反证了 `rainAmount` 是"逐小时量"而非"自起报累计"（后者会单调增长）。
   本模块把"日内 raAccu 差分 == rainAmount"实现为守恒哨兵：不一致即告警，
   绝不静默入库。跨自然日边界不做差分（累计在那里重置）。
8. **降水窗口方向**：中国天气网未明示 `rainAmount@t` 覆盖 (t−1h, t] 还是
   (t, t+1h]。本项目按"前 1 小时累计"同标注直接透传（与彩云/和风/中科星图
   口径一致），与观测 `rain@t`=(t−1h,t] 对齐。待样本积累后按 README"降水口径"
   一节的 −1/0/+1h 平移标定法复核，若某平移显著且稳定占优再落在本模块并留档
   （同 accuweather 的 precip_alignment 处置原则）。
9. **空间语义：最近城市吸附**（与 AccuWeather 同类，非格点）：请求坐标被吸附
   到中国天气网的城市站点，回显在 `source.id`（如 101300603）、
   `source.location.Name`（如"万秀"）、`source.coordinates`。实测 4 站吸附距离
   约 1.7–7.3 km。本源得分代表"最近城市"，快照留档 `location_distance_km`
   与 `location_name` 供报告披露（绝不静默当作点预报）。
10. **时效上限**：225 点 ≈ 9.3 天，最大 lead 224h → 覆盖天桶 1..10，而评估配置
    `hourly_lead_days=16`。总榜"覆盖时效"列会自动显示 10d，跨源比较时需知情
    （短覆盖的源合并样本偏"易"，README 已披露该原理性混杂）。
11. **幂等**：issue 为整点，随 `lastUpdated` 按小时推进。同一小时内的重复抓取
   被 storage 的幂等键（站×模型×起报）跳过——语义正确：那是同一份滚动更新数据，
   且"起报快照错过即无法追补"的原则不允许覆盖已封存快照。
12. **分片失败即熔断**：10 个分片打的是同一端点、只差一个 `day` 查询参数，
   分片失败（退避重试穷尽后）是**整源级信号**——剩余分片必然以同样方式失败。
   逐片重试满退避 ≈ 9 × 21s ≈ 3 分钟/站（4 站约 12 分钟/轮）纯属浪费，且把
   "页面改版/被风控"这类确定性故障拖成超时。故首个分片失败后即中止剩余分片的
   抓取（与 accuweather 的配额熔断同款取舍）。代价：一次瞬时抖动会牺牲该轮
   剩余时效；收益：故障快速可见、CI 不被无谓拖长，下一轮（新 issue）自动补回。
"""
from __future__ import annotations

import base64
import json
import logging
import re
import time
from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from typing import Any
from urllib.parse import quote

import requests

from .base import ForecastProvider
from ..timeutil import BEIJING, floor_to_hour, parse_iso

logger = logging.getLogger(__name__)

BASE_URL = "https://www.msn.cn/zh-cn/weather/forecast"
HEADERS = {"User-Agent": "weather-api-eval/0.1 (+https://github.com/)"}
SOURCE = "msn"
MODEL_NAME = "msn_v1"

# day=1..10 有效（docstring 第 4 条）；>10 会被服务端静默回退为滚动窗口
MAX_DAY = 10

# SSR 内嵌状态块：整页无其它可取数通道，抽取失败即契约漂移
_REDUX_RE = re.compile(
    r'<script id="redux-data" type="application/json">(.*?)</script>', re.S | re.I
)

# 降水守恒哨兵容差：rainAmount 与 raAccu 均按 2 位小数回传，逐点舍入误差 ≤0.02，
# 留 0.05 以吸收浮点与边界；超出即视为口径/字段语义已变化。
_RAIN_TOLERANCE_MM = 0.05

# 版本漂移后整体重抓的次数上限（docstring 第 5 条）：1 次即能把"这一轮报废"
# 换成"这一轮拿到新版本完整序列"，再漂移就中止，避免无限重抓。
MAX_VERSION_RESTARTS = 1


class ReduxStateMissing(RuntimeError):
    """页面内找不到 redux-data 状态块（契约漂移的确定性信号，重试无意义）。"""


class _Deterministic(Exception):
    """确定性失败（重试无意义）：4xx 或页面结构异常。"""


# ------------------------------------------------------------------ 工具函数
def loc_param(lat: float, lon: float) -> str:
    """构造 loc 查询参数：URL-safe base64 的 {"x":"经度","y":"纬度"}。

    x=经度 / y=纬度 是本接口最容易写反的地方（docstring 第 2 条）；
    调用方只需传 (纬度, 经度)，映射在此集中完成，避免各调用点各自写反。
    """
    raw = json.dumps({"x": f"{lon}", "y": f"{lat}"}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii")


def forecast_url(lat: float, lon: float, day: int | None = None) -> str:
    """构造 MSN 预报页 URL（day 省略时服务端按滚动 24h 窗口渲染）。"""
    url = (f"{BASE_URL}/in-{lat},{lon}?loc={quote(loc_param(lat, lon), safe='')}"
           f"&weadegreetype=C")
    if day is not None:
        url += f"&day={int(day)}"
    return url


def _num_or_none(v: Any) -> float | None:
    """null/bool/NaN/非法串一律归 None，绝不把缺测伪装成 0 参与评估。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """haversine 大圆距离（km），量化"最近城市吸附"偏离站点的程度。"""
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return round(6371.0 * 2 * asin(sqrt(a)), 3)


def extract_state(html: str) -> dict:
    """从页面 HTML 中抽取 SSR 内嵌状态 → WeatherData 的 `_@STATE@_` 子树。

    抽取失败（页面改版/返回的是错误页/被风控拦截）抛 ReduxStateMissing ——
    这是确定性信号，重试不会改变结果，绝不能退化为空快照让上层误判"抓取成功"。
    """
    if not isinstance(html, str) or not html:
        raise ReduxStateMissing("MSN 响应为空或非文本")
    m = _REDUX_RE.search(html)
    if not m:
        raise ReduxStateMissing(
            f"MSN 页面中未找到 redux-data 状态块（页面结构可能已改版或被风控拦截），"
            f"响应前 200 字符: {html[:200]!r}"
        )
    try:
        blob = json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise ReduxStateMissing(
            f"MSN redux-data 状态块不是合法 JSON（契约漂移）: {e}"
        ) from e
    wd = (blob or {}).get("WeatherData")
    state = wd.get("_@STATE@_") if isinstance(wd, dict) else None
    if not isinstance(state, dict):
        raise ReduxStateMissing(
            "MSN redux-data 中缺少 WeatherData._@STATE@_（契约漂移）: "
            f"顶层键={list(blob)[:8] if isinstance(blob, dict) else type(blob).__name__}"
        )
    return state


def _parse_last_updated(s: Any) -> datetime | None:
    """`lastUpdated`（如 `2026-09-05T15:22:15+08:00`）→ 北京时 naive datetime。

    无偏移的异常形态按北京时墙钟处理（本土站点当地时即北京时）。
    解析失败返回 None，由调用方降级到时间轴首点（不因此丢弃整份快照）。
    """
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        dt = datetime.fromisoformat(s.strip())
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(BEIJING).replace(tzinfo=None)
    return dt


def _parse_entry_time(entry: dict) -> datetime | None:
    """逐小时条目 → 北京时 naive 整点。

    优先用 `time.dataValue`（UTC ISO，形态稳定，实测恒为 `...T07:00:00.000Z`）；
    缺失时回退 `timeStr`（当地时带偏移，实测两种形态混用：
    `2026-09-05T15:00:00.000+08:00` 与 `2026-09-05T20:00:00+08:00`，
    故不可按固定长度切片）。两者都不可用时该点丢弃（不猜时间）。
    """
    node = entry.get("time")
    if isinstance(node, dict):
        v = node.get("dataValue")
        if isinstance(v, str) and v.strip():
            try:
                dt = datetime.fromisoformat(v.strip().replace("Z", "+00:00"))
                return dt.astimezone(BEIJING).replace(tzinfo=None, minute=0,
                                                      second=0, microsecond=0)
            except ValueError:
                pass
    ts = entry.get("timeStr")
    if isinstance(ts, str) and ts.strip():
        try:
            dt = datetime.fromisoformat(ts.strip())
        except ValueError:
            return None
        if dt.tzinfo is not None:
            dt = dt.astimezone(BEIJING).replace(tzinfo=None)
        return dt.replace(minute=0, second=0, microsecond=0)
    return None


def iter_hourly(state: dict) -> list[tuple[datetime, dict]]:
    """展开 state 的全部逐小时条目 → [(北京时整点, 条目)]，按时间升序、整点去重。

    注意：不按 `day` 参数去猜"该取 forecast 的第几项"——SSR 对 day=1 会同时
    渲染 forecast[0] 与 forecast[1]，对 day=2..10 只渲染对应日。统一全量展开、
    按整点去重（保留首见）在两种形态下都正确，且不依赖 day→索引的映射约定。
    """
    out: dict[datetime, dict] = {}
    for day in state.get("forecast") or []:
        if not isinstance(day, dict):
            continue
        for entry in day.get("hourly") or []:
            if not isinstance(entry, dict):
                continue
            dt = _parse_entry_time(entry)
            if dt is None:
                continue
            out.setdefault(dt, entry)
    return sorted(out.items())


def check_rain_conservation(points: list[tuple[datetime, float | None, float | None]]
                            ) -> list[str]:
    """降水守恒哨兵：同一自然日内 `raAccu` 的逐时增量应等于当小时 `rainAmount`。

    `raAccu` 按自然日重置（docstring 第 7 条），故**只在日内做差分**，跨日边界
    重置为"首小时累计 == 自身"。返回不一致的描述列表（供调用方告警，限条数）。
    """
    problems: list[str] = []
    prev_day: str | None = None
    prev_accu: float | None = None
    for dt, rain, accu in points:
        day = dt.strftime("%Y-%m-%d")
        if day != prev_day:
            # 新的一天：累计重置，首点应满足 accu == rain
            if rain is not None and accu is not None and abs(accu - rain) > _RAIN_TOLERANCE_MM:
                problems.append(
                    f"{dt:%Y-%m-%dT%H:%M} 日累计重置点 raAccu={accu} != rainAmount={rain}")
            prev_day, prev_accu = day, accu
            continue
        if rain is not None and accu is not None and prev_accu is not None:
            delta = accu - prev_accu
            if abs(delta - rain) > _RAIN_TOLERANCE_MM:
                problems.append(
                    f"{dt:%Y-%m-%dT%H:%M} raAccu 增量 {delta:.2f} != rainAmount {rain:.2f}")
        # 缺测点直接把基准置空：宁可让下一点无法校验，也不跨过缺口去差分
        # （跨缺口差分会产出假告警，而假告警比不校验更伤——它会训练人忽略告警）
        prev_accu = accu
    return problems


# ------------------------------------------------------------------ Provider
class MsnProvider(ForecastProvider):
    """MSN 天气（中国天气网）逐小时预报快照器：无需凭据，单模型快照 dict。"""

    def __init__(
        self,
        timeout: int | tuple = (10, 30),
        retries: int = 3,
        session: requests.Session | None = None,
        max_day: int = MAX_DAY,
    ):
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        # day>10 服务端静默回退为滚动窗口（docstring 第 4 条），上界不可越过
        self.max_day = max(1, min(int(max_day), MAX_DAY))

    # ------------------------------------------------------------------ 对外
    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> dict:
        merged: dict[datetime, dict] = {}
        base_lu: str | None = None
        state0: dict | None = None
        # dropped = 最终快照实际缺失的分片；drift_history = 版本漂移审计。
        # 两者必须分开：整体重抓成功后快照并不残缺，但漂移史仍需留档——
        # 若混用，重抓成功也会打出"快照残缺"的自相矛盾日志并污染审计字段。
        dropped: list[str] = []
        drift_history: list[str] = []
        fetched = 0
        restarts = 0
        day = 1
        while day <= self.max_day:
            try:
                state = self._fetch_day(station, day)
            except Exception as e:  # noqa: BLE001
                # 熔断（docstring 第 12 条）：该分片已跑满自身退避仍失败，而 10 个
                # 分片打的是同一端点、只差一个 day 参数——剩余分片必然同样失败。
                # 继续逐片重试只会把确定性故障拖成超时（≈3 分钟/站）。
                logger.error("MSN 站点 %s day=%d 抓取失败（已重试 %d 次），"
                             "中止该站剩余分片: %s", station.id, day, self.retries, e)
                dropped.append(f"day={day}: {e}")
                break
            lu = state.get("lastUpdated")
            if base_lu is None:
                base_lu, state0 = lu, state
            elif lu != base_lu:
                # 版本漂移（docstring 第 5 条）：核心不变量是一份快照不跨版本。
                # 绝不"保留旧分片 + 补新分片"——那正是跨版本混合。
                note = f"day={day}: 数据版本漂移 {lu} != {base_lu}"
                drift_history.append(note)
                if restarts < MAX_VERSION_RESTARTS:
                    restarts += 1
                    logger.warning(
                        "MSN 站点 %s day=%d 数据版本漂移（%s -> %s），舍弃旧版本分片"
                        "并以新版本整体重抓（第 %d/%d 次）",
                        station.id, day, base_lu, lu, restarts, MAX_VERSION_RESTARTS)
                    merged, fetched, dropped = {}, 0, []
                    base_lu, state0 = lu, state
                    day = 1
                    continue
                logger.warning("MSN 站点 %s day=%d 数据版本再次漂移（%s != %s），已达"
                               "重抓上限，中止该站剩余分片", station.id, day, lu, base_lu)
                dropped.append(note)
                break
            fetched += 1
            before = len(merged)
            for dt, entry in iter_hourly(state):
                merged.setdefault(dt, entry)
            logger.debug("MSN 站点 %s day=%d 新增 %d 个整点",
                         station.id, day, len(merged) - before)
            day += 1

        if not merged:
            raise RuntimeError(
                f"MSN 站点 {station.id} 未能取得任何逐小时预报（{self.max_day} 个分片全部失败）: "
                f"{dropped or drift_history or '无具体错误'}"
            )
        if fetched < self.max_day:
            # 残缺入库是显式决策：起报快照错过即无法追补，部分时效胜过零样本；
            # 但必须让残缺可见（日志 + 快照字段），否则榜单的"覆盖时效"会被误读。
            logger.warning("MSN 站点 %s 快照残缺：%d/%d 个分片可用，丢弃项=%s",
                           station.id, fetched, self.max_day, dropped)

        points = sorted(merged.items())
        temps: list[float | None] = []
        precips: list[float | None] = []
        accu_triplet: list[tuple[datetime, float | None, float | None]] = []
        for dt, entry in points:
            temps.append(_num_or_none(entry.get("temperature")))
            rain = _num_or_none(entry.get("rainAmount"))
            precips.append(rain)
            accu_triplet.append((dt, rain, _num_or_none(entry.get("raAccu"))))

        # ---- 起报时刻：floor_hour(lastUpdated)（docstring 第 5 条）
        issue_dt: datetime | None = None
        lu_dt = _parse_last_updated(base_lu)
        if lu_dt is not None:
            issue_dt = floor_to_hour(lu_dt)
        first_dt = points[0][0]
        if issue_dt is None:
            # lastUpdated 缺失/不可解析：降级为时间轴首点（仍可用，lead 从 0 起算），
            # 但必须告警——起报锚点退化会让"覆盖时效"的语义变松。
            logger.warning("MSN 站点 %s 响应缺少可解析的 lastUpdated，退化以时间轴首点"
                           "作为起报时刻", station.id)
            issue_dt = first_dt
        elif abs((first_dt - issue_dt).total_seconds()) > 3600:
            # 哨兵：正常契约下序列首点应当就是起报整点（或紧邻的下一整点）。
            # 偏离 >1h 说明时间语义已变化，lead 分组会整体错位，必须暴露。
            logger.warning(
                "MSN 站点 %s 时间轴首点 %s 与起报时刻 %s（lastUpdated=%s）偏离超过 1 小时，"
                "服务端时间契约可能已变化，lead 分组可能整体错位",
                station.id, first_dt.strftime("%Y-%m-%dT%H:%M"),
                issue_dt.strftime("%Y-%m-%dT%H:%M"), base_lu,
            )

        hourly_time = [dt.strftime("%Y-%m-%dT%H:%M") for dt, _ in points]
        issue_iso = issue_dt.strftime("%Y-%m-%dT%H:%M")

        # ---- 质量哨兵（全部只告警，不静默也不阻断）
        if all(v is None for v in temps):
            logger.warning("MSN 站点 %s 温度序列全部缺测，服务端契约可能已变化", station.id)
        if all(v is None for v in precips):
            logger.warning("MSN 站点 %s 降水序列全部缺测（rainAmount 恒为 null/0），"
                           "本快照降水将计为缺测", station.id)
        problems = check_rain_conservation(accu_triplet)
        if problems:
            logger.warning(
                "MSN 站点 %s 有 %d/%d 个整点不满足「raAccu 增量 == rainAmount」"
                "（容差 %.2fmm），降水字段语义可能已变化；前 3 例: %s",
                station.id, len(problems), len(accu_triplet), _RAIN_TOLERANCE_MM,
                problems[:3],
            )
        # 时效覆盖：本源上限约 9.3 天，短于配置的 16 天属预期，但被截断要可见
        expected = 24 * self.max_day
        if len(hourly_time) < expected // 2:
            logger.warning("MSN 站点 %s 仅合并出 %d 个整点（预期约 %d），长时效可能被"
                           "服务端截断或分片大量失败", station.id,
                           len(hourly_time), expected)

        # ---- 吸附元数据（docstring 第 9 条：最近城市语义，绝不当作点预报）
        src = (state0 or {}).get("source") or {}
        coords = src.get("coordinates") or {}
        g_lat = _num_or_none(coords.get("lat"))
        g_lon = _num_or_none(coords.get("lon"))
        if g_lat is None or g_lon is None:
            logger.warning("MSN 站点 %s 响应缺少 source.coordinates，以请求坐标留档",
                           station.id)
            g_lat, g_lon = float(station.lat), float(station.lon)
        loc_info = src.get("location") or {}
        provider = (state0 or {}).get("provider") or {}

        snapshot = {
            "issue_iso": issue_iso,
            "station_id": station.id,
            "source": SOURCE,
            "models": [MODEL_NAME],
            # 吸附后的城市站点坐标（非站点点位）
            "grid_lat": g_lat,
            "grid_lon": g_lon,
            "elevation": None,  # 该接口无海拔字段
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            "location_id": src.get("id"),
            "location_name": loc_info.get("Name"),
            "location_distance_km": _distance_km(
                float(station.lat), float(station.lon), g_lat, g_lon),
            "provider_name": provider.get("name"),
            "provider_url": provider.get("url"),
            "last_updated": base_lu,         # 服务端数据更新时间（起报锚点来源）
            "days_fetched": fetched,         # 实际合并的分片数（残缺时 < max_day）
            "dropped_days": dropped,         # 最终快照实际缺失的分片，供事后审计
            "drift_history": drift_history,  # 版本漂移史（重抓成功时快照并不残缺）
            "version_restarts": restarts,    # 整体重抓次数
            "precip_alignment": (
                "rainAmount@t 的累计窗口方向中国天气网未明示；按「前 1 小时累计」"
                "同标注直接透传，与观测 rain@t=(t-1h,t] 对齐。已用 raAccu（按自然日"
                "重置的累计降水）做守恒校验。待样本积累后按 README「降水口径」的 "
                "-1/0/+1h 平移标定法复核。详见 forecast/msn.py docstring 第 8 条。"
            ),
            "hourly_time": hourly_time,
            "data": {MODEL_NAME: {
                "temperature_2m": temps,
                "precipitation": precips,
            }},
        }
        logger.info("站点 %s 已抓取 MSN 起报 %s（%d 个整点，分片 %d/%d，城市 %s，"
                    "距站点 %.1f km）", station.id, issue_iso, len(hourly_time),
                    fetched, self.max_day, snapshot["location_name"],
                    snapshot["location_distance_km"])
        return snapshot

    # ------------------------------------------------------------------ 内部
    def _fetch_day(self, station: Any, day: int) -> dict:
        """抓取并解析某一天分片对应的 SSR 状态。"""
        url = forecast_url(station.lat, station.lon, day)
        html = self._request(url)
        return extract_state(html)

    def _request(self, url: str) -> str:
        """带退避地取回页面文本。

        4xx（除 429）是确定性失败：重试不改变结果，立即上抛以免烧满退避。
        注意：本方法只负责"取回文本"，状态块解析在 `_fetch_day` 里——页面结构
        漂移（ReduxStateMissing）因此只会消耗 1 次请求就触发上层熔断，不会
        被当成网络错误重试。
        """
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            resp = None
            try:
                resp = self.session.get(url, headers=HEADERS, timeout=self.timeout)
                status = getattr(resp, "status_code", None)
                if status == 200:
                    try:
                        # 服务端实测返回 charset=utf-8；缺失时 requests 会退回
                        # chardet 猜测，而本源整条解析链依赖中文正确解码，
                        # 故显式钉死 UTF-8，不让编码猜测成为静默失败源。
                        if not getattr(resp, "encoding", None):
                            resp.encoding = "utf-8"
                        return resp.text
                    except Exception as e:  # noqa: BLE001
                        raise _Deterministic(f"响应体读取失败: {e}") from e
                digest = ""
                try:
                    digest = (resp.text or "")[:200]
                except Exception:  # noqa: BLE001
                    pass
                if isinstance(status, int) and 400 <= status < 500 and status != 429:
                    raise _Deterministic(f"HTTP {status} body={digest!r}")
                last_err = RuntimeError(f"HTTP {status} body={digest!r}")
            except _Deterministic:
                raise
            except Exception as e:  # noqa: BLE001  网络类异常/5xx/429 → 可重试
                last_err = e
            logger.warning("MSN 请求失败（第%d次）: %s", attempt + 1, last_err)
            if attempt < self.retries:
                time.sleep(min(30, 3 * 2 ** attempt))
        raise RuntimeError(f"MSN 请求最终失败: {last_err}") from last_err

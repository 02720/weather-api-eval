"""AccuWeather 逐小时预报快照器。

数据源：AccuWeather Forecast API v1（官方开放 API，Key 见 developer.accuweather.com）：
  GET https://dataservice.accuweather.com/forecasts/v1/hourly/{hours}hour/{locationKey}
      ?language=zh-cn&details=true&metric=true   （Key 走 Authorization: Bearer 头）

─────────────────────────────────────────────────────────────────────
端点与契约（第一性原理，关键设计约束，2026-08 核对官方文档）
─────────────────────────────────────────────────────────────────────
1. 定位是"最近城市吸附"而非格点：AccuWeather 不支持按经纬度点播，必须先用
   Locations API 把站点坐标解析为最近城市的 locationKey：
     GET /locations/v1/cities/geoposition/search?q={lat},{lon}   （Key 走头）
   q 为"纬度,经度"（注意与和风 v7 的"经度,纬度"相反，不可混用）。解析出的
   城市坐标/海拔/Key 全部留档（grid_lat/grid_lon/location_*），并用 haversine
   计算吸附距离 location_distance_km——该源的得分代表"最近城市"，与站点
   点位可能相距数十公里，解读时必须知情。
2. 时效档位 1/12/24/48/72/120 小时，订阅档位决定可用上限：请求超出订阅的
   档位返回 403/400。按 120→72→48→24→12→1 逐级回退（geovis 同款）。档位是
   账号级属性：首次成功后进程内缓存、跨站点复用；不做跨运行持久化——免费档
   50 次/天下每轮重探最多浪费 3 次调用（总用量约 33 次/天，仍宽裕），换来
   双向自愈：瞬时 403 降级不会被 git 自动提交的状态文件永久封顶时效。
   实际使用的档位记入快照 tier 字段，实际拿到的点数记入 hours 字段。
3. 配额：免费档 50 次/天，超限返回 503（偶发 403）。本源每站每轮 2 次调用
   （1 定位 + 1 预报），4 站 × 3 轮/天 ≈ 24 次，档位探测另计约 9 次。503 按
   瞬时过载退避重试，穷尽后置"配额熔断"标志；档位梯子全被拒同样熔断——两种
   账号级确定性失败都让同次运行内后续站点直接失败、不再烧退避/重探（若其实
   是瞬时故障，下次运行自愈）。最终错误信息写明配额账本。
4. 鉴权：官方认证文档（2026-06-10 修订版）改为 Bearer 头——每个请求必须带
   Authorization: Bearer {key}；旧版"?apikey="query 参数契约已停用（2026-08-29
   实测：有效 Key 走 query 也一律 401，4 站全挂）。Key 不再进 URL，但掩码防御
   保留——底层异常/服务端回显仍可能外带凭据，日志/异常消息一律先经 _masked。
5. 时间：DateTime 为 ISO8601 带当地时区偏移（如 2026-08-28T15:00:00+08:00；
   文档亦给出 +08 两数字形态，fromisoformat 均可直接解析），astimezone 转北京
   时并下取整整点（同彩云/和风口径）；无偏移的异常形态按北京时墙钟兜底并
   WARNING（本土站点当地时即北京时）。
6. 数值：details=true 才返回 TotalLiquid（该小时液态降水总量），metric=true
   时 ℃/mm。响应自带 Unit 字段——按单位自适应换算（F→℃、inch→mm），未
   知单位 → None + WARNING；Unit 键缺失的异常形态按公制处理并 WARNING
   （fuxi_data 同款取舍）。绝不静默把华氏度当摄氏度入库。
7. 降水移位（承重假设，官方文档未明示区间方向）：判定 TotalLiquid@t 覆盖
   (t, t+1h]（小时段起点在 t）。依据：① "1hour" 产品若描述的是刚过去的一
   小时便不成其为预报；② 官网小时块把"当前小时"条目展示为正在发生的未来
   降水；③ 观测类字段显式命名 PastHour，预报字段无此标注。而观测 rain@t 的
   口径是 (t−1h, t]，故快照 precipitation@t := TotalLiquid@(t−1h)——把
   TotalLiquid 序列整体后移 1 小时入库（移位由 liquid_slot 映射按"整点键"
   实现，序列缺口不会把上一窗的雨错配到下一小时）。代价：首个时效点的降水
   无对应窗记 None，末点的降水被丢弃。若日后发现该源降水相对观测系统性滞后
   1 小时（整体前移 1h 反而更准时），应优先复核此假设。快照 precip_alignment
   字段留档，与风乌 expansion 字段同例。
8. issue_iso 取时间轴首点（评估引擎排除 lead=0），与 qweather/geovis 一致。
"""
from __future__ import annotations

import logging
import os
import time
import unicodedata
from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from typing import Any

import requests

from .base import ForecastProvider
from ..timeutil import BEIJING

logger = logging.getLogger(__name__)

BASE_URL = "https://dataservice.accuweather.com"
LOCATION_URL = f"{BASE_URL}/locations/v1/cities/geoposition/search"
FORECAST_URL = f"{BASE_URL}/forecasts/v1/hourly/{{hours}}hour/{{key}}"

SOURCE = "accuweather"
MODEL_NAME = "accuweather_v1"
KEY_ENV = "ACCUWEATHER_API_KEY"

# 官方逐小时档位（小时），从大到小用于订阅档位回退
TIERS = (120, 72, 48, 24, 12, 1)
DEFAULT_HOURS = 120
HEADERS = {"User-Agent": "weather-api-eval/0.1 (+https://github.com/)"}

# 单位自适应：AccuWeather 数值对象自带 Unit。温度目标 ℃、降水目标 mm；
# 键为响应 Unit 的小写形态，值为 (乘数, 加数) 线性换算。
_TEMP_TO_C = {"c": (1.0, 0.0), "f": (5.0 / 9.0, -32.0 * 5.0 / 9.0)}
_LIQUID_TO_MM = {"mm": (1.0, 0.0), "in": (25.4, 0.0), "inch": (25.4, 0.0)}


class _QuotaHint(Exception):
    """503 穷尽重试后携带配额说明的失败（内部占位，最终转 RuntimeError）。"""


def _tier_for(hours: int) -> int:
    """把请求时效吸附到官方档位：取不超过 hours 的最大档，无则取最小档 1。"""
    usable = [t for t in TIERS if t <= hours]
    return max(usable) if usable else min(TIERS)


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


def _metric_value(obj: Any, table: dict[str, tuple[float, float]]) -> tuple[float | None, str]:
    """把 {'Value': x, 'Unit': u} 量纲对象按表换算，返回 (值, 单位键)。

    已知单位线性换算；Unit 键缺失的异常形态按目标单位处理（fuxi_data 同款
    取舍：保数据 + 由调用方聚合告警）；未知单位绝不猜测，值归 None。
    单位键交给调用方聚合计数（逐条目告警会刷屏），Value 为 null 时不计单位。
    """
    if obj is None or not isinstance(obj, dict):
        return None, ""
    v = _num_or_none(obj.get("Value"))
    if v is None:
        return None, ""
    unit = obj.get("Unit")
    key = str(unit).strip().lower() if unit is not None else ""
    if not key:
        # Unit 键缺失：按目标单位处理（fuxi_data 同款取舍），"" 由调用方聚合告警
        return v, ""
    conv = table.get(key)
    if conv is None:
        return None, key
    mul, add = conv
    return mul * v + add, key


def _warn_units(units: dict[str, int], table: dict[str, tuple[float, float]],
                expect: str, what: str, station_id: str) -> None:
    """按（单位, 条目数）聚合告警：未声明单位/未知单位/非公制单位各一条。"""
    for u, n in sorted(units.items()):
        if u == expect:
            continue
        if u == "":
            logger.warning("AccuWeather 站点 %s 有 %d 条 %s 条目未声明单位，按 %s 处理"
                           "（若与实际不符请检查契约）", station_id, n, what, expect)
        elif u not in table:
            logger.warning("AccuWeather 站点 %s 有 %d 条 %s 条目单位 %r 未知，已按缺测处理",
                           station_id, n, what, u)
        else:
            logger.warning("AccuWeather 站点 %s 有 %d 条 %s 条目返回非公制单位 %r，已换算为 %s",
                           station_id, n, what, u, expect)


def _parse_dt(s: Any) -> datetime:
    """ISO8601（带偏移）→ 北京时 naive 并下取整整点。

    兼容 "+08:00"/"+08"/"Z" 等形态（3.12 fromisoformat 原生支持）；
    无偏移的异常形态视为已是北京时墙钟（本土站点当地时即北京时）并 WARNING。
    """
    if not isinstance(s, str) or not s.strip():
        raise ValueError(f"非法时间戳: {s!r}")
    try:
        dt = datetime.fromisoformat(s.strip())
    except ValueError as e:
        raise ValueError(f"无法解析 AccuWeather 时间戳: {s!r}") from e
    if dt.tzinfo is not None:
        dt = dt.astimezone(BEIJING).replace(tzinfo=None)
    else:
        logger.warning("AccuWeather 时间戳无时区偏移（%r），按北京时墙钟处理", s)
    return dt.replace(minute=0, second=0, microsecond=0)


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """haversine 大圆距离（km），用于量化"最近城市吸附"偏离站点的程度。"""
    p1, p2 = radians(lat1), radians(lat2)
    dp, dl = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dp / 2) ** 2 + cos(p1) * cos(p2) * sin(dl / 2) ** 2
    return round(6371.0 * 2 * asin(sqrt(a)), 3)


def _body_digest(body: Any) -> str:
    """响应体摘要（定长），用于错误信息定位契约漂移。"""
    if body is None:
        return "<非 JSON>"
    if isinstance(body, (dict, list)):
        return repr(body)[:200]
    return repr(str(body)[:200])


def _quota_hint(detail: str) -> str:
    return (
        f"AccuWeather 请求持续失败（{detail}）。最常见原因是免费档 50 次/天配额耗尽"
        "（本源每站每轮 2 次调用：1 定位 + 1 预报，4 站 × 3 轮/天 ≈ 24 次，档位探测"
        "与手动测试另计），也可能是服务端过载；请前往 developer.accuweather.com "
        "控制台核对当日用量。"
    )


def parse_hourly_payload(payload: Any, station_id: str,
                         requested_hours: int) -> dict[str, list]:
    """解析逐小时预报响应 → {"time", "temperature_2m", "precipitation"}。

    - 时间轴取温度条目的整点集合（升序、去重保留首见、按请求档位截断）；
    - 降水按 docstring 第 7 条整体后移 1 小时入库：precipitation@t =
      TotalLiquid@(t−1h)，用"整点键映射"实现，序列缺口不会错配邻窗降水；
    - 无有效条目、条目数为 0 时抛错（空快照入库会被同 issue 幂等锁死）；
    - details=true 未生效（响应无 TotalLiquid 键）、温度/降水全缺测、点数少于
      档位等降级均以 WARNING 暴露，绝不静默。
    """
    if not isinstance(payload, list) or not payload:
        raise RuntimeError(f"AccuWeather 站点 {station_id} 响应为空或非数组: {_body_digest(payload)}")

    temp_map: dict[datetime, float | None] = {}
    liquid_slot: dict[datetime, float | None] = {}
    temp_units: dict[str, int] = {}
    liquid_units: dict[str, int] = {}
    total_liquid_keys = 0
    for ent in payload:
        if not isinstance(ent, dict):
            continue
        try:
            dt = _parse_dt(ent.get("DateTime"))
        except ValueError:
            logger.warning("AccuWeather 站点 %s 出现无法解析的 DateTime=%r，跳过",
                           station_id, ent.get("DateTime"))
            continue
        if dt in temp_map:  # 重复整点保留首见
            continue
        tv, tkey = _metric_value(ent.get("Temperature"), _TEMP_TO_C)
        temp_map[dt] = tv
        # 只有真实解释过数值（换算/假设/拒绝）才计入单位统计：
        # Value 为 null 的缺测条目不参与，避免误报"未声明单位"
        if tv is not None or tkey:
            temp_units[tkey] = temp_units.get(tkey, 0) + 1
        lv, lkey = _metric_value(ent.get("TotalLiquid"), _LIQUID_TO_MM)
        liquid_slot[dt + timedelta(hours=1)] = lv
        if lv is not None or lkey:
            liquid_units[lkey] = liquid_units.get(lkey, 0) + 1
        if "TotalLiquid" in ent:
            total_liquid_keys += 1

    if not temp_map:
        raise RuntimeError(f"AccuWeather 站点 {station_id} 未解析出任何有效逐小时条目")
    _warn_units(temp_units, _TEMP_TO_C, "c", "气温", station_id)
    _warn_units(liquid_units, _LIQUID_TO_MM, "mm", "降水", station_id)

    if total_liquid_keys == 0:
        logger.warning("AccuWeather 站点 %s 响应无 TotalLiquid 字段（details=true 疑未生效），"
                       "本快照降水将全缺测", station_id)

    times = sorted(temp_map)
    if len(times) > requested_hours:
        times = times[:requested_hours]
    elif len(times) < requested_hours:
        logger.warning("AccuWeather 站点 %s 仅返回 %d 个逐小时点（档位 %dh）。订阅档位或"
                       "服务端截断可能限制了时效，本次评估对该源的有效时效约 %dh。",
                       station_id, len(times), requested_hours, len(times))

    temps = [temp_map[dt] for dt in times]
    precips = [liquid_slot.get(dt) for dt in times]
    if all(v is None for v in temps):
        logger.warning("AccuWeather 站点 %s 温度序列全部缺测，服务端契约可能已变化", station_id)
    if total_liquid_keys and all(v is None for v in precips):
        logger.warning("AccuWeather 站点 %s 降水序列全部缺测（TotalLiquid 值均为 null），"
                       "本快照降水将计为缺测", station_id)

    return {
        "time": [dt.strftime("%Y-%m-%dT%H:%M") for dt in times],
        "temperature_2m": temps,
        "precipitation": precips,
    }


class AccuWeatherProvider(ForecastProvider):
    """AccuWeather 逐小时预报快照器：需 ACCUWEATHER_API_KEY，单模型快照 dict。"""

    def __init__(self, api_key: str | None = None, hours: int = DEFAULT_HOURS,
                 language: str = "zh-cn", timeout: int | tuple = (10, 60),
                 retries: int = 3, session: requests.Session | None = None):
        self.key = api_key if api_key is not None else os.environ.get(KEY_ENV)
        if self.key:
            # CI Secret/本地 .env 常携带首尾空白或换行——剥除后再进 Authorization 头
            self.key = self.key.strip()
            if any(unicodedata.category(c) == "Cc" for c in self.key):
                # Cc 覆盖 C0/DEL/C1 全部控制码位，非法字符绝不入 Bearer 头
                raise RuntimeError(
                    f"AccuWeather API Key 含非法控制字符（经 {KEY_ENV} 注入），"
                    "请检查凭据来源是否被污染"
                )
        if not self.key:
            raise RuntimeError(
                f"AccuWeather API Key 未提供：请在 developer.accuweather.com 创建应用获取，"
                f"经环境变量 {KEY_ENV} 注入，或在构造 AccuWeatherProvider 时显式传入 api_key。"
            )
        self.hours = _tier_for(hours)  # 请求时效吸附到官方档位
        self.language = language
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self._headers = {
            **HEADERS,
            # 官方 2026-06-10 修订契约：Key 经 Authorization: Bearer 头传递
            # （旧版 query 参数已停用），并按文档要求显式声明 Accept-Encoding
            "Accept-Encoding": "gzip, deflate",
            "Authorization": f"Bearer {self.key}",
        }
        self._tier_cache: int | None = None   # 本进程已验证的可用档位（账号级，跨站点复用）
        self._quota_suspect = False           # 503 穷尽重试后的熔断标志
        self._tiers_exhausted = False         # 档位梯子全被拒后的熔断标志

    # ------------------------------------------------------------------ 对外
    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> dict:
        if self._quota_suspect:
            # 同一次运行内 503 已穷尽过退避：后续站点快速失败，不烧配额与时间
            raise RuntimeError(_quota_hint("熔断：本次运行已有请求持续 503"))
        if self._tiers_exhausted:
            # 档位梯子全被拒（订阅未开放逐小时预报或配额以 403 形态耗尽）：
            # 属账号级确定性失败，后续站点不再重复 6 连探测
            raise RuntimeError(
                "AccuWeather 熔断：本次运行内预报各档位均被拒（订阅未开放逐小时预报"
                "或配额耗尽），不再对后续站点重复尝试")
        loc = self._resolve_location(station)
        tier, payload = self._forecast_any_tier(loc["key"], station.id)
        parsed = parse_hourly_payload(payload, station.id, tier)
        snapshot = {
            "issue_iso": parsed["time"][0],
            "station_id": station.id,
            "source": SOURCE,
            "models": [MODEL_NAME],
            # "最近城市吸附"语义：grid_* 是解析到的城市坐标而非站点点位
            "grid_lat": loc["lat"],
            "grid_lon": loc["lon"],
            "elevation": loc["elevation"],
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            "location_key": loc["key"],
            "location_name": loc["name"],
            "location_distance_km": _distance_km(
                float(station.lat), float(station.lon), loc["lat"], loc["lon"]),
            "tier": tier,                     # 实际使用的时效档位（订阅决定）
            "hours": len(parsed["time"]),     # 实际拿到的时间点数（被截断时 < tier）
            "precip_alignment": (
                "TotalLiquid@t 官方口径推断为 (t, t+1h]（小时段起点在 t）；"
                "快照 precipitation@t := TotalLiquid@(t-1h) 整体后移 1h，与观测 "
                "rain@t=(t-1h,t] 对齐。详见 forecast/accuweather.py docstring 第 7 条。"
            ),
            "hourly_time": parsed["time"],
            "data": {MODEL_NAME: {
                "temperature_2m": parsed["temperature_2m"],
                "precipitation": parsed["precipitation"],
            }},
        }
        logger.info("站点 %s 已抓取 AccuWeather 起报 %s（档位 %dh，实际 %d 点，城市 %s，"
                    "距站点 %.1f km）", station.id, parsed["time"][0], tier,
                    len(parsed["time"]), loc["name"], snapshot["location_distance_km"])
        return snapshot

    # ------------------------------------------------------------------ 内部
    def _resolve_location(self, station: Any) -> dict:
        """站点坐标 → 最近城市的 locationKey 与元数据（q 为"纬度,经度"）。"""
        status, body = self._get(LOCATION_URL, {
            "q": f"{station.lat},{station.lon}",
            "language": self.language,
        })
        if status == 401:
            raise RuntimeError(
                f"AccuWeather 定位鉴权失败（HTTP 401）：Key 已按官方现行契约经 "
                f"Authorization: Bearer 头传递仍被拒，请到 developer.accuweather.com "
                f"核对 {KEY_ENV} 是否有效/启用"
            )
        if status != 200 or not isinstance(body, dict):
            hint = ("响应结构异常（应为单个 location 对象，疑似契约漂移）" if status == 200
                    else "常见原因：配额耗尽/无权限（403）、坐标无匹配城市（404）")
            raise RuntimeError(
                f"AccuWeather 站点 {station.id} 定位查询失败（HTTP {status}，{hint}）: "
                f"{_masked(_body_digest(body), self.key)}"
            )
        key = body.get("Key")
        if not key:
            raise RuntimeError(
                f"AccuWeather 站点 {station.id} 定位响应缺少 locationKey（疑似契约漂移）: "
                f"{_masked(_body_digest(body), self.key)}"
            )
        geo = body.get("GeoPosition") or {}
        lat, lon = _num_or_none(geo.get("Latitude")), _num_or_none(geo.get("Longitude"))
        if lat is None or lon is None:
            logger.warning("AccuWeather 站点 %s 定位响应缺 GeoPosition，以请求坐标留档", station.id)
            lat, lon = float(station.lat), float(station.lon)
        elevation = _num_or_none(((geo.get("Elevation") or {}).get("Metric") or {}).get("Value"))
        return {
            "key": str(key),
            "name": body.get("LocalizedName") or body.get("EnglishName") or str(key),
            "lat": lat,
            "lon": lon,
            "elevation": elevation,
        }

    def _forecast_any_tier(self, loc_key: str, station_id: str) -> tuple[int, Any]:
        """按档位梯子请求逐小时预报：403/400 逐级下探，成功档位进程内缓存（跨站点复用）。

        无跨运行持久化：免费档 50 次/天下，每轮从默认档重探最多浪费 3 次调用
        （4 站 × 3 轮/天 ≈ 33 次，含探测），换来双向自愈——瞬时 403 降级不会像
        状态文件那样被 git 自动提交永久封顶时效。
        """
        tier = self._tier_cache or self.hours
        rejected: list[str] = []
        while True:
            status, body = self._get(
                FORECAST_URL.format(hours=tier, key=loc_key),
                {"language": self.language, "details": "true", "metric": "true"},
            )
            if status == 200:
                self._tier_cache = tier
                return tier, body
            if status == 401:
                raise RuntimeError(
                    f"AccuWeather 鉴权失败（HTTP 401）：Key 已按官方现行契约经 "
                    f"Authorization: Bearer 头传递仍被拒，请核对 {KEY_ENV} 是否有效/启用"
                )
            if status in (400, 403):
                rejected.append(f"{tier}h:HTTP {status} {_masked(_body_digest(body), self.key)}")
                smaller = [t for t in TIERS if t < tier]
                if not smaller:
                    self._tiers_exhausted = True
                    raise RuntimeError(
                        "AccuWeather 预报各档位均被拒绝（订阅未开放逐小时预报或配额耗尽）"
                        f"，尝试记录: {rejected}"
                    )
                logger.warning("AccuWeather 站点 %s 预报档位 %dh 被拒（HTTP %s），降级尝试 %dh",
                               station_id, tier, status, smaller[0])
                tier = smaller[0]
                continue
            raise RuntimeError(
                f"AccuWeather 站点 {station_id} 预报请求失败（HTTP {status}）: "
                f"{_masked(_body_digest(body), self.key)}"
            )

    def _get(self, url: str, params: dict) -> tuple[int | None, Any]:
        """请求一个端点。200 或确定性 4xx（除 429）返回 (status, body)；
        网络错误/5xx/429 指数退避重试；503 穷尽后置熔断标志并携带配额说明抛错。"""
        last_err: Exception | None = None
        last_status: int | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(url, params=params, headers=self._headers,
                                        timeout=self.timeout)
                status = getattr(resp, "status_code", None)
                try:
                    body = resp.json()
                except ValueError:
                    body = (getattr(resp, "text", "") or "")[:200]
                if status == 200:
                    return status, body
                if isinstance(status, int) and 400 <= status < 500 and status != 429:
                    # 鉴权/权限/档位类确定性错误：重试不会改变结果，交上层判定
                    return status, body
                last_status = status
                last_err = RuntimeError(f"HTTP {status} {_masked(_body_digest(body), self.key)}")
            except Exception as e:  # noqa: BLE001  网络类异常 → 可重试
                last_status = None
                # 归一为脱敏后的 RuntimeError：最终 raise 挂 __cause__ 链时，
                # 原始异常消息（可能含服务端回显的凭据）不会绕过掩码外泄
                last_err = RuntimeError(_masked(str(e), self.key))
            logger.warning("AccuWeather 请求失败（第%d次）: %s", attempt + 1,
                           _masked(str(last_err), self.key))
            if attempt < self.retries:
                time.sleep(min(30, 3 * 2 ** attempt))
        if last_status == 503:
            self._quota_suspect = True
            raise RuntimeError(_quota_hint("HTTP 503 已退避重试穷尽")) from last_err
        raise RuntimeError(f"AccuWeather 请求最终失败: {_masked(str(last_err), self.key)}") from last_err


def _masked(text: str, key: str) -> str:
    """Key 已走 Authorization 头（官方 2026-06-10 修订契约），URL 不再携带凭据；
    但底层异常/服务端回显仍可能外带 Key——入日志前一律掩码，防御纵深。"""
    if key:
        text = text.replace(key, "***")
    return text

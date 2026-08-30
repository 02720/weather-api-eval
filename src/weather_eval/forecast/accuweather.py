"""AccuWeather 逐小时预报快照器。

数据源：AccuWeather Enterprise API（官方文档 apidev.accuweather.com，2026-08-29 核对）：
  GET https://api.accuweather.com/forecasts/v1/hourly/{hours}hour/{locationKey}
      ?language=zh-cn&details=true&metric=true&apikey={key}

─────────────────────────────────────────────────────────────────────
端点与契约（第一性原理，关键设计约束，2026-08-29 迁移 Enterprise 时核对官方文档）
─────────────────────────────────────────────────────────────────────
0. 入口：2026-08-29 起走 Enterprise 入口，替代自助开发者入口 dataservice。
   官方 Overview 列双环境：生产 api.accuweather.com、开发 apidev.accuweather.com
   （构造参数 base_url 切换，默认生产）。host/鉴权/档位集/配额语义是同一入口
   契约的四个面，迁移必须整体切换——只改域名会把 Bearer 头请求打进只认
   apikey query 的 Enterprise，全站 401。
1. 定位是"最近城市吸附"而非格点：AccuWeather 不支持按经纬度点播，必须先用
   Locations API 把站点坐标解析为最近城市的 locationKey：
     GET /locations/v1/cities/geoposition/search?q={lat},{lon}   （Key 走 query）
   （Enterprise 文档的 Locations 指南页为客户端渲染、无法直接核对路径，按官方
   同一 /locations/v1 端点族沿用；契约漂移时定位失败有含响应摘要的可诊断错误
   兜底，取得真实 Key 后应做一次冒烟确认。）
   q 为"纬度,经度"（注意与和风 v7 的"经度,纬度"相反，不可混用）。解析出的
   城市坐标/海拔/Key 全部留档（grid_lat/grid_lon/location_*），并用 haversine
   计算吸附距离 location_distance_km——该源的得分代表"最近城市"，与站点
   点位可能相距数十公里，解读时必须知情。
2. 时效档位取官方 Enterprise 集：官方 Forecasts 页明列 1/12/24/72/120/240/360
   小时——**无 48h**（沿用 dataservice 旧梯子会对 48 白烧一次必拒调用）。
   默认请求 240h（~10 天；2026-08-29 真实 Key 实测订阅最高开放档，评估链路
   hourly_lead_days=16 天可完整覆盖），请求超出订阅的档位返回 403/400，按
   240→120→72→24→12→1 逐级回退（geovis 同款）。
   档位是账号级属性：首次成功后进程内缓存、跨站点复用；不做跨运行持久化——
   每轮重探最坏浪费 5 次调用（梯子全被拒时；订阅正常开放 240h 则为 0），
   换来双向自愈：瞬时 403 降级不会被 git 自动提交的状态文件永久封顶时效。
   实际使用的档位记入快照 tier 字段，实际拿到的点数记入 hours 字段。
3. 配额：Enterprise 配额与订阅合同挂钩，超限官方以 HTTP 409 表达（Overview
   状态表 "Allowed request limit has been exceeded"）——账号级确定性失败，
   不退避立即熔断；503 仍按瞬时过载退避重试、穷尽后熔断。档位梯子全被拒
   同样熔断——账号级确定性失败都让同次运行内后续站点直接失败、不再烧
   退避/重探（若其实是瞬时故障，下次运行自愈）。本源每站每轮 2 次调用
   （1 定位 + 1 预报），4 站 × 3 轮/天 ≈ 24 次，档位探测进程内仅一轮
   （最坏 5 次，梯子全被拒时；订阅正常开放 240h 则为 0）。官方另
   提供 allowError 参数可把错误码压平成 200——刻意不用：状态码是重试/熔断
   状态机的输入，压平会把契约漂移/配额耗尽静默成伪 200，违背"绝不静默"。
4. 鉴权：Enterprise 契约是 ?apikey= query 参数（官方认证页 "Include the apikey
   query parameter on every request"），与 dataservice 入口 2026-06-10 修订的
   Bearer 头契约相反、互不通用。Key 由 Enterprise 订阅签发（sales@accuweather.com，
   非自助创建）。Key 由此重新进入 URL——掩码防御从"纵深"升级为"承重"：
   底层异常/服务端回显仍可能外带 URL 凭据，日志/异常消息一律先经 _masked
   （两层：apikey=<值> 参数形态通用脱敏兜住 percent-encode/回显形态 + 原文
   替换兜底其余泄漏面）；构造期即拒绝含 URL 保留字符的 Key——其编码形态会
   使脱敏失配（审查实证），合法 Key 均为 URL 安全字符。
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
   小时便不成其为预报——注意该论据并不严格成立：所有条目时刻都在起报之后，
   "过去一小时"对预报而言仍是未来，(t−1h, t] 标注与预报语义并不矛盾；
   ② 官网小时块把"当前小时"条目展示为正在发生的未来降水；③ 观测类字段显式
   命名 PastHour，预报字段无此标注。而观测 rain@t 的口径是 (t−1h, t]，故快照
   precipitation@t := TotalLiquid@(t−1h)——把 TotalLiquid 序列整体后移 1 小时
   入库（移位由 liquid_slot 映射按"整点键"实现，序列缺口不会把上一窗的雨错配
   到下一小时）。代价：首个时效点的降水无对应窗记 None，末点的降水被丢弃。
   ── 2026-08-30 假设复核（docstring 预注册的证伪条件："若发现整体前移 1h 反而
   更准时，应优先复核此假设"）：对已积累样本做 −1/0/+1h 整体平移标定，d=+1
   （即不移位）名义更优（POD 48.9%→68.9%、FAR 56.9%→45.6%，McNemar p≈0.06），
   但事件级证据互相矛盾（同一站两次过程分别支持两种口径）、且样本仅 2 天——
   证据不足，移位假设保留不动（不因 p≈0.06 翻动存档口径）；待样本积累 ≥1 个月
   后用同一标定复核，若 d=+1 持续显著占优则整体前移并迁移历史快照（meta 留档）。
   标定方法见审查报告/README"降水口径"一节。快照 precip_alignment 字段留档，
   与风乌 expansion 字段同例。
8. issue_iso 取时间轴首点（评估引擎排除 lead=0），与 qweather/geovis 一致。
"""
from __future__ import annotations

import logging
import os
import re
import time
import unicodedata
from datetime import datetime, timedelta
from math import asin, cos, radians, sin, sqrt
from typing import Any
from urllib.parse import quote

import requests

from .base import ForecastProvider
from ..timeutil import BEIJING

logger = logging.getLogger(__name__)

# Enterprise 入口（官方 Overview 双环境）：生产 api / 开发 apidev。
# host/鉴权/档位/配额是同一入口契约的四个面，切换必须整体进行（docstring 第 0 条）
BASE_URL = "https://api.accuweather.com"
DEV_BASE_URL = "https://apidev.accuweather.com"
# 占位符用单花括号普通字符串（f-string 会吃掉花括号，.format 才是格式化时机）
_LOCATION_PATH = "/locations/v1/cities/geoposition/search"
_FORECAST_PATH = "/forecasts/v1/hourly/{hours}hour/{key}"
LOCATION_URL = f"{BASE_URL}{_LOCATION_PATH}"
FORECAST_URL = f"{BASE_URL}{_FORECAST_PATH}"

SOURCE = "accuweather"
MODEL_NAME = "accuweather_v1"
KEY_ENV = "ACCUWEATHER_API_KEY"

# Enterprise 官方逐小时档位（小时）：官方 Forecasts 页明列 1/12/24/72/120/240/360，
# 无 48——沿用 dataservice 旧梯子（含 48）会对 48 白烧一次必拒调用。
# 从大到小用于订阅档位回退；360 供显式加大 hours 时使用（订阅升级后可用满）。
TIERS = (360, 240, 120, 72, 24, 12, 1)
DEFAULT_HOURS = 240  # 真实 Key 实测订阅最高开放档（2026-08-29）
HEADERS = {"User-Agent": "weather-api-eval/0.1 (+https://github.com/)"}

# 单位自适应：AccuWeather 数值对象自带 Unit。温度目标 ℃、降水目标 mm；
# 键为响应 Unit 的小写形态，值为 (乘数, 加数) 线性换算。
_TEMP_TO_C = {"c": (1.0, 0.0), "f": (5.0 / 9.0, -32.0 * 5.0 / 9.0)}
_LIQUID_TO_MM = {"mm": (1.0, 0.0), "in": (25.4, 0.0), "inch": (25.4, 0.0)}


class _QuotaHint(Exception):
    """503 穷尽重试后携带配额说明的失败（内部占位，最终转 RuntimeError）。"""


def _tier_for(hours: int) -> int:
    """把请求时效吸附到官方档位：取不超过 hours 的最大档，无则取最小档 1。

    非官方档位（如 dataservice 时代的 48，Enterprise 已无此档）落在此路径，
    INFO 留痕避免静默降档。"""
    usable = [t for t in TIERS if t <= hours]
    tier = max(usable) if usable else min(TIERS)
    if hours not in TIERS:
        logger.info("请求时效 %dh 非官方档位，吸附到 %dh（Enterprise 官方档位 %s）",
                    hours, tier, "/".join(map(str, TIERS[::-1])))
    return tier


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
        f"AccuWeather Enterprise 请求失败（{detail}）。Enterprise 超限的官方形态是 "
        "HTTP 409（Allowed request limit has been exceeded），请核对 Enterprise 订阅"
        "的当日用量与合同限额（本源每站每轮 2 次调用：1 定位 + 1 预报，档位探测另计）；"
        "503 则多为服务端过载，退避穷尽仍持续失败时也应核对配额账本。"
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
                 retries: int = 3, session: requests.Session | None = None,
                 base_url: str = BASE_URL):
        self.key = api_key if api_key is not None else os.environ.get(KEY_ENV)
        if self.key:
            # CI Secret/本地 .env 常携带首尾空白或换行——剥除后再进 apikey 参数
            self.key = self.key.strip()
            if any(unicodedata.category(c) == "Cc" for c in self.key):
                # Cc 覆盖 C0/DEL/C1 全部控制码位，非法字符绝不入请求 URL
                raise RuntimeError(
                    f"AccuWeather API Key 含非法控制字符（经 {KEY_ENV} 注入），"
                    "请检查凭据来源是否被污染"
                )
            if quote(self.key, safe="") != self.key:
                # URL 保留字符（+&=/、空格、% 等）会被 requests 编码成与原文不同的
                # 形态：异常/日志里的 URL 携带 percent-encode 形态，_masked 的原文
                # 替换匹配不到 → 凭据泄漏（审查实证）。合法 Key 均为 URL 安全字符，
                # 含保留字符即凭据来源可疑，直接拒绝而非静默掩码
                raise RuntimeError(
                    f"AccuWeather API Key 含 URL 保留字符（经 {KEY_ENV} 注入），"
                    "percent-encode 形态会使日志脱敏失效，请检查凭据来源是否正确"
                )
        if not self.key:
            raise RuntimeError(
                f"AccuWeather API Key 未提供：Enterprise 订阅 Key（sales@accuweather.com "
                f"签发，非自助创建）请经环境变量 {KEY_ENV} 注入，或在构造 "
                f"AccuWeatherProvider 时显式传入 api_key。"
            )
        self.hours = _tier_for(hours)  # 请求时效吸附到官方档位
        self.language = language
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        # Enterprise 双环境（docstring 第 0 条）：默认生产入口，可切开发环境
        self.base_url = base_url.rstrip("/")
        if not self.base_url:
            raise RuntimeError(
                "AccuWeather base_url 不能为空（应为 Enterprise 入口 URL，如 "
                f"{BASE_URL}）"
            )
        self.location_url = f"{self.base_url}{_LOCATION_PATH}"
        self.forecast_url = f"{self.base_url}{_FORECAST_PATH}"
        # Enterprise 契约：Key 经 apikey query 参数传递（_get 集中注入），
        # 请求头不携带凭据；dataservice 入口 2026-06-10 的 Bearer 头契约与本入口
        # 互不通用，不得混用
        self._headers = dict(HEADERS)
        self._tier_cache: int | None = None   # 本进程已验证的可用档位（账号级，跨站点复用）
        self._quota_suspect = False           # 409/503 穷尽后的熔断标志
        self._tiers_exhausted = False         # 档位梯子全被拒后的熔断标志

    # ------------------------------------------------------------------ 对外
    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> dict:
        if self._quota_suspect:
            # 同一次运行内配额失败已确认（409 直判 / 503 退避穷尽）：
            # 后续站点快速失败，不烧配额与时间
            raise RuntimeError(_quota_hint("熔断：本次运行已有请求确认配额失败"))
        if self._tiers_exhausted:
            # 档位梯子全被拒（订阅未开放逐小时预报或配额以 403 形态耗尽）：
            # 属账号级确定性失败，后续站点不再重复梯子探测
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
        status, body = self._get(self.location_url, {
            "q": f"{station.lat},{station.lon}",
            "language": self.language,
        })
        if status == 401:
            raise RuntimeError(
                f"AccuWeather 定位鉴权失败（HTTP 401，官方认证页语义：Missing or invalid "
                f"API key）：Key 已按 Enterprise 契约经 apikey query 参数传递仍被拒，"
                f"请核对 Enterprise 订阅的 {KEY_ENV} 是否有效/启用"
            )
        if status != 200 or not isinstance(body, dict):
            hint = ("响应结构异常（应为单个 location 对象，疑似契约漂移）" if status == 200
                    else "官方状态表：403=未随请求提供有效 Key、404=无匹配路由或坐标无匹配城市")
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

        无跨运行持久化：每轮从默认档重探最多浪费 3 次调用，换来双向自愈——
        瞬时 403 降级不会像状态文件那样被 git 自动提交永久封顶时效。
        """
        tier = self._tier_cache or self.hours
        rejected: list[str] = []
        while True:
            status, body = self._get(
                self.forecast_url.format(hours=tier, key=loc_key),
                {"language": self.language, "details": "true", "metric": "true"},
            )
            if status == 200:
                self._tier_cache = tier
                return tier, body
            if status == 401:
                raise RuntimeError(
                    f"AccuWeather 鉴权失败（HTTP 401，官方认证页语义：Missing or invalid "
                    f"API key）：Key 已按 Enterprise 契约经 apikey query 参数传递仍被拒，"
                    f"请核对 Enterprise 订阅的 {KEY_ENV} 是否有效/启用"
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
        """请求一个端点，并按 Enterprise 契约集中注入 apikey query 参数
        （端点 params 只管业务语义，鉴权不可能被单个调用点遗漏）。
        200 或确定性 4xx（除 429/409）返回 (status, body)；409 为官方"订阅允许
        的请求上限已被超出"——账号级确定性配额失败，不退避、立即置熔断标志并
        携带配额说明抛错；网络错误/5xx/429 指数退避重试；503 穷尽后同样熔断。
        409 分支在 try 之外判定：raise 若落在 try 内会被网络异常分支捕获、
        被当成可重试错误烧满退避。"""
        params = {**params, "apikey": self.key}
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
            except Exception as e:  # noqa: BLE001  网络类异常 → 可重试
                last_status = None
                # 归一为脱敏后的 RuntimeError：最终 raise 挂 __cause__ 链时，
                # 原始异常消息（可能含服务端回显的凭据）不会绕过掩码外泄
                last_err = RuntimeError(_masked(str(e), self.key))
            else:
                if status == 200:
                    return status, body
                if status == 409:
                    # Enterprise 官方语义：Allowed request limit has been
                    # exceeded——确定性配额失败，重试无意义；置熔断让同次运行内
                    # 后续站点快速失败、不再烧配额
                    self._quota_suspect = True
                    raise RuntimeError(_quota_hint("HTTP 409 已超出订阅允许的请求上限"))
                if isinstance(status, int) and 400 <= status < 500 and status != 429:
                    # 鉴权/权限/档位类确定性错误：重试不会改变结果，交上层判定
                    return status, body
                last_status = status
                last_err = RuntimeError(f"HTTP {status} {_masked(_body_digest(body), self.key)}")
            logger.warning("AccuWeather 请求失败（第%d次）: %s", attempt + 1,
                           _masked(str(last_err), self.key))
            if attempt < self.retries:
                time.sleep(min(30, 3 * 2 ** attempt))
        if last_status == 503:
            self._quota_suspect = True
            raise RuntimeError(_quota_hint("HTTP 503 已退避重试穷尽")) from last_err
        raise RuntimeError(f"AccuWeather 请求最终失败: {_masked(str(last_err), self.key)}") from last_err


# URL 参数形态的 apikey 值（大小写不敏感）：_masked 的通用脱敏兜底
_APIKEY_IN_URL_RE = re.compile(r"(apikey=)[^&\s'\"<>]+", re.IGNORECASE)


def _masked(text: str, key: str) -> str:
    """Enterprise 契约下 Key 经 apikey query 参数进入 URL——日志/异常中的 URL、
    底层异常消息、服务端回显都可能外带 Key，入日志前一律掩码（承重防御）。

    两层：先按 `apikey=<值>` 的 URL 参数形态通用脱敏（兜住 percent-encode、
    服务端回显等与原文不同的形态——原文替换对编码形态必然失配，审查实证）；
    再按原文替换兜底其余泄漏面（如异常消息里裸出现的 Key）。"""
    text = _APIKEY_IN_URL_RE.sub(r"\1***", text)
    if key:
        text = text.replace(key, "***")
    return text

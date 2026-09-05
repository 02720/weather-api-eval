"""和风天气（QWeather）逐小时预报快照器。

认证方式：API Key 经 ``X-QW-Api-Key`` 请求头传递（文档亦支持 key= 查询参数，
但同一请求不得混用两种方式）。API Host 为控制台「设置」中分配的每账号专属域名；
原公共域名 devapi/api.qweather.com 自 2026 年起逐步停止服务，故 host 可经环境
变量 QWEATHER_API_HOST 注入（未注入时退回 devapi 并显式告警提醒迁移）。

─────────────────────────────────────────────────────────────────────
端点选择（第一性原理，关键设计约束）
─────────────────────────────────────────────────────────────────────
1. 首选新版 ``GET {host}/weather/v1/hourly/{纬度}/{经度}``：
   - 路径参数"纬度在前"，且坐标契约要求不超过 2 位小数，必须先取整再拼路径，
     否则按无效参数被拒（对应业务码 400）；
   - ``hours`` 取值 1..240（省略时仅返回 24 点）；
   - 条目字段 forecastTime 形如 ``2024-05-31T03:00Z``，为 UTC 时间——而本评估
     体系全程使用北京时 naive 墙钟并与整点观测按字符串精确配对，因此必须把它
     astimezone 到北京时、去掉偏移并下取整到整点，否则与整点观测零配对；
   - 温度取 temperature.value（°C，number），降水取 precipitation.amount
     （当小时累计降水量，mm），缺测时为 null/缺失键，统一归一为 None。
2. 响应没有 HTTP 200 却缺 ``hours``，或命中 404 这类"路由不存在"信号时，说明
   该凭据/API Host 尚未开通 weather/v1，自动降级到旧版
   ``GET {host}/v7/weather/{tier}h?location={经度},{纬度}``（tier ∈ {24,72,168}
   取不超过所请求时效的最大档）继续完成抓取。v7 的数值为字符串且免费档仅开放
   24h，这些差异都以 WARNING 明示，绝不静默缩短评估时效。401 属于账号级鉴权
   失败，在两套端点上结果相同，直接失败而不做无意义的第二次请求。

快照口径与 Open-Meteo/彩云一致：issue_iso 取共享时间轴首点（即起报轮次的当前
小时，lead=0 本就被评估引擎排除）；时间轴为北京时 naive 整点字符串；模型名固定
为 qweather_v1，保证跨运行快照目录稳定（无论实际由哪一代接口返回数据）。

─────────────────────────────────────────────────────────────────────
降水口径（2026-08-30 标定，详见 README"降水口径"一节）
─────────────────────────────────────────────────────────────────────
和风官方文档对逐小时 precip 的描述即"目标小时/当前小时累计降水量"，方向未逐字
明示。本项目按"前 1 小时累计"（值@t 覆盖 (t−1h, t]，与观测 rain@t 同标注）直接
透传入库，不做移位。已对首月样本做 −1/0/+1h 整体平移标定：短时效 d=−1 名义最优
但与 d=0 差异在噪声内（TS 0.205 vs 0.198），长时效 d=0 明确最优（TS 0.271 vs
0.250/0.262）——无 1h 错位证据。其 FAR 偏高与同场全球模式相当，属预报特性而非
对齐问题。若日后标定翻转，应把换算落在本模块并在此留档。

限速遵循官方"指数退避"建议：对网络错误/5xx/429 以 3、6、12…秒退避重试；
确定性 4xx（鉴权、权限、参数）不重试、直接上抛交由调用方处理。

─────────────────────────────────────────────────────────────────────
逐日预报块（可选扩展，契约见 forecast/base.py）
─────────────────────────────────────────────────────────────────────
逐小时（240h=10 天）之外，本源还有覆盖更远的逐日产品——逐小时一断供，按天评估
可由日产品接续（本源是逐日补位产生**真实时效增益**的主要源之一）：
- 选用旧版 `GET {host}/v7/weather/{tier}d`（tier ∈ 3/7/10/15/30d，从大到小逐档
  回退，档位随订阅而定、进程内缓存）：daily[].fxDate（'YYYY-MM-DD'，站点所在时区
  自然日——评估站均在国内即北京时自然日）、tempMax/tempMin（字符串 ℃）、
  precip（**当日总降水量**，mm，直取即得，无需任何推导）。
- 不接新版 weather/v1 daily（days≤10，与逐小时上限持平，**零时效增益**；且无
  整日降水量字段，只有白天/夜间两个半天量，求和引入日界窗口假设）。
- v7 已官方宣布 2027-02-01 停止服务（存量订阅用户仍可用）：届时各档将确定性
  失败，本块自动退化为"不带逐日块"（WARNING 可见），按天轨道回到纯逐小时口径，
  绝不拖垮整份起报快照。
- 逐日抓取失败（任一形态）只降级为不带该块的快照：逐小时是主干，日产品只是
  延长线，快照错过即无法追补，绝不因延长线丢主干。
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

import requests

from .base import ForecastProvider
from ..timeutil import BEIJING

logger = logging.getLogger(__name__)

DEFAULT_NAME = "qweather_v1"
KEY_ENV = "QWEATHER_API_KEY"
HOST_ENV = "QWEATHER_API_HOST"
# 旧公共地址自 2026 年起逐步停服；仅在用户未提供专属 Host 时作为兜底并告警。
DEPRECATED_FALLBACK_HOST = "devapi.qweather.com"

DEFAULT_HOURS = 240          # 新版接口支持上限（1..240）
_V7_TIERS = (24, 72, 168)    # 旧版逐小时接口的三档时效
# 旧版逐日接口的档位（从大到小用于订阅回退；docstring"逐日预报块"一节）
_V7_DAILY_TIERS = (30, 15, 10, 7, 3)
HEADERS = {"User-Agent": "weather-api-eval/0.1 (+https://github.com/)"}


class _Transient(Exception):
    """需要退避重试的瞬时失败占位（网络错误 / 5xx / 429）。"""


def _redact(text: str, key: str) -> str:
    """抹掉文本中的 API Key，避免它随日志/异常泄露。"""
    if key:
        text = text.replace(key, "***")
    return text


def _to_float(v: Any) -> float | None:
    """把和风的数值统一转为 float。

    新版是 number 或 null；旧版 v7 是字符串（如 "28"、"0.0"），缺测为空串。
    无法解释的值一律归 None，绝不把缺测伪装成 0 参与评估。
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _metric_value(obj: Any) -> float | None:
    """新版带单位量纲统一为 {'value': number, 'unit': str}（如 temperature、
    precipitation.amount——注意 amount 本身就是该对象，数值在 .value 里）；
    个别字段/服务端形态也可能直接给标量，两种都兼容，缺解释时归 None。"""
    if isinstance(obj, dict):
        return _to_float(obj.get("value"))
    return _to_float(obj)


def _parse_qweather_dt(s: Any) -> datetime:
    """把和风时间戳解析为北京时 naive datetime 并下取整到整点。

    兼容三种形态：
      "2026-08-27T07:00Z"        （新版 weather/v1，UTC，须 +8 换算）
      "2026-08-27T15:00+08:00"   （旧版 v7 fxTime，带偏移）
      "2026-08-27T15:00"         （异常形态兜底，视为已是墙钟）
    """
    if not isinstance(s, str):
        raise ValueError(f"非法时间戳: {s!r}")
    raw = s.strip()
    if raw.endswith(("Z", "z")):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError as e:
        raise ValueError(f"无法解析和风时间戳: {s!r}") from e
    if dt.tzinfo is not None:
        dt = dt.astimezone(BEIJING).replace(tzinfo=None)
    return dt.replace(minute=0, second=0, microsecond=0)


def _normalize_host(host: str) -> str:
    # 域名大小写不敏感，统一小写以便比较与日志一致
    h = (host or "").strip().lower().rstrip("/")
    for p in ("https://", "http://"):
        if h.startswith(p):
            h = h[len(p):]
    return h


def _describe_error(body: Any) -> str:
    """尽量从错误响应体中提取可读信息（如 error.invalidParams）。"""
    if isinstance(body, dict):
        parts = []
        if body.get("error"):
            parts.append(f"error={body['error']!r}")
        if body.get("code"):
            parts.append(f"code={body['code']!r}")
        if parts:
            return ", ".join(parts)
        return f"body_keys={sorted(body.keys())!r}"
    if body is None:
        return "body=<非 JSON>"
    return f"body={body!r:.200}"


def _v7_tier_for(hours: int) -> int:
    """旧版只有 24/72/168 三档，取不超过请求时效的最大档。

    请求时效低于最小档（hours<24）时取最小档 24h——多余点由
    ``_merge_series`` 按 expected=min(hours,tier) 截断，不会进入快照。
    """
    usable = [t for t in _V7_TIERS if t <= hours]
    return max(usable) if usable else min(_V7_TIERS)


def _parse_daily_entries(entries: Any, station_id: str,
                         model: str = DEFAULT_NAME) -> dict | None:
    """解析 v7 daily[] → {"time": [...], "data": {模型: {...}}}（北京时自然日）。

    fxDate 非法/缺失的条目跳过；重复日保留首见；时间轴升序去重（契约要求）。
    解析不出任何有效日返回 None（调用方告警并退化为不带该块的快照）。
    """
    if not isinstance(entries, list):
        return None
    rows: dict[str, tuple[float | None, float | None, float | None]] = {}
    for ent in entries:
        if not isinstance(ent, dict):
            continue
        d = ent.get("fxDate")
        if not isinstance(d, str):
            logger.warning("和风站点 %s 逐日条目缺 fxDate（%r），跳过", station_id, d)
            continue
        try:
            datetime.strptime(d.strip(), "%Y-%m-%d")
        except ValueError:
            logger.warning("和风站点 %s 逐日 fxDate 非法（%r），跳过", station_id, d)
            continue
        day = d.strip()
        if day in rows:  # 重复日保留首见
            continue
        rows[day] = (_to_float(ent.get("tempMax")),
                     _to_float(ent.get("tempMin")),
                     _to_float(ent.get("precip")))
    if not rows:
        return None
    days = sorted(rows)
    return {
        "time": days,
        "data": {model: {
            "temp_max": [rows[d][0] for d in days],
            "temp_min": [rows[d][1] for d in days],
            "precipitation": [rows[d][2] for d in days],
        }},
    }


def _merge_series(
    pairs: list[tuple[datetime, float | None, float | None]],
    station_id: str,
    requested_hours: int,
) -> tuple[list[str], list[float | None], list[float | None]]:
    """把 (时刻, 温度, 降水) 元组序列整理为共享时间轴。

    - 按 datetime 升序排序并对重复整点去重（保留首见），保证评估引擎可以安全地
      用 i 索引同时取出温度与降水（两条数组共用同一条时间轴）；
    - 超过所请求时效的点截断丢弃，保持与其他源的快照口径一致；
    - 解析不出任何有效点时报错而非产出空快照（上游会静默变成"样本不足"）；
    - 点数少于请求值（例如免费档被限制为 24 点）时以 WARNING 暴露降级。
    """
    seen: dict[datetime, tuple[float | None, float | None]] = {}
    skipped = 0
    for dt, tv, pv in pairs:
        seen.setdefault(dt, (tv, pv))
    ordered = sorted(seen.items())[:requested_hours]
    if len(seen) != len(pairs):
        skipped = len(pairs) - len(seen)
        logger.warning("和风站点 %s 存在 %d 个重复/无效整点，已去重", station_id, skipped)

    if not ordered:
        raise RuntimeError(f"和风站点 {station_id} 未解析出任何有效逐小时数据")

    hourly_time = [dt.strftime("%Y-%m-%dT%H:%M") for dt, _ in ordered]
    temps = [tv for _, (tv, _) in ordered]
    precips = [pv for _, (_, pv) in ordered]

    if len(hourly_time) < requested_hours:
        logger.warning(
            "和风站点 %s 仅返回 %d 个逐小时点（请求 %dh）。免费开发版或其他订阅限制可能截断了时效，"
            "本次评估对该源的有效时效约 %dh。",
            station_id, len(hourly_time), requested_hours, len(hourly_time),
        )
    return hourly_time, temps, precips


class QWeatherProvider(ForecastProvider):
    def __init__(
        self,
        key: str | None = None,
        host: str | None = None,
        name: str = DEFAULT_NAME,
        hours: int = DEFAULT_HOURS,
        timeout: int = 60,
        retries: int = 3,
        session: requests.Session | None = None,
    ):
        self.key = key if key is not None else os.environ.get(KEY_ENV)
        if not self.key:
            raise RuntimeError(
                f"和风 API Key 未提供：请在环境变量 {KEY_ENV} 中设置，"
                f"或在构造 QWeatherProvider 时显式传入 key。"
            )
        host_from_arg = _normalize_host(host) if host else None
        host_from_env = _normalize_host(os.environ.get(HOST_ENV, ""))
        if host_from_arg:
            self.host = host_from_arg
        elif host_from_env:
            self.host = host_from_env
        else:
            self.host = DEPRECATED_FALLBACK_HOST
            logger.warning(
                "未设置 %s，退回旧公共地址 %s。该公共地址自 2026 年起逐步停止服务，"
                "请尽快改用你控制台「设置」中的专属 API Host。",
                HOST_ENV, DEPRECATED_FALLBACK_HOST,
            )
        self.name = name
        self.hours = max(1, min(int(hours), 240))
        self.timeout = timeout
        self.retries = retries
        # 逐日档位：订阅级属性，首次成功后进程内缓存、跨站点复用（与逐小时梯子同理）
        self._daily_tier_cache: int | None = None
        # Key 只经请求头传递，绝不进入 URL/query（避免随日志/代理泄露）。
        self._headers = {**HEADERS, "X-QW-Api-Key": self.key}
        self.session = session or requests.Session()

    # ------------------------------------------------------------------ 对外
    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> dict:
        r_lat = round(float(station.lat), 2)   # 和风坐标契约：小数不超过 2 位
        r_lon = round(float(station.lon), 2)

        v1_url = f"https://{self.host}/weather/v1/hourly/{r_lat}/{r_lon}"
        status_v1, body_v1 = self._get(v1_url, {"hours": self.hours})
        entries = body_v1.get("hours") if isinstance(body_v1, dict) else None
        desc_v1 = _describe_error(body_v1)

        if status_v1 == 401:
            raise RuntimeError(
                f"和风鉴权失败（HTTP 401）：请检查 API Key 是否有效、是否放行当前 API Host "
                f"（{self.host}）。{desc_v1}"
            )

        if isinstance(entries, list) and entries:
            pairs: list[tuple[datetime, float | None, float | None]] = []
            for ent in entries:
                if not isinstance(ent, dict):
                    continue
                try:
                    dt = _parse_qweather_dt(ent.get("forecastTime"))
                except ValueError:
                    logger.warning("和风站点 %s 出现无法解析的 forecastTime=%r，跳过",
                                   station.id, ent.get("forecastTime"))
                    continue
                temp_obj = ent.get("temperature")
                prec_obj = ent.get("precipitation")
                tv = _metric_value(temp_obj) if isinstance(temp_obj, dict) else None
                # precipitation={"amount":{"value":..,"unit":"mm"},...}：取内层 value
                pv = (_metric_value(prec_obj.get("amount"))
                      if isinstance(prec_obj, dict) else None)
                pairs.append((dt, tv, pv))
            used_api = "weather/v1"
            expected = self.hours
        else:
            # weather/v1 不可用：404 等路由缺失、订阅未开通该路由（403/400），
            # 或 200 却不带 hours（服务不兼容）。不把 403 快速失败：免费凭据可能
            # 只是尚未开通 weather/v1 而 v7 可用；若为 Host 错误/额度耗尽等账号级
            # 问题，v7 会同样失败并抛出携带两侧状态码的联合错误，根因仍可定位。
            hint = ""
            if status_v1 == 403:
                hint = "；403 常见原因：额度不足/账单逾期/API Host 不符/无权限"
            elif status_v1 == 429:
                hint = "；429 表示触发限流（请求已按退避策略重试过）"
            logger.warning(
                "站点 %s 的 weather/v1 不可用（HTTP %s：%s%s），自动改用旧版 "
                "/v7/weather 接口（数值同为温度°C/小时降水mm，但免费档时效仅 24h）",
                station.id, status_v1, desc_v1, hint,
            )
            tier = _v7_tier_for(self.hours)
            v7_url = f"https://{self.host}/v7/weather/{tier}h"
            status_v7, body_v7 = self._get(v7_url, {"location": f"{r_lon},{r_lat}"})
            code = str(body_v7.get("code")) if isinstance(body_v7, dict) else None
            if status_v7 != 200 or code != "200":
                raise RuntimeError(
                    f"和风两代接口均请求失败：weather/v1 -> HTTP {status_v1}（{desc_v1}）；"
                    f"v7/{tier}h -> HTTP {status_v7}, code={code}"
                    f"{'' if not isinstance(body_v7, dict) else ', ' + _describe_error(body_v7)}"
                )
            items = body_v7.get("hourly")
            if not isinstance(items, list) or not items:
                raise RuntimeError(f"和风站点 {station.id} 旧版接口未返回逐小时条目")
            pairs = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                try:
                    dt = _parse_qweather_dt(it.get("fxTime"))
                except ValueError:
                    logger.warning("和风站点 %s 出现无法解析的 fxTime=%r，跳过",
                                   station.id, it.get("fxTime"))
                    continue
                pairs.append((dt, _to_float(it.get("temp")), _to_float(it.get("precip"))))
            used_api = f"v7/{tier}h"
            expected = min(self.hours, tier)

        hourly_time, temps, precips = _merge_series(pairs, station.id, expected)

        daily_block = self._fetch_daily_block(r_lon, r_lat, station.id)

        snapshot = {
            "issue_iso": hourly_time[0],
            "station_id": station.id,
            "source": "qweather",
            "models": [self.name],
            # 和风不做格点吸附：以实际参与查询的 2 位小数坐标为准（contract 上界），
            # 响应该端点无 elevation 字段，记 None。
            "grid_lat": r_lat,
            "grid_lon": r_lon,
            "elevation": None,
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            "hourly_time": hourly_time,
            "data": {
                self.name: {"temperature_2m": temps, "precipitation": precips}
            },
        }
        if daily_block:
            snapshot["daily_time"] = daily_block["time"]
            snapshot["daily"] = daily_block["data"]
        logger.info(
            "站点 %s 已抓取和风起报 %s（%s），时间点数 %d，日产品 %d 天",
            station.id, hourly_time[0], used_api, len(hourly_time),
            len(daily_block["time"]) if daily_block else 0,
        )
        return snapshot

    # ------------------------------------------------------------------ 内部
    def _fetch_daily_block(self, r_lon: float, r_lat: float,
                           station_id: str) -> dict | None:
        """v7 逐日预报块：档位从大到小逐档回退，成功档位进程内缓存。

        任何失败（档位被拒/业务错误码/网络穷尽/解析为空）都不抛出——逐日块是
        可选延长线，退化为"不带该块的快照"并以 WARNING 暴露（docstring 契约）；
        401 同样只降级：此时逐小时已成功，Key 有效性已被证实，逐日 401 属产品
        权限/契约问题，降档无意义也不应拖垮快照。
        """
        tiers: list[int] = []
        if self._daily_tier_cache is not None:
            tiers.append(self._daily_tier_cache)
        tiers += [t for t in _V7_DAILY_TIERS if t != self._daily_tier_cache]
        for tier in tiers:
            try:
                status, body = self._get(
                    f"https://{self.host}/v7/weather/{tier}d",
                    {"location": f"{r_lon},{r_lat}"},
                )
            except Exception as e:  # noqa: BLE001  网络穷尽等：降档继续
                logger.warning("和风站点 %s 逐日预报（v7/%dd）请求失败: %s",
                               station_id, tier, _redact(str(e), self.key))
                continue
            code = str(body.get("code")) if isinstance(body, dict) else None
            entries = body.get("daily") if isinstance(body, dict) else None
            if status == 200 and code == "200" and entries:
                parsed = _parse_daily_entries(entries, station_id, model=self.name)
                if parsed is not None:
                    self._daily_tier_cache = tier
                    return parsed
                logger.warning("和风站点 %s 逐日响应未解析出任何有效日（契约漂移），"
                               "本次快照不带逐日预报块", station_id)
                return None
            logger.info("和风站点 %s 逐日档位 %dd 不可用（HTTP %s code=%s），降档尝试",
                        station_id, tier, status, code)
        logger.warning(
            "和风站点 %s 逐日预报各档位均不可用（订阅/产品权限或 v7 停服），"
            "本次快照不带逐日预报块，按天评估将只用逐小时聚合", station_id)
        return None

    def _get(self, url: str, params: dict) -> tuple[int | None, Any]:
        """请求一个端点。HTTP 200 或确定性 4xx 时返回 (status, 已解析的 json 或 None)；
        网络错误/5xx/429 按官方建议指数退避重试，全部失败后抛 RuntimeError。"""
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(
                    url, params=params, headers=self._headers, timeout=self.timeout
                )
                status = getattr(resp, "status_code", None)
                try:
                    body = resp.json()
                except ValueError:
                    body = None
                if status == 200:
                    return status, body
                if isinstance(status, int) and 400 <= status < 500 and status != 429:
                    # 鉴权/权限/参数错误重试也不会改变结果，立即交给上层判定。
                    return status, body
                last_err = _Transient(f"HTTP {status}")
            except Exception as e:  # noqa: BLE001  网络类异常 → 可重试
                last_err = e
            logger.warning(
                "和风请求失败（第%d次）: %s", attempt + 1,
                _redact(f"{url}: {last_err}", self.key),
            )
            if attempt < self.retries:
                time.sleep(min(30, 3 * 2 ** attempt))  # 官方建议指数退避
        raise RuntimeError(
            f"和风请求最终失败: {_redact(str(last_err), self.key)}"
        ) from last_err

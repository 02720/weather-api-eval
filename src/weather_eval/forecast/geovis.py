"""中科星图（GeoVis Earth）全国逐小时预报快照器。

数据源：中科星图维天信天气 API「全国90天/逐小时预报」组的
《全国城市120小时逐小时预报》（文档：datacloud.geovisearth.com/support/meteorological/
chinaCity120HourForecast，入口即 meteorological/summary 页）：
  GET https://tiles.geovisearth.com/meteorology/v1/weather/cn/forecast/hour/area
      [/professional|/basic]?location={lon,lat}&token={token}

选型说明：评估站均在国内、需要逐小时气温+降水，该产品（3300+ 市区县、支持经纬度
直查、1h 分辨率、最长 120h、每天更新 7 次、`tem`=℃、`pre`=1h 降水量 mm）与评估
契约最契合；全球公里级网格预报（25km、5 天后仅 3h 步长）作备选未接入。

─────────────────────────────────────────────────────────────────────
关键契约（依据官方文档 2026-08 版，改动前必须复核）
─────────────────────────────────────────────────────────────────────
1. 鉴权：需注册 + 开发者认证，`token` 走 query 参数（经环境变量 GEVIS_TOKEN 注入）；
   档位订阅决定可用时效，本提供方按 专业(120h) → 进阶(48h) → 基础(24h) 顺序回退，
   实际使用的档位记入快照 meta（tier 字段）。档位不可用有两种表达：HTTP 4xx，
   或 HTTP 200 + 业务失败（status!=0 / datas 空）——两者都纳入降档回退，档位只在
   解析成功后固定，不依赖"档位错误必为 HTTP 4xx"的未验证假设。
2. 时间语义：`fc_time`/`start`/`end` 为 **yyyyMMddHH 当地时间**（响应 date.timeZone
   = "Asia/Shanghai"），直接按北京时墙钟解析；数据从查询时刻（= start）起逐小时。
   issue_iso 取 result.start —— 因数据"查询时间起报"，start 即起报时刻。
3. 数值口径：tem=℃；pre=该小时降水量 mm（前 1 小时累计，与观测 rain@t 直接同口径，
   无需近似）；异常值 **999999** → None（气温/降水等 double 字段的官方缺测标记）。
4. `status != 0` 为业务失败（如 token 无效/无权限），直接报错，附带响应摘要。
5. 传入经纬度时响应无 areaCode/location 字段，坐标以请求值为准（城市级后处理产品，
   无格点吸附语义）。
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Any

import requests

from .base import ForecastProvider

logger = logging.getLogger(__name__)

AREA_URL = ("https://tiles.geovisearth.com/meteorology/v1/weather/cn/"
            "forecast/hour/area")
HEADERS = {"User-Agent": "weather-api-eval/0.1 (+https://github.com/)"}

SOURCE = "geovis"
MODEL_NAME = "geovis_v1"
# 档位 → URL 后缀（专业 120h → 进阶 48h → 基础 24h）
TIERS: list[tuple[str, str]] = [("professional", "/professional"), ("48h", ""), ("basic", "/basic")]
# 官方异常值 999999；对 9999/99999 等变体一并剔除（气温/降水的真实值不可能 ≥9999）
MISSING_TOL = 9999.0

TOKEN_ENV = "GEVIS_TOKEN"


def parse_fc_time(s: Any) -> str | None:
    """yyyyMMddHH（北京时）→ "YYYY-MM-DDTHH:00"；非法返回 None。"""
    if not isinstance(s, str) or len(s) != 10 or not s.isdigit():
        return None
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:00"


def _num_or_none(v: Any) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f:
        return None
    return None if abs(f) >= MISSING_TOL else f


class _TierUnavailable(RuntimeError):
    """业务层"该档位不可用"（status!=0 / datas 空）：应降档重试而非整源失败。"""


def parse_area_response(payload: Any) -> dict[str, Any]:
    """解析逐小时预报响应 → {"issue_iso", "time", "temperature_2m", "precipitation"}。

    status != 0 / datas 为空抛 _TierUnavailable（RuntimeError 子类，档位回退捕捉）；
    start 非法等其他异常仍为普通 RuntimeError（契约漂移，降档无意义，直接上抛）；
    fc_time 无法解析的条目跳过。
    """
    if not isinstance(payload, dict):
        raise RuntimeError(f"星图响应异常: {payload!r}"[:300])
    status = payload.get("status")
    if status not in (0, "0"):
        raise _TierUnavailable(
            f"星图业务错误: status={status!r} resp={str(payload)[:200]!r}"
        )
    result = payload.get("result") or {}
    datas = result.get("datas")
    if not isinstance(datas, list) or not datas:
        raise _TierUnavailable("星图响应 result.datas 为空（token 无权限或产品未覆盖该点）")
    start = parse_fc_time(result.get("start"))
    if start is None:
        raise RuntimeError(f"星图响应 result.start 非法: {result.get('start')!r}")
    pairs: dict[str, tuple[float | None, float | None]] = {}
    for d in datas:
        if not isinstance(d, dict):
            continue
        t = parse_fc_time(d.get("fc_time"))
        if t is None:
            continue
        pairs.setdefault(t, (_num_or_none(d.get("tem")), _num_or_none(d.get("pre"))))
    times = sorted(pairs)
    return {
        "issue_iso": start,
        "time": times,
        "temperature_2m": [pairs[t][0] for t in times],
        "precipitation": [pairs[t][1] for t in times],
    }


class GevisProvider(ForecastProvider):
    """中科星图逐小时预报快照器：需 GEVIS_TOKEN，单模型快照 dict。"""

    def __init__(self, timeout: int | tuple = (10, 60), retries: int = 3,
                 session: requests.Session | None = None, token: str | None = None):
        self.token = token if token is not None else os.environ.get(TOKEN_ENV, "")
        if not self.token:
            raise RuntimeError(
                f"缺少中科星图 token：请在 datacloud.geovisearth.com 注册并完成开发者"
                f"认证后获取，经环境变量 {TOKEN_ENV} 注入"
            )
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self._tier_cache: str | None = None  # 可用档位（账号级属性，跨站点复用）

    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> dict:
        # 档位在首次成功（含解析成功）后固定（权限是账号级属性，跨站点复用）
        parsed, tier = self._query_any_tier(station)
        if parsed["temperature_2m"] and all(v is None for v in parsed["temperature_2m"]):
            logger.warning("星图站点 %s 温度序列全部缺测，服务端契约可能已变化", station.id)
        if parsed["precipitation"] and all(v is None for v in parsed["precipitation"]):
            logger.warning("星图站点 %s 降水序列全部缺测，本快照降水将计为缺测", station.id)
        snapshot = {
            "issue_iso": parsed["issue_iso"],
            "station_id": station.id,
            "source": SOURCE,
            "models": [MODEL_NAME],
            "grid_lat": float(station.lat),
            "grid_lon": float(station.lon),
            "elevation": None,
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            "tier": tier,
            "hourly_time": parsed["time"],
            "data": {MODEL_NAME: {
                "temperature_2m": parsed["temperature_2m"],
                "precipitation": parsed["precipitation"],
            }},
        }
        if not parsed["time"]:
            # 空时间轴快照一旦入库会被同 issue 幂等锁死，正常数据永远进不来
            raise RuntimeError("星图响应无可解析的 fc_time 条目（疑似契约漂移），拒绝入库")
        logger.info("站点 %s 已抓取星图逐小时预报（档位 %s）起报 %s，时间点数 %d",
                    station.id, tier, parsed["issue_iso"], len(parsed["time"]))
        return snapshot

    # ------------------------------------------------------------------ 内部
    def _query_any_tier(self, station: Any) -> tuple[dict, str]:
        """按档位梯子查询并解析，返回 (parse_area_response 结果, 档位名)。

        探测梯内所有失败（HTTP 4xx、网络穷尽、"200 + 业务失败"）都降档重试，
        档位只在解析成功后缓存——200 不等于该档位可用，避免把无权限档位固定进
        _tier_cache；全部档位穷尽后抛聚合错误（保留各档失败原因，根因可定位）。
        已固定档位复用时仅业务级失效回梯子自愈，其余错误按原样上抛。
        """
        if self._tier_cache is not None:
            tier = self._tier_cache
            suffix = dict(TIERS)[tier]
            try:
                return parse_area_response(self._request(station, suffix)), tier
            except _TierUnavailable as e:
                # 已固定档位中途变为业务不可用（配额/权限变更等）：作废缓存回梯子，
                # 其余 RuntimeError（契约漂移）按原样上抛，不回梯子
                logger.warning("星图已固定档位 %s 业务层不可用，作废缓存重新降档: %s", tier, e)
                self._tier_cache = None
        errors: list[str] = []
        for tier, suffix in TIERS:
            try:
                payload = self._request(station, suffix)
                parsed = parse_area_response(payload)
                self._tier_cache = tier
                return parsed, tier
            except _TierUnavailable as e:
                errors.append(f"{tier}: {e}")
                logger.warning("星图档位 %s 业务层不可用，降档: %s", tier, e)
            except RuntimeError as e:
                errors.append(f"{tier}: {e}")
                logger.warning("星图档位 %s 查询失败: %s", tier, e)
        raise RuntimeError(f"星图全部档位查询失败: {errors!r}")

    def _request(self, station: Any, suffix: str) -> dict:
        params = {
            "location": f"{station.lon},{station.lat}",
            "token": self.token,
        }

        def _masked(err: Any) -> str:
            # token 走 URL query，底层异常消息会带完整 URL——入日志前必须掩码
            return str(err).replace(self.token, "***")

        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(AREA_URL + suffix, params=params,
                                        headers=HEADERS, timeout=self.timeout)
                status = getattr(resp, "status_code", None)
                if status == 200:
                    return resp.json()
                try:
                    body_digest = _masked((resp.text or ""))[:200]
                except Exception:  # noqa: BLE001
                    body_digest = ""
                if status == 401 or status == 403:
                    raise RuntimeError(
                        f"星图鉴权/权限失败（HTTP {status} {body_digest!r}）——"
                        f"请确认 {TOKEN_ENV} 有效且账号已开通逐小时预报产品"
                    )
                if isinstance(status, int) and 400 <= status < 500 and status != 429:
                    raise _Rejected(f"HTTP {status} body={body_digest!r}")
                last_err = RuntimeError(f"HTTP {status} body={body_digest!r}")
            except RuntimeError as e:
                if "鉴权" in str(e):
                    raise
                last_err = RuntimeError(_masked(e))
            except _Rejected as e:
                # 确定性失败（4xx）重试无意义，直接上抛
                raise RuntimeError(f"星图请求被拒: {e}") from e
            except Exception as e:  # noqa: BLE001  网络类异常/5xx → 可重试
                last_err = RuntimeError(_masked(e))
            logger.warning("星图请求失败（第%d次）: %s", attempt + 1, last_err)
            if attempt < self.retries:
                time.sleep(min(30, 3 * 2 ** attempt))
        raise RuntimeError(f"星图请求最终失败: {last_err}") from last_err


class _Rejected(Exception):
    """确定性失败（4xx，重试无意义）。"""

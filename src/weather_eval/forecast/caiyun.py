"""彩云天气（Caiyun Weather）v2.6 预报快照器。

认证方式：v2.6 约定 Token 直接置于 URL 路径，默认从环境变量 CAIYUN_TOKEN 读取。
端点：https://api.caiyunapp.com/v2.6/{token}/{经度,纬度}/weather.json

─────────────────────────────────────────────────────────────────────
时间对齐（第一性原理，关键设计约束）
─────────────────────────────────────────────────────────────────────
彩云返回的逐小时时间序列锚定在"请求时刻"，时间戳形如
``2026-08-27T15:10+08:00``：分钟随请求时刻波动，且带 ``+08:00`` 时区偏移。
而评估体系的观测（eia-data）落在整点（如 ``15:00``），评估引擎按时间字符串
精确配对（obs_map.get(tstr)）。因此必须把彩云时间戳：
  1. 解析为北京时（去掉偏移，得到墙钟时间）；
  2. 下取整到整点；
才能与整点观测配对。逐小时量级下 5~10 分钟的偏差对 1 小时间隔评估可忽略。
若不下取整，则 ``15:10`` 永远匹配不到 ``15:00`` 的观测，导致全部样本丢失。

时效范围（重要！）：本评估请求 hourlysteps=384（覆盖约 16 天）。经验证，彩云对该
测试 Token 的"长时效"返回与 User-Agent 强相关——使用默认 python-requests UA 仅返回
约 48 个逐小时点，而本项目固定 UA 可返回完整 384 点。因此这里必须固定 UA，
否则评估会静默丢失约 14 天样本且无任何报错。提供方在返回点数明显少于请求时也会
记录 WARNING，把这种静默降级暴露出来。
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

ENDPOINT = "https://api.caiyunapp.com/v2.6/{token}/{lonlat}/weather.json"
HEADERS = {"User-Agent": "weather-api-eval/0.1 (+https://github.com/)"}
DEFAULT_NAME = "caiyun_v2_6"
TOKEN_ENV = "CAIYUN_TOKEN"
# 请求大值由服务端截断为实际上限（约 48），不影响解析。
MAX_HOURLY_STEPS = 384


def _parse_caiyun_dt(s: str) -> datetime:
    """把 '2026-08-27T15:10+08:00' 解析为北京时 naive datetime 并下取整到整点。

    彩云时间已带 +08:00，与北京时一致，故转换后仅取墙钟并去偏移；
    若响应异常缺少偏移，回退为按 'YYYY-MM-DDTHH:MM' 解析后下取整。
    """
    try:
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is not None:
            dt = dt.astimezone(BEIJING).replace(tzinfo=None)
        return dt.replace(minute=0, second=0, microsecond=0)
    except ValueError:
        base = s[:16]
        return datetime.strptime(base, "%Y-%m-%dT%H:%M").replace(
            minute=0, second=0, microsecond=0
        )


def _redact(text: str, token: str) -> str:
    """把文本中的 Token 抹掉，避免它随异常/日志泄露（Token 在 URL 路径里）。"""
    if token:
        text = text.replace(token, "***")
    return text


class CaiyunProvider(ForecastProvider):
    def __init__(
        self,
        token: str | None = None,
        name: str = DEFAULT_NAME,
        timeout: int = 60,
        retries: int = 3,
        session: requests.Session | None = None,
    ):
        self.token = token if token is not None else os.environ.get(TOKEN_ENV)
        if not self.token:
            raise RuntimeError(
                f"彩云 Token 未提供：请在环境变量 {TOKEN_ENV} 中设置，"
                f"或在构造 CaiyunProvider 时显式传入 token。"
            )
        self.name = name
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()

    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> dict:
        lonlat = f"{station.lon},{station.lat}"
        url = ENDPOINT.format(token=self.token, lonlat=lonlat)
        params = {
            "hourlysteps": MAX_HOURLY_STEPS,
            "dailysteps": 1,
            "unit": "metric",
            "lang": "zh_CN",
        }
        payload = self._request(url, params)

        if payload.get("status") != "ok" or payload.get("api_status") != "active":
            err = payload.get("error", "unknown")
            raise RuntimeError(
                f"彩云 API 返回异常（疑似鉴权失败）: status={payload.get('status')!r} "
                f"api_status={payload.get('api_status')!r} error={err!r}"
            )

        hourly = payload.get("result", {}).get("hourly", {})
        temp_series = hourly.get("temperature") or []
        precip_series = hourly.get("precipitation") or []
        if not temp_series:
            raise RuntimeError(f"彩云站点 {station.id} 返回空逐小时温度序列")

        # 以温度序列时间为基准时间轴；降水按（下取整后）时间对齐，缺失补 None
        p_by_dt: dict[datetime, Any] = {}
        for p in precip_series:
            try:
                p_by_dt[_parse_caiyun_dt(p["datetime"])] = p.get("value")
            except (KeyError, ValueError):
                continue

        hourly_time: list[str] = []
        temps: list[float | None] = []
        precips: list[float | None] = []
        for t in temp_series:
            try:
                dt = _parse_caiyun_dt(t["datetime"])
            except (KeyError, ValueError):
                continue
            hourly_time.append(dt.strftime("%Y-%m-%dT%H:%M"))
            temps.append(t.get("value"))
            precips.append(p_by_dt.get(dt))

        if not hourly_time:
            raise RuntimeError(f"彩云站点 {station.id} 未解析出有效逐小时时间点")

        # 长时效可能被 Token/User-Agent 限制截断（默认 UA 仅返回约 48 点），
        # 此时评估会静默丢失样本——显式告警以暴露降级。
        if len(hourly_time) < MAX_HOURLY_STEPS // 2:
            logger.warning(
                "彩云站点 %s 仅返回 %d 个逐小时点（请求 %d，约 %dh）。长时效可能被 Token/User-Agent 限制截断，"
                "评估仅覆盖约 %dh；若预期为完整 16 天，请检查 User-Agent 与 Token 权限。",
                station.id, len(hourly_time), MAX_HOURLY_STEPS, len(hourly_time), len(hourly_time),
            )

        # 以响应 location[lon,lat] 为实际格点（彩云不做吸附，通常等于请求坐标）
        loc = payload.get("location")
        if isinstance(loc, (list, tuple)) and len(loc) == 2:
            grid_lon, grid_lat = float(loc[0]), float(loc[1])
        else:
            grid_lat, grid_lon = station.lat, station.lon

        issue_iso = hourly_time[0]
        snapshot = {
            "issue_iso": issue_iso,
            "station_id": station.id,
            "source": "caiyun",
            "models": [self.name],
            # 彩云不做格点吸附；以响应 location[lon,lat] 为准（通常等于请求坐标），响应无 elevation
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "elevation": None,
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            "hourly_time": hourly_time,
            "data": {
                self.name: {"temperature_2m": temps, "precipitation": precips}
            },
        }
        logger.info(
            "站点 %s 已抓取彩云起报 %s，时间点数 %d（逐小时时效约 %.0f h）",
            station.id, issue_iso, len(hourly_time), len(hourly_time),
        )
        return snapshot

    def _request(self, url: str, params: dict) -> dict:
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(
                    url, params=params, headers=HEADERS, timeout=self.timeout
                )
                resp.raise_for_status()
                return resp.json()
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("彩云站点请求失败（第%d次）: %s", attempt + 1, _redact(str(e), self.token))
                time.sleep(3 * (attempt + 1))
        raise RuntimeError(f"彩云请求失败: {_redact(str(last_err), self.token)}") from last_err

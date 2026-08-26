"""Open-Meteo 预报快照器。

请求多模型时，返回键名带模型后缀（temperature_2m_ecmwf_ifs 等）；单模型不带后缀。
本实现统一请求多模型并做兼容解析。坐标会被 Open-Meteo 吸附到最近格点，
响应中的 latitude/longitude/elevation 即真实格点，存档记录。
"""
from __future__ import annotations

import logging
import time
from typing import Any

import requests

from .base import ForecastProvider

logger = logging.getLogger(__name__)

ENDPOINT = "https://api.open-meteo.com/v1/forecast"
HEADERS = {"User-Agent": "weather-api-eval/0.1 (+https://github.com/)"}


def _model_key(hourly_units: dict, base: str, model: str) -> str | None:
    """在多模型/单模型两种返回形态下，找到 base 变量对应的真实键名。"""
    suffixed = f"{base}_{model}"
    if suffixed in hourly_units:
        return suffixed
    if base in hourly_units:
        return base
    return None


class OpenMeteoProvider(ForecastProvider):
    def __init__(self, timeout: int = 60, retries: int = 3, session: requests.Session | None = None):
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()

    def fetch_snapshot(self, station: Any, models: list[str]) -> dict:
        params = {
            "latitude": station.lat,
            "longitude": station.lon,
            "hourly": "temperature_2m,precipitation",
            "models": ",".join(models),
            "forecast_days": 16,
            "timezone": "Asia/Shanghai",
            "temperature_unit": "celsius",
            "precipitation_unit": "mm",
        }
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(ENDPOINT, params=params, headers=HEADERS, timeout=self.timeout)
                resp.raise_for_status()
                payload = resp.json()
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("Open-Meteo 站点 %s 请求失败（第%d次）: %s", station.id, attempt + 1, e)
                time.sleep(3 * (attempt + 1))
        else:
            raise RuntimeError(f"Open-Meteo 站点 {station.id} 请求失败: {last_err}") from last_err

        hourly = payload.get("hourly", {})
        hourly_units = payload.get("hourly_units", {})
        times = hourly.get("time", [])
        if not times:
            raise RuntimeError(f"Open-Meteo 站点 {station.id} 返回空时间序列")

        issue_iso = times[0]  # 起报时刻 = 响应共享时间轴首点（北京时），与 hourly_time 同口径
        data: dict[str, dict] = {}
        for model in models:
            tkey = _model_key(hourly_units, "temperature_2m", model)
            pkey = _model_key(hourly_units, "precipitation", model)
            if tkey is None or pkey is None:
                logger.warning("模型 %s 在返回中缺失，跳过", model)
                continue
            tarr = hourly.get(tkey)
            parr = hourly.get(pkey)
            if tarr is None or parr is None:
                logger.warning("模型 %s 的数据数组缺失，跳过", model)
                continue
            data[model] = {"temperature_2m": tarr, "precipitation": parr}
        if not data:
            raise RuntimeError("Open-Meteo 未返回任何请求的模型数据")

        snapshot = {
            "issue_iso": issue_iso,
            "station_id": station.id,
            "source": "open-meteo",
            "models": list(data.keys()),
            "grid_lat": payload.get("latitude"),
            "grid_lon": payload.get("longitude"),
            "elevation": payload.get("elevation"),
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            "hourly_time": times,
            "data": data,
        }
        logger.info("站点 %s 已抓取起报 %s，模型 %s，时间点数 %d",
                    station.id, issue_iso, list(data.keys()), len(times))
        return snapshot

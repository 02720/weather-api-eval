"""Open-Meteo 预报快照器。

请求多模型时，返回键名带模型后缀（temperature_2m_ecmwf_ifs 等）；单模型不带后缀。
本实现统一请求多模型并做兼容解析。坐标会被 Open-Meteo 吸附到最近格点，
响应中的 latitude/longitude/elevation 即真实格点，存档记录。

─────────────────────────────────────────────────────────────────────
起报锚点语义（与其他源的已知差异，评估/解读时必须知情）
─────────────────────────────────────────────────────────────────────
Open-Meteo 不回显底层模式的真实起报轮次，共享时间轴首点固定是北京时"当日 00:00"
（timezone=Asia/Shanghai 的当日零点）。它与真实 init（00/06/12/18Z 最近一轮）相差
−8~+12h 且随抓取时刻漂移，而彩云/和风/星图/AccuWeather 锚定抓取时刻（滚动预报）、
天机/伏羲/风乌锚定真实轮次——同一"提前 N 天"桶内各源的实际预报难度因此略有不同，
跨源解读分时效榜时应把该差异计入（详见 README"起报锚点"说明）。
同时本源快照按（站×模型×当日 00:00）幂等，每天仅产生 1 份快照（13/20 点的抓取
全部被幂等跳过），样本新鲜度构成与其他逐时快照的源也不可比。评估按快照自身的
锚点计算 lead，口径内部自洽；差异属"披露给读者"而非"可修正"项。

─────────────────────────────────────────────────────────────────────
逐日预报块（可选扩展，契约见 forecast/base.py）
─────────────────────────────────────────────────────────────────────
一次请求同时要 hourly 与 daily 两组变量，成本为零（仍是 1 次调用）。daily 块
的语义（本源是"逐日补位"的参考实现，其他源照此接入）：
- `timezone=Asia/Shanghai`，故 daily.time 的日期就是**北京时自然日**，与按天
  评估的日界一致（其他源接入前必须先确认它的日界是否也是北京时自然日——
  中国天气网那类"白天/夜间"日界与本口径不同，不能照抄）。
- `temperature_2m_max/min` 是 Open-Meteo 按逐小时序列取的日内极值，与本项目
  逐小时聚合口径等价；`precipitation_sum` 是当日 24h 累计（00:00–24:00）。
- 日界差异（必须知情）：观测侧日降水是 Σ rain@t（t 取当日 00:00–23:00 整点），
  窗口实为 (前日 23:00, 当日 23:00]；日产品的窗口是 [00:00, 24:00]。两者相差
  约 1 小时的边界，属补位口径的固有差异（README 已披露，不修正、不隐藏）。
- 本源逐小时已覆盖完整 16 天（锚点当日 00:00 + 384h，末日均整 24 点），补位实际
  不会触发；这里封存日产品的价值是**参考实现 + 口径对照**（2026-09-05 实测：
  日产品的日极值与逐小时聚合逐日完全一致，最大偏差 0.0°C；日降水逐日相等），
  并让逐日补位链路在本仓库的 CI 里被真实数据走到。逐小时覆盖短于逐日的源
  （和风 10d、中科星图 5d、风乌 7d、AccuWeather 10d、MSN 9.3d…）接入同一契约
  后才产生真正的时效增益。
- 实测响应形态：daily.time 16 天但**末日的日产品值为 null**（逐小时却有完整
  24 点）——尾部 null 与数组越界同样按缺测处理，绝不前移对齐、绝不造值。
- daily 块缺失（契约漂移）以 WARNING 暴露并退化为不带该块的快照：逐小时是主干，
  日产品只是延长线，绝不因它丢掉整份起报快照。
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


def _model_key(units: dict, base: str, model: str, allow_bare: bool = True) -> str | None:
    """在多模型/单模型两种返回形态下，找到 base 变量对应的真实键名。

    逐小时与逐日两组变量共用本函数（键名规则相同：多模型带模型后缀）。

    多模型请求（allow_bare=False）只接受带模型后缀的键：裸键回退仅对单模型请求
    开放，否则若多模型响应里混入裸键，会让多个模型静默映射到同一数组。
    """
    suffixed = f"{base}_{model}"
    if suffixed in units:
        return suffixed
    if allow_bare and base in units:
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
            # 逐日块与逐小时同一次请求取回（不额外占调用次数）：用于逐小时断供后
            # 的按天评估补位，语义见模块 docstring"逐日预报块"一节
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
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

        # 锚点语义见模块 docstring：Open-Meteo 无真实起报轮次回显，首点=当日 00:00
        issue_iso = times[0]
        allow_bare = len(models) == 1  # 仅单模型请求允许裸键回退（见 _model_key）
        data: dict[str, dict] = {}
        for model in models:
            tkey = _model_key(hourly_units, "temperature_2m", model, allow_bare=allow_bare)
            pkey = _model_key(hourly_units, "precipitation", model, allow_bare=allow_bare)
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

        daily_block = _parse_daily(payload, models, allow_bare, station.id)
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
        if daily_block:
            snapshot["daily_time"] = daily_block["time"]
            snapshot["daily"] = daily_block["data"]
        logger.info("站点 %s 已抓取起报 %s，模型 %s，时间点数 %d，日产品 %d 天",
                    station.id, issue_iso, list(data.keys()), len(times),
                    len(daily_block.get("time", ())) if daily_block else 0)
        return snapshot


# 逐日块的三要素 → daily 响应键前缀（缺任一即该模型不入块）
_DAILY_VARS = (
    ("temperature_2m_max", "temp_max"),
    ("temperature_2m_min", "temp_min"),
    ("precipitation_sum", "precipitation"),
)


def _parse_daily(payload: dict, models: list[str], allow_bare: bool,
                 station_id: str) -> dict | None:
    """解析响应中的 daily 组 → {"time": [...], "data": {model: {...}}}。

    缺块/缺时间轴/所有模型都缺变量时返回 None（调用方退化为不带该块的快照）：
    逐日块是可选扩展，绝不因为它缺失而丢掉整份逐小时起报快照——起报快照错过
    即无法追补，而日产品只是延长线。缺块本身以 WARNING 暴露（契约漂移可见）。
    """
    daily = payload.get("daily")
    dtimes = daily.get("time") if isinstance(daily, dict) else None
    if not isinstance(dtimes, list) or not dtimes:
        logger.warning("Open-Meteo 站点 %s 响应缺少 daily 组（契约漂移），"
                       "本次快照不带逐日预报块，按天评估将只用逐小时聚合", station_id)
        return None
    units = payload.get("daily_units") or {}
    data: dict[str, dict] = {}
    for model in models:
        resolved: list[tuple[str, str]] = []   # [(响应键, 输出名)]
        for base, out in _DAILY_VARS:
            key = _model_key(units, base, model, allow_bare=allow_bare)
            if key is None:
                logger.warning("模型 %s 在 daily 组中缺变量 %s，该模型不写入逐日块",
                               model, base)
                resolved = []
                break
            arr = daily.get(key)
            resolved.append((key, out))
            if not isinstance(arr, list):
                logger.warning("模型 %s 的 daily 变量 %s 缺失或非数组，该变量按全缺测"
                               "写入逐日块", model, key)
        if not resolved:
            continue
        data[model] = {out: (daily.get(resp) if isinstance(daily.get(resp), list) else None)
                       for resp, out in resolved}
    if not data:
        logger.warning("Open-Meteo 站点 %s 的 daily 组未解析出任何模型，"
                       "本次快照不带逐日预报块", station_id)
        return None
    return {"time": [str(t)[:10] for t in dtimes], "data": data}

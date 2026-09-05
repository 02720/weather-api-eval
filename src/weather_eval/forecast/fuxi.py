"""伏羲中期（FuXi-C88）预报快照器 —— 抓取 fuxi-ai.cn 可视化页面背后的接口。

数据源：https://fuxi-ai.cn/visual/weather（伏羲大模型可视化）页面背后的自有网关
（网页抓取，游客态可用、无需任何凭据）：
  GET  https://fuxi-ai.cn/gw/weather/api/v1/weather/queryWeatherTile
  POST https://fuxi-ai.cn/gw/weather/api/v1/weather/queryWeatherInfo
       body: {"lat": <纬度>, "lon": <经度>, "forecastType": "1"}

─────────────────────────────────────────────────────────────────────
关键契约（2026-08 线上实测 + 前端 JS 逆向，改动前必须复核）
─────────────────────────────────────────────────────────────────────
1. 模型标识：forecastType "1"=伏羲中期（FuXi-C88）、"2"=伏羲次季节。本提供方只接
   中期（"1"）；伏羲确定性（FuXi-Det）走另一数据服务 API（见 fuxi_data.py），二者
   是不同产品线，不得混用。
2. 起报锚点（最关键的隐性契约）：queryWeatherInfo 的响应里**没有任何绝对时间**，
   只有 step 1..360；起报时刻必须从 queryWeatherTile 拿：
     data[forecastType=="1"].startTime = "YYYYMMDDHH"，**UTC 语义**
     （前端用 moment.utc(...).local() 解析；且实测 startTime=...12 时 step1 的
     ssrd≈-0.2、t2m=27.5，对应北京时 21 时夜间——若按北京时解析则应为正午、
     ssrd 应达数百 W/m²，矛盾）。因此北京时起报 = startTime(UTC) + 8h，
     逐点时刻 = 北京时起报 + step 小时。
3. 发布延迟：c88 可视化产品线发布明显滞后（实测 15 时最新锚点仍是前一日 12Z 轮），
   属正常现象；本提供方不回退探测（接口无起报参数，锚点与点位数据由服务端绑定）。
4. 数值口径（响应值为字符串，须 float 化；缺测/非法 → None）：
   - t2m：℃（已换算，与数据服务 API 的 K 不同）
   - tp：页面图例 unit="mm/h"（逐小时降水率），作为"前 1 小时累计"的近似与观测
     rain@t 配对（与中科天机 pratesfc 同一口径）。注意数据服务 /models 元数据把
     c88 的 tp 标为"Total precipitation, mm"——两条产品线口径可能不同，若后续
     对比发现日降水系统性偏大 ~6 倍，应复核此处。
5. tile 锚点与点位数据是两个接口，理论上有极小概率读到不同轮次（点位先更新）。
   无从校验（点位响应不回显起报），属已知风险，anchor 变化时会自然形成新快照。

逐日预报块（不接入，2026-09-05 留档）：可视化产品线只有逐小时（step 1..360），
无逐日极值产品；数据服务的次季节线（forecastType "2"，FuXi-s2s）虽有 t_min/t_max，
但那是 1.5° 次季节分位值产品线（docstring 第 1 条明令不得与中期混用），且非
自然日口径。本源逐小时已覆盖 15 天，仅第 16 天按天样本缺口接受为已知边界。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import requests

from .base import ForecastProvider

logger = logging.getLogger(__name__)

TILE_URL = "https://fuxi-ai.cn/gw/weather/api/v1/weather/queryWeatherTile"
INFO_URL = "https://fuxi-ai.cn/gw/weather/api/v1/weather/queryWeatherInfo"
HEADERS = {"User-Agent": "weather-api-eval/0.1 (+https://github.com/)"}

SOURCE = "fuxi"
MODEL_NAME = "fuxi_c88"
FORECAST_TYPE_C88 = "1"          # 伏羲中期；"2" 为次季节（不接入）
EXPECTED_HOURS = 360             # c88 逐小时 15 天（与 tile totalStep 一致）


def _num(v: Any) -> float | None:
    """响应值为字符串：非法/缺测一律 None，绝不伪装成 0。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def parse_tile_start_time(payload: Any) -> str:
    """从 queryWeatherTile 响应取 c88（forecastType="1"）的 UTC 起报 YYYYMMDDHH。"""
    if not isinstance(payload, dict) or not payload.get("success"):
        raise RuntimeError(f"伏羲 tile 响应异常: {payload!r}"[:300])
    for item in payload.get("data") or []:
        if isinstance(item, dict) and str(item.get("forecastType")) == FORECAST_TYPE_C88:
            st = item.get("startTime")
            if isinstance(st, str) and len(st) == 10 and st.isdigit():
                return st
            raise RuntimeError(f"伏羲 tile 的 startTime 非法: {st!r}")
    raise RuntimeError("伏羲 tile 响应中无 forecastType=1（伏羲中期）条目，产品线可能已下线")


def tile_start_to_issue_iso(start_utc: str) -> str:
    """UTC YYYYMMDDHH → 北京时 issue "YYYY-MM-DDTHH:00"（UTC+8，见模块 docstring 第 2 点）。"""
    utc = datetime.strptime(start_utc, "%Y%m%d%H")
    bj = utc + timedelta(hours=8)
    return bj.strftime("%Y-%m-%dT%H:00")


def parse_weather_info(payload: Any, issue_iso: str) -> dict[str, list]:
    """把 queryWeatherInfo 响应展开为以北京时起报为锚的逐小时序列。

    返回 {"time": [...], "temperature_2m": [...], "precipitation": [...]}；
    时刻 = 北京时起报 + step 小时。step 缺失/非法的条目跳过。
    """
    if not isinstance(payload, dict) or not payload.get("success"):
        raise RuntimeError(f"伏羲点位响应异常: {payload!r}"[:300])
    data = payload.get("data") or {}
    items = data.get("weatherInfoList")
    if not isinstance(items, list) or not items:
        raise RuntimeError("伏羲点位响应 weatherInfoList 为空（起报可能尚未发布或契约漂移）")
    issue = datetime.strptime(issue_iso, "%Y-%m-%dT%H:%M")
    rows: list[tuple[datetime, float | None, float | None]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            step = int(it.get("step"))
        except (TypeError, ValueError):
            continue
        t = issue + timedelta(hours=step)
        rows.append((t, _num(it.get("t2m")), _num(it.get("tp"))))
    rows.sort(key=lambda r: r[0])
    # 去重（同一 step 重复时保留首见），保证时间轴严格递增
    seen: dict[str, tuple[float | None, float | None]] = {}
    for t, tv, pv in rows:
        seen.setdefault(t.strftime("%Y-%m-%dT%H:00"), (tv, pv))
    times = list(seen.keys())
    return {
        "time": times,
        "temperature_2m": [seen[t][0] for t in times],
        "precipitation": [seen[t][1] for t in times],
    }


class FuxiC88Provider(ForecastProvider):
    """伏羲中期（FuXi-C88）快照器：单模型、共享时间轴，返回单份快照 dict。"""

    def __init__(self, timeout: int | tuple = (10, 60), retries: int = 3,
                 session: requests.Session | None = None):
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self._tile_cache: str | None = None  # tile 锚点为产品级属性，跨站点复用

    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> dict:
        start_utc = self._tile_cache
        if start_utc is None:
            payload = self._request(TILE_URL, method="GET")
            start_utc = parse_tile_start_time(payload)
            self._tile_cache = start_utc
        issue_iso = tile_start_to_issue_iso(start_utc)

        info = self._request(
            INFO_URL, method="POST",
            json_body={"lat": station.lat, "lon": station.lon,
                       "forecastType": FORECAST_TYPE_C88},
        )
        series = parse_weather_info(info, issue_iso)
        if not series["time"]:
            # 空时间轴快照一旦入库会被同 issue 幂等锁死，正常数据永远进不来
            raise RuntimeError(
                "伏羲点位响应解析出 0 个有效逐小时点（疑似契约漂移），拒绝入库")
        if len(series["time"]) < EXPECTED_HOURS:
            logger.warning(
                "伏羲中期站点 %s 仅返回 %d 个逐小时点（预期约 %d），长时效可能被截断",
                station.id, len(series["time"]), EXPECTED_HOURS,
            )
        if series["temperature_2m"] and all(v is None for v in series["temperature_2m"]):
            logger.warning("伏羲中期站点 %s 温度序列全部缺测，服务端契约可能已变化", station.id)
        if series["precipitation"] and all(v is None for v in series["precipitation"]):
            logger.warning(
                "伏羲中期站点 %s 降水序列全部缺测，本快照降水将计为缺测（评估显示样本不足）",
                station.id,
            )
        snapshot = {
            "issue_iso": issue_iso,
            "station_id": station.id,
            "source": SOURCE,
            "models": [MODEL_NAME],
            # 该 API 不做格点吸附（响应不回显坐标）；无 elevation 字段
            "grid_lat": float(station.lat),
            "grid_lon": float(station.lon),
            "elevation": None,
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            "hourly_time": series["time"],
            "data": {MODEL_NAME: {
                "temperature_2m": series["temperature_2m"],
                "precipitation": series["precipitation"],
            }},
        }
        logger.info("站点 %s 已抓取伏羲中期起报 %s，时间点数 %d",
                    station.id, issue_iso, len(series["time"]))
        return snapshot

    # ------------------------------------------------------------------ 内部
    def _request(self, url: str, *, method: str = "GET",
                 json_body: dict | None = None) -> Any:
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                if method == "GET":
                    resp = self.session.get(url, headers=HEADERS, timeout=self.timeout)
                else:
                    resp = self.session.post(url, json=json_body, headers=HEADERS,
                                             timeout=self.timeout)
                status = getattr(resp, "status_code", None)
                if status == 200:
                    return resp.json()
                try:
                    body_digest = (resp.text or "")[:200]
                except Exception:  # noqa: BLE001
                    body_digest = ""
                if isinstance(status, int) and 400 <= status < 500 and status != 429:
                    raise _Rejected(f"HTTP {status} body={body_digest!r}")
                last_err = RuntimeError(f"HTTP {status} body={body_digest!r}")
            except _Rejected as e:
                raise RuntimeError(f"伏羲请求被拒: {e}") from e
            except Exception as e:  # noqa: BLE001  网络类异常/5xx → 可重试
                last_err = e
            logger.warning("伏羲请求失败（第%d次）: %s", attempt + 1, last_err)
            if attempt < self.retries:
                time.sleep(min(30, 3 * 2 ** attempt))
        raise RuntimeError(f"伏羲请求最终失败: {last_err}") from last_err


class _Rejected(Exception):
    """确定性失败（4xx，重试无意义）。"""

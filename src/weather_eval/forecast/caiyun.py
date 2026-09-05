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

─────────────────────────────────────────────────────────────────────
降水口径（2026-08-30 标定，详见 README"降水口径"一节）
─────────────────────────────────────────────────────────────────────
彩云官方文档未明示逐小时 precipitation 的累计窗口方向。本项目按"前 1 小时累计"
（值@t 覆盖 (t−1h, t]，与观测 rain@t 同标注）直接透传入库，不做移位。已对首月
样本做 −1/0/+1h 整体平移标定：d=0 与 d=+1 的 TS 差异在噪声内（0.181 vs 0.185），
且彩云 POD 显著低于同场其他源的现象在任何平移下都存在（错位会随平移消失，
技巧差不会）——判断为预报技巧差异而非 1h 错位，无移位依据。若日后标定翻转
（某平移显著且稳定占优），应把换算落在本模块并在此留档。

─────────────────────────────────────────────────────────────────────
逐日预报块（可选扩展，契约见 forecast/base.py）
─────────────────────────────────────────────────────────────────────
同一请求把 dailysteps 提到 15（官方上限，默认 5）即可顺带取回逐日产品，零额外
调用。字段语义（v2.6 官方文档《天级别预报》实测核对）：
- `daily.temperature[]` 条目形如 {"date": "2026-09-06T00:00+08:00", "max": 27,
  "min": 18, "avg": ...}——max/min 即**全天最高/最低气温**，date 为当地自然日
  （站点均在国内，+08:00 即北京时自然日，与按天评估的日界一致）。
- **只接温度，降水置全 null**：`daily.precipitation[]` 条目是 {"max", "min",
  "avg", "probability"}——日内逐时降水量的统计量与概率，**没有 24h 累计字段**。
  "avg×24 = 全天累计"依赖"avg 为 24h 平均降水率"的未核实读法（对湿时平均/量级
  档等其它读法会系统性失真），在无 Token 实测对拍核实前不接入；本源逐小时本就
  覆盖满 16 天（384h），逐日产品（15 天）对补位无时效增益，接入价值在逐小时被
  Token/User-Agent 截断时（约 48h）温度轨道仍能继续。若日后取得 Token 复核实
  了 avg 语义，可在本模块补上降水并把口径留档。
- daily 块缺失/解析为空以 WARNING 暴露并退化为不带该块的快照（逐小时是主干）。
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
# 逐日预报天数：官方上限 15（默认 5，一览表明示 dailysteps=15 可取满 15 天）。
# 不请求 16：超上限行为未文档化，不值得为 offset=15 一天拿主干请求冒险。
MAX_DAILY_STEPS = 15


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


def _daily_date(s: Any) -> str | None:
    """逐日条目 date（如 '2026-09-06T00:00+08:00'）→ 北京时自然日 'YYYY-MM-DD'。

    彩云 v2.3+ 的逐日时间带当地时区偏移；站点均在国内（+08:00），换算到北京时
    不改变日历日。非法/缺失返回 None，由调用方剔除该条。
    """
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        dt = datetime.fromisoformat(s.strip())
    except ValueError:
        return None
    if dt.tzinfo is not None:
        dt = dt.astimezone(BEIJING).replace(tzinfo=None)
    return dt.date().isoformat()


def _num(v: Any) -> float | None:
    """逐日数值归一：null/非法一律 None，绝不伪装成 0（与逐小时路径同口径）。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if f != f else f


def parse_daily_block(payload: Any, station_id: str,
                      model: str = DEFAULT_NAME) -> dict | None:
    """解析响应中的 daily 组 → {"time": [...], "data": {模型: {...}}}。

    只接温度（docstring"逐日预报块"一节）：temp_max/temp_min 取
    daily.temperature[].max/min，precipitation 按契约给全 null 数组。
    块缺失/temperature 缺失或解析不出任何有效日 → 返回 None（调用方退化为
    不带该块的快照并以 WARNING 暴露）；单条 date 非法跳过、max/min 缺测为 None。
    """
    result = payload.get("result") if isinstance(payload, dict) else None
    daily = result.get("daily") if isinstance(result, dict) else None
    temps = daily.get("temperature") if isinstance(daily, dict) else None
    if not isinstance(temps, list) or not temps:
        logger.warning("彩云站点 %s 响应缺少 daily.temperature（契约漂移或 dailysteps "
                       "被截断），本次快照不带逐日预报块，按天评估将只用逐小时聚合",
                       station_id)
        return None
    rows: dict[str, tuple[float | None, float | None]] = {}
    for ent in temps:
        if not isinstance(ent, dict):
            continue
        day = _daily_date(ent.get("date"))
        if day is None:
            logger.warning("彩云站点 %s 逐日条目 date 无法解析（%r），跳过",
                           station_id, ent.get("date"))
            continue
        rows.setdefault(day, (_num(ent.get("max")), _num(ent.get("min"))))
    if not rows:
        logger.warning("彩云站点 %s 的 daily.temperature 未解析出任何有效日，"
                       "本次快照不带逐日预报块", station_id)
        return None
    days = sorted(rows)
    data = {
        model: {
            "temp_max": [rows[d][0] for d in days],
            "temp_min": [rows[d][1] for d in days],
            # 逐日累计降水产品不存在（日内统计量≠累计）：全 null，绝不折算
            "precipitation": [None] * len(days),
        },
    }
    return {"time": days, "data": data}


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
            # 逐日块与逐小时同一次请求取回（不额外占调用次数）：用于逐小时断供后
            # 的按天评估补位，语义见模块 docstring"逐日预报块"一节
            "dailysteps": MAX_DAILY_STEPS,
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
        daily_block = parse_daily_block(payload, station.id, model=self.name)
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
        if daily_block:
            snapshot["daily_time"] = daily_block["time"]
            snapshot["daily"] = daily_block["data"]
        logger.info(
            "站点 %s 已抓取彩云起报 %s，时间点数 %d（逐小时时效约 %.0f h），日产品 %d 天",
            station.id, issue_iso, len(hourly_time), len(hourly_time),
            len(daily_block["time"]) if daily_block else 0,
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

"""FengWu-GHR-9km 预报快照器 —— 抓取 fengwuai.com「简易气象查询」页面背后的接口。

数据源：https://fengwuai.com/simple-query 页面背后的公开查询 API（网页抓取）：
  GET https://fengwuai.com/api/v1/weather/availability?model_type=FengWu-GHR-9km&region=cn
  GET https://fengwuai.com/api/open/v1/weather/visual/query
      ?longitude={lon}&latitude={lat}&model_type=FengWu-GHR-9km&region=cn
      &forecast_time={YYYY-MM-DDTHH:00:00Z}

─────────────────────────────────────────────────────────────────────
关键契约（2026-08 线上实测 + 前端 JS 逆向，改动前必须复核）
─────────────────────────────────────────────────────────────────────
1. 鉴权与时效（API_KEY 的作用）：
   - 游客态：无任何凭据即可查询，但服务端把逐小时序列**截断到起报后 166h（7 天）**，
     且时间步长为 **3 小时**（56 个点）。
   - 填 API_KEY（环境变量 FENGWU_API_KEY，经 `Authorization: Bearer <key>` 请求头
     传递，与页面登录态一致）后解锁模型完整时效：逐小时、最长 360h（15 天）。
     Key 无效时服务端返回 401，本提供方直接报错而非静默退回游客时效。
2. 起报轮次：每天 4 轮 00/06/12/18 UTC。availability.api_end_time = 最新可查起报
   （UTC ISO），以它为首选；若查询 400（该轮次实际不可查/availability 滞后）则向
   过去逐轮（-6h）回退，最多 MAX_ISSUE_FALLBACK 轮。
3. 时间语义：请求 forecast_time 与响应 data[].time、forecast_time 均为 **UTC ISO**
   （Z 结尾）；逐点统一转北京时 naive 墙钟。游客序列从起报后 +1h 开始（+1,+4,+7…）。
4. 坐标：响应回显吸附到 9km 网格的实际坐标（如 111.304→111.33），作为 grid_lat/lon。
5. ── 时间分辨率与 6 小时降水的处理（本提供方的核心口径）──
   原始字段：t2m（K，须减 273.15）、tp6h（mm，**6 小时累计降水**，窗口为
   (t-6h, t]，即"截至该时刻的 6 小时累计"——由 ssrd1h/ssr6h 并存与 ERA5 惯例推证，
   实测起报后 +1h 的 tp6h 窗口大部分落在起报前，属产品固有形态）、u10/v10 等。
   评估契约需要逐小时序列，故做如下展开（在快照 meta 中留档）：
   - 温度：对 3h 采样做**线性插值**到逐小时（温度场平滑，插值误差可忽略）；
     若本身即为逐小时（有 Key），插值退化为恒等。
   - 降水：原始字段 tp6h（mm，**6 小时累计**，窗口为 (t-6h, t]，即"截至该时刻的
     6 小时累计"——由 ssrd1h/ssr6h 并存与 ERA5 惯例推证）。因采样间隔 3h < 窗口
     6h，相邻窗口重叠，直接逐窗均摊会重复计总量；本提供方取**相位平铺子集**
     （以首个采样为相位、每 6h 一个窗口端点，窗口两两无缝拼接），把每个平铺窗口
     的累计均摊 /6 到其 6 个小时（precip[h] = tp6h(t)/6，详见 spread_precip_6h）：
       a) 每小时恰属一个窗口，总量严格守恒（日降水 BIAS 不失真）；
       b) mm/h 速率与观测 rain@t（前 1h 累计）配对口径与中科天机 pratesfc 一致；
       c) 均摊对 0.1mm 晴雨阈值偏保守（短时强降水被摊薄），属已知局限。
     不改用"相邻窗口差分恢复逐小时"：tp6h 是滑动窗口（(t-6h, t]）而非自起报累计，
     W(t)−W(t−1) = rain(t−1,t] − rain(t−7h,t−6h]，差分残留 6 小时前的另一个 1h 量，
     序列无种子值不可解——差分在本数据形态下数学上不可恢复 1h 分辨率，均摊是
     唯一总量守恒的确定性展开。
6. region 固定 "cn"（评估站点均在国内；global 对国内点同样可用但范围校验不同）。

─────────────────────────────────────────────────────────────────────
逐日预报块（实测后决定不接入，2026-09-05 留档）
─────────────────────────────────────────────────────────────────────
官方变量表 `GET /api/v1/weather/variables?model_type=FengWu-GHR-9km&region=cn`
实测仅 16 个逐小时/短时累计要素（u10/v10/u100/v100/ws/wd/t2m/msl/sp/tp6h/tcc/
ssr6h/ssrd1h/ssr1h），**无任何逐日极值或逐日累计要素**；前端 API 面
（weather/models、weather/variables、weather/query、weather/availability、
api/open/v1/weather/visual/query）亦无逐日端点。本源按天轨道的缺口（游客 7 天/
持 Key 15 天逐小时）接受为已知边界；若日后上线逐日要素，按 base.py 契约接入。
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .base import ForecastProvider

logger = logging.getLogger(__name__)

AVAIL_URL = "https://fengwuai.com/api/v1/weather/availability"
QUERY_URL = "https://fengwuai.com/api/open/v1/weather/visual/query"
HEADERS = {"User-Agent": "weather-api-eval/0.1 (+https://github.com/)"}

SOURCE = "fengwu"
MODEL_NAME = "fengwu_ghr_9km"
MODEL_TYPE = "FengWu-GHR-9km"
REGION = "cn"
MAX_ISSUE_FALLBACK = 4          # 起报回退轮数（每轮 -6h）

KEY_ENV = "FENGWU_API_KEY"


def parse_iso_z(s: str) -> datetime:
    """UTC ISO（Z 结尾）→ naive UTC datetime。

    非零时区偏移（如 "+08:00" 结尾）会整体错 8h，显式拒绝以防契约漂移被静默误读。
    """
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1]
    elif s.endswith("z"):
        s = s[:-1]
    if len(s) == 16:
        s += ":00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        if dt.utcoffset() != timedelta(0):
            raise ValueError(f"风乌时间串带非零时区偏移，契约应为 UTC(Z): {s!r}")
        return dt.replace(tzinfo=None)
    return dt  # noqa: DTZ005  契约即 UTC


def to_bj(dt_utc: datetime) -> datetime:
    return dt_utc + timedelta(hours=8)


def _num(v: Any) -> float | None:
    """数值归一：非法/NaN/inf → None，绝不伪装成 0（与其余源同口径）。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    # NaN != NaN；inf 的 isfinite 为 False
    return f if (f == f and f not in (float("inf"), float("-inf"))) else None


def interpolate_hourly(samples: list[tuple[datetime, float | None]]) \
        -> list[tuple[datetime, float | None]]:
    """把按时间升序的采样点线性插值到逐小时（首末采样之间；不外推）。

    端点缺测的区段（任一端为 None）不插值，产出 None；既有整点采样原样保留。
    """
    if len(samples) < 2:
        return list(samples)
    out: list[tuple[datetime, float | None]] = []
    for (t0, v0), (t1, v1) in zip(samples, samples[1:]):
        out.append((t0, v0))
        if v0 is None or v1 is None:
            continue
        span = int((t1 - t0).total_seconds() // 3600)
        for k in range(1, span):
            frac = k / span
            out.append((t0 + timedelta(hours=k), v0 + (v1 - v0) * frac))
    out.append(samples[-1])
    # 按小时取整并去重（采样可能落在非整点，先地板到整点）
    seen: dict[datetime, float | None] = {}
    for t, v in out:
        th = t.replace(minute=0, second=0, microsecond=0)
        seen.setdefault(th, v)
    return sorted(seen.items())


def spread_precip_6h(samples: list[tuple[datetime, float | None]],
                     hours: list[datetime]) -> list[float | None]:
    """6 小时累计降水 → 逐小时 mm/h 速率（非重叠平铺窗口均摊 /6）。

    采样间隔 3h < 窗口 6h，相邻窗口 (t_k-6h, t_k] 相互重叠——若把每个窗口都摊到
    自己的 6 个小时会重复计总量。故取**相位平铺子集**：以首个采样为相位、每隔 6h
    取一个窗口端点（如采样在 +1,+4,+7,+10… 则取 +1,+7,+13,…），这些窗口
    (t-6h, t] 两两无缝拼接、恰好覆盖时间轴；每个小时 h 归属包含它的唯一平铺窗口
    （端点 t ∈ [h, h+6h)），precip[h] = tp6h(t)/6。效果：
      a) 每小时恰属一个窗口，任意完整覆盖跨度上的求和 == 原始累计总量（BIAS 不失真）；
      b) mm/h 速率与观测 rain@t（前 1h 累计）配对口径 = 中科天机 pratesfc 同款近似；
      c) 6h 窗口均摊对 0.1mm 晴雨阈值偏保守（短时强降水被摊薄），属已知局限。
    未被平铺窗口覆盖的小时（首窗之前/末窗之后）为 None。
    """
    by_time: dict[datetime, float | None] = {}
    for t, v in samples:
        by_time.setdefault(t, v)  # 重复时刻保留首见（与温度插值去重口径一致）
    times = sorted(by_time)
    if not times or not hours:
        return [None] * len(hours)
    t0 = times[0]
    tile = [t for t in times
            if (t - t0) % timedelta(hours=6) == timedelta(0)]
    out: list[float | None] = []
    for h in hours:
        tk = next((t for t in tile if h <= t < h + timedelta(hours=6)), None)
        v = by_time[tk] if tk is not None else None
        out.append(None if v is None else v / 6.0)
    return out


class FengWuProvider(ForecastProvider):
    """FengWu-GHR-9km 快照器：单模型快照 dict；FENGWU_API_KEY 可选（延长时效）。"""

    def __init__(self, timeout: int | tuple = (10, 60), retries: int = 3,
                 session: requests.Session | None = None, api_key: str | None = None):
        self.api_key = api_key if api_key is not None else os.environ.get(KEY_ENV, "")
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self._issue_cache: datetime | None = None  # 最新可查起报（产品级，跨站点复用）

    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> dict:
        issue_utc = self._resolve_issue()
        payload = self._query(station, issue_utc)
        return self._build_snapshot(station, payload)

    # ------------------------------------------------------------------ 内部
    def _resolve_issue(self) -> datetime:
        """最新可查起报：availability.api_end_time 优先，查询 400 时逐轮回退。"""
        if self._issue_cache is not None:
            return self._issue_cache
        payload = self._request(
            AVAIL_URL, params={"model_type": MODEL_TYPE, "region": REGION})
        end = (payload or {}).get("api_end_time") if isinstance(payload, dict) else None
        if isinstance(end, str) and end:
            try:
                self._issue_cache = parse_iso_z(end)
                return self._issue_cache
            except ValueError:
                logger.warning("风乌 availability.api_end_time 无法解析: %r", end)
        # availability 不可用 → 从最近的整 6h 轮次直接猜
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None, minute=0,
                                                     second=0, microsecond=0)
        self._issue_cache = now_utc - timedelta(hours=now_utc.hour % 6)
        return self._issue_cache

    def _query(self, station: Any, issue_utc: datetime) -> dict:
        """查询点位序列；该起报 400（尚未可查）时向过去回退最多 MAX_ISSUE_FALLBACK 轮。"""
        last_err: Exception | None = None
        issue = issue_utc
        for _ in range(MAX_ISSUE_FALLBACK):
            try:
                payload = self._request(
                    QUERY_URL,
                    params={
                        "longitude": station.lon, "latitude": station.lat,
                        "model_type": MODEL_TYPE, "region": REGION,
                        "forecast_time": issue.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                )
                if isinstance(payload, dict) and isinstance(payload.get("data"), list) \
                        and payload["data"]:
                    if issue != issue_utc:
                        logger.info("风乌起报 %s 不可查，回退使用 %s",
                                    issue_utc.strftime("%Y-%m-%dT%H:%MZ"),
                                    issue.strftime("%Y-%m-%dT%H:%MZ"))
                    self._issue_cache = issue
                    return payload
                raise _Empty("响应 data 为空")
            except (_Rejected, _Empty) as e:
                last_err = e
                logger.warning("风乌起报 %s 查询失败: %s，尝试更早轮次",
                               issue.strftime("%Y-%m-%dT%H:%MZ"), e)
                issue -= timedelta(hours=6)
        raise RuntimeError(
            f"风乌最近 {MAX_ISSUE_FALLBACK} 个起报轮次均不可查: {last_err}")

    def _build_snapshot(self, station: Any, payload: dict) -> dict:
        issue_utc = parse_iso_z(str(payload.get("forecast_time")))
        issue_iso = to_bj(issue_utc).strftime("%Y-%m-%dT%H:00")
        rows: list[tuple[datetime, float | None, float | None]] = []
        for item in payload.get("data") or []:
            if not isinstance(item, dict):
                continue
            try:
                t = to_bj(parse_iso_z(str(item.get("time"))))
            except (ValueError, TypeError):
                continue
            values = item.get("values") or {}
            t2m = _num(values.get("t2m"))
            tp = _num(values.get("tp6h"))
            rows.append((
                t,
                None if t2m is None else t2m - 273.15,
                tp,
            ))
        if not rows:
            raise RuntimeError("风乌响应无可解析的数据点")
        rows.sort(key=lambda r: r[0])
        temp_samples = [(t, v) for t, v, _ in rows]
        tp_samples = [(t, v) for t, _, v in rows]
        hourly = interpolate_hourly(temp_samples)
        hours = [t for t, _ in hourly]
        precips = spread_precip_6h(tp_samples, hours)
        temps = [v for _, v in hourly]
        # 缺测健康度告警
        if all(v is None for v in temps):
            logger.warning("风乌站点 %s 温度序列全部缺测，服务端契约可能已变化", station.id)
        if all(v is None for v in precips):
            logger.warning("风乌站点 %s 降水序列全部缺测，本快照降水将计为缺测", station.id)
        grid = payload.get("longitude"), payload.get("latitude")
        grid_lat, grid_lon = station.lat, station.lon
        try:
            if grid[0] is not None and grid[1] is not None:
                grid_lon, grid_lat = float(grid[0]), float(grid[1])  # 响应先经度后纬度
        except (TypeError, ValueError):
            pass
        snapshot = {
            "issue_iso": issue_iso,
            "station_id": station.id,
            "source": SOURCE,
            "models": [MODEL_NAME],
            "grid_lat": grid_lat,
            "grid_lon": grid_lon,
            "elevation": None,
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            # 口径留档：3h 采样 → 逐小时插值；tp6h 窗口 (t-6h, t] 均摊 /6
            "expansion": "linear-interp-1h; precip=tp6h/6 spread over (t-6h,t]",
            "hourly_time": [t.strftime("%Y-%m-%dT%H:00") for t in hours],
            "data": {MODEL_NAME: {"temperature_2m": temps, "precipitation": precips}},
        }
        logger.info("站点 %s 已抓取风乌 GHR-9km 起报 %s，采样点 %d → 逐小时 %d",
                    station.id, issue_iso, len(rows), len(hours))
        return snapshot

    def _request(self, url: str, *, params: dict | None = None) -> Any:
        headers = dict(HEADERS)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(url, params=params, headers=headers,
                                        timeout=self.timeout)
                status = getattr(resp, "status_code", None)
                if status == 200:
                    return resp.json()
                try:
                    body_digest = (resp.text or "")[:200]
                except Exception:  # noqa: BLE001
                    body_digest = ""
                if status == 401:
                    # Key 无效属账号级错误，任何起报轮次都会同样失败——
                    # 绝不能当"该起报不可查"逐轮回退（浪费请求且错误消息误导）
                    raise RuntimeError(
                        f"风乌鉴权失败（HTTP 401 {body_digest!r}）——"
                        f"{KEY_ENV} 无效或已过期")
                if status == 400:
                    # 起报轮次不可查/参数问题：调用方按需回退，不重试
                    raise _Rejected(f"HTTP 400 body={body_digest!r}")
                if isinstance(status, int) and 400 <= status < 500 and status != 429:
                    raise RuntimeError(f"风乌请求被拒: HTTP {status} body={body_digest!r}")
                last_err = RuntimeError(f"HTTP {status} body={body_digest!r}")
            except RuntimeError as e:
                if "鉴权失败" in str(e) or "请求被拒" in str(e):
                    raise
                last_err = e
            except _Rejected:
                raise
            except Exception as e:  # noqa: BLE001  网络类异常/5xx → 可重试
                last_err = e
            logger.warning("风乌请求失败（第%d次）: %s", attempt + 1, last_err)
            if attempt < self.retries:
                time.sleep(min(30, 3 * 2 ** attempt))
        raise RuntimeError(f"风乌请求最终失败: {last_err}") from last_err


class _Rejected(Exception):
    """确定性失败（4xx，重试无意义）。"""


class _Empty(Exception):
    """200 但数据为空。"""

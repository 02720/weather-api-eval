"""伏羲确定性（FuXi-Det）预报快照器 —— fuxi-ai.cn 数据服务 API（fuxi-data 页）。

数据源：https://fuxi-ai.cn/fuxi-data（「伏羲数据」页）背后的数据服务网关
（需登录后于页面获取查询 Token，经环境变量 FUXI_DATA_TOKEN 注入）：
  POST https://fuxi-ai.cn/gw/fuxi-data/api/v1/initTime/isAvail
       body: {"modelId": "4", "initTime": "YYYY-MM-DD"}          # 游客可用
  POST https://fuxi-ai.cn/gw/fuxi-data/api/v1/queryWeatherInfo
       body: {"lon": ..., "lat": ..., "vars": ["t2m","tp"],
              "initTime": "YYYY-MM-DD HH:00:00", "model": "FuXi-Det"}
       headers: {"Authorization": "<token>"}                      # 原始 token，无 Bearer 前缀

─────────────────────────────────────────────────────────────────────
关键契约（2026-08 线上实测 + 前端 JS 逆向 + 官方页面内嵌示例代码，改动前必须复核）
─────────────────────────────────────────────────────────────────────
1. 模型：/models（游客可用）返回 FuXi-c88(id=1)/FuXi-s2s(id=2)/FuXi-ens(id=3)/
   FuXi-Det(id=4)。本提供方**只接 FuXi-Det**（伏羲确定性，0.1°，每天 00/06/12/18Z
   四轮）；FuXi-c88（中期）必须走可视化接口（fuxi.py），不得在此接入。
2. 起报探测：initTime/isAvail 的 initTime 只传**日期** "YYYY-MM-DD"（**UTC 日期**，
   服务端解析失败会回 "日期格式错误...invalid literal for int()"），响应 data 为该日
   可用 UTC 小时列表（如 ["00","06","12","18"]）。提供方从 UTC 今天向过去回退探测
   （跨日边界安全），取首个"非空小时列表"的最大小时作为起报。
3. initTime 时区：queryWeatherInfo 的 initTime = UTC 日期 + " " + UTC 小时 +
   ":00:00"（页面小时按钮带 "z" 后缀、官方示例即此格式）。
4. 响应结构（官方页面"响应参数"文档 + 代码逆向）：
     data.location   [纬度, 经度]（0.1° 网格吸附后实际坐标）
     data.time_fcst  起报时间标识（与请求 initTime 对应）
     data.timestamp  ["...Z", ...] UTC 预报时刻数组（时间轴）
     data.var_names  变量缩写列表（与请求 vars 对应）
     data.units      与 var_names 一一对应的单位（文档示例 "m/s、K、Pa" → t2m 可能为 K）
     data.values     values[i][j] = 第 i 个变量在第 j 个时刻的值
   单位自适应：t2m 按其 unit 判断（含 "K" → 减 273.15 转 ℃；℃/°C → 原值）。
5. 降水口径：tp 单位 mm（"Total precipitation"）。累计窗口官方未说明（FuXi 系基于
   ERA5 训练，可能为逐小时或 6 小时累计）。当前按"逐时刻累计 ≈ 前 1 小时累计"与
   观测配对；并做**单调性哨兵**：若 tp 序列近乎单调不减（起报累计式口径的典型形态），
   记 WARNING 提示口径存疑，便于事后发现（绝不静默）。
6. token 获取：登录 fuxi-ai.cn → fuxi-data 页 → 页面会调
   POST /gw/user/api/v1/api/v1/admin/accessToken/queryToken 换取查询 token
   （页面"获取查询令牌"）。401 "请求头中缺少id字段" = 未带/无效 token。
7. 快照粒度：单模型，返回单份快照 dict（共享时间轴形态）。

逐日预报块（不接入，2026-09-05 留档）：`/models`（游客可用，实测）中 FuXi-Det
的 vars_raw 仅逐小时要素（t2m/tp/u10/…），无逐日极值变量（无 mx2t/mn2t 类）；
带 t_max/t_min 的是 FuXi-s2s 次季节线（id=2，1.5°，分位值），与本产品线不同，
不接入。逐小时断供后的按天轨道缺口接受为已知边界。
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

BASE_URL = "https://fuxi-ai.cn/gw/fuxi-data/api/v1"
AVAIL_URL = f"{BASE_URL}/initTime/isAvail"
QUERY_URL = f"{BASE_URL}/queryWeatherInfo"
HEADERS = {"User-Agent": "weather-api-eval/0.1 (+https://github.com/)"}

SOURCE = "fuxi-data"
MODEL_NAME = "fuxi_det"
MODEL_CODE = "FuXi-Det"
MODEL_ID = "4"                    # /models 返回的 id，isAvail 要求字符串
VARS = ["t2m", "tp"]
MAX_DATE_FALLBACK = 4             # 起报探测：从 UTC 今天向过去最多回退天数

TOKEN_ENV = "FUXI_DATA_TOKEN"


def utc_now() -> datetime:
    """当前 UTC（naive）。独立出来便于测试注入。"""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def candidate_dates(now: datetime, count: int = MAX_DATE_FALLBACK) -> list[str]:
    """从 UTC 今天向过去生成候选日期 "YYYY-MM-DD"（降序）。"""
    return [(now - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(count)]


def parse_avail_hours(payload: Any) -> list[str]:
    """解析 isAvail 响应 → 可用 UTC 小时列表（如 ["00","06","12","18"]）。

    响应 msgCode=="10000" 且 data 为列表视为成功；data 为空列表 = 该日无数据（回退）。
    """
    if not isinstance(payload, dict):
        raise RuntimeError(f"伏羲数据 isAvail 响应异常: {payload!r}"[:300])
    msg_code = payload.get("msgCode")
    if msg_code is None and payload.get("data") is None:
        # 无 msgCode 且无 data 的响应（如 {"success":false,"msg":"token 无效"}）
        # 不能当作"该日无数据"静默回退
        raise RuntimeError(f"伏羲数据 isAvail 响应异常（无 msgCode/data）: {payload!r}"[:300])
    if msg_code is not None and msg_code not in ("10000", 10000, "200", 200):
        raise RuntimeError(
            f"伏羲数据 isAvail 业务错误: msg={payload.get('msg')!r} "
            f"msgCode={msg_code!r}"
        )
    data = payload.get("data")
    if data is None:
        return []
    if not isinstance(data, list):
        raise RuntimeError(f"伏羲数据 isAvail data 非列表: {data!r}")
    return [str(h) for h in data]


def parse_query_response(payload: Any) -> dict[str, list]:
    """把 queryWeatherInfo 响应展开为统一序列。

    返回 {"issue_echo": str|None, "grid_lat": float|None, "grid_lon": float|None,
          "time": ["YYYY-MM-DDTHH:00", ...]（北京时）, "t2m": [...], "tp": [...],
          "t2m_unit": str|None}
    timestamp 为 UTC 时刻数组，逐点转北京时墙钟；values[i][j] 按 var_names 定位。
    """
    if not isinstance(payload, dict):
        raise RuntimeError(f"伏羲数据响应异常: {payload!r}"[:300])
    msg_code = payload.get("msgCode")
    if msg_code not in ("10000", 10000, "200", 200, None):
        raise RuntimeError(
            f"伏羲数据 queryWeatherInfo 业务错误: msg={payload.get('msg')!r} "
            f"msgCode={msg_code!r}"
        )
    data = payload.get("data")
    if not isinstance(data, dict):
        raise RuntimeError("伏羲数据响应 data 缺失或非对象（token 失效/无数据）")
    timestamps = data.get("timestamp")
    var_names = data.get("var_names")
    values = data.get("values")
    if not (isinstance(timestamps, list) and timestamps
            and isinstance(var_names, list) and isinstance(values, list)):
        raise RuntimeError(
            f"伏羲数据响应结构不完整: keys={sorted(data.keys())}"
        )
    lower_names = [str(v).lower() if v is not None else "" for v in var_names]
    try:
        i_t2m = lower_names.index("t2m")
    except ValueError:
        raise RuntimeError(f"伏羲数据响应缺 t2m 变量: var_names={var_names}") from None
    try:
        i_tp = lower_names.index("tp")
    except ValueError:
        raise RuntimeError(f"伏羲数据响应缺 tp 变量: var_names={var_names}") from None

    units = data.get("units") if isinstance(data.get("units"), list) else []
    t2m_unit = str(units[i_t2m]) if i_t2m < len(units) and units[i_t2m] is not None else None

    def conv(idx: int) -> list[float | None]:
        row = values[idx] if idx < len(values) else []
        out: list[float | None] = []
        for v in row:
            if v is None or isinstance(v, bool):
                out.append(None)
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                out.append(None)
                continue
            if f != f:  # NaN
                out.append(None)
            elif idx == i_t2m and t2m_unit and "K" in t2m_unit.upper():
                # 单位为开尔文（文档示例 "m/s、K、Pa"）→ 转 ℃
                out.append(f - 273.15)
            else:
                out.append(f)
        return out

    t2m_raw = conv(i_t2m)
    # 量级哨兵：未声明 K 但数值普遍在开尔文量级（>150）→ 口径漂移预警
    if (t2m_unit is None or "K" not in t2m_unit.upper()):
        nums = [v for v in t2m_raw if v is not None]
        if len(nums) >= 6 and sum(1 for v in nums if v > 150) >= 0.9 * len(nums):
            logger.warning(
                "伏羲数据 t2m 未声明开尔文单位(%r)但数值呈 K 量级（如 %.1f）——"
                "疑似服务端口径漂移，请人工复核", t2m_unit, nums[0])
    if t2m_unit is None:
        logger.warning("伏羲数据响应未提供 t2m 单位，按 ℃ 处理（若实为 K 将失真）")

    # 时间轴与数值逐列对齐：无法解析的时刻整列剔除（绝不产出空串时间键毒化存档）
    times: list[str] = []
    kept: list[int] = []
    for j, ts in enumerate(timestamps):
        t = _parse_utc_ts(ts)
        if t is None:
            logger.warning("伏羲数据无法解析的时刻 %r 已跳过（该列数值一并剔除）", ts)
            continue
        kept.append(j)
        # timestamp 为 UTC 时刻 → 北京时墙钟（+8h，系统全程北京时配对）
        times.append((t + timedelta(hours=8)).strftime("%Y-%m-%dT%H:00"))

    def col(idx: int) -> list[float | None]:
        row = conv(idx)
        return [row[j] if j < len(row) else None for j in kept]

    loc = data.get("location")
    grid_lat = grid_lon = None
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        try:
            grid_lat, grid_lon = float(loc[0]), float(loc[1])
        except (TypeError, ValueError):
            pass
    return {
        "issue_echo": data.get("time_fcst") if isinstance(data.get("time_fcst"), str) else None,
        "grid_lat": grid_lat,
        "grid_lon": grid_lon,
        "time": times,
        "t2m": col(i_t2m),
        "tp": col(i_tp),
        "t2m_unit": t2m_unit,
    }


def _parse_utc_ts(ts: Any) -> datetime | None:
    """解析 UTC 时刻字符串 → naive UTC。

    容忍 "...Z"/"...z"/毫秒 ".000Z"/带 "+00:00" 偏移/无时区标记；
    带非零偏移的字符串按其偏移换算到 UTC。解析失败返回 None（调用方剔除该列）。
    """
    if not isinstance(ts, str) or not ts:
        return None
    s = ts.strip()
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00").replace("z", "+00:00"))
        if dt.tzinfo is not None:
            return dt.astimezone(timezone.utc).replace(tzinfo=None)
        return dt  # 无时区标记：契约即 UTC
    except ValueError:
        return None


def looks_like_running_accumulation(values: list[float | None]) -> bool:
    """降水序列是否呈"起报累计"形态：非缺测样本近乎单调不减且整体显著增长。

    用于发现口径漂移（tp 若为自起报累计，逐小时配对会系统性失真）。
    """
    vals = [v for v in values if v is not None]
    if len(vals) < 12:
        return False
    nondec = sum(1 for a, b in zip(vals, vals[1:]) if b >= a - 1e-9)
    return nondec >= 0.95 * (len(vals) - 1) and vals[-1] > vals[0] + 5.0


class FuxiDetProvider(ForecastProvider):
    """伏羲确定性（FuXi-Det）快照器：需 FUXI_DATA_TOKEN，单模型快照 dict。"""

    def __init__(self, timeout: int | tuple = (10, 60), retries: int = 3,
                 session: requests.Session | None = None, token: str | None = None,
                 now: datetime | None = None):
        self.token = token if token is not None else os.environ.get(TOKEN_ENV, "")
        if not self.token:
            raise RuntimeError(
                f"缺少伏羲数据服务查询 Token：请登录 fuxi-ai.cn/fuxi-data 获取，"
                f"并经环境变量 {TOKEN_ENV} 注入"
            )
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self._now = now
        # isAvail 结果按日期缓存（产品级属性，跨站点复用）；None 表示该日无数据
        self._avail_cache: dict[str, list[str] | None] = {}

    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> dict:
        init_time = self._resolve_init_time()      # "YYYY-MM-DD HH:00:00"（UTC）
        issue_utc = datetime.strptime(init_time, "%Y-%m-%d %H:%M:%S")
        issue_iso = (issue_utc + timedelta(hours=8)).strftime("%Y-%m-%dT%H:00")

        resp = self._request(
            QUERY_URL, method="POST",
            json_body={
                "lon": station.lon, "lat": station.lat,
                "vars": VARS, "initTime": init_time, "model": MODEL_CODE,
            },
            auth=True,
        )
        parsed = parse_query_response(resp)
        if parsed["issue_echo"] and parsed["issue_echo"] != init_time:
            logger.warning(
                "伏羲数据响应起报(%s)与请求(%s)不一致，lead 口径可能错位",
                parsed["issue_echo"], init_time,
            )
        if not parsed["time"]:
            raise RuntimeError("伏羲数据响应 timestamp 为空（该起报可能未发布）")
        n = len(parsed["time"])
        t2m = (parsed["t2m"] + [None] * n)[:n]
        tp = (parsed["tp"] + [None] * n)[:n]
        if all(v is None for v in t2m):
            logger.warning("伏羲确定性站点 %s 温度序列全部缺测，服务端契约可能已变化", station.id)
        if looks_like_running_accumulation(tp):
            logger.warning(
                "伏羲确定性站点 %s 降水序列呈单调不减形态——tp 可能是自起报累计而非"
                "逐小时累计，逐小时配对口径存疑，请人工复核（本快照仍按逐时刻值入库）",
                station.id,
            )
        if parsed["t2m_unit"]:
            logger.info("伏羲数据 t2m 单位: %r", parsed["t2m_unit"])
        snapshot = {
            "issue_iso": issue_iso,
            "station_id": station.id,
            "source": SOURCE,
            "models": [MODEL_NAME],
            "grid_lat": parsed["grid_lat"] if parsed["grid_lat"] is not None else float(station.lat),
            "grid_lon": parsed["grid_lon"] if parsed["grid_lon"] is not None else float(station.lon),
            "elevation": None,
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            "hourly_time": parsed["time"],
            "data": {MODEL_NAME: {"temperature_2m": t2m, "precipitation": tp}},
        }
        logger.info("站点 %s 已抓取伏羲确定性起报 %s（UTC %s），时间点数 %d",
                    station.id, issue_iso, init_time, n)
        return snapshot

    # ------------------------------------------------------------------ 内部
    def _resolve_init_time(self) -> str:
        """探测最新可用起报：UTC 今天起向过去逐日查 isAvail，取该日最大可用小时。"""
        now = self._now or utc_now()
        for date in candidate_dates(now):
            if date in self._avail_cache:
                hours = self._avail_cache[date]
            else:
                payload = self._request(
                    AVAIL_URL, method="POST",
                    json_body={"modelId": MODEL_ID, "initTime": date},
                )
                hours = parse_avail_hours(payload) or None
                self._avail_cache[date] = hours
            if hours:
                # 轮次取整数值最大（不依赖零填充字典序）
                return f"{date} {max(hours, key=int)}:00:00"
        raise RuntimeError(
            f"伏羲确定性最近 {MAX_DATE_FALLBACK} 天（UTC）均无可用起报轮次"
        )

    def _request(self, url: str, *, method: str = "GET", json_body: dict | None = None,
                 auth: bool = False) -> Any:
        headers = dict(HEADERS)
        if auth:
            headers["Authorization"] = self.token  # 官方示例：原始 token，无 Bearer 前缀
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.request(
                    method, url, json=json_body, headers=headers, timeout=self.timeout,
                )
                status = getattr(resp, "status_code", None)
                if status == 200:
                    return resp.json()
                try:
                    body_digest = (resp.text or "")[:200]
                except Exception:  # noqa: BLE001
                    body_digest = ""
                if status == 401:
                    raise RuntimeError(
                        f"伏羲数据服务鉴权失败（HTTP 401 {body_digest!r}）——"
                        f"请确认 {TOKEN_ENV} 为 fuxi-data 页面获取的最新查询 Token"
                    )
                if isinstance(status, int) and 400 <= status < 500 and status != 429:
                    raise _Rejected(f"HTTP {status} body={body_digest!r}")
                last_err = RuntimeError(f"HTTP {status} body={body_digest!r}")
            except RuntimeError as e:
                if "鉴权失败" in str(e):
                    raise
                last_err = e
            except _Rejected as e:
                # 确定性失败（4xx）重试无意义，直接上抛
                raise RuntimeError(f"伏羲数据请求被拒: {e}") from e
            except Exception as e:  # noqa: BLE001  网络类异常/5xx → 可重试
                last_err = e
            logger.warning("伏羲数据请求失败（第%d次）: %s", attempt + 1, last_err)
            if attempt < self.retries:
                time.sleep(min(30, 3 * 2 ** attempt))
        raise RuntimeError(f"伏羲数据请求最终失败: {last_err}") from last_err


class _Rejected(Exception):
    """确定性失败（4xx，重试无意义）。"""

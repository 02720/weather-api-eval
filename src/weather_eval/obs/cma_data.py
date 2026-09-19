"""中国气象数据网（data.cma.cn）实况观测抓取器 —— eia-data 的**备用源**。

为什么需要它
------------
全项目的观测此前只有 eia-data.com 一个第三方页面（README §10 把它列为已知缺口：
"观测源仍是单点"）。观测是本项目**唯一的真值来源**，单点依赖是整个评估里最脆弱的
一环——页面一改版，所有结论立刻停摆，而且没有任何第二个出口可以顶上。

接口契约（2026-09 逆向 data.cma.cn/dataGis/static/gridgis 前端 bundle 得到，逐条实测）
-----------------------------------------------------------------------------
站点页 https://data.cma.cn/dataGis/static/gridgis/#/pcindex 的实况数据由

    /app/Rest/liveDataService/station/{站号}/latest?datetime=<YYYYMMDDHHMMSS>

提供（`type=1`）。它不是公开 API——无官方契约、无 SLA、随时可能变——但**游客态
可用、无需凭据**，且寻址用的是 WMO 站号：正好是 config/stations.yaml 里已经为
`cma_public` 配好的 `cma_id`，无需新增任何配置字段。

**时间语义：`datetime` 参数按 UTC 解释，返回的 `D_datetime` 是北京时。**
这是本模块最要紧的一条。若把请求时刻当成北京时直接传进去，会静默拿到 **+8 小时**
的观测——不报错、不告警，只是每一个小时都对错了对象（本项目对 EW4ALL 的 UTC 接口
已经吃过一次同型的亏，见 forecast/ew4all.py）。证据（站号 59265，2026-09-19 实测）：

    ┌──────────────┬──────────────────────┐   ┌──────────────┬──────────────────────┐
    │ 请求         │ 返回 D_datetime      │   │ 请求         │ 返回 D_datetime      │
    ├──────────────┼──────────────────────┤   ├──────────────┼──────────────────────┤
    │ 00:00:00     │ 2026-09-19 08:00:00  │   │ 13:00:00     │ 2026-09-19 21:00:00  │
    │ 01:00:00     │ 2026-09-19 09:00:00  │   │ 14:00:00     │ 2026-09-19 22:00:00  │
    │ 06:00:00     │ 2026-09-19 14:00:00  │   │ 15:00:00     │ 22:00:00（封顶）     │
    │ 12:00:00     │ 2026-09-19 20:00:00  │   │ 23:00:00     │ 22:00:00（封顶）     │
    └──────────────┴──────────────────────┘

前 15 行严格满足 `回显 = 请求 + 8h`；其后封顶于"当前最新可得时刻"（数据还没发布）。
温度序列同样是物理自洽的日变化（08:00 26.4℃ → 15:00 33.9℃ → 22:00 28.9℃），
排除了"其实返回的是别的时刻"的可能。

本模块据此采取三条硬约束：
1. 请求参数 = 目标北京时 − 8h；
2. **时间键一律取响应自述的 `D_datetime`**，绝不把请求时刻当作观测时刻；
3. 回显与请求的关系被**分类计数**并告警：回显晚于请求 = 接口语义已漂移（严重，
   必须有人知道）；回显早于请求 = 该时刻尚未发布、封顶到最新（正常）。
   这把"接口哪天变了"从一次静默错标变成一声喊。

要素码（与 eia-data 存档逐点对拍确认）
--------------------------------------
    V12001  气温（℃）         V13003  相对湿度（%）
    V10004  气压（hPa）       V11293  风速（m/s）
    V13019  前 1 小时降水（mm）
    V20001  能见度（m）       V20003T 天气现象文本（未入库）

对拍证据（4 站 × 12 个连续整点 + 8 个降水事件，逐点与 `data/obs/` 存档比对）：
气温 |ΔT| = 0.00 ℃；降水 |ΔR| = 0.00 mm——含万宁 2026-09-19 13:00 的 8.8 mm
与 14:00 的 6.6 mm 这类强降水，四个站号（59265/59449/59255/59951）全部可用。

> ⚠️ 需要说清楚的是：这**不是**一个独立于 eia-data 的观测源，而是**同一批 CMA 观测
> 的另一个出口**（两边逐点完全相等即是证据）。所以它的价值不在"交叉验证"，而在
> "eia-data 改版时观测不断供"。真要交叉验证，需要另一个**非 CMA** 的观测来源。

失效形态：`HTTP 200 + content: {}`
----------------------------------
站号写错、站号为空、时刻超出可得范围、`datetime` 非法——全部返回

    {"message":"成功！","status":0,"code":200,"type":1,"content":{}}

**空 content 是"该时刻没有观测"，不是错误**（与 forecast/cma_public.py 的
"HTTP 200 + 空 data"、EW4ALL 的"空 data"同一形态）。因此遵守同一条不变量：
**空数据绝不建记录**——单点为空 → 该小时缺测（绝不折算成 0.0）；整窗全空 → 抛错，
视为抓取失败，交给上层降级或标红。

反爬：**UA 必须在白名单内**
---------------------------
实测（2026-09-19）：不带 UA → HTTP 403；`python-requests/2.31.0` → 服务端直接断开
连接（无响应）；浏览器 UA 通过。403 属确定性失败，`request_with_retries` 会立即
熔断而不烧退避。UA 与 Referer 集中在本模块的 `HEADERS` 单点定义——若日后仓库
统一 UA 被误伤，只改这一处。
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta
from typing import Any, Iterable

import requests

from .base import ObsSource
from ..forecast.http import DEFAULT_TIMEOUT, TimeBudget, request_with_retries
from ..timeutil import iso, now_beijing, parse_obs_time

logger = logging.getLogger(__name__)

# 站点实况接口（type=1 = 地面实况）
BASE_URL = "https://data.cma.cn/app/Rest/liveDataService/station/{station}/latest"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://data.cma.cn/dataGis/static/gridgis/",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# 要素码 → 入库字段。顺序即"这是一条要素型记录"的声明。
ELEMENT_MAP: dict[str, str] = {
    "V12001": "temp",
    "V13003": "humidity",
    "V10004": "pressure",
    "V11293": "wind_speed",
    "V13019": "rain",
    "V20001": "visibility",
}

# 入库记录的来源标记（可追溯：这一小时的实况是哪条链路抓回来的）
SOURCE_TAG = "cma_data"

# `datetime` 参数按 UTC 解释（返回的 D_datetime 是北京时）：请求 = 北京时 − 该偏移
PARAM_UTC_OFFSET_HOURS = 8

# 默认回看窗口：24h 满窗 + 2h 冗余（覆盖观测到报延迟与上轮运行间隔）
DEFAULT_LOOKBACK_HOURS = 26
# 复核窗口：最近若干小时内即便本地已有记录也重新抓一次，吸收第三方对近时观测的修正
DEFAULT_REVISION_HOURS = 6
# 请求间隔（秒）：单轮 4 站 × ~26 次请求，节流以免被当作爬虫
DEFAULT_SLEEP_SECONDS = 0.1

# 标称窗口内至少要拿到多少小时，才算"这个源这轮可用"（低于此值判为窗口被截断）。
# 26h 窗口正常应有 ~25 条；门槛取 6 是"明显截断"的松边界——真正的可用性判定
# 还有新鲜度与异常两条，见 obs/chain.py。
DEFAULT_MIN_HOURS = 6


def _to_float(v: Any) -> float | None:
    """把要素值转为 float；缺测/非数值一律 None（**绝不折算成 0.0**）。"""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in ("none", "nan", "na", "null", "-", "—"):
            return None
        try:
            f = float(s)
        except ValueError:
            return None
    else:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
    return None if f != f else f  # 去掉 NaN


def _target_hours(lookback_hours: int, now: datetime) -> list[datetime]:
    """目标整点（北京时，升序）：从 now 下取整往前推 lookback_hours 小时。"""
    end = now.replace(minute=0, second=0, microsecond=0)
    out: list[datetime] = []
    cur = end - timedelta(hours=lookback_hours)
    while cur <= end:
        out.append(cur)
        cur += timedelta(hours=1)
    return out


def _parse_echo(s: Any) -> datetime | None:
    """解析响应自述的观测时刻（`D_datetime`，如 '2026-09-19 22:00:00'，北京时）。"""
    if not isinstance(s, str) or not s.strip():
        return None
    try:
        return parse_obs_time(s)
    except ValueError:
        return None


def _record_from_content(content: dict, obs_dt: datetime) -> dict:
    """把一份 content 转成入库记录；时间键用响应自述的时刻（北京时整点）。"""
    rec: dict[str, Any] = {
        "time": iso(obs_dt.replace(minute=0, second=0, microsecond=0)),
        "source": SOURCE_TAG,
    }
    for code, field in ELEMENT_MAP.items():
        rec[field] = _to_float(content.get(code))
    return rec


class CmaDataObsSource(ObsSource):
    """按 WMO 站号逐整点抓取中国气象数据网实况（近 lookback_hours 窗口）。

    与 EiaDataObsSource 的接口差异：eia-data 一次请求返回整页 24h；本接口一次只
    返回**一个**时刻，故按窗口内每个整点各请求一次。为把请求量压回可接受范围：
      · `skip_hours`：调用方已持有的时刻（通常是上一轮刚写入的）不再重复请求；
      · 复核窗口内的少数时刻例外（吸收第三方对近时观测的修正）。
    实测 12 次连发约 5.3s，无限流迹象；单源总耗时仍受 `TimeBudget` 约束。
    """

    def __init__(
        self,
        *,
        lookback_hours: int = DEFAULT_LOOKBACK_HOURS,
        revision_hours: int = DEFAULT_REVISION_HOURS,
        min_hours: int = DEFAULT_MIN_HOURS,
        timeout: Any = DEFAULT_TIMEOUT,
        retries: int = 2,
        sleep_seconds: float = DEFAULT_SLEEP_SECONDS,
        session: requests.Session | None = None,
        skip_hours: Iterable[str] | None = None,
        budget: TimeBudget | None = None,
    ):
        self.lookback_hours = lookback_hours
        self.revision_hours = revision_hours
        self.min_hours = min_hours
        self.timeout = timeout
        self.retries = retries
        self.sleep_seconds = sleep_seconds
        self.session = session or requests.Session()
        # 调用方已知的时刻（ISO 北京时）：不必重复请求——除非落在复核窗口内
        self.skip_hours: set[str] = set(skip_hours or ())
        self.budget = budget if budget is not None else TimeBudget()

    # ---------------------------------------------------------------- 内部
    def _fetch_hour(self, station_id: str, station_no: str, target: datetime) -> dict | None:
        """抓单个整点；返回 content（可能为空 dict），请求失败则抛异常。"""
        param = (target - timedelta(hours=PARAM_UTC_OFFSET_HOURS)).strftime("%Y%m%d%H%M%S")
        url = BASE_URL.format(station=station_no)
        resp = request_with_retries(
            self.session, url,
            params={"datetime": param},
            headers=HEADERS, timeout=self.timeout, retries=self.retries,
            source=f"CMA 实况站点 {station_id}",
            budget=self.budget,
        )
        try:
            payload = resp.json()
        except (json.JSONDecodeError, ValueError, AttributeError) as e:
            raise RuntimeError(
                f"站点 {station_id} 实况响应不是合法 JSON（可能被反爬页/维护页替换）: {e}"
            ) from e
        if payload.get("code") not in (200, "200"):
            raise RuntimeError(
                f"站点 {station_id} 实况接口返回异常: {payload.get('message') or payload!r}"
            )
        content = payload.get("content")
        return content if isinstance(content, dict) else {}

    def fetch(self, station: Any) -> list[dict]:
        station_no = getattr(station, "cma_id", None)
        if not station_no:
            # 与 cma_public 同一条纪律：绝不猜站号。缺配置 = 响亮失败。
            raise ValueError(
                f"站点 {station.id} 未配置 cma_id（WMO 站号），无法使用中国气象数据网实况源"
            )

        now = now_beijing()
        hours = _target_hours(self.lookback_hours, now)
        revision_cutoff = now - timedelta(hours=self.revision_hours)

        records: dict[str, dict] = {}   # time_iso -> record（按响应自述时刻去重）
        n_empty = 0
        n_skipped = 0
        n_late_echo = 0     # 回显晚于请求：接口语义漂移（严重）
        n_early_echo = 0    # 回显早于请求：该时刻未发布、封顶到最新（正常）

        for target in hours:
            t_iso = iso(target)
            if t_iso in self.skip_hours and target < revision_cutoff:
                n_skipped += 1
                continue
            if self.budget.expired():
                logger.error("站点 %s CMA 实况抓取时间预算耗尽（%s），提前收尾",
                             station.id, self.budget.describe())
                break
            content = self._fetch_hour(station.id, station_no, target)
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
            if not content:
                n_empty += 1
                continue
            obs_dt = _parse_echo(content.get("D_datetime"))
            if obs_dt is None:
                # 有 content 却没有可解析的观测时刻：无法确定这条实况属于哪一小时，
                # 宁可丢掉也不能凭请求时刻猜——错标的观测会污染全部评估。
                logger.warning("站点 %s 实况缺少可解析的 D_datetime（%r），丢弃该条",
                               station.id, content.get("D_datetime"))
                continue
            if obs_dt > target:
                n_late_echo += 1
                if n_late_echo <= 3:
                    logger.warning(
                        "站点 %s 实况回显时刻 %s 晚于请求时刻 %s，超出已知的 UTC+8 语义——"
                        "接口时间约定可能已变化，入库仍以回显时刻为准",
                        station.id, iso(obs_dt), t_iso)
            elif obs_dt < target:
                n_early_echo += 1
            records[iso(obs_dt.replace(minute=0, second=0, microsecond=0))] = \
                _record_from_content(content, obs_dt)

        if n_late_echo:
            logger.error(
                "站点 %s 有 %d/%d 个回显时刻晚于请求时刻：CMA 实况接口的 UTC 偏移语义"
                "可能已变（当前按 %dh 处理），请重新标定 obs/cma_data.py",
                station.id, n_late_echo, len(hours), PARAM_UTC_OFFSET_HOURS)

        out = [records[k] for k in sorted(records, reverse=True)]  # 最新在前（与 eia-data 一致）

        if not out:
            # 整窗全空：视为抓取失败（站号错配 / 接口改版 / 反爬），让上层降级或标红。
            # 绝不返回空列表冒充"成功"——空数据不建记录，也不伪装成"本轮没有新观测"。
            raise RuntimeError(
                f"站点 {station.id}（站号 {station_no}）CMA 实况整窗无数据"
                f"（{len(hours)} 个整点全空；可能站号未在该平台收录、接口已改版或被限流）"
            )

        if len(out) < self.min_hours:
            logger.warning(
                "站点 %s CMA 实况仅取到 %d 小时（窗口 %d 小时，空气 %d 个）——"
                "窗口可能被截断，将由编排层判定是否降级",
                station.id, len(out), len(hours), n_empty)

        logger.info(
            "站点 %s CMA 实况取到 %d 条（窗口 %d 小时，空气 %d，跳过已知 %d，"
            "回显封顶 %d）", station.id, len(out), len(hours), n_empty, n_skipped,
            n_early_echo)
        return out

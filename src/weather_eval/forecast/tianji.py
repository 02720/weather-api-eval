"""中科天机（TianJi，tjweather.com）单点预报快照器。

数据源：https://www.tjweather.com/vis/ 可视化页面背后的单点查询接口（网页抓取）：
  GET https://www.tjweather.com/meteorological/spas/single-point/query
      ?lat={纬度}&lon={经度}&mode={模式}&baseTime={YYYYMMDDHH}
      &production={产品码}&region=global&factorCode={要素码}

─────────────────────────────────────────────────────────────────────
关键契约（均经线上实测验证，改动前必须复核）
─────────────────────────────────────────────────────────────────────
1. 鉴权：游客态无需任何 Token/请求头（登录态才注入 Authorization；游客接口不校验）。
2. 时间语义（最容易错的地方）：
   - baseTime 参数与响应 baseTimeString 均为**北京时** YYYYMMDDHH。服务端会把
     baseTime=2026082708 回显为 UTC ISO ``2026-08-27T00:00:00+00:00``（08 BJ = 00 UTC）。
   - forecastTimeString 同为北京时 YYYYMMDDHH（与 forecastTime 的 UTC ISO 恒差 8h），
     逐小时步进、从起报后 1 小时开始，起报当刻本身不出现在序列里。
   - 本评估体系全程使用北京时 naive 整点字符串配对，因此这里直接按北京时墙钟解析，
     无需任何时区换算（见 timeutil 模块说明）。
   - 起报以响应回显的 baseTimeString 为准：若服务端以"就近替换"响应了非所请求的
     轮次，数据属于回显轮次，用它计算 issue/lead 才不错位（仅告警不拒绝，数据保留）。
3. 起报轮次：北京时每天 08/20 时两轮（国内 NWP 业务惯例）。最新轮次有发布延迟，
   且**各模式的发布进度互不同步**——用未来轮次查询会返回 200 但 forecastDetails
   为空。因此逐模式向过去回退探测（最多 MAX_BASE_FALLBACK 轮）直到拿到非空数据，
   并把解析结果按模式缓存在实例上（起报可用性是产品级属性，跨站点复用）。
4. 模型与产品映射（用户口径 → 官方产品）：
   - 公里级融合 → mode=nextgen：温度 production=c1km/factorCode=tmp2m，
     降水 production=c2_5km/factorCode=pratesfc（融合产品按要素用不同网格码）；
   - T2-Early（天机2/DA，快速同化）→ mode=early&production=t2；
   - T2（天机2/ND，未同化）     → mode=late&production=t2；
   - T1（天机1/ND）             → mode=t1_ai&production=t1（该产品本身即 AI 驱动，
     站点上不存在非 AI 的 t1 轮次）；
   - T1-AI（T1H-AI 高分辨率版） → mode=early&production=t1h。
5. 数值口径：温度 ℃；pratesfc 为逐小时降水率 mm/h，作为"前 1 小时累计"的近似与
   观测 rain@t 配对（与 Open-Meteo precipitation@t 同一假设，README 已注明）。
   value 是数组：标量要素取 [0]；空数组/缺测/非法值一律归 None，绝不伪装成 0。
6. 快照粒度：各模式最新可用起报可能不同步，而快照结构只有一个全局 issue_iso——
   若多模型共享一份快照会导致时效（lead）分组错位。因此本提供方**按模型返回独立
   快照列表**（每份各自 issue_iso=该模型起报时刻、各自时间轴），由 CLI 逐份存档；
   单个模型失败只跳过该模型（告警可见），不拖垮同站其余模型。
7. 探测缓存假设：起报可用性是**产品级**属性（跨站点一致），故按模式缓存并跨站复用。
   若某站在缓存轮次上意外拿到空序列（产品回溯清理等罕见情形），会作废缓存并重探一次。
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any

import requests

from .base import ForecastProvider
from ..timeutil import now_beijing

logger = logging.getLogger(__name__)

ENDPOINT = "https://www.tjweather.com/meteorological/spas/single-point/query"
HEADERS = {"User-Agent": "weather-api-eval/0.1 (+https://github.com/)"}
REGION = "global"

# 起报轮次（北京时）与回退探测参数
CYCLE_HOURS = (8, 20)
MAX_BASE_FALLBACK = 4

SOURCE = "tianji"

# 模型名 → (mode, 温度 production, 温度 factorCode, 降水 production, 降水 factorCode, 预期点数)
MODEL_SPECS: dict[str, tuple[str, str, str, str, str, int]] = {
    # 公里级融合：按要素区分产品网格码（温度 1km / 降水 2.5km）
    "tj_km_fusion": ("nextgen", "c1km", "tmp2m", "c2_5km", "pratesfc", 240),
    "tj_t2_early": ("early", "t2", "t2mz", "t2", "pratesfc", 240),
    "tj_t2": ("late", "t2", "t2mz", "t2", "pratesfc", 240),
    "tj_t1": ("t1_ai", "t1", "t2mz", "t1", "pratesfc", 360),
    "tj_t1h_ai": ("early", "t1h", "t2mz", "t1h", "pratesfc", 360),
}


def candidate_base_times(now: datetime, count: int = MAX_BASE_FALLBACK) -> list[str]:
    """从北京时 now 向过去生成最近的起报轮次候选（YYYYMMDDHH，降序）。

    轮次固定在北京时 08/20 时。now=08-27 06:00 → 首个候选为 08-26 20 时。
    """
    cycles: list[str] = []
    day = now.date()
    for _ in range(count // 2 + 2):
        for h in CYCLE_HOURS:
            cycles.append(day.strftime("%Y%m%d") + f"{h:02d}")
        day -= timedelta(days=1)
    cycles.sort(reverse=True)
    now_str = now.strftime("%Y%m%d%H")
    return [c for c in cycles if c <= now_str][:count]


def _value_of(entry: Any) -> float | None:
    """把 forecastDetails 条目的 value 数组归一为 float（缺测 → None）。"""
    if not isinstance(entry, dict):
        return None
    val = entry.get("value")
    if not isinstance(val, (list, tuple)) or not val:
        return None
    try:
        f = float(val[0])
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f


def _parse_tj_response(payload: Any, factor_code: str) -> dict[str, list[float | None]]:
    """从单点查询响应中提取某要素的时间序列。

    返回 {"time": ["YYYY-MM-DDTHH:00", ...], "value": [...]}；
    响应 code!=200 抛错；该要素缺失或 forecastDetails 为空返回空序列（调用方区分
    "起报轮次无数据"与"请求失败"）。
    """
    if not isinstance(payload, dict) or payload.get("code") != 200:
        msg = payload.get("message") if isinstance(payload, dict) else payload
        raise RuntimeError(f"中科天机响应异常: code={payload.get('code') if isinstance(payload, dict) else None} message={msg!r}")
    data = payload.get("data") or {}
    for item in data.get("forecast") or []:
        if item.get("factorCode") != factor_code:
            continue
        details = item.get("forecastDetails") or []
        pairs: list[tuple[str, float | None]] = []
        for d in details:
            tstr = d.get("forecastTimeString")
            if not isinstance(tstr, str) or len(tstr) != 10 or not tstr.isdigit():
                continue
            # 北京时 YYYYMMDDHH → "YYYY-MM-DDTHH:00"
            t = f"{tstr[0:4]}-{tstr[4:6]}-{tstr[6:8]}T{tstr[8:10]}:00"
            pairs.append((t, _value_of(d)))
        pairs.sort(key=lambda x: x[0])
        # 去重（保留首见），保证时间轴严格递增且与数值按下标对齐
        seen: dict[str, float | None] = {}
        for t, v in pairs:
            seen.setdefault(t, v)
        times = list(seen.keys())
        return {"time": times, "value": [seen[t] for t in times]}
    return {"time": [], "value": []}


class _Deterministic(Exception):
    """确定性失败（重试无意义）：4xx 或带业务错误码的响应体。"""


class TianjiProvider(ForecastProvider):
    """中科天机预报快照器：按模型返回独立快照列表（见模块 docstring 第 6 点）。"""

    def __init__(
        self,
        timeout: int | tuple = (10, 30),
        retries: int = 3,
        session: requests.Session | None = None,
        now: datetime | None = None,
    ):
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self._now = now  # 测试注入；None 则运行时取当前北京时
        self._base_cache: dict[str, str] = {}      # model -> baseTime(YYYYMMDDHH)
        self._failed_models: set[str] = set()      # 本次运行内已失败的模型（产品级故障，跳过以免每站重复探测）

    # ------------------------------------------------------------------ 对外
    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> list[dict]:
        wanted = [m for m in (models or list(MODEL_SPECS)) if m in MODEL_SPECS]
        if not wanted:
            raise RuntimeError(f"中科天机无可识别的模型: {models}")
        snapshots: list[dict] = []
        errors: dict[str, Exception] = {}
        for model in wanted:
            # 单模型失败只跳过该模型：任一产品下线/延迟不应造成整站快照全部丢弃
            # （存档按 站×模型×起报 幂等，后续运行对失败模型自动重试）。
            if model in self._failed_models:
                continue
            try:
                snapshots.append(self._fetch_one(station, model))
            except Exception as e:  # noqa: BLE001
                self._failed_models.add(model)
                errors[model] = e
                logger.error("中科天机站点 %s 模型 %s 抓取失败: %s", station.id, model, e)
        if not snapshots:
            raise RuntimeError(f"中科天机站点 {station.id} 全部模型抓取失败: {errors!r}")
        return snapshots

    def _fetch_one(self, station: Any, model: str) -> dict:
        mode, t_prod, t_factor, p_prod, p_factor, expected = MODEL_SPECS[model]
        temp: dict[str, list] | None = None
        base = self._base_cache.get(model)
        if base is not None:
            temp, echoed = self._fetch_series(station, mode, t_prod, t_factor, base)
            if echoed:
                # 数据属于回显轮次：以回显为准修正起报并刷新缓存，防 lead 整体错位
                base = echoed
                self._base_cache[model] = base
            if not temp["time"]:
                # 缓存轮次意外无数据（罕见：产品回溯清理），作废缓存重探一次
                logger.warning(
                    "中科天机模型 %s 缓存起报 %s 无数据，作废缓存重新探测", model, base)
                self._base_cache.pop(model, None)
                base = None
                temp = None
        if temp is None:
            base, temp = self._resolve_base_time(model, mode, t_prod, t_factor, station)
            self._base_cache[model] = base
        # 探测刚返回的 temp 序列与正式请求同参同址，直接复用（省一次请求）。

        prec, _prec_echo = self._fetch_series(station, mode, p_prod, p_factor, base)
        if not temp["time"]:
            raise RuntimeError(
                f"中科天机站点 {station.id} 模型 {model} 起报 {base} 未返回温度序列"
            )
        if temp["time"] and all(v is None for v in temp["value"]):
            logger.warning(
                "中科天机站点 %s 模型 %s 起报 %s 温度序列全部缺测，服务端契约可能已变化",
                station.id, model, base,
            )
        if not prec["time"]:
            # 温度/降水是不同产品线（公里级融合尤其如此），发布可能不同步：
            # 只告警不跳过——温度照常入库，该轮降水为缺测（评估显示"样本不足"）。
            logger.warning(
                "中科天机站点 %s 模型 %s 起报 %s 降水产品(%s/%s)无数据，"
                "本快照降水将全为缺测（该轮降水可能尚未发布或瞬时故障）",
                station.id, model, base, p_prod, p_factor,
            )
        if len(temp["time"]) < expected:
            logger.warning(
                "中科天机站点 %s 模型 %s 仅返回 %d 个逐小时点（预期约 %d），"
                "长时效可能被服务端截断",
                station.id, model, len(temp["time"]), expected,
            )
        # 以温度序列为时间轴；降水按整点字符串对齐，缺失补 None
        p_by_t = dict(zip(prec["time"], prec["value"]))
        temps = temp["value"]
        precips = [p_by_t.get(t) for t in temp["time"]]

        issue_iso = f"{base[0:4]}-{base[4:6]}-{base[6:8]}T{base[8:10]}:00"
        snapshot = {
            "issue_iso": issue_iso,
            "station_id": station.id,
            "source": SOURCE,
            "models": [model],
            # 该 API 不做格点吸附，响应回显请求坐标；无 elevation 字段
            "grid_lat": float(station.lat),
            "grid_lon": float(station.lon),
            "elevation": None,
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            "hourly_time": temp["time"],
            "data": {model: {"temperature_2m": temps, "precipitation": precips}},
        }
        logger.info(
            "站点 %s 已抓取中科天机模型 %s 起报 %s，时间点数 %d",
            station.id, model, issue_iso, len(temp["time"]),
        )
        return snapshot

    # ------------------------------------------------------------------ 内部
    def _resolve_base_time(self, model: str, mode: str, prod: str,
                           factor: str, station: Any) -> tuple[str, dict[str, list]]:
        """探测该模型最新可用起报：从最近轮次向过去回退，取首个非空数据轮。

        返回 (生效起报, 命中轮次的序列)——生效起报以响应回显的 baseTimeString 为准
        （服务端"静默就近替换"时，数据属于回显轮次，用它计算 lead 才不错位）；
        序列与正式温度请求同参同址，直接复用省一次请求。注意：非 200 / code!=200
        属于请求级失败，直接上抛；只有"200 但该轮尚未发布（空序列）"才回退到更早轮次。
        """
        now = self._now or now_beijing()
        first = candidate_base_times(now, 1)[0]
        for cand in candidate_base_times(now):
            series, echoed = self._fetch_series(station, mode, prod, factor, cand)
            if series["time"]:
                effective = echoed or cand
                if effective != first:
                    logger.info("中科天机模型 %s 最新轮次未发布，回退使用起报 %s", model, effective)
                return effective, series
        raise RuntimeError(
            f"中科天机模型 {model} 在最近 {MAX_BASE_FALLBACK} 个起报轮次均无数据"
        )

    def _fetch_series(self, station: Any, mode: str, prod: str,
                      factor: str, base: str) -> tuple[dict[str, list], str | None]:
        """请求某要素在某起报轮次的序列，返回 (序列, 回显的轮次 YYYYMMDDHH | None)。"""
        params = {
            "lat": station.lat,
            "lon": station.lon,
            "mode": mode,
            "baseTime": base,
            "production": prod,
            "region": REGION,
            "factorCode": factor,
        }
        payload = self._request(params)
        series = _parse_tj_response(payload, factor)
        # 回读校验：确认服务端响应的就是所请求的轮次，防止"静默就近替换"导致 lead 错位。
        # 不一致时告警并返回回显值，由调用方以回显为准（数据属于它真实所属的轮次）；
        # 回显值必须通过 YYYYMMDDHH 格式校验才采信，否则忽略（防契约漂移产出垃圾 issue）。
        echoed = (payload or {}).get("data", {}).get("baseTimeString") if isinstance(payload, dict) else None
        if isinstance(echoed, str) and len(echoed.strip()) == 10 and echoed.strip().isdigit():
            echoed = echoed.strip()
        else:
            echoed = None
        if echoed is not None and echoed != base:
            logger.warning(
                "中科天机响应轮次(%s)与请求(%s)不一致，以回显轮次为准计算起报/时效",
                echoed, base,
            )
        return series, echoed

    def _request(self, params: dict) -> Any:
        last_err: Exception | None = None
        for attempt in range(self.retries + 1):
            resp = None
            try:
                resp = self.session.get(
                    ENDPOINT, params=params, headers=HEADERS, timeout=self.timeout
                )
                status = getattr(resp, "status_code", None)
                if status == 200:
                    return resp.json()
                try:
                    body_digest = (resp.text or "")[:200]
                except Exception:  # noqa: BLE001
                    body_digest = ""
                # 该服务把参数/产品类确定性错误也包在 HTTP 500 + 业务错误码里返回
                # （如 {"code":11001,"message":"缺失参数:..."}）——重试无意义，直接上抛
                business_code = None
                try:
                    bj = resp.json()
                    if isinstance(bj, dict):
                        business_code = bj.get("code")
                except Exception:  # noqa: BLE001
                    pass
                if (isinstance(status, int) and 400 <= status < 500 and status != 429) \
                        or (business_code is not None and business_code != 200):
                    raise _Deterministic(
                        f"HTTP {status} code={business_code} body={body_digest!r}"
                    )
                last_err = RuntimeError(f"HTTP {status} body={body_digest!r}")
            except _Deterministic as e:
                raise RuntimeError(f"中科天机请求被拒: {e}") from e
            except Exception as e:  # noqa: BLE001  网络类异常/5xx → 可重试
                last_err = e
            logger.warning("中科天机请求失败（第%d次）: %s", attempt + 1, last_err)
            if attempt < self.retries:
                time.sleep(min(30, 3 * 2 ** attempt))
        raise RuntimeError(f"中科天机请求最终失败: {last_err}") from last_err

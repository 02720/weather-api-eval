"""eia-data.com 观测抓取器。

每个气象站页面（http://eia-data.com/<站名>气象站基本信息/）服务端直出近 24h
逐小时实况，页面内嵌 `const wd = {"time":[...],"temp":[...],"rain":[...],...}`。
首选解析该内嵌 JSON（字段干净、无单位噪声），失败则回退解析 HTML 表格。

新鲜度检查（P2-1 教训）：解析成功后检查最新一条观测是否明显陈旧（>3h），陈旧
以 WARNING 暴露但不阻断入库——观测文件本身可能已正常写入，只是页面数据滞后，
这正是"抓取跑通但数据停摆"的静默失效场景。检查曾在 `raise` 之后的死代码里
（永远不会执行）且用错解析函数（parse_obs_time 解析不了 iso() 的 'T' 分隔格式，
即便执行也必然静默失败）——现移到两条成功返回路径共用的正常路径上。
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import requests
from bs4 import BeautifulSoup

from .base import ObsSource
from ..forecast.http import request_with_retries
from ..timeutil import iso, parse_iso, parse_obs_time, now_beijing

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9",
}

# 内嵌 JSON：const wd = {...};（兼容 var/let、末尾分号可有可无）
WD_RE = re.compile(r"(?:const|var|let)\s+wd\s*=\s*(\{.*?\});?", re.DOTALL)

# 数值字段（来自 wd）
NUMERIC_KEYS = {
    "temp": "temp",
    "pressure": "pressure",
    "humidity": "humidity",
    "rain": "rain",
    "wind_speed": "wind_speed",
    "wind_dir": "wind_dir",
}

# 最新观测滞后超过该小时数视为"数据可能停摆"，告警暴露
STALE_HOURS = 3.0


def _to_float(v: Any) -> float | None:
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() in ("none", "nan", "na", "null"):
            return None
        try:
            return float(s)
        except ValueError:
            return None
    try:
        f = float(v)
        return None if f != f else f  # 去掉 NaN
    except (TypeError, ValueError):
        return None


class _NonHourCounter:
    """单次 fetch 内聚合"非整点观测时刻"计数（P2-4）。

    此前是模块级全局计数器：进程全局、非线程安全，且多站串行抓取时跨站共享——
    站点 A 产生的计数会在站点 B 的告警里出现（归因错位）。改为每次 fetch 新建
    一个实例，计数随请求生命周期产生与消费，归因天然按站隔离。
    """

    def __init__(self) -> None:
        self.seen = 0

    def floor(self, dt) -> Any:
        """观测时刻下取整到整点，并统计带（秒/分）的非整点时刻。

        评估按整点字符串精确配对，带分钟的观测（如 15:10）永远配不上整点预报、
        会被静默丢样。eia-data 实测全为整点，此为防御：若页面开始返回带分钟
        时刻，下取整并聚合告警（取整对 1h 累计量的标注误差 ≤ 读取延迟，
        优于整条丢失）。
        """
        floored = dt.replace(minute=0, second=0, microsecond=0)
        if floored != dt:
            self.seen += 1
        return floored

    def warn(self, station_id: str) -> None:
        if self.seen:
            logger.warning(
                "站点 %s 有 %d 个带分钟/秒的观测时刻，已下取整到整点参与配对"
                "（页面时间格式可能已变化，请核对 eia-data 页面）", station_id, self.seen)


def _check_freshness(records: list[dict], station_id: str) -> None:
    """最新一条观测明显陈旧时告警（不阻断入库——入库的数据仍是可用的历史实况）。"""
    try:
        latest = max(parse_iso(r["time"]) for r in records if r.get("time"))
        age_h = (now_beijing() - latest).total_seconds() / 3600
        if age_h > STALE_HOURS:
            logger.warning(
                "站点 %s 最新观测已陈旧 %.1f 小时（预期近 1 小时内）——"
                "抓取链路正常但页面数据可能停摆，请核对 eia-data 页面",
                station_id, age_h)
    except (ValueError, KeyError, TypeError) as e:
        logger.warning("站点 %s 新鲜度检查失败（不影响入库）: %s", station_id, e)


def _records_from_wd(wd: dict, cnt: _NonHourCounter) -> list[dict]:
    times = wd.get("time") or []
    if not times:
        return []
    n = len(times)
    out: list[dict] = []
    for i in range(n):
        try:
            dt = cnt.floor(parse_obs_time(times[i]))
        except ValueError:
            continue
        rec = {"time": iso(dt), "source": "wd"}
        for src_key, out_key in NUMERIC_KEYS.items():
            arr = wd.get(src_key)
            rec[out_key] = _to_float(arr[i]) if (arr and i < len(arr)) else None
        out.append(rec)
    return out


def _records_from_table(html: str, cnt: _NonHourCounter) -> list[dict]:
    """回退：解析观测表（表头须含"气温"与"降水量"等观测字段，避免误抓预报表）。"""
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", class_="modern-table")
    target = None
    for table in tables:
        header = [th.get_text(strip=True) for th in table.find_all("tr")[0].find_all(["th", "td"])] if table.find("tr") else []
        text = " ".join(header)
        # 观测表签名：含 气温 + 降水量（预报表只有"温度/降水"且无气压/湿度）
        if "气温" in text and "降水量" in text:
            target = table
            break
    if target is None:
        return []
    rows = target.find_all("tr")
    if len(rows) < 2:
        return []
    header = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]
    col = {}
    for idx, h in enumerate(header):
        if "时间" in h:
            col["time"] = idx
        elif "气温" in h:
            col["temp"] = idx
        elif "降水量" in h or "降水" in h:
            col["rain"] = idx
        elif "气压" in h:
            col["pressure"] = idx
        elif "相对湿度" in h or "湿度" in h:
            col["humidity"] = idx
        elif "风速" in h:
            col["wind_speed"] = idx
        elif "风向" in h:
            col["wind_dir"] = idx
    out: list[dict] = []
    for tr in rows[1:]:
        tds = tr.find_all("td")
        if "time" not in col or len(tds) <= col["time"]:
            continue
        try:
            dt = cnt.floor(parse_obs_time(tds[col["time"]].get_text(strip=True)))
        except ValueError:
            continue
        rec = {"time": iso(dt), "source": "table"}
        for out_key, idx in col.items():
            if out_key == "time":
                continue
            rec[out_key] = _to_float(tds[idx].get_text(strip=True)) if idx < len(tds) else None
        out.append(rec)
    return out


class EiaDataObsSource(ObsSource):
    def __init__(self, timeout: int = 30, retries: int = 2, session: requests.Session | None = None):
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()

    def fetch(self, station: Any) -> list[dict]:
        url = station.obs_url
        if not url:
            raise ValueError(f"站点 {station.id} 未配置 obs_url")
        resp = request_with_retries(
            self.session, url, headers=HEADERS, timeout=self.timeout,
            retries=self.retries, source=f"观测站点 {station.id}",
        )
        html = resp.text  # requests 已按声明/探测编码解码

        cnt = _NonHourCounter()
        records = None
        m = WD_RE.search(html)
        if m:
            try:
                wd = json.loads(m.group(1))
                recs = _records_from_wd(wd, cnt)
                if recs:
                    logger.info("站点 %s 解析 wd JSON 得到 %d 条", station.id, len(recs))
                    records = recs
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("站点 %s 内嵌 wd JSON 解析失败: %s", station.id, e)

        if records is None:
            recs = _records_from_table(html, cnt)
            if recs:
                logger.info("站点 %s 回退解析表格得到 %d 条", station.id, len(recs))
                records = recs

        if not records:
            # 页面 200 但无任何观测：视为抓取失败（可能是反爬/登录页/改版），让上层标红
            raise RuntimeError(
                f"站点 {station.id} 页面未解析到任何观测记录（可能页面改版或返回异常页）"
            )
        # 新鲜度检查在成功路径上（两条解析路径汇合后、返回前）：页面数据停摆时告警
        _check_freshness(records, station.id)
        cnt.warn(station.id)
        return records

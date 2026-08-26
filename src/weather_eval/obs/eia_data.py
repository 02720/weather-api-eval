"""eia-data.com 观测抓取器。

每个气象站页面（http://eia-data.com/<站名>气象站基本信息/）服务端直出近 24h
逐小时实况，页面内嵌 `const wd = {"time":[...],"temp":[...],"rain":[...],...}`。
首选解析该内嵌 JSON（字段干净、无单位噪声），失败则回退解析 HTML 表格。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import requests
from bs4 import BeautifulSoup

from .base import ObsSource
from ..timeutil import iso, parse_obs_time, now_beijing

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


def _records_from_wd(wd: dict) -> list[dict]:
    times = wd.get("time") or []
    if not times:
        return []
    n = len(times)
    out: list[dict] = []
    for i in range(n):
        try:
            dt = parse_obs_time(times[i])
        except ValueError:
            continue
        rec = {"time": iso(dt), "source": "wd"}
        for src_key, out_key in NUMERIC_KEYS.items():
            arr = wd.get(src_key)
            rec[out_key] = _to_float(arr[i]) if (arr and i < len(arr)) else None
        out.append(rec)
    return out


def _records_from_table(html: str) -> list[dict]:
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
            dt = parse_obs_time(tds[col["time"]].get_text(strip=True))
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
        last_err = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.get(url, headers=HEADERS, timeout=self.timeout)
                resp.raise_for_status()
                html = resp.text  # requests 已按声明/探测编码解码
                break
            except Exception as e:  # noqa: BLE001
                last_err = e
                logger.warning("抓取 %s 失败（第%d次）: %s", url, attempt + 1, e)
                time.sleep(2 * (attempt + 1))
        else:
            raise RuntimeError(f"抓取 {url} 失败: {last_err}") from last_err

        m = WD_RE.search(html)
        if m:
            try:
                wd = json.loads(m.group(1))
                records = _records_from_wd(wd)
                if records:
                    logger.info("站点 %s 解析 wd JSON 得到 %d 条", station.id, len(records))
                    return records
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("站点 %s 内嵌 wd JSON 解析失败: %s", station.id, e)

        records = _records_from_table(html)
        if records:
            logger.info("站点 %s 回退解析表格得到 %d 条", station.id, len(records))
        else:
            # 页面 200 但无任何观测：视为抓取失败（可能是反爬/登录页/改版），让上层标红
            raise RuntimeError(
                f"站点 {station.id} 页面未解析到任何观测记录（可能页面改版或返回异常页）"
            )
        # 新鲜度检查：最新一条是否明显陈旧
        try:
            latest = max(parse_obs_time(r["time"]) for r in records if r.get("time"))
            age_h = (now_beijing() - latest).total_seconds() / 3600
            if age_h > 3:
                logger.warning("站点 %s 最新观测已陈旧 %.1f 小时（预期近 1 小时内）", station.id, age_h)
        except Exception:
            pass
        return records

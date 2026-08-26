"""北京时（Asia/Shanghai）时间工具。

设计原则：系统全程使用"无时区标记的 naive datetime"来表示北京时墙钟时间，
避免引入 UTC/DST 转换错误。中国不实行夏令时，全年固定 UTC+8，因此：
  - 当前北京时 = (UTC now).astimezone(Asia/Shanghai) 后去掉时区。
  - 所有文件命名、配对比较、JSON 时间键都用该 naive 北京时。
绝不混用 UTC 与北京时做算术。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

BEIJING = ZoneInfo("Asia/Shanghai")


def now_beijing() -> datetime:
    """返回当前北京时（naive datetime，已是墙钟，无 tzinfo）。"""
    return datetime.now(timezone.utc).astimezone(BEIJING).replace(tzinfo=None)


def floor_to_hour(dt: datetime) -> datetime:
    return dt.replace(minute=0, second=0, microsecond=0)


def iso(dt: datetime) -> str:
    """'YYYY-MM-DDTHH:MM'（北京时）。"""
    return dt.strftime("%Y-%m-%dT%H:%M")


def parse_iso(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%dT%H:%M")


def parse_obs_time(s: str) -> datetime:
    """解析 eia-data 页面时间字符串，如 '2026-08-26 20:00' 或带秒。"""
    s = s.strip()
    for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise ValueError(f"无法解析时间字符串: {s!r}")


def ym(dt: datetime) -> str:
    return dt.strftime("%Y-%m")


def ymd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def hour_bucket_days(lead_hours: int) -> int:
    """把时效（小时）映射到 1..16 的"天桶"标签。

    lead 1..24 -> 1, 25..48 -> 2, ..., 361..384 -> 16；lead=0 归入第 1 桶。
    """
    if lead_hours <= 0:
        return 1
    return (lead_hours - 1) // 24 + 1


def lead_label(bucket_days: int) -> str:
    return f"{bucket_days}d"


def add_days(dt: datetime, n: int) -> datetime:
    return dt + timedelta(days=n)

"""北京时（Asia/Shanghai）时间工具。

设计原则：系统全程使用"无时区标记的 naive datetime"来表示北京时墙钟时间，
避免引入 UTC/DST 转换错误。中国不实行夏令时，全年固定 UTC+8，因此：
  - 当前北京时 = (UTC now).astimezone(Asia/Shanghai) 后去掉时区。
  - 所有文件命名、配对比较、JSON 时间键都用该 naive 北京时。
绝不混用 UTC 与北京时做算术。
"""
from __future__ import annotations

import bisect
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
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
    return _parse_iso_cached(s)


@lru_cache(maxsize=20_000)
def _parse_iso_cached(s: str) -> datetime:
    """时间戳解析（带缓存）。

    报告构建会对百万级时间字符串调用本函数，而各模型共享同一条时间轴，
    实际唯一值只有几千个——缓存把 strptime 的完整格式解析开销从构建热点中
    消掉（实测占构建时间约 41%）。datetime 不可变，缓存共享实例是安全的。

    容量（P2-3）：原先 200,000 会让长生命周期进程（`all` 一次跑完全流程）
    持有 20 万个 datetime 且永不释放——单次运行的实际唯一值仅数千，20,000 已
    留足余量，同时把内存上界钉在一个可预期的量级。
    """
    return datetime.fromisoformat(s)


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

# ------------------------------------------------------------------ 时间轴剪枝
# 存档时间轴（`hourly_time`）是**分钟分辨率**的定长 ISO 'YYYY-MM-DDTHH:MM' 且升序，
# 而定长 ISO 的字典序与时间序一致。于是"落在窗口内 / 早于抓取时刻 / 提前天数在
# 区间内"这类逐点时间比较，可以一次二分定界，取代百万次的 datetime 解析与减法——
# 这是 collect 与诊断层最内层循环的主要开销来源（实测 collect 独占 9.3 s、诊断层
# iter_error_samples 独占 6.2 s 里的绝大部分）。
#
# 下面三个函数把"带秒的那一侧"处理干净，保证与原来的 datetime 比较**逐点等价**：
# `fetched_at_bj` 带秒而时间轴只到分，抓取恰好落在同一分钟之内时，判定方向取决于
# 秒是否为零（实测 3224 份带抓取时刻的快照里 3198 份秒非零、134 份的抓取分钟正好
# 是时间轴上的一个点）。忽略这一点会把"同一分钟"的样本错判成未封存——该排除的没
# 排除属于静默改判，本项目不接受。
_ISO_MIN_FMT = "%Y-%m-%dT%H:%M"


def lower_split(times: list[str], dt: datetime) -> int:
    """第一个满足 `时刻 >= dt` 的下标（它之前的都 < dt）。

    dt 带非零秒/微秒时，与 dt 同一分钟的时间轴点仍然**早于** dt，必须一起排除，
    故用 bisect_right；dt 无秒时同分钟即相等、应当保留，故用 bisect_left。
    """
    head = dt.strftime(_ISO_MIN_FMT)
    if dt.second or dt.microsecond:
        return bisect.bisect_right(times, head)
    return bisect.bisect_left(times, head)


def upper_split(times: list[str], dt: datetime) -> int:
    """第一个满足 `时刻 > dt` 的下标（半开区间右界：<= dt 的都保留）。

    时间轴点是分钟分辨率（秒恒为 0），故 `vt <= dt` 等价于 `分钟(vt) <= 分钟(dt)`，
    与 dt 带不带秒无关——带秒时同分钟的点照样 <= dt，bisect_right 正好覆盖它。
    """
    return bisect.bisect_right(times, dt.strftime(_ISO_MIN_FMT))


def unfrozen_split(times: list[str], fetched_at_bj: str | None
                   ) -> tuple[int, datetime | None]:
    """"有效时刻早于抓取时刻"（未封存）这一段的右界。

    返回 `(边界, 抓取时刻)`：下标 < 边界的点判为未封存。与逐点 `vt < fetched` 等价
    ——秒非零时同分钟的点更早（进未封存段），秒为零时同分钟即同时封存（不进）。
    抓取字段缺失返回 `(0, None)`：无从判定，一律按已封存处理。
    """
    if not fetched_at_bj:
        return 0, None
    dt = parse_iso(fetched_at_bj)
    head = dt.strftime(_ISO_MIN_FMT)
    if dt.second or dt.microsecond:
        return bisect.bisect_right(times, head), dt
    return bisect.bisect_left(times, head), dt

# 时间字符串 → 自 1970-01-01 00:00 起的**分钟序号**（整数）的缓存。
_EPOCH_ORD = date(1970, 1, 1).toordinal()
_MINUTE_INDEX: dict[str, int | None] = {}
_MISSING = object()
# 容量上界：见 minute_index 的说明。正常路径靠 clear_minute_index() 维护，
# 这一条只为"从没被清理的调用路径"（诊断层、worker 进程）兜住单调增长。
_MINUTE_INDEX_MAX = 40_000


def minute_index(s: str) -> int | None:
    """`'YYYY-MM-DDTHH:MM'` → 分钟序号；不符合这一格式（带秒/带时区/畸形）→ None。

    collect 的内层循环原本对每个有效时刻都要 `parse_iso` + timedelta 减法 +
    `total_seconds()` + 两次 `.date()`——在 155 万次迭代里是纯粹的对象分配开销
    （实测这一段占 collect 的 1.9 s）。而整条时间轴的**唯一值只有 1,394 个**
    （各家模型共享同一批时刻），于是把时刻压成一个整数分钟序号后，时效与天桶
    都退化成两次整数除法。

    等价性（为什么可以放心用）：填表时用 `strptime` 严格解析，顺带完成格式校验，
    所以缓存值与 datetime 口径同源；分钟分辨率下 `vt − issue` 是 60 秒的整数倍，
    `(im − issue_im) // 60` 与 `int((vt − issue).total_seconds() // 3600)` 逐位相等
    （该量级远小于 2^53，浮点精确）。天桶同理：`im // 1440` 就是以 1970-01-01
    为原点编号的自然日序，两个序之差 == `(vt.date() − issue.date()).days`。

    **返回 None 时调用方必须回退 datetime 路径**——绝不把"解析不了"当成 0，
    那是把缺测伪装成数值的同一种错误。

    缓存的生命周期：调用方（collect）在入口 `clear_minute_index()`；但**并非所有
    调用路径都会清**（诊断层的 iter_error_samples、并行 worker 都会填这张表）。
    因此额外加一道容量上界：满了就整表丢弃重建，而不是让它单调增长。上界取
    单次运行实测键数（约 9,600）的 4 倍——正常路径永远撞不到，只有"确实没人
    清理"时才兜底，且丢弃只是重算一次 strptime，不产生错误结果。

    **返回 None 时调用方必须回退 datetime 路径**——绝不把"解析不了"当成 0，
    那是把缺测伪装成数值的同一种错误。
    """
    v = _MINUTE_INDEX.get(s, _MISSING)
    if v is not _MISSING:
        return v
    try:
        dt = datetime.strptime(s, _ISO_MIN_FMT)
    except (ValueError, TypeError):
        v = None
    else:
        v = (dt.toordinal() - _EPOCH_ORD) * 1440 + dt.hour * 60 + dt.minute
    if len(_MINUTE_INDEX) >= _MINUTE_INDEX_MAX:
        _MINUTE_INDEX.clear()
    _MINUTE_INDEX[s] = v
    return v


def clear_minute_index() -> None:
    """清空分钟序号缓存（每次 collect 入口调用）。"""
    _MINUTE_INDEX.clear()

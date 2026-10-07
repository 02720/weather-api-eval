"""时间轴剪枝的等价性锁定：`lower_split` / `upper_split` / `unfrozen_split` /
`_days_between` 必须与"逐点 datetime 比较"**逐位一致**。

为什么单独开一个文件（而不是靠 golden 对拍兜住）：对拍跑的是**当期存档**，它只能
证明"这批数据上没出分歧"，不能证明"判据本身等价"。剪枝的正确性依赖两条契约：

1. 时间轴是**分钟分辨率**的定长 ISO（`'YYYY-MM-DDTHH:MM'`）且升序；
2. 比较的另一侧（窗口界 / `fetched_at_bj`）可以带秒。

第 2 条是真正的坑：`fetched_at_bj` 带秒（实测 3224 份里 3198 份秒非零），而抓取
分钟**正好落在某个有效时刻**上有 134 例。这些点上"秒是否为零"决定了该样本算不算
未封存——写错方向就是"该排除的事后取数样本被放进来"，属于静默污染评估，比慢严重
得多。所以这里用随机穷举把判据本身钉死：任何一侧的秒、重复时间戳、空轴、边界外
窗口都要覆盖。
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

import pytest

from weather_eval.evaluate import _days_between
from weather_eval.timeutil import (
    _ISO_MIN_FMT,
    clear_minute_index,
    lower_split,
    minute_index,
    unfrozen_split,
    upper_split,
)

pytestmark = pytest.mark.unit

def _axis(n: int, rng: random.Random) -> list[str]:
    """随机生成分钟分辨率、升序（**允许重复**）的时间轴。

    重复是必须覆盖的：同一时刻出现两次时 bisect 的左右界语义差别会直接暴露
    left/right 用错的问题。
    """
    cur = datetime(2026, 8, 1, 0, 0)
    out = []
    for _ in range(n):
        cur = cur + timedelta(minutes=rng.choice([0, 0, 30, 60, 60, 120]))
        out.append(cur.strftime(_ISO_MIN_FMT))
    return out


def _per_point_ge(times: list[str], dt: datetime) -> int:
    """朴素参照：第一个 `时刻 >= dt` 的下标（逐点 datetime 比较）。"""
    return next((i for i, t in enumerate(times) if datetime.fromisoformat(t) >= dt),
                len(times))


def _per_point_gt(times: list[str], dt: datetime) -> int:
    return next((i for i, t in enumerate(times) if datetime.fromisoformat(t) > dt),
                 len(times))


def _per_point_unfrozen(times: list[str], dt: datetime) -> int:
    """朴素参照：第一个**不再**满足 `时刻 < dt` 的下标（= 未封存段的右界）。"""
    return next((i for i, t in enumerate(times)
                 if not (datetime.fromisoformat(t) < dt)), len(times))


@pytest.mark.parametrize("n", [0, 1, 2, 5, 40])
def test_lower_split_matches_per_point(n):
    rng = random.Random(1000 + n)
    times = _axis(n, rng)
    for _ in range(200):
        d = datetime(2026, 8, 1) + timedelta(
            minutes=rng.randint(-240, 1500),
            seconds=rng.choice([0, 0, 0, 1, 30, 59]),
            microseconds=rng.choice([0, 0, 500_000]))
        assert lower_split(times, d) == _per_point_ge(times, d), (times, d)


@pytest.mark.parametrize("n", [0, 1, 2, 5, 40])
def test_upper_split_matches_per_point(n):
    rng = random.Random(2000 + n)
    times = _axis(n, rng)
    for _ in range(200):
        d = datetime(2026, 8, 1) + timedelta(
            minutes=rng.randint(-240, 1500),
            seconds=rng.choice([0, 0, 0, 1, 30, 59]),
            microseconds=rng.choice([0, 0, 500_000]))
        assert upper_split(times, d) == _per_point_gt(times, d), (times, d)


@pytest.mark.parametrize("n", [0, 1, 2, 5, 40])
def test_unfrozen_split_matches_per_point(n):
    rng = random.Random(3000 + n)
    times = _axis(n, rng)
    for _ in range(200):
        d = datetime(2026, 8, 1) + timedelta(
            minutes=rng.randint(-240, 1500),
            seconds=rng.choice([0, 0, 0, 1, 30, 59]),
            microseconds=rng.choice([0, 0, 500_000]))
        got, got_dt = unfrozen_split(times, d.isoformat())
        assert got == _per_point_unfrozen(times, d), (times, d)
        assert got_dt == d


def test_unfrozen_split_without_fetch_time_is_conservative():
    """无抓取时刻 → 边界 0 且时间为 None：一律按"已封存"处理，绝不凭空抹样本。"""
    times = ["2026-09-01T05:00", "2026-09-01T06:00"]
    assert unfrozen_split(times, None) == (0, None)
    assert unfrozen_split(times, "") == (0, None)


# ------------------------------------------------------------------ 关键边界（手挑，不靠随机）
AXIS = ["2026-09-01T05:00", "2026-09-01T06:00", "2026-09-01T07:00"]


def test_fetch_exactly_on_the_minute_does_not_unfreeze_that_hour():
    """抓取时刻秒=0 且正好等于某个有效时刻：该小时 **不算** 未封存。

    这是最容易写反的一格：`vt == fetched` 时 `vt < fetched` 为假。若把整串
    'YYYY-MM-DDTHH:MM:SS' 直接拿去和 16 字符的时间轴点比，定长前缀规则会让
    '...T06:00' < '...T06:00:00' 成立，于是把一个**已封存**的样本错判成未封存
    并剔掉——样本无声减少，榜单却看起来一切正常。
    """
    assert unfrozen_split(AXIS, "2026-09-01T06:00:00")[0] == 1      # 只 05:00 未封存


def test_fetch_with_seconds_unfreezes_the_same_minute():
    """抓取时刻秒≠0：同一分钟的有效时刻确实更早，**应**判未封存。"""
    assert unfrozen_split(AXIS, "2026-09-01T06:00:30")[0] == 2      # 05:00 与 06:00


def test_window_bounds_are_inclusive_on_both_sides():
    """评估窗口两端都是闭区间（原口径 `vt < start` / `vt > end` 才丢弃）。"""
    start = datetime(2026, 9, 1, 6, 0)
    end = datetime(2026, 9, 1, 6, 0)
    assert lower_split(AXIS, start) == 1
    assert upper_split(AXIS, end) == 2
    # 起点带秒 → 同一分钟的点落在窗口外（原口径 vt=06:00 < start=06:00:30）
    assert lower_split(AXIS, start + timedelta(seconds=30)) == 2
    # 终点带秒 → 同一分钟的点仍在窗口内（原口径 vt=06:00 <= end=06:00:30）
    assert upper_split(AXIS, end + timedelta(seconds=30)) == 2


def test_empty_axis_and_out_of_range_bounds():
    assert lower_split([], datetime(2026, 1, 1)) == 0
    assert upper_split([], datetime(2026, 1, 1)) == 0
    assert lower_split(AXIS, datetime(2027, 1, 1)) == len(AXIS)
    assert upper_split(AXIS, datetime(2025, 1, 1)) == 0


# ------------------------------------------------------------------ 天桶对齐
@pytest.mark.parametrize("issue,valid,expect", [
    ("2026-08-01", "2026-08-01", 0),
    ("2026-08-01", "2026-08-02", 1),
    ("2026-08-31", "2026-09-01", 1),          # 跨月
    ("2026-12-31", "2027-01-01", 1),          # 跨年
    ("2028-02-28", "2028-02-29", 1),          # 闰年
    ("2026-02-28", "2026-03-01", 1),          # 非闰年 2/28 的次日是 3/1
    ("2026-08-02", "2026-08-01", -1),         # 负数也要如实给出
])
def test_days_between_matches_date_subtraction(issue, valid, expect):
    """`_days_between` 必须等于原来的 `(vt.date() - issue.date()).days`。"""
    assert _days_between(issue, valid) == expect
    a = datetime.fromisoformat(issue + "T13:00")
    b = datetime.fromisoformat(valid + "T07:00")     # 时刻故意不同：只看自然日
    assert (b.date() - a.date()).days == _days_between(issue, valid)


# ------------------------------------------------------------------ 分钟序号快路径
def test_minute_index_matches_datetime_arithmetic():
    """`lead = (im − issue_im) // 60` 与 `int((vt−issue).total_seconds() // 3600)`
    必须逐位相等；`im // 1440` 之差必须等于 `(vt.date()−issue.date()).days`。

    这是 collect 内层循环的快路径判据。覆盖跨月/跨年/闰日/负差，以及"起报
    不在整分"这一必须回退的情形。
    """
    rng = random.Random(4404)
    base = datetime(2026, 1, 1)
    for _ in range(3000):
        issue = base + timedelta(minutes=rng.randint(0, 500_000))
        vt = issue + timedelta(minutes=rng.randint(-3000, 3000))
        i_s = issue.strftime(_ISO_MIN_FMT)
        v_s = vt.strftime(_ISO_MIN_FMT)
        im, iim = minute_index(v_s), minute_index(i_s)
        assert im is not None and iim is not None
        assert (im - iim) // 60 == int((vt - issue).total_seconds() // 3600)
        assert im // 1440 - iim // 1440 == (vt.date() - issue.date()).days


def test_minute_index_rejects_non_minute_formats():
    """带秒 / 带时区 / 空格分隔 / 畸形 → None（调用方必须回退，绝不按 0 处理）。"""
    for bad in ("2026-09-01T06:00:30", "2026-09-01T06:00+08:00",
                "2026-09-01 06:00", "2026-13-45T99:99", "", "not-a-date"):
        assert minute_index(bad) is None, bad


def test_minute_index_cache_is_cleared_and_stable():
    """缓存清空后重算值不变（生命周期纪律：可清、清后等价）。"""
    key = "2026-08-01T13:00"
    first = minute_index(key)
    clear_minute_index()
    assert minute_index(key) == first
    # None 也要被缓存：畸形串反复查不应每次都走 strptime 异常路径
    clear_minute_index()
    assert minute_index("2026-09-01T06:00:30") is None
    assert minute_index("2026-09-01T06:00:30") is None

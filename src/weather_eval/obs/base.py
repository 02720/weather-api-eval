"""观测数据源抽象基类。"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any


class ObsSource(ABC):
    """观测源：给定站点，返回其最近实况记录列表。

    每条记录为 dict：{"time": "<ISO 北京时>", "temp": float|None, "rain": float|None, ...}
    """

    @abstractmethod
    def fetch(self, station: Any) -> list[dict]:
        ...


# 最新观测滞后阈值（小时）：判定"主源是否停摆"。此前 chain.py 与 eia_data.py
# 各持一份 3.0、同口径靠注释维持——一边改动另一边静默漂移：源内告警说"陈旧了"
# 而编排层判"可用"（或反之）。单一出处，两侧引用同一常量。
STALE_HOURS = 3.0

# 覆盖门槛（小时）：一轮至少要有多少小时才算"窗口没被截断"。eia-data 正常
# 返回 24 条；门槛只拦"明显截断"，真正可用性还有异常/新鲜度两条（见 chain.py）。
MIN_HOURS = 6


def parse_obs_float(v: Any) -> float | None:
    """观测要素值的统一数值归一：缺测/非数值/非有限一律 None（**绝不折算成 0.0**）。

    此前 eia_data 与 cma_data 各持一份 95% 相同的 `_to_float`，靠注释约定同步
    ——复制后各自演化正是口径漂移的温床（同一批观测经两源解析，一边 None 一边
    异常值，`_same_obs` 判等随之失真）。缺测词表取两源并集：""/none/nan/na/
    null/"-"/"—"（float 本就解析不了 "-" 与 "—"，显式列出是为了与真数值失败
    区分开，也让词表可读）。

    bool 必须先挡：**float(True) == 1.0**，页面里一个 `true` 会变成 1°C 实况
    写进归档（forecast 侧 `_num` 家族均已挡，观测侧这里是唯一的统一入口）。
    """
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
    # NaN != NaN；inf 的 isfinite 为 False——非有限值经 plausible 的合理域
    # 多半也拦得住，但在数值层就地归一，让"缺测"只有一种到达方式。
    return f if math.isfinite(f) else None


# 要素的物理合理域（第四轮 P2-10）：越界即缺测并计数。气象哨兵值（如风向不明码
# 999017、占位 999.9）曾原样落库 58 次——temp/rain 恰好干净只是运气，通道已被
# 证明是活的。"缺测绝不伪装成数值"必须包含"非物理值"。
OBS_PLAUSIBLE_RANGES: dict[str, tuple[float, float]] = {
    "temp": (-60.0, 60.0),
    "rain": (0.0, 500.0),
    "pressure": (800.0, 1100.0),
    "humidity": (0.0, 100.0),
    "wind_speed": (0.0, 120.0),
    "wind_dir": (0.0, 360.0),
    "visibility": (0.0, 100_000.0),
}


def plausible(field: str, v: float | None) -> float | None:
    """要素值落在物理合理域内原样返回；越界/缺测一律 None（绝不折算 0）。"""
    if v is None:
        return None
    rng = OBS_PLAUSIBLE_RANGES.get(field)
    if rng is None:
        return v
    lo, hi = rng
    return v if lo <= v <= hi else None

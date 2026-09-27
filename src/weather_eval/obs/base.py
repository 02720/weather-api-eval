"""观测数据源抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ObsSource(ABC):
    """观测源：给定站点，返回其最近实况记录列表。

    每条记录为 dict：{"time": "<ISO 北京时>", "temp": float|None, "rain": float|None, ...}
    """

    @abstractmethod
    def fetch(self, station: Any) -> list[dict]:
        ...


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

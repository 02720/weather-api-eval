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

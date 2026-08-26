"""预报数据源抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ForecastProvider(ABC):
    """预报源：给定站点与模型列表，返回一次"起报快照"。

    快照 dict 结构：
    {
      "issue_iso": "YYYY-MM-DDTHH:MM",   # 起报时刻（北京时，按轮次实际时间）
      "station_id": str,
      "source": "open-meteo",
      "models": [model, ...],
      "grid_lat": float, "grid_lon": float, "elevation": float,  # 实际吸附格点
      "hourly_time": ["YYYY-MM-DDTHH:MM", ...],   # 共享时间轴（北京时）
      "data": { model: {"temperature_2m": [...], "precipitation": [...]} }
    }
    """

    @abstractmethod
    def fetch_snapshot(self, station: Any, models: list[str]) -> dict:
        ...

"""预报数据源抽象基类。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ForecastProvider(ABC):
    """预报源：给定站点与模型列表，返回一次"起报快照"。

    返回形态（CLI 两种都支持）：
    1. dict —— 共享同一条时间轴与同一 issue 的多模型快照（Open-Meteo/彩云/和风）：
    {
      "issue_iso": "YYYY-MM-DDTHH:MM",   # 起报时刻（北京时，按轮次实际时间）
      "station_id": str,
      "source": "open-meteo",
      "models": [model, ...],
      "grid_lat": float, "grid_lon": float, "elevation": float,  # 实际吸附格点
      "hourly_time": ["YYYY-MM-DDTHH:MM", ...],   # 共享时间轴（北京时）
      "data": { model: {"temperature_2m": [...], "precipitation": [...]} }
    }
    2. list[dict] —— 各模型独立的快照列表，每份结构与上述 dict 相同但只含一个模型、
       各自的 issue_iso 与 hourly_time。适用于"各模式最新可用起报可能不同步"的源
       （中科天机）：独立 issue 保证时效（lead）分组不被跨模式错位污染。
    """

    @abstractmethod
    def fetch_snapshot(self, station: Any, models: list[str]) -> dict | list[dict]:
        ...

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

    可选的"逐日预报"块（**向后兼容的可选扩展**，缺省即旧行为）：
    多数 API 的逐日预报比逐小时预报覆盖得更远（如逐小时到 5~10 天、逐日到 15 天）。
    若该源能提供自己的逐日预报，按下述结构一并封存，评估引擎即可在逐小时断供后
    继续用日产品做**按天**评估：
    {
      "daily_time": ["YYYY-MM-DD", ...],        # 北京时自然日（升序、去重）
      "daily": { model: {
          "temp_max": [...],                    # 当日最高气温 ℃
          "temp_min": [...],                    # 当日最低气温 ℃
          "precipitation": [...],               # 当日 24h 累计降水 mm
      }}
    }
    三条数组与 daily_time 等长、按索引对齐；缺测一律 null（绝不填 0）。
    允许**只接温度**的半块：源若无逐日累计降水产品（只有概率/量级码/日内统计量），
    precipitation 给全 null 数组并留档原因——把概率、量级档或"日内均值×24"折算成
    累计 mm 都是凭空造值，绝不允许（星图 pre_day/pre_night 恒 5.0 量级码、彩云
    daily.precipitation 仅日内统计量，即为反例证据，见各自 docstring）。

    使用边界（第一性原理，不可越界）：
    - 日产品**只用于按天评估补位**（日最高/最低温、日降水），且仅在该日逐小时覆盖
      不足（< daily_min_hours）时才启用——逐小时覆盖充足时仍用逐小时聚合，保证与
      历史存档同一口径。
    - **绝不由日产品反推逐小时序列**（插值/均摊都是凭空造出日内变化），逐小时
      指标与排行榜不因此产生任何样本。
    - 日界窗口与观测口径存在固有差异（详见 README"逐日预报补位"一节），各源口径
      假设必须像降水口径那样写进对应 provider 的 docstring 并留档。
    - 日产品抓取失败**不得**拖垮整份起报快照：逐小时是主干，日产品只是延长线，
      失败时降级为不带该块的快照（WARNING 可见），快照错过即无法追补。
    """

    @abstractmethod
    def fetch_snapshot(self, station: Any, models: list[str]) -> dict | list[dict]:
        ...

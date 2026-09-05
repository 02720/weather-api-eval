"""配置加载：stations.yaml。"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "stations.yaml"

# 评估默认参数（可被 stations.yaml 的 eval 段覆盖）
DEFAULT_EVAL = {
    "temp_accuracy_limits": [1, 2],   # ±1°C、±2°C 准确率
    "rain_threshold_mm": 0.1,          # 有无降水阈值（国内业务：≥0.1mm 记为有降水）
    "hourly_lead_days": 16,            # 逐小时评估最大时效（天），即 lead 1..384h
    "daily_max_offset_days": 16,       # 按天评估最大日偏移（天），即 offset 1..16
    "daily_min_hours": 20,             # 按天评估的日覆盖门槛：观测/预报任一侧当天
                                       # 非缺测小时数低于此值，该天该要素不入样
                                       # （防"缺测折算 0.0"与"部分日累计偏低"伪装成技巧）
    "min_sample": 5,                    # 样本数低于此值视为"样本不足"，不出结论
    "daily_source_fallback": True,      # 逐小时覆盖不足时，允许用快照自带的逐日
                                        # 预报（daily_time/daily 块）为按天评估补位。
                                        # 只补按天轨道，绝不反推逐小时；关掉即回到
                                        # 纯逐小时聚合的旧口径（用于对照/回归）
}


class Station:
    def __init__(self, data: dict):
        self.id: str = data["id"]
        self.name: str = data.get("name", data["id"])
        self.lat: float = float(data["lat"])
        self.lon: float = float(data["lon"])
        self.obs_url: str = data.get("obs_url", "")


class Config:
    def __init__(self, data: dict):
        self.raw = data
        self.models: list[str] = list(data.get("models", []))
        self.stations: list[Station] = [Station(s) for s in data.get("stations", [])]
        self.eval: dict[str, Any] = {**DEFAULT_EVAL, **(data.get("eval") or {})}

    @property
    def station_ids(self) -> list[str]:
        return [s.id for s in self.stations]


@lru_cache(maxsize=1)
def load_config(path: str | Path | None = None) -> Config:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with open(p, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))

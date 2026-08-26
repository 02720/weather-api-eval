"""数据存档层。

所有数据以 JSON 落盘到 <项目根>/data/ 下，并在 git 中跟踪：
  data/obs/{station_id}/{YYYY-MM}.json           观测（按时间键去重合并）
  data/forecasts/{station_id}/{model}/{issue}.json  起报快照（幂等）
  data/metrics/{period}/{file}.json              评估结果（可选缓存）

写入采用"临时文件 + 原子 rename"避免半文件；读取容错（缺失返回空）。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .timeutil import ym, ymd

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _root() -> Path:
    return Path(os.environ.get("WEATHER_EVAL_DATA_ROOT", PROJECT_ROOT / "data"))


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ------------------------------------------------------------------ 观测
def save_obs(station_id: str, records: list[dict]) -> int:
    """把一批观测记录合并写入该站当月文件，按时间键去重；返回新增/更新的条数。"""
    if not records:
        return 0
    months: dict[str, dict] = {}
    for r in records:
        months.setdefault(ym(parse_dt(r["time"])), {})[r["time"]] = r
    updated = 0
    for month, rec_map in months.items():
        path = _root() / "obs" / station_id / f"{month}.json"
        existing: dict = {}
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        for k, v in rec_map.items():
            if k not in existing or v != existing[k]:
                updated += 1
            existing[k] = v
        _atomic_write_json(path, existing)
    return updated


def load_obs(station_id: str, month: str | None = None) -> dict[str, dict]:
    """读取某站全部观测（指定 month 时只取该月）。返回 time_iso -> record。"""
    result: dict[str, dict] = {}
    base = _root() / "obs" / station_id
    if not base.exists():
        return result
    for p in sorted(base.glob("*.json")):
        if month and p.stem != month:
            continue
        with open(p, "r", encoding="utf-8") as f:
            result.update(json.load(f))
    return result


# ------------------------------------------------------------------ 预报快照
def _issue_filename(issue_iso: str) -> str:
    return issue_iso.replace(":", "") + ".json"


def save_forecast_snapshot(station_id: str, model: str, snapshot: dict) -> bool:
    """写入起报快照；若同站同模型同起报时刻已存在则跳过（幂等）。返回是否新建。"""
    issue_iso = snapshot["issue_iso"]
    path = _root() / "forecasts" / station_id / model / _issue_filename(issue_iso)
    if path.exists():
        return False
    _atomic_write_json(path, snapshot)
    return True


def list_forecast_snapshots(station_id: str, model: str) -> list[dict]:
    base = _root() / "forecasts" / station_id / model
    if not base.exists():
        return []
    out = []
    for p in sorted(base.glob("*.json")):
        with open(p, "r", encoding="utf-8") as f:
            out.append(json.load(f))
    return out


# ------------------------------------------------------------------ 指标（可选缓存，当前实现每次重算）
def save_metrics(period: str, name: str, obj: Any) -> None:
    _atomic_write_json(_root() / "metrics" / period / f"{name}.json", obj)


def parse_dt(s: str) -> Any:
    from .timeutil import parse_iso, parse_obs_time
    try:
        return parse_iso(s)
    except ValueError:
        return parse_obs_time(s)

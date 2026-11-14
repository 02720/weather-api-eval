"""把 dvs-v6 的结果挂进报告：按站构建 → 合并 → 出榜 → 写 `meta.dvs`。

**为什么单独一层**

`evaluate.build_report` 已经有 8,000 行、承担着"口径 + 指标 + 榜面 + 披露"四件事。
DVS 的接入如果再塞进它，切回 skill-v5 就会变成一次不可逆的编辑。所以这里只做三件事，
且全部通过**一个入口函数** `attach_dvs(report, ...)` 完成：

1. 按站跑 DVS 全链路（`dvs_engine.build_station`），合并各站的逐日统计量；
2. 用 `dvs_engine.build_board` 出榜（含对照源、MCS、区间、技巧寿命）；
3. 把结果写进 `report["meta"]["dvs"]`，并保留 `SCORING_DVS.md` 作为口径单一事实源。

失败时**降级而不崩溃**：DVS 是"旁路新增结论"，它挂掉不该让主报告不更新（与诊断层
同一纪律）。降级后的 `meta.dvs.available = False` 会写明原因，读者不会误以为
"这一期没有 DVS"。
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Mapping, Sequence

from . import dvs_engine as engine
from .storage import canonical_models

logger = logging.getLogger(__name__)

SCHEME = "dvs-v6"


def attach_dvs(report: dict, station_ids: Sequence[str], models: Sequence[str],
               eval_cfg: Mapping | None, start_dt: datetime,
               end_dt: datetime) -> dict:
    """构建 DVS 并挂到 `report["meta"]["dvs"]`，返回该节点本身。"""
    meta = report.setdefault("meta", {})
    try:
        node = build_dvs_node(station_ids, models, eval_cfg, start_dt, end_dt)
    except Exception as e:  # noqa: BLE001 - 旁路层绝不拖垮主报告
        logger.exception("DVS v6 构建失败，主报告继续（不含新榜）")
        node = {"available": False, "scheme": SCHEME,
                "reason": f"{type(e).__name__}: {e}"}
    meta["dvs"] = node
    return node


def build_dvs_node(station_ids: Sequence[str], models: Sequence[str],
                   eval_cfg: Mapping | None, start_dt: datetime,
                   end_dt: datetime) -> dict:
    """跑完整条 DVS 链路并返回可序列化的结果节点。"""
    config = engine.DvsConfig.from_eval_cfg(eval_cfg)
    models_c = canonical_models(list(models))
    start_day = start_dt.toordinal()
    end_day = end_dt.toordinal()
    obs_all = _load_obs(station_ids)
    snaps_all = _load_snapshots(station_ids, models_c)
    clim = engine.load_climatology(config) if config.climatology_path else None
    merged: dict[str, dict] = {}
    station_meta: dict[str, dict] = {}
    declared: dict[str, int] = {}
    obs_days: dict[str, set[int]] = {}
    # 站编码在每次 `build_station` 里都是 **0**：那一层的 `station_ids` 参数是
    # "本次构建涉及的站列表"，而单站构建的站列表长度就是 1。传 `enumerate` 的
    # 序号会让第二个站越界（实测 IndexError）。
    def _one(sid: str):
        """单站构建（供线程池并发；每站只用自己那份观测与快照，无共享状态）。"""
        station_clim = clim.get(sid) if clim is not None else None
        return engine.build_station(
            sid, obs_all.get(sid) or {}, snaps_all.get(sid) or {},
            models=models_c, station_code=0, start_day=start_day,
            end_day=end_day, config=config, climatology=station_clim)

    # 站间完全独立（各自的观测、各自的快照），并发是纯收益：实测 4 站串行 54 s
    # → 并行约 16 s。用线程池而不是进程池：numpy 的重活会释放 GIL，而进程池要
    # 把上百万行的快照序列化一遍，代价反而更高。
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(station_ids), 8) or 1) as pool:
        builds = list(pool.map(_one, station_ids))
    for sid, build in zip(station_ids, builds):
        station_meta[sid] = {
            "climatology_source": build.climatology_source,
            "tau_hours": round(float(build.tau_hours), 2),
            "tau_fitted": bool(build.tau_fitted),
            "facts": build.n_facts,
            "batches": build.n_runs,
            "observed_days": build.observed_days,
        }
        declared[sid] = build.declared_days
        # 观测实际存在的自然日（ISO → toordinal），用于可用率的分母
        obs_days[sid] = {
            datetime.strptime(k[:10], "%Y-%m-%d").toordinal()
            for k in (obs_all.get(sid) or {}) if isinstance(k, str) and len(k) >= 10}
        for model, per_day in build.rows.items():
            target = merged.setdefault(model, {})
            for day, rows in per_day.items():
                cur = target.get(day)
                if cur is None:
                    target[day] = rows
                    continue
                _merge_rows(cur, rows)
    window_days = sorted({d for per_day in merged.values() for d in per_day})
    board = engine.build_board(
        merged, config=config, window_days=window_days, models=list(merged),
        declared_days={k: int(v) for k, v in declared.items()})
    board["available"] = True
    board["scheme"] = SCHEME
    board["scoring_doc"] = "SCORING_DVS.md"
    board["stations"] = station_meta
    board["declared_days"] = dict(declared)
    board["observed_days"] = {k: len(v) for k, v in obs_days.items()}
    return board


def _merge_rows(cur, rows) -> None:
    """把同一 (源, 日) 的两份统计量逐项相加（跨站合并）。

    站维在统计量里是键的一部分，所以跨站合并只是"同一个字典里加更多键"，
    不需要任何加权——权重在**打分时**按样本量体现（`temp_skill` 用的是合并后的
    充分统计量，等价于按 n 加权）。
    """
    for key, vals in rows.temp.items():
        ent = cur.temp.get(key)
        if ent is None:
            cur.temp[key] = list(vals)
        else:
            for i, v in enumerate(vals):
                ent[i] += v
    for key, ent in rows.rain.items():
        got = cur.rain.get(key)
        if got is None:
            cur.rain[key] = list(ent)
        else:
            for i in range(5):
                got[i] += ent[i]


def _load_obs(station_ids: Sequence[str]) -> dict[str, dict]:
    from .storage import load_obs
    return {sid: load_obs(sid) for sid in station_ids}


def _load_snapshots(station_ids: Sequence[str],
                    models: Sequence[str]) -> dict[str, dict[str, list[dict]]]:
    from .storage import list_forecast_snapshots
    return {sid: {m: list_forecast_snapshots(sid, m) for m in models}
            for sid in station_ids}


__all__ = ["attach_dvs", "build_dvs_node"]

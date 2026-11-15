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

#: 站构建并发度。**按内存定，不按站数定**（理由见 `build_dvs_node` 内的注释：
#: 进程峰值 ≈ 并发站数 × 单站峰值，而站构建只占 DVS 总时长的零头）。
STATION_WORKERS = 2


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
    obs_all, snaps_all = _preloaded_inputs(station_ids, models_c)
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

    # 站间完全独立（各自的观测、各自的快照），并发是纯收益：实测 4 站串行 ~58 s
    # → 4 路并行 ~20 s。用线程池而不是进程池：numpy 的重活会释放 GIL，而进程池
    # 要把上百万行的快照序列化一遍，代价反而更高。
    #
    # 并发度按**内存**定，不按站数定：单站构建的峰值来自列式事实表（真实源 33 源
    # + 对照源 8 份，逐小时 ~7.6M 行）与日聚合/累积的叠加，站与站之间不共享，于是
    # 进程峰值 ≈ 并发站数 × 单站峰值。同一台 4 核机器、同一窗口实测：
    #   4 路并发 → 峰值 7.1 GB（耗时 216~271 s，两次同口径跑出来的抖动就有 25%）
    #   2 路并发 → 峰值 5.1 GB（耗时 203 s）
    # 也就是说"拉满并发"既不多买时间（站段只占零头，自助 200 次与出榜是串行的，
    # numpy 的 BLAS 线程在 4 核上互抢反而更慢），却要多占 2 GB —— 在"13 个月数据
    # 下 ~20 GB 会 OOM"这条硬约束下，并发度必须按内存定（预算见 perf_budget.json）。
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(station_ids), STATION_WORKERS)
                            or 1) as pool:
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


def _preloaded_inputs(station_ids: Sequence[str], models: Sequence[str]
                      ) -> tuple[dict[str, dict], dict[str, dict[str, list[dict]]]]:
    """取观测与快照：**复用 evaluate 的进程内预加载缓存**。

    为什么必须复用：`build_report` 里 dvs-v6 分支是**叠加**在 skill-v5 影子口径
    之后的，影子口径已经用 `evaluate._preload` 把全部观测与快照读进内存；这里
    再 `list_forecast_snapshots` / `load_obs` 读一遍，等于同一进程把同一批文件
    （数千份快照 JSON）解析两遍。成本是双份的：那两遍解析（本机 page cache 已热
    时仍要 4 s，CI runner 冷盘上更高），以及两份已解析字典同时驻留的峰值内存。
    正确性不依赖任何时序假设：缓存键含 `data_version()`，写路径落盘即失效；不命中
    时 `_preload` 自己读盘，返回内容与直接调用 storage **逐位相同**
    （`_preload` 只做 listing，不清洗、不过滤——清洗在 `_snaps_for` 里按 `id()`
    记忆并返回副本，不回写缓存对象）。

    独立进程（例如只跑 DVS 的调用方）语义不变：一样是"读一次、用一次"。
    """
    from .evaluate import _preload
    models_c = canonical_models(list(models))
    obs_maps, snaps = _preload(list(station_ids), list(models_c))
    return obs_maps, {sid: {m: snaps.get((sid, m)) or [] for m in models_c}
                      for sid in station_ids}


__all__ = ["attach_dvs", "build_dvs_node"]

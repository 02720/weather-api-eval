"""报告页的"统计诊断层"：把对抗式审查里"该披露但还没算"的几项落成真数字。

这一层的定位（第一性原理）：评估层回答"谁第几名"，诊断层回答"这个第几名值多少
信任"。两者必须分开——诊断层只读冻结快照与已算好的榜单，**不改任何指标口径**，
因此它算出来的东西永远不会改变总榜逐位结果（回归测试锁定）。

当前提供的诊断：

1. ``cross_source_correlation``（审查 P0-1）
   从冻结快照重建各家在同一批"站点 × 有效时刻 × 提前天数"上的温度误差序列，估计
   两两相关。26 家同榜里 11 家共享 Open-Meteo 的同一条时间轴与同一个起报锚点，
   5 个天机变体出自同一家产品——它们不是 26 份独立证据。输出 ρ̄、相关族、以及
   k_eff = m/(1+(m−1)ρ̄)（多重比较的有效检验数）。
2. ``station_rho``（审查 P1-4）
   按**站对**披露 ρ̄（不再是"全部站对的平均"这一个标量），并给出有效独立站数。
3. ``fingerprint_drift``（审查 P1-5）
   本项目最危险的失效形态是"HTTP 200 + 语义已变"：降水从累计变瞬时、时间从 UTC
   变北京时、返回点数被静默截断。这些都写在快照契约里，逐轮比对即可发现。
4. ``bridge_and_jackknife``（审查 P1-1）
   长时效段只有少数源覆盖，短时效段人人都在——两段之间的相对位置靠少数"桥梁源"
   串起来。逐个剔除入设计的源重算行分，直接回答"这张榜是不是被一两家撑着"。
5. ``corridor_sensitivity``（审查 P0-2）
   敏感性分析此前只扰动了**权重**，但决定名次的还有**口径**：n_eff 入围门槛、
   ridge 收缩、格子权重门槛、长尾桶门槛。这些改变的不是"各项占多少"，而是
   "哪些样本算数"。逐个重算总榜，报告 Spearman 与前十换了几家。

全部为**纯函数式**的离线计算：输入是 data/ 与已算好的榜单，输出是一份可序列化
的 dict。接入方式是把它并进 ``meta.diagnostics`` 由报告页展示——**本轮尚未接入**：
报告页仍沿用远端的单文件内联设计，诊断结果目前只能由调用方自行读取。
"""
from __future__ import annotations

import glob
import gzip
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .. import stats as st

logger = logging.getLogger(__name__)

# 参与跨源相关的提前天数区间：与总榜的入榜时效范围一致（1~10 天），
# 超出这个区间的格子只有极少数源覆盖，相关估计会被个别源的缺测模式主导。
CROSS_SOURCE_LEAD_DAYS = (1, 10)
# 指纹漂移比对的时间分界：把周期切成前后两半，比较两半的契约字段众数
FINGERPRINT_SPLIT_FRACTION = 0.5


# ------------------------------------------------------------------ 数据读取
def _read_json(path: str) -> dict | None:
    try:
        if path.endswith(".gz"):
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:            # 单份损坏快照不该让整份诊断失败
        logger.warning("快照读取失败 %s：%s", path, exc)
        return None


def _snapshot_files(station_dir: Path) -> list[str]:
    return sorted(glob.glob(str(station_dir / "*.json"))) + sorted(
        glob.glob(str(station_dir / "*.json.gz")))


def _parse_iso(s: str) -> datetime | None:
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return None


def load_observations(obs_dir: Path, months: Iterable[str]) -> dict[str, dict]:
    """读实况：{time_iso: {"temp": float|None, "rain": float|None}}。"""
    out: dict[str, dict] = {}
    for month in months:
        p = obs_dir / f"{month}.json"
        if not p.exists():
            continue
        raw = _read_json(str(p))
        if not isinstance(raw, dict):
            continue
        for t, rec in raw.items():
            if not isinstance(rec, dict):
                continue
            out[t] = {"temp": rec.get("temp"), "rain": rec.get("rain")}
    return out


def iter_error_samples(forecasts_root: Path, obs_by_station: dict[str, dict],
                       months: list[str], start: str | None = None,
                       end: str | None = None,
                       ) -> tuple[dict[str, dict[tuple, float]],
                                   dict[str, dict[tuple, float]]]:
    """遍历全部冻结快照，重建"逐样本误差"。

    返回 ``(temp_err, rain_err)``，结构均为 ``{model: {(station, valid_time, lead_day): error}}``。

    键为什么是三元组：跨源相关必须在**同一个物理样本**上比。同一有效时刻、同一
    提前天数、同一站点的两家预报，面对的是同一批天气与同一个锚点误差——这才是
    "同源冗余"的物理来源。缺测小时自然跳过，不做任何插补（插补会凭空制造相关）。

    error 的符号统一为「预报 − 实况」：相关的符号不依赖方向，但一致的定义便于复核。
    """
    temp_err: dict[str, dict[tuple, float]] = {}
    rain_err: dict[str, dict[tuple, float]] = {}
    lo, hi = CROSS_SOURCE_LEAD_DAYS
    t_start = _parse_iso(start) if start else None
    t_end = _parse_iso(end) if end else None

    for station_dir in sorted(forecasts_root.iterdir()):
        if not station_dir.is_dir():
            continue
        station = station_dir.name
        obs = obs_by_station.get(station) or {}
        if not obs:
            continue
        for model_dir in sorted(station_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            t_map = temp_err.setdefault(model, {})
            r_map = rain_err.setdefault(model, {})
            for path in _snapshot_files(model_dir):
                snap = _read_json(path)
                if not isinstance(snap, dict):
                    continue
                issue = _parse_iso(snap.get("issue_iso") or "")
                if issue is None:
                    continue
                times = snap.get("hourly_time") or []
                data = snap.get("data") or {}
                if not isinstance(data, dict) or not times:
                    continue
                # 一份快照可能同时封装多个模型（Open-Meteo 批量接口）
                for mid, series in data.items():
                    if not isinstance(series, dict):
                        continue
                    key_model = mid if mid in (snap.get("models") or []) else model
                    temps = series.get("temperature_2m") or []
                    rains = series.get("precipitation") or []
                    tg = t_map if key_model == model else temp_err.setdefault(key_model, {})
                    rg = r_map if key_model == model else rain_err.setdefault(key_model, {})
                    for idx, t_iso in enumerate(times):
                        vt = _parse_iso(t_iso)
                        if vt is None:
                            continue
                        if t_start and vt < t_start:
                            continue
                        if t_end and vt > t_end:
                            continue
                        lead_days = int((vt - issue) // timedelta(days=1))
                        if lead_days < lo or lead_days > hi:
                            continue
                        rec = obs.get(t_iso)
                        if not rec:
                            continue
                        key = (station, t_iso, lead_days)
                        if idx < len(temps) and rec.get("temp") is not None:
                            fv = temps[idx]
                            if isinstance(fv, (int, float)) and abs(float(fv)) < 900:
                                tg[key] = float(fv) - float(rec["temp"])
                        if idx < len(rains) and rec.get("rain") is not None:
                            fv = rains[idx]
                            if isinstance(fv, (int, float)) and float(fv) >= 0:
                                # 降水用"有无"的偏离（观测 −0.5/预报 +0.5 会互相抵消）
                                rg[key] = float(fv) - float(rec["rain"])
    return temp_err, rain_err


# ------------------------------------------------------------------ P0-1 跨源相关
def cross_source_correlation(temp_err: dict[str, dict[tuple, float]],
                             families: dict[str, str] | None = None,
                             min_overlap: int = st.SOURCE_CORR_MIN_OVERLAP,
                             ) -> dict[str, Any]:
    """跨源冗余的完整诊断（审查 P0-1 的主产出）。"""
    usable = {m: s for m, s in temp_err.items() if len(s) >= min_overlap}
    if len(usable) < 3:
        return {"available": False,
                "reason": f"可用于估计跨源相关的源不足 3 家（实得 {len(usable)} 家）",
                "n_models": len(usable)}
    out = st.source_corr_cluster(usable, families=families, min_overlap=min_overlap)
    out["samples_per_model"] = {m: len(s) for m, s in sorted(usable.items())}
    out["excluded_small"] = sorted(set(temp_err) - set(usable))
    return out


# ------------------------------------------------------------------ P1-4 站对相关
def station_rho(temp_err: dict[str, dict[tuple, float]],
                min_overlap: int = st.CROSS_STATION_MIN_OVERLAP) -> dict[str, Any]:
    """按站对披露 ρ̄：逐源算再按 Fisher-z 合并，而不是把所有源混成一个序列。

    混成一个序列会让"某源在某站缺测"变成噪声；逐源算、再合并，才是"站间冗余"
    这个量的正确口径（与 cross_station_rho 的既有口径一致）。
    """
    stations = sorted({k[0] for s in temp_err.values() for k in s})
    if len(stations) < 2:
        return {"available": False, "reason": "站点数不足 2", "stations": stations}

    acc: dict[tuple[str, str], list[tuple[float, int]]] = {}
    for model, series in temp_err.items():
        by_station: dict[str, dict[tuple, float]] = {}
        for (sta, t, lead), v in series.items():
            by_station.setdefault(sta, {})[(t, lead)] = v
        res = st.station_rho_matrix(by_station, min_overlap=min_overlap)
        for pair in res["pairs"]:
            if pair["rho"] is None:
                continue
            key = (pair["a"], pair["b"])
            # n 用公共样本数近似（Fisher-z 的权重），这里用该源在该站对上的最小长度
            n_approx = min(len(by_station.get(pair["a"], {})),
                           len(by_station.get(pair["b"], {})))
            acc.setdefault(key, []).append((pair["rho"], max(n_approx, 4)))

    pairs = []
    flat = []
    for (a, b), items in sorted(acc.items()):
        rs = [r for r, _ in items]
        ns = [n for _, n in items]
        combined = st.fisher_z_combine(rs, ns)
        pairs.append({"a": a, "b": b, "rho": None if combined is None else round(combined, 3),
                      "mean_rho": round(float(np.mean(rs)), 3), "n_sources": len(rs)})
        if combined is not None:
            flat.append(combined)
    mean_rho = float(np.mean(flat)) if flat else None
    return {
        "available": bool(flat),
        "stations": stations,
        "pairs": pairs,
        "mean_rho": None if mean_rho is None else round(mean_rho, 4),
        "k_eff": round(st.effective_independent_count(len(stations), mean_rho), 2),
        "n_stations": len(stations),
    }


# ------------------------------------------------------------------ P1-5 数据指纹漂移
_FINGERPRINT_FIELDS = (
    "resolution_hours", "precip_unit", "precip_accum_window_hours",
    "issue_source", "source",
)


def fingerprint_drift(forecasts_root: Path, months: list[str],
                      split_fraction: float = FINGERPRINT_SPLIT_FRACTION) -> dict[str, Any]:
    """逐源比对快照契约字段的前后半周期众数，漂移即告警（审查 P1-5）。

    为什么这比 health 的"最新快照超过 N 小时未更新"更关键：本项目源的典型失效
    形态是 HTTP 200 + 结构还在、语义已变（降水从累计变瞬时、时间从 UTC 变北京时、
    UA 被改后静默截断到 48 小时）。这类失效 health 完全看不见，但契约字段会变。
    """
    per_model: dict[str, list[tuple[datetime, dict]]] = {}
    for station_dir in sorted(forecasts_root.iterdir()):
        if not station_dir.is_dir():
            continue
        for model_dir in sorted(station_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model = model_dir.name
            for path in _snapshot_files(model_dir):
                snap = _read_json(path)
                if not isinstance(snap, dict):
                    continue
                issue = _parse_iso(snap.get("issue_iso") or "")
                if issue is None:
                    continue
                fp = {k: snap.get(k) for k in _FINGERPRINT_FIELDS}
                fp["actual_hours"] = snap.get("actual_hours")
                fp["complete"] = snap.get("complete")
                fp["missing_shards"] = bool(snap.get("missing_shards"))
                per_model.setdefault(model, []).append((issue, fp))

    out_models: dict[str, Any] = {}
    drifted: list[str] = []
    for model, rows in sorted(per_model.items()):
        rows.sort(key=lambda x: x[0])
        if len(rows) < 4:
            out_models[model] = {"snapshots": len(rows), "drift": False,
                                 "note": "快照数不足 4，不做漂移判定"}
            continue
        cut = max(1, int(len(rows) * split_fraction))
        early, late = rows[:cut], rows[cut:]

        def mode(field: str, chunk: list[tuple[datetime, dict]]) -> Any:
            vals = [fp.get(field) for _, fp in chunk if fp.get(field) is not None]
            if not vals:
                return None
            return max(set(vals), key=vals.count)

        changes = []
        for field in (*_FINGERPRINT_FIELDS, "actual_hours"):
            a, b = mode(field, early), mode(field, late)
            if a is not None and b is not None and a != b:
                changes.append({"field": field, "early": a, "late": b})
        incomplete_rate_late = sum(1 for _, fp in late
                                   if fp.get("complete") is False or fp.get("missing_shards"))
        incomplete_pct = round(100.0 * incomplete_rate_late / len(late), 1)
        entry = {
            "snapshots": len(rows),
            "drift": bool(changes),
            "changes": changes,
            "incomplete_pct_recent": incomplete_pct,
            "actual_hours_early": mode("actual_hours", early),
            "actual_hours_late": mode("actual_hours", late),
        }
        if changes:
            drifted.append(model)
        out_models[model] = entry

    return {
        "fields": list(_FINGERPRINT_FIELDS) + ["actual_hours"],
        "models": out_models,
        "drifted": drifted,
        "n_models": len(out_models),
        "checked_snapshots": sum(len(v) for v in per_model.values()),
    }


# ------------------------------------------------------------------ P1-1 桥梁源 + 留一源
def _board_matrix(report: dict, board_key: str = "all") -> tuple[np.ndarray, list[str], list[str]] | None:
    """从已算好的分桶榜单重建 (m, b) 分数矩阵与其权重矩阵。"""
    meta = report.get("meta") or {}
    dw = (meta.get("difficulty_window") or {}).get(board_key) or {}
    buckets = dw.get("buckets") or []
    lbs = report.get("leaderboards") or {}
    if not buckets:
        return None
    models: list[str] = []
    for b in buckets:
        for row in lbs.get(b) or []:
            if row.get("model") and row["model"] not in models:
                models.append(row["model"])
    if not models:
        return None
    idx = {m: i for i, m in enumerate(models)}
    S = np.full((len(models), len(buckets)), np.nan)
    W = np.full((len(models), len(buckets)), np.nan)
    for j, b in enumerate(buckets):
        for row in lbs.get(b) or []:
            m = row.get("model")
            if m not in idx:
                continue
            s = row.get("score")
            if isinstance(s, (int, float)):
                S[idx[m], j] = float(s)
            for wkey in ("n_eff", "n"):
                w = row.get(wkey)
                if isinstance(w, (int, float)) and w and w > 0:
                    W[idx[m], j] = float(w)
                    break
    return S, models, buckets


def board_matrices(report: dict, board_key: str = "all"
                   ) -> tuple[np.ndarray, list[str], list[str]] | None:
    """从已算好的分桶榜单重建总榜的 (分数矩阵 S, 模型序, 桶序)。

    **为什么等权**：正式总榜的格子权重是 √(两维有效样本量的较小者)，但逐格
    n_eff 并未随报告导出——拿不到就无法忠实复现。而实测表明等权重建与正式总榜
    的 Spearman 达 0.99（``reconstruction_fidelity`` 会把这个数一并披露），
    说明正式总榜本身就落在等权路径上（``_board_cell_weights`` 在逐格 n_eff
    整体缺失时的安全网会退回等权）。用等权而非"拿 n 当 n_eff 凑一个权重矩阵"
    ——后者是伪造精度，会造出一条与正式榜不同却自称同源的基线。

    保真度由调用方显式披露：诊断是"在重建设计上做相对比较"，不是"重算正式榜"。
    """
    return _board_matrix(report, board_key)


def reconstruction_fidelity(report: dict, board_key: str = "all") -> dict[str, Any]:
    """重建设计与正式总榜的一致性（Spearman / 冠军是否同一家）。

    这个数字本身就是一项披露：如果它掉到 0.9 以下，说明诊断用的重建设计与
    正式榜已经不是同一张设计，走廊结论不该被引用。
    """
    built = board_matrices(report, board_key)
    official = {row.get("model"): row.get("score")
                for row in (report.get("leaderboards") or {}).get(board_key) or []}
    if built is None or not official:
        return {"available": False}
    S, models, buckets = built
    built_scores = _adjust(S, report, buckets)
    order = [(models[i], float(built_scores[i])) for i in range(len(models))
             if np.isfinite(built_scores[i])]
    order.sort(key=lambda x: -x[1])
    rank_b = {m: i for i, (m, _) in enumerate(order)}
    rank_o = {m: i for i, m in enumerate(sorted(
        [m for m in official if official[m] is not None],
        key=lambda m: -official[m]))}
    common = [m for m in rank_b if m in rank_o]
    rho = None
    if len(common) >= 3:
        rho = st._spearman(np.array([rank_o[m] for m in common], dtype=float),
                           np.array([rank_b[m] for m in common], dtype=float))
    return {
        "available": True,
        "spearman": None if rho is None else round(float(rho), 4),
        "n_compared": len(common),
        "champion_official": next(iter(rank_o), None),
        "champion_rebuilt": order[0][0] if order else None,
        "champion_same": bool(order) and order[0][0] == next(iter(rank_o), None),
    }


def _adjust(S: np.ndarray, report: dict, buckets: list[str],
            ridge: float | None = None, **over) -> np.ndarray:
    """在当前 meta 口径下跑一次双向劈分（等权），返回行分。"""
    meta = report.get("meta") or {}
    opts = dict(min_col=st.MIN_MODELS_PER_BUCKET,
                min_row=st.MIN_BUCKETS_PER_MODEL,
                min_col_frac=float(meta.get("board_min_col_frac") or 0.0),
                min_cell_weight=0.0,
                segment_sizes=_segment_sizes(buckets))
    opts.update(over)
    lam = float(meta.get("board_ridge") or 0.0) if ridge is None else float(ridge)
    return st.two_way_adjust(S, ridge=lam, **opts)["scores"]


def bridge_and_jackknife(report: dict, board_key: str = "all",
                         ridge_values: tuple[float, ...] = (0.0, 1.0)) -> dict[str, Any]:
    """桥梁源披露 + 留一源 jackknife + ridge 并列名次（审查 P1-1 / P1-2）。

    留一源的成本是 O(m) 次 ALS，m≈27、每次毫秒级——这是"完全跑得起"的诊断，
    此前不做只是因为没人做。它直接回答：这张榜是不是被一两家撑着。
    """
    built = board_matrices(report, board_key)
    if built is None:
        return {"available": False, "reason": "无法重建分数矩阵"}
    S, models, buckets = built
    meta = report.get("meta") or {}
    base_res = st.two_way_adjust(S, ridge=float(meta.get("board_ridge") or 0.0),
                                 min_col=st.MIN_MODELS_PER_BUCKET,
                                 min_row=st.MIN_BUCKETS_PER_MODEL,
                                 min_col_frac=float(meta.get("board_min_col_frac") or 0.0),
                                 min_cell_weight=0.0,
                                 segment_sizes=_segment_sizes(buckets))
    base_scores = base_res["scores"]
    row_keep = base_res["row_keep"]

    # ---- 桥梁源：在两个赛段（小时 / 日）都有格子的源
    seg = _segment_sizes(buckets)
    bridge: list[str] = []
    segment_coverage: dict[str, Any] = {}
    if seg and len(seg) == 2:
        a, b = seg
        V = np.isfinite(S)
        bridge = [models[i] for i in range(len(models))
                  if V[i, :a].any() and V[i, a:a + b].any()]
        start_col = 0
        for si, size in enumerate(seg):
            chunk = np.isfinite(S[:, start_col:start_col + size])
            segment_coverage[f"seg{si}"] = {
                "label": "小时轨" if si == 0 else "日轨",
                "buckets": buckets[start_col:start_col + size],
                "n_models": int(chunk.any(axis=1).sum()),
            }
            start_col += size

    # ---- 留一源 jackknife
    order_base = _rank_order(base_scores, models)
    jack: list[dict[str, Any]] = []
    for i, m in enumerate(models):
        if not row_keep[i]:
            continue
        keep = np.ones(len(models), dtype=bool)
        keep[i] = False
        alt = _adjust(S[keep], report, buckets)
        alt_scores = np.full(len(models), np.nan)
        alt_scores[keep] = alt
        common = np.isfinite(base_scores) & np.isfinite(alt_scores)
        if int(common.sum()) < 3:
            continue
        rho = st._spearman(base_scores[common], alt_scores[common])
        if rho is None:
            continue
        jack.append({"model": m, "spearman": round(float(rho), 4),
                     "top10_changed": _top_moved(order_base,
                                                 _rank_order(alt_scores, models))})
    jack.sort(key=lambda x: x["spearman"])

    # ---- ridge 并列名次（P1-2：把"已论证过的风险"从默认值里拿出来，变成可核对的数字）
    ridge_runs = []
    for lam in ridge_values:
        alt = _adjust(S, report, buckets, ridge=lam)
        common = np.isfinite(base_scores) & np.isfinite(alt)
        rho = st._spearman(base_scores[common], alt[common]) if int(common.sum()) >= 3 else None
        ridge_runs.append({
            "ridge": float(lam),
            "spearman_vs_default": None if rho is None else round(float(rho), 4),
            "top10_changed": 0 if rho is None else _top_moved(
                order_base, _rank_order(alt, models)),
            "champion": _champion(alt, models),
        })

    return {
        "available": True,
        "board": board_key,
        "n_models": int(row_keep.sum()),
        "n_buckets": len(buckets),
        "bridge_sources": bridge,
        "n_bridge": len(bridge),
        "segment_coverage": segment_coverage,
        "jackknife": {
            "worst": jack[0] if jack else None,
            "best": jack[-1] if jack else None,
            "mean": round(float(np.mean([j["spearman"] for j in jack])), 4) if jack else None,
            "n": len(jack),
            "all": jack,
        },
        "ridge_runs": ridge_runs,
        "default_ridge": float(meta.get("board_ridge") or 0.0),
        "fidelity": reconstruction_fidelity(report, board_key),
    }


def _segment_sizes(buckets: list[str]) -> tuple[int, ...] | None:
    hourly = sum(1 for b in buckets if b.startswith("hourly:"))
    daily = sum(1 for b in buckets if b.startswith("daily:"))
    if hourly and daily:
        return (hourly, daily)
    return None


def _rank_order(scores: np.ndarray, models: list[str] | None = None,
                top: int = 10) -> list[str]:
    order = [i for i in range(scores.size) if np.isfinite(scores[i])]
    order.sort(key=lambda i: -scores[i])
    if models is None:
        return [str(i) for i in order[:top]]
    return [models[i] for i in order[:top]]


def _top_moved(a: list[str], b: list[str]) -> int:
    return len(set(a[:10]) - set(b[:10]))


def _champion(scores: np.ndarray, models: list[str]) -> str | None:
    finite = [i for i in range(scores.size) if np.isfinite(scores[i])]
    if not finite:
        return None
    return models[max(finite, key=lambda i: scores[i])]


# ================================================================== P0-2 口径走廊
def corridor_sensitivity(report: dict, board_key: str = "all",
                         neff_gates: tuple[float, ...] = (20.0, 60.0, 100.0, 150.0),
                         ridges: tuple[float, ...] = (0.0, 1.0),
                         col_fracs: tuple[float, ...] = (0.0, 0.5, 0.7)) -> dict[str, Any]:
    """名次对**口径参数**的敏感度（审查 P0-2）。

    权重敏感性已经做了（``meta.weight_sensitivity``），但权重只改变"各项占多少"；
    门槛类参数改变的是"**哪些样本算数**"——README 自己实测过降水阈值能把 ETS
    从 0.054 抬到 0.25，量级远大于 ±40% 的权重扰动，而这一类从未进过敏感性分析。

    这里对能低成本重算的三项逐个重跑总榜：n_eff 入围门槛、ridge 收缩、长尾桶
    门槛。降水阈值 / ``daily_min_hours`` / 逐格权重门槛需要重跑全量评估或导出
    逐格 n_eff，由 ``scripts/sensitivity_corridors.py`` 离线产出——本函数不假装能算。
    """
    built = board_matrices(report, board_key)
    if built is None:
        return {"available": False, "reason": "无法重建分数矩阵"}
    S, models, buckets = built
    meta = report.get("meta") or {}
    lbs = report.get("leaderboards") or {}
    default_gate = float(meta.get("min_board_neff") or 30.0)
    default_frac = float(meta.get("board_min_col_frac") or 0.0)
    default_ridge = float(meta.get("board_ridge") or 0.0)

    # 基线取**重建设计上**的行分，而不是正式总榜的分。
    # 为什么：走廊要回答的是"改这一个旋钮，名次动不动"——同一条设计上前后对比
    # 才有意义。拿正式榜当基线会把"重建与正式之间那 0.7% 的系统性偏差"混进每个
    # 走廊，于是每个走廊都报告"冠军换了"——那是尺子的偏差，不是旋钮的效应。
    # 重建与正式的一致性另由 fidelity 单独披露（当前 0.993）。
    base_scores = _adjust(S, report, buckets)
    base_order = _rank_order(base_scores, models)
    base_champion = _champion(base_scores, models)
    corridors: list[dict[str, Any]] = []

    # ① n_eff 入围门槛：n_eff 不足的源整行剔出设计后重算
    for gate in neff_gates:
        if abs(gate - default_gate) < 1e-9:
            continue
        S2 = S.copy()
        board_rows = {row.get("model"): row.get("n_eff")
                      for row in (report.get("leaderboards") or {}).get(board_key) or []}
        for i, m in enumerate(models):
            row = _row_neff(lbs, m, buckets, board_rows)
            if row is None or row < gate:
                S2[i, :] = np.nan
        alt = _adjust(S2, report, buckets)
        corridors.append(_compare("min_board_neff", f"{gate:g}", default_gate,
                                  base_scores, base_order, base_champion, alt, models))
    # ② ridge 收缩（P1-2）
    for lam in ridges:
        if abs(lam - default_ridge) < 1e-9:
            continue
        alt = _adjust(S, report, buckets, ridge=lam)
        corridors.append(_compare("board_ridge", f"{lam:g}", default_ridge,
                                  base_scores, base_order, base_champion, alt, models))
    # ③ 长尾桶门槛
    for frac in col_fracs:
        if abs(frac - default_frac) < 1e-9:
            continue
        alt = _adjust(S, report, buckets, min_col_frac=frac)
        corridors.append(_compare("board_min_col_frac", f"{frac:g}", default_frac,
                                  base_scores, base_order, base_champion, alt, models))

    rhos = [c["spearman"] for c in corridors if c["spearman"] is not None]
    return {
        "available": True,
        "board": board_key,
        "corridors": corridors,
        "min_spearman": min(rhos) if rhos else None,
        "max_top10_changed": max((c["top10_changed"] for c in corridors), default=0),
        "champion_rotations": sum(1 for c in corridors if c["champion_changed"]),
        "n_corridors": len(corridors),
        "fidelity": reconstruction_fidelity(report, board_key),
        "note": "降水阈值 / daily_min_hours / 逐格权重门槛走廊需重跑全量评估或导出逐格 n_eff，由 scripts/sensitivity_corridors.py 离线产出",
    }


def _row_neff(lbs: dict, model: str, buckets: list[str],
              board_rows: dict | None = None) -> float | None:
    """某源在总榜上的 n_eff——**必须取总榜行自己的 n_eff**，不能拿逐桶的 n。

    门槛 ``min_board_neff`` 判的是"这个源在整个总榜上攒够样本了没有"，口径就是
    总榜行的 n_eff。用逐桶的名义样本量 n（动辄 2368）会让任何门槛都剔不掉人，
    走廊于是永远给出 ρ=1.0 的"名次纹丝不动"——一个看起来很稳、其实是口径错了的
    假结论。
    """
    if board_rows and model in board_rows:
        w = board_rows[model]
        if isinstance(w, (int, float)) and w > 0:
            return float(w)
    vals = []
    for b in buckets:
        for row in lbs.get(b) or []:
            if row.get("model") == model:
                w = row.get("n_eff")
                if isinstance(w, (int, float)) and w > 0:
                    vals.append(float(w))
    return max(vals) if vals else None


def _compare(name: str, value: str, default: float, base_scores: np.ndarray,
             base_order: list[str], base_champion: str | None,
             alt: np.ndarray, models: list[str]) -> dict[str, Any]:
    common = np.isfinite(base_scores) & np.isfinite(alt)
    rho = st._spearman(base_scores[common], alt[common]) if int(common.sum()) >= 3 else None
    alt_order = _rank_order(alt, models)
    champ = _champion(alt, models)
    return {
        "param": name,
        "value": value,
        "default": default,
        "spearman": None if rho is None else round(float(rho), 3),
        "top10_changed": _top_moved(base_order, alt_order),
        "champion": champ,
        "champion_changed": champ != base_champion,
        "n_ranked": int(np.isfinite(alt).sum()),
    }


# ------------------------------------------------------------------ 汇总入口
def compute_all(report: dict, data_root: Path, months: list[str] | None = None,
                families: dict[str, str] | None = None,
                with_fingerprint: bool = True) -> dict[str, Any]:
    """一次性算出全部诊断项，返回可直接并进 ``meta`` 的 dict。"""
    meta = report.get("meta") or {}
    months = months or [meta.get("period_label") or ""]
    months = [m for m in months if m]
    obs_root = Path(data_root) / "obs"
    fc_root = Path(data_root) / "forecasts"
    out: dict[str, Any] = {"generated_from": "data/ 冻结快照 + 已算好的分桶榜单"}

    obs_by_station = {}
    if obs_root.exists():
        for st_dir in sorted(obs_root.iterdir()):
            if st_dir.is_dir():
                obs_by_station[st_dir.name] = load_observations(st_dir, months)

    if fc_root.exists() and obs_by_station:
        start = (meta.get("start") or "").replace(" ", "T")
        end = (meta.get("end") or "").replace(" ", "T")
        temp_err, rain_err = iter_error_samples(
            fc_root, obs_by_station, months, start=start or None, end=end or None)
        out["source_correlation"] = cross_source_correlation(temp_err, families=families)
        out["station_rho"] = station_rho(temp_err)
        out["_debug"] = {"models_with_series": len(temp_err),
                         "samples": sum(len(s) for s in temp_err.values())}
        if with_fingerprint:
            out["fingerprint"] = fingerprint_drift(fc_root, months)
    else:
        out["source_correlation"] = {"available": False, "reason": "未找到 data/forecasts"}
        out["station_rho"] = {"available": False, "reason": "未找到 data/forecasts"}

    out["bridge"] = bridge_and_jackknife(report)
    out["corridor"] = corridor_sensitivity(report)
    return out

"""λ（评分换算斜率）敏感性走廊——离线审计。

**为什么单独跑一次**：这是 `docs/score_slopes.md` §七 与 `README.md`
「本轮审查留下的、尚未解决的已知缺口」里同时挂着的那一条——冠军频率敏感性只
扰动**子分权重**与温度:降水宏观权重（`stats.weight_champion_distribution`），
扰不到换算斜率 λ 本身；`scripts/sensitivity_corridors.py` 覆盖的是降水阈值与
`daily_min_hours`。λ 是"按子分桶内跨源 sd = 8 分"反解并冻结的**标准量**
（`config.eval.score_slopes`），如果名次对 λ 呈刀锋状敏感，"标定"就只是一层
包装；本脚本回答：λ 整体缩放 / 逐项抖动 / 随机联合扰动时，名次还剩下多少。

**口径纪律**（与 `scripts/sensitivity_corridors.py` 同）：
  * B/C 段一次只动一个旋钮（均匀缩放 = 一个旋钮，联合扰动段显式声明）；
  * `cfg.eval` **深拷贝**后再改（`config.py` P1-3 的警告：就地改
    `score_slopes[k]` 会永久污染 `DEFAULT_EVAL` 与 `lru_cache` 的 Config）；
  * λ 只经 `build_report` 的 `eval_cfg` 生效（`evaluate.apply_score_slopes`
    在入口装载，换算函数调用时读模块级 `SCORE_SLOPES`）；
  * E 段的权重表 `evaluate.TEMP_SCORE_PARTS` / `PRECIP_SCORE_PARTS` 是模块级
    元组——只在**本进程内**重绑、`finally` 还原，不落盘、不改文件。这已是
    "只扰动权重"唯一可用的缝；宏观温度:降水分（`track_cells` 里的 `_mean2`）
    与"同轨两量各半均值"共用同一个辅助函数，**没有干净的缝**，故不做（见
    输出 JSON 的 notes）。
  * 只读：本脚本不写 `config/`、`src/`、`tests/`、`docs/`、`reports/`，
    产物落在 `.work/`。

**窗口**：默认为已发布主报告 `reports/index.html` 的对账窗口（读内联
`meta.start` / `meta.end`），保证走廊与读者看到的榜单**同轴**、可逐家对拍；
可用 `--start/--end` 或 `--month` 覆盖。`--month` 走自然月，与跨月累计的
主页总榜**不同轴**，此时只能自己与自己的基线比。

**分段**（总 ≈ 25 次全量 `build_report`）：
    A 基线（1）
    B 全部 λ 均匀 × {0.75, 1.5}（2；1.0 = A）
    C 离散度贡献最大的 6 项逐项 × {0.5, 2.0}（12）
    D λ 联合随机扰动 U(0.7, 1.4)，seed 42，6 抽（6）
    E λ × 权重联合扰动（λ 同上；15 项名义权重 × U(0.6, 1.4) 后按族归一）（4）

用法：
    .venv312/bin/python scripts/audit_lambda_corridor.py                  # 全走廊
    .venv312/bin/python scripts/audit_lambda_corridor.py --passes baseline
    .venv312/bin/python scripts/audit_lambda_corridor.py --jobs 3 --resume
    .venv312/bin/python scripts/audit_lambda_corridor.py --month 2026-10
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import random
import statistics
import sys
import time
from multiprocessing import get_context
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from weather_eval.config import load_config  # noqa: E402
from weather_eval.stats import _spearman  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("lambda-corridor")

WORK = ROOT / ".work"
PASS_CACHE = WORK / "lambda_passes"

# 子进程（fork）继承的窗口与配置——避免每个 pass 重新装载
_G_CFG = None
_G_START = None
_G_END = None
_G_LABEL = None

# render.py:258-287 的 lb 行数组下标（键名只出现一次、模型只传索引）：
# [rank_idx, score, temp, precip, acc2, ts, acc1, rmse, mae, mbe, r, slope,
#  mbe_bdisp, accr, ets, pod, far, bias, amt_mae, amt_bias, grade_ets]
_LB_COL: dict[int, str] = {
    4: "acc2", 6: "acc1", 7: "rmse", 8: "mae", 9: "mbe", 10: "r", 11: "slope",
    12: "mbe_bdisp", 14: "ets", 15: "pod", 16: "far", 17: "bias",
    18: "amt_mae", 19: "amt_bias", 20: "grade_ets",
}
_LB_WIDTH = 21


# ----------------------------------------------------------------- 工具
def _read_published_report() -> dict | None:
    """读 `reports/index.html` 内联的 `const report = {...}`（只读，不写）。"""
    p = ROOT / "reports" / "index.html"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8")
    i = txt.find("const report = {")
    if i < 0:
        return None
    j = txt.find("{", i)
    depth = 0
    instr = esc = False
    k = j
    while k < len(txt):
        ch = txt[k]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        else:
            if ch == '"':
                instr = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
        k += 1
    try:
        return json.loads(txt[j:k + 1])
    except json.JSONDecodeError:
        return None


def _published_board_order(rep: dict | None) -> list[str]:
    if not rep:
        return []
    return [r["m"] for r in (rep.get("board") or []) if r.get("s") is not None]


def _lb_dispersion(rep: dict | None, slopes: dict) -> dict[str, dict]:
    """从内联分时效榜估算"名义权重 × λ × 跨源离散度"（= 影响力，未取份额）。

    逐榜（hourly/daily/all × 提前天数）求各指标跨源 sd（ddof=1，≥3 家），
    再对榜取算术平均。离散度用的是**原始指标值**，λ 是斜率——两者相乘即
    `docs/score_slopes.md` 定义的影响力公式（未裁剪前的近似）。
    """
    from weather_eval.evaluate import TEMP_SCORE_PARTS, PRECIP_SCORE_PARTS

    nom = {k: float(w) for k, w, *_ in list(TEMP_SCORE_PARTS) + list(PRECIP_SCORE_PARTS)}
    per_board: dict[str, list[float]] = {k: [] for k in _LB_COL.values()}
    if rep:
        for track in (rep.get("lb") or {}).values():
            for rows in track.values():
                for idx, key in _LB_COL.items():
                    vals = [r[idx] for r in rows if len(r) == _LB_WIDTH and r[idx] is not None]
                    if len(vals) < 3:
                        continue
                    per_board[key].append(float(np.std(np.asarray(vals, dtype=float), ddof=1)))
    out = {}
    for key, sds in per_board.items():
        if not sds:
            continue
        sd = statistics.fmean(sds)
        out[key] = {"sd_mean": round(sd, 4), "n_boards": len(sds),
                    "nominal_weight": nom.get(key),
                    "lambda": slopes.get(key),
                    "influence": round((nom.get(key) or 0.0) * (slopes.get(key) or 0.0) * sd, 4)}
    return out


def _slope_influence_ranking(slope_influence: dict | None) -> list[dict]:
    """把 `meta.diagnostics.slope_influence` 折算成跨轨可比的贡献（w × swing）。

    该诊断本就把摆幅 swing 定义在**已换算、已裁剪**的子分上（桶内 ≥3 家、
    逐桶中心化后池化的 sd），正是项目自己的"实测影响力"口径；逐轨（hourly /
    daily）、逐维（temp / precip）分别报，这里对同一指标的可用轨取平均。
    """
    if not slope_influence or not slope_influence.get("available"):
        return []
    acc: dict[str, list[float]] = {}
    shares: dict[str, list[float]] = {}
    labels: dict[str, str] = {}
    for _tname, dims in (slope_influence.get("tracks") or {}).items():
        for dim in ("temp", "precip"):
            for row in (dims or {}).get(dim) or []:
                key = row.get("key")
                w, sw = row.get("nominal_weight"), row.get("swing")
                if key is None or w is None or sw is None:
                    continue
                acc.setdefault(key, []).append(float(w) * float(sw))
                if row.get("effective_share") is not None:
                    shares.setdefault(key, []).append(float(row["effective_share"]))
                labels[key] = row.get("label") or key
    out = []
    for k, v in acc.items():
        row = {"key": k, "contribution": round(statistics.fmean(v), 4), "n_tracks": len(v),
               "label": labels.get(k, k)}
        if shares.get(k):
            row["mean_effective_share"] = round(statistics.fmean(shares[k]), 3)
        out.append(row)
    return sorted(out, key=lambda r: -r["contribution"])


def _reweight(parts: tuple, weights: dict) -> tuple:
    """把 (key, w, label, 说明, 换算函数) 表里的 w 换成 weights[key]（其余字段原样）。"""
    return tuple((row[0], float(weights[row[0]]), *row[2:]) if row[0] in weights else row
                 for row in parts)


# ----------------------------------------------------------------- 单次 pass
def _run_pass(spec: dict) -> dict:
    import weather_eval.evaluate as ev_mod
    from weather_eval.evaluate import build_report

    cfg = _G_CFG
    ev = copy.deepcopy(dict(cfg.eval))          # 深拷贝：绝不动 cfg.eval 本体
    ev["score_slopes"] = dict(spec["slopes"])
    saved = None
    if spec.get("weights"):
        saved = (ev_mod.TEMP_SCORE_PARTS, ev_mod.PRECIP_SCORE_PARTS)
        ev_mod.TEMP_SCORE_PARTS = _reweight(saved[0], spec["weights"])
        ev_mod.PRECIP_SCORE_PARTS = _reweight(saved[1], spec["weights"])
    t0 = time.time()
    try:
        rep = build_report(cfg.station_ids, cfg.models, ev, _G_START, _G_END,
                           period_label=_G_LABEL)
    finally:
        if saved is not None:                   # 还原模块级表（同进程内不留痕）
            ev_mod.TEMP_SCORE_PARTS, ev_mod.PRECIP_SCORE_PARTS = saved
    elapsed = time.time() - t0

    rows = (rep.get("leaderboards") or {}).get("all") or []
    out = {
        "order": [r["model"] for r in rows if r.get("score") is not None],
        "scores": {r["model"]: r.get("score") for r in rows},
        "qualified": {r["model"]: bool(r.get("qualified", True)) for r in rows},
        "elapsed_s": round(elapsed, 1),
    }
    if spec.get("want_diagnostics"):
        diag = (rep.get("meta") or {}).get("diagnostics") or {}
        out["slope_influence"] = diag.get("slope_influence")
    return {**{k: spec.get(k) for k in
               ("pid", "kind", "metric", "factor", "seed", "note", "multipliers",
                "weights")}, **out}


# ----------------------------------------------------------------- 走廊定义
def _build_passes(base_slopes: dict, chosen: list[str], families: dict,
                  weights_nominal: dict) -> list[dict]:
    keys = list(base_slopes)
    passes: list[dict] = [{
        "pid": "A-baseline", "kind": "baseline", "metric": None, "factor": 1.0,
        "seed": None, "note": "冻结 λ 原值", "slopes": dict(base_slopes),
        "want_diagnostics": True,
    }]

    # B 均匀缩放：子分是 λ 的仿射函数，理应近似保秩——不保秩 = 裁剪/饱和在
    # 不同源上以不同速率生效，是一个需要解释的发现（不是"扰动可接受"）。
    for f in (0.75, 1.5):
        passes.append({
            "pid": f"B-uniform-x{f}", "kind": "uniform", "metric": None, "factor": f,
            "seed": None, "note": "全部 λ 同乘一个因子",
            "slopes": {k: v * f for k, v in base_slopes.items()},
        })

    # C 逐项抖动：只挑"离散度贡献最大"的 6 项（贡献由基线诊断决定，见 main）。
    for m in chosen:
        for f in (0.5, 2.0):
            s = dict(base_slopes)
            s[m] = base_slopes[m] * f
            passes.append({
                "pid": f"C-single-{m}-x{f}", "kind": "single", "metric": m,
                "factor": f, "seed": None, "note": "仅该项 λ 乘因子，其余冻结",
                "slopes": s,
            })

    # D λ 联合随机扰动：固定 seed 42，独立同分布
    rng = random.Random(42)
    for i in range(6):
        mult = {k: rng.uniform(0.7, 1.4) for k in keys}
        passes.append({
            "pid": f"D-random-{i + 1}", "kind": "random", "metric": None, "factor": None,
            "seed": 42, "note": "每项 λ × U(0.7,1.4) 独立抽样",
            "multipliers": {k: round(v, 4) for k, v in mult.items()},
            "slopes": {k: base_slopes[k] * mult[k] for k in keys},
        })

    # E λ × 权重联合扰动：权重表是模块级元组，进程内重绑 + finally 还原
    rng = random.Random(42)
    for i in range(4):
        mult = {k: rng.uniform(0.7, 1.4) for k in keys}
        wmult = {k: rng.uniform(0.6, 1.4) for k in keys}
        weights = {}
        for fam_keys in families.values():
            tot = sum(weights_nominal[k] * wmult.get(k, 1.0) for k in fam_keys)
            for k in fam_keys:
                weights[k] = weights_nominal[k] * wmult.get(k, 1.0) / tot
        passes.append({
            "pid": f"E-joint-weight-{i + 1}", "kind": "joint_weight", "metric": None,
            "factor": None, "seed": 42,
            "note": "λ × U(0.7,1.4) 与 15 项名义权重 × U(0.6,1.4)（按族归一）联合扰动",
            "multipliers": {k: round(v, 4) for k, v in mult.items()},
            "weights": {k: round(v, 6) for k, v in weights.items()},
            "slopes": {k: base_slopes[k] * mult[k] for k in keys},
        })
    return passes


# ----------------------------------------------------------------- 比较
def _compare(base_order: list[str], base_scores: dict, res: dict) -> dict:
    idx = {m: i for i, m in enumerate(base_order)}
    order = res["order"]
    common = [m for m in order if m in idx]
    rho = None
    if len(common) >= 3:
        rho = _spearman(np.asarray([idx[m] for m in common], dtype=float),
                        np.asarray([order.index(m) for m in common], dtype=float))
    top10_changed = len(set(base_order[:10]) - set(order[:10]))
    top3_changed = len(set(base_order[:3]) - set(order[:3]))
    shifts = [abs(base_scores[m] - res["scores"][m]) for m in common
              if base_scores.get(m) is not None and res["scores"].get(m) is not None]
    shifts_top10 = [abs(base_scores[m] - res["scores"][m]) for m in common
                    if m in set(base_order[:10])]
    champ = order[0] if order else None
    in_order = set(order)
    common_top10 = [m for m in base_order[:10] if m in in_order]
    return {
        "spearman": None if rho is None else round(float(rho), 4),
        "spearman_top10": None if len(common_top10) < 3 else round(float(_spearman(
            np.asarray([idx[m] for m in common_top10], dtype=float),
            np.asarray([order.index(m) for m in common_top10], dtype=float))), 4),
        "top10_changed": top10_changed,
        "top3_changed": top3_changed,
        "champion": champ,
        "champion_changed": champ != base_order[0],
        "champion_base": base_order[0],
        "n_ranked": len(order),
        "n_common": len(common),
        "max_score_shift": None if not shifts else round(max(shifts), 2),
        "max_score_shift_top10": None if not shifts_top10 else round(max(shifts_top10), 2),
        "mean_abs_rank_shift": None if not common else round(
            statistics.fmean(abs(idx[m] - order.index(m)) for m in common), 2),
    }


# ----------------------------------------------------------------- 主流程
def main() -> int:
    ap = argparse.ArgumentParser(description="λ 敏感性走廊（离线，只读生产数据）")
    ap.add_argument("--month", default=None, help="YYYY-MM；自然月窗口（与主页总榜不同轴）")
    ap.add_argument("--start", default=None, help="ISO，如 2026-08-01T00:00")
    ap.add_argument("--end", default=None, help="ISO，如 2026-10-18T13:00")
    ap.add_argument("--config", default=None)
    ap.add_argument("--jobs", type=int, default=3, help="并行进程数（默认 3）")
    ap.add_argument("--resume", action="store_true", help="复用 .work/lambda_passes 里已完成的 pass")
    ap.add_argument("--passes", default="all",
                    help="all | baseline | A,B,C,D,E 的逗号组合")
    ap.add_argument("--out", default=str(WORK / "lambda_corridor.json"))
    ap.add_argument("--md", default=str(WORK / "lambda_corridor.md"))
    args = ap.parse_args()

    global _G_CFG, _G_START, _G_END, _G_LABEL
    cfg = load_config(args.config)
    from weather_eval.timeutil import parse_iso

    published = _read_published_report()
    board_start = board_end = None
    if published:
        meta = published.get("meta") or {}
        if meta.get("start") and meta.get("end"):
            board_start = parse_iso(meta["start"].replace(" ", "T"))
            board_end = parse_iso(meta["end"].replace(" ", "T"))

    if args.start and args.end:
        start, end = parse_iso(args.start), parse_iso(args.end)
        label = f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d}"
        window_src = "--start/--end"
    elif args.month:
        from weather_eval.__main__ import _month_window
        start, end = _month_window(args.month)
        label = args.month
        window_src = "--month（自然月）"
    elif board_start and board_end:
        start, end = board_start, board_end
        label = f"{start:%Y-%m-%d} ~ {end:%Y-%m-%d}"
        window_src = "reports/index.html meta（与已发布总榜同轴）"
    else:
        from weather_eval.__main__ import _live_window
        start, end, label = _live_window(cfg)
        window_src = "_live_window(cfg)（退回实时窗口）"

    same_as_board = bool(board_start and board_end and start == board_start and end == board_end)
    log.info("窗口：%s → %s（%s；与已发布总榜同轴：%s）", start, end, window_src, same_as_board)

    _G_CFG, _G_START, _G_END, _G_LABEL = cfg, start, end, label

    base_slopes = dict(cfg.eval.get("score_slopes") or {})
    if len(base_slopes) < 15:
        log.warning("config 的 score_slopes 只有 %d 项（预期 15）——按实际项数跑",
                    len(base_slopes))
    from weather_eval.evaluate import TEMP_SCORE_PARTS, PRECIP_SCORE_PARTS
    families = {
        "temp": [k for k, *_ in TEMP_SCORE_PARTS],
        "precip": [k for k, *_ in PRECIP_SCORE_PARTS],
    }
    weights_nominal = {k: float(w) for k, w, *_ in list(TEMP_SCORE_PARTS) + list(PRECIP_SCORE_PARTS)}
    missing = [k for k in base_slopes if k not in weights_nominal]
    if missing:
        log.warning("λ 里有评分表外的键（将照常扰动，但无名义权重）：%s", missing)

    PASS_CACHE.mkdir(parents=True, exist_ok=True)
    wanted = {"A": "all", "B": "all", "C": "all", "D": "all", "E": "all"}
    sel = args.passes.strip().lower()
    if sel not in ("all",):
        want = {"A"} if sel == "baseline" else set(
            s.strip().upper() for s in sel.split(",") if s.strip())
        wanted = {k: (k if k in want else "skip") for k in wanted}
    run_kinds = {k for k, v in wanted.items() if v != "skip"}

    results: dict[str, dict] = {}
    # A 段无条件跑（或复用缓存）：它是所有走廊点的比较基准，不能缺
    phase1 = [{"pid": "A-baseline", "kind": "baseline", "metric": None, "factor": 1.0,
               "seed": None, "note": "冻结 λ 原值", "slopes": dict(base_slopes),
               "want_diagnostics": True}]

    def _cache_path(pid: str) -> Path:
        return PASS_CACHE / f"{pid}.json"

    def _run_phase(specs: list[dict], jobs: int) -> None:
        todo = []
        for spec in specs:
            cp = _cache_path(spec["pid"])
            if args.resume and cp.exists():
                results[spec["pid"]] = json.loads(cp.read_text(encoding="utf-8"))
                log.info("复用缓存 %s", spec["pid"])
            else:
                todo.append(spec)
        if not todo:
            return
        if jobs <= 1:
            for spec in todo:
                res = _run_pass(spec)
                results[res["pid"]] = res
                _cache_path(res["pid"]).write_text(
                    json.dumps(res, ensure_ascii=False), encoding="utf-8")
                log.info("完成 %s（%.0fs）", res["pid"], res["elapsed_s"])
            return
        ctx = get_context("fork")
        with ctx.Pool(processes=jobs) as pool:
            for res in pool.imap_unordered(_run_pass, todo):
                results[res["pid"]] = res
                _cache_path(res["pid"]).write_text(
                    json.dumps(res, ensure_ascii=False), encoding="utf-8")
                log.info("完成 %s（%.0fs，共 %d/%d）", res["pid"], res["elapsed_s"],
                         len(results), len(specs))

    t_all = time.time()
    _run_phase(phase1, 1)                      # A 先单独跑：C 的选点依赖它
    if not results:
        log.error("基线 pass 缺失（--passes 只选了其它段？A 必须完成）")
        return 2
    base = results["A-baseline"]
    base_order = base["order"]

    # C 的选点：以基线诊断（实测影响力 w×swing）为主，内联榜估算为交叉核验
    infl = _slope_influence_ranking(base.get("slope_influence"))
    lb_infl = _lb_dispersion(published, base_slopes)
    if infl:
        ranked = [r["key"] for r in infl]
        method = ("meta.diagnostics.slope_influence（项目自己的实测影响力口径："
                  "名义权重 × 桶内中心化的换算子分 sd，跨轨平均）")
    else:
        ranked = [k for k, _ in sorted(lb_infl.items(), key=lambda kv: -kv[1]["influence"])]
        method = ("内联分时效榜估算（名义权重 × λ × 原始指标跨源 sd，逐榜平均）"
                  "——基线诊断不可用时的降级路径")
    chosen = ranked[:6]
    log.info("逐项抖动选点（%s）：%s", "诊断" if infl else "内联估算", ", ".join(chosen))

    specs = _build_passes(base_slopes, chosen, families, weights_nominal)
    keep = {"uniform": "B", "single": "C", "random": "D", "joint_weight": "E"}
    specs = [s for s in specs if s["kind"] == "baseline" or keep[s["kind"]] in run_kinds]
    _run_phase([s for s in specs if s["pid"] != "A-baseline"], args.jobs)

    # ---------------- 汇总 ----------------
    corridor = []
    for spec in specs:
        if spec["pid"] == "A-baseline":
            continue
        res = results.get(spec["pid"])
        if not res:
            log.warning("缺少结果：%s", spec["pid"])
            continue
        cmp = _compare(base_order, base["scores"], res)
        corridor.append({
            "pid": spec["pid"],
            "kind": spec["kind"],
            "metric_or_null": spec["metric"],
            "factor": spec["factor"],
            "spearman": cmp["spearman"],
            **{k: cmp[k] for k in ("top10_changed", "champion", "champion_changed",
                                   "n_ranked", "max_score_shift")},
            "spearman_top10": cmp["spearman_top10"],
            "top3_changed": cmp["top3_changed"],
            "max_score_shift_top10": cmp["max_score_shift_top10"],
            "mean_abs_rank_shift": cmp["mean_abs_rank_shift"],
            "n_common": cmp["n_common"],
            "seed": spec.get("seed"),
            "multipliers": spec.get("multipliers"),
            "weights": spec.get("weights"),
            "note": spec.get("note"),
            "elapsed_s": res.get("elapsed_s"),
        })

    rhos = [c["spearman"] for c in corridor if c["spearman"] is not None]
    pub_order = _published_board_order(published)
    mism = [{"rank": i + 1, "published": m, "computed": base_order[i] if i < len(base_order) else None}
            for i, m in enumerate(pub_order) if i >= len(base_order) or base_order[i] != m]

    out = {
        "task": "lambda_corridor",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window": {
            "start": start.strftime("%Y-%m-%dT%H:%M"),
            "end": end.strftime("%Y-%m-%dT%H:%M"),
            "period_label": label,
            "source": window_src,
            "published_board_window": None if not board_start else {
                "start": board_start.strftime("%Y-%m-%dT%H:%M"),
                "end": board_end.strftime("%Y-%m-%dT%H:%M")},
            "same_as_published_board": same_as_board,
        },
        "lambda_keys": list(base_slopes),
        "lambda_values": {k: float(v) for k, v in base_slopes.items()},
        "baseline": {
            "champion": base_order[0] if base_order else None,
            "order": base_order,
            "scores": base["scores"],
            "qualified": base.get("qualified"),
            "n_ranked": len(base_order),
            "elapsed_s": base.get("elapsed_s"),
            "published_order": pub_order,
            "matches_published_order": not mism,
            "published_mismatches": mism,
        },
        "selected_metrics": {
            "method": method,
            "ranking": infl or None,
            "lb_based_ranking": [
                {"key": k, **v} for k, v in sorted(lb_infl.items(), key=lambda kv: -kv[1]["influence"])],
            "chosen": chosen,
        },
        "corridor": corridor,
        "summary": {
            "n_points": len(corridor),
            "min_spearman": min(rhos) if rhos else None,
            "median_spearman": round(statistics.median(rhos), 4) if rhos else None,
            "max_spearman": max(rhos) if rhos else None,
            "champion_rotations": sum(1 for c in corridor if c["champion_changed"]),
            "n_distinct_champions": len({c["champion"] for c in corridor if c["champion"]}),
            "top10_fully_stable_points": sum(1 for c in corridor if c["top10_changed"] == 0),
            "wall_clock_s": round(time.time() - t_all, 1),
        },
        "notes": [
            "λ 只经 build_report 的 eval_cfg 生效（evaluate.apply_score_slopes 在入口装载）；"
            "每次都深拷贝 cfg.eval 后再改，绝不触碰 cfg.eval 本体。",
            "E 段在子进程内重绑 evaluate.TEMP_SCORE_PARTS / PRECIP_SCORE_PARTS 并在 finally "
            "还原；不落盘、不改文件。",
            "宏观温度:降水分（track_cells 里的 _mean2 各半合成）与日轨两量均值共用同一个"
            "辅助函数，改它会同时改掉无关的合成（小时:日轨合成、显示均值）——没有干净的缝，"
            "故本走廊不扰动宏观权重（该维度已由 stats.weight_champion_distribution 的 "
            "U(30%,70%) 冠军频率覆盖）。",
            "综合分技能指数缩放（score_index_gain，build_report 入口装载）会同比放大"
            " max_score_shift（显示层）；本走廊的名次类指标不受增益影响。",
        ],
    }

    dst = Path(args.out)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("写出：%s", dst)

    _write_md(out, Path(args.md))
    log.info("写出：%s", args.md)
    log.info("点数 %d，最低 Spearman %s，中位 %s，冠军轮换 %d 次，用时 %.0fs",
             out["summary"]["n_points"], out["summary"]["min_spearman"],
             out["summary"]["median_spearman"], out["summary"]["champion_rotations"],
             out["summary"]["wall_clock_s"])
    return 0


def _fmt(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _write_md(out: dict, dst: Path) -> None:
    w = out["window"]
    base = out["baseline"]
    rows = sorted(out["corridor"], key=lambda c: -(abs(1 - (c["spearman"] or 1.0))))
    lines: list[str] = []
    lines.append("# λ（评分换算斜率）敏感性走廊")
    lines.append("")
    lines.append(f"- 生成：{out['generated_at']}　用时 {out['summary']['wall_clock_s']}s"
                 f"（{out['summary']['n_points']} 个走廊点，每点一次全量 build_report）")
    lines.append(f"- 窗口：**{w['start']} → {w['end']}**（{w['source']}）；"
                 f"与已发布总榜同轴：**{w['same_as_published_board']}**")
    lines.append(f"- 基线冠军：**{base['champion']}**（{base['n_ranked']} 家入榜）；"
                 f"与 `reports/index.html` 名次逐家一致：**{base['matches_published_order']}**")
    lines.append(f"- 逐项抖动选点依据：{out['selected_metrics']['method']}")
    lines.append(f"- 选中的 6 项：{', '.join(out['selected_metrics']['chosen'])}")
    lines.append("")
    lines.append("## 走廊表（按 |1 − Spearman| 降序）")
    lines.append("")
    lines.append("| 段 | 扰动 | 指标 | 因子 | Spearman | 前10 Spearman | 前十换 | 前三换 "
                 "| 平均位移 | 冠军 | 换冠军 | max\\|Δ分\\| |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    kind_zh = {"baseline": "A 基线", "uniform": "B 均匀", "single": "C 逐项",
               "random": "D 联合随机", "joint_weight": "E λ×权重"}
    for c in rows:
        lines.append(
            f"| {kind_zh.get(c['kind'], c['kind'])} | {c['pid'].split('-', 1)[1]} | "
            f"{c['metric_or_null'] or '全部'} | {_fmt(c['factor'], 2)} | "
            f"{_fmt(c['spearman'])} | {_fmt(c['spearman_top10'])} | "
            f"{c['top10_changed']} | {c['top3_changed']} | "
            f"{_fmt(c['mean_abs_rank_shift'], 2)} | {c['champion']}"
            f"{' ⚠️' if c['champion_changed'] else ''} | "
            f"{'是' if c['champion_changed'] else '否'} | {_fmt(c['max_score_shift'], 2)} |")
    lines.append("")
    s = out["summary"]
    lines.append("## 汇总")
    lines.append("")
    lines.append(f"- Spearman：最低 **{_fmt(s['min_spearman'])}**、中位 **{_fmt(s['median_spearman'])}**、"
                 f"最高 {_fmt(s['max_spearman'])}")
    lines.append(f"- 冠军轮换：**{s['champion_rotations']} / {s['n_points']}** 个点换人"
                 f"（出现过的冠军 {s['n_distinct_champions']} 家）")
    lines.append(f"- 前十完全不变的点：{s['top10_fully_stable_points']} / {s['n_points']}")
    lines.append("")
    lines.append("## 解读")
    lines.append("")
    for t in out.get("interpretation") or []:
        lines.append(f"- {t}")
    lines.append("")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

"""口径走廊敏感性（离线）——审查 P0-2 里"需要重跑全量评估"的那几条。

报告页的「名次对口径参数的敏感度」一节只重算了能低成本重算的三项
（n_eff 入围门槛 / ridge 收缩 / 长尾桶门槛）。剩下这三条改变的是**指标定义本身**，
必须从原始快照重新走一遍评估，没法从已算好的榜单反推：

    rain_hourly_threshold_mm   1.0 → 0.5 / 2.0
    rain_daily_threshold_mm    1.0 → 0.5 / 2.0
    daily_min_hours            20  → 18 / 22

为什么值得单独跑一次：README 自己实测过，降水阈值 0.1→1.0 能让 BIAS 中位数
从 2.69 降到 1.26、ETS 从 0.078 升到 0.111——量级远大于 ±40% 的权重扰动。
也就是说，决定名次的"作者选择"里最大的一块，此前从未进过敏感性分析。

为什么不塞进每日流水线：每换一套阈值就要重跑一遍全量评估（分钟级），
一天三次跑不起。它是"季度体检"，不是"每轮验收"。

用法：
    .venv/bin/python scripts/sensitivity_corridors.py --month 2026-09
    .venv/bin/python scripts/sensitivity_corridors.py --month 2026-09 --out reports/data/corridors.json

产出一份 JSON，人工贴进 meta.diagnostics.corridor_offline（或直接读文件展示）。
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from weather_eval.config import load_config  # noqa: E402
from weather_eval.evaluate import build_report  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("corridors")

# 每条走廊：只改一个参数，其余完全不动——"一次只动一个旋钮"是敏感性分析的底线
CORRIDORS = [
    {"param": "rain_hourly_threshold_mm", "values": [0.5, 1.0, 2.0]},
    {"param": "rain_daily_threshold_mm", "values": [0.5, 1.0, 2.0]},
    {"param": "daily_min_hours", "values": [18, 20, 22]},
]


def _spearman(a: list[float], b: list[float]) -> float | None:
    """秩相关（并列取平均秩），与 stats._spearman 同口径。"""
    def rank(x):
        order = sorted(range(len(x)), key=lambda i: x[i])
        r = [0.0] * len(x)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    n = len(a)
    if n < 3 or n != len(b):
        return None
    ra, rb = rank(a), rank(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((ra[i] - ma) * (rb[i] - mb) for i in range(n))
    da = sum((ra[i] - ma) ** 2 for i in range(n)) ** 0.5
    db = sum((rb[i] - mb) ** 2 for i in range(n)) ** 0.5
    if da == 0 or db == 0:
        return None
    return num / (da * db)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default=None, help="YYYY-MM，缺省为本月")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="输出 JSON 路径")
    args = ap.parse_args()

    cfg = load_config(args.config)
    from weather_eval.timeutil import now_beijing, ym, floor_to_hour, parse_iso
    from weather_eval.__main__ import _month_window

    if args.month:
        start, end = _month_window(args.month)
        month = args.month
    else:
        now = floor_to_hour(now_beijing())
        month = ym(now)
        start = parse_iso(f"{month}-01T00:00")
        end = now

    log.info("基线评估（%s，%s → %s）…", month, start, end)
    base_eval = copy.deepcopy(dict(cfg.eval))
    base = build_report(cfg.station_ids, cfg.models, base_eval, start, end,
                        period_label=month)
    base_rows = (base.get("leaderboards") or {}).get("all") or []
    base_order = [r["model"] for r in base_rows if r.get("score") is not None]
    base_rank = {m: i for i, m in enumerate(base_order)}
    base_scores = {r["model"]: r.get("score") for r in base_rows}
    base_champ = base_order[0] if base_order else None
    log.info("  基线冠军：%s，入榜 %d 家", base_champ, len(base_order))

    out = {"month": month, "base_champion": base_champ, "corridors": []}
    for spec in CORRIDORS:
        param, values = spec["param"], spec["values"]
        default = base_eval.get(param)
        for v in values:
            if v == default:
                continue
            ev = copy.deepcopy(dict(cfg.eval))
            ev[param] = v
            log.info("重跑 %s = %s …", param, v)
            try:
                rep = build_report(cfg.station_ids, cfg.models, ev, start, end,
                                   period_label=month)
            except Exception as exc:
                log.warning("  %s=%s 重跑失败：%s", param, v, exc)
                out["corridors"].append({"param": param, "value": v, "error": str(exc)})
                continue
            rows = (rep.get("leaderboards") or {}).get("all") or []
            order = [r["model"] for r in rows if r.get("score") is not None]
            common = [m for m in order if m in base_rank and m in base_scores]
            a = [base_rank[m] for m in common]
            b = [order.index(m) for m in common]
            rho = _spearman([float(x) for x in a], [float(x) for x in b])
            champ = order[0] if order else None
            out["corridors"].append({
                "param": param, "value": v, "default": default,
                "spearman": None if rho is None else round(rho, 4),
                "top10_changed": len(set(base_order[:10]) - set(order[:10])),
                "champion": champ,
                "champion_changed": champ != base_champ,
                "n_ranked": len(order),
            })
            log.info("  → Spearman %s，前十换 %d 家，冠军 %s",
                     "—" if rho is None else f"{rho:.3f}",
                     out["corridors"][-1]["top10_changed"], champ)

    rhos = [c["spearman"] for c in out["corridors"] if c.get("spearman") is not None]
    out["min_spearman"] = min(rhos) if rhos else None
    out["champion_rotations"] = sum(1 for c in out["corridors"] if c.get("champion_changed"))
    out["note"] = ("改变的是指标定义本身（哪些小时算「下雨了」、哪些天算「有完整观测」），"
                   "必须从原始快照重跑，无法从已算好的榜单反推")

    dst = Path(args.out) if args.out else ROOT / ".work" / "corridors.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log.info("写出：%s", dst)
    log.info("最低 Spearman %s，冠军轮换 %d 次",
             out["min_spearman"], out["champion_rotations"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

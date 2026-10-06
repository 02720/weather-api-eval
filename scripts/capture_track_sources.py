"""导出生产管线的逐桶指标（track_sources），供斜率标定/冗余审计/口径对拍复用。

产出 JSON 形如 {"track_sources": {"hourly": {"temp": {模型: {桶: 指标}}, ...}, ...}}，
与 scripts/calibrate_score_slopes.py 的 --from-json 口径一致。

运行：
  PYTHONPATH=src python scripts/capture_track_sources.py --days 60 --out /tmp/track_sources.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta

sys.path.insert(0, "src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", default="/tmp/track_sources.json")
    ap.add_argument("--start", default=None, help="YYYY-MM-DDTHH:MM（缺省 = end − days）")
    ap.add_argument("--end", default=None, help="YYYY-MM-DDTHH:MM（缺省 = 当前北京时整点）")
    args = ap.parse_args()

    from weather_eval.config import load_config
    from weather_eval.evaluate import build_report
    from weather_eval.timeutil import now_beijing

    cfg = load_config()
    if args.end:
        end = __import__("datetime").datetime.fromisoformat(args.end)
    else:
        end = now_beijing().replace(minute=0, second=0, microsecond=0)
    start = (__import__("datetime").datetime.fromisoformat(args.start)
             if args.start else end - timedelta(days=args.days))
    print(f"构建 {start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M}（{len(cfg.models)} 源 /"
          f" {len(cfg.station_ids)} 站）…", flush=True)
    rep = build_report(cfg.station_ids, cfg.models, cfg.eval, start, end,
                       period_label="capture")

    ts = {"hourly": {"temp": rep["temp_hourly"], "precip": rep["precip_hourly_score"]},
          "daily": {"temp": rep["temp_daily"], "precip": rep["precip_score_daily"]}}
    # 诊断轨（0.1mm 口径 + 分级）也一并留档，供"全指标"审计用
    diag = {"precip_hourly": rep.get("precip_hourly"),
            "precip_daily": rep.get("precip_daily"),
            "scorecard": rep.get("scorecard")}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"window": {"start": f"{start:%Y-%m-%d %H:%M}",
                              "end": f"{end:%Y-%m-%d %H:%M}"},
                   "track_sources": ts, "diagnostics": diag}, f,
                  ensure_ascii=False)
    n_h = len(ts["hourly"]["temp"])
    buckets = sorted({b for m in ts["hourly"]["temp"].values() for b in m})
    print(f"已写出 {args.out}：小时轨 {n_h} 源 / {len(buckets)} 桶")


if __name__ == "__main__":
    main()

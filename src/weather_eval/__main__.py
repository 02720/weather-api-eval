"""命令行入口。

用法：
  python -m weather_eval fetch-obs                抓取 4 站近 24h 实况并归档
  python -m weather_eval fetch-forecast           抓取 3 模型起报快照并归档
  python -m weather_eval report                   生成本月至今运行报告
  python -m weather_eval monthly [--month YYYY-MM] 生成月度汇总报告
  python -m weather_eval all                       抓取观测+预报+生成本次运行报告（GitHub Action 调用）
"""
from __future__ import annotations

import argparse
import calendar
import logging
import re
import sys
from datetime import timedelta, datetime

from .config import load_config
from .timeutil import now_beijing, ymd, parse_iso, floor_to_hour, ym
from .storage import save_obs, save_forecast_snapshot
from .obs import EiaDataObsSource
from .forecast import OpenMeteoProvider, CaiyunProvider
from .evaluate import build_report
from .report import write_run_report, write_monthly_report, write_index

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("weather_eval")


def _month_window(month_str: str):
    start = parse_iso(f"{month_str}-01T00:00")
    _, last = calendar.monthrange(start.year, start.month)
    end = parse_iso(f"{month_str}-{last:02d}T23:00")
    return start, end


def _default_month() -> str:
    now = now_beijing()
    prev = (now.replace(day=1) - timedelta(days=1))
    return prev.strftime("%Y-%m")


def cmd_fetch_obs(args):
    cfg = load_config(args.config)
    src = EiaDataObsSource()
    failures = 0
    for st in cfg.stations:
        try:
            recs = src.fetch(st)
            n = save_obs(st.id, recs)
            log.info("站点 %s 写入 %d 条（累计去重后）", st.id, n)
        except Exception as e:  # noqa: BLE001
            failures += 1
            log.error("站点 %s 抓取失败: %s", st.id, e)
    return failures


def cmd_fetch_forecast(args):
    cfg = load_config(args.config)
    source = getattr(args, "source", "open_meteo")
    if source == "caiyun":
        prov = CaiyunProvider()
        model_list = [prov.name]
    else:
        prov = OpenMeteoProvider()
        # 仅把"非彩云"的模型交给 Open-Meteo，避免把 caiyun_v2_6 当作缺失模型而刷警告
        model_list = [m for m in cfg.models if m != CaiyunProvider.DEFAULT_NAME]
    failures = 0
    for st in cfg.stations:
        try:
            snap = prov.fetch_snapshot(st, model_list)
            for m in snap["models"]:
                sub = dict(snap)
                sub["models"] = [m]
                sub["data"] = {m: snap["data"][m]}
                save_forecast_snapshot(st.id, m, sub)
            log.info("站点 %s 起报 %s 已存档（模型 %s）", st.id, snap["issue_iso"], snap["models"])
        except Exception as e:  # noqa: BLE001
            failures += 1
            log.error("站点 %s 预报抓取失败: %s", st.id, e)
    return failures


def _prune_runs(keep_days: int = 60) -> None:
    """清理 reports/runs 中超过 keep_days 天的旧报告，避免无限增长。"""
    from .report.render import REPORTS_ROOT
    runs_dir = REPORTS_ROOT / "runs"
    if not runs_dir.exists():
        return
    cutoff = datetime.now() - timedelta(days=keep_days)
    stamp_re = re.compile(r"(\d{8}-\d{4})$")
    for p in runs_dir.glob("*.html"):
        m = stamp_re.search(p.stem)
        if not m:
            continue
        try:
            if datetime.strptime(m.group(1), "%Y%m%d-%H%M") < cutoff:
                p.unlink()
        except ValueError:
            continue


def cmd_report(args):
    cfg = load_config(args.config)
    now = floor_to_hour(now_beijing())
    month = ym(now)  # YYYY-MM
    start = parse_iso(f"{month}-01T00:00")
    data = build_report(cfg.station_ids, cfg.models, cfg.eval, start, now, period_label=month)
    out = write_run_report(data)
    _prune_runs()
    write_index()
    log.info("运行报告已生成: %s", out)


def cmd_monthly(args):
    cfg = load_config(args.config)
    month = args.month or _default_month()
    start, end = _month_window(month)
    data = build_report(cfg.station_ids, cfg.models, cfg.eval, start, end,
                        period_label=month, is_monthly=True)
    out = write_monthly_report(data)
    write_index()
    log.info("月度报告已生成: %s", out)


def cmd_all(args):
    f1 = cmd_fetch_obs(args)
    f2 = cmd_fetch_forecast(args)
    cmd_report(args)
    total = (f1 or 0) + (f2 or 0)
    if total:
        log.error("本次运行有 %d 个站点抓取失败", total)
        sys.exit(1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="weather_eval", description="天气预报 API 准确度评估")
    p.add_argument("--config", default=None, help="stations.yaml 路径")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("fetch-obs")
    p_fetch = sub.add_parser("fetch-forecast")
    p_fetch.add_argument("--source", choices=["open_meteo", "caiyun"], default="open_meteo",
                         help="预报源：open_meteo（默认）或 caiyun（v2.6 Token 认证，需 CAIYUN_TOKEN 环境变量）")
    sub.add_parser("report")
    pm = sub.add_parser("monthly")
    pm.add_argument("--month", default=None, help="YYYY-MM，默认上一自然月")
    sub.add_parser("all")

    args = p.parse_args(argv)
    {
        "fetch-obs": cmd_fetch_obs,
        "fetch-forecast": cmd_fetch_forecast,
        "report": cmd_report,
        "monthly": cmd_monthly,
        "all": cmd_all,
    }[args.cmd](args)


if __name__ == "__main__":
    main()

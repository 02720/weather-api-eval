"""命令行入口。

用法：
  python -m weather_eval fetch-obs                抓取 4 站近 24h 实况并归档
  python -m weather_eval fetch-forecast           抓取 3 模型起报快照并归档
  python -m weather_eval report                   用本月至今数据更新主报告 reports/index.html
  python -m weather_eval monthly [--month YYYY-MM] 生成月度归档报告 reports/monthly/YYYY-MM.html
  python -m weather_eval all                       抓取观测+预报+更新主报告（GitHub Action 调用）

报告体系（2026-08 重设计）：
  index.html 是"本月至今"的累积视图，每次运行覆盖更新（不再保留每次运行一份的 runs/）；
  monthly/ 每月归档一份冻结的历史月份，主报告页脚自动列出归档链接。
"""
from __future__ import annotations

import argparse
import calendar
import logging
import sys
from datetime import timedelta

from .config import load_config
from .timeutil import now_beijing, ymd, parse_iso, floor_to_hour, ym
from .storage import save_obs, save_forecast_snapshot
from .obs import EiaDataObsSource
from .forecast import OpenMeteoProvider, CaiyunProvider, QWeatherProvider
from .forecast.caiyun import DEFAULT_NAME as CAIYUN_DEFAULT_MODEL
from .forecast.qweather import DEFAULT_NAME as QWEATHER_DEFAULT_MODEL
# 非第三方模式名的"独立抓取源"，Open-Meteo 抓取分支必须排除，
# 否则会被当作 Open-Meteo 响应里缺失的模型而刷警告。
NON_OPENMETEO_MODELS = {CAIYUN_DEFAULT_MODEL, QWEATHER_DEFAULT_MODEL}
from .evaluate import build_report
from .report import write_live_report, write_monthly_report

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


def _build_provider(source: str, cfg):
    """按 --source 构造预报快照器及其模型列表。

    独立源的凭据缺失属于配置错误：在构造期即失败并给出可操作提示，
    而非带着 traceback 崩溃（此前缺 Token 时会直接抛出未捕获异常）。
    """
    if source == "caiyun":
        prov = CaiyunProvider()
        return prov, [prov.name]
    if source == "qweather":
        prov = QWeatherProvider()
        return prov, [prov.name]
    # Open-Meteo 无需凭据；仅交其自身模型，避免把 caiyun/qweather 当作缺失模型刷警告
    return OpenMeteoProvider(), [m for m in cfg.models if m not in NON_OPENMETEO_MODELS]


def cmd_fetch_forecast(args):
    cfg = load_config(args.config)
    source = getattr(args, "source", "open_meteo")
    try:
        prov, model_list = _build_provider(source, cfg)
    except Exception as e:  # noqa: BLE001
        log.error("预报源 %s 初始化失败（请检查相应环境变量/凭据配置）: %s", source, e)
        sys.exit(1)
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


def _update_live_report(cfg):
    """用"本月 1 号至今"的累计数据重建主报告 reports/index.html（覆盖写）。"""
    now = floor_to_hour(now_beijing())
    month = ym(now)  # YYYY-MM
    start = parse_iso(f"{month}-01T00:00")
    data = build_report(cfg.station_ids, cfg.models, cfg.eval, start, now, period_label=month)
    out = write_live_report(data, station_labels={s.id: s.name for s in cfg.stations})
    return month, out


def cmd_report(args):
    cfg = load_config(args.config)
    month, out = _update_live_report(cfg)
    log.info("主报告已更新（%s 累积至今）: %s", month, out)


def cmd_monthly(args):
    """把某个自然月冻结为月度归档 reports/monthly/YYYY-MM.html（默认上一自然月）。"""
    cfg = load_config(args.config)
    month = args.month or _default_month()
    start, end = _month_window(month)
    data = build_report(cfg.station_ids, cfg.models, cfg.eval, start, end,
                        period_label=month, is_monthly=True)
    out = write_monthly_report(data, station_labels={s.id: s.name for s in cfg.stations})
    log.info("月度归档已生成: %s", out)
    # 归档列表是主报告渲染时快照的：立即重建一次主报告，
    # 让新归档在本次部署就出现在首页页脚，而不是等下一次定时运行。
    _update_live_report(cfg)


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
    p_fetch.add_argument(
        "--source", choices=["open_meteo", "caiyun", "qweather"], default="open_meteo",
        help="预报源：open_meteo（默认）、caiyun（需 CAIYUN_TOKEN）或 qweather"
             "（和风天气，API Key 认证，需 QWEATHER_API_KEY 环境变量，"
             "可选 QWEATHER_API_HOST 指定专属 API Host）",
    )
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

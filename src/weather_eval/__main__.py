"""命令行入口。

用法：
  python -m weather_eval fetch-obs                抓取 4 站近 24h 实况并归档
  python -m weather_eval fetch-forecast           抓取 Open-Meteo 多模型起报快照并归档
  python -m weather_eval fetch-forecast --source caiyun    抓取彩云天气 v2.6 起报
  python -m weather_eval fetch-forecast --source qweather  抓取和风天气起报
  python -m weather_eval fetch-forecast --source tianji    抓取中科天机起报（网页接口，无需凭据）
  python -m weather_eval fetch-forecast --source fuxi      抓取伏羲中期 FuXi-C88 起报（网页接口，无需凭据）
  python -m weather_eval fetch-forecast --source fuxi_data 抓取伏羲确定性 FuXi-Det 起报（需 FUXI_DATA_TOKEN）
  python -m weather_eval fetch-forecast --source fengwu    抓取风乌 FengWu-GHR-9km 起报（可选 FENGWU_API_KEY）
  python -m weather_eval fetch-forecast --source geovis    抓取中科星图逐小时预报起报（需 GEVIS_TOKEN）
  python -m weather_eval fetch-forecast --source accuweather 抓取 AccuWeather 逐小时预报起报（需 ACCUWEATHER_API_KEY）
  python -m weather_eval fetch-forecast --source msn        抓取 MSN 天气（中国天气网）起报（网页接口，无需凭据）
  python -m weather_eval report                   用本月至今数据更新主报告 reports/index.html
  python -m weather_eval monthly [--month YYYY-MM] 生成月度归档报告 reports/monthly/YYYY-MM.html
  python -m weather_eval all                       抓取观测+预报+更新主报告（GitHub Action 调用）

报告体系（2026-08 重设计）：
  index.html 是"本月至今"的累积视图，每次运行覆盖更新（不再保留每次运行一份的 runs/）；
  monthly/ 每月归档一份冻结的历史月份，主报告页脚自动列出归档链接。

快照粒度说明：Open-Meteo/彩云/和风的一次抓取共享同一条时间轴与起报口径，按模型拆分
存档；中科天机各模式最新可用起报轮次可能不同步（发布进度独立），故其提供方直接按
模型返回独立快照（各自 issue_iso 与时间轴），保证时效（lead）分组不被跨模式错位污染。
"""
from __future__ import annotations

import argparse
import calendar
import logging
import re
import sys
from datetime import timedelta
from typing import Any

from .config import load_config
from .timeutil import now_beijing, ymd, parse_iso, floor_to_hour, ym
from .storage import save_obs, save_forecast_snapshot
from .obs import EiaDataObsSource
from .forecast import (
    OpenMeteoProvider, CaiyunProvider, QWeatherProvider, TianjiProvider,
    FuxiC88Provider, FuxiDetProvider, FengWuProvider, GevisProvider,
    AccuWeatherProvider, MsnProvider,
)
from .forecast.caiyun import DEFAULT_NAME as CAIYUN_DEFAULT_MODEL
from .forecast.qweather import DEFAULT_NAME as QWEATHER_DEFAULT_MODEL
from .forecast.tianji import MODEL_SPECS as TJ_MODEL_SPECS
from .forecast.fuxi import MODEL_NAME as FUXI_C88_MODEL
from .forecast.fuxi_data import MODEL_NAME as FUXI_DET_MODEL
from .forecast.fengwu import MODEL_NAME as FENGWU_MODEL
from .forecast.geovis import MODEL_NAME as GEVIS_MODEL
from .forecast.accuweather import MODEL_NAME as ACCUWEATHER_MODEL
from .forecast.msn import MODEL_NAME as MSN_MODEL
# 独立抓取源（非 Open-Meteo 模型）的登记处：source -> (模型集合, 提供方类)。
# 单一数据源：模型集合与提供方类必须同步登记，此前分成两张表（SOURCE_MODELS 与
# _build_provider 内的内联字典）手工同步，新增源漏登其一会退化成运行期 KeyError
# 且只在运行到该源时才暴露。这里合并后两张派生表自动同步，新增源只需改一处。
# 提供方登记为**零参工厂**而非类对象：lambda 体内对模块级符号的解析发生在调用时，
# 保留了晚绑定（测试以 monkeypatch 替换 m.<Xxx>Provider 来注入假实现；若在此处按值
# 绑定类对象，替换将失效——这不是测试细节，而是"CLI 应可被注入"的可测性契约）。
SOURCE_SPECS: dict[str, tuple[set[str], Any]] = {
    # 伏羲中期：可视化接口，游客可用
    "fuxi": ({FUXI_C88_MODEL}, lambda: FuxiC88Provider()),
    # 伏羲确定性：数据服务，需 FUXI_DATA_TOKEN
    "fuxi_data": ({FUXI_DET_MODEL}, lambda: FuxiDetProvider()),
    "fengwu": ({FENGWU_MODEL}, lambda: FengWuProvider()),       # 风乌 GHR-9km：可选 FENGWU_API_KEY
    "geovis": ({GEVIS_MODEL}, lambda: GevisProvider()),         # 中科星图：需 GEVIS_TOKEN
    # AccuWeather：需 ACCUWEATHER_API_KEY（Enterprise 入口）
    "accuweather": ({ACCUWEATHER_MODEL}, lambda: AccuWeatherProvider()),
    "msn": ({MSN_MODEL}, lambda: MsnProvider()),                # MSN 天气：无凭据，底层中国天气网
}
# 各独立源的模型集合（config 中按此过滤，防止把别家的模型传进去刷缺失警告）
SOURCE_MODELS: dict[str, set[str]] = {s: ms for s, (ms, _) in SOURCE_SPECS.items()}
# 非第三方模式名的"独立抓取源"，Open-Meteo 抓取分支必须排除，
# 否则会被当作 Open-Meteo 响应里缺失的模型而刷警告。
NON_OPENMETEO_MODELS = {
    CAIYUN_DEFAULT_MODEL, QWEATHER_DEFAULT_MODEL, *TJ_MODEL_SPECS,
    *(m for ms in SOURCE_MODELS.values() for m in ms),
}
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


# 合法月份：YYYY-MM，月 01-12（避免 "2026-13" 之类的输入裸 traceback）
_MONTH_RE = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


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
    if source == "tianji":
        # 中科天机无需凭据；仅交其自身模型（config 中以 tj_ 前缀区分）
        tjs = [m for m in cfg.models if m in TJ_MODEL_SPECS]
        if not tjs:
            raise RuntimeError(
                "config models 中未配置任何中科天机模型（tj_*），无法抓取该源"
            )
        return TianjiProvider(), tjs
    if source in SOURCE_MODELS:
        wanted = [m for m in cfg.models if m in SOURCE_MODELS[source]]
        if not wanted:
            raise RuntimeError(
                f"config models 中未配置任何 {source} 源模型（{sorted(SOURCE_MODELS[source])}），"
                "无法抓取该源"
            )
        _, provider_factory = SOURCE_SPECS[source]
        return provider_factory(), wanted
    # Open-Meteo 无需凭据；仅交其自身模型，避免把独立源模型当作缺失模型刷警告
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
            if isinstance(snap, dict):
                # 共享时间轴的多模型快照（Open-Meteo/彩云/和风）：按模型拆为独立存档单元
                subs = []
                for m in snap["models"]:
                    sub = dict(snap)
                    sub["models"] = [m]
                    sub["data"] = {m: snap["data"][m]}
                    subs.append(sub)
            else:
                # 中科天机：各模式最新可用起报可能不同步，提供方直接返回按模型独立的快照列表
                # （每份各自 issue_iso 与时间轴），保证时效（lead）分组不被跨模式错位污染。
                subs = list(snap)
            for sub in subs:
                save_forecast_snapshot(st.id, sub["models"][0], sub)
            log.info("站点 %s 起报已存档 %d 份（模型 %s）",
                     st.id, len(subs), [s["models"][0] for s in subs])
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
    """把某个自然月冻结为月度归档 reports/monthly/YYYY-MM.html（默认上一自然月）。

    已存在的归档默认拒绝重写（冻结档案永不改动）；--force 才允许重建。
    """
    cfg = load_config(args.config)
    month = args.month or _default_month()
    if not _MONTH_RE.match(month):
        raise SystemExit(f"无效月份: {month!r}（应为 YYYY-MM，如 2026-07）")
    start, end = _month_window(month)
    data = build_report(cfg.station_ids, cfg.models, cfg.eval, start, end,
                        period_label=month, is_monthly=True)
    out = write_monthly_report(data, station_labels={s.id: s.name for s in cfg.stations},
                               force=args.force)
    log.info("月度归档就绪（已存在的冻结档案保留不动）: %s", out)
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
        "--source", choices=["open_meteo", "caiyun", "qweather", "tianji",
                             "fuxi", "fuxi_data", "fengwu", "geovis",
                             "accuweather", "msn"],
        default="open_meteo",
        help="预报源：open_meteo（默认）、caiyun（需 CAIYUN_TOKEN）、qweather"
             "（需 QWEATHER_API_KEY）、tianji（网页接口，无需凭据）、fuxi（伏羲中期"
             " FuXi-C88，网页接口，无需凭据）、fuxi_data（伏羲确定性 FuXi-Det，需 "
             "FUXI_DATA_TOKEN）、fengwu（FengWu-GHR-9km，可选 FENGWU_API_KEY 延长"
             "时效）、geovis（中科星图，需 GEVIS_TOKEN）、accuweather（AccuWeather"
             " Enterprise，需 ACCUWEATHER_API_KEY）、msn（MSN 天气/中国天气网，"
             "无需凭据，时效上限约 9.3 天）",
    )
    sub.add_parser("report")
    pm = sub.add_parser("monthly")
    pm.add_argument("--month", default=None, help="YYYY-MM，默认上一自然月")
    pm.add_argument("--force", action="store_true",
                    help="已存在同名归档时强制重写（默认拒绝改动冻结档案）")
    sub.add_parser("all")

    args = p.parse_args(argv)
    rc = {
        "fetch-obs": cmd_fetch_obs,
        "fetch-forecast": cmd_fetch_forecast,
        "report": cmd_report,
        "monthly": cmd_monthly,
        "all": cmd_all,
    }[args.cmd](args)
    # 抓取类命令的失败数必须反映到退出码：否则单独运行 fetch-forecast 失败也会
    # 以 0 退出，CI 的 continue-on-error 步骤（彩云/和风/中科天机）连"失败标注"都不会出现。
    if rc:
        log.error("本次运行有 %d 项失败", rc)
        sys.exit(1)


if __name__ == "__main__":
    main()

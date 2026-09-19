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
  python -m weather_eval fetch-forecast --source cma_public 抓取中国气象局公众网（weather.cma.cn）起报（公开接口，无需凭据）
  python -m weather_eval report                   用本月至今数据更新主报告 reports/index.html
  python -m weather_eval monthly [--month YYYY-MM] 生成月度归档报告 reports/monthly/YYYY-MM.html
  python -m weather_eval archive [--days 60] [--apply]  把超窗口的旧快照 gzip 归档（默认 dry-run）
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
    AccuWeatherProvider, MsnProvider, Ew4allProvider, CmaPublicProvider,
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
from .forecast.ew4all import MODEL_SPECS as EW4ALL_MODEL_SPECS
from .forecast.cma_public import MODEL_NAME as CMA_PUBLIC_MODEL
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
    # EW4ALL（云上早期预警支撑系统）：网页接口，无凭据；一次抓取返回两个模型
    # 各自的独立快照（起报轮次发布进度不同步，见 forecast/ew4all.py）
    "ew4all": (set(EW4ALL_MODEL_SPECS), lambda: Ew4allProvider()),
    # 中国气象局公众气象服务网：公开 JSON 接口，无凭据（站点须配 cma_id = WMO 站号）
    "cma_public": ({CMA_PUBLIC_MODEL}, lambda: CmaPublicProvider()),
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
                    # 逐日预报块（可选扩展，见 forecast/base.py）：与 hourly 一样
                    # 按模型拆分，否则拆分后的子快照会带着别家的日产品
                    if isinstance(snap.get("daily"), dict) and m in snap["daily"]:
                        sub["daily"] = {m: snap["daily"][m]}
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
    # 哈希链清单（§7.1）：把当轮全部快照的 Merkle 根落盘，供 verify 与月度归档公示
    from .storage import save_manifest
    save_manifest(month, {
        **data["meta"].get("integrity", {}),
        "period_label": month,
        "generated_at": data["meta"]["generated_at"],
    })
    # 源健康度看板与主报告同批刷新（P1-8）：让"静默死亡"变成"一眼可见"。
    # 它不进主报告（主报告面向读者，健康度面向维护者），但必须与主报告同时是新
    # 的——否则看板会变成一份没人更新的摆设。
    n_stale = write_health_report(cfg, stale_hours=int(cfg.eval.get("source_stale_hours", 30)))
    if n_stale:
        log.warning("有 %d 个源已陈旧，详见 reports/health.html", n_stale)
    return month, out


def write_health_report(cfg, stale_hours: int = 30):
    """写 reports/health.html，返回陈旧源个数（0 = 全部健康）。"""
    from . import health as _health
    from .evaluate import _preload
    from .report.render import write_health_page
    from .timeutil import now_beijing as _now

    _obs_maps, snapshots = _preload(cfg.station_ids, cfg.models)
    h = _health.of(snapshots, cfg.station_ids, cfg.models)
    h = _health.evaluate_staleness(h, stale_hours=stale_hours)
    now = _now()
    meta = {
        "period_label": ym(now),
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "start": f"{ym(now)}-01 00:00",
        "end": floor_to_hour(now).strftime("%Y-%m-%d %H:%M"),
    }
    write_health_page(_health.render_health_html(h, meta, stale_hours))
    return len(_health.stale_sources(h))


def cmd_health(args):
    """源健康度检查（CI 用）：陈旧源存在时以非零退出，供自动开 Issue 步骤捕捉。"""
    cfg = load_config(args.config)
    n_stale = write_health_report(cfg, stale_hours=args.stale_hours)
    if n_stale:
        log.error("有 %d 个源超过 %d 小时未成功抓取（详见 reports/health.html）",
                  n_stale, args.stale_hours)
        return n_stale
    log.info("全部源健康（阈值 %d 小时）", args.stale_hours)
    return 0


def cmd_verify(args):
    """重算全部快照的 Merkle 根并与清单比对（§7.1）。

    这是"预报必须在实况之前封存"这条地基从**自我声明**变成**可机器核验**的那一步：
    清单（data/manifest/{period}.json）记录了当轮全部快照的哈希聚合根，任何人克隆
    仓库后重算一遍即可验证存档未被事后改写。根不一致 = 有文件被改过/被补写/丢失，
    以非零退出让 CI 变红——绝不静默通过。
    """
    from .snapshot_meta import integrity_summary
    from .storage import load_manifest

    cfg = load_config(args.config)
    _obs, snapshots = _preload_snapshots(cfg)
    all_snaps = [s for lst in snapshots.values() for s in lst]
    live = integrity_summary(all_snaps)
    stored = load_manifest(args.period)
    if not stored:
        log.warning("未找到哈希链清单（data/manifest/*.json），跳过校验；"
                    "下一次 report/all 运行会自动生成")
        return 0
    if stored.get("merkle_root") is None:
        log.warning("清单中没有 Merkle 根（可能来自更早的版本），跳过校验")
        return 0
    if live["merkle_root"] != stored.get("merkle_root"):
        log.error(
            "哈希链校验失败：存档内容与清单不一致。\n"
            "  清单（%s）：%s（%s 份快照）\n  实测：%s（%s 份快照）\n"
            "  含义：有快照文件在清单生成之后被修改、补写或删除。",
            stored.get("period_label") or "最新", stored.get("merkle_root"),
            stored.get("n_snapshots"), live["merkle_root"], live["n_snapshots"])
        return 1
    log.info("哈希链校验通过：%d 份快照，Merkle 根 %s（抓取时刻 %s ~ %s）",
             live["n_snapshots"], live["merkle_root"],
             live["fetched_first"], live["fetched_last"])
    return 0


def _preload_snapshots(cfg):
    from .evaluate import _preload
    return _preload(cfg.station_ids, cfg.models)


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


def cmd_archive(args):
    """把 issue 早于保留窗口的旧快照 gzip 归档（P3-5）。

    默认 dry-run 只列出候选；--apply 才真正压缩并删除源 .json。评估读取侧对
    .json/.json.gz 一视同仁，归档不改变任何指标口径。git 侧表现为"删除 N 个
    .json + 新增 N 个 .json.gz"，随下一次自动提交入库。"""
    from .storage import archive_old_snapshots
    candidates = archive_old_snapshots(args.days, apply=False)
    n = len(candidates)
    if n == 0:
        log.info("没有早于 %d 天的快照需要归档", args.days)
        return 0
    if not args.apply:
        log.info("dry-run：%d 份快照早于 %d 天可归档（加 --apply 执行）", n, args.days)
        for p in candidates[:10]:
            log.info("  %s", p)
        if n > 10:
            log.info("  … 及另外 %d 份", n - 10)
        return 0
    total_in = sum(p.stat().st_size for p in candidates)
    archive_old_snapshots(args.days, apply=True)
    gz_sizes = [p.with_name(p.name + ".gz").stat().st_size for p in candidates]
    log.info("已归档 %d 份旧快照（%.1f MB -> %.1f MB，%.1f×）",
             n, total_in / 1e6, sum(gz_sizes) / 1e6,
             (total_in / sum(gz_sizes)) if sum(gz_sizes) else 0.0)
    return 0


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
                             "accuweather", "msn", "ew4all", "cma_public"],
        default="open_meteo",
        help="预报源：open_meteo（默认）、caiyun（需 CAIYUN_TOKEN）、qweather"
             "（需 QWEATHER_API_KEY）、tianji（网页接口，无需凭据）、fuxi（伏羲中期"
             " FuXi-C88，网页接口，无需凭据）、fuxi_data（伏羲确定性 FuXi-Det，需 "
             "FUXI_DATA_TOKEN）、fengwu（FengWu-GHR-9km，可选 FENGWU_API_KEY 延长"
             "时效）、geovis（中科星图，需 GEVIS_TOKEN）、accuweather（AccuWeather"
             " Enterprise，需 ACCUWEATHER_API_KEY）、msn（MSN 天气/中国天气网，"
             "无需凭据，时效上限约 9.3 天）、ew4all（CMA 云上早期预警支撑系统，"
             "无需凭据，含 CMA-NDFS 与 风清AI模式；温度 10/15 天，降水 10 天）、"
             "cma_public（中国气象局公众气象服务网 weather.cma.cn，无需凭据，"
             "以 WMO 站号寻址、站点须配 cma_id；3 小时分辨率，覆盖约 7 天）",
    )
    sub.add_parser("report")
    pm = sub.add_parser("monthly")
    pm.add_argument("--month", default=None, help="YYYY-MM，默认上一自然月")
    pm.add_argument("--force", action="store_true",
                    help="已存在同名归档时强制重写（默认拒绝改动冻结档案）")
    pa = sub.add_parser("archive")
    pa.add_argument("--days", type=int, default=60,
                    help="保留窗口（天）：issue 早于该窗口的快照被归档，默认 60")
    pa.add_argument("--apply", action="store_true",
                    help="真正执行压缩并删除源文件（默认 dry-run 只列出候选）")
    ph = sub.add_parser("health")
    ph.add_argument("--stale-hours", type=int, default=30,
                    help="陈旧阈值（小时）：超过该时长未成功抓取的源会使命令以非零退出")
    pv = sub.add_parser("verify")
    pv.add_argument("--period", default=None,
                    help="要核对的清单月份 YYYY-MM（缺省 = 最新一份）")
    sub.add_parser("all")

    args = p.parse_args(argv)
    rc = {
        "fetch-obs": cmd_fetch_obs,
        "fetch-forecast": cmd_fetch_forecast,
        "report": cmd_report,
        "monthly": cmd_monthly,
        "archive": cmd_archive,
        "health": cmd_health,
        "verify": cmd_verify,
        "all": cmd_all,
    }[args.cmd](args)
    # 抓取类命令的失败数必须反映到退出码：否则单独运行 fetch-forecast 失败也会
    # 以 0 退出，CI 的 continue-on-error 步骤（彩云/和风/中科天机）连"失败标注"都不会出现。
    if rc:
        log.error("本次运行有 %d 项失败", rc)
        sys.exit(1)


if __name__ == "__main__":
    main()

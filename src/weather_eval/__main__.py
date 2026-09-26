"""命令行入口。

用法：
  python -m weather_eval fetch-obs                抓取 4 站近 24h 实况并归档（多源编排：
                                                  主源 eia-data，失败/陈旧时自动降级到
                                                  中国气象数据网备用源）
  python -m weather_eval fetch-obs --source cma_data   只用中国气象数据网实况源（对照/排障）
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
  python -m weather_eval compact [--retain-months 13] [--apply]  月度冻结 + 超期出仓
                                                   （体积治理主线：默认 dry-run，幂等）
  python -m weather_eval footprint [--warn-mb 400] [--fail-mb 900]  仓库体量看门狗
                                                   （超硬阈值非零退出，CI 据此告警）
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
import json
import logging
import re
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

from .config import DEFAULT_EVAL, load_config
from .timeutil import now_beijing, ymd, parse_iso, floor_to_hour, ym
from .storage import (
    PROJECT_ROOT, compact_snapshots, data_footprint, period_summary_path, reports_footprint,
    save_obs,
    save_forecast_snapshot,
)
from .obs import EiaDataObsSource, ObsChain
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


def _known_obs_hours(station_id: str) -> set[str]:
    """该站已入库的观测时刻（ISO 北京时）。

    给 CMA 备用源做 `skip_hours`：一次只返回一个时刻的接口，不该把上一轮刚写过的
    24 小时再问一遍。只读最近两个月的文件——回看窗口只有 26 小时，读全量历史没有意义。
    """
    from .storage import load_obs
    from .timeutil import ym as _ym
    now = now_beijing()
    months = {_ym(now), _ym(now - timedelta(days=31))}
    out: set[str] = set()
    for m in months:
        out.update(load_obs(station_id, m))
    return out


def _build_obs_sources(cfg, source: str, station):
    """按 --source 构造观测源字典（顺序由 cfg.obs_sources 决定）。

    auto = 把配置里登记的全部源都交给编排层，主源可用时不惊动备用源；
    指定单个源 = 只跑该源（用于对照、故障定位与单源复算）。
    """
    from .obs import CmaDataObsSource, EiaDataObsSource
    wanted = list(cfg.obs_sources) if source == "auto" else [source]
    out = {}
    for name in wanted:
        if name == "eia_data":
            out[name] = EiaDataObsSource()
        elif name == "cma_data":
            out[name] = CmaDataObsSource(skip_hours=_known_obs_hours(station.id))
        else:
            raise RuntimeError(
                f"未知观测源 {name!r}（可用：eia_data、cma_data；"
                "见 obs/chain.py 的 OBS_SOURCE_PRIORITY）")
    if not out:
        raise RuntimeError("观测源列表为空：请检查 config 的 obs_sources")
    return out


def cmd_fetch_obs(args):
    """抓取各站观测并归档。

    多源编排（2026-09）：主源（eia-data）失败、或抓通了但**数据陈旧/窗口截断**时，
    自动降级到备用源（中国气象数据网）补位；合并结果的来源构成写进日志留痕。
    观测是评估里唯一的真值来源，单点依赖等于把全部结论押在一个第三方页面上。
    """
    cfg = load_config(args.config)
    source = getattr(args, "source", "auto")
    stale_hours = float(cfg.eval.get("obs_stale_hours", 3.0))
    min_hours = int(cfg.eval.get("obs_min_hours", 6))
    failures = 0
    for st in cfg.stations:
        try:
            sources = _build_obs_sources(cfg, source, st)
            chain = ObsChain(sources, priority=tuple(cfg.obs_sources),
                             stale_hours=stale_hours, min_hours=min_hours)
            recs, report = chain.fetch(st)
            n = save_obs(st.id, recs)
            log.info("站点 %s 写入 %d 条（累计去重后）；来源构成：%s",
                     st.id, n, report.describe())
            if report.degraded:
                log.warning("站点 %s 本轮观测已降级（%d 条由备用源补位）——"
                            "请核对首选源是否改版或停摆", st.id, report.n_filled)
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
                # 日产品评测范围必须**显式**随调用传入：save 路径虽有缺省回退
                # （读默认配置），但本 CLI 支持 --config 覆盖——缺省回退只认
                # 仓库默认配置，会让自定义配置的评测范围与入库截断口径分裂
                save_forecast_snapshot(
                    st.id, sub["models"][0], sub,
                    daily_max_offset_days=int(cfg.eval.get(
                        "daily_max_offset_days",
                        DEFAULT_EVAL["daily_max_offset_days"])))
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

    同时把该月**结论**固化成 data/metrics/{month}/summary.json（体积治理的前提）：
    原始快照将在保留期后出仓，出仓之前必须先有这份摘要，否则"删掉原始数据"就等于
    "结论不可复核"。`compact` 出仓前会检查它是否存在，不存在就拒绝删除。
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
    _write_period_summary(month, data)
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


def _write_period_summary(period: str, data: dict) -> Path:
    """把某月的**结论**固化成一份轻量摘要，写进 data/metrics/{period}/summary.json。

    这是"先固化、后删除"里的那一步固化，也是体积治理能安全成立的**前提**：
    原始快照终将从仓库出仓（默认保留 13 个月），出仓之后，"那个月谁最准、差多少"
    这个问题只能靠这份摘要回答。因此这里只保留**结论与口径**（排行榜、评分卡、
    元数据、覆盖率），不保留逐样本的明细数组（temp_hourly / heatmap / timeseries
    / per_station）——那些是重算用的原料，正是要出仓的东西。

    摘要的体积本身必须小且稳定：它永久留在仓库里，不能变成新的增长源。
    """
    summary = {
        "period": period,
        "frozen_at": now_beijing().strftime("%Y-%m-%d %H:%M"),
        "note": "原始预报快照出仓后，本文件是该月结论的唯一可复核来源",
        "meta": data.get("meta", {}),
        "coverage": data.get("coverage"),
        "scorecard": data.get("scorecard"),
        "leaderboards": data.get("leaderboards"),
    }
    path = period_summary_path(period)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, sort_keys=False)
    size = path.stat().st_size
    log.info("月度结论已固化：%s（%.0f KB，含总榜与分时效榜）", path, size / 1024)
    return path


def cmd_compact(args):
    """月度冻结 + 超期出仓：让仓库能**持续**自动化运行的那一步。

    为什么必须做：起报快照是只增不减的证据流（实测 ~170 份/天、单份中位 18.9 KB、
    约 2.8 MB/天，一年 ≈ 6 万文件 / 1 GB 原始 JSON）。不治理，git 的每次
    add/status/clone 都会肉眼可见地变慢，最终把"每天自动跑三次"变成不可能。

    分层与不变量（详见 storage.compact_snapshots 的 docstring）：
        当月                逐份 .json（热层，追加写入、随时可读）
        已结束的自然月      一份 {YYYY-MM}.json.gz（冻结，永不重写）
        超过保留月数        出仓（先确认该月结论摘要已固化，否则拒绝删除）

    默认 dry-run。--apply 才落盘；幂等（再跑一次无候选）。
    """
    cfg = load_config(args.config)
    grace = args.grace_days if args.grace_days is not None \
        else int(cfg.eval.get("compact_grace_days", 2))
    retain = args.retain_months if args.retain_months is not None \
        else int(cfg.eval.get("compact_retain_months", 13))

    fp0 = data_footprint()
    log.info("治理前：data/ 共 %.1f MB / %d 个文件（热层 %d 个，冷层 %d 个）",
             fp0["total_bytes"] / 1e6, fp0["total_files"],
             fp0["forecast_hot_files"], fp0["forecast_cold_files"])

    rep = compact_snapshots(grace_days=grace, retain_months=retain,
                            apply=args.apply, force=args.force)
    n_new = len(rep["bundles_created"])
    n_pending = rep["files_pending"]
    n_removed = rep["files_removed"]
    saved = rep["bytes_before"] - rep["bytes_after"]

    if n_new == 0 and not rep["expired"] and not rep["expiry_blocked"]:
        log.info("没有需要冻结或出仓的月份（保留期 %d 个月，宽限 %d 天）", retain, grace)
    if n_new:
        verb = "已冻结" if args.apply else "可冻结"
        log.info("%s %d 个月度 bundle：%.2f MB → %.2f MB（%.1f×），%s %d 个散装快照",
                 verb, n_new, rep["bytes_before"] / 1e6, rep["bytes_after"] / 1e6,
                 (rep["bytes_before"] / rep["bytes_after"]) if rep["bytes_after"] else 0.0,
                 "已删除" if args.apply else "将删除",
                 n_removed if args.apply else n_pending)
    if rep["bundles_skipped"]:
        log.info("%d 个包已冻结（冻结档案永不重写），本次跳过", len(rep["bundles_skipped"]))
    if rep["expired"]:
        log.info("%s %d 个超期 bundle（%.1f MB）",
                 "已出仓" if args.apply else "可出仓",
                 len(rep["expired"]), rep["expired_bytes"] / 1e6)
    if rep["expiry_blocked"]:
        log.error("有 %d 个超期 bundle 因结论摘要缺失被拒绝出仓（先跑 monthly 固化该月，"
                  "或用 --force 明确接受代价）", len(rep["expiry_blocked"]))

    if args.apply and (n_new or rep["expired"]):
        fp1 = data_footprint()
        log.info("治理后：data/ 共 %.1f MB / %d 个文件（热层 %d 个，冷层 %d 个）",
                 fp1["total_bytes"] / 1e6, fp1["total_files"],
                 fp1["forecast_hot_files"], fp1["forecast_cold_files"])
    elif not args.apply and n_new:
        log.info("dry-run 结束：预计释放 %.1f MB（加 --apply 执行）", saved / 1e6)

    if rep["errors"]:
        for e in rep["errors"][:10]:
            log.error("治理错误：%s", e)
        return len(rep["errors"])
    return 0


def cmd_footprint(args):
    """仓库体量看门狗：报告 data/ 分层占用与 .git 体积，超阈值即非零退出。

    为什么需要它：体积治理的所有机制（bundle 冻结、月度出仓）都只在"按月"这个
    节奏上生效，而一次意外的写入（某源疯狂重试、分片爆炸、误提交大文件）可以在
    一天内把仓库推高几个数量级。没有看门狗，这类事故的表现是"某天起 clone 变慢"，
    等被发现时已经很难收拾——那正是这个项目最想避免的"静默劣化"。

    阈值语义：**超过 --fail-mb 让作业变红并自动开 Issue**（CI 会据此喊人）；
    --warn-mb 只打印提醒。默认值参考 GitHub 的实际约束：仓库软上限 1 GB、
    单次 push 建议 2 GB 以内，超出后 Pages 部署与 clone 都会开始明显变慢。
    """
    from .storage import _root  # 数据根可能被 WEATHER_EVAL_DATA_ROOT 覆盖
    fp = data_footprint()
    root = _root()
    git_dir = PROJECT_ROOT / ".git"
    git_bytes = 0
    if git_dir.is_dir():
        for p in git_dir.rglob("*"):
            try:
                if p.is_file():
                    git_bytes += p.stat().st_size
            except OSError:
                continue

    log.info("data/      ：%.1f MB / %d 个文件", fp["total_bytes"] / 1e6, fp["total_files"])
    log.info("  ├ 热层（当月散装快照）：%.1f MB / %d 个文件",
             fp["forecast_hot_bytes"] / 1e6, fp["forecast_hot_files"])
    log.info("  ├ 冷层（月度 bundle）  ：%.1f MB / %d 个文件",
             fp["forecast_cold_bytes"] / 1e6, fp["forecast_cold_files"])
    log.info("  ├ 观测档案             ：%.1f MB / %d 个文件",
             fp["obs_bytes"] / 1e6, fp["obs_files"])
    log.info("  └ 清单/其他            ：%.1f MB / %d 个文件",
             (fp["manifest_bytes"] + fp["other_bytes"]) / 1e6,
             fp["manifest_files"] + fp["other_files"])
    log.info(".git/      ：%.1f MB（历史不可逆：删掉的文件仍留在历史里）", git_bytes / 1e6)
    # reports/ 此前完全不在看门狗视野里（对抗式审查 P0-3）：它是"每轮全量重写、
    # 且体积是冷层 bundle 三倍"的产物，却无人测量。补上之后，".git 涨了"这类
    # 归因才不会一律落到 data/ 头上。
    rf = reports_footprint(PROJECT_ROOT / "reports")
    log.info("reports/   ：%.1f MB / %d 个文件", rf["total_bytes"] / 1e6, rf["total_files"])
    log.info("  ├ 主报告（每轮重写）  ：%.1f MB", rf["main_bytes"] / 1e6)
    log.info("  ├ 月度归档（只增不改）：%.1f MB / %d 个文件",
             rf["monthly_bytes"] / 1e6, rf["monthly_files"])
    log.info("  ├ 外置数据 JSON（data/） ：%.1f MB", rf["data_bytes"] / 1e6)
    log.info("  └ 静态资源（assets/vendor）：%.1f MB",
             (rf["assets_bytes"] + rf["vendor_bytes"]) / 1e6)
    for item in rf["over_single_threshold"]:
        log.warning("单文件超软阈值：%s %.2f MB", item["file"], item["bytes"] / 1e6)
    log.info("数据根     ：%s", root)
    log.info("可回收量估计：.git %.1f MB − data/ %.1f MB − reports/ %.1f MB ≈ %.1f MB"
             " 历史可回收量（粗估，需用 git count-objects -vH 定量）",
             git_bytes / 1e6, fp["total_bytes"] / 1e6, rf["total_bytes"] / 1e6,
             max(0.0, (git_bytes - fp["total_bytes"] - rf["total_bytes"]) / 1e6))

    rc = 0
    if rf["over_fail"]:
        log.error("reports/ %.1f MB 已超过硬阈值 %.0f MB",
                  rf["total_bytes"] / 1e6, rf["fail_bytes"] / 1e6)
        rc = 1
    elif rf["over_warn"]:
        log.warning("reports/ %.1f MB 已超过提醒阈值 %.0f MB",
                    rf["total_bytes"] / 1e6, rf["warn_bytes"] / 1e6)
    if git_bytes > args.fail_mb * 1e6:
        log.error("仓库体积 %.0f MB 已超过硬阈值 %d MB：请执行 compact "
                  "并考虑 README「历史体积的人工回收」一节", git_bytes / 1e6, args.fail_mb)
        rc = 1
    elif git_bytes > args.warn_mb * 1e6:
        log.warning("仓库体积 %.0f MB 已超过提醒阈值 %d MB（硬阈值 %d MB）",
                    git_bytes / 1e6, args.warn_mb, args.fail_mb)
    if fp["total_files"] > args.max_files:
        log.error("data/ 文件数 %d 已超过阈值 %d：热层可能未按月冻结，请检查 compact 步骤",
                  fp["total_files"], args.max_files)
        rc = 1
    if rc == 0:
        log.info("体积检查通过")
    return rc


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

    p_obs = sub.add_parser("fetch-obs")
    p_obs.add_argument(
        "--source",
        choices=["auto", "eia_data", "cma_data"],
        default="auto",
        help="观测源：auto（默认，按 config 的 obs_sources 编排：主源可用时不惊动"
             "备用源，主源失败/陈旧/截断时自动降级补位）、eia_data（环境气象数据"
             "服务平台，页面内嵌 JSON，一次返回近 24h）、cma_data（中国气象数据网"
             " data.cma.cn 站点实况接口，按 cma_id 的 WMO 站号逐整点抓取）",
    )
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
    pc = sub.add_parser("compact")
    pc.add_argument("--grace-days", type=int, default=None,
                    help="月度冻结宽限期（天）：自然月结束满该天数后，该月快照才合并为"
                         "月度 bundle。缺省取 config 的 compact_grace_days（2）")
    pc.add_argument("--retain-months", type=int, default=None,
                    help="月度 bundle 保留月数，更早的出仓。缺省取 config 的"
                         " compact_retain_months（13）")
    pc.add_argument("--apply", action="store_true",
                    help="真正落盘（冻结 bundle 并删除超期冷层）；默认 dry-run 只报告")
    pc.add_argument("--force", action="store_true",
                    help="出仓时跳过'该月结论摘要必须已固化'的检查（慎用：会让该月"
                         "结论失去可复核性）")
    pf = sub.add_parser("footprint")
    pf.add_argument("--warn-mb", type=int, default=400,
                    help=".git 体积提醒阈值（MB），默认 400")
    pf.add_argument("--fail-mb", type=int, default=900,
                    help=".git 体积硬阈值（MB），超过即非零退出，默认 900")
    pf.add_argument("--max-files", type=int, default=20000,
                    help="data/ 文件数硬阈值，超过即非零退出（热层可能未按月冻结）")
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
        "compact": cmd_compact,
        "footprint": cmd_footprint,
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

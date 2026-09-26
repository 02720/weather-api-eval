"""HTML 报告渲染：Jinja2 模板 + ECharts（仓库内本地副本）。

报告体系（2026-09 再设计）：

- ``reports/index.html``           **主报告（本月至今累积）**。每次 Action 运行覆盖更新，
  GitHub Pages 首页打开即是它 —— 数据在 ``data/`` 里持续积累，报告只是当前累计数据
  的一个"视图"，没必要每次运行留一份文件（那是旧版 reports/runs/ 的做法，已废弃）。
- ``reports/monthly/YYYY-MM.html`` **月度归档**。每月 1 号把上个月的数据冻结成一份
  永久档案；主报告页脚会自动列出所有归档链接。

2026-09 第一性原理重构：页面只回答一个问题——"谁家预报最准？"。结构收敛为
  答案（Hero）→ 证据（一张服务端渲染的总榜）→ 理解（五张各司其职的图）→
  深挖（折叠的方法论与词典）。两张配套约束：

- **内联体积 ∝ 页面画的东西**：``_slim_report`` 只内联图表真正用到的字段
  （完整数据在 data/ 与 data/metrics/ 里随仓库公开）；
- **能服务端渲染的不留给 JS**：总榜表格、走势线 SVG、Hero 实况曲线都在
  Python 侧生成——无 JS 也能读到完整榜单，"榜单静默消失"级事故失去载体。

模板里所有面向读者的措辞都按"小白能看懂"的标准撰写，图表配一行"怎么看"提示；
改动文案请保持同一风格。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..evaluate import PRECIP_SCORE_PARTS, TEMP_SCORE_PARTS
from ..timeutil import now_beijing

logger = logging.getLogger(__name__)

TPL_DIR = Path(__file__).resolve().parent / "templates"
env = Environment(loader=FileSystemLoader(str(TPL_DIR)), autoescape=True)

# 模型的中文显示名：原始 id（如 ecmwf_ifs）对读者不友好，
# 页面上的排行榜、图表图例统一用这里的名字；未收录的模型回退为原始 id。
MODEL_LABELS = {
    "ecmwf_ifs": "欧洲 ECMWF",
    "ncep_gfs_global": "美国 GFS",
    "dwd_icon_global": "德国 ICON",
    "best_match": "OM 最优匹配",
    "cma_grapes_global": "中国 GRAPES",
    "cmc_gem_gdps": "加拿大 GEM",
    "jma_gsm": "日本 GSM",
    "ukmo_global_deterministic_10km": "英国 UKMO",
    "ecmwf_ifs025": "ECMWF IFS 0.25°",
    "ecmwf_aifs025_single": "ECMWF AIFS(AI)",
    "ncep_aigfs025": "NCEP AI-GFS",
    "ncep_hgefs025_ensemble_mean": "NCEP HGEFS 集合",
    "caiyun_v2_6": "彩云天气",
    "qweather_v1": "和风天气",
    "tj_km_fusion": "天机·公里级融合",
    "tj_t2_early": "天机2/DA (T2-Early)",
    "tj_t2": "天机2/ND (T2)",
    "tj_t1": "天机1 (T1)",
    "tj_t1h_ai": "T1H-AI (T1-AI)",
    "fuxi_c88": "伏羲中期 (FuXi-C88)",
    "fuxi_det": "伏羲确定性 (FuXi-Det)",
    "fengwu_ghr_9km": "风乌 GHR-9km",
    "geovis_v1": "中科星图逐小时",
    "accuweather_v1": "AccuWeather 逐小时",
    "msn_v1": "MSN 天气（中国天气网）",
    "cma_ndfs": "CMA-NDFS 智能网格",
    "fengqing_ai": "风清AI模式",
    "cma_public_v1": "中国气象局公众网",
}

# 模型家族分组（选源器按此分组展示）。源多了以后，读者按"这家是什么来头"找源，
# 比在一长串列表里按颜色找快得多；组内展示顺序由前端按排行榜名次排。
# 守卫：tests/test_render.py 的 test_all_models_registered_in_report_layer 会
# 校验 config 的每个模型都登记在 LABELS/COLORS/FAMILIES 三张表里——新源接入
# 漏登任何一张表都会在 CI 红（MSN 曾以原始 id 显示在总榜第 3 名附近，P2-2）。
MODEL_FAMILIES = [
    {"icon": "🌐", "name": "Open-Meteo 全球模式", "models": [
        "best_match",
        "ecmwf_ifs", "ecmwf_ifs025", "ecmwf_aifs025_single",
        "ncep_gfs_global", "ncep_aigfs025", "ncep_hgefs025_ensemble_mean",
        "dwd_icon_global", "cma_grapes_global", "cmc_gem_gdps",
        "jma_gsm", "ukmo_global_deterministic_10km",
    ]},
    {"icon": "🏢", "name": "商业天气 API", "models": [
        "caiyun_v2_6", "qweather_v1", "accuweather_v1", "msn_v1",
    ]},
    {"icon": "🔬", "name": "中科天机", "models": [
        "tj_km_fusion", "tj_t2_early", "tj_t2", "tj_t1", "tj_t1h_ai",
    ]},
    {"icon": "🤖", "name": "AI 气象大模型", "models": [
        "fuxi_c88", "fuxi_det", "fengwu_ghr_9km", "geovis_v1",
    ]},
    {"icon": "🇨🇳", "name": "中国气象局（CMA）", "models": [
        "cma_ndfs", "fengqing_ai", "cma_public_v1",
    ]},
]

# 每个模型的固定配色（图表/排行榜共用，全站一致，方便读者形成"颜色=模型"的记忆）。
MODEL_COLORS = {
    "ecmwf_ifs": "#2563eb",      # 蓝
    "ncep_gfs_global": "#f59e0b",  # 橙
    "dwd_icon_global": "#10b981",  # 绿
    "best_match": "#0ea5e9",     # 天蓝
    "cma_grapes_global": "#dc2626",  # 深红
    "cmc_gem_gdps": "#eab308",   # 黄
    "jma_gsm": "#f97316",        # 橙红
    "ukmo_global_deterministic_10km": "#14b8a6",  # 青
    "ecmwf_ifs025": "#3b82f6",   # 亮蓝
    "ecmwf_aifs025_single": "#6366f1",  # 靛蓝
    "ncep_aigfs025": "#a855f7",  # 亮紫
    "ncep_hgefs025_ensemble_mean": "#d946ef",  # 品红
    "caiyun_v2_6": "#8b5cf6",    # 紫
    "qweather_v1": "#ef4444",    # 红
    "tj_km_fusion": "#e11d48",   # 玫红
    "tj_t2_early": "#22c55e",    # 亮绿
    "tj_t2": "#84cc16",          # 黄绿
    "tj_t1": "#f472b6",          # 粉
    "tj_t1h_ai": "#c084fc",      # 浅紫
    "fuxi_c88": "#0d9488",       # 深青
    "fuxi_det": "#06b6d4",       # 青
    "fengwu_ghr_9km": "#f43f5e", # 玫红偏红
    "geovis_v1": "#6b7280",      # 灰
    "accuweather_v1": "#b45309", # 棕橙（AccuWeather 橙红系，与现有橙/红均拉开明度）
    "msn_v1": "#4d7c0f",         # 橄榄绿（与既有亮绿/黄绿拉开明度与色相）
    "cma_ndfs": "#1e3a8a",       # 深海军蓝（明显暗于既有各蓝）
    "fengqing_ai": "#c026d3",    # 洋红（与亮紫/品红拉开明度）
    "cma_public_v1": "#0e7490",  # 深青（CMA 家族色：与 cma_ndfs 深海军蓝同族；与 #06b6d4/#14b8a6 靠明度拉开）
}


def _r(v, nd: int = 1):
    """四舍五入到 nd 位小数；None 原样返回（缺测在页面显示 —）。"""
    return None if v is None else round(v, nd)


def _slim_report(report_data: dict) -> dict:
    """构建内联进页面的**精简数据视图**。

    旧版把 evaluate 的完整输出（≈1.8MB）原样内联，其中九成字段页面根本不画：
    明细表数据、难度对齐的设计矩阵、每桶全指标……全量数据已在 data/ 与
    data/metrics/ 里随仓库公开，页面只内联"画图真正需要的字段"，并把数值
    取整到显示精度。这是页面体积的第一性约束：**内联体积 ∝ 页面画的东西，
    而不是 ∝ 评估算过的东西**。字段命名取短键（JSON 体积大头是键名）。
    """
    meta = report_data.get("meta", {})
    lbs = report_data.get("leaderboards", {})

    board = []
    for i, row in enumerate(lbs.get("all") or [], 1):
        board.append({
            "m": row.get("model"), "rk": i,
            "s": row.get("score"), "t": row.get("temp_score"),
            "p": row.get("precip_score"), "a2": _r(row.get("acc2")),
            "rmse": _r(row.get("rmse")), "ts": row.get("ts"),
            "ets": row.get("ets"), "lead": row.get("lead_days"),
            "n": row.get("n"), "ci": row.get("ci90"),
            "sig": row.get("sig_vs_top"), "disp": bool(row.get("disputed")),
            "ok": bool(row.get("qualified")),
        })

    def _curve(track: dict) -> dict:
        return {m: {b: _r(v) for b, v in (per or {}).items()}
                for m, per in track.items()}

    trend = {}
    for trk, dims in (report_data.get("score_trend") or {}).items():
        trend[trk] = {dim: _curve(per) for dim, per in dims.items()}

    temp = {}
    for m, buckets in (report_data.get("temp_hourly") or {}).items():
        temp[m] = {b: {k: _r(v, 2) for k, v in d.items()
                       if k in ("acc1", "acc2", "rmse", "mae", "mbe", "r")}
                   for b, d in buckets.items()}

    rain = {}
    for m, buckets in (report_data.get("precip_hourly") or {}).items():
        rain[m] = {b: {k: _r(v, 2) for k, v in d.items()
                       if k in ("acc", "pod", "far", "ts", "ets", "bias",
                                "rmse", "mae")}
                   for b, d in buckets.items()}

    # 热力图：行 = 模型（按总榜名次），列 = 日期；双矩阵（准确率 + 样本数）
    heat_rows = report_data.get("heatmap") or []
    order = [r["m"] for r in board] or [r["model"] for r in heat_rows]
    dates = sorted({r["date"] for r in heat_rows})
    heat = {"dates": [d[5:] for d in dates],
            "models": [r for r in order if any(x["model"] == r for x in heat_rows)],
            "acc": [], "n": []}
    heat_idx = {(r["model"], r["date"]): r for r in heat_rows}
    for m in heat["models"]:
        heat["acc"].append([_r(heat_idx[(m, d)]["acc2"]) if (m, d) in heat_idx else None
                            for d in dates])
        heat["n"].append([heat_idx[(m, d)]["n"] if (m, d) in heat_idx else None
                          for d in dates])

    # 实况对比：每站最近 72 小时，全部压成数值数组（键名只出现一次）
    WIN = 72
    ts = {}
    for st, payload in (report_data.get("timeseries") or {}).items():
        obs = payload.get("obs") or []
        obs = obs[-WIN:]
        models_ts = payload.get("models") or {}
        src = {}
        for m, series in models_ts.items():
            series = series[-WIN:]
            src[m] = {"t": [_r(x.get("temp")) for x in series],
                      "r": [_r(x.get("rain"), 2) for x in series]}
        ts[st] = {"t": [x["t"][5:16] for x in obs],
                  "obs": {"t": [_r(x.get("temp")) for x in obs],
                          "r": [_r(x.get("rain"), 2) for x in obs]},
                  "src": src}

    # 分站对比：每站每源只留主要指标
    stations = {}
    for st, per_model in (report_data.get("per_station") or {}).items():
        stations[st] = {}
        for m, dims in per_model.items():
            t, p = dims.get("temp") or {}, dims.get("precip") or {}
            stations[st][m] = {
                "t": {k: _r(t.get(k), 2) for k in ("acc2", "rmse", "mbe")},
                "r": {k: _r(p.get(k), 2) for k in ("ts", "ets")},
            }

    slim_meta = {
        "period_label": meta.get("period_label"),
        "is_monthly": bool(meta.get("is_monthly")),
        "start": meta.get("start"), "end": meta.get("end"),
        "models": meta.get("models"), "stations": meta.get("stations"),
        "rain_threshold_mm": meta.get("rain_threshold_mm"),
        "rain_daily_threshold_mm": meta.get("rain_daily_threshold_mm"),
        "min_sample": meta.get("min_sample"),
    }
    return {"meta": slim_meta, "board": board, "trend": trend,
            "temp": temp, "rain": rain, "heat": heat, "ts": ts,
            "stations": stations, "coverage": {
                k: report_data.get("coverage", {}).get(k)
                for k in ("coverage_pct", "first_obs", "last_obs")}}


def _sparkline_svg(values: list, med: float | None = None,
                   w: int = 110, h: int = 30, color: str = "#205fa7") -> str:
    """服务端生成"得分随时效衰减"的行内走势 SVG（无 JS、无全局状态）。

    values 为按提前 1..N 天排列的综合分，None = 该档样本不足（折线在此断开）。
    med 为该榜全体源的中位分（灰色虚线参考线）——旧版这条线由前端 JS 按
    "当前轨道"注入全局变量，出现过"拆了声明忘了改引用→榜单整表消失"的
    事故（fc7c402 回归）；改为服务端渲染后，这条 bug 类别被整类消灭。
    """
    pts = [(i, v) for i, v in enumerate(values) if v is not None]
    if not pts:
        return ""
    xs = [p[0] for p in pts]
    vs = [p[1] for p in pts]
    lo, hi = min(vs + ([med] if med is not None else [])), \
        max(vs + ([med] if med is not None else []))
    span = (hi - lo) or 1.0
    pad = 3.0
    px = lambda i: round(pad + i * (w - 2 * pad) / max(len(values) - 1, 1), 1)
    py = lambda v: round(h - pad - (v - lo) * (h - 2 * pad) / span, 1)
    # None 断开处拆成多段 polyline
    segs, cur = [], [pts[0]]
    for prev, item in zip(pts, pts[1:]):
        if item[0] == prev[0] + 1:
            cur.append(item)
        else:
            segs.append(cur)
            cur = [item]
    segs.append(cur)
    lines = "".join(
        f'<polyline fill="none" stroke="{color}" stroke-width="1.6" '
        f'stroke-linejoin="round" stroke-linecap="round" '
        f'points="{" ".join(f"{px(i)},{py(v)}" for i, v in seg)}"/>' for seg in segs)
    med_line = ""
    if med is not None:
        med_line = (f'<line x1="{pad}" x2="{w - pad}" y1="{py(med)}" y2="{py(med)}" '
                    f'stroke="#94a3b8" stroke-width="1" stroke-dasharray="3 3"/>')
    dot = (f'<circle cx="{px(pts[-1][0])}" cy="{py(pts[-1][1])}" r="2.2" '
           f'fill="{color}"/>')
    return (f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
            f'aria-hidden="true">{med_line}{lines}{dot}</svg>')


def _hero_curve_svg(report_data: dict, w: int = 1440, h: int = 220) -> str:
    """Hero 背景装饰：第一站最近 72 小时**实况气温**的平滑曲线。

    报告的全部结论都建立在"拿实况说话"上——把真实观测画进版面，装饰即方法论。
    温度是逐小时原值，跨日夜的锯齿正是"天气"本身，不做平滑处理。
    """
    ts = report_data.get("timeseries") or {}
    if not ts:
        return ""
    first = next(iter(ts.values()))
    temps = [x.get("temp") for x in (first.get("obs") or []) if x.get("temp") is not None]
    if len(temps) < 24:
        return ""
    lo, hi = min(temps), max(temps)
    span = (hi - lo) or 1.0
    n = len(temps)
    pts = " ".join(
        f"{round(i * w / (n - 1), 1)},{round(h - 30 - (v - lo) * (h - 70) / span, 1)}"
        for i, v in enumerate(temps))
    return (f'<svg viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'aria-hidden="true"><polyline fill="none" stroke="rgba(125,211,252,.55)" '
            f'stroke-width="2" points="{pts}"/></svg>')


def _reports_root() -> Path:
    """报告输出根目录；每次调用动态读环境变量，便于测试用 monkeypatch 重定向。"""
    return Path(os.environ.get(
        "WEATHER_EVAL_REPORTS_ROOT",
        Path(__file__).resolve().parents[3] / "reports",
    ))


def _list_archives(root: Path) -> list[str]:
    """扫描月度归档目录，返回月份列表（新→旧，如 ["2026-08", "2026-07"]）。"""
    monthly_dir = root / "monthly"
    if not monthly_dir.exists():
        return []
    return sorted((p.stem for p in monthly_dir.glob("*.html")), reverse=True)


def _js_json(obj) -> str:
    """序列化为紧凑 JSON 并转义 </：防止数据中的 </script> 提前闭合内联脚本。

    separators 显式去掉默认的 ", "/": " 空格——内联 JSON 达 MB 级，紧凑分隔符
    白省 10~15% 页面体积（P3-4）。对所有内联进 <script> 的 JSON 统一走这里
    （与 storage 的原子写一样，是"写进仓库的每一份产物都要过"的基础防护）。"""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")) \
        .replace("</", "<\\/")


def _atomic_write_text(path: Path, text: str) -> None:
    """临时文件 + 原子 rename 落盘，避免中途失败留下半截 HTML
    （半文件会被 git-auto-commit 收走并部署到 Pages）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)   # mkstemp 默认 0600，恢复常规读权限
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _score_parts_rows(parts) -> list[dict]:
    """把得分构成表转成模板可渲染的行（权重百分比 + 白话换算说明）；函数本体不序列化。"""
    return [{"key": k, "weight": w, "pct": round(w * 100), "label": label, "map": mp}
            for (k, w, label, mp, _fn) in parts]


def render_report_html(report_data: dict, title: str | None = None,
                       base: str = "./", archives: list[str] | None = None,
                       station_labels: dict[str, str] | None = None) -> str:
    """渲染一份报告 HTML。

    base：本报告文件到 reports/ 根的相对前缀（根目录 ./ ，monthly/ 子目录 ../ ），
          用于定位 vendor/echarts.min.js 与归档链接。
    archives：月度归档月份列表，显示在页脚"月度归档"区。
    station_labels：站点 id → 中文名（如 wuzhou → 梧州气象站），缺省时页面显示 id。
    """
    tpl = env.get_template("report.html.j2")
    slim_json = _js_json(_slim_report(report_data))
    labels_json = _js_json(MODEL_LABELS)
    colors_json = _js_json(MODEL_COLORS)

    # 排行榜每行的"得分随时效衰减"走势线在服务端算好（小时口径，1~16 天）：
    # 中位参考线也在此计算——旧版这条线由前端 JS 按"当前轨道"注入全局变量，
    # 出现过"拆了声明忘了改引用 → 榜单整表消失"的事故（fc7c402 回归）；
    # 改为服务端渲染后，这一 bug 类别被整类消灭。
    trend_hourly = ((report_data.get("score_trend") or {}).get("hourly") or {})
    overall_trend = trend_hourly.get("overall") or {}
    lb_all_rows = (report_data.get("leaderboards") or {}).get("all") or []
    _med_vals = sorted(v for per in overall_trend.values()
                       for v in per.values() if v is not None)
    spark_med = _med_vals[len(_med_vals) // 2] if _med_vals else None
    spark_svg = {r["model"]: _sparkline_svg(
        [overall_trend.get(r["model"], {}).get(f"{i}d") for i in range(1, 17)],
        spark_med)
        for r in lb_all_rows}
    hero_curve = _hero_curve_svg(report_data)

    return tpl.render(
        report=report_data,
        slim_json=slim_json,
        spark_med=spark_med,
        spark_svg=spark_svg,
        hero_curve=hero_curve,
        model_labels=MODEL_LABELS,
        model_colors=MODEL_COLORS,
        model_labels_json=labels_json,
        model_colors_json=colors_json,
        score_temp_parts=_score_parts_rows(TEMP_SCORE_PARTS),
        score_precip_parts=_score_parts_rows(PRECIP_SCORE_PARTS),
        station_labels=station_labels or {},
        station_labels_json=_js_json(station_labels or {}),
        archives=archives or [],
        title=title or "天气预报准确度检验报告",
        generated_at=now_beijing().strftime("%Y-%m-%d %H:%M"),
        base=base,
    )


def write_health_page(html: str) -> Path:
    """写 reports/health.html（源健康度看板，P1-8）。与主报告同根、同样原子写。"""
    root = _reports_root()
    root.mkdir(parents=True, exist_ok=True)
    out = root / "health.html"
    _atomic_write_text(out, html)
    return out


def write_live_report(report_data: dict, station_labels: dict[str, str] | None = None) -> Path:
    """写主报告：覆盖 reports/index.html（Pages 首页）。每次运行都基于当月全部数据重算。"""
    root = _reports_root()
    root.mkdir(parents=True, exist_ok=True)
    out = root / "index.html"
    _atomic_write_text(out, render_report_html(report_data, base="./",
                                               archives=_list_archives(root),
                                               station_labels=station_labels))
    return out


def write_monthly_report(report_data: dict, station_labels: dict[str, str] | None = None,
                         force: bool = False) -> Path:
    """写月度归档：reports/monthly/YYYY-MM.html，写后不再变动（冻结档案）。

    已存在的归档默认拒绝重写——防止手动 dispatch 或归档条件重复触发悄悄改写
    冻结数据；确需重建时显式传 force=True。
    归档页脚的归档列表包含自身——读者在任意归档页应看到完整的归档导航。
    """
    root = _reports_root()
    monthly_dir = root / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    month = report_data["meta"]["period_label"]
    out = monthly_dir / f"{month}.html"
    if out.exists() and not force:
        logger.warning("月度归档 %s 已存在，跳过重写（冻结档案不再变动；确需重建请使用 --force）", out)
        return out
    archives = sorted({*_list_archives(root), month}, reverse=True)
    _atomic_write_text(out, render_report_html(
        report_data, title=f"{month} 月度归档 · 天气预报准确度检验报告",
        base="../", archives=archives,
        station_labels=station_labels))
    return out

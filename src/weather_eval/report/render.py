"""HTML 报告渲染：Jinja2 模板 + ECharts（仓库内本地副本）。

报告体系（2026-08 重设计，面向非专业读者）：

- ``reports/index.html``           **主报告（本月至今累积）**。每次 Action 运行覆盖更新，
  GitHub Pages 首页打开即是它 —— 数据在 ``data/`` 里持续积累，报告只是当前累计数据
  的一个"视图"，没必要每次运行留一份文件（那是旧版 reports/runs/ 的做法，已废弃）。
- ``reports/monthly/YYYY-MM.html`` **月度归档**。每月 1 号把上个月的数据冻结成一份
  永久档案；主报告页脚会自动列出所有归档链接。

模板里所有面向读者的措辞都按"小白能看懂"的标准撰写，图表配"怎么看"提示，
页面底部内置名词词典；改动文案请保持同一风格。
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..timeutil import now_beijing

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
}

# 模型家族分组（选源器按此分组展示）。源多了以后，读者按"这家是什么来头"找源，
# 比在一长串列表里按颜色找快得多；组内展示顺序由前端按排行榜名次排。
MODEL_FAMILIES = [
    {"icon": "🌐", "name": "Open-Meteo 全球模式", "models": [
        "best_match",
        "ecmwf_ifs", "ecmwf_ifs025", "ecmwf_aifs025_single",
        "ncep_gfs_global", "ncep_aigfs025", "ncep_hgefs025_ensemble_mean",
        "dwd_icon_global", "cma_grapes_global", "cmc_gem_gdps",
        "jma_gsm", "ukmo_global_deterministic_10km",
    ]},
    {"icon": "🏢", "name": "商业天气 API", "models": [
        "caiyun_v2_6", "qweather_v1", "accuweather_v1",
    ]},
    {"icon": "🔬", "name": "中科天机", "models": [
        "tj_km_fusion", "tj_t2_early", "tj_t2", "tj_t1", "tj_t1h_ai",
    ]},
    {"icon": "🤖", "name": "AI 气象大模型", "models": [
        "fuxi_c88", "fuxi_det", "fengwu_ghr_9km", "geovis_v1",
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
}


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
    """序列化为 JSON 并转义 </：防止数据中的 </script> 提前闭合内联脚本。
    对所有内联进 <script> 的 JSON 统一走这里（与 storage 的原子写一样，
    是"写进仓库的每一份产物都要过"的基础防护）。"""
    return json.dumps(obj, ensure_ascii=False).replace("</", "<\\/")


def _atomic_write_text(path: Path, text: str) -> None:
    """临时文件 + 原子 rename 落盘，避免中途失败留下半截 HTML
    （半文件会被 git-auto-commit 收走并部署到 Pages）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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
    report_json = _js_json(report_data)
    labels_json = _js_json(MODEL_LABELS)
    colors_json = _js_json(MODEL_COLORS)
    return tpl.render(
        report=report_data,
        report_json=report_json,
        model_labels=MODEL_LABELS,
        model_labels_json=labels_json,
        model_colors_json=colors_json,
        model_families_json=_js_json(MODEL_FAMILIES),
        station_labels=station_labels or {},
        station_labels_json=_js_json(station_labels or {}),
        archives=archives or [],
        title=title or "天气预报准确度检验报告",
        generated_at=now_beijing().strftime("%Y-%m-%d %H:%M"),
        base=base,
    )


def write_live_report(report_data: dict, station_labels: dict[str, str] | None = None) -> Path:
    """写主报告：覆盖 reports/index.html（Pages 首页）。每次运行都基于当月全部数据重算。"""
    root = _reports_root()
    root.mkdir(parents=True, exist_ok=True)
    out = root / "index.html"
    _atomic_write_text(out, render_report_html(report_data, base="./",
                                               archives=_list_archives(root),
                                               station_labels=station_labels))
    return out


def write_monthly_report(report_data: dict, station_labels: dict[str, str] | None = None) -> Path:
    """写月度归档：reports/monthly/YYYY-MM.html，写后不再变动（冻结档案）。"""
    root = _reports_root()
    monthly_dir = root / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    month = report_data["meta"]["period_label"]
    out = monthly_dir / f"{month}.html"
    _atomic_write_text(out, render_report_html(
        report_data, title=f"{month} 月度归档 · 天气预报准确度检验报告",
        base="../", archives=_list_archives(root),
        station_labels=station_labels))
    return out

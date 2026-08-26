"""HTML 报告渲染：Jinja2 模板 + ECharts（仓库内本地副本）。"""
from __future__ import annotations

import json
import os
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ..timeutil import now_beijing, ymd

TPL_DIR = Path(__file__).resolve().parent / "templates"
REPORTS_ROOT = Path(os.environ.get(
    "WEATHER_EVAL_REPORTS_ROOT",
    Path(__file__).resolve().parents[3] / "reports",
))
env = Environment(loader=FileSystemLoader(str(TPL_DIR)), autoescape=True)


def render_report_html(report_data: dict, title: str | None = None, base: str = "./") -> str:
    """base：相对本报告文件到 reports/ 根的路径前缀（根目录 ./ ，子目录 ../ ）。"""
    tpl = env.get_template("report.html.j2")
    # 内联 JSON 时转义 </ 防止 </script> 提前闭合导致脚本注入/提前终止
    report_json = json.dumps(report_data, ensure_ascii=False).replace("</", "<\\/")
    return tpl.render(
        report=report_data,
        report_json=report_json,
        title=title or "天气预报 API 准确度评估报告",
        generated_at=now_beijing().strftime("%Y-%m-%d %H:%M"),
        base=base,
    )


def write_run_report(report_data: dict) -> Path:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    runs_dir = REPORTS_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    period = report_data["meta"]["period_label"]
    stamp = now_beijing().strftime("%Y%m%d-%H%M")
    out = runs_dir / f"{period}-{stamp}.html"
    out.write_text(render_report_html(report_data, base="../"), encoding="utf-8")
    latest = REPORTS_ROOT / "latest.html"
    latest.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    return out


def write_monthly_report(report_data: dict) -> Path:
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    monthly_dir = REPORTS_ROOT / "monthly"
    monthly_dir.mkdir(parents=True, exist_ok=True)
    month = report_data["meta"]["period_label"]
    out = monthly_dir / f"{month}.html"
    out.write_text(render_report_html(report_data, title=f"月度评估报告 {month}", base="../"), encoding="utf-8")
    return out


def write_index() -> Path:
    """生成报告门户 index.html：列出 runs 与 monthly 下的报告。"""
    REPORTS_ROOT.mkdir(parents=True, exist_ok=True)
    runs, monthly = [], []
    runs_dir = REPORTS_ROOT / "runs"
    if runs_dir.exists():
        for p in sorted(runs_dir.glob("*.html"), reverse=True):
            runs.append(p.name)
    monthly_dir = REPORTS_ROOT / "monthly"
    if monthly_dir.exists():
        for p in sorted(monthly_dir.glob("*.html"), reverse=True):
            monthly.append(p.name)
    tpl = env.get_template("index.html.j2")
    html = tpl.render(runs=runs, monthly=monthly, generated_at=now_beijing().strftime("%Y-%m-%d %H:%M"))
    out = REPORTS_ROOT / "index.html"
    out.write_text(html, encoding="utf-8")
    return out

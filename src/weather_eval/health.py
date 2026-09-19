"""源健康度看板（对抗式审查 P1-8）。

问题：`eval.yml` 里 9 个抓取步骤全部 `continue-on-error: true`，失败只写
`log.error`。于是一个源连续 30 轮抓不到数据，唯一信号是"作业状态偶尔变黄"——
而对于"预报快照错过即无法追补"的项目，这等于**用静默换取了样本期的空洞**。

本模块把"这家还活着吗"变成一个可以一眼看出的页面（reports/health.html）与一个
可以失败的退出码：

- 每个源的**最近成功时刻**（issue 与 fetched_at 两个口径都给：issue 陈旧说明
  服务端不再更新，fetched_at 陈旧说明我们抓不到）；
- 快照数、残缺数、起报锚点语义构成；
- **陈旧判定**：曾经有数据、但最新快照已超过 stale_hours 未更新 → 标红。
  从未有过数据的源只标"未接入"，**不让它把构建搞红**——可选源（需 Secret）
  没配是正常状态，不是故障。

这样"静默死亡"变成"一眼可见 + 构建失败 + 自动开 Issue"。
"""
from __future__ import annotations

from datetime import timedelta

from .snapshot_meta import ISSUE_SOURCE_LABELS, snapshot_complete
from .timeutil import now_beijing, parse_iso


def of(snapshots: dict, station_ids: list[str], models: list[str]) -> dict:
    """汇总每个源的接入健康度。

    snapshots: {(station_id, model): [snapshot, ...]}
    返回 {model: {...}}，字段见 docstring 顶部。
    """
    out: dict[str, dict] = {}
    for m in models:
        snaps: list[dict] = []
        per_station: dict[str, int] = {}
        for sid in station_ids:
            lst = snapshots.get((sid, m)) or []
            per_station[sid] = len(lst)
            snaps.extend(lst)
        if not snaps:
            out[m] = {
                "status": "no_data", "snapshots": 0, "truncated": 0,
                "stations_covered": 0, "n_stations": len(station_ids),
                "last_issue": None, "last_fetched": None,
                "issue_sources": {}, "stale": False,
            }
            continue
        issues = [s.get("issue_iso") for s in snaps if s.get("issue_iso")]
        fetched = [s.get("fetched_at_bj") for s in snaps if s.get("fetched_at_bj")]
        srcs: dict[str, int] = {}
        truncated = 0
        for s in snaps:
            if not snapshot_complete(s):
                truncated += 1
            key = str(s.get("issue_source") or "unknown")
            srcs[key] = srcs.get(key, 0) + 1
        out[m] = {
            "status": "ok",
            "snapshots": len(snaps),
            "truncated": truncated,
            "stations_covered": sum(1 for v in per_station.values() if v),
            "n_stations": len(station_ids),
            "last_issue": max(issues) if issues else None,
            "last_fetched": max(fetched) if fetched else None,
            # 有多少份快照连抓取时刻都没有：这个数字应当随新契约逐步归零，
            # 它是"可审计性是否已经覆盖全部存档"的直接指标
            "without_fetched_at": sum(1 for s in snaps if not s.get("fetched_at_bj")),
            "issue_sources": {k: v for k, v in sorted(srcs.items())},
            "stale": False,
            "per_station": per_station,
        }
    return out


def evaluate_staleness(health: dict, stale_hours: int = 30) -> dict:
    """按"最新一次成功抓取距今多久"判定陈旧。

    以 `fetched_at_bj` 为准；老存档没有该字段时退回 `issue_iso`（口径更松，
    但总比不判定好）。`stale_hours` 默认 30 小时：定时任务是每 7 小时一轮，
    30 小时 = 连续 4 轮没成功，已不是偶发抖动。

    只对**曾经有数据**的源判陈旧：从未接入的源（缺 Secret、尚未接线）
    不算故障——把正常状态当成故障会让告警迅速失去意义。
    """
    now = now_beijing()
    limit = timedelta(hours=stale_hours)
    out: dict[str, dict] = {}
    for m, h in health.items():
        if h["status"] == "no_data":
            out[m] = {**h, "stale": False, "age_hours": None, "unavailable": False}
            continue
        ref = h.get("last_fetched") or h.get("last_issue")
        age = None
        if ref:
            try:
                age = (now - parse_iso(str(ref)[:16])).total_seconds() / 3600.0
            except (ValueError, TypeError):
                age = None
        stale = bool(age is not None and age > stale_hours)
        out[m] = {**h, "stale": stale,
                  "age_hours": round(age, 1) if age is not None else None,
                  "unavailable": age is None}
    return out


def stale_sources(health: dict) -> list[str]:
    """陈旧源名单（供 CLI 决定退出码与 CI 决定是否开 Issue）。"""
    return sorted(m for m, h in health.items() if h.get("stale"))


def render_health_html(health: dict, meta: dict, stale_hours: int) -> str:
    """渲染源健康度看板（自包含 HTML，无外部依赖，可离线打开）。"""
    from html import escape

    rows = []
    ordered = sorted(health.items(),
                     key=lambda kv: (kv[1]["status"] != "ok",
                                     -(kv[1].get("age_hours") or 0)))
    for m, h in ordered:
        if h["status"] != "ok":
            badge, cls = "未接入", "muted"
            age = "—"
        elif h.get("stale"):
            badge, cls = "陈旧", "bad"
            age = f"{h['age_hours']} h"
        elif h.get("unavailable"):
            badge, cls = "无抓取时刻", "warn"
            age = "—"
        else:
            badge, cls = "正常", "ok"
            age = f"{h['age_hours']} h"
        srcs = "、".join(f"{ISSUE_SOURCE_LABELS.get(k, k)}({v})"
                         for k, v in (h.get("issue_sources") or {}).items()) or "—"
        rows.append(
            "<tr>"
            f"<td>{escape(str(m))}</td>"
            f'<td><span class="b {cls}">{badge}</span></td>'
            f"<td class=num>{h['snapshots']}</td>"
            f"<td class=num>{h.get('stations_covered', 0)}/{h.get('n_stations', 0)}</td>"
            f"<td class=num>{h.get('truncated', 0)}</td>"
            f"<td class=num>{h.get('without_fetched_at', 0)}</td>"
            f"<td class=num>{age}</td>"
            f"<td>{escape(str(h.get('last_issue') or '—'))}</td>"
            f"<td class=src>{escape(srcs)}</td>"
            "</tr>")

    stale = stale_sources(health)
    alert = (f'<p class="alert">⚠️ 有 {len(stale)} 个源已超过 {stale_hours} 小时未成功抓取：'
             f'{escape("、".join(stale))}。预报快照错过即无法追补，请尽快核查。</p>'
             if stale else
             f'<p class="fine">✅ 所有已接入的源都在 {stale_hours} 小时内成功抓取过。</p>')

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>预报源健康度 · {escape(str(meta.get('period_label', '')))}</title>
<style>
:root{{--ink:#1c1f23;--muted:#6b7280;--line:#e3e6ea;--ok:#0f7b4f;--bad:#b3261e;
--warn:#8a5a00;--bg:#fbfbfa;--card:#fff}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 ui-sans-serif,system-ui,"Noto Sans SC",sans-serif}}
main{{max-width:1080px;margin:0 auto;padding:40px 20px 80px}}
h1{{font-size:24px;margin:0 0 4px}}
p.sub{{color:var(--muted);margin:0 0 24px;font-size:13px}}
.alert{{background:#fdecea;border-left:3px solid var(--bad);padding:12px 14px;
margin:0 0 20px;border-radius:0 6px 6px 0}}
.fine{{background:#eaf6ef;border-left:3px solid var(--ok);padding:12px 14px;
margin:0 0 20px;border-radius:0 6px 6px 0}}
table{{width:100%;border-collapse:collapse;background:var(--card);
border:1px solid var(--line);font-size:13.5px}}
th,td{{padding:9px 11px;border-bottom:1px solid var(--line);text-align:left;
vertical-align:top}}
th{{font-size:12px;color:var(--muted);font-weight:600;
background:#f6f7f8;position:sticky;top:0}}
td.num{{text-align:right;font-variant-numeric:tabular-nums}}
td.src{{font-size:12px;color:var(--muted);max-width:280px}}
tr:last-child td{{border-bottom:none}}
.b{{display:inline-block;padding:1px 8px;border-radius:99px;font-size:12px;
border:1px solid transparent}}
.b.ok{{color:var(--ok);background:#eaf6ef;border-color:#bfe3d0}}
.b.bad{{color:var(--bad);background:#fdecea;border-color:#f3c6c1}}
.b.warn{{color:var(--warn);background:#fdf5e6;border-color:#efd9a8}}
.b.muted{{color:var(--muted);background:#f1f2f4;border-color:#dfe2e6}}
footer{{margin-top:28px;color:var(--muted);font-size:12px}}
</style></head>
<body><main>
<h1>预报源健康度</h1>
<p class="sub">生成于 {escape(str(meta.get('generated_at', '')))} ·
评估窗口 {escape(str(meta.get('start', '')))} ~ {escape(str(meta.get('end', '')))} ·
判定阈值 {stale_hours} 小时（定时任务每 7 小时一轮，超过 4 轮失败即视为陈旧）</p>
{alert}
<table>
<thead><tr>
<th>预报源</th><th>状态</th><th>快照数</th><th>覆盖站点</th><th>残缺</th>
<th>缺抓取时刻</th><th>距上次成功</th><th>最近起报</th><th>起报锚点语义构成</th>
</tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
<footer>「残缺」= 快照契约标了 complete=false（分片未取全）；
「缺抓取时刻」= 该存档早于 2026-09 的元数据契约、没有 fetched_at 字段，
属于历史存量，随归档自然递减。</footer>
</main></body></html>
"""

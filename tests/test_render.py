from datetime import datetime, timedelta

from weather_eval import storage
from weather_eval.evaluate import build_report
from weather_eval.report.render import render_report_html, write_live_report, write_monthly_report
from weather_eval.timeutil import iso


def _populate(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 1, 0, 0)
    # 30 天观测（用确定性模式）
    obs = []
    for h in range(30 * 24):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 5),
                    "rain": 1.0 if h % 12 == 0 else 0.0})
    storage.save_obs("s1", obs)
    # 多个起报快照，使按天 offset 与逐小时桶都有足够样本
    for day in range(0, 6):
        issue = start + timedelta(days=day, hours=0)
        times = [iso(start + timedelta(days=day, hours=hh)) for hh in range(24 * 4)]
        snap = {
            "issue_iso": iso(issue), "station_id": "s1", "source": "open-meteo",
            "models": ["ecmwf_ifs"], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
            "hourly_time": times,
            "data": {"ecmwf_ifs": {
                "temperature_2m": [20.0 + (day + hh % 5) % 5 + 0.5 for hh in range(24 * 4)],
                "precipitation": [1.0 if hh % 12 == 0 else 0.0 for hh in range(24 * 4)],
            }},
        }
        storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap)


def test_render_monthly_html_nonempty(tmp_path, monkeypatch):
    _populate(tmp_path, monkeypatch)
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08", is_monthly=True)
    html = render_report_html(data, title="月度报告测试")

    assert "echarts.min.js" in html
    assert "const report = {" in html
    # 评分卡应有真实数值（非全 None）
    assert data["scorecard"]["ecmwf_ifs"]["temp_24h"]["n"] > 0
    # 分时效排行榜无条件计算（主报告的表格排行榜/冠军横幅依赖它）
    # 分时效榜按分辨率分开命名（2026-09 跨分辨率重构）
    assert data["leaderboards"]["hourly:1d"] and data["leaderboards"]["hourly:1d"][0]["score"] is not None
    assert data["leaderboards"]["daily:1d"] is not None
    # 全时效总榜同样无条件计算（主报告冠军横幅默认读总榜）
    assert data["leaderboards"]["all"] and data["leaderboards"]["all"][0]["score"] is not None
    assert isinstance(data["heatmap"], list) and len(data["heatmap"]) > 0
    # 得分趋势（§02 衰减曲线的数据源）与全指标图表容器
    assert data["score_trend"]["hourly"]["overall"]["ecmwf_ifs"]["1d"] is not None
    assert data["score_trend"]["daily"]["overall"]["ecmwf_ifs"]["1d"] is not None
    # 2026-09 重构后的页面结构：五张图 + 服务端渲染的榜单表格
    for el_id in ("chartDecay", "chartWx", "chartHeat", "chartTs", "chartStation"):
        assert f'id="{el_id}"' in html, f"缺图表容器 {el_id}"
    assert 'id="lbBody"' in html and 'id="lbTable"' in html
    # 榜单行必须在服务端渲染出来（无 JS 可读是硬约束），且含温度计与走势线
    assert html.count('<tr class="') >= 1 and "thermo" in html and "<svg" in html
    # 评分构成表（服务端渲染）与"±2°C 准确率"白话标签
    assert "综合分怎么算" in html and "±2°C 准确率" in html
    # 2026-10 呈现重构：换算白话拆成「主式（括注）」+ 份额条，起报锚点表改成
    # "挂旗在前 + 按语义归组"，逐行重复的括注长句不再出现（旧版一句重复 27 遍，
    # 且 nowrap 把表顶出卡片、body 的 overflow-x:clip 直接把溢出裁掉）
    assert "w-map" in html and "w-bar" in html
    assert "anchor-cell" in html and "rowspan=" in html   # 按语义归组的第二张表
    assert "随归档递减" not in html
    # 内联数据必须是裁剪视图（_slim_report），不再内联全量评估输出
    assert '"board"' in html and '"scorecard"' not in html


def test_read_board_note_always_includes_sitting_champion(tmp_path, monkeypatch):
    """读榜须知与冠军横幅必须自洽：权重敏感性只列前二时不得丢掉现榜冠军。

    回归（2026-10-10 实测）：冠军伏羲中期在 500 次权重扰动中夺冠 25.0%、排第三，
    旧模板截断前二（MSN 48.4% / ECMWF 26.6%）且标题写作"冠军归属"，同一张卡片
    上下两处各说各话。披露的宾语是"现榜冠军有多稳"，故展示集合 = 频率前二 ∪
    现榜冠军——冠军一次未夺冠（不在列表里）时频率记 0% 也不得省略。
    """
    _populate(tmp_path, monkeypatch)
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5,
           "sensitivity_runs": 100}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08")
    assert data["leaderboards"]["all"][0]["model"] == "ecmwf_ifs"

    def note_of(html):
        return html.split("读榜须知", 1)[1].split("看完整榜单", 1)[0]

    # 情形 1：冠军有夺冠频率但不在前二（排第三）→ 必须补进且标注"现榜冠军"
    data["meta"]["weight_sensitivity"]["champions"] = [
        {"model": "msn_v1", "pct": 48.4},
        {"model": "ecmwf_aifs025_single", "pct": 26.6},
        {"model": "ecmwf_ifs", "pct": 25.0}]
    note = note_of(render_report_html(data))
    assert "ECMWF IFS HRES 9km 25.0%" in note and "现榜冠军" in note

    # 情形 2：冠军一次未夺冠（敏感性列表里没有它）→ 频率记 0%，不得省略
    data["meta"]["weight_sensitivity"]["champions"] = [
        {"model": "msn_v1", "pct": 48.4},
        {"model": "ecmwf_aifs025_single", "pct": 26.6}]
    note = note_of(render_report_html(data))
    assert "ECMWF IFS HRES 9km 0%（一次未夺冠）" in note and "现榜冠军" in note

    # 情形 3：冠军就在前二（常见情形）→ 不重复追加，总数仍为两条
    data["meta"]["weight_sensitivity"]["champions"] = [
        {"model": "ecmwf_ifs", "pct": 51.4},
        {"model": "msn_v1", "pct": 33.0}]
    note = note_of(render_report_html(data))
    assert "现榜冠军 ECMWF IFS HRES 9km 51.4%" in note
    assert note.count("ECMWF IFS HRES 9km") == 1


def test_inline_json_is_slim(tmp_path, monkeypatch):
    """2026-09 体积守卫：内联 JSON ∝ 页面画的东西，而不是 ∝ 评估算过的东西。

    旧版把完整评估输出（≈1.8MB）原样内联；重构后只内联图表用字段。
    本测试用"多源大窗口"的真实形状数据锁住上限——任何把全量数据重新塞回
    页面的改动都会在这里变红。"""
    from weather_eval.report.render import _slim_report

    _populate(tmp_path, monkeypatch)
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08")
    slim = _slim_report(data)
    import json as _json
    size = len(_json.dumps(slim, ensure_ascii=False))
    assert size < 60_000, f"内联视图 {size:,} 字节，超出预算（检查 _slim_report 是否泄漏了全量字段）"
    # 页面产物整体也设上限（真实数据 ≈330KB；放宽到 600KB 覆盖模板自身变化）
    html = render_report_html(data)
    assert len(html.encode()) < 600_000


def test_write_live_report_overwrites_index(tmp_path, monkeypatch):
    """主报告每次运行覆盖更新 reports/index.html（不再往 runs/ 堆文件），并带归档链接。"""
    _populate(tmp_path, monkeypatch)
    monkeypatch.setenv("WEATHER_EVAL_REPORTS_ROOT", str(tmp_path / "reports"))
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08")
    # 分时效排行榜对非月度报告也必须可用（主报告冠军横幅依赖它；默认读总榜 "all"）
    # 分时效榜按分辨率分开命名（2026-09 跨分辨率重构）
    assert data["leaderboards"]["hourly:1d"] and data["leaderboards"]["hourly:1d"][0]["score"] is not None
    assert data["leaderboards"]["daily:1d"] is not None
    assert data["leaderboards"]["all"] and data["leaderboards"]["all"][0]["score"] is not None

    out = write_live_report(data, station_labels={"s1": "一号站"})
    assert out.name == "index.html"
    html = out.read_text(encoding="utf-8")
    assert "一号站" in html                 # 站点中文名生效
    assert "monthly/2026-07.html" not in html  # 无归档时不显示链接

    # 预置一份归档后，主报告应自动列出归档链接（覆盖重写，同一文件）
    archive_dir = tmp_path / "reports" / "monthly"
    archive_dir.mkdir(parents=True)
    (archive_dir / "2026-07.html").write_text("x", encoding="utf-8")
    write_live_report(data, station_labels={"s1": "一号站"})
    html = out.read_text(encoding="utf-8")
    assert "monthly/2026-07.html" in html


def test_write_monthly_report_creates_frozen_archive(tmp_path, monkeypatch):
    """月度归档写入 monthly/YYYY-MM.html：相对路径前缀 ../、返回链接、归档列表。"""
    _populate(tmp_path, monkeypatch)
    monkeypatch.setenv("WEATHER_EVAL_REPORTS_ROOT", str(tmp_path / "reports"))
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08", is_monthly=True)

    out = write_monthly_report(data, station_labels={"s1": "一号站"})
    assert out.name == "2026-08.html"
    html = out.read_text(encoding="utf-8")
    # 子目录页面：ECharts 与返回链接都走 ../ 前缀
    assert '"../vendor/echarts.min.js"' in html or "'../vendor/echarts.min.js'" in html \
        or "../vendor/echarts.min.js" in html
    assert "index.html" in html                # 返回本月实时报告的链接
    assert "月度归档" in html                   # 冻结徽章
    assert "一号站" in html
    # 归档页脚的归档列表包含自身（读者在任意归档页看到完整归档导航）
    assert "monthly/2026-08.html" in html


def test_write_monthly_report_never_rewrites_frozen_archive(tmp_path, monkeypatch):
    """冻结档案保护：已存在的归档默认拒绝重写（内容与 mtime 均不变），--force 才重建。"""
    _populate(tmp_path, monkeypatch)
    monkeypatch.setenv("WEATHER_EVAL_REPORTS_ROOT", str(tmp_path / "reports"))
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08", is_monthly=True)

    first = write_monthly_report(data)
    mtime = first.stat().st_mtime_ns
    content = first.read_text(encoding="utf-8")
    # 再次写入（如同 1 号当天的后续运行/手动 dispatch）：内容与 mtime 均不变
    again = write_monthly_report(data)
    assert again == first
    assert first.stat().st_mtime_ns == mtime
    assert first.read_text(encoding="utf-8") == content
    # --force 显式重建：允许重写
    write_monthly_report(data, force=True)
    assert first.stat().st_mtime_ns != mtime


def test_all_models_registered_in_report_layer():
    """P2-2 守卫：config 的每个模型必须登记在报告层三张表里。

    README 的扩展契约写明"新增源只需改 SOURCE_SPECS + config + CI，评估与报告
    逻辑无需改动"——但 MSN 接入时漏登了 MODEL_LABELS/MODEL_COLORS/MODEL_FAMILIES，
    它以原始 id 显示在总榜第 3 名附近且不属于任何源分组。这条测试把该契约变成
    机器可校验的：任何新源漏登任何一张表，CI 直接红。"""
    from weather_eval.config import load_config
    from weather_eval.report import render

    cfg = load_config()
    fam = {m for f in render.MODEL_FAMILIES for m in f["models"]}
    for m in cfg.models:
        assert m in render.MODEL_LABELS, f"{m} 缺中文名（MODEL_LABELS）"
        assert m in render.MODEL_COLORS, f"{m} 缺配色（MODEL_COLORS）"
        assert m in fam, f"{m} 缺源分组（MODEL_FAMILIES）"


def test_inline_json_is_compact():
    """P3-4：内联 JSON 用紧凑分隔符（MB 级内联数据白省 10~15% 体积）。"""
    from weather_eval.report.render import _js_json
    out = _js_json({"a": [1, 2], "b": "中文</script>"})
    assert ": " not in out and ", " not in out
    assert "<\\/" in out            # </ 转义仍在
    import json as _json
    assert _json.loads(out.replace("<\\/", "</")) == {"a": [1, 2], "b": "中文</script>"}


def test_sparkline_svg_server_rendered(tmp_path, monkeypatch):
    """走势线服务端渲染回归（2026-09 重构后的守卫）。

    历史：走势线曾由前端 JS 渲染，中位参考线依赖按轨道注入的全局变量——
    fc7c402 把全局 sparkDomain 拆成 SPARK_DOMAIN{all,hourly,daily} 后，sparkSVG
    内部仍读旧全局，榜单渲染到第一行就抛 ReferenceError，被外层 try/catch 吞掉，
    表头在、表体空。重构后走势线在 Python 侧生成（_sparkline_svg 接收 med 形参、
    不读任何全局），这条 bug 类别被整类消灭——本测试锁住"服务端出图"这一性质，
    防止走势线又被改回前端渲染。"""
    import re
    _populate(tmp_path, monkeypatch)
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08")
    html = render_report_html(data, title="走势线服务端渲染回归")

    # 榜单行内直接嵌服务端生成的 SVG（无 JS 参与），且带中位虚线参考线
    assert '<td class="l c-spark"><svg' in html
    assert 'stroke-dasharray="3 3"' in html
    # 渲染函数签名保持"中位线由调用方按轨道传入"的形状（防同类回归的语义锚点）
    m = re.search(r"def _sparkline_svg\([^)]*\)", html) or \
        re.search(r"def _sparkline_svg\([^)]*\)",
                  open(render_mod_path(), encoding="utf-8").read())
    assert m and "med" in m.group(0)
    # 前端不再有任何 spark 渲染函数
    assert "sparkSVG" not in html


def test_board_metric_dimension_tabs(tmp_path, monkeypatch):
    """§01 指标列页签（2026-10）：总榜呈现全部已评测指标，横排不拉长。

    15 项已评测指标全铺成列会把横排拉到两千像素开外；页签把温度/降水两维各收成
    一个列组（radio + 兄弟选择器，纯 CSS，无 JS 也可切换），默认「概览」只看总分。
    「选了哪一维就按哪一维排」的排序语义见 test_board_dimension_tab_is_sort_order。
    单项指标与综合分走同一张劈分设计（_build_board_rows 的 _aligned），行内
    data-* 供列排序使用；分时效榜经 slim 行数组携带同列指标（页签全榜可用）。"""
    _populate(tmp_path, monkeypatch)
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08")
    html = render_report_html(data, title="指标页签")

    # 页签结构：三个 radio + label（概览默认选中），共享列/族专属列成对出现
    assert 'id="lbdim-ov" checked' in html
    assert 'id="lbdim-temp"' in html and 'id="lbdim-rain"' in html
    assert 'class="sh-temp h-temp"' in html and 'class="sh-rain h-rain"' in html
    assert 'class="t-only h-temp"' in html and 'class="r-only h-rain"' in html
    # 概览只有总分两列：±2°C 与晴雨 TS 归入各自族页签（t-only / r-only）
    assert '<td class="sh-temp" data-label="±2°C">' not in html
    assert '<td class="sh-rain" data-label="晴雨TS">' not in html
    assert '<td class="t-only" data-label="±2°C">' in html
    assert '<td class="r-only" data-label="晴雨TS">' in html
    # 表头瘦身：长括注不再进表头（口径说明由 title/hint 承担；冠军卡上的
    # 内联标签不是表头，保留完整措辞）
    assert ">晴雨 TS（两轨对齐）</th>" not in html and ">晴雨 TS</th>" in html
    assert ">±2°C</th>" in html and ">±1°C</th>" in html
    # 全部已评测指标都在表头（数据通路锁点：_board_row 透传 + 对齐矩阵）
    for key in ("acc1", "rmse", "mae", "mbe", 'data-key="r"', "slope",
                "accr", "ets", "pod", "far", "bias", "amt_mae", "amt_bias",
                "grade_ets"):
        assert key in html, f"表头缺指标列 {key}"
    # 行内 data-* 与页签列一一对应（列排序的数据源）
    assert 'data-grade_ets="' in html and 'data-bdisp="' in html
    # 隐藏 radio 显式钉在页签行（top:0/left:0）——零尺寸 abspos 的 static
    # position 在 flex 容器里会被 Chrome 解析到远处，聚焦即页面下跳（实测回归）
    assert ".lb-dims input { position:absolute; top:0; left:0;" in html
    # 单项指标与综合分同一张设计：总榜行带对齐后的指标键（单模型 fixture 走
    # 降级路径、数值可为 None，但键缺失 = 数据通路断了）
    row = data["leaderboards"]["all"][0]
    for key in ("acc1", "mae", "mbe", "r", "slope", "mbe_bdisp",
                "accr", "pod", "far", "bias", "amt_mae", "amt_bias", "grade_ets"):
        assert key in row, f"总榜行缺指标键 {key}"
    # accr 与 pod 出自同一张 2×2 列联表，缺有必须一致——锁 _board_row 的
    # 改名约定（p 里传原始键 acc，_build_board_rows 曾传 accr 导致整列 None）
    assert (row["accr"] is None) == (row["pod"] is None)
    row_h = data["leaderboards"]["hourly:1d"][0]
    assert "acc1" in row_h and (row_h["accr"] is None) == (row_h["pod"] is None)
    # 分榜 slim 行数组与 LB_METRIC_COLS 的下标约定同步：21 槽全指标
    from weather_eval.report.render import _slim_report
    slim = _slim_report(data)
    bad = [(tk, d, len(r)) for tk, days in slim["lb"].items()
           for d, rows in days.items() for r in rows if len(r) != 21]
    assert slim["lb"] and not bad, f"slim lb 行数组应为 21 槽，异常：{bad[:5]}"
    # 切到分榜时页签不再隐藏（页签在所有榜单下可用）
    assert 'id="lbDims"' in html and ".lb-dims.sub" not in html


def test_board_dimension_tab_is_sort_order(tmp_path, monkeypatch):
    """§01 维度页签 = 排序口径（2026-10）：选了温度/降水就该按该维分数排。

    第一性原则：榜单的排序键是视图的属性，不是永远固定的——读者把页签切到
    「🌡️ 温度」，意图就是"按温度分看谁强"。旧版维度页签只做列过滤（纯 CSS），
    行序与服务端渲染一样恒按综合分，且综合分列仍挂在表上：读者拿着总分的
    高低去读一个声称"温度视图"的序。本测试锁住新行为的三个部件：

      * 列：综合分列标 ov-only（概览专属），温度/降水页签 CSS 隐藏；
      * 序：LB_DIM_KEY 把页签映射到排序键，lbApplyOrder 按该键重排并
        重算名次（# 列 = 当前视图第几名），表头箭头随排序键走；
      * 说明：lbDesc 逐页签写明"当前按什么排、总分去哪了"。
    """
    _populate(tmp_path, monkeypatch)
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08")
    html = render_report_html(data, title="维度即排序")

    # ---- 列：综合分 = 概览专属（ov-only），子项页签下隐去 ----
    assert '<th class="ov-only" data-key="score"' in html
    assert '<td class="c-score ov-only" data-label="综合分">' in html
    assert "#lbdim-temp:checked ~ .lb-scroll .ov-only" in html
    assert "#lbdim-rain:checked ~ .lb-scroll .ov-only" in html
    # 温度分/降水分列在子项页签是共享列（留在表上），但不是概览专属
    assert 'class="sh-temp h-temp"' in html and 'class="sh-rain h-rain"' in html
    # ---- 得分条只给"当前视图的排序键"（2026-10 用户口径：概览只要综合分的条）----
    # 三个"分"列同构：格内是 .score-bar（条，默认隐藏）+ .score-num（数字）
    assert '<td class="sh-temp" data-label="温度分"><div class="score-bar">' in html
    assert '<td class="sh-rain" data-label="降水分"><div class="score-bar">' in html
    # 分榜行：LB_METRIC_COLS 的 thermo 格式生成同款格（条 + 数字）
    assert '["sh-temp", "温度分", "temp", 2, "thermo"]' in html
    assert '["sh-rain", "降水分", "precip", 3, "thermo"]' in html
    assert 'kind === "thermo"' in html and 'const thermoV =' in html
    assert '<div class="score-bar">' in html and '<b class="score-num">' in html
    # 条默认隐藏（概览：三个分列并排是噪声，且两维量纲不同条长不可直接比）；
    # 只在该维页签显示该维的条（CSS 随页签切换，无 JS 同样成立）
    assert ".score-bar { display:none; }" in html
    assert "#lbdim-temp:checked ~ .lb-scroll .sh-temp .score-bar," in html
    assert "#lbdim-rain:checked ~ .lb-scroll .sh-rain .score-bar { display:inline-block;" in html
    # 概览下子项分退回普通数字（不跟总综合分抢"主分"的分量）
    assert "#lbdim-ov:checked ~ .lb-scroll .sh-temp .score-num," in html
    assert "color:inherit; }" in html
    # 移动端：总综合分是卡片主分（标签 + 条 + 大字），子项分仍折成小字 chips
    assert ".lb-table td.c-score[data-label] { display:block" in html
    assert ".lb-table td.c-score[data-label]::before { content:attr(data-label)" in html

    # ---- 序：页签 → 排序键，重排 + 名次重算 + 箭头 ----
    assert 'LB_DIM_KEY = {ov: "score", temp: "temp", rain: "precip"}' in html
    assert "function lbApplyOrder()" in html
    assert "function lbRefreshRanks(tbody, key" in html
    # 重排读 data-temp / data-precip（分榜行由 LB_METRIC_COLS 生成同款属性）
    assert "tr.dataset[key]" in html
    # 三条写路径（维度重排 / 分榜渲染 / 列头排序）之后都刷新名次
    assert html.count("lbRefreshRanks(") >= 3
    # 表头箭头能落到温度/降水分列（.ind 占位）+ 页签 change 触发重排
    assert '<th class="sh-temp h-temp" data-key="temp" data-type="num">温度分<span class="ind"></span></th>' in html
    assert '<th class="sh-rain h-rain" data-key="precip" data-type="num">降水分<span class="ind"></span></th>' in html
    assert "function bindDimTabs()" in html and "bindDimTabs();" in html
    assert 'input[name="lbdim"]' in html
    # renderBoard 渲染后必须重排（分榜 × 维度两轴正交）
    assert "lbApplyOrder();" in html and html.count("lbApplyOrder()") >= 2

    # ---- 说明：口径说明逐维度写明排序键与总分去向 ----
    assert "当前按难度对齐综合分降序排列" in html
    assert "当前按「温度分」降序排列" in html
    assert "当前按「降水分」降序排列" in html
    assert "lbDesc(lbMode, lbDim)" in html

def test_issue_anchor_rows_puts_flags_first_then_groups():
    """起报锚点披露重排（2026-10 呈现重构）：例外优先、其余按语义归组。

    旧版是一张平表：同一句「（另有 N 份历史存档未声明，随归档递减）」逐行重复、
    20 行存疑列是一串「—」，真正要看的挂旗源埋在灰字里（且长句 nowrap 把表顶出
    卡片）。这里锁住新结构的性质：disputed 单独成块在前、其余按语义归组且组内
    保序、未声明只留一个数字。
    """
    from weather_eval.report.render import _issue_anchor_rows

    anchors = {
        "a1": {"issue_source": "axis_start",
               "issue_source_label": "时间轴首点（产品的起始时刻）",
               "issue_source_undeclared": 100,
               "issue_source_note": "仍有 100 份快照未声明锚点语义",
               "disputed": False, "disputed_reasons": []},
        "a2": {"issue_source": "axis_start",
               "issue_source_label": "时间轴首点（产品的起始时刻）",
               "issue_source_undeclared": 96,
               "disputed": False, "disputed_reasons": []},
        "b1": {"issue_source": "model_run", "issue_source_label": "模式轮次（真实起报时次）",
               "issue_source_undeclared": 188,
               "disputed": False, "disputed_reasons": []},
        "bad": {"issue_source": "request_floor",
                "issue_source_label": "请求时刻下取整（≈抓取时刻）",
                "issue_source_undeclared": 316, "disputed": True,
                "disputed_reasons": ["起报锚点语义为「请求时刻下取整（≈抓取时刻）」",
                                     "最近城市吸附（平均 3.5 km，代表城市而非站点）"]},
        "none": {"issue_source": None, "issue_source_undeclared": 0,
                 "disputed": False, "disputed_reasons": []},   # 无可披露 → 不进表
    }
    out = _issue_anchor_rows({"meta": {"issue_anchors": anchors}})

    assert (out["n_total"], out["n_flagged"], out["n_clean"]) == (4, 1, 3)
    # 例外在前：挂旗的源单独成块，理由原样保留（= 榜上 ⚠️ 悬停看到的同一段话）
    assert [r["m"] for r in out["flagged"]] == ["bad"]
    assert out["flagged"][0]["reason_text"] == "；".join(anchors["bad"]["disputed_reasons"])
    # 其余按语义归组，组内保持榜内序（dict 插入序）
    assert [g["semantic"] for g in out["groups"]] == ["时间轴首点（产品的起始时刻）",
                                                      "模式轮次（真实起报时次）"]
    assert [r["m"] for r in out["groups"][0]["members"]] == ["a1", "a2"]
    # 未声明只留数字；逐行复述该数字的 note 不上屏（重复即噪音）
    assert out["groups"][0]["members"][0]["undeclared"] == 100
    rows = out["flagged"] + [r for g in out["groups"] for r in g["members"]]
    assert all("note" not in r for r in rows)
    # 一份可披露信息都没有 → None，模板据此整节不渲染
    assert _issue_anchor_rows({"meta": {}}) is None


def test_score_parts_rows_split_formula_from_note():
    """评分构成表把换算白话拆成「主式 + 括注」：信息一条不丢（可拼回原文）。

    SCORE_MAP_TEMPLATES 的每条都是 ``主式（说明）`` 的形状；页面里主式要一眼
    可读、括注退灰并放开换行——旧版整句一格、且 nowrap，正是把表顶出卡片的长句。
    """
    from weather_eval.report.render import _score_parts_rows, _split_map, TEMP_SCORE_PARTS

    assert _split_map("主式（括注）") == ("主式", "括注")
    assert _split_map("没有括注") == ("没有括注", "")

    rows = _score_parts_rows(TEMP_SCORE_PARTS)
    assert len(rows) == 8          # 2026-10 全指标审计 7 项 + 第二轮新增站间一致性
    for r in rows:
        rebuilt = f"{r['formula']}（{r['note']}）" if r["note"] else r["formula"]
        assert rebuilt == r["map"], "拆分不得丢字：主式+括注必须拼回原文"
        assert r["formula"]
    acc2 = next(r for r in rows if r["key"] == "acc2")
    assert acc2["formula"].startswith("百分比×") and "个百分点扣 1 分" in acc2["note"]


def render_mod_path():
    from weather_eval.report import render
    return render.__file__


# --------------------------------------------------------------- 视图随代码重建
# 第一性原理（2026-10）：报告页 = render(该月数据, 代码)。数据冻结、视图随代码
# 重求值——产物头部盖一份"视图指纹"，重建前逐份比对，一致即跳过、不一致才重渲。
# 这几条测试锁住该机制：指纹入产物、陈旧触发重建、最新不重写、无原料不崩。

def _monthly_fixture(tmp_path, monkeypatch):
    """铺 2026-08 的观测+快照，返回 (cfg_stub 用不到, 该月 build_report 结果)。"""
    _populate(tmp_path, monkeypatch)
    monkeypatch.setenv("WEATHER_EVAL_REPORTS_ROOT", str(tmp_path / "reports"))
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 30, 23, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    return build_report(["s1"], ["ecmwf_ifs"], cfg, start, end,
                        "2026-08", is_monthly=True)


class _StubCfg:
    """refresh_monthly_archives 只从 cfg 取 stations（用于站点中文名）。"""
    stations: list = []


def test_report_embeds_view_fingerprint(tmp_path, monkeypatch):
    """主报告与月榜都盖视图指纹 meta，且可由 _embedded_fingerprint 读回。"""
    from weather_eval.report import render

    data = _monthly_fixture(tmp_path, monkeypatch)
    fp = render.view_fingerprint()
    assert fp and len(fp) >= 8

    html = render.render_report_html(data)
    assert f'name="wx-render-fp" content="{fp}"' in html

    out = render.write_monthly_report(data, force=True)
    assert render._embedded_fingerprint(out) == fp
    # 无指纹/非本模板产物回退为 None（调用方据此判过期）
    plain = tmp_path / "reports" / "monthly" / "2020-01.html"
    plain.write_text("<html><head></head></html>", encoding="utf-8")
    assert render._embedded_fingerprint(plain) is None


def test_refresh_rebuilds_stale_archive(tmp_path, monkeypatch):
    """产物指纹陈旧时，refresh 用最新代码重建，并盖上当前指纹。"""
    from weather_eval.report import render
    import weather_eval.__main__ as m

    data = _monthly_fixture(tmp_path, monkeypatch)
    out = render.write_monthly_report(data, force=True)

    # 模拟"代码变了"：当前指纹换成一个与产物不同的值（重建数据源已备好，
    # 直接注入，跳过昂贵的全量重算——本测试只锁指纹判定与落盘这一环）
    monkeypatch.setattr(render, "view_fingerprint", lambda: "deadbeefcafe0000")
    monkeypatch.setattr(m, "_rebuild_month_data", lambda cfg, month: data)

    n = m.refresh_monthly_archives(_StubCfg())
    assert n == 1
    assert render._embedded_fingerprint(out) == "deadbeefcafe0000"


def test_refresh_skips_when_fingerprint_current(tmp_path, monkeypatch):
    """指纹与当前代码一致 → 零重建、零重算，产物 mtime 不变。"""
    from weather_eval.report import render
    import weather_eval.__main__ as m

    data = _monthly_fixture(tmp_path, monkeypatch)
    out = render.write_monthly_report(data, force=True)
    mtime = out.stat().st_mtime_ns

    called = []
    monkeypatch.setattr(m, "_rebuild_month_data",
                        lambda cfg, month: (called.append(month), data)[1])
    assert m.refresh_monthly_archives(_StubCfg()) == 0
    assert called == []                            # 未触发任何重算
    assert out.stat().st_mtime_ns == mtime         # 未重写


def test_refresh_keeps_unrebuildable_month(tmp_path, monkeypatch, caplog):
    """原始快照与结论摘要皆缺失的月份：告警并保留原样，不写坏产物。"""
    import logging
    from weather_eval.report import render
    import weather_eval.__main__ as m

    data = _monthly_fixture(tmp_path, monkeypatch)
    out = render.write_monthly_report(data, force=True)
    before = out.read_text(encoding="utf-8")

    monkeypatch.setattr(render, "view_fingerprint", lambda: "newfpmustdiffer")
    monkeypatch.setattr(m, "_rebuild_month_data", lambda cfg, month: None)

    with caplog.at_level(logging.WARNING):
        assert m.refresh_monthly_archives(_StubCfg()) == 0
    assert "无法用最新代码重建" in caplog.text
    assert out.read_text(encoding="utf-8") == before


def test_rebuild_month_data_source_priority(tmp_path, monkeypatch):
    """重建数据源优先级：保留期内原始快照 > 结论摘要 > 无（None）。"""
    import json as _json
    import weather_eval.__main__ as m

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))   # 空数据根

    class _C:
        station_ids: list = []
        models: list = []
        eval: dict = {}

    # 两者皆无 → None（如 2026-07）
    assert m._rebuild_month_data(_C(), "2026-07") is None

    # 只有结论摘要 → 用它渲染结论级页面（期号/总榜非空才判为已固化）
    storage._atomic_write_json(storage.period_summary_path("2026-07"), {
        "period": "2026-07", "meta": {"period_label": "2026-07"},
        "leaderboards": {"all": [{"model": "ecmwf_ifs", "score": 1.0}]},
    })
    data = m._rebuild_month_data(_C(), "2026-07")
    assert isinstance(data, dict) and data["period"] == "2026-07"
    assert _json.loads(storage.period_summary_path("2026-07").read_text(encoding="utf-8"))


def test_summary_only_report_hides_detail_sections(tmp_path, monkeypatch):
    """结论级页面（原始快照已出仓、只剩 summary.json）：隐藏需逐样本原料的图节，
    榜单仍服务端渲染，并给出"结论摘要版"说明——不留空图框。

    摘要形状与 ``__main__._write_period_summary`` 固化的完全一致（meta/coverage/
    scorecard/leaderboards，无逐样本明细数组），故用真实 build_report 结果裁剪而来。
    """
    from weather_eval.report import render

    data = _monthly_fixture(tmp_path, monkeypatch)
    summary = {
        "period": data["meta"]["period_label"], "frozen_at": "2026-09-01 06:00",
        "note": "原始快照出仓后的唯一可复核来源",
        "meta": data["meta"], "coverage": data.get("coverage"),
        "scorecard": data.get("scorecard"), "leaderboards": data["leaderboards"],
    }
    html = render.render_report_html(summary, title="摘要版")

    assert 'id="summary-only"' in html            # 摘要版说明在
    for el in ("chartDecay", "chartWx", "chartHeat", "chartTs", "chartStation"):
        assert f'id="{el}"' not in html, f"无原料却渲染了 {el}"
    assert 'id="lbTable"' in html                 # 榜单照常服务端渲染




# ---------------------------------------------------------------------------
# P2 复算不变量（score_review_proposal §2 P2）：日榜温度分必须能由**发布的
# 分半指标**复算出来。为什么单独锁：这条不变量守的是"发布契约"——键集、精度档
# 位、max/min 半映射、行对齐。之前它只在真实整跑里生效，测试套件的
# `_slim_report` 调用喂的是合成 fixture、`temp_halves` 整体缺席，于是
# `_verify_daily_temp_halves` 每格都走 `continue` 分支——**锁从未被测过**。
# 实际后果：它一度按内部 2 位精度去比发布 1 位精度的分数，在**第一个**正常格上
# 就炸（lb.daily[1][0] 发布 97.0 / 复算 96.95），即"渲染期必崩"，而测试全绿。
# ---------------------------------------------------------------------------


def _half_case(mb_max=-1.22, mb_min=0.32):
    """一个日温度格的发布三件套：分半指标 + 它的真分数。

    mbe 刻意取**异号**——那正是 P2 的病根（显示列取均值会让两半偏差相消，
    分数不会），异号时"用均值复算"必然对不上，用分半复算才对得上。
    """
    from weather_eval.evaluate import _DAILY_TEMP_VIEW_KEYS, daily_temp_score
    mx = {"acc2": 80.4, "acc1": 45.2, "rmse": 1.63, "mae": 1.29,
          "mbe": mb_max, "r": 0.71, "slope": 0.96, "mbe_bdisp": 0.42}
    mn = {"acc2": 91.7, "acc1": 61.3, "rmse": 1.05, "mae": 0.86,
          "mbe": mb_min, "r": 0.83, "slope": 1.02, "mbe_bdisp": 0.31}
    keys = list(_DAILY_TEMP_VIEW_KEYS)
    return (
        {"daily": {"1": [[0, None, None, None] + [None] * 17]},
         "hourly": {}},                                   # 行占位，仅第 2 列(温度分)参与
        {"1": [{"mx": [mx[k] for k in keys], "mn": [mn[k] for k in keys]}]},
        daily_temp_score({"max": mx, "min": mn}),
    )


def test_daily_temp_halves_invariant_reconstructs_published_score():
    """发布精度下，分半指标必须逐位复现发布的日温度分。"""
    from weather_eval.report.render import _verify_daily_temp_halves
    lb, lbtemp, expect = _half_case()
    lb["daily"]["1"][0][2] = round(expect, 1)          # lb 行经 _r 发布，1 位小数
    _verify_daily_temp_halves(lb, lbtemp)     # 不抛 = 通过
    assert abs(expect - round(expect, 1)) <= 0.05


def test_daily_temp_halves_invariant_catches_corrupted_half_value():
    """任一半的指标被改动/键集被改窄，复算必须立刻对不上并炸出来。

    **这条锁抓不到的一类错误（有意记在这里，别误以为它有兜底）**：max/min
    两半**整体互换**。daily_temp_score = (score(max) + score(min)) / 2，对两半
    对称，互换后分数逐位不变——复算当然照样通过。真要抓它得另立一条"分半
    与源序列一致"的检查（需要日最高/最低原值，当前未发布），那是另一件事。
    这里只锁**可锁的**契约：一旦有人动键集、动精度、动某一半的数值，就地炸。
    """
    import pytest
    from weather_eval.report.render import _verify_daily_temp_halves
    lb, lbtemp, expect = _half_case()
    lb["daily"]["1"][0][2] = round(expect, 1)

    lbtemp["1"][0]["mn"][0] = 12.3          # 只动 min 侧的 acc2，其余原样
    with pytest.raises(RuntimeError, match="复算不变量"):
        _verify_daily_temp_halves(lb, lbtemp)


def test_daily_temp_halves_invariant_catches_dropped_key():
    """键集被改窄（少发一项指标）必须被抓——半映射/键集是这条锁的主要宾语。"""
    import pytest
    from weather_eval.evaluate import _DAILY_TEMP_VIEW_KEYS
    from weather_eval.report.render import _verify_daily_temp_halves
    lb, lbtemp, expect = _half_case()
    lb["daily"]["1"][0][2] = round(expect, 1)
    # 少发 slope（它在 TEMP_SCORE_PARTS 里，且非线性），复算必然漂
    idx = list(_DAILY_TEMP_VIEW_KEYS).index("slope")
    lbtemp["1"][0]["mx"][idx] = None
    lbtemp["1"][0]["mn"][idx] = None
    with pytest.raises(RuntimeError, match="复算不变量"):
        _verify_daily_temp_halves(lb, lbtemp)


def test_daily_temp_halves_invariant_catches_row_misalignment():
    """lbtemp 与 lb.daily 行数不同（错位）必须被抓，不许静默按序错配。"""
    import pytest
    from weather_eval.report.render import _verify_daily_temp_halves
    lb, lbtemp, expect = _half_case()
    lb["daily"]["1"][0][2] = round(expect, 1)
    lb["daily"]["1"].append([0, None, None, None] + [None] * 17)
    with pytest.raises(RuntimeError, match="行数不一致"):
        _verify_daily_temp_halves(lb, lbtemp)


def test_daily_temp_halves_invariant_tolerates_absent_halves():
    """旧结论摘要重渲染时没有分半字段：无可对账，不得误伤。"""
    from weather_eval.report.render import _verify_daily_temp_halves
    lb, lbtemp, _ = _half_case()
    lbtemp["1"][0]["mx"] = [None] * 8
    lbtemp["1"][0]["mn"] = [None] * 8
    lb["daily"]["1"][0][2] = 97.0
    _verify_daily_temp_halves(lb, lbtemp)     # 不抛 = 通过


def test_render_replays_score_slopes_before_any_recompute():
    """斜率回放必须发生在任何复算**之前**（时序缺陷回归锁）。

    场景：summary.json 重放路径（月度归档按最新代码重渲）——进程没跑过
    build_report，模块全局 SCORE_SLOPES 还是代码缺省；归档的 meta.score_slopes
    与缺省不同（季度重标定后必然如此）。旧实现把 apply_score_slopes 放在
    _slim_report 之后：复算不变量用**缺省斜率**去对账**旧斜率发布的分半值**
    → AssertionError 把整条月度刷新链炸断。修复后：先按 meta 装载，再复算，
    同一份分半值在其发布斜率下逐位自洽。
    """
    from weather_eval.evaluate import (DEFAULT_SCORE_SLOPES, _DAILY_TEMP_VIEW_KEYS,
                                       apply_score_slopes, daily_temp_score)
    import weather_eval.evaluate as ev
    from weather_eval.report.render import render_report_html
    # 选一个与代码缺省显著不同的斜率（rmse 的换算斜率 ×5）
    slopes = {**DEFAULT_SCORE_SLOPES, "rmse": DEFAULT_SCORE_SLOPES["rmse"] * 5}
    assert slopes["rmse"] != DEFAULT_SCORE_SLOPES["rmse"]
    # 在 slopes 尺度下构造一份自洽的发布数据（分半值 → 分数 → 发布精度）
    try:
        apply_score_slopes(slopes)
        lb, lbtemp, expect = _half_case()
        lb["daily"]["1"][0][2] = round(expect, 1)
        halves = lbtemp
        keys = list(_DAILY_TEMP_VIEW_KEYS)
        assert round(daily_temp_score({
            "max": dict(zip(keys, halves["1"][0]["mx"])),
            "min": dict(zip(keys, halves["1"][0]["mn"])),
        }), 1) == round(expect, 1)
    finally:
        apply_score_slopes(None)               # 还原进程缺省（重放路径的起点状态）
    # ⚠️ 经模块属性读取：apply_score_slopes 是**整体重绑**模块全局，
    # from-import 拿到的旧 dict 引用不会跟随更新
    assert ev.SCORE_SLOPES["rmse"] == DEFAULT_SCORE_SLOPES["rmse"]
    report_full = {
        "meta": {"score_slopes": slopes},
        "leaderboards": {**lb, "all": []},
        "score_trend": {}, "timeseries": {}, "temp_hourly": {},
        "precip_hourly": {}, "temp_daily": {}, "precip_daily": {},
        "per_station": {}, "coverage": {"first_obs": "2026-08-01"},
    }
    html = render_report_html(report_full)     # 旧代码在此 AssertionError
    assert html and "复算不变量" not in html
    # 回放语义：渲染把归档斜率装进全局，同一进程内的后续渲染同尺
    assert ev.SCORE_SLOPES["rmse"] == slopes["rmse"]
    apply_score_slopes(None)


def test_render_replays_score_gains_in_fresh_process():
    """缩放回放与斜率同一条纪律：渲染不得依赖"同进程里先跑过 build_report"。

    场景与上一条相同（summary.json 重放），但缺的是 score_index_gain /
    score_composite_gain。旧实现只回放斜率：进程全局仍是代码缺省 gain，
    复算出的日温度分与归档发布值不符 → P2 复算不变量 AssertionError。
    """
    import weather_eval.evaluate as ev
    import weather_eval.stats as st
    from weather_eval.evaluate import (_DAILY_TEMP_VIEW_KEYS, apply_composite_gain,
                                       apply_score_index_gain, daily_temp_score)
    from weather_eval.report.render import _slim_report

    center = 50.0
    try:
        apply_score_index_gain(2.0, center)
        apply_composite_gain(1.5, center)
        lb, lbtemp, expect = _half_case()
        keys = list(_DAILY_TEMP_VIEW_KEYS)
        assert round(daily_temp_score({
            "max": dict(zip(keys, lbtemp["1"][0]["mx"])),
            "min": dict(zip(keys, lbtemp["1"][0]["mn"])),
        }), 1) == round(expect, 1)
        lb["daily"]["1"][0][2] = round(expect, 1)
    finally:
        apply_score_index_gain(1.0, center)
        apply_composite_gain(1.0, center)
    report = {
        "meta": {"score_slopes": None, "score_index_gain": 2.0,
                 "score_index_center": center, "score_composite_gain": 1.5},
        "leaderboards": {**lb, "all": []},
        "score_trend": {}, "timeseries": {}, "temp_hourly": {},
        "precip_hourly": {}, "temp_daily": {}, "precip_daily": {},
        "per_station": {}, "coverage": {"first_obs": "2026-08-01"},
    }
    _slim_report(report)                        # 旧代码在此 AssertionError
    assert st.SCORE_INDEX_GAIN == 2.0
    assert st.SCORE_COMPOSITE_GAIN == 1.5
    apply_score_index_gain(None)
    apply_composite_gain(None)
    assert st.SCORE_INDEX_GAIN == ev.DEFAULT_SCORE_INDEX_GAIN

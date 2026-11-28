"""评测结果数据导出（reports/eval_all.json）测试。

外部程序消费的唯一数据入口（write_eval_data_export）：挂在 _update_live_report
尾部随每轮 action 重写并入库。测试钉死外部调用方的五条契约——

1. 落点与形态：文件写在 reports 根目录（Pages 部署根 + 随仓库提交），顶层带
   schema 版本、导出时间戳（北京时）与 months_available；
2. 结果层白名单：meta 口径参数 + 榜单/评分卡/分站/曲线/趋势/来源构成/覆盖度；
   重算原料（timeseries/heatmap/逐桶明细表/meta.difficulty_window）绝不出现；
3. 精度口径：结果表浮点收敛到 3 位小数（页面展示精度），meta 保持原精度；
4. 严格 RFC 8259：非有限浮点必须折成 None（_json_safe 口径），任何语言的
   解析器都能读；
5. 幂等重写：同一数据重复导出内容逐位相同，且不留 .tmp 半截文件。

逐月冻结结论（data/metrics/{m}/summary.json）不复制进本文件：跨月累计窗口
已覆盖全部月份的累计口径，两份产物分工（见 __main__.write_eval_data_export
docstring），months_available 只是指针。
"""
import json

from weather_eval.__main__ import write_eval_data_export
from weather_eval import storage

# 重算原料：与指标列不同，这些只服务于页面图表与内部重算（可从快照还原）
_MATERIAL_KEYS = (
    "timeseries", "heatmap", "temp_hourly", "temp_daily", "precip_hourly",
    "precip_daily", "precip_score_daily", "precip_hourly_score",
)


def _seed(tmp_path, monkeypatch, months=("2026-08", "2026-09")):
    """重定向两个输出根并造观测月文件（available_months 只扫文件名）。

    reports 根不重定向会让测试产物**覆盖真实的 reports/eval_all.json**——
    测试绝不能写进真仓库。
    """
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("WEATHER_EVAL_REPORTS_ROOT", str(tmp_path / "reports"))
    for m in months:
        p = tmp_path / "obs" / "s1" / f"{m}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("[]", encoding="utf-8")


def _report():
    """一份带原料键的结果层样本：榜单行含全精度浮点，meta 含标定参数。"""
    return {
        "meta": {"period_label": "跨月累计", "difficulty_window": {"heavy": 1},
                 "score_skill_k": 1.23456789012},
        "coverage": {"ecmwf_ifs": {"n": 100}},
        "scorecard": {"rows": [1, 2]},
        "leaderboards": {"all": [{"model": "ecmwf_ifs", "score": 71.26,
                                  "r": 0.9561116339029627, "rmse": 1.173,
                                  "n_eff": 8836}]},
        "per_station": {"wuzhou": {"ecmwf_ifs": {"temp": {"rmse": 1.23456}}},
                        "best_match": {}},
        "temp_lead_curve": {"hourly": [1.23456]},
        "score_trend": {"weekly": [0.123456]},
        "daily_source_mix": {"ecmwf_ifs": {"hourly": 1, "daily": 0}},
        # ---- 原料键：导出必须整体剔除 ----
        **{k: {"raw": [1, 2, 3]} for k in _MATERIAL_KEYS},
    }


def test_export_shape_and_whitelist(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    out = write_eval_data_export(_report())
    assert out.exists() and out.name == "eval_all.json"

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "weather-eval/export@v2"
    assert payload["timezone"] == "Asia/Shanghai"
    assert payload["exported_at"]
    assert payload["months_available"] == ["2026-08", "2026-09"]

    report = payload["report"]
    assert set(report) == {"meta", "coverage", "scorecard", "leaderboards",
                           "per_station", "temp_lead_curve", "score_trend",
                           "daily_source_mix"}
    # 原料不进结果导出（可从快照重算，指标列已全部在榜单行里）
    assert not (set(report) & set(_MATERIAL_KEYS))
    assert "difficulty_window" not in report["meta"]
    # 结果透传：数值与键序原样（白名单过滤不做二次加工）
    assert report["leaderboards"]["all"][0]["model"] == "ecmwf_ifs"


def test_export_precision_contract(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    out = write_eval_data_export(_report())
    payload = json.loads(out.read_text(encoding="utf-8"))["report"]
    # 结果表：浮点收敛到 3 位小数（展示精度），整数不受影响
    row = payload["leaderboards"]["all"][0]
    assert row["r"] == 0.956 and row["rmse"] == 1.173 and row["n_eff"] == 8836
    assert payload["per_station"]["wuzhou"]["ecmwf_ifs"]["temp"]["rmse"] == 1.235
    assert payload["temp_lead_curve"]["hourly"] == [1.235]
    assert payload["score_trend"]["weekly"] == [0.123]
    # meta 标定参数保持原精度（那是复算分数的输入，不能失真）
    assert payload["meta"]["score_skill_k"] == 1.23456789012


def test_export_folds_nonfinite_to_null(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, months=("2026-08",))
    report = _report()
    report["leaderboards"]["all"][0]["ci90"] = [float("inf"), float("nan"), 1.5]
    out = write_eval_data_export(report)
    # 逐语言解析器契约：文件必须是不含 Infinity/NaN 字面量的合法 JSON
    text = out.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    row = json.loads(text)["report"]["leaderboards"]["all"][0]
    assert row["ci90"] == [None, None, 1.5]


def test_export_compact_and_idempotent(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch, months=("2026-08",))
    report = _report()
    out = write_eval_data_export(report)
    text = out.read_text(encoding="utf-8")
    # 紧凑写（indent=None）：结构化内容里不出现缩进换行
    assert "\n" not in text
    first = out.read_bytes()
    assert write_eval_data_export(report).read_bytes() == first
    # 原子写纪律：不留半截 .tmp
    assert not list(out.parent.glob("*.tmp"))
    # storage._atomic_write_json 默认 indent=2 未被本次改动破坏（人的留痕数据
    # 保持 diff 可读；紧凑只给机器消费的大文件用）
    p = tmp_path / "keep" / "x.json"
    storage._atomic_write_json(p, {"a": 1})
    assert "\n" in p.read_text(encoding="utf-8")

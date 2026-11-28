"""全量评测数据导出（reports/eval_all.json）测试。

外部程序消费的唯一数据入口（write_eval_data_export）：挂在 _update_live_report
尾部随每轮 action 重写。测试钉死外部调用方的四条契约——

1. 落点与形态：文件写在 reports 根目录（Pages 部署根），顶层带 schema 版本、
   导出时间戳（北京时）与 months_available，report 字段即 build_report 全量
   产物原样透传（单一事实源：页面与 JSON 同一次 build_report）；
2. 严格 RFC 8259：非有限浮点必须折成 None（_json_safe 口径），任何语言的
   解析器都能读；
3. 紧凑写：indent=None（机器整读，体积优先）；
4. 幂等重写：同一数据重复导出内容逐位相同，且不留 .tmp 半截文件。

逐月冻结结论（data/metrics/{m}/summary.json）不复制进本文件：跨月累计窗口
已覆盖全部月份的累计口径，两份产物分工（见 __main__.write_eval_data_export
docstring），months_available 只是指针。
"""
import json

from weather_eval.__main__ import write_eval_data_export
from weather_eval import storage


def _seed_months(tmp_path, monkeypatch, months):
    """用观测月文件让 available_months() 有东西可扫（只做文件名级扫描）。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    for m in months:
        p = tmp_path / "obs" / "s1" / f"{m}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("[]", encoding="utf-8")


def test_export_shape_and_passthrough(tmp_path, monkeypatch):
    _seed_months(tmp_path, monkeypatch, ["2026-08", "2026-09"])
    report = {"meta": {"period_label": "跨月累计", "score": 88.5},
              "scorecard": {"any": [1, 2]}}
    out = write_eval_data_export(report)
    assert out.exists() and out.name == "eval_all.json"

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["schema"] == "weather-eval/export@v1"
    assert payload["timezone"] == "Asia/Shanghai"
    assert payload["exported_at"]  # 每次导出都带时间戳
    assert payload["months_available"] == ["2026-08", "2026-09"]
    assert payload["report"] == report  # 全量透传，不做任何二次加工


def test_export_folds_nonfinite_to_null(tmp_path, monkeypatch):
    _seed_months(tmp_path, monkeypatch, ["2026-08"])
    report = {"meta": {}, "detail": [float("inf"), float("-inf"), float("nan"), 1.5]}
    out = write_eval_data_export(report)
    # 逐语言解析器契约：文件必须是不含 Infinity/NaN 字面量的合法 JSON
    text = out.read_text(encoding="utf-8")
    assert "Infinity" not in text and "NaN" not in text
    assert json.loads(text)["report"]["detail"] == [None, None, None, 1.5]


def test_export_compact_and_idempotent(tmp_path, monkeypatch):
    _seed_months(tmp_path, monkeypatch, ["2026-08"])
    report = {"meta": {}, "scorecard": {"a": 1}}
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

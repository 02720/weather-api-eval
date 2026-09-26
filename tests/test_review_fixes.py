"""对抗式审查修复的回归。

第一轮（P0-2/P0-3/P0-6/P1-3/P1-4）：这些缺陷的共同点是**榜单照样出、数字照样有**，
只是含义变了。所以每条都写成"换个数据集、结论必须相应改变/不改变"的形式，实现回退
就立刻变红。

第二轮：跨源相关与站对相关（P0-1/P1-4）的统计性质，以及 reports/ 体积看门狗（P0-3）
——后者是"此前无人测量"的那块盲区，测试锁住的是分层口径而不是某个具体字节数。
"""
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from weather_eval import storage
from weather_eval.evaluate import build_report
from weather_eval.timeutil import iso

CFG = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
       "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 1,
       "min_board_neff": 1, "min_board_neff_rain": 1,
       "bootstrap_runs": 30, "sensitivity_runs": 30,
       # 夹具规模小，格子门槛与长尾门槛必须相应放开，否则设计会被清空——
       # 真实数据下的门槛值由 test_board_gates_on_real_scale 覆盖
       "min_cell_neff": 0, "board_min_col_frac": 0.0}


def _obs(start, hours, temp=20.0, rain=0.0):
    out = []
    for h in range(hours):
        t = start + timedelta(hours=h)
        out.append({"time": iso(t), "temp": temp + (h % 3), "rain": rain})
    return out


def _snap(issue, hours, model="m", temp_bias=1.0, rain_fn=None, **extra):
    times = [iso(issue + timedelta(hours=h)) for h in range(hours)]
    snap = {
        "issue_iso": iso(issue), "station_id": "s1", "source": "test",
        "models": [model], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
        "hourly_time": times,
        "data": {model: {
            "temperature_2m": [20.0 + (h % 3) + temp_bias for h in range(hours)],
            "precipitation": ([rain_fn(h) for h in range(hours)] if rain_fn
                              else [1.0 if h % 6 == 0 else 0.0 for h in range(hours)]),
        }},
    }
    snap.update(extra)
    return snap


# --------------------------------------------------- P0-3 温度/降水同窗对齐
def test_temp_buckets_share_the_calendar_window_with_precip(tmp_path, monkeypatch):
    """「提前 N 天」对温度与降水必须指**同一个自然日**。

    旧口径用 (lead−1)//24+1 分温度桶：那对发报时刻不是 00:00 的源是"滚动 24 小时"，
    而降水侧是"有效日 − 起报日"。实测同一档里有的源 96% 样本来自起报当天、
    有的源 72% 来自起报次日——两维日历构成不同，却被平均成同一个数字。
    """
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 20, 9, 0)          # 发报 09:00：滚动窗口与日历日必然不同
    storage.save_obs("s1", _obs(start, 72))
    storage.save_forecast_snapshot("s1", "m", _snap(start, 72))
    data = build_report(["s1"], ["m"], CFG, start, start + timedelta(hours=71), "2026-08")

    # 1d 桶 = 起报日之后第 1 个自然日（08-21）全天 24 点
    t1 = data["temp_hourly"]["m"]["1d"]
    assert t1["n"] == 24, t1["n"]
    d1 = data["precip_score_daily"]["m"]["1d"]
    assert d1["n"] == 1, d1["n"]                  # 按天轨道：该自然日 1 条配对

    # 直接核对逐桶的日历归属：温度桶 N 的全部样本必须落在"起报日 + N 天"
    from weather_eval.evaluate import collect
    hourly, _daily = collect(["s1"], ["m"], start, start + timedelta(hours=71), 16, 16, 20)
    issue_day = start.date()
    for r in hourly:
        if r["bucket"] >= 1:
            got = datetime.fromisoformat(r["valid_iso"]).date()
            assert got == issue_day + timedelta(days=r["bucket"]), r
    # 起报当日（bucket=0）仍在记录里，但不进任何天桶
    assert any(r["bucket"] == 0 for r in hourly)
    # 1d/2d 是完整的自然日；3d（08-23）只被序列覆盖到 08:00，故 9 个点
    assert data["temp_hourly"]["m"]["2d"]["n"] == 24
    assert data["temp_hourly"]["m"]["3d"]["n"] == 9
    assert all(r["bucket"] != 4 for r in hourly)   # 序列只到第 3 个自然日


def test_board_row_counts_only_scored_samples(tmp_path, monkeypatch):
    """总榜行的样本数只统计**进天桶**的样本（bucket ≥ 1），否则会用
    一批没参与打分的样本虚报这一行的证据量。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 20, 0, 0)
    storage.save_obs("s1", _obs(start, 24))       # 只有起报当天
    storage.save_forecast_snapshot("s1", "m", _snap(start, 24))
    data = build_report(["s1"], ["m"], CFG, start, start + timedelta(hours=23), "2026-08")
    row = data["leaderboards"]["all"][0]
    assert row["n"] == 0                 # 没有一天进天桶
    assert row["n_all_leads"] == 23      # 但全 lead 的配对数如实披露
    assert row["lead_days"] is None


# -------------------------------------------------  P1-3 派生列与综合分同格
def test_derived_columns_use_the_same_cell_mask(tmp_path, monkeypatch):
    """某源某桶"因降水缺测而没有综合分"，就不该带着它的温度值去参与
    "难度对齐 ±2°C"的估计——否则页面两列来自两批不同的格子。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 20, 0, 0)
    storage.save_obs("s1", _obs(start, 96))

    def build(bias_c):
        # C 在 2d 桶的按天降水缺失（降水全 0 且不足门槛？用逐日覆盖不足构造）
        for m in ("a", "b"):
            path = tmp_path / "forecasts" / "s1" / m
            if path.exists():
                import shutil
                shutil.rmtree(path)
        snap_a = _snap(start, 96, model="a", temp_bias=1.0)
        snap_b = _snap(start, 96, model="b", temp_bias=2.0)
        snap_c = _snap(start, 96, model="c", temp_bias=bias_c)
        # C 的降水整段置 None -> 该源所有天桶都没有降水分 -> 综合分为 None
        snap_c["data"]["c"]["precipitation"] = [None] * 96
        for sid, m, s in (("s1", "a", snap_a), ("s1", "b", snap_b), ("s1", "c", snap_c)):
            storage.save_forecast_snapshot(sid, m, s)
        return build_report(["s1"], ["a", "b", "c"], CFG, start,
                            start + timedelta(hours=96 - 1), "2026-08")

    d1 = build(1.0)
    acc1 = {r["model"]: r["acc2"] for r in d1["leaderboards"]["all"]}
    d2 = build(9.0)      # 只改 C 的温度（C 没有综合分，本不应影响任何对齐列）
    acc2 = {r["model"]: r["acc2"] for r in d2["leaderboards"]["all"]}
    assert acc1 == acc2, (acc1, acc2)

    dw = d1["meta"]["difficulty_window"]["all"]
    assert dw["cell_valid"], "设计掩膜必须随披露一起落盘，供读者核对"


# --------------------------------------------- P0-6 残缺快照与锚点语义
def test_incomplete_snapshot_excluded_by_default(tmp_path, monkeypatch):
    """残缺快照（complete=false）默认不进评估：残缺样本与完整样本同权，
    会把"这家少抓了一半数据"混进"这家报得准不准"。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 20, 0, 0)
    storage.save_obs("s1", _obs(start, 48))
    storage.save_forecast_snapshot("s1", "m", _snap(
        start, 48, complete=False, missing_shards=["day=4"]))
    data = build_report(["s1"], ["m"], CFG, start, start + timedelta(hours=47), "2026-08")
    assert data["leaderboards"]["all"][0]["n"] == 0
    assert data["meta"]["excluded_incomplete"] == {"m": 1}

    # 显式关掉门槛时样本回来（口径开关是"可选择排除"，不是硬删除）
    data2 = build_report(["s1"], ["m"], dict(CFG, require_complete_snapshots=False),
                         start, start + timedelta(hours=47), "2026-08")
    assert data2["leaderboards"]["all"][0]["n"] == 24


def test_disputed_only_with_positive_evidence(tmp_path, monkeypatch):
    """争议标记只认正面证据：provider 显式声明了嫌疑锚点/残缺/吸附/量化温度。

    历史存档没有 issue_source 时若也判争议，全榜会同时标红——标记一旦对所有人
    都亮，就等于没有标记。
    """
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 20, 0, 0)
    storage.save_obs("s1", _obs(start, 48))
    # declared：带 meta_schema + 嫌疑锚点
    storage.save_forecast_snapshot("s1", "declared", _snap(
        start, 48, model="declared", issue_source="request_floor",
        meta_schema=2, quantized_temp=True))
    # legacy：没有 schema，锚点未声明
    storage.save_forecast_snapshot("s1", "legacy", _snap(start, 48, model="legacy"))
    data = build_report(["s1"], ["declared", "legacy"], CFG, start,
                        start + timedelta(hours=47), "2026-08")
    anchors = data["meta"]["issue_anchors"]
    assert anchors["declared"]["disputed"] is True
    assert any("请求时刻" in r for r in anchors["declared"]["disputed_reasons"])
    assert any("量化" in r for r in anchors["declared"]["disputed_reasons"])
    assert anchors["legacy"]["disputed"] is False
    assert anchors["legacy"]["issue_source"] == "unknown"
    assert anchors["legacy"]["issue_source_undeclared"] == 1
    # 行上带出锚点信息，页面才能标注
    row = next(r for r in data["leaderboards"]["all"] if r["model"] == "declared")
    assert row["disputed"] is True and row["issue_source"] == "request_floor"


# ------------------------------------ P0-2 / P1-4 技巧剖面与长尾参考榜
def test_disclosure_block_is_complete(tmp_path, monkeypatch):
    """报告必须自带"这份名次有多结实"的全部数字（P0-2/P1-4 的核心诉求）：
    交互方差占比、跨桶名次一致性、加权敏感度、格子样本量、长尾档清单。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 20, 0, 0)
    storage.save_obs("s1", _obs(start, 96))
    for m, bias in (("a", 0.5), ("b", 1.5), ("c", 2.5)):
        storage.save_forecast_snapshot("s1", m, _snap(start, 96, model=m, temp_bias=bias))
    data = build_report(["s1"], ["a", "b", "c"], CFG, start,
                        start + timedelta(hours=95), "2026-08")
    dw = data["meta"]["difficulty_window"]["all"]
    for key in ("variance", "rank_stability", "rank_sensitivity", "cell_weights",
                "profile", "min_cell_neff", "cell_weighting", "gate_relaxed",
                "dropped_thin_cells"):
        assert key in dw, key
    vd = dw["variance"]
    assert vd["residual_share"] is not None
    assert abs(vd["row_share"] + vd["col_share"] + vd["residual_share"] - 1.0) < 1e-6
    # 技巧剖面三段齐备，且每段都有名次与分数
    prof = dw["profile"]
    assert set(prof) >= {"short", "mid", "long"}
    assert prof["short"]["buckets"] == ["hourly:1d", "hourly:2d", "hourly:3d",
                                        "daily:1d", "daily:2d", "daily:3d"]
    assert prof["short"]["scores"]["a"] is not None
    row = next(r for r in data["leaderboards"]["all"] if r["model"] == "a")
    assert row["profile_rank"]["short"] is not None
    # 加权 vs 等权的对照分数一并披露
    assert dw["rank_sensitivity"]["spearman"] is not None
    assert set(dw["equal_weight_scores"]) == {"a", "b", "c"}


def test_long_tail_buckets_are_separated_not_mixed(tmp_path, monkeypatch):
    """长尾桶（家数不足最热闹桶一半）单独成参考榜，不参与总榜名次（P1-4）。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 20, 0, 0)
    storage.save_obs("s1", _obs(start, 16 * 24))
    # 三家都覆盖前 3 天；只有一家覆盖到第 10 天 -> 第 4 天起的"家数"不足
    for m, bias in (("a", 0.5), ("b", 1.5), ("c", 2.5)):
        storage.save_forecast_snapshot("s1", m, _snap(start, 4 * 24, model=m, temp_bias=bias))
    long_only = _snap(start, 10 * 24, model="a", temp_bias=0.5)
    long_only["issue_iso"] = iso(start)
    import json
    p = (tmp_path / "forecasts" / "s1" / "a"
         / (iso(start).replace(":", "") + ".json"))
    existing = json.loads(p.read_text(encoding="utf-8"))
    # 直接把 a 的序列拉长到 10 天（模拟"只有它覆盖长时效"）
    existing["hourly_time"] = long_only["hourly_time"]
    existing["data"]["a"] = long_only["data"]["a"]
    existing.pop("payload_sha256", None)
    p.write_text(json.dumps(existing, ensure_ascii=False), encoding="utf-8")

    cfg = dict(CFG, board_min_col_frac=0.5, min_cell_neff=0)
    data = build_report(["s1"], ["a", "b", "c"], cfg, start,
                        start + timedelta(hours=10 * 24 - 1), "2026-08")
    dw = data["meta"]["difficulty_window"]["all"]
    assert dw["excluded_long_tail_buckets"], dw
    assert not set(dw["excluded_long_tail_buckets"]) & set(dw["buckets"])
    assert dw["long_tail"]["buckets"] == dw["excluded_long_tail_buckets"]
    assert "不参与总榜名次" in dw["long_tail"]["note"]
    # 总榜的名次只用主流桶（a 的分数不应等于含长尾桶的平均）。
    # 2026-09 跨分辨率重构后总榜的列是 (天桶 × 分辨率) 的笛卡尔积：三家共同
    # 覆盖的小时榜与日榜各 3 桶进主设计，更远的长尾桶单独成参考榜。
    assert dw["buckets"] == ["hourly:1d", "hourly:2d", "hourly:3d",
                             "daily:1d", "daily:2d", "daily:3d"]
    assert dw["bucket_indices"] == [1, 2, 3, 1, 2, 3]   # 天桶号按各自分辨率计


def test_board_gate_scale_is_consistent(tmp_path, monkeypatch):
    """门槛与权重的量纲必须一致：拿 √n_eff 的权去比 n_eff 的门槛，会让门槛
    要么形同虚设、要么一次清空全部格子——两种失败都表现为"设计被降级"，不报错。"""
    from weather_eval.evaluate import _board_cell_weights
    neff_t = np.array([[49.0, 4.0]])
    neff_r = np.array([[36.0, 25.0]])
    w, min_w = _board_cell_weights(neff_t, neff_r, "neff", min_cell_neff=25)
    assert np.isclose(w[0, 0], 6.0)          # √min(49,36)=6
    assert np.isclose(w[0, 1], 2.0)          # √min(4,25)=2
    assert np.isclose(min_w, 5.0)            # √25 —— 与权重量纲一致
    assert not (w >= min_w)[0, 1]            # 第二格确实低于门槛
    w2, min_w2 = _board_cell_weights(neff_t, neff_r, "equal", min_cell_neff=25)
    assert w2 is None and min_w2 == 0.0


# ================= 第二轮：跨源相关与站对相关（P0-1 / P1-4） =================
def test_duplicate_source_does_not_double_independent_count():
    """把一个源复制成两份完全相同的数据，k_eff 不应近似翻倍。

    这是"重叠家数虚高"这个缺陷的最小可复现：按源数 m 直读时，27 家复制成
    28 家会让"独立信源数"从 27 变 28；做了跨源相关校正后，新增的那一份与
    原份 ρ=1，k_eff 应当几乎不动。
    """
    from weather_eval import stats as st

    rng = np.random.default_rng(20260925)
    base = rng.normal(size=400)
    series = {}
    for i in range(6):
        noise = rng.normal(scale=0.6, size=400)
        series[f"m{i}"] = {j: float(base[j] * 0.8 + noise[j]) for j in range(400)}
    before = st.source_corr_cluster(series)["k_eff"]

    series["m0_copy"] = dict(series["m0"])          # 完全同源的一份
    after = st.source_corr_cluster(series)["k_eff"]

    assert after < before * 1.25, \
        f"复制一份同源数据后 k_eff 从 {before} 涨到 {after}——跨源相关没有生效"


def test_effective_independent_count_math():
    from weather_eval import stats as st

    assert st.effective_independent_count(10, 0.0) == pytest.approx(10.0)
    # ρ=1 会被 SOURCE_RHO_CLAMP 截断到 0.95（防止退化序列把 k_eff 压到 0），
    # 所以这里是 10/(1+9×0.95) 而不是 1.0——截断行为本身也是要被测试钉住的
    assert st.effective_independent_count(10, 1.0) == pytest.approx(
        10 / (1 + 9 * st.SOURCE_RHO_CLAMP))
    # 负相关没有物理意义，夹紧到 m
    assert st.effective_independent_count(10, -0.5) == pytest.approx(10.0)
    assert st.effective_independent_count(1, 0.5) == pytest.approx(1.0)


def test_station_rho_matrix_reports_pairs():
    """站对相关必须按站对披露，而不是一个平均值（审查 P1-4）。"""
    from weather_eval import stats as st

    rng = np.random.default_rng(7)
    common = rng.normal(size=300)
    series = {
        "a": {j: float(common[j] + rng.normal(scale=.2)) for j in range(300)},
        "b": {j: float(common[j] + rng.normal(scale=.2)) for j in range(300)},
        "c": {j: float(rng.normal()) for j in range(300)},
    }
    out = st.station_rho_matrix(series, min_overlap=30)
    assert out["n_stations"] == 3
    assert len(out["pairs"]) == 3
    by = {(p["a"], p["b"]): p["rho"] for p in out["pairs"]}
    assert by[("a", "b")] > 0.9, "同驱动的站对应当高度相关"
    assert by[("a", "c")] < 0.3, "独立站对不应相关"


# ================= 第二轮：reports/ 体积看门狗（P0-3） =================
REPORTS = Path(__file__).resolve().parents[1] / "reports"


def test_reports_footprint_covers_reports_dir():
    """footprint 必须能看到 reports/ —— 此前它是看门狗唯一的盲区。"""
    fp = storage.reports_footprint(REPORTS)
    assert fp["total_bytes"] > 0
    assert fp["main_bytes"] > 0 or fp["monthly_bytes"] > 0
    assert "warn_bytes" in fp and "fail_bytes" in fp
    # 分层必须分开计：主报告每天重写，归档只增不改，两层混在一起就分不清谁在涨
    assert "monthly_bytes" in fp and "assets_bytes" in fp and "vendor_bytes" in fp


def test_reports_footprint_single_file_threshold():
    """单个 HTML 超软阈值必须被点名——这是"报告又胖了"当轮可见的机制。"""
    fp = storage.reports_footprint(REPORTS)
    for item in fp["over_single_threshold"]:
        assert item["bytes"] > storage.REPORT_SINGLE_WARN_BYTES
        assert item["file"].endswith(".html")


def test_reports_footprint_ignores_hidden_dirs(tmp_path):
    """隐藏目录（.baseline 之类）是本地留档，不进仓库也不部署，不得计入总量。"""
    (tmp_path / "index.html").write_text("x" * 100, encoding="utf-8")
    hidden = tmp_path / ".baseline"
    hidden.mkdir()
    (hidden / "big.html").write_text("y" * 5000, encoding="utf-8")
    fp = storage.reports_footprint(tmp_path)
    assert fp["total_bytes"] == 100

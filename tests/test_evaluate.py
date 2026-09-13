from datetime import datetime, timedelta
import math

from weather_eval import storage
from weather_eval.evaluate import (
    build_report, temp_metrics, precip_metrics,
    temp_score, precip_score, overall_score,
)
from weather_eval.timeutil import iso


def test_temp_metrics_handcalc():
    # 误差 [0, 2, 0, 0] -> |err|<=2 全部成立；|err|<=1 仅有 3/4
    m = temp_metrics([20, 21, 19, 22], [20, 23, 19, 22], [1, 2], 0)
    assert m["acc2"] == 100.0
    assert m["acc1"] == 75.0
    assert abs(m["rmse"] - 1.0) < 1e-6
    assert abs(m["mae"] - 0.5) < 1e-6
    # cyeva 全量指标键齐备（相关系数/斜率 4 点样本可算）
    for k in ("mbe", "rss", "chi2", "r", "slope"):
        assert m[k] is not None, k


def test_temp_metrics_tiny_sample_regress_none():
    # 2 个点不足以做线性回归：r/slope 应为 None 而非崩溃
    m = temp_metrics([20, 21], [20, 23], [1, 2], 0)
    assert m["rmse"] is not None
    assert m["r"] is None and m["slope"] is None


def test_precip_metrics_handcalc():
    # obs=[0,1,1,0,0,1] fcst=[0,1,0,0,1,1]  (>=0.1)
    # 命中 idx1,idx5 -> a=2；空报 idx4 -> b=1；漏报 idx2 -> c=1；正确否定 idx0,idx3 -> d=2
    # TS(a/(a+b+c))=0.5（比值）；acc/pod/far/miss 为百分数；bias 为比值
    m = precip_metrics([0, 1, 1, 0, 0, 1], [0, 1, 0, 0, 1, 1], 0.1, 0)
    assert abs(m["ts"] - 0.5) < 1e-6
    assert abs(m["acc"] - 100 * 2 / 3) < 1e-2
    assert abs(m["pod"] - 100 * 2 / 3) < 1e-2
    assert abs(m["far"] - 100 * 1 / 3) < 1e-2
    assert abs(m["bias"] - 1.0) < 1e-6
    # ETS 手算：(a - a_ref)/(a+b+c-a_ref)，a_ref=(a+c)(a+b)/total=3*3/6=1.5 -> 0.5/2.5=0.2
    assert abs(m["ets"] - 0.2) < 1e-6
    # 空报频率 POFD = b/(b+d) = 1/3
    assert abs(m["farate"] - 100 / 3) < 1e-2
    # 连续量指标键齐备
    for k in ("rmse", "mae", "mbe"):
        assert m[k] is not None, k


def test_precip_metrics_graded_structure():
    m = precip_metrics([0, 5, 30, 0], [0, 3, 20, 0.2], 0.1, 0,
                       kind="24h", graded_levs=("+1", "+2", "+3"))
    assert set(m["graded"].keys()) == {"+1", "+2", "+3"}
    for lev, g in m["graded"].items():
        assert set(g.keys()) == {"acc", "pod", "far", "miss", "ts", "ets", "bias"}, lev


def test_precip_metrics_inf_mask_aligns_with_cyeva():
    # cyeva 的 drop_nan 只剔 NaN、保留 inf（inf 会被 threshold_binarize 判为"有雨"）。
    # ets/farate 的手工二值化必须同口径：若误用 isfinite 掩膜，下例 idx2（obs=inf）
    # 会被丢弃，TS 从 0.5 变 0、ETS 从 0.25 变 None。
    obs = [0.0, 5.0, float("inf")]
    fcst = [0.0, 0.0, 0.5]
    m = precip_metrics(obs, fcst, 0.1, 0)
    # ob=[F,T,T] fb=[F,F,T]：命中1/漏报1/空报0 -> TS=1/2
    assert abs(m["ts"] - 0.5) < 1e-6
    # ETS：hits_ref=(1+1)*(1+0)/3=2/3 -> (1-2/3)/(1+0+1-2/3)=0.25
    assert abs(m["ets"] - 0.25) < 1e-6
    # 空报频率 POFD = fa/(fa+cr) = 0/1 = 0
    assert abs(m["farate"] - 0.0) < 1e-6


def test_min_sample_suppresses():
    m = temp_metrics([20, 21], [20, 23], [1, 2], 5)
    assert m["rmse"] is None
    assert m["n"] == 2


def test_scores_weighted_multi_metric():
    # 温度分：只有 acc2/rmse 时（权重各 0.25）等价于两者均分
    t = {"acc2": 90.0, "rmse": 1.0}
    assert temp_score(t) == (90.0 + 95.0) / 2
    # 全项在位：按 TEMP_SCORE_PARTS 权重加权
    t_full = {"acc2": 80, "rmse": 2.0, "r": 0.9, "acc1": 60,
              "mae": 1.5, "mbe": 0.5, "slope": 1.1}
    # 子分：80, 90, 90, 60, 92.5, 95, 90；权重 .25/.25/.15/.10/.10/.10/.05
    assert abs(temp_score(t_full) - 85.25) < 1e-9
    # 降水分（2026-09 重构）：acc 不再入分（主要由气候基率决定）——
    # ts 单独在位即 ts×100；acc 无论多高/多低都不改变分数
    p = {"ts": 0.5, "acc": 80.0}
    assert precip_score(p) == 50.0
    assert precip_score({"ts": 0.5, "acc": 0.0}) == 50.0
    # ETS 首位（0.35）：ets 与 ts 同值时 ETS 话语权更大
    p2 = {"ets": 0.5, "ts": 0.5}
    assert precip_score(p2) == 50.0
    # 子分截断到 [0,100]：ETS 为负记 0 分，不拖成负总分（缺项按剩余权重归一：
    # 0×0.35 与 50×0.25 在剩余权重 0.60 上归一 -> 20.83）
    assert precip_score({"ets": -0.5, "ts": 0.5}) == round(50.0 * 0.25 / 0.60, 2)
    # RMSE 大到换算分为负时截断为 0
    assert temp_score({"rmse": 30.0}) == 0.0
    # 缺项按剩余权重归一：只有 r 时常数为 r×100
    assert temp_score({"r": 0.8}) == 80.0
    # 综合分 = 两者均分
    assert overall_score(t, p) == (92.5 + 50.0) / 2
    # 缺项不计：只有温度分时综合分 = 温度分
    assert overall_score(t, {"ts": None, "ets": None}) == 92.5
    # 全缺 -> None
    assert overall_score({}, {}) is None


def test_score_parts_contract():
    # 权重表契约：指标键、权重和为 1、每项带白话标签与换算函数
    from weather_eval.evaluate import PRECIP_SCORE_PARTS, TEMP_SCORE_PARTS
    for parts, keys in (
        (TEMP_SCORE_PARTS, {"acc2", "rmse", "r", "acc1", "mae", "mbe", "slope"}),
        (PRECIP_SCORE_PARTS, {"ets", "ts", "pod", "far", "bias"}),
    ):
        assert {k for k, *_ in parts} == keys
        assert abs(sum(w for _k, w, *_ in parts) - 1.0) < 1e-9
        for _k, _w, label, mp, fn in parts:
            assert label and mp and callable(fn)
    # P0-3 契约：acc 不入分（气候基率主导）；ETS 权重首位；
    # FAR+BIAS 权重之和不低于 POD（不奖励"多报占便宜"）
    keys = [k for k, *_ in PRECIP_SCORE_PARTS]
    assert "acc" not in keys
    w = {k: weight for k, weight, *_ in PRECIP_SCORE_PARTS}
    assert list(w) == sorted(w, key=w.get, reverse=True)   # ETS 权重最大
    assert w["far"] + w["bias"] >= w["pod"]


def test_score_clamps_and_dimension_conventions():
    # 换算方向与截断的守卫：任一翻脸即红
    assert temp_score({"r": -0.5}) == 0.0            # r 为负 -> 截断 0
    assert temp_score({"slope": 0.0}) == 0.0         # 斜率 0（幅度全丢）-> 0 分
    assert temp_score({"slope": 1.0}) == 100.0       # 斜率恰为 1 -> 满分
    assert temp_score({"mbe": -2.5}) == 75.0         # |MBE| 双向对称（负偏差同样扣分）
    assert precip_score({"ts": 0.0}) == 0.0          # TS=0 不因截断变 None
    assert precip_score({"bias": 4.6}) == 0.0        # BIAS 极端 -> 截断 0
    assert precip_score({"bias": 1.0}) == 100.0      # 频率偏差恰为 1 -> 满分
    # far 是百分比(0~100)的关键约定：100 − 100/3 ≈ 66.7；
    # 若上游漂移成比值(0~1)，这里会得到 ≈100 分，即暴露量纲回归
    assert abs(precip_score({"far": 100.0 / 3}) - (100.0 - 100.0 / 3)) < 0.01
    # far 缺项（如从不报雨的源）按剩余权重归一，不整行出局
    assert precip_score({"far": None, "ts": 0.5}) == 50.0


def test_build_report_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    obs = []
    for h in range(48):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 3), "rain": 1.0 if h % 6 == 0 else 0.0})
    storage.save_obs("s1", obs)

    times = [iso(start + timedelta(hours=h)) for h in range(48)]
    snap = {
        "issue_iso": iso(start), "station_id": "s1", "source": "open-meteo",
        "models": ["ecmwf_ifs"], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
        "hourly_time": times,
        "data": {"ecmwf_ifs": {
            "temperature_2m": [20.0 + (h % 3) + 1.0 for h in range(48)],
            "precipitation": [1.0 if h % 6 == 0 else 0.0 for h in range(48)],
        }},
    }
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap)

    end = start + timedelta(hours=47)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 1,
           "min_board_neff_rain": 1}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08")

    # 逐小时 24h 桶：lead 1..24 共 24 对，且预报=观测+1 -> 误差恒为 1 -> ±2°C 准确率 100%，RMSE=1
    sc_t = data["scorecard"]["ecmwf_ifs"]["temp_24h"]
    assert sc_t["n"] == 24
    assert sc_t["acc2"] == 100.0
    assert abs(sc_t["rmse"] - 1.0) < 1e-6

    # 降水：预报与观测逐小时完全一致 -> TS=1，准确率=100
    pb = data["scorecard"]["ecmwf_ifs"]["precip_24h"]
    assert abs(pb["ts"] - 1.0) < 1e-6
    assert pb["acc"] == 100.0

    # 逐小时各天桶都有数据（lead 上限 47，故 2d 桶含 lead 25..47 共 23 个样本）
    assert data["temp_hourly"]["ecmwf_ifs"]["1d"]["n"] == 24
    assert data["temp_hourly"]["ecmwf_ifs"]["2d"]["n"] == 23

    # 逐小时降水桶含 1h 雨强分级结构；按天降水桶含 24h 累计分级结构
    assert set(data["precip_hourly"]["ecmwf_ifs"]["1d"]["graded"].keys()) == {"1", "2", "3", "4", "5"}
    assert set(data["precip_daily"]["ecmwf_ifs"]["1d"]["graded"].keys()) == {"+1", "+2", "+3", "+4", "+5", "+6"}

    # 按天：offset 1（有效日 08-25）存在样本
    assert data["temp_daily"]["ecmwf_ifs"]["1d"]["max"]["n"] == 1

    # 时间序列与热力图结构存在
    assert "s1" in data["timeseries"]
    assert isinstance(data["heatmap"], list)

    # 分时效排行榜：温度分来自逐小时轨道、降水分来自按天累计轨道（P0-3 主轨道切换）
    from weather_eval.evaluate import overall_score, _mean_or_none
    lb1 = data["leaderboards"]["1d"]
    assert lb1 and lb1[0]["model"] == "ecmwf_ifs"
    row = lb1[0]
    assert row["score"] is not None and row["temp_score"] is not None and row["precip_score"] is not None
    assert 0 <= row["score"] <= 100 and 0 <= row["temp_score"] <= 100
    # 榜单分数与该桶两条轨道完全一致，冠军横幅与榜单不分叉
    t1 = data["temp_hourly"]["ecmwf_ifs"]["1d"]
    p1 = data["precip_score_daily"]["ecmwf_ifs"]["1d"]
    assert row["score"] == overall_score(t1, p1)
    assert row["acc2"] == t1["acc2"] and row["ts"] == p1["ts"]
    # 全部时效桶都有榜单行（无数据的桶行内分数为 None、沉底），另有总榜 "all"
    assert set(data["leaderboards"]) == {"all"} | {f"{i}d" for i in range(1, 17)}
    assert all(len(rows) == 1 for rows in data["leaderboards"].values())
    assert data["leaderboards"]["5d"][0]["score"] is None

    # 全时效总榜（P0-1 macro 化）：各"两维齐备"天桶综合分的等权平均，
    # 既不是全部样本池化，也不把单维桶算作综合分
    all_row = data["leaderboards"]["all"][0]
    assert all_row["model"] == "ecmwf_ifs" and all_row["score"] is not None
    assert all_row["n"] == 47 and all_row["lead_days"] == 2
    bucket_overalls = [
        overall_score(data["temp_hourly"]["ecmwf_ifs"][f"{b}d"],
                      data["precip_score_daily"]["ecmwf_ifs"][f"{b}d"])
        for b in (1, 2)
        if temp_score(data["temp_hourly"]["ecmwf_ifs"][f"{b}d"]) is not None
        and precip_score(data["precip_score_daily"]["ecmwf_ifs"][f"{b}d"]) is not None]
    assert all_row["score"] == _mean_or_none(bucket_overalls)
    assert all_row["n_buckets"] == len(bucket_overalls)
    # 不确定性（P0-2）：90% 置信区间、冠军频率、n_eff 门槛达标标记齐备。
    # 误差恒定的确定性夹具 → 每次重采样分数相同 → CI 坍缩为点值、冠军频率 100%
    assert all_row["ci90"] is not None and all_row["ci90"][0] <= all_row["score"] <= all_row["ci90"][1]
    assert all_row["champion_pct"] == 100.0
    assert all_row["n_eff"] == 47 and all_row["qualified"] is True
    assert all_row["n_buckets"] == len(bucket_overalls)

    # 得分趋势：综合 = 温度/降水的均分，且逐桶键齐备；与榜单共用同一套桶得分
    st = data["score_trend"]
    assert set(st.keys()) == {"overall", "temp", "precip", "baseline"}
    for b in ("1d", "2d"):
        tv = st["temp"]["ecmwf_ifs"][b]
        pv = st["precip"]["ecmwf_ifs"][b]
        ov = st["overall"]["ecmwf_ifs"][b]
        # 综合分 = 温度/降水的均分，缺项不计（本夹具 2d 桶无按天降水样本）
        expected = round((tv + pv) / 2, 2) if tv is not None and pv is not None else tv
        assert abs(ov - expected) < 0.011, b


def test_build_report_empty_is_safe(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 1, 0, 0)
    end = datetime(2026, 8, 2, 0, 0)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, end, "2026-08")
    # 无数据时不崩溃，指标均为 None/空，得分为 None
    assert data["scorecard"]["ecmwf_ifs"]["temp_24h"]["n"] == 0
    assert data["temp_hourly"]["ecmwf_ifs"]["1d"]["rmse"] is None
    assert data["leaderboards"]["1d"][0]["score"] is None
    assert data["score_trend"]["overall"]["ecmwf_ifs"]["1d"] is None
    # 总榜在无数据时同样安全：分数与覆盖时效均为 None
    assert data["leaderboards"]["all"][0]["score"] is None
    assert data["leaderboards"]["all"][0]["lead_days"] is None


def test_overall_board_pools_all_leads_and_discloses_coverage(tmp_path, monkeypatch):
    """总榜 macro 化后的语义：各天桶等权平均 + 覆盖时效按有效样本披露。

    两个模型：short_range 只覆盖前 24h（lead 1..23，n=23，且预报与实况完全一致 ->
    满分），ecmwf_ifs 覆盖 48h（lead 1..47，温度恒偏高 1°C）。short_range 只有
    第 1 桶且桶内完美 -> macro = 100 仍第一；ecmwf 的 macro 是其两个桶的平均。
    lead_days 按实际参与计算的样本计（1 / 2 天）。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    obs = []
    for h in range(48):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 3), "rain": 1.0 if h % 6 == 0 else 0.0})
    storage.save_obs("s1", obs)

    def make_snap(model, hours, temp_bias):
        times = [iso(start + timedelta(hours=h)) for h in range(hours)]
        return {
            "issue_iso": iso(start), "station_id": "s1", "source": "test",
            "models": [model], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
            "hourly_time": times,
            "data": {model: {
                "temperature_2m": [20.0 + (h % 3) + temp_bias for h in range(hours)],
                "precipitation": [1.0 if h % 6 == 0 else 0.0 for h in range(hours)],
            }},
        }

    storage.save_forecast_snapshot("s1", "short_range", make_snap("short_range", 24, 0.0))
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", make_snap("ecmwf_ifs", 48, 1.0))

    end = start + timedelta(hours=47)
    # min_sample=1：单站单起报的按天降水只有 1 对样本，也让它入样，
    # 使"维度齐备"门槛（温度+降水都有分）可以独立于样本量被观察
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 1,
           "min_board_neff": 10, "min_board_neff_rain": 1}
    data = build_report(["s1"], ["ecmwf_ifs", "short_range"], cfg, start, end, "2026-08")

    board = data["leaderboards"]["all"]
    # short_range 只覆盖逐小时前 23h：没有任何按天降水样本 -> 温度单维度，
    # "综合"未到位 -> 未达标（哪怕桶内满分）；ecmwf 两维度齐备 -> 达标第一
    assert [r["model"] for r in board] == ["ecmwf_ifs", "short_range"]
    by_model = {r["model"]: r for r in board}
    assert by_model["short_range"]["n"] == 23 and by_model["short_range"]["lead_days"] == 1
    assert by_model["ecmwf_ifs"]["n"] == 47 and by_model["ecmwf_ifs"]["lead_days"] == 2
    assert by_model["short_range"]["qualified"] is False
    # 只有温度一个维度 -> "综合分"根本没到位：macro 只统计两维齐备的桶，故为 None
    assert by_model["short_range"]["score"] is None
    assert by_model["ecmwf_ifs"]["qualified"] is True
    # n_eff 门槛（P0-2.3）：把门槛抬到 100 -> ecmwf（n_eff=47）也不得入围，
    # 冠军频率与显著性全部清零（156 条样本争冠军的教训）
    data_100 = build_report(["s1"], ["ecmwf_ifs", "short_range"],
                            dict(cfg, min_board_neff=100), start, end, "2026-08")
    board_100 = {r["model"]: r for r in data_100["leaderboards"]["all"]}
    assert board_100["ecmwf_ifs"]["qualified"] is False
    assert board_100["ecmwf_ifs"]["champion_pct"] == 0.0
    # 完美预报满分；有偏差的源低于满分（macro 是各自桶分的平均，两个桶都 <100）
    assert 0 < by_model["ecmwf_ifs"]["score"] < 100
    # macro 语义：总榜分 == 各"两维齐备"天桶综合分的等权平均
    # （既不等于全样本池化的分，也不把单维桶当成综合分）
    from weather_eval.evaluate import overall_score, _mean_or_none
    for m in ("short_range", "ecmwf_ifs"):
        buckets = [overall_score(data["temp_hourly"][m][f"{b}d"],
                                 data["precip_score_daily"][m][f"{b}d"])
                   for b in range(1, 17)
                   if temp_score(data["temp_hourly"][m][f"{b}d"]) is not None
                   and precip_score(data["precip_score_daily"][m][f"{b}d"]) is not None]
        assert by_model[m]["score"] == _mean_or_none(buckets), m
        assert by_model[m]["n_buckets"] == len(buckets), m
    # 分时效榜：short_range 在 2d 桶无样本，分数为 None 沉底；
    # ecmwf 在 2d 桶只有温度维（无按天降水）-> 维度不齐 -> 未达标（分数保留）
    b2 = data["leaderboards"]["2d"]
    assert [r["model"] for r in b2] == ["ecmwf_ifs", "short_range"]
    assert b2[1]["score"] is None
    e2 = next(r for r in b2 if r["model"] == "ecmwf_ifs")
    assert e2["qualified"] is False and e2["temp_score"] is not None \
        and e2["precip_score"] is None
    # 展示用 n 仍是全时效池化的配对数（天桶对 lead 1..N*24 完整划分）
    for m in ("short_range", "ecmwf_ifs"):
        sum_buckets = sum(
            next(r for r in data["leaderboards"][f"{i}d"] if r["model"] == m)["n"]
            for i in range(1, 17))
        assert by_model[m]["n"] == sum_buckets, m


def test_overall_board_boundaries_and_pooling_benefit(tmp_path, monkeypatch):
    """总榜的边界语义、样本量门槛与"两维齐备才进 macro"。

    - lead_days 上限边界：lead 24h -> 1 天；lead 383h -> 16 天。
    - 展示 n == 各分桶 n 之和（天桶对 lead 完整划分）。
    - 样本量门槛（P0-2.3）：sparse 模型每个天桶只有 6 个样本但**没有任何按天
      降水样本**（逐小时只给到 6 个孤立时刻，日聚合覆盖不足被门槛挡下）——
      旧池化口径会把这些样本凑成一份"总榜分数"，macro 化 + 两维齐备门槛后
      总榜不给分、行未达标（qualified=False），杜绝"156 条样本争冠军"类假结论。
    - day1 只覆盖第 1 桶的温度（没有按天降水样本）-> 维度不齐 -> 无综合分。

    用 5 个站：按天轨道每个日偏移 5 条样本，达到 min_sample=5 的门槛，
    使"维度齐备"能被独立于样本量观察。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 1, 0, 0)
    stations = [f"s{i}" for i in range(1, 6)]
    obs = []
    for h in range(16 * 24):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 3), "rain": 1.0 if h % 6 == 0 else 0.0})
    for sid in stations:
        storage.save_obs(sid, obs)

    def make_snap(station, model, hours, temp_bias):
        times = [iso(start + timedelta(hours=h)) for h in range(hours)]
        return {
            "issue_iso": iso(start), "station_id": station, "source": "test",
            "models": [model], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
            "hourly_time": times,
            "data": {model: {
                "temperature_2m": [20.0 + (h % 3) + temp_bias for h in range(hours)],
                "precipitation": [1.0 if h % 6 == 0 else 0.0 for h in range(hours)],
            }},
        }

    for sid in stations:
        storage.save_forecast_snapshot(sid, "day1", make_snap(sid, "day1", 25, 0.0))
        storage.save_forecast_snapshot(sid, "wide", make_snap(sid, "wide", 16 * 24, 1.0))
        # sparse：6 个互不同桶的孤立时刻（lead 1/25/49/73/97/121 -> 桶 1..6）
        sparse_snap = make_snap(sid, "sparse", 0, 1.0)
        sparse_times = [1, 25, 49, 73, 97, 121]
        sparse_snap["hourly_time"] = [iso(start + timedelta(hours=h)) for h in sparse_times]
        sparse_snap["data"]["sparse"] = {
            "temperature_2m": [21.0 + (h % 3) for h in sparse_times],
            "precipitation": [1.0 if h % 6 == 0 else 0.0 for h in sparse_times],
        }
        storage.save_forecast_snapshot(sid, "sparse", sparse_snap)

    end = start + timedelta(hours=16 * 24 - 1)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(stations, ["day1", "wide", "sparse"], cfg, start, end, "2026-08")

    board = data["leaderboards"]["all"]
    info = {r["model"]: r for r in board}
    # ceil 边界：24h -> 1 天；383h -> 16 天；121h -> 6 天（n 为 5 站合计）
    assert info["day1"]["lead_days"] == 1 and info["day1"]["n"] == 24 * 5
    assert info["wide"]["lead_days"] == 16 and info["wide"]["n"] == 383 * 5
    assert info["sparse"]["lead_days"] == 6 and info["sparse"]["n"] == 6 * 5
    # day1 只有温度维 -> 无综合分；wide 两维齐备 -> 有分；sparse 无按天降水 -> 无分沉底
    assert info["day1"]["score"] is None and info["day1"]["qualified"] is False
    assert 0 < info["wide"]["score"] < 100 and info["wide"]["qualified"] is True
    assert info["sparse"]["score"] is None and info["sparse"]["qualified"] is False
    assert [r["model"] for r in board][-1] == "sparse"
    # 展示 n 不变量：总榜 n == 分桶 n 之和
    for m in ("day1", "wide", "sparse"):
        sum_buckets = sum(
            next(r for r in data["leaderboards"][f"{i}d"] if r["model"] == m)["n"]
            for i in range(1, 17))
        assert info[m]["n"] == sum_buckets, m


# ---------------------------------------------------------------- 回归：按天口径
def _save_day_snapshot(station, model, issue, day_hours, temps, precs, **extra):
    """构造并保存一份快照：day_hours 为 (day_offset, 小时数) 的逐日样本数。"""
    times, tvals, pvals = [], [], []
    for off, hours in day_hours:
        for h in range(hours):
            times.append(iso(issue + timedelta(days=off, hours=h)))
            tvals.append(temps(off, h))
            pvals.append(precs(off, h))
    snap = {
        "issue_iso": iso(issue), "station_id": station, "source": "test",
        "models": [model], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
        "hourly_time": times,
        "data": {model: {"temperature_2m": tvals, "precipitation": pvals}},
    }
    snap.update(extra)
    storage.save_forecast_snapshot(station, model, snap)
    return snap


CFG = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
       "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}


def test_daily_all_none_precip_day_not_folded_to_zero(tmp_path, monkeypatch):
    """H1 回归：某天降水全缺测时，该天绝不折算成 0.0 的假"预报无雨"进入按天降水评估。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    obs = []
    for h in range(72):   # 3 天完整观测，第 3 天有明显降雨
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 3),
                    "rain": 5.0 if h >= 48 else 0.0})
    storage.save_obs("s1", obs)

    # 快照覆盖 3 天：第 3 天（offset 2）温度完整、降水全缺测（如超出模式真实时效）
    def temps(off, h):
        return 21.0
    def precs(off, h):
        return None if off == 2 else 0.0
    _save_day_snapshot("s1", "ecmwf_ifs", start, [(0, 24), (1, 24), (2, 24)],
                       temps, precs)

    data = build_report(["s1"], ["ecmwf_ifs"], CFG, start, start + timedelta(hours=71),
                        "2026-08")
    # 修复前：offset 2 的降水以 rain_fcst=0.0 参与评估，把实况 5mm/h 判成一片"漏报"
    # 修复后：该天降水不入样（n=0、指标 None），温度指标不受影响
    p2 = data["precip_daily"]["ecmwf_ifs"]["2d"]
    assert p2["n"] == 0 and p2["ts"] is None
    t2 = data["temp_daily"]["ecmwf_ifs"]["2d"]
    assert t2["max"]["n"] == 1


def test_daily_partial_coverage_day_gated(tmp_path, monkeypatch):
    """覆盖门槛：预报只覆盖当天部分小时（模式时效边界）时，该天不入按天评估。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    obs = []
    for h in range(24 * 3):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 3), "rain": 0.0})
    storage.save_obs("s1", obs)

    # offset 1 仅覆盖 9 个小时（< daily_min_hours=20），offset 2 完整
    def temps(off, h):
        return 21.0 if (off == 1 and h >= 9) or off == 2 else None
    def precs(off, h):
        return 0.0 if temps(off, h) is not None else None
    _save_day_snapshot("s1", "ecmwf_ifs", start, [(0, 24), (1, 24), (2, 24)],
                       temps, precs)

    data = build_report(["s1"], ["ecmwf_ifs"], CFG, start, start + timedelta(hours=71),
                        "2026-08")
    assert data["temp_daily"]["ecmwf_ifs"]["1d"]["max"]["n"] == 0   # 9/24 小时 -> 剔除
    assert data["temp_daily"]["ecmwf_ifs"]["2d"]["max"]["n"] == 1   # 完整天保留


def test_daily_multi_model_snapshot_attributed_separately(tmp_path, monkeypatch):
    """M5 回归：多模型共享时间轴的存档，按天聚合必须按模型分别展开，
    不能把所有模型混算后全部记到最后一个模型名下。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    obs = []
    for h in range(48):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 3), "rain": 0.0})
    storage.save_obs("s1", obs)

    times = [iso(start + timedelta(hours=h)) for h in range(48)]
    snap = {
        "issue_iso": iso(start), "station_id": "s1", "source": "test",
        "models": ["ecmwf_ifs", "other_model"], "grid_lat": 23.0, "grid_lon": 111.0,
        "elevation": 50, "hourly_time": times,
        "data": {
            "ecmwf_ifs": {"temperature_2m": [20.0] * 48, "precipitation": [0.0] * 48},
            "other_model": {"temperature_2m": [30.0] * 48, "precipitation": [0.0] * 48},
        },
    }
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap)   # 存档目录只挂一个模型

    cfg = dict(CFG, min_sample=1)   # 单样本天也出指标（本测试只关心按模型分账）
    data = build_report(["s1"], ["ecmwf_ifs", "other_model"], cfg,
                        start, start + timedelta(hours=47), "2026-08")
    # 两个模型各自与观测的偏差被分开记账（修复前 other_model 独占全部按天样本）
    assert data["temp_daily"]["ecmwf_ifs"]["1d"]["max"]["n"] == 1
    assert data["temp_daily"]["other_model"]["1d"]["max"]["n"] == 1
    # 日最高温：观测 max=22°C（20/21/22 周期），预报恒 20 / 30
    assert data["temp_daily"]["ecmwf_ifs"]["1d"]["max"]["rmse"] == 2.0   # |20-22|
    assert data["temp_daily"]["other_model"]["1d"]["max"]["rmse"] == 8.0  # |30-22|


def test_coverage_denominator_truncated_to_first_obs(tmp_path, monkeypatch):
    """L9 回归：覆盖率分母按各站实际有实况的时段截断，不把接入前的整月时段
    算成缺失。观测从窗口第 3 天开始且逐小时完整 -> 覆盖率应为 100%。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 1, 0, 0)
    obs = []
    for h in range(3 * 24, 6 * 24):   # 08-04 00:00 起 3 天完整逐小时
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0, "rain": 0.0})
    storage.save_obs("s1", obs)

    from weather_eval.evaluate import _coverage
    # 窗口 08-01 起：若不截断，分母是 08-01 起的 144 小时；截断后分母 = 08-04 起的 72 小时
    cov = _coverage(["s1"], start, datetime(2026, 8, 6, 23, 0))
    assert cov["got_hours"] == cov["expected_hours"] == 3 * 24
    assert cov["coverage_pct"] == 100.0
    assert cov["first_obs"] == "2026-08-04T00:00"


def test_model_caveats_surfaced_from_snapshot_meta(tmp_path, monkeypatch):
    """M2 回归：快照留档的"最近城市吸附"元数据进入报告 meta.model_caveats。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    obs = [{"time": iso(start + timedelta(hours=h)), "temp": 20.0, "rain": 0.0}
           for h in range(24)]
    storage.save_obs("s1", obs)

    def temps(off, h):
        return 20.0
    def precs(off, h):
        return 0.0
    snap = _save_day_snapshot("s1", "accuweather_v1", start, [(0, 24)], temps, precs,
                              location_key="323883", location_name="梧州市",
                              location_distance_km=38.2)
    assert snap["location_distance_km"] == 38.2

    data = build_report(["s1"], ["accuweather_v1"], CFG, start,
                        start + timedelta(hours=23), "2026-08")
    note = data["meta"]["model_caveats"]["accuweather_v1"]
    assert "38.2" in note and "梧州" in note


def test_model_caveats_lists_all_station_names_not_just_first(tmp_path, monkeypatch):
    """吸附距离是全站均值，而各站吸附到的城市不同——只写第一站名会误导读者
    （MSN/中国天气网 4 站各吸附到不同城市，该缺陷因新源接入而显性化）。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    for sid in ("s1", "s2"):
        obs = [{"time": iso(start + timedelta(hours=h)), "temp": 20.0, "rain": 0.0}
               for h in range(24)]
        storage.save_obs(sid, obs)

    for sid, nm, km in (("s1", "万秀", 1.8), ("s2", "wanning", 7.0)):
        _save_day_snapshot(sid, "msn_v1", start, [(0, 24)],
                           lambda off, h: 20.0, lambda off, h: 0.0,
                           location_name=nm, location_distance_km=km)

    data = build_report(["s1", "s2"], ["msn_v1"], CFG, start,
                        start + timedelta(hours=23), "2026-08")
    note = data["meta"]["model_caveats"]["msn_v1"]
    assert "4.4" in note                     # 均值 (1.8+7.0)/2
    assert "万秀" in note and "wanning" in note   # 两站城市名都必须出现


# ------------------------------------------------- 逐日预报补位（2026-09 新增）
def _snap_with_daily(model, issue, hourly_days, temps, precs,
                     daily_days, daily_max, daily_min, daily_rain, **extra):
    """构造带可选逐日预报块的快照。

    hourly_days: [(日偏移, 该日逐小时点数)]；daily_*: 与 daily_days 等长的日产品值。
    """
    times, tvals, pvals = [], [], []
    for off, hours in hourly_days:
        for h in range(hours):
            times.append(iso(issue + timedelta(days=off, hours=h)))
            tvals.append(temps(off, h))
            pvals.append(precs(off, h))
    snap = {
        "issue_iso": iso(issue), "station_id": "s1", "source": "test",
        "models": [model], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
        "hourly_time": times,
        "data": {model: {"temperature_2m": tvals, "precipitation": pvals}},
    }
    if daily_days is not None:
        snap["daily_time"] = [str(issue.date() + timedelta(days=off)) for off in daily_days]
        snap["daily"] = {model: {"temp_max": daily_max, "temp_min": daily_min,
                                 "precipitation": daily_rain}}
    snap.update(extra)
    return snap


def _obs_days(start, days, temp=20.0, rain=0.0):
    obs = []
    for h in range(days * 24):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": temp + (h % 3), "rain": rain})
    return obs


def test_daily_fallback_extends_beyond_hourly_coverage(tmp_path, monkeypatch):
    """核心回归：逐小时只覆盖前 2 天，第 3~5 天靠源自带日产品继续做按天评估。

    补位前 offset 3~5 完全无样本（按天评估止于第 2 天）；补位后这些日子的
    日最高/最低温与日降水都入样，且来源标记为 "daily"。
    """
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    obs = []
    for h in range(5 * 24):     # 5 天完整观测：前 2 天无雨，后 3 天有雨
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 3),
                    "rain": 5.0 if h >= 2 * 24 else 0.0})
    storage.save_obs("s1", obs)

    snap = _snap_with_daily(
        "ecmwf_ifs", start,
        hourly_days=[(0, 24), (1, 24)],          # 逐小时只覆盖到 offset 1（08-25）
        temps=lambda off, h: 21.0, precs=lambda off, h: 0.0,
        # 日产品覆盖 offset 2~4（08-26..28）—— 完美预报：日最高 22 / 最低 20 / 日雨 120
        daily_days=[2, 3, 4],
        daily_max=[22.0, 22.0, 22.0],
        daily_min=[20.0, 20.0, 20.0],
        daily_rain=[120.0, 120.0, 120.0],
    )
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap)

    cfg = dict(CFG, min_sample=1)   # 单样本天也要出指标（本测试只关心入样与来源）
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, start + timedelta(hours=5 * 24 - 1),
                        "2026-08")
    # 观测侧：offset 2~4 每天 5mm/h × 24h = 120mm，日最高 22 / 最低 20
    for off in (2, 3, 4):
        t = data["temp_daily"]["ecmwf_ifs"][f"{off}d"]
        assert t["max"]["n"] == 1, off          # 补位前为 0
        assert abs(t["max"]["rmse"]) < 1e-9     # 日最高完全吻合
        assert abs(t["min"]["rmse"]) < 1e-9
        p = data["precip_daily"]["ecmwf_ifs"][f"{off}d"]
        assert p["n"] == 1 and p["ts"] == 1.0   # 有雨且报中
    # 来源构成：offset 2~4 三桶全部标记为日产品补位
    mix = data["daily_source_mix"]["ecmwf_ifs"]
    for off in (2, 3, 4):
        assert mix[f"{off}d"] == {"temp": {"hourly": 0, "daily": 1},
                                  "rain": {"hourly": 0, "daily": 1}}, off
    # 逐小时轨道完全不受影响：日产品绝不反推逐小时样本
    assert data["temp_hourly"]["ecmwf_ifs"]["2d"]["n"] == 23   # 仅 lead 25..47
    assert data["temp_hourly"]["ecmwf_ifs"]["3d"]["n"] == 0
    assert data["leaderboards"]["all"][0]["lead_days"] == 2


def test_daily_prefers_hourly_aggregation_when_covered(tmp_path, monkeypatch):
    """逐小时覆盖充足的日子继续用逐小时聚合——补位不得改变已有口径的样本。

    构造一个"逐小时聚合与日产品给出不同答案"的日子：日最高 22（逐小时）/ 30（日产品）。
    若实现误把日产品当首选，指标会立刻翻脸。
    """
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    storage.save_obs("s1", _obs_days(start, 2, temp=20.0))

    snap = _snap_with_daily(
        "ecmwf_ifs", start,
        hourly_days=[(0, 24), (1, 24)],
        temps=lambda off, h: 22.0, precs=lambda off, h: 0.0,   # 逐小时恒温 22
        daily_days=[0, 1], daily_max=[30.0, 30.0], daily_min=[30.0, 30.0],
        daily_rain=[99.0, 99.0],                                # 日产品故意给出不同答案
    )
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap)

    data = build_report(["s1"], ["ecmwf_ifs"], dict(CFG, min_sample=1), start,
                        start + timedelta(hours=47), "2026-08")
    t1 = data["temp_daily"]["ecmwf_ifs"]["1d"]
    # 观测日最高 22（20+(h%3) -> 22），逐小时聚合亦为 22 -> RMSE=0；若误用日产品则为 8
    assert abs(t1["max"]["rmse"]) < 1e-9
    assert data["daily_source_mix"]["ecmwf_ifs"]["1d"]["temp"] == {"hourly": 1, "daily": 0}


def test_daily_fallback_fills_partially_covered_day(tmp_path, monkeypatch):
    """时效边界日（逐小时只有部分小时）由日产品补位：原本这整天会被覆盖门槛剔除。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    storage.save_obs("s1", _obs_days(start, 2, temp=20.0))

    # offset 1 只预报了 9 个小时（< daily_min_hours=20）
    def temps(off, h):
        return 22.0 if off == 0 or (off == 1 and h < 9) else None
    snap = _snap_with_daily(
        "ecmwf_ifs", start,
        hourly_days=[(0, 24), (1, 24)],
        temps=temps, precs=lambda off, h: 0.0 if temps(off, h) is not None else None,
        daily_days=[1], daily_max=[22.0], daily_min=[20.0], daily_rain=[0.0],
    )
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap)

    data = build_report(["s1"], ["ecmwf_ifs"], CFG, start, start + timedelta(hours=47),
                        "2026-08")
    mix = data["daily_source_mix"]["ecmwf_ifs"]["1d"]
    assert mix == {"temp": {"hourly": 0, "daily": 1}, "rain": {"hourly": 0, "daily": 1}}
    assert data["temp_daily"]["ecmwf_ifs"]["1d"]["max"]["n"] == 1
    # 关掉开关则回到旧行为：该天被剔除
    off_cfg = dict(CFG, daily_source_fallback=False)
    data_off = build_report(["s1"], ["ecmwf_ifs"], off_cfg, start,
                            start + timedelta(hours=47), "2026-08")
    assert data_off["temp_daily"]["ecmwf_ifs"]["1d"]["max"]["n"] == 0
    assert data_off["daily_source_mix"] == {}


def test_daily_fallback_never_invents_missing_values(tmp_path, monkeypatch):
    """补位也不得把缺测伪装成数值：日产品该要素为 null 时该要素照样不入样。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    obs = []
    for h in range(3 * 24):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 3), "rain": 5.0 if h >= 2 * 24 else 0.0})
    storage.save_obs("s1", obs)

    snap = _snap_with_daily(
        "ecmwf_ifs", start,
        hourly_days=[(0, 24), (1, 24)],
        temps=lambda off, h: 21.0, precs=lambda off, h: 0.0,
        daily_days=[2], daily_max=[22.0], daily_min=[20.0], daily_rain=[None],
    )
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap)

    data = build_report(["s1"], ["ecmwf_ifs"], CFG, start, start + timedelta(hours=3 * 24 - 1),
                        "2026-08")
    p2 = data["precip_daily"]["ecmwf_ifs"]["2d"]
    assert p2["n"] == 0 and p2["ts"] is None     # 日产品降水缺测 -> 绝不折算 0.0
    t2 = data["temp_daily"]["ecmwf_ifs"]["2d"]
    assert t2["max"]["n"] == 1                   # 温度照常入样
    # 降水缺测不影响温度来源标记，两者分开记账（未配对成样本的要素不出现在构成里）
    assert data["daily_source_mix"]["ecmwf_ifs"]["2d"] == {"temp": {"hourly": 0, "daily": 1}}


def test_daily_fallback_multimodel_snapshot_attributed_per_model(tmp_path, monkeypatch):
    """多模型共享时间轴的存档带日产品块时，各模型只取自己的那一列。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    storage.save_obs("s1", _obs_days(start, 3, temp=20.0))
    times = [iso(start + timedelta(hours=h)) for h in range(24)]
    snap = {
        "issue_iso": iso(start), "station_id": "s1", "source": "test",
        "models": ["ecmwf_ifs", "other_model"],
        "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
        "hourly_time": times,
        "data": {
            "ecmwf_ifs": {"temperature_2m": [22.0] * 24, "precipitation": [0.0] * 24},
            "other_model": {"temperature_2m": [22.0] * 24, "precipitation": [0.0] * 24},
        },
        "daily_time": [str(start.date() + timedelta(days=1)),
                       str(start.date() + timedelta(days=2))],
        "daily": {
            "ecmwf_ifs": {"temp_max": [22.0, 22.0], "temp_min": [20.0, 20.0],
                          "precipitation": [0.0, 0.0]},
            "other_model": {"temp_max": [40.0, 40.0], "temp_min": [40.0, 40.0],
                            "precipitation": [50.0, 50.0]},
        },
    }
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap)

    cfg = dict(CFG, min_sample=1)
    data = build_report(["s1"], ["ecmwf_ifs", "other_model"], cfg,
                        start, start + timedelta(hours=3 * 24 - 1), "2026-08")
    # 两家各自记账：ecmwf 的日产品与观测吻合（RMSE 0），other_model 明显偏差（18）
    t2 = data["temp_daily"]
    assert abs(t2["ecmwf_ifs"]["2d"]["max"]["rmse"]) < 1e-9
    assert abs(t2["other_model"]["2d"]["max"]["rmse"] - 18.0) < 1e-9
    # 降水（雨量 RMSE 在全零样本上仍有定义）：ecmwf 与实况一致 -> 0；
    # other_model 用了自己那一列的 50mm -> 50。若两家混用同一列，两者会相等
    assert abs(data["precip_daily"]["ecmwf_ifs"]["2d"]["rmse"]) < 1e-9
    assert abs(data["precip_daily"]["other_model"]["2d"]["rmse"] - 50.0) < 1e-9


def test_daily_block_missing_or_malformed_is_backward_compatible(tmp_path, monkeypatch):
    """旧存档无 daily 块 / 结构畸形：行为完全回退到纯逐小时聚合，且不崩溃。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    storage.save_obs("s1", _obs_days(start, 2, temp=20.0))

    base = _snap_with_daily("ecmwf_ifs", start, [(0, 24), (1, 24)],
                            lambda off, h: 22.0, lambda off, h: 0.0,
                            None, None, None, None)
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", base)   # 无 daily 块

    cfg = dict(CFG, min_sample=1)
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start, start + timedelta(hours=47),
                        "2026-08")
    assert data["temp_daily"]["ecmwf_ifs"]["1d"]["max"]["n"] == 1
    assert data["daily_source_mix"]["ecmwf_ifs"]["1d"]["temp"] == {"hourly": 1, "daily": 0}

    # 畸形块不得拖垮报告：每份快照只有一种畸形形态，故各自独立跑一次
    for bad in ({"daily_time": ["2026-08-25"], "daily": {}},                    # 缺模型
                {"daily_time": "2026-08-25", "daily": {"ecmwf_ifs": {}}},       # 时间轴非数组
                {"daily_time": ["2026-08-25"], "daily": {"ecmwf_ifs": {
                    "temp_max": ["x"], "temp_min": [None], "precipitation": [0.0]}}},  # 非数值
                {"daily_time": [123], "daily": {"ecmwf_ifs": {                  # 日期非字符串
                    "temp_max": [1.0], "temp_min": [1.0], "precipitation": [0.0]}}}):
        import shutil
        shutil.rmtree(tmp_path / "forecasts", ignore_errors=True)
        broken = dict(base); broken.update(bad)
        storage.save_forecast_snapshot("s1", "ecmwf_ifs", broken)
        d = build_report(["s1"], ["ecmwf_ifs"], cfg, start, start + timedelta(hours=47),
                         "2026-08")
        assert d["temp_daily"]["ecmwf_ifs"]["1d"]["max"]["n"] == 1   # 逐小时口径不受影响
        assert d["temp_daily"]["ecmwf_ifs"]["1d"]["max"]["rmse"] == 0.0


def test_daily_fallback_respects_observation_gate(tmp_path, monkeypatch):
    """补位不放松观测侧门槛：实况当天不足 daily_min_hours 时该天照样不入样。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    obs = []   # 第 3 天只有 10 个小时的实况
    for h in range(2 * 24 + 10):
        obs.append({"time": iso(start + timedelta(hours=h)), "temp": 20.0, "rain": 0.0})
    storage.save_obs("s1", obs)

    snap = _snap_with_daily("ecmwf_ifs", start, [(0, 24), (1, 24)],
                            lambda off, h: 21.0, lambda off, h: 0.0,
                            [2], [21.0], [21.0], [0.0])
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap)

    data = build_report(["s1"], ["ecmwf_ifs"], CFG, start,
                        start + timedelta(hours=2 * 24 + 9), "2026-08")
    assert data["temp_daily"]["ecmwf_ifs"]["2d"]["max"]["n"] == 0
    assert data["daily_source_mix"]["ecmwf_ifs"].get("2d") is None


def test_daily_all_none_precip_excluded_even_if_gate_disabled(tmp_path, monkeypatch):
    """第二轮审查回归：即使 daily_min_hours 被配成 0（门槛关闭），降水全缺测的
    天也绝不折算成 0.0 —— "缺测不折算"是无条件下限，门槛只是额外的公平性要求。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    obs = []
    for h in range(72):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 3),
                    "rain": 5.0 if h >= 48 else 0.0})
    storage.save_obs("s1", obs)

    def temps(off, h):
        return 21.0
    def precs(off, h):
        return None if off == 2 else 0.0
    _save_day_snapshot("s1", "ecmwf_ifs", start, [(0, 24), (1, 24), (2, 24)],
                       temps, precs)

    cfg = dict(CFG, daily_min_hours=0, min_sample=1)
    data = build_report(["s1"], ["ecmwf_ifs"], cfg, start,
                        start + timedelta(hours=71), "2026-08")
    p2 = data["precip_daily"]["ecmwf_ifs"]["2d"]
    assert p2["n"] == 0 and p2["ts"] is None   # 假 0.0 依然被拒绝


# ------------------------------------------------- 统计推断层（2026-09-06 新增）
def test_lead_days_uses_valid_pairs_not_series_length(tmp_path, monkeypatch):
    """P1-1 回归：「覆盖时效」按实际参与计算的样本（两侧同时非缺测）计。

    cma_grapes 类源的部分模型序列尾部是大片 null：全量 record 的 max(lead)
    会把尾部虚报成覆盖（旧实现 384h 序列只有 119h 有效却披露 16 天）。
    构造 lead 1..96 中只有前 48h 有温度值 -> 覆盖时效应为 2 天而非 4 天。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 24, 0, 0)
    obs = []
    for h in range(48):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 3), "rain": 0.0})
    storage.save_obs("s1", obs)

    times = [iso(start + timedelta(hours=h)) for h in range(96)]
    temps = [20.0 + (h % 3) if h < 48 else None for h in range(96)]
    snap = {
        "issue_iso": iso(start), "station_id": "s1", "source": "test",
        "models": ["trailing_null"], "grid_lat": 23.0, "grid_lon": 111.0,
        "elevation": 50, "hourly_time": times,
        "data": {"trailing_null": {"temperature_2m": temps,
                                   "precipitation": [0.0] * 96}},
    }
    storage.save_forecast_snapshot("s1", "trailing_null", snap)

    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
    data = build_report(["s1"], ["trailing_null"], cfg, start,
                        start + timedelta(hours=95), "2026-08")
    row = data["leaderboards"]["all"][0]
    assert row["lead_days"] == 2          # 温度有效样本最远到 lead 48 -> 2 天
    assert row["n"] == 47                 # 只有 lead 1..47 有观测可配对
    # 观测只有 2 天：观测缺位同样限制有效覆盖，绝不按 96h 序列长度虚报


def test_neff_effective_sample_size_and_gate():
    """P1-2：有效样本量与门槛。

    - AR(1) 误差（ρ≈0.9）的 n_eff ≈ n·(1−ρ)/(1+ρ)（约 1/19）；
    - 白噪声误差 n_eff ≈ n；
    - min_sample 门槛作用于 n_eff：名义 n 达标但 n_eff 不达标同样不出结论。"""
    import numpy as np
    from weather_eval.evaluate import temp_metrics, _n_eff_temp
    from weather_eval.stats import effective_n

    rng = np.random.default_rng(7)
    ar = np.empty(600)
    ar[0] = 0.0
    for i in range(1, 600):
        ar[i] = 0.9 * ar[i - 1] + rng.normal(0, 0.2)
    assert 20 < effective_n(ar) < 80          # 理论 ~31
    white = rng.normal(0, 1, 600)
    assert effective_n(white) > 400           # 无自相关几乎不折损

    # 名义 n=600 但误差强自相关 -> n_eff < min_sample 时指标置 None（不报假精度）
    obs = [20.0] * 600
    fcst = [20.0 + float(e) for e in ar]
    m = temp_metrics(obs, fcst, [1, 2], 5, n_eff=3)
    assert m["rmse"] is None and m["n"] == 600 and m["n_eff"] == 3
    # n_eff 达标则正常出指标
    m2 = temp_metrics(obs, fcst, [1, 2], 5, n_eff=500)
    assert m2["rmse"] is not None

    # _n_eff_temp 沿时间轴去重（同一有效时刻只保留最新一版起报）
    recs = []
    for h in range(100):
        recs.append({"station": "s1", "valid_iso": f"2026-08-{h // 24 + 1:02d}T{h % 24:02d}:00",
                     "lead": 24, "temp_obs": 20.0, "temp_fcst": 20.1})
    recs.append({"station": "s1", "valid_iso": "2026-08-02T00:00", "lead": 48,
                 "temp_obs": 20.0, "temp_fcst": 25.0})   # 旧起报对同一时刻（被最新版覆盖）
    assert _n_eff_temp(recs) == 100                      # 100 个不同时刻、误差近常数


def test_temp_metrics_per_station_fisher_z(tmp_path):
    """P1-3：r/slope 站内计算后 Fisher-z / n 加权合并；池化值并列披露。

    两站：A 站误差与观测强相关（r≈1）、B 站反相关（r≈−1）。池化后两站的
    站间均值差会主导 r（池化 r 接近 1，纯粹的"复现气候差异"红利）；
    站内合并的 r 应显著低于池化值——这才度量"站内起伏同步程度"。"""
    import numpy as np
    from weather_eval.evaluate import temp_metrics

    rng = np.random.default_rng(11)
    n = 200
    obs_a = 20 + rng.normal(0, 2, n)          # A 站均温 20
    fcst_a = obs_a + rng.normal(0, 0.3, n)    # 同步起伏 r≈1
    obs_b = 32 + rng.normal(0, 2, n)          # B 站均温 32（站间差 12°C）
    # B 站镜像预报（绕均值 32 反相）：站内 r≈−1，与 A 站 r≈+1 合并后应接近 0
    fcst_b = 64.0 - obs_b + rng.normal(0, 0.3, n)

    groups = [(obs_a, fcst_a), (obs_b, fcst_b)]
    m = temp_metrics(np.concatenate([obs_a, obs_b]),
                     np.concatenate([fcst_a, fcst_b]),
                     [1, 2], 5, groups=groups)
    assert m["r_pooled"] is not None and m["r"] is not None
    # 池化 r 因站间均值差虚高（接近 1）；站内合并（+1 与 −1 抵消）应接近 0
    assert m["r_pooled"] > 0.9
    assert m["r"] < m["r_pooled"] - 0.5
    assert abs(m["r"]) < 0.3
    # 单组时站内合并退化为该组自身
    m1 = temp_metrics(obs_a, fcst_a, [1, 2], 5, groups=[(obs_a, fcst_a)])
    assert abs(m1["r"] - m1["r_pooled"]) < 1e-9


def test_numpy_fast_paths_match_cyeva(tmp_path, monkeypatch):
    """快速路径（曲线 rmse/acc2、按天二分类 6 项）与 cyeva 全量路径同数值。

    用真实随机数据对拍：报告里的数字不允许"曲线一套、表格一套"。"""
    import numpy as np
    from weather_eval.evaluate import (temp_curve_metrics, precip_binary_metrics,
                                       precip_metrics)
    rng = np.random.default_rng(3)
    o = rng.normal(25, 2, 500).round(1)
    f = o + rng.normal(0, 1.5, 500).round(1)

    fast = temp_curve_metrics(o, f, 5)
    full = temp_metrics(o, f, [2], 5)
    assert fast["rmse"] == full["rmse"] and fast["acc2"] == full["acc2"]

    # 按天二分类（1mm 阈值）与 cyeva threshold 路径
    ro = (rng.random(400) < 0.15).astype(float) * rng.uniform(0, 30, 400).round(1)
    rf = (rng.random(400) < 0.3).astype(float) * rng.uniform(0, 25, 400).round(1)
    mine = precip_binary_metrics(ro, rf, 1.0, 5)
    cy = precip_metrics(ro, rf, 1.0, 5)
    for k in ("acc", "pod", "far", "ts", "ets", "bias"):
        assert mine[k] == cy[k], (k, mine[k], cy[k])


def test_bootstrap_ci_deterministic_and_honest(tmp_path, monkeypatch):
    """P0-2：bootstrap 置信区间。

    - 同一天数据 + 固定种子 -> 完全相同的 CI（可复现性）；
    - 噪声数据 -> CI 有真实宽度且冠军频率不武断（无一家 100% 时不应报 100%）。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 1, 0, 0)
    import numpy as np
    rng = np.random.default_rng(5)
    obs = []
    for h in range(16 * 24):
        t = start + timedelta(hours=h)
        obs.append({"time": iso(t), "temp": 20.0 + (h % 3),
                    "rain": 1.0 if h % 12 == 0 else 0.0})
    storage.save_obs("s1", obs)
    for model, noise in (("noisy_a", 1.0), ("noisy_b", 1.2)):
        times = [iso(start + timedelta(hours=h)) for h in range(16 * 24 - 1)]
        vals = [20.0 + (h % 3) + float(rng.normal(0, noise)) for h in range(16 * 24 - 1)]
        rains = [1.0 if h % 12 == 0 else 0.0 for h in range(16 * 24 - 1)]
        snap = {"issue_iso": iso(start), "station_id": "s1", "source": "test",
                "models": [model], "grid_lat": 23.0, "grid_lon": 111.0,
                "elevation": 50, "hourly_time": times,
                "data": {model: {"temperature_2m": vals,
                                 "precipitation": rains}}}
        storage.save_forecast_snapshot("s1", model, snap)

    # min_sample=1：单站单起报的按天降水只有 1 对样本，也让它入样，
    # 使两源达到"两维度齐备"的入围条件（本测试关注 CI 与冠军频率本身）
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 1,
           "bootstrap_runs": 200, "min_board_neff_rain": 1}
    d1 = build_report(["s1"], ["noisy_a", "noisy_b"], cfg, start,
                      start + timedelta(hours=16 * 24 - 2), "2026-08")
    d2 = build_report(["s1"], ["noisy_a", "noisy_b"], cfg, start,
                      start + timedelta(hours=16 * 24 - 2), "2026-08")
    r1 = {r["model"]: r for r in d1["leaderboards"]["all"]}
    r2 = {r["model"]: r for r in d2["leaderboards"]["all"]}
    for m in ("noisy_a", "noisy_b"):
        assert r1[m]["ci90"] == r2[m]["ci90"]
        assert r1[m]["ci90"][0] < r1[m]["ci90"][1]        # 噪声数据 CI 有宽度
    champ_total = sum(r["champion_pct"] for r in r1.values())
    assert 99.0 <= champ_total <= 101.0                   # 冠军频率归一


def test_weight_sensitivity_in_meta(tmp_path, monkeypatch):
    """P0-2.2：权重敏感性进报告 meta——冠军分布、扰动次数齐备。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 1, 0, 0)
    obs = [{"time": iso(start + timedelta(hours=h)), "temp": 20.0 + (h % 3),
            "rain": 1.0 if h % 12 == 0 else 0.0} for h in range(48)]
    storage.save_obs("s1", obs)
    for model, bias in (("m_a", 0.0), ("m_b", 1.0)):
        times = [iso(start + timedelta(hours=h)) for h in range(47)]
        snap = {"issue_iso": iso(start), "station_id": "s1", "source": "test",
                "models": [model], "grid_lat": 23.0, "grid_lon": 111.0,
                "elevation": 50, "hourly_time": times,
                "data": {model: {
                    "temperature_2m": [20.0 + (h % 3) + bias for h in range(47)],
                    "precipitation": [1.0 if h % 12 == 0 else 0.0 for h in range(47)]}}}
        storage.save_forecast_snapshot("s1", model, snap)
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 1,
           "sensitivity_runs": 100, "min_board_neff_rain": 1}
    data = build_report(["s1"], ["m_a", "m_b"], cfg, start,
                        start + timedelta(hours=46), "2026-08")
    ws = data["meta"]["weight_sensitivity"]
    assert ws["runs"] == 100
    assert ws["champions"] and sum(c["pct"] for c in ws["champions"]) <= 100.5
    names = {c["model"] for c in ws["champions"]}
    assert names <= {"m_a", "m_b"}


def test_bootstrap_rain_sample_size_counts_all_cells_not_hits():
    """P0-1 回归：降水的样本量门槛用 h+fa+mi+c，不是命中数 hits。

    构造一个"真实样本 32 条、但命中只有 2 次"的桶：晴天多的桶命中数天然稀少，
    按 hits 判门槛会把 32 条样本的降水分误剔，桶分退化为纯温度分（虚高）。
    """
    import numpy as np
    from weather_eval import stats
    from weather_eval.evaluate import TEMP_SCORE_PARTS, PRECIP_SCORE_PARTS

    n_m, n_b, n_s, n_d = 1, 2, 1, 3
    T = np.zeros((n_m, n_b, n_s, n_d, len(stats._TEMP_STATS)))
    R = np.zeros((n_m, n_b, n_s, n_d, len(stats._RAIN_STATS)))
    # 桶 1：三天合计 h=2 / fa=0 / mi=6 / c=24 → 真实样本 32（> min_sample=5），
    # 但命中数只有 2（落在 1~4 之间，正是旧代码误判为"样本不足"的区间）
    R[0, 0, 0, 0] = [1, 0, 2, 8]
    R[0, 0, 0, 1] = [1, 0, 2, 8]
    R[0, 0, 0, 2] = [0, 0, 2, 8]
    # 温度：同样 32 条样本，恒定误差 0.5°C（r/slope 因零方差退化，按缺项处理）
    for d, n in enumerate((11, 11, 10)):
        T[0, 0, 0, d] = [n, n * 0.25, n * 0.5, n * 0.5, n, n,
                         n * 20.0, n * 19.5, n * 390.0, n * 400.0, n * 380.25]
    W = np.ones((1, n_d))                      # 退化权重：不做重采样
    macro = stats.macro_scores_from_weights(
        W, T, R, TEMP_SCORE_PARTS, PRECIP_SCORE_PARTS, min_sample=5)
    temp_only = float(stats._temp_scores_from_aggregate(
        stats.aggregate_day_stats(W, T), TEMP_SCORE_PARTS)[0, 0, 0])
    rain = float(stats._rain_scores_from_aggregate(
        stats.aggregate_day_stats(W, R), PRECIP_SCORE_PARTS)[0, 0, 0])
    assert abs(rain - 34.5) < 0.05, rain          # 降水分（ETS/TS/POD/FAR/BIAS 加权）
    # 桶分必须含降水分：若被误剔，macro 会等于纯温度分
    assert abs(macro[0, 0] - (temp_only + rain) / 2) < 0.05
    assert abs(macro[0, 0] - temp_only) > 30.0    # 与"只剩温度分"明确区分


def test_degenerate_bootstrap_reproduces_point_estimate(tmp_path, monkeypatch):
    """不变量：权重全置 1 的退化 bootstrap 必须精确复现点估计的桶 macro 分。

    这条不变量一次性兜住所有"bootstrap 与点估计口径漂移"类缺陷（P0-1 的教训：
    注释里写了"与点估计缺项口径一致"，却没有机器校验）。容差 0.05 分——
    点估计的指标经 round3 后再算分，与未舍入的聚合路径有 ≤0.02 的差。
    """
    import numpy as np
    from weather_eval import stats
    from weather_eval.evaluate import (TEMP_SCORE_PARTS, PRECIP_SCORE_PARTS,
                                       collect, temp_score, precip_score)
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 1, 0, 0)
    n_days = 8
    obs = [{"time": iso(start + timedelta(hours=h)),
            "temp": 24.0 + 4.0 * np.sin(h / 6.0),
            "rain": (2.0 if h % 24 in (0, 1) and (h // 24) % 3 == 0 else 0.0)}
           for h in range(n_days * 24)]
    storage.save_obs("s1", obs)
    storage.save_obs("s2", obs)
    models = ["m_a", "m_b"]
    for mi, model in enumerate(models):
        rng = np.random.default_rng(9 + mi)
        for issue_d in range(n_days - 1):     # 多个起报轮次 → 桶内样本充足
            issue = start + timedelta(days=issue_d)
            times, vals, rains = [], [], []
            for h in range(1, 16 * 24):
                t = issue + timedelta(hours=h)
                times.append(iso(t))
                vals.append(24.0 + 4.0 * np.sin((issue_d * 24 + h) / 6.0)
                            + float(rng.normal(0, 0.6 + 0.3 * mi)))
                rains.append(2.0 if (issue_d * 24 + h) % 72 in (0, 1) else 0.0)
            storage.save_forecast_snapshot(
                "s1", model, {"issue_iso": iso(issue), "station_id": "s1",
                              "source": "test", "models": [model], "grid_lat": 23.0,
                              "grid_lon": 111.0, "elevation": 50, "hourly_time": times,
                              "data": {model: {"temperature_2m": vals,
                                               "precipitation": rains}}})
            storage.save_forecast_snapshot("s2", model, {
                "issue_iso": iso(issue), "station_id": "s2", "source": "test",
                "models": [model], "grid_lat": 23.0, "grid_lon": 111.0,
                "elevation": 50, "hourly_time": times,
                "data": {model: {"temperature_2m": vals, "precipitation": rains}}})
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5,
           "bootstrap_runs": 60, "sensitivity_runs": 30}
    end = start + timedelta(hours=n_days * 24 - 1)
    data = build_report(["s1", "s2"], models, cfg, start, end, "2026-08")
    hourly, daily = collect(["s1", "s2"], models, start, end, 16, 16, 20, True)
    days, T, R = stats.build_day_stat_tables(hourly, daily, models, 16, 1.0)
    W = np.ones((1, len(days)))
    tv = np.array([[temp_score(data["temp_hourly"][m].get(f"{b}d") or {}) is not None
                    for b in range(1, 17)] for m in models])
    rv = np.array([[precip_score(data["precip_score_daily"][m].get(f"{b}d") or {}) is not None
                    for b in range(1, 17)] for m in models])
    macro = stats.macro_scores_from_weights(
        W, T, R, TEMP_SCORE_PARTS, PRECIP_SCORE_PARTS, 5, tv, rv, tv & rv)
    board = {r["model"]: r for r in data["leaderboards"]["all"]}
    checked = 0
    for mi, m in enumerate(models):
        if board[m]["score"] is None:
            continue
        assert abs(macro[0, mi] - board[m]["score"]) < 0.05, (m, macro[0, mi], board[m]["score"])
        checked += 1
    assert checked >= 1
    # 同一条不变量对"共同窗口分"也必须成立——名次换尺子，不变量不能跟着换
    common = [b - 1 for b in (data["meta"].get("common_window") or {}).get("buckets") or []]
    if common:
        macro_c = stats.macro_scores_from_weights(
            W, T, R, TEMP_SCORE_PARTS, PRECIP_SCORE_PARTS, 5, tv, rv, tv & rv, common)
        for mi, m in enumerate(models):
            sc = board[m].get("score_common")
            if sc is None:
                continue
            assert abs(macro_c[0, mi] - sc) < 0.05, (m, macro_c[0, mi], sc)
    # 验证日数 / 降水维有效样本量（P0-2 的日历维度披露）
    for m in models:
        assert board[m]["n_days"] >= 1
        assert board[m]["n_eff_rain"] is not None


def test_holm_bonferroni_step_down():
    """P1-3：Holm–Bonferroni 逐步校正（一旦不显著，此后全部不显著）。"""
    from weather_eval.stats import holm_bonferroni
    # α=0.1、4 次检验：阈值依次为 .025 / .033 / .05 / .1
    assert holm_bonferroni([0.01, 0.02, 0.04, 0.09], 0.1) == [True, True, True, True]
    assert holm_bonferroni([0.01, 0.04, 0.04, 0.09], 0.1) == [True, False, False, False]
    assert holm_bonferroni([0.9, 0.9], 0.1) == [False, False]
    assert holm_bonferroni([], 0.1) == []
    # None（样本不足无法求 p）视为不显著，并阻断其后的判定
    assert holm_bonferroni([0.001, None, 0.001], 0.1) == [True, False, True]
    # 校正必然不比未校正更宽松
    ps = [0.01, 0.04, 0.2]
    flags = holm_bonferroni(ps, 0.1)
    assert all((not f) or (p <= 0.1) for f, p in zip(flags, ps))


def test_block_length_respects_decorrelation_and_block_count():
    """P1-1：块长取误差去相关时间，但块数不足时收缩（退化比偏短更糟）。"""
    from weather_eval.stats import (resolve_block_days, MIN_BOOTSTRAP_BLOCKS,
                                    MAX_BOOTSTRAP_BLOCK_DAYS)
    # ρ=0.5 → τ=3 天；12 天只有 4 块 < 下限 6 → 收缩到 2
    assert resolve_block_days(12, None, 0.5) == 2
    # 18 天 → 6 块，够用 → 3
    assert resolve_block_days(18, None, 0.5) == 3
    # 无自相关信息（ρ 未知/为 None）→ 1 天
    assert resolve_block_days(30, None, None) == 1
    # ρ=0.9 → τ=19 → 截断到上限
    assert resolve_block_days(200, None, 0.9) == MAX_BOOTSTRAP_BLOCK_DAYS
    # 显式指定同样受块数下限约束
    assert resolve_block_days(12, 5, None) == 2
    assert resolve_block_days(0, None, None) == 1
    assert resolve_block_days(12, None, 0.5) <= 12 // MIN_BOOTSTRAP_BLOCKS + 1


def test_day_block_weights_weights_sum_to_n_days():
    """块重采样：每次重复的总权重 == 天数（与点估计同量级），块内整体抽。"""
    import numpy as np
    from weather_eval.stats import day_block_weights
    for L in (1, 2, 3):
        W = day_block_weights(50, 12, L, seed=7)
        assert W.shape == (50, 12)
        assert np.allclose(W.sum(axis=1), 12)
        # 块长 L 时，同一块内的天必然同进退（权重相等）
        for row in W:
            for s in range(0, 12, L):
                grp = row[s:min(s + L, 12)]
                assert len(set(grp.tolist())) == 1


def test_common_window_removes_coverage_bias(tmp_path, monkeypatch):
    """P0-2：总榜名次只在入围源共同覆盖的天桶上比较。

    构造两家（同样的"误差随时效增长"规律，只是起点不同）：
      short 只覆盖第 1 桶，且第 1 桶报得**更差**（起点偏差 0.6°C）；
      wide  覆盖 1~5 桶，第 1 桶报得**更准**（起点偏差 0.05°C）。
    全窗口 macro 下 short 只吃最容易的第 1 桶 -> 名次被覆盖长度顶上去；
    共同窗口（第 1 桶，两家同台）下 wide 凭真本事第一。这正是要消掉的偏置。
    """
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 1, 0, 0)
    stations = [f"s{i}" for i in range(1, 6)]
    obs = [{"time": iso(start + timedelta(hours=h)), "temp": 24.0,
            "rain": 2.0 if h % 24 < 2 else 0.0} for h in range(8 * 24)]
    for sid in stations:
        storage.save_obs(sid, obs)

    def err(h, bias0):
        # 误差 = 起点偏差 + 随时效线性增长 + 低自相关的确定性抖动
        return bias0 + 0.9 * (h / 24.0) + (((h * 7919) % 100) / 100.0 - 0.5) * 0.8

    def snap(sid, model, hours, bias0):
        times, vals, rains = [], [], []
        for h in range(1, hours + 1):
            times.append(iso(start + timedelta(hours=h)))
            vals.append(24.0 + err(h, bias0))
            rains.append(2.0 if h % 24 < 2 else 0.0)
        return {"issue_iso": iso(start), "station_id": sid, "source": "test",
                "models": [model], "grid_lat": 23.0, "grid_lon": 111.0,
                "elevation": 50, "hourly_time": times,
                "data": {model: {"temperature_2m": vals, "precipitation": rains}}}

    for sid in stations:
        storage.save_forecast_snapshot(sid, "short", snap(sid, "short", 48, 0.6))
        storage.save_forecast_snapshot(sid, "wide", snap(sid, "wide", 6 * 24, 0.05))
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5,
           "min_board_neff": 5, "min_board_neff_rain": 5,
           "bootstrap_runs": 30, "sensitivity_runs": 30}
    end = start + timedelta(hours=7 * 24)
    data = build_report(stations, ["short", "wide"], cfg, start, end, "2026-08")
    board = data["leaderboards"]["all"]
    by_model = {r["model"]: r for r in board}
    assert by_model["short"]["qualified"] and by_model["wide"]["qualified"]
    # 共同窗口 = 两家都覆盖的第 1 桶；wide 覆盖 5 桶、short 只有 1 桶
    assert data["meta"]["common_window"]["buckets"] == [1]
    assert by_model["wide"]["n_buckets"] > by_model["short"]["n_buckets"] == 1
    # 第 1 桶 wide 更准 -> 共同窗口分更高 -> 名次第一（真本事说话）
    assert by_model["wide"]["score_common"] > by_model["short"]["score_common"]
    assert [r["model"] for r in board][0] == "wide"
    # 若按全窗口 macro 排名，只覆盖最容易一档的 short 反而会登顶——偏置的实证
    assert by_model["short"]["score"] > by_model["wide"]["score"]


def test_baseline_persistence_is_zero_skill_reference(tmp_path, monkeypatch):
    """P1-2：persistence 基准（明天 = 今天）存在，且预报源相对它有正技巧。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 1, 0, 0)
    stations = [f"s{i}" for i in range(1, 6)]
    # 温度有明显日变化（persistence 会明显出错），预报接近实况
    obs = [{"time": iso(start + timedelta(hours=h)), "temp": 24.0 + 3.0 * math.sin(h / 4.0),
            "rain": 2.0 if h % 48 < 3 else 0.0} for h in range(8 * 24)]
    for sid in stations:
        storage.save_obs(sid, obs)
    for sid in stations:
        times = [iso(start + timedelta(hours=h)) for h in range(1, 8 * 24)]
        storage.save_forecast_snapshot(sid, "good", {
            "issue_iso": iso(start), "station_id": sid, "source": "test",
            "models": ["good"], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
            "hourly_time": times,
            "data": {"good": {
                "temperature_2m": [24.0 + 3.0 * math.sin(h / 4.0) + 0.2
                                   for h in range(1, 8 * 24)],
                "precipitation": [2.0 if h % 48 < 3 else 0.0 for h in range(1, 8 * 24)]}}})
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5,
           "min_board_neff": 10, "min_board_neff_rain": 5,
           "bootstrap_runs": 30, "sensitivity_runs": 30}
    data = build_report(stations, ["good"], cfg, start,
                        start + timedelta(hours=7 * 24), "2026-08")
    base = data["meta"]["baseline_persistence"]
    assert base["1d"]["n_temp"] > 0 and base["1d"]["overall"] is not None
    row = data["leaderboards"]["all"][0]
    # 基准是零技巧参照：好的预报必须显著高于它
    assert row["skill"] is not None and row["skill"] > 10
    assert row["baseline_score"] < row["score_common"]
    # 趋势图带基准参考线数据（与曲线同一坐标系）
    assert data["score_trend"]["baseline"]["overall"]["1d"] is not None


def test_snapshot_quality_and_model_status_disclosed(tmp_path, monkeypatch):
    """P2-1/P2-2/P2-3：起报轮次、残缺快照、零数据状态进报告。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    start = datetime(2026, 8, 1, 0, 0)
    obs = [{"time": iso(start + timedelta(hours=h)), "temp": 20.0,
            "rain": 0.0} for h in range(72)]
    storage.save_obs("s1", obs)
    # full：3 个起报轮次，序列长度一致；trunc：第 3 份只有 12 小时
    for k in range(3):
        hours = 48 if k < 2 else 12
        times = [iso(start + timedelta(days=k, hours=h)) for h in range(hours)]
        storage.save_forecast_snapshot("s1", "full", {
            "issue_iso": iso(start + timedelta(days=k)), "station_id": "s1",
            "source": "test", "models": ["full"], "grid_lat": 23.0, "grid_lon": 111.0,
            "elevation": 50, "hourly_time": times,
            "data": {"full": {"temperature_2m": [20.0] * hours,
                              "precipitation": [0.0] * hours}}})
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 1}
    data = build_report(["s1"], ["full", "ghost"], cfg, start,
                        start + timedelta(hours=71), "2026-08")
    assert data["meta"]["model_status"]["full"] == "ok"
    assert data["meta"]["model_status"]["ghost"] == "no_data"
    q = data["meta"]["snapshot_quality"]["full"]
    assert q["snapshots"] == 3 and q["truncated"] == 1 and q["median_len"] == 48
    row = next(r for r in data["leaderboards"]["all"] if r["model"] == "full")
    assert row["n_issues"] == 3      # 起报轮次数（不是覆盖天数）

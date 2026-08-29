from datetime import datetime, timedelta

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
    # 降水分：TS×100(.30) 与 准确率(.15) 的加权 -> (15+12)/0.45
    p = {"ts": 0.5, "acc": 80.0}
    assert precip_score(p) == 60.0
    # 子分截断到 [0,100]：ETS 为负记 0 分，不拖成负总分
    assert precip_score({"ets": -0.5, "acc": 100.0}) == (0.0 * 0.25 + 100.0 * 0.15) / 0.40
    # RMSE 大到换算分为负时截断为 0
    assert temp_score({"rmse": 30.0}) == 0.0
    # 缺项按剩余权重归一：只有 r 时常数为 r×100
    assert temp_score({"r": 0.8}) == 80.0
    # 综合分 = 两者均分
    assert overall_score(t, p) == (92.5 + 60.0) / 2
    # 缺项不计：只有温度分时综合分 = 温度分
    assert overall_score(t, {"ts": None, "acc": None}) == 92.5
    # 全缺 -> None
    assert overall_score({}, {}) is None


def test_score_parts_contract():
    # 权重表契约：指标键、权重和为 1、每项带白话标签与换算函数
    from weather_eval.evaluate import PRECIP_SCORE_PARTS, TEMP_SCORE_PARTS
    for parts, keys in (
        (TEMP_SCORE_PARTS, {"acc2", "rmse", "r", "acc1", "mae", "mbe", "slope"}),
        (PRECIP_SCORE_PARTS, {"ts", "ets", "acc", "pod", "far", "bias"}),
    ):
        assert {k for k, *_ in parts} == keys
        assert abs(sum(w for _k, w, *_ in parts) - 1.0) < 1e-9
        for _k, _w, label, mp, fn in parts:
            assert label and mp and callable(fn)


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
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 5}
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

    # 分时效排行榜：行结构与名次（唯一有数据模型应排第一且分数与评分卡一致）
    from weather_eval.evaluate import overall_score
    lb1 = data["leaderboards"]["1d"]
    assert lb1 and lb1[0]["model"] == "ecmwf_ifs"
    row = lb1[0]
    assert row["score"] is not None and row["temp_score"] is not None and row["precip_score"] is not None
    assert 0 <= row["score"] <= 100 and 0 <= row["temp_score"] <= 100
    # 榜单分数与 24h 评分卡（同一样本总体）完全一致，冠军横幅与榜单不分叉
    sc = data["scorecard"]["ecmwf_ifs"]
    assert row["score"] == overall_score(sc["temp_24h"], sc["precip_24h"])
    assert row["acc2"] == sc["temp_24h"]["acc2"] and row["ts"] == sc["precip_24h"]["ts"]
    # 全部时效桶都有榜单行（无数据的桶行内分数为 None、沉底）
    assert len(data["leaderboards"]) == 16
    assert all(len(rows) == 1 for rows in data["leaderboards"].values())
    assert data["leaderboards"]["5d"][0]["score"] is None

    # 得分趋势：综合 = 温度/降水的均分，且逐桶键齐备
    st = data["score_trend"]
    assert set(st.keys()) == {"overall", "temp", "precip"}
    for b in ("1d", "2d"):
        tv = st["temp"]["ecmwf_ifs"][b]
        pv = st["precip"]["ecmwf_ifs"][b]
        ov = st["overall"]["ecmwf_ifs"][b]
        assert abs(ov - round((tv + pv) / 2, 2)) < 0.011, b


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

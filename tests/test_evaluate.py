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


def test_scores():
    # 温度分：±2°C 准确率与 100−RMSE×5 的均分
    t = {"acc2": 90.0, "rmse": 1.0}
    assert temp_score(t) == (90.0 + 95.0) / 2
    # 降水分：TS×100 与准确率的均分
    p = {"ts": 0.5, "acc": 80.0}
    assert precip_score(p) == (50.0 + 80.0) / 2
    # 综合分 = 两者均分
    assert overall_score(t, p) == (92.5 + 65.0) / 2
    # 缺项不计：只有温度分时综合分 = 温度分
    assert overall_score(t, {"ts": None, "acc": None}) == 92.5
    # 全缺 -> None
    assert overall_score({}, {}) is None


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

    # 排行榜：分数分解 + 排序稳定
    ranking = data["ranking"]
    assert ranking and ranking[0]["model"] == "ecmwf_ifs"
    row = ranking[0]
    assert row["score"] is not None and row["temp_score"] is not None and row["precip_score"] is not None
    assert abs(row["score"] - row["temp_score"]) <= 100.0  # 分数在 0~100 内
    assert 0 <= row["score"] <= 100 and 0 <= row["temp_score"] <= 100

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
    assert data["ranking"][0]["score"] is None
    assert data["score_trend"]["overall"]["ecmwf_ifs"]["1d"] is None

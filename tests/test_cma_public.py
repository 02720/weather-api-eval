"""CmaPublicProvider（中国气象局公众气象服务网 weather.cma.cn）接入测试。

mock 契约来自 2026-09-13 线上实测（4 站 × 各自 53 个整点逐点核对，
见 forecast/cma_public.py docstring）：

  GET https://weather.cma.cn/api/hourly/{WMO 站号}
  → {"msg":"success","code":0,
     "data":[{"date":"YYYY/MM/DD","list":[
        {"stationid","publishTime","forecastTime","hour","temperature",
         "precipitation",...}]}, ...(7 个日块，块间有重叠)...]}

重点覆盖五处容易**静默**出错的契约：
1. 时区（接口原生北京时，非 UTC）——错 8 小时不报错，只让样本静默归零；
2. 降水口径（后向 3 小时累计）——按 1 小时量入库会把雨量放大 3 倍；
3. 缺测哨兵 `999.9`——误当毫米入库会单条摧毁降水评分；
4. 起报锚点由接口自述的 `hour` 反解（而非 publishTime，后者晚 4 小时）；
5. "HTTP 200 + 空 data" 的失效形态——空快照会被幂等键永久锁死。
"""
import json
import logging
from datetime import datetime, timedelta

import pytest
from conftest import FakeResp

from weather_eval.forecast.cma_public import (
    BASE_URL,
    MODEL_NAME,
    SENTINEL_MIN,
    CmaPublicPayloadError,
    CmaPublicProvider,
    derive_issue,
    extract_points,
    parse_dt_bj,
)

BASE = datetime(2026, 9, 13, 8, 0)          # 起报基准（由 forecastTime − hour 反解）
PUBLISH = "2026/09/13 12:00"                # 产品循环的名义标签（**不是**锚点，见契约第 5 条）


class Station:
    id, name, lat, lon = "wuzhou", "梧州", 23.4783, 111.304
    cma_id = "59265"


class StationWithoutCode:
    id, name, lat, lon = "nodefault", "无站号站", 23.0, 111.0
    cma_id = None


class RoutingSession:
    """按 URL 路由的假会话；记录调用次数供"确定性失败不重试"断言。"""

    def __init__(self, responder):
        self.responder = responder
        self.calls = 0

    def request(self, method, url, **kw):
        self.calls += 1
        status, text = self.responder(url)
        return FakeResp(text, status_code=status)

    def get(self, url, **kw):
        return self.request("GET", url, **kw)


def _entry(t: datetime, hour: int, temp, precip, publish=PUBLISH) -> dict:
    return {"stationid": "59265", "publishTime": publish,
            "forecastTime": t.strftime("%Y/%m/%d %H:%M"), "hour": hour,
            "temperature": temp, "humidity": 80.0, "precipitation": precip,
            "weather": 3, "text": "阵雨"}


def _payload(points: list[tuple[datetime, dict]], per: int = 5,
             stride: int = 3) -> dict:
    """把点列切成**互相重叠**的日块（模拟线上块间重叠的形态：首块 8 点、次块从
    第 5 点起，重叠 3 点）。stride >= per 时退化为不重叠的分块。"""
    data = []
    for i in range(0, max(1, len(points) - per + 1), stride):
        chunk = points[i:i + per]
        if not chunk:
            break
        data.append({"date": chunk[0][0].strftime("%Y/%m/%d"),
                     "list": [e for _, e in chunk]})
    if len(points) >= per and (len(points) - per) % stride:
        chunk = points[-per:]
        data.append({"date": chunk[0][0].strftime("%Y/%m/%d"),
                     "list": [e for _, e in chunk]})
    return {"msg": "success", "code": 0, "data": data}


def _grid(n: int = 9, step_h: int = 3, start_h: int = 9, temp0: float = 20.0,
          precip: float = 3.0, publish: str = PUBLISH):
    """构造从 BASE+start_h 起、每 step_h 一采样、共 n 个的 3 小时网格。"""
    pts = []
    for i in range(n):
        t = BASE + timedelta(hours=start_h + step_h * i)
        pts.append((t, _entry(t, start_h + step_h * i, temp0 + i, precip, publish)))
    return pts


def _provider(responder, **kw) -> CmaPublicProvider:
    return CmaPublicProvider(session=RoutingSession(responder), **kw)


def _ok(payload) -> RoutingSession:
    return RoutingSession(lambda url: (200, json.dumps(payload)))


# ------------------------------------------------------------------ 纯函数：时间
def test_parse_dt_bj_accepts_contract_format():
    assert parse_dt_bj("2026/09/13 17:00") == datetime(2026, 9, 13, 17, 0)
    assert parse_dt_bj(" 2026/9/3 5:00 ") == datetime(2026, 9, 3, 5, 0)


def test_parse_dt_bj_rejects_other_formats():
    """接口若改回 ISO / 带偏移形态，必须响亮失败而非静默错 8 小时。"""
    for bad in ("2026-09-13T17:00:00+08:00", "2026-09-13T09:00:00Z",
                "2026-09-13 17:00:00", "17:00", None):
        with pytest.raises(ValueError):
            parse_dt_bj(bad)


# ------------------------------------------------------------------ 纯函数：解析
def test_extract_points_dedupes_overlapping_blocks():
    """日块重叠是接口的常态（首块覆盖到次日 14 时、次块从次日 08 时起），按时刻去重。"""
    pts = _grid(9)
    payload = _payload(pts)
    assert len([e for blk in payload["data"] for e in blk["list"]]) > len(pts)  # 确有重叠
    got, meta = extract_points(payload)
    assert [t for t, _ in got] == [t for t, _ in pts]
    assert meta["conflicts"] == 0 and meta["publish_raw"] == PUBLISH


def test_extract_points_flags_conflicting_duplicates():
    """同一时刻在两个日块里数值不同 = 一份响应里混了两个数据版本，必须可见。"""
    pts = _grid(5)
    payload = _payload(pts, per=len(pts))
    dup = dict(payload["data"][0]["list"][0])
    dup["temperature"] = 99.9
    payload["data"][0]["list"].append(dup)
    _, meta = extract_points(payload)
    assert meta["conflicts"] == 1


def test_extract_points_rejects_empty_data():
    """站号无效/契约漂移都表现为 HTTP 200 + 空 data（实测 data 为 ""），必须响亮失败。"""
    for data in ("", [], None):
        with pytest.raises(CmaPublicPayloadError):
            extract_points({"msg": "success", "code": 0, "data": data})


def test_extract_points_rejects_bad_envelope():
    with pytest.raises(CmaPublicPayloadError):
        extract_points({"code": 400, "msg": "参数错误", "data": []})
    with pytest.raises(CmaPublicPayloadError):
        extract_points(["not", "a", "dict"])


def test_extract_points_counts_unparsable_time():
    pts = _grid(4)
    payload = _payload(pts, per=len(pts))
    payload["data"][0]["list"].append({"forecastTime": "13/09/2026 17:00", "hour": 9})
    got, meta = extract_points(payload)
    assert len(got) == 4 and meta["bad_time"] == 1


# ------------------------------------------------------------------ 纯函数：起报锚点
def test_derive_issue_uses_hour_offset():
    """正常契约：forecastTime − hour 恒为同一常量（00Z 起报），不依赖硬编码偏移。"""
    pts = _grid(9)
    notes: list[str] = []
    issue, src = derive_issue(pts, PUBLISH, notes)
    assert issue == BASE and src == "hour_offset" and not notes


def test_derive_issue_uses_majority_when_a_few_hours_are_corrupt():
    """少数点的 `hour` 异常时取**多数**基准，而不是整体退回 publishTime。

    整体退回会把全部样本的 lead 系统性平移 (publishTime − 基准) 小时；而个别点
    损坏并不影响其余点给出的那个基准——降级必须是最后一招。异常本身仍要留档。
    """
    pts = _grid(6)
    pts[3][1]["hour"] = 999            # 1/6 异常
    notes: list[str] = []
    issue, src = derive_issue(pts, PUBLISH, notes)
    assert src == "hour_offset" and issue == BASE
    assert notes, "异常必须留档（快照 issue_notes）"


def test_derive_issue_falls_back_on_ambiguous_bases():
    """反解出**并列**的多个基准（无多数）时无法判定，退回 publishTime 并留档。"""
    pts = _grid(4)
    for _, e in pts[:2]:
        e["hour"] -= 6                 # 2 个点给出 14:00 基准，另 2 个给出 08:00
    notes: list[str] = []
    issue, src = derive_issue(pts, PUBLISH, notes)
    assert src == "publish_time" and issue == datetime(2026, 9, 13, 12, 0)
    assert notes


def test_derive_issue_rejects_base_after_first_valid_time():
    """反解出的基准晚于序列首点 = 会出现负 lead（评估侧静默丢弃）——必须拒绝并降级。"""
    pts = _grid(6, start_h=15)                 # 首点 23:00
    for _, e in pts:
        e["hour"] -= 12                        # 基准被推到 20:00，早于首点：可用
    notes: list[str] = []
    issue, src = derive_issue(pts, PUBLISH, notes)
    assert src == "hour_offset" and issue == datetime(2026, 9, 13, 20, 0)

    pts2 = _grid(6)                            # 首点 17:00
    for _, e in pts2:
        e["hour"] -= 12                        # 基准 20:00 晚于首点 → 不采用
    notes2: list[str] = []
    issue2, src2 = derive_issue(pts2, PUBLISH, notes2)
    assert src2 == "publish_time" and issue2 == datetime(2026, 9, 13, 12, 0)
    assert any("晚于序列首点" in n for n in notes2)


def test_derive_issue_falls_back_to_first_point_without_any_anchor():
    pts = [(t, {}) for t, _ in _grid(3)]
    notes: list[str] = []
    issue, src = derive_issue(pts, None, notes)
    assert src == "first_point" and issue == pts[0][0] and notes


def test_issue_follows_hour_not_publish_time():
    """**回归守卫**：起报锚点必须跟着 `hour` 反解走，而不是跟着 `publishTime`。

    2026-09-13 线上实测到产品换循环：publishTime 前进 8h（12:00→20:00）而 hour 基准
    前进 12h（08:00→20:00），即 publishTime − 起报基准并非常量（+4h → +0h）。
    若把 publishTime 当锚点，同一批有效时刻的 lead 会整体偏移数小时——而 lead 正是
    「提前 N 天」分桶的唯一依据。本条按循环 B 的真实形态构造（首点 23:00、基准 20:00、
    publishTime 仍是 12:00 未变），把两个候选彻底分离，钉住正确的那一个。
    """
    pts = _grid(6, start_h=15)                      # 首点 23:00（同循环 B）
    for _, e in pts:
        e["hour"] -= 12                             # 基准平移到 09-13 20:00
    notes: list[str] = []
    issue, src = derive_issue(pts, PUBLISH, notes)  # publishTime 仍是 12:00
    assert src == "hour_offset"
    assert issue == datetime(2026, 9, 13, 20, 0), "锚点必须跟 hour 走，不是 publishTime"
    assert not notes


# ------------------------------------------------------------------ Provider：口径
def test_snapshot_issue_and_hourly_axis():
    snap = _provider(lambda url: (200, json.dumps(_payload(_grid(9))))) \
        .fetch_snapshot(Station, [MODEL_NAME])
    assert snap["models"] == [MODEL_NAME] and snap["source"] == "cma_public"
    assert snap["issue_iso"] == "2026-09-13T08:00"
    # 契约枚举（P0-6）：各源统一到五种语义；provider 自有词表留在 detail
    assert snap["issue_source"] == "model_run"
    assert snap["issue_source_detail"] == "hour_offset"
    assert snap["resolution_hours"] and snap["precip_accum_window_hours"]
    # publishTime 只是审计线索，**不是**锚点：它与起报基准的差值实测非常量（+4h / +0h）
    assert snap["publish_minus_issue_hours"] == 4
    # 逐小时轴：首末采样之间逐小时展开（9 个 3 小时采样 → 25 个整点）
    assert snap["hourly_time"][0] == "2026-09-13T17:00"
    assert snap["hourly_time"][-1] == "2026-09-14T17:00"
    assert len(snap["hourly_time"]) == 25
    # lead 必须为正（评估按 lead 分天桶，落在起报之前会被整条丢掉）
    assert all(datetime.strptime(t, "%Y-%m-%dT%H:%M") > datetime(2026, 9, 13, 8, 0)
               for t in snap["hourly_time"])


def test_temperature_is_interpolated_linearly():
    snap = _provider(lambda url: (200, json.dumps(_payload(_grid(4, temp0=10.0))))) \
        .fetch_snapshot(Station, [MODEL_NAME])
    T = snap["data"][MODEL_NAME]["temperature_2m"]
    # 采样 17/20/23/02 时 = 10/11/12/13 → 17~20 时之间线性插值
    assert T[0] == 10.0 and T[3] == 11.0
    assert abs(T[1] - (10.0 + 1 / 3)) < 1e-9
    assert abs(T[2] - (10.0 + 2 / 3)) < 1e-9


def test_precipitation_is_three_hour_accumulation_spread():
    """降水是**后向 3 小时累计**：每个 3 小时窗口均摊到其覆盖的 3 个小时。

    这是本源最关键的口径假设（docstring 第 6 条的四条证据）；若实为 1 小时量，
    这里的断言会失败——那是必须重新标定而不是改断言的场景。"""
    snap = _provider(lambda url: (200, json.dumps(_payload(_grid(5, precip=3.0))))) \
        .fetch_snapshot(Station, [MODEL_NAME])
    P = snap["data"][MODEL_NAME]["precipitation"]
    assert snap["precip_interval_hours"] == 3
    # 5 个采样、每个 3.0mm → 除首个窗口只覆盖 1 小时外，其余每小时 1.0mm
    assert P[0] == 1.0
    assert P[1:] == [1.0] * (len(P) - 1)
    # 守恒：逐小时求和 == 采样总和 − 首窗未覆盖的 2 小时份额
    assert abs(sum(P) - (5 * 3.0 - 2 * 3.0 / 3)) < 1e-9


def test_sentinel_999_9_becomes_missing_not_rainfall():
    """`999.9` 是长时效的缺测占位。误当毫米入库会单条摧毁降水评分。"""
    pts = _grid(5, precip=3.0)
    pts[3][1]["precipitation"] = 999.9
    snap = _provider(lambda url: (200, json.dumps(_payload(pts)))) \
        .fetch_snapshot(Station, [MODEL_NAME])
    P = snap["data"][MODEL_NAME]["precipitation"]
    assert SENTINEL_MIN not in P and 999.9 not in P
    assert snap["missing_sentinel_precip"] == 1
    # 哨兵点（第 4 个采样 02:00）所在的 3 小时窗口整体降缺测——窗口总量无从得知，
    # 绝不借用邻窗值（轴自 17:00 起，该窗口覆盖轴上的第 8~10 个小时）
    assert P[7:10] == [None, None, None]
    # 温度不受影响：实测同一行温度仍是有效值
    assert snap["data"][MODEL_NAME]["temperature_2m"][6] is not None


def test_nonphysical_values_become_missing():
    pts = _grid(5)
    pts[1][1]["precipitation"] = -0.5          # 负降水物理上不存在
    pts[2][1]["temperature"] = 999.9           # 温度的防御性物理上界
    snap = _provider(lambda url: (200, json.dumps(_payload(pts)))) \
        .fetch_snapshot(Station, [MODEL_NAME])
    assert snap["negative_precip"] == 1 and snap["implausible_temp"] == 1
    P = snap["data"][MODEL_NAME]["precipitation"]
    T = snap["data"][MODEL_NAME]["temperature_2m"]
    assert P[1:4] == [None, None, None]        # 负值（第 2 个采样）所在窗口整体缺测
    assert T[3] is not None                     # 缺测点是 23:00，20:00 的插值端点仍可用
    assert T[6] is None                         # 23:00 本尊按缺测


def test_row_with_missing_precipitation_is_not_zero():
    """缺测绝不折算 0.0——那会被读成"预报无雨"，是凭造技巧。"""
    pts = _grid(4, precip=3.0)
    pts[2][1]["precipitation"] = None
    snap = _provider(lambda url: (200, json.dumps(_payload(pts)))) \
        .fetch_snapshot(Station, [MODEL_NAME])
    P = snap["data"][MODEL_NAME]["precipitation"]
    assert None in P and 0.0 not in P[3:9]


def test_all_missing_refuses_to_store():
    """温度与降水双双全缺测时拒绝入库（空快照被幂等锁死即永久占位）。"""
    pts = _grid(4)
    for _, e in pts:
        e["temperature"] = None
        e["precipitation"] = None
    with pytest.raises(RuntimeError) as ei:
        _provider(lambda url: (200, json.dumps(_payload(pts)))) \
            .fetch_snapshot(Station, [MODEL_NAME])
    assert "拒绝入库空快照" in str(ei.value)


# ------------------------------------------------------------------ Provider：失败语义
def test_station_without_cma_id_raises_actionably():
    with pytest.raises(RuntimeError) as ei:
        _provider(lambda url: (200, "{}")).fetch_snapshot(StationWithoutCode, [MODEL_NAME])
    assert "cma_id" in str(ei.value) and "stations.yaml" in str(ei.value)


def test_empty_business_data_raises():
    with pytest.raises(CmaPublicPayloadError) as ei:
        _provider(lambda url: (200, json.dumps(
            {"msg": "success", "code": 0, "data": ""}))).fetch_snapshot(Station, [MODEL_NAME])
    assert "data 为空" in str(ei.value)


def test_http_403_is_fatal_and_not_retried():
    """实测该站按 UA 反爬名单拒绝（python-requests/curl 被拒）：确定性失败。"""
    sess = RoutingSession(lambda url: (403, "<html>403 Forbidden</html>"))
    prov = CmaPublicProvider(session=sess, retries=3)
    with pytest.raises(CmaPublicPayloadError) as ei:
        prov.fetch_snapshot(Station, [MODEL_NAME])
    assert "403" in str(ei.value) and "User-Agent" in str(ei.value)
    assert sess.calls == 1                     # 不烧退避


def test_non_json_200_raises():
    with pytest.raises(CmaPublicPayloadError) as ei:
        _provider(lambda url: (200, "<html>bad gateway</html>")) \
            .fetch_snapshot(Station, [MODEL_NAME])
    assert "非 JSON" in str(ei.value)


def test_request_url_uses_station_code():
    seen: list[str] = []

    def responder(url):
        seen.append(url)
        return 200, json.dumps(_payload(_grid(4)))

    _provider(responder).fetch_snapshot(Station, [MODEL_NAME])
    assert seen == [f"{BASE_URL}/59265"]


def test_5xx_is_retried():
    state = {"n": 0}

    def responder(url):
        state["n"] += 1
        if state["n"] < 2:
            return 503, "upstream busy"
        return 200, json.dumps(_payload(_grid(4)))

    snap = _provider(responder, retries=2).fetch_snapshot(Station, [MODEL_NAME])
    assert state["n"] == 2 and snap["issue_iso"] == "2026-09-13T08:00"


def test_health_warns_on_short_coverage(caplog):
    """长时效被截断必须可见，否则会被误读成"模式长时效技巧差"。"""
    with caplog.at_level(logging.WARNING):
        _provider(lambda url: (200, json.dumps(_payload(_grid(3))))) \
            .fetch_snapshot(Station, [MODEL_NAME])
    assert "最大时效仅" in caplog.text or "时效可能被严重截断" in caplog.text


def test_health_warns_on_sentinel(caplog):
    pts = _grid(4)
    pts[1][1]["precipitation"] = 999.9
    with caplog.at_level(logging.WARNING):
        _provider(lambda url: (200, json.dumps(_payload(pts)))) \
            .fetch_snapshot(Station, [MODEL_NAME])
    assert "占位哨兵值" in caplog.text


# ------------------------------------------------------------------ CLI 端到端
def test_cli_cma_public_archives_snapshot(tmp_path, monkeypatch):
    import weather_eval.__main__ as m
    from weather_eval import storage

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))

    class FakeProvider:
        def fetch_snapshot(self, station, model_list):
            return {
                "issue_iso": "2026-09-13T08:00", "station_id": station.id,
                "source": "cma_public", "models": ["cma_public_v1"],
                "grid_lat": station.lat, "grid_lon": station.lon, "elevation": None,
                "hourly_time": ["2026-09-13T17:00", "2026-09-13T18:00"],
                "data": {"cma_public_v1": {"temperature_2m": [28.5, 27.7],
                                           "precipitation": [1.0, 1.0]}},
            }

    monkeypatch.setattr(m, "CmaPublicProvider", lambda: FakeProvider())
    assert m.main(["fetch-forecast", "--source", "cma_public"]) in (None, 0)
    snaps = storage.list_forecast_snapshots("wuzhou", "cma_public_v1")
    assert len(snaps) == 1 and snaps[0]["issue_iso"] == "2026-09-13T08:00"
    assert snaps[0]["data"]["cma_public_v1"]["precipitation"] == [1.0, 1.0]


def test_cli_cma_public_is_registered():
    """源必须同时登记在 SOURCE_SPECS 与 argparse 的 choices 里（漏一即运行期 KeyError）。"""
    import weather_eval.__main__ as m
    assert "cma_public" in m.SOURCE_SPECS
    assert m.SOURCE_MODELS["cma_public"] == {"cma_public_v1"}


# ------------------------------------------------------------------ 端到端配对
def test_snapshot_pairs_with_observations_hourly(tmp_path, monkeypatch):
    """**时区/口径的终局回归**：入库快照必须与实况按北京时整点配上。

    评估引擎按整点精确配对——时区错 8 小时不会报任何错，只会让该源样本数直接
    变成 0（榜单上表现为"该源消失"）。这条把 provider → storage → build_report
    整条链路走通，并让每条观测严格等于预报 + 1℃，据此反证对齐正确。
    """
    from weather_eval import storage
    from weather_eval.evaluate import build_report
    from weather_eval.timeutil import parse_iso

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    snap = _provider(lambda url: (200, json.dumps(_payload(_grid(9))))) \
        .fetch_snapshot(Station, [MODEL_NAME])
    storage.save_forecast_snapshot(Station.id, MODEL_NAME, snap)

    obs = []
    for t, v in zip(snap["hourly_time"], snap["data"][MODEL_NAME]["temperature_2m"]):
        if v is not None:
            obs.append({"time": t, "temp": v + 1.0, "rain": 0.0})
    assert len(obs) == 25
    storage.save_obs(Station.id, obs)

    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 1}
    start = parse_iso(snap["hourly_time"][0])
    end = parse_iso(snap["hourly_time"][-1])
    data = build_report([Station.id], [MODEL_NAME], cfg, start, end, "2026-09")
    sc = data["scorecard"][MODEL_NAME]["temp_all"]
    assert sc["n"] == 25, "时刻必须逐点配上（错 8 小时则 n=0）"
    assert abs(sc["rmse"] - 1.0) < 1e-6


def test_second_fetch_in_same_issue_is_idempotent(tmp_path, monkeypatch):
    """同一 issue 重复抓取不产生冗余快照（一轮内首份即封存，符合存档先行）。"""
    from weather_eval import storage

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    prov = _provider(lambda url: (200, json.dumps(_payload(_grid(4)))))
    snap = prov.fetch_snapshot(Station, [MODEL_NAME])
    assert storage.save_forecast_snapshot(Station.id, MODEL_NAME, snap) is True
    assert storage.save_forecast_snapshot(Station.id, MODEL_NAME, snap) is False
    assert len(storage.list_forecast_snapshots(Station.id, MODEL_NAME)) == 1

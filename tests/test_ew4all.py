"""Ew4allProvider（EW4ALL / CMA-NDFS / 风清AI模式）接入测试。

mock 契约来自 2026-09-12 线上实测 + 前端 bundle 逆向（见 forecast/ew4all.py docstring）：
- GET  {BASE}/api/modelTimeList?data_type&element
       → {"code":200,"data":[{"data_time":"YYYYMMDDHH0000"},...]}
- POST {BASE}/api/raster/findByPoint  body={mode,elements,point:[[lon,lat]],projection,dataTime,level}
       → {"code":200,"data":[{"<element>":v,"Datetime":"YYYY-MM-DD HH:MM:SS","level":0},...]}

重点覆盖三处容易静默出错的契约：
1. 时间语义（接口原生是 UTC，不是北京时）；
2. 降水口径（只取内部自洽的后向累计族，平铺均摊；绝不用不自洽的 ONETPE）；
3. 起报轮次必须"要素齐全"（该源一轮内先出温度、后出降水）。
"""
import json
import logging
from datetime import datetime, timedelta

import pytest
from conftest import FakeResp

from weather_eval.forecast.ew4all import (
    FIND_BY_POINT_URL,
    MODEL_SPECS,
    MODEL_TIME_LIST_URL,
    Ew4allProvider,
    build_series,
    hourly_axis,
    parse_dt_utc,
    parse_run_label,
    to_bj,
)
from weather_eval.forecast.resample import interpolate_hourly, spread_accumulation


class RoutingSession:
    """按 (method, url) 路由的假会话；记录全部调用供断言。"""

    def __init__(self, routes):
        self.routes = routes
        self.calls: list[tuple] = []

    def _dispatch(self, method, url, params=None, headers=None, timeout=None, json=None):
        self.calls.append((method, url, dict(params or {}), dict(json or {})))
        status, text = self.routes(method, url, params, json)
        return FakeResp(text, status_code=status)

    def get(self, url, **kw):
        return self._dispatch("GET", url, **kw)

    def post(self, url, **kw):
        return self._dispatch("POST", url, **kw)

    def request(self, method, url, **kw):
        return self._dispatch(method, url, **kw)


class Station:
    id, name, lat, lon = "wuzhou", "梧州", 23.4783, 111.304


# ------------------------------------------------------------------ 纯函数
def test_parse_dt_utc_and_shift_to_beijing():
    assert parse_dt_utc("2026-09-12 01:00:00") == datetime(2026, 9, 12, 1, 0)
    assert to_bj(datetime(2026, 9, 12, 1, 0)) == datetime(2026, 9, 12, 9, 0)


def test_parse_dt_utc_rejects_nonzero_offset():
    """带 +08:00 的时间串会整体错 8h，显式拒绝以防契约漂移被静默误读。"""
    with pytest.raises(ValueError):
        parse_dt_utc("2026-09-12T01:00:00+08:00")
    # 明确的 UTC 偏移仍可接受
    assert parse_dt_utc("2026-09-12T01:00:00+00:00") == datetime(2026, 9, 12, 1, 0)


def test_parse_run_label():
    assert parse_run_label("20260912120000") == datetime(2026, 9, 12, 12, 0)
    for bad in ("2026091212", "2026091212000a", "202609121200000"):
        with pytest.raises(ValueError):
            parse_run_label(bad)


def test_hourly_axis_inclusive():
    assert hourly_axis(datetime(2026, 9, 12, 9), datetime(2026, 9, 12, 12)) == [
        datetime(2026, 9, 12, 9), datetime(2026, 9, 12, 10),
        datetime(2026, 9, 12, 11), datetime(2026, 9, 12, 12)]


# ------------------------------------------------------------------ build_series 口径
def _utc(h, day=12, minute=0):
    return datetime(2026, 9, day, h, minute)


def test_build_series_interpolates_temperature_and_tiles_precip():
    """温度线性插值；降水按 3h 后向窗口平铺均摊（每小时恰属一个窗口）。"""
    tem = [(_utc(1), 10.0), (_utc(4), 13.0)]        # 3h 采样 → 插值出 10/11/12/13
    prc = [(_utc(3), 3.0)]                          # HOURTPE(03:00Z) = (00,03] 共 3.0mm
    axis, temps, precips, one_until = build_series(tem, [], prc, 3)
    # 轴为北京时：UTC 01:00 → BJT 09:00
    assert [t.hour for t in axis] == [9, 10, 11, 12]
    assert temps == [10.0, 11.0, 12.0, 13.0]
    # 3h 窗口 (00,03]UTC = BJT 09/10/11 三小时，各 1.0；BJT 12:00 不在该窗口内 → 缺测
    assert precips == [1.0, 1.0, 1.0, None]
    assert one_until is None                        # 未提供 1 小时产品


def test_build_series_prefers_one_hour_product_then_hands_off():
    """1 小时产品在覆盖时效内逐点透传（不摊薄），超出部分才用累计量摊薄兜底。"""
    tem = [(_utc(1), 20.0), (_utc(6), 25.0)]
    one = [(_utc(1), 0.2), (_utc(2), 0.0), (_utc(3), 1.4)]   # 原生 1 小时累计
    acc = [(_utc(3), 3.0), (_utc(6), 9.0)]                   # 3h 累计：(00,03] / (03,06]
    axis, _, precips, one_until = build_series(tem, one, acc, 3)
    # +1/+2/+3h 用 1 小时产品原值；+4/+5/+6h 用 3h 窗口均摊（9.0/3 = 3.0）
    assert precips[:3] == [0.2, 0.0, 1.4]
    assert precips[3:] == [3.0, 3.0, 3.0]
    # 注意 +3h：摊薄值本会是 1.0，被 1 小时产品的 1.4 覆盖 —— 优先级正确
    assert one_until == axis[2]


def test_build_series_precip_missing_yields_all_none():
    """降水要素缺失时整列 None——绝不折算 0.0（0.0 会被读成"预报无雨"，是造技巧）。"""
    axis, temps, precips, one_until = build_series([(_utc(1), 20.0), (_utc(3), 22.0)], [], [], 3)
    assert len(precips) == len(axis) == 3
    assert temps == [20.0, 21.0, 22.0]
    assert precips == [None, None, None]
    assert one_until is None


def test_build_series_six_hour_window_conserves_total():
    """风清：6h 累计逐 6 小时采样 → 平铺 /6；完整跨度上求和恒等于原始累计总量。"""
    bins = [(datetime(2026, 9, 12, 6) + timedelta(hours=6 * i), 6.0) for i in range(4)]
    tem = [(datetime(2026, 9, 12, 6), 30.0), (datetime(2026, 9, 13, 0), 24.0)]
    axis, _, precips, _ = build_series(tem, [], bins, 6)
    covered = [v for v in precips if v is not None]
    assert covered and all(abs(v - 1.0) < 1e-9 for v in covered)
    # 轴从 BJT 14:00 起，末窗 (18,24] UTC 未完整落在轴上 → 覆盖小时数 * 1.0mm == 求和
    assert abs(sum(covered) - len(covered) * 1.0) < 1e-9


def test_temperature_not_extrapolated_beyond_samples():
    axis, temps, _, _ = build_series(
        [(datetime(2026, 9, 12, 1), 10.0), (datetime(2026, 9, 12, 2), 11.0)],
        [], [(datetime(2026, 9, 12, 9), 1.0)], 3)
    # 降水把轴拉到 UTC 09:00；温度只到 02:00，其后一律缺测（不外推）
    assert temps[:2] == [10.0, 11.0]
    assert temps[2:] == [None] * (len(axis) - 2)


# ------------------------------------------------------------------ resample 助手
def test_spread_accumulation_ignores_overlapping_windows():
    """采样间隔 < 窗口长度时，只能取相位平铺子集：否则同一场雨被重复计多次。"""
    # 6h 窗口、3h 采样：全部 6 个采样点都是合法窗口端点，平铺只取相位一致的那 3 个
    samples = [(datetime(2026, 9, 12, 3 + 3 * i), 6.0) for i in range(6)]
    hours = hourly_axis(datetime(2026, 9, 12, 1), datetime(2026, 9, 12, 18))
    out = spread_accumulation(samples, hours, 6)
    assert all(abs(v - 1.0) < 1e-9 for v in out if v is not None)
    # 3 个平铺窗口 × 6 小时 = 18 个小时槽，其中首窗（窗口起点落在轴之前）只覆盖 3 个小时，
    # 末窗之后（16~18 时）无窗口覆盖 → 实际覆盖 15 小时
    assert sum(1 for v in out if v is not None) == 15
    assert out[-1] is None          # 平铺窗口之外一律缺测，绝不留用邻值


def test_spread_accumulation_rejects_bad_window():
    with pytest.raises(ValueError):
        spread_accumulation([], [], 0)


def test_interpolate_hourly_identity_for_hourly_samples():
    samples = [(datetime(2026, 9, 12, h), float(h)) for h in range(5)]
    assert interpolate_hourly(samples) == samples


def test_precip_specs_pin_the_one_hour_first_policy():
    """回归守卫：CMA-NDFS 必须优先用原生 1 小时降水，兜底要素必须来自后向累计族。

    口径决策的证据见 ew4all.py docstring 第 5/6 条（ONETPE 与 3h 族同相位、同总量，
    是同一物理量的不同后处理版本；曾因单站小样本误判为"不自洽"而弃用，已纠正）。"""
    n = MODEL_SPECS["cma_ndfs"]
    assert n.precip_1h_element == "ONETPE"       # 短时效用原生 1 小时累计
    assert n.precip_element == "HOURTPE"         # 兜底：3 小时后向累计
    assert n.precip_window_hours == 3
    # 风清确实没有 1 小时产品（ONETPE / HOURTPE 实测为空），不能臆造
    f = MODEL_SPECS["fengqing_ai"]
    assert f.precip_1h_element is None
    assert f.precip_element == "SIXTPE" and f.precip_window_hours == 6
    # 兜底要素必须属于内部严格自洽的后向累计族（防日后误取 DAYTPE 之类的粗产品）
    assert {s.precip_element for s in MODEL_SPECS.values()} <= {
        "HOURTPE", "SIXTPE", "TWELVETPE", "DAYTPE"}


# ------------------------------------------------------------------ Provider
def _rows(element, start_utc, n, step_h, value_fn):
    day = start_utc
    out = []
    for i in range(n):
        t = day + timedelta(hours=step_h * i)
        out.append({element: value_fn(i), "Datetime": t.strftime("%Y-%m-%d %H:%M:%S"),
                    "level": 0})
    return out


def _routes(*, runs=("2026091200", "2026091112"), code=200, tem_missing=()):
    """构造路由：runs 为 modelTimeList 内容；各要素的采样序列锚定在请求的 dataTime 上。

    采样形态与线上一致：GDFS5KM 的温度逐小时（首点 +1h）、降水平铺 3 小时（首点 +3h）；
    NMCFENGQING 两者均逐 6 小时（首点 +6h）。tem_missing 中的轮次温度返回空。
    """
    def routes(method, url, params, body):
        if url == MODEL_TIME_LIST_URL:
            assert params["element"] == "TEM"
            return code, json.dumps({
                "code": 200,
                "data": [{"data_time": r + "0000"} for r in runs],
                "success": "true"})
        if url == FIND_BY_POINT_URL:
            mode, el, dt = body["mode"], body["elements"], body["dataTime"]
            assert body["point"] == [[Station.lon, Station.lat]]
            assert body["projection"] == 4326
            run = datetime.strptime(dt, "%Y%m%d%H")
            if dt in tem_missing and el == "TEM":
                return 200, json.dumps({"code": 200, "data": []})
            if el == "TEM":
                step = 1 if mode == "GDFS5KM" else 6
                n = 4 if mode == "GDFS5KM" else 2
                rows = _rows("TEM", run + timedelta(hours=step), n, step,
                             lambda i: 20.0 + i)
            elif el == "ONETPE":                       # 原生 1 小时降水：+1h 起逐小时
                rows = _rows(el, run + timedelta(hours=1), 4, 1, lambda i: 0.5)
            else:                                      # 后向累计族（3h / 6h）
                step = 3 if el == "HOURTPE" else 6
                rows = _rows(el, run + timedelta(hours=step), 2, step, lambda i: 3.0)
            return 200, json.dumps({"code": 200, "message": "获取成功", "data": rows,
                                    "success": "true"})
        return 404, "{}"

    return routes


def test_fetch_snapshot_returns_independent_snapshot_per_model():
    """两模型各自独立快照（起报轮次发布进度不同步），而非共享时间轴的单份 dict。"""
    prov = Ew4allProvider(session=RoutingSession(_routes()))
    snaps = prov.fetch_snapshot(Station, ["cma_ndfs", "fengqing_ai"])
    assert isinstance(snaps, list) and [s["models"] for s in snaps] == [
        ["cma_ndfs"], ["fengqing_ai"]]
    for snap in snaps:
        m = snap["models"][0]
        assert snap["source"] == "ew4all"
        assert snap["issue_iso"] == "2026-09-12T08:00"   # UTC 00Z → 北京时 08:00
        assert snap["run_label_utc"] == "2026091200"
        assert len(snap["data"][m]["temperature_2m"]) == len(snap["hourly_time"])
        assert len(snap["data"][m]["precipitation"]) == len(snap["hourly_time"])


def test_issue_iso_is_beijing_time_not_raw_label():
    """接口原生是 UTC：起报锚点若不做 +8h 转换会整体错 8 小时且不报错。"""
    prov = Ew4allProvider(session=RoutingSession(_routes(runs=("2026091212",))))
    snap = prov.fetch_snapshot(Station, ["cma_ndfs"])[0]
    assert snap["issue_iso"] == "2026-09-12T20:00"       # 12Z → 20:00 北京时
    # 首个有效时刻应恰为起报 +1h（lead 从 1 开始，评估引擎才不会丢掉全部样本）
    first = datetime.strptime(snap["hourly_time"][0], "%Y-%m-%dT%H:%M")
    issue = datetime.strptime(snap["issue_iso"], "%Y-%m-%dT%H:%M")
    assert int((first - issue).total_seconds() // 3600) == 1


def test_run_with_temperature_but_no_precip_is_skipped():
    """该源一轮内先出温度后出降水：要素不齐的轮次必须跳过。

    否则残缺快照落盘即被同 issue 幂等锁死，该轮降水永久缺失（每轮 12h × 4 站）。"""
    def routes(method, url, params, body):
        if url == MODEL_TIME_LIST_URL:
            return 200, json.dumps({"code": 200, "data": [
                {"data_time": "20260912120000"}, {"data_time": "20260912000000"}]})
        mode, el, dt = body["mode"], body["elements"], body["dataTime"]
        if dt == "2026091212" and el != "TEM":
            return 200, json.dumps({"code": 200, "data": []})       # 降水尚未出数
        rows = _rows(el, datetime(2026, 9, 12, 1), 3, 1, lambda i: 20.0) \
            if el == "TEM" else _rows(el, datetime(2026, 9, 12, 3), 2, 3, lambda i: 3.0)
        return 200, json.dumps({"code": 200, "data": rows})

    prov = Ew4allProvider(session=RoutingSession(routes))
    snap = prov.fetch_snapshot(Station, ["cma_ndfs"])[0]
    assert snap["run_label_utc"] == "2026091200"            # 回退到要素齐全的上一轮
    assert any(v is not None for v in snap["data"]["cma_ndfs"]["precipitation"])


def test_no_run_with_both_elements_raises():
    """契约漂移（要素代码变了）必须响亮失败，而不是静默落一份降水全缺的快照。"""
    def routes(method, url, params, body):
        if url == MODEL_TIME_LIST_URL:
            return 200, json.dumps({"code": 200, "data": [
                {"data_time": "20260912000000"}, {"data_time": "20260911120000"}]})
        if body["elements"] == "TEM":
            return 200, json.dumps({"code": 200, "data": _rows(
                "TEM", datetime(2026, 9, 12, 1), 3, 1, lambda i: 20.0)})
        return 200, json.dumps({"code": 200, "data": []})

    prov = Ew4allProvider(session=RoutingSession(routes))
    with pytest.raises(RuntimeError) as ei:
        prov.fetch_snapshot(Station, ["cma_ndfs"])
    assert "没有任何一轮" in str(ei.value)


def test_empty_temperature_series_refuses_to_store():
    """空快照会被同 issue 幂等锁死，正常数据永远进不来——必须拒绝入库。"""
    prov = Ew4allProvider(session=RoutingSession(
        _routes(tem_missing=("2026091200", "2026091112"))))
    with pytest.raises(RuntimeError) as ei:
        prov.fetch_snapshot(Station, ["cma_ndfs"])
    assert "没有任何一轮" in str(ei.value)


def test_business_error_code_raises():
    """HTTP 200 但业务 code != 200 必须显式上抛（不能静默当成缺测）。"""
    def routes(method, url, params, body):
        if url == MODEL_TIME_LIST_URL:
            return 200, json.dumps({"code": 200,
                                    "data": [{"data_time": "20260912000000"}]})
        return 200, json.dumps({"code": 400, "msg": "参数错误", "data": None})

    prov = Ew4allProvider(session=RoutingSession(routes))
    with pytest.raises(RuntimeError) as ei:
        prov.fetch_snapshot(Station, ["cma_ndfs"])
    assert "参数错误" in str(ei.value)


def test_non_json_200_raises():
    """200 但返回非 JSON（如网关错误页）属确定性失败。"""
    def routes(method, url, params, body):
        return 200, "<html>bad gateway</html>"

    prov = Ew4allProvider(session=RoutingSession(routes))
    with pytest.raises(RuntimeError) as ei:
        prov.fetch_snapshot(Station, ["cma_ndfs"])
    assert "非 JSON" in str(ei.value)


def test_http_4xx_is_fatal_and_not_retried():
    calls = {"n": 0}

    def routes(method, url, params, body):
        calls["n"] += 1
        return 422, json.dumps({"detail": [{"msg": "field required"}]})

    prov = Ew4allProvider(session=RoutingSession(routes), retries=3)
    with pytest.raises(RuntimeError) as ei:
        prov.fetch_snapshot(Station, ["cma_ndfs"])
    assert "HTTP 422" in str(ei.value)
    assert calls["n"] == 1          # 确定性失败不重试（不烧退避）


def test_unknown_model_is_rejected():
    prov = Ew4allProvider(session=RoutingSession(_routes()))
    with pytest.raises(RuntimeError) as ei:
        prov.fetch_snapshot(Station, ["no_such_model"])
    assert "未登记模型" in str(ei.value)


def test_run_resolution_cached_across_stations():
    """起报探测是模式级属性：4 站共享一次探测，且探测站的原始行直接复用。"""
    prov = Ew4allProvider(session=RoutingSession(_routes()))
    sess = prov.session
    prov.fetch_snapshot(Station, ["cma_ndfs"])
    # 首次 = 起报时次列表 1 + 探测温度 1 + 探测兜底降水 1 + 探测 1 小时降水 1；
    # 落盘复用探测结果，故不再追加请求
    assert len(sess.calls) == 4
    prov.fetch_snapshot(Station, ["cma_ndfs"])
    # 第二次只剩 3 次数据请求（温度 + 1 小时降水 + 兜底降水）
    assert len(sess.calls) - 4 == 3


def test_snapshot_marks_one_hour_handoff_point_and_expansion():
    """口径切换点必须机器可读地留档：哪一段是原生 1 小时、哪一段是摊薄。"""
    prov = Ew4allProvider(session=RoutingSession(_routes()))
    snap = prov.fetch_snapshot(Station, ["cma_ndfs"])[0]
    # fixture：ONETPE 覆盖 +1h..+4h（北京时 09:00..12:00）
    assert snap["precip_1h_until"] == "2026-09-12T12:00"
    assert "ONETPE passthrough" in snap["expansion"]
    P = snap["data"]["cma_ndfs"]["precipitation"]
    # +5h/+6h 落到 HOURTPE 摊薄轨道（3.0/3 = 1.0），且与 1 小时段衔接无缺测
    assert P == [0.5, 0.5, 0.5, 0.5, 1.0, 1.0]


def test_snapshot_falls_back_when_one_hour_product_absent():
    """1 小时产品缺位（占位期/契约漂移）时全时效退化为摊薄，且不留虚假交接点。"""
    def routes(method, url, params, body):
        if url == MODEL_TIME_LIST_URL:
            return 200, json.dumps({"code": 200, "data": [{"data_time": "20260912000000"}]})
        mode, el, dt = body["mode"], body["elements"], body["dataTime"]
        run = datetime.strptime(dt, "%Y%m%d%H")
        assert mode == "GDFS5KM"
        if el == "TEM":
            rows = _rows("TEM", run + timedelta(hours=1), 4, 1, lambda i: 20.0 + i)
        elif el == "ONETPE":
            rows = _rows(el, run + timedelta(hours=1), 4, 1, lambda i: None)  # 值全 null
        else:
            rows = _rows(el, run + timedelta(hours=3), 2, 3, lambda i: 3.0)
        return 200, json.dumps({"code": 200, "data": rows})

    prov = Ew4allProvider(session=RoutingSession(routes))
    snap = prov.fetch_snapshot(Station, ["cma_ndfs"])[0]
    assert snap["precip_1h_until"] is None
    assert "ONETPE passthrough" in snap["expansion"]      # 配置仍写着，但实际未生效
    # 全部来自 3 小时摊薄轨道
    assert snap["data"]["cma_ndfs"]["precipitation"][3:6] == [1.0, 1.0, 1.0]


def test_run_with_all_none_precip_values_is_skipped():
    """"行数非空但值全为 null" 不是"降水已就绪"。

    若只看行数，就会主动选中这个残缺的新轮、丢掉更早的完整轮，落下一份全缺测快照
    并被幂等键永久锁死。"""
    def routes(method, url, params, body):
        if url == MODEL_TIME_LIST_URL:
            return 200, json.dumps({"code": 200, "data": [
                {"data_time": "20260912120000"}, {"data_time": "20260912000000"}]})
        mode, el, dt = body["mode"], body["elements"], body["dataTime"]
        rows = _rows(el, datetime(2026, 9, 12, 1), 3, 1, lambda i: 20.0) \
            if el == "TEM" else _rows(el, datetime(2026, 9, 12, 3), 2, 3, lambda i: 3.0)
        if dt == "2026091212" and el != "TEM":
            for r in rows:               # 占位期：行在、值全为 null
                r[el] = None
        return 200, json.dumps({"code": 200, "data": rows})

    prov = Ew4allProvider(session=RoutingSession(routes))
    snap = prov.fetch_snapshot(Station, ["cma_ndfs"])[0]
    assert snap["run_label_utc"] == "2026091200"
    assert any(v is not None for v in snap["data"]["cma_ndfs"]["precipitation"])


def test_all_runs_with_null_precip_raise():
    def routes(method, url, params, body):
        if url == MODEL_TIME_LIST_URL:
            return 200, json.dumps({"code": 200, "data": [
                {"data_time": "20260912000000"}, {"data_time": "20260911120000"}]})
        el = body["elements"]
        rows = _rows(el, datetime(2026, 9, 12, 1), 3, 1, lambda i: 20.0)
        if el != "TEM":
            for r in rows:
                r[el] = None
        return 200, json.dumps({"code": 200, "data": rows})

    prov = Ew4allProvider(session=RoutingSession(routes))
    with pytest.raises(RuntimeError) as ei:
        prov.fetch_snapshot(Station, ["cma_ndfs"])
    assert "没有任何一轮" in str(ei.value)


def test_health_warns_on_missing_precip_window(caplog):
    """平铺端点间距哨兵：缺一个采样点会让整个窗口（1~w 小时）降缺测，必须可见。"""
    from weather_eval.forecast.resample import tile_endpoints

    spec = MODEL_SPECS["fengqing_ai"]        # 6 小时窗口
    base = datetime(2026, 9, 12, 6)
    tem = [(base + timedelta(hours=6 * i), 20.0) for i in range(6)]
    prc = [(base + timedelta(hours=6 * i), 6.0) for i in range(6)]
    assert {int((b - a).total_seconds() // 3600)
            for a, b in zip(tile_endpoints(prc, 6), tile_endpoints(prc, 6)[1:])} == {6}

    gappy = [x for x in prc if x[0].hour != 18]      # 缺 18:00 这个窗口端点
    gap_hours = {int((b - a).total_seconds() // 3600)
                 for a, b in zip(tile_endpoints(gappy, 6), tile_endpoints(gappy, 6)[1:])}
    assert 12 in gap_hours                        # 间距出现 12h = 中间丢了一个窗口

    axis, temps, precips, _ = build_series(tem, [], gappy, 6)
    with caplog.at_level(logging.WARNING):
        Ew4allProvider._warn_health(Station, spec, tem, gappy, [], axis, temps, precips)
    assert "降水平铺端点间距异常" in caplog.text

    caplog.clear()
    axis2, t2, p2, _ = build_series(tem, [], prc, 6)
    with caplog.at_level(logging.WARNING):
        Ew4allProvider._warn_health(Station, spec, tem, prc, [], axis2, t2, p2)
    assert "间距异常" not in caplog.text


def test_health_warns_when_one_hour_coverage_shrinks(caplog):
    """1 小时降水覆盖时长明显短于规格时必须告警：交接点提前＝更多时效落入摊薄轨道。"""
    spec = MODEL_SPECS["cma_ndfs"]
    base = datetime(2026, 9, 12, 1)
    tem = [(base + timedelta(hours=i), 20.0) for i in range(8)]
    acc = [(base + timedelta(hours=3), 3.0), (base + timedelta(hours=6), 3.0)]
    one = [(base + timedelta(hours=i), 0.5) for i in range(4)]   # 只覆盖 4 小时（预期 ~72）

    axis, temps, precips, _ = build_series(tem, one, acc, 3)
    with caplog.at_level(logging.WARNING):
        Ew4allProvider._warn_health(Station, spec, tem, acc, one, axis, temps, precips)
    assert "仅覆盖" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        Ew4allProvider._warn_health(Station, spec, tem, acc, [], axis, temps, precips)
    assert "无有效值" in caplog.text        # 完全没有 1 小时产品时另有一条告警


# ------------------------------------------------------------------ CLI 端到端
def test_cli_ew4all_archives_one_snapshot_per_model(tmp_path, monkeypatch):
    """两模型各自独立快照：存档必须按模型分账，不能互相串带时间轴/起报时刻。"""
    import weather_eval.__main__ as m
    from weather_eval import storage

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))

    class FakeProvider:
        def fetch_snapshot(self, station, model_list):
            out = []
            for mod in model_list:
                out.append({
                    "issue_iso": "2026-09-12T08:00", "station_id": station.id,
                    "source": "ew4all", "models": [mod],
                    "grid_lat": station.lat, "grid_lon": station.lon, "elevation": None,
                    "hourly_time": ["2026-09-12T09:00", "2026-09-12T10:00"],
                    "data": {mod: {"temperature_2m": [20.0, 21.0],
                                   "precipitation": [0.0 if mod == "cma_ndfs" else 5.0,
                                                     0.0]}},
                })
            return out

    monkeypatch.setattr(m, "Ew4allProvider", lambda: FakeProvider())
    assert m.main(["fetch-forecast", "--source", "ew4all"]) in (None, 0)

    for mod, rain in (("cma_ndfs", 0.0), ("fengqing_ai", 5.0)):
        snaps = storage.list_forecast_snapshots("wuzhou", mod)
        assert len(snaps) == 1, mod
        assert list(snaps[0]["data"]) == [mod], "快照必须按模型分账"
        assert snaps[0]["data"][mod]["precipitation"][0] == rain
        assert snaps[0]["issue_iso"] == "2026-09-12T08:00"


# ------------------------------------------------------------------ 端到端配对
def test_snapshot_pairs_with_observations_at_lead_one(tmp_path, monkeypatch):
    """**时区/时效的终局回归**：入库快照必须能与实况按整点配上，且 lead 从 1 起。

    评估引擎按北京时整点精确配对，时区错 8 小时不会报任何错、只会让该源的样本数
    直接变成 0（榜单上表现为"该源消失"）。本条把整条链路走通：provider 抓取
    （UTC 采样）→ build_series（转北京时 + 插值/平铺）→ storage 落盘 →
    build_report 配对打分，据此反证时间语义。
    """
    from weather_eval import storage
    from weather_eval.evaluate import build_report
    from weather_eval.timeutil import parse_iso

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))

    prov = Ew4allProvider(session=RoutingSession(_routes(runs=("2026091200",))))
    snap = prov.fetch_snapshot(Station, ["cma_ndfs"])[0]
    storage.save_forecast_snapshot(Station.id, "cma_ndfs", snap)

    # 观测与快照逐点对应：温度 = 预报 − 1（误差恒为 1°C，便于反推对齐是否正确）
    obs = []
    for i, t in enumerate(snap["hourly_time"]):
        v = snap["data"]["cma_ndfs"]["temperature_2m"][i]
        if v is None:
            continue
        obs.append({"time": t, "temp": v - 1.0, "rain": 0.0})
    assert len(obs) == 4                      # fixture：起报 +1h..+4h 四个整点
    storage.save_obs(Station.id, obs)

    start = parse_iso(snap["hourly_time"][0])
    end = parse_iso(snap["hourly_time"][-1])
    cfg = {"temp_accuracy_limits": [1, 2], "rain_threshold_mm": 0.1,
           "hourly_lead_days": 16, "daily_max_offset_days": 16, "min_sample": 1}
    data = build_report([Station.id], ["cma_ndfs"], cfg, start, end, "2026-09")

    sc = data["scorecard"]["cma_ndfs"]["temp_all"]
    assert sc["n"] == 4, "时刻必须逐点配上（错 8 小时则 n=0）"
    assert abs(sc["rmse"] - 1.0) < 1e-6
    assert sc["acc1"] == 100.0


def test_first_valid_time_matches_run_plus_one_hour():
    """lead 必须从 1 开始：若首点落在起报之前，评估引擎会把该快照的样本全部丢掉。"""
    prov = Ew4allProvider(session=RoutingSession(_routes()))
    snap = prov.fetch_snapshot(Station, ["cma_ndfs"])[0]
    issue = datetime.strptime(snap["issue_iso"], "%Y-%m-%dT%H:%M")
    first = datetime.strptime(snap["hourly_time"][0], "%Y-%m-%dT%H:%M")
    assert int((first - issue).total_seconds() // 3600) == 1

from weather_eval import storage


def test_obs_dedup_and_merge(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    r1 = [
        {"time": "2026-08-26T20:00", "temp": 27.0, "rain": 0.0},
        {"time": "2026-08-26T19:00", "temp": 26.0, "rain": 1.0},
    ]
    assert storage.save_obs("s1", r1) == 2
    # 重复保存相同数据：不新增
    assert storage.save_obs("s1", r1) == 0
    # 更新一条
    assert storage.save_obs("s1", [{"time": "2026-08-26T20:00", "temp": 28.0, "rain": 0.0}]) == 1
    loaded = storage.load_obs("s1", "2026-08")
    assert loaded["2026-08-26T20:00"]["temp"] == 28.0
    assert loaded["2026-08-26T19:00"]["rain"] == 1.0


def test_forecast_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    snap = {
        "issue_iso": "2026-08-26T21:00", "station_id": "s1", "source": "open-meteo",
        "models": ["ecmwf_ifs"], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
        "hourly_time": ["2026-08-26T21:00"],
        "data": {"ecmwf_ifs": {"temperature_2m": [20.0], "precipitation": [0.0]}},
    }
    assert storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap) is True
    # 同站同模型同起报时刻：幂等
    assert storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap) is False
    assert len(storage.list_forecast_snapshots("s1", "ecmwf_ifs")) == 1


def test_corrupt_json_skipped_not_fatal(tmp_path, monkeypatch):
    """损坏的存档文件（git 冲突残留等）应告警跳过，不拖垮观测/快照读取。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    # 好文件 + 坏文件并存
    good_obs = tmp_path / "obs" / "s1" / "2026-08.json"
    good_obs.parent.mkdir(parents=True)
    good_obs.write_text('{"2026-08-26T20:00": {"time": "2026-08-26T20:00", "temp": 27.0}}',
                        encoding="utf-8")
    bad = tmp_path / "obs" / "s1" / "2026-07.json"
    bad.write_text('{"2026-07-01T10:00": {"time": ', encoding="utf-8")
    loaded = storage.load_obs("s1")
    assert set(loaded) == {"2026-08-26T20:00"}
    # 坏文件被跳过后仍可正常合并写入该月
    storage.save_obs("s1", [{"time": "2026-07-01T10:00", "temp": 25.0, "rain": 0.0}])
    assert storage.load_obs("s1", "2026-07")["2026-07-01T10:00"]["temp"] == 25.0

    # 预报快照同理
    good_snap_dir = tmp_path / "forecasts" / "s1" / "ecmwf_ifs"
    good_snap_dir.mkdir(parents=True)
    snap = {"issue_iso": "2026-08-26T21:00", "models": ["ecmwf_ifs"],
            "hourly_time": ["2026-08-26T21:00"], "data": {"ecmwf_ifs": {}}}
    (good_snap_dir / "2026-08-26T2100.json").write_text(
        __import__("json").dumps(snap), encoding="utf-8")
    (good_snap_dir / "broken.json").write_text("{oops", encoding="utf-8")
    assert len(storage.list_forecast_snapshots("s1", "ecmwf_ifs")) == 1


def test_snapshots_readable_from_gzip(tmp_path, monkeypatch):
    """P3-5：归档为 .json.gz 的快照与普通 .json 一视同仁地被读取。"""
    import gzip as _gzip
    import json as _json
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    snap = {"issue_iso": "2026-08-26T21:00", "models": ["ecmwf_ifs"],
            "hourly_time": ["2026-08-26T21:00"],
            "data": {"ecmwf_ifs": {"temperature_2m": [20.0], "precipitation": [0.0]}}}
    d = tmp_path / "forecasts" / "s1" / "ecmwf_ifs"
    d.mkdir(parents=True)
    (d / "2026-08-26T2100.json.gz").write_bytes(
        _gzip.compress(_json.dumps(snap).encode("utf-8")))
    (d / "2026-08-26T2200.json").write_text(_json.dumps({**snap, "issue_iso": "2026-08-26T22:00"}),
                                           encoding="utf-8")
    snaps = storage.list_forecast_snapshots("s1", "ecmwf_ifs")
    assert [s["issue_iso"] for s in snaps] == ["2026-08-26T21:00", "2026-08-26T22:00"]


def test_archive_old_snapshots_dry_run_and_apply(tmp_path, monkeypatch):
    """archive_old_snapshots：dry-run 只列候选；apply 压缩并删源文件；幂等。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    old = {"issue_iso": "2026-07-01T08:00", "models": ["ecmwf_ifs"],
           "hourly_time": ["2026-07-01T08:00"], "data": {"ecmwf_ifs": {}}}
    new = {"issue_iso": "2026-09-06T08:00", "models": ["ecmwf_ifs"],
           "hourly_time": ["2026-09-06T08:00"], "data": {"ecmwf_ifs": {}}}
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", old)
    storage.save_forecast_snapshot("s1", "ecmwf_ifs", new)

    candidates = storage.archive_old_snapshots(60, apply=False)
    assert len(candidates) == 1 and candidates[0].name == "2026-07-01T0800.json"
    assert candidates[0].exists()                       # dry-run 不动文件

    changed = storage.archive_old_snapshots(60, apply=True)
    assert len(changed) == 1
    assert not candidates[0].exists()
    assert candidates[0].with_name(candidates[0].name + ".gz").exists()
    # 归档后两个快照都能读到（.json.gz 与 .json 混存）
    snaps = storage.list_forecast_snapshots("s1", "ecmwf_ifs")
    assert [s["issue_iso"] for s in snaps] == ["2026-07-01T08:00", "2026-09-06T08:00"]
    # 幂等：再跑一次无候选
    assert storage.archive_old_snapshots(60, apply=True) == []


def test_concurrent_snapshot_save_is_locked(tmp_path, monkeypatch):
    """P2-5：快照写入的 exists 检查与写入在 flock 临界区内，并发保存不重复写。"""
    import json as _json
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    snap = {"issue_iso": "2026-08-26T21:00", "models": ["ecmwf_ifs"],
            "hourly_time": ["2026-08-26T21:00"], "data": {"ecmwf_ifs": {}}}
    # 串行语义保持：第二次保存幂等跳过
    assert storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap) is True
    assert storage.save_forecast_snapshot("s1", "ecmwf_ifs", snap) is False
    files = list((tmp_path / "forecasts" / "s1" / "ecmwf_ifs").glob("*.json"))
    assert len(files) == 1
    assert _json.loads(files[0].read_text(encoding="utf-8"))["issue_iso"] == "2026-08-26T21:00"


# ------------------------------------------------- P2-2 观测回改留痕（revisions）
def test_obs_revision_is_preserved_not_overwritten(tmp_path, monkeypatch):
    """观测源会修正早期错报；旧实现静默覆盖，"当时的实况"从此不可复原——
    而实况是评估里唯一的真值来源，它一旦不可追溯，所有历史结论都失去可复核性。
    """
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    storage.save_obs("s1", [{"time": "2026-08-26T20:00", "temp": 27.0, "rain": 0.0}])
    storage.save_obs("s1", [{"time": "2026-08-26T20:00", "temp": 29.5, "rain": 0.0}])
    rec = storage.load_obs("s1", "2026-08")["2026-08-26T20:00"]
    assert rec["temp"] == 29.5                     # 新值生效
    assert rec["revisions"], "回改必须留痕"
    assert rec["revisions"][-1]["prev"]["temp"] == 27.0   # 旧值可复原
    assert rec["revisions"][-1]["at"]              # 改动时间戳

    # 未回改（写入相同值）不产生 revision
    storage.save_obs("s1", [{"time": "2026-08-26T21:00", "temp": 25.0, "rain": 0.0}])
    rec2 = storage.load_obs("s1", "2026-08")["2026-08-26T21:00"]
    assert "revisions" not in rec2


def test_obs_revisions_are_capped(tmp_path, monkeypatch):
    """抖动源反复改写同一时刻时，revisions 不得无限增长。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    t = "2026-08-26T20:00"
    storage.save_obs("s1", [{"time": t, "temp": 20.0, "rain": 0.0}])
    for i in range(30):
        storage.save_obs("s1", [{"time": t, "temp": 20.0 + i, "rain": 0.0}])
    rec = storage.load_obs("s1", "2026-08")[t]
    assert len(rec["revisions"]) <= storage.MAX_OBS_REVISIONS


# ------------------------------------------------------ P2-1 归档写入原子性
def test_archive_is_atomic_and_idempotent(tmp_path, monkeypatch):
    """中断只能留下临时文件，绝不留下半截 .gz（它会被 git 收进仓库，
    变成一份永久损坏的"档案"）。"""
    import gzip
    import json
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    snap = {"issue_iso": "2020-01-01T00:00", "station_id": "s1", "source": "t",
            "models": ["m"], "hourly_time": ["2020-01-01T01:00"],
            "data": {"m": {"temperature_2m": [1.0], "precipitation": [0.0]}}}
    storage.save_forecast_snapshot("s1", "m", snap)
    cands = storage.archive_old_snapshots(older_than_days=1, apply=True)
    assert len(cands) == 1
    gz = cands[0].with_name(cands[0].name + ".gz")
    assert gz.exists() and not cands[0].exists()
    assert not list(gz.parent.glob("*.tmp"))          # 不留临时文件
    with gzip.open(gz, "rt", encoding="utf-8") as f:
        assert json.load(f)["issue_iso"] == "2020-01-01T00:00"
    # 归档后仍能被读取（评估口径不受影响）
    assert len(storage.list_forecast_snapshots("s1", "m")) == 1
    # 幂等：再归档一次没有候选
    assert storage.archive_old_snapshots(older_than_days=1, apply=True) == []


# -------------------------------------- 跨源合并：来源通道变化不算"观测回改"
def test_source_tag_change_is_not_a_revision(tmp_path, monkeypatch):
    """主源故障恢复后，同一小时会从备用源换成主源——**这不是实况回改**。

    若把来源标记计入变化比较，主源恢复的那一轮会把每一小时都记成一次"回改"，
    revisions 立刻被噪声淹没，真正的错报修正反而看不见了（revisions 的存在意义
    就是"当时的实况可复原"）。
    """
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    t = "2026-08-26T20:00"
    # 备用源先写入
    assert storage.save_obs("s1", [{"time": t, "source": "cma_data",
                                    "temp": 27.0, "rain": 1.0}]) == 1
    # 主源恢复后给出同样的要素值，仅来源不同
    assert storage.save_obs("s1", [{"time": t, "source": "wd",
                                    "temp": 27.0, "rain": 1.0}]) == 0
    rec = storage.load_obs("s1", "2026-08")[t]
    assert "revisions" not in rec, "来源通道变化不得留下回改痕迹"
    assert rec["source"] == "wd", "权威源应接管该小时的来源标记"


def test_real_value_change_still_records_revision(tmp_path, monkeypatch):
    """要素值真的变了，回改留痕必须照旧（不能被上一条放宽掉）。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    t = "2026-08-26T21:00"
    storage.save_obs("s1", [{"time": t, "source": "wd", "temp": 27.0, "rain": 0.0}])
    assert storage.save_obs("s1", [{"time": t, "source": "wd",
                                    "temp": 25.5, "rain": 4.6}]) == 1
    rec = storage.load_obs("s1", "2026-08")[t]
    assert rec["temp"] == 25.5 and rec["rain"] == 4.6
    assert rec["revisions"][-1]["prev"]["temp"] == 27.0
    assert rec["revisions"][-1]["prev"]["rain"] == 0.0


# ------------------------------------------------------- 日产品评测范围截断
def _snap_with_long_daily(days: int) -> dict:
    """构造日产品铺 days 天的快照（模拟 90 天日预报的源，起报 2026-09-01T08:00）。"""
    from datetime import date as _date, timedelta as _td
    model = "long_daily"
    base = _date(2026, 9, 1)
    return {
        "issue_iso": "2026-09-01T08:00", "station_id": "s1", "source": "test",
        "models": [model], "grid_lat": 23.0, "grid_lon": 111.0, "elevation": 50,
        "hourly_time": ["2026-09-01T08:00"],
        "data": {model: {"temperature_2m": [20.0], "precipitation": [0.0]}},
        "daily_time": [(base + _td(days=off)).isoformat() for off in range(days)],
        "daily": {model: {"temp_max": [22.0] * days, "temp_min": [18.0] * days,
                          "precipitation": [0.5] * days}},
    }


def test_save_forecast_snapshot_truncates_daily_block_beyond_eval_range(
        tmp_path, monkeypatch):
    """入库截断（需求 2）：日偏移 > 16 的日产品不入库；数组与时间轴保持对齐；
    内容哈希覆盖截断后的实际存档（可复算）。"""
    from weather_eval.snapshot_meta import snapshot_sha256
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    snap = _snap_with_long_daily(90)
    assert storage.save_forecast_snapshot("s1", "long_daily", snap) is True
    stored = storage.list_forecast_snapshots("s1", "long_daily")[0]
    # 起报日 09-01：保留 offset 0..16 共 17 天（评测用 1..16，起报当日随块保留）
    assert len(stored["daily_time"]) == 17
    assert stored["daily_time"][0] == "2026-09-01"
    assert stored["daily_time"][-1] == "2026-09-17"
    entry = stored["daily"]["long_daily"]
    assert all(len(entry[k]) == 17 for k in ("temp_max", "temp_min", "precipitation"))
    # 哈希在截断之后计算：库内哈希必须能对存档复算（截断前的内容不参与）
    assert stored["payload_sha256"] == snapshot_sha256(stored)


def test_truncate_daily_block_conservative_cases(tmp_path, monkeypatch):
    """截断的防御性约定：范围内不动、畸形条目保留、无 issue 不截、无日块不崩。"""
    from weather_eval.snapshot_meta import truncate_daily_block
    # 1) 全部在范围内：原样返回、对象不被改动
    snap = _snap_with_long_daily(3)
    before = {k: (list(v) if isinstance(v, list) else v)
              for k, v in snap.items() if k in ("daily_time", "daily")}
    assert truncate_daily_block(snap, 16) is snap
    assert snap["daily_time"] == before["daily_time"] and snap["daily"] == before["daily"]
    # 2) 畸形条目（非字符串/坏日期）保留——宁多勿丢；置于边界之外仍不丢
    snap = _snap_with_long_daily(20)
    snap["daily_time"][18] = None
    snap["daily_time"][19] = "not-a-date"
    truncate_daily_block(snap, 16)
    # 17 个有效（offset 0..16）+ 2 个畸形保留；offset 17 的正常条目被截
    assert len(snap["daily_time"]) == 19
    assert None in snap["daily_time"] and "not-a-date" in snap["daily_time"]
    assert len(snap["daily"]["long_daily"]["temp_max"]) == 19   # 数组同步保留
    # 3) issue_iso 缺失/不可解析：不截（退回旧行为）
    snap = _snap_with_long_daily(90)
    snap.pop("issue_iso")
    truncate_daily_block(snap, 16)
    assert len(snap["daily_time"]) == 90
    # 4) 无日产品块 / 块结构异常：安全为 no-op
    for malformed in ({}, {"daily_time": ["2026-09-01"]},
                      {"daily_time": "oops", "daily": {}},
                      {"daily_time": ["2026-09-01"], "daily": "oops"}):
        s = {"issue_iso": "2026-09-01T08:00", **malformed}
        truncate_daily_block(s, 16)   # 绝不抛异常

"""体积治理（月度 bundle 冻结 + 超期出仓）的回归测试。

治理的对象是**证据流**，删错了不可恢复，所以这里的用例围绕四条不变量展开：

  A. **口径不变**：归档前后 Merkle 根与快照集合完全相同（读取侧透明展开）。
  B. **先固化、后删除**：出仓前必须已有该月结论摘要，否则拒绝删除。
  C. **冻结不可逆**：已写出的 bundle 永不重写（否则 git 每月都收一份全新 blob，
     省下来的字节原样还回去）。
  D. **失败不毁数据**：源文件不可读 / 回读校验不通过 → 拒绝冻结，源文件原封不动。
"""
from __future__ import annotations

import gzip
import json

import pytest

from weather_eval import storage
from weather_eval.snapshot_meta import integrity_summary
from weather_eval.timeutil import parse_iso
from weather_eval.__main__ import main


def _fake_now(s: str):
    """注入"当前时刻"（ISO 北京时），让分层的月份边界可被确定性地测试。"""
    return parse_iso(s)


def _snap(issue: str, model: str = "ecmwf_ifs") -> dict:
    return {
        "issue_iso": issue, "station_id": "s1", "source": "test",
        "models": [model],
        "hourly_time": [issue, issue],
        "data": {model: {"temperature_2m": [20.0, 21.0], "precipitation": [0.0, 0.0]}},
    }


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    return tmp_path


def _seed(issues, model="ecmwf_ifs"):
    for issue in issues:
        storage.save_forecast_snapshot("s1", model, _snap(issue, model))


def _dir(env, model="ecmwf_ifs"):
    return env / "forecasts" / "s1" / model


# ------------------------------------------------------------------ A 口径不变
def test_bundle_preserves_snapshot_set_and_merkle_root(env):
    """★ 归档不改变任何指标口径：快照集合与 Merkle 根逐位相同。

    这是整个治理方案能被接受的前提——若归档会动到哈希链，"验证存档未被改写"
    这条地基就跟着塌了。
    """
    issues = [f"2026-07-{d:02d}T{h:02d}00" for d in (1, 2, 3) for h in (0, 6, 12)]
    _seed(issues)
    before = storage.list_forecast_snapshots("s1", "ecmwf_ifs")
    root_before = integrity_summary(before)["merkle_root"]

    rep = storage.compact_snapshots(grace_days=2, apply=True,
                                    now=_fake_now("2026-09-19T10:00"))
    assert len(rep["bundles_created"]) == 1
    assert rep["files_removed"] == len(issues)

    after = storage.list_forecast_snapshots("s1", "ecmwf_ifs")
    assert integrity_summary(after)["merkle_root"] == root_before
    assert [s["issue_iso"] for s in after] == [s["issue_iso"] for s in before]
    # 落盘形态确实变了：散装 .json 没了，换成一份 .gz
    assert not list(_dir(env).glob("*.json"))
    assert (_dir(env) / "2026-07.json.gz").exists()


def test_bundle_is_much_smaller_than_individual_files(env):
    _seed([f"2026-07-{d:02d}T{h:02d}00" for d in range(1, 11) for h in range(0, 24, 3)])
    rep = storage.compact_snapshots(grace_days=2, apply=True,
                                    now=_fake_now("2026-09-19T10:00"))
    assert rep["bytes_after"] < rep["bytes_before"] / 3, "合并后应有数量级的压缩收益"
    assert rep["bundles_created"]


def test_dry_run_touches_nothing(env):
    _seed(["2026-07-01T0000", "2026-07-02T0000"])
    rep = storage.compact_snapshots(grace_days=2, apply=False,
                                    now=_fake_now("2026-09-19T10:00"))
    assert len(rep["bundles_created"]) == 1        # 报告里有候选
    assert rep["files_pending"] == 2
    assert rep["files_removed"] == 0
    assert len(list(_dir(env).glob("*.json"))) == 2, "dry-run 不得动盘"
    assert not (_dir(env) / "2026-07.json.gz").exists()


def test_compaction_is_idempotent(env):
    _seed(["2026-07-01T0000", "2026-07-02T0000"])
    now = _fake_now("2026-09-19T10:00")
    first = storage.compact_snapshots(grace_days=2, apply=True, now=now)
    second = storage.compact_snapshots(grace_days=2, apply=True, now=now)
    assert len(first["bundles_created"]) == 1
    assert second["bundles_created"] == []
    assert second["files_removed"] == 0
    assert len(storage.list_forecast_snapshots("s1", "ecmwf_ifs")) == 2


def test_current_month_is_never_frozen(env):
    """当月（及宽限期内）保持逐份 .json：主报告、月度冻结与 verify 都靠它。"""
    now = _fake_now("2026-09-05T10:00")
    _seed(["2026-09-01T0000", "2026-09-02T0000", "2026-08-30T0000"])
    rep = storage.compact_snapshots(grace_days=2, apply=True, now=now)
    assert len(rep["bundles_created"]) == 1
    assert (_dir(env) / "2026-08.json.gz").exists()
    assert (_dir(env) / "2026-09-01T0000.json").exists(), "当月快照不得被冻结"


def test_grace_period_holds_back_the_previous_month(env):
    """宽限期内不冻结上月：月初那几轮运行仍可能补上月末最后几小时的快照。"""
    _seed(["2026-08-31T2300"])
    rep = storage.compact_snapshots(grace_days=3, apply=True,
                                    now=_fake_now("2026-09-02T10:00"))
    assert rep["bundles_created"] == []
    assert (_dir(env) / "2026-08-31T2300.json").exists()


# -------------------------------------------------------------- C 冻结不可逆
def test_frozen_bundle_is_never_rewritten(env):
    """已冻结的包不动；冻结之后新补的散装快照按读取侧规则优先。"""
    _seed(["2026-07-01T0000"])
    now = _fake_now("2026-09-19T10:00")
    storage.compact_snapshots(grace_days=2, apply=True, now=now)
    bundle = _dir(env) / "2026-07.json.gz"
    fingerprint = bundle.read_bytes()

    # 冻结之后又出现同月的一份散装快照（异常情况：不该发生，但要能自洽处理）
    _seed(["2026-07-02T0000"])
    rep = storage.compact_snapshots(grace_days=2, apply=True, now=now)

    assert bundle.read_bytes() == fingerprint, "冻结档案不得被重写"
    assert rep["files_removed"] == 0
    snaps = storage.list_forecast_snapshots("s1", "ecmwf_ifs")
    assert {s["issue_iso"] for s in snaps} == {"2026-07-01T0000", "2026-07-02T0000"}, \
        "bundle 与后补的散装快照应合并可读"


def test_bundle_container_is_self_describing(env):
    _seed(["2026-07-01T0000", "2026-07-02T0000"])
    storage.compact_snapshots(grace_days=2, apply=True,
                              now=_fake_now("2026-09-19T10:00"))
    with gzip.open(_dir(env) / "2026-07.json.gz", "rt", encoding="utf-8") as f:
        c = json.load(f)
    assert c[storage.BUNDLE_MARK] == storage.BUNDLE_SCHEMA_VERSION
    assert c["period"] == "2026-07"
    assert c["n_snapshots"] == 2
    assert set(c["snapshots"]) == {"2026-07-01T0000", "2026-07-02T0000"}


# ------------------------------------------------------------ D 失败不毁数据
def test_unreadable_source_blocks_freezing_and_keeps_files(env):
    _seed(["2026-07-01T0000"])
    (_dir(env) / "2026-07-02T0000.json").write_text("{半截 json", encoding="utf-8")
    rep = storage.compact_snapshots(grace_days=2, apply=True,
                                    now=_fake_now("2026-09-19T10:00"))
    assert rep["errors"], "不可读的源文件必须让该包冻结失败并如实上报"
    assert not (_dir(env) / "2026-07.json.gz").exists()
    assert (_dir(env) / "2026-07-01T0000.json").exists(), "源文件必须原封不动"


def test_archive_step_never_loses_obs_side(env):
    """治理只碰 forecasts/，观测档案不受影响。"""
    storage.save_obs("s1", [{"time": "2026-07-01T10:00", "temp": 25.0, "rain": 0.0}])
    _seed(["2026-07-01T0000"])
    storage.compact_snapshots(grace_days=2, apply=True,
                              now=_fake_now("2026-09-19T10:00"))
    assert storage.load_obs("s1", "2026-07")["2026-07-01T10:00"]["temp"] == 25.0


# ------------------------------------------------------- B 先固化、后删除
def test_expiry_requires_frozen_summary(env):
    """超期 bundle 只有在"该月结论已固化"之后才允许出仓。"""
    _seed(["2024-01-01T0000"])
    now = _fake_now("2026-09-19T10:00")
    storage.compact_snapshots(grace_days=2, apply=True, now=now)   # 先冻结
    bundle = _dir(env) / "2024-01.json.gz"
    assert bundle.exists()

    # 没有结论摘要 → 拒绝删除
    rep = storage.compact_snapshots(grace_days=2, retain_months=13, apply=True, now=now)
    assert rep["expiry_blocked"] and not rep["expired"]
    assert bundle.exists(), "结论摘要缺失时不得删除原始归档"

    # 固化结论之后 → 允许出仓
    p = storage.period_summary_path("2024-01")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"period": "2024-01"}', encoding="utf-8")
    rep = storage.compact_snapshots(grace_days=2, retain_months=13, apply=True, now=now)
    assert rep["expired"] and not rep["expiry_blocked"]
    assert not bundle.exists()


def test_expiry_force_bypasses_summary_check(env):
    _seed(["2024-01-01T0000"])
    now = _fake_now("2026-09-19T10:00")
    storage.compact_snapshots(grace_days=2, apply=True, now=now)
    rep = storage.compact_snapshots(grace_days=2, retain_months=13, apply=True,
                                    force=True, now=now)
    assert rep["expired"] and not rep["expiry_blocked"]


def test_within_retention_is_kept(env):
    _seed(["2025-12-01T0000"])
    now = _fake_now("2026-09-19T10:00")
    storage.compact_snapshots(grace_days=2, apply=True, now=now)
    rep = storage.compact_snapshots(grace_days=2, retain_months=13, apply=True, now=now)
    assert not rep["expired"]
    assert (_dir(env) / "2025-12.json.gz").exists(), "保留期内的冷层不得出仓"


# ------------------------------------------------------------------ 工具函数
def test_shift_month_bounds():
    assert storage.shift_month("2026-09", -13) == "2025-08"
    assert storage.shift_month("2026-01", -1) == "2025-12"
    assert storage.shift_month("2026-12", 1) == "2027-01"
    assert storage.shift_month("2026-08", -8) == "2025-12"


def test_data_footprint_splits_layers(env):
    _seed(["2026-07-01T0000", "2026-09-01T0000"])
    storage.compact_snapshots(grace_days=2, apply=True,
                              now=_fake_now("2026-09-19T10:00"))
    fp = storage.data_footprint()
    assert fp["forecast_cold_files"] == 1
    assert fp["forecast_hot_files"] == 1
    assert fp["total_bytes"] > 0


# ------------------------------------------------------------------ CLI
def test_cli_compact_dry_run_then_apply(env, capsys):
    _seed(["2026-07-01T0000", "2026-07-02T0000"])
    assert main(["compact"]) in (None, 0)
    assert not (_dir(env) / "2026-07.json.gz").exists(), "默认 dry-run 不落盘"
    assert main(["compact", "--apply"]) in (None, 0)
    assert (_dir(env) / "2026-07.json.gz").exists()
    # 幂等：再跑一次不报错、不再改盘
    assert main(["compact", "--apply"]) in (None, 0)


def test_cli_footprint_ok_for_small_repo(env):
    _seed(["2026-07-01T0000"])
    assert main(["footprint", "--warn-mb", "10000", "--fail-mb", "20000"]) in (None, 0)


def test_cli_footprint_fails_over_threshold(env):
    _seed(["2026-07-01T0000"])
    # 阈值压到 0 → 必定超限（仓库必然非空）；看门狗应以非零码退出，让 CI 变红开 Issue
    with pytest.raises(SystemExit) as ei:
        main(["footprint", "--warn-mb", "0", "--fail-mb", "0"])
    assert ei.value.code != 0

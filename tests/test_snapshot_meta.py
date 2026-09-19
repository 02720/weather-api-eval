"""快照元数据契约与哈希链（P0-6 / P1-1 / §7.1）的回归。

这些字段的价值全在"有没有"上：缺失时不会报错，只会让"预报是否在实况之前封存"
永远不可考。所以每一条都必须是"删掉就红"。
"""
import json

import pytest

from weather_eval import storage
from weather_eval.snapshot_meta import (
    META_SCHEMA_VERSION, canonical_bytes, integrity_summary, merkle_root,
    snapshot_complete, snapshot_sha256, stamp_snapshot,
)


def _snap(**extra):
    base = {
        "issue_iso": "2026-09-01T00:00", "station_id": "s1", "source": "test",
        "models": ["m"], "hourly_time": ["2026-09-01T01:00"] * 5,
        "data": {"m": {"temperature_2m": [1, 2, 3, 4, 5], "precipitation": [0] * 5}},
    }
    base.update(extra)
    return base


# --------------------------------------------------------------- 盖章与兼容
def test_stamp_fills_contract_fields():
    s = stamp_snapshot(_snap(), "2026-09-01T06:00:00", "2026-08-31T22:00:00Z")
    assert s["meta_schema"] == META_SCHEMA_VERSION
    assert s["fetched_at_bj"] == "2026-09-01T06:00:00"
    assert s["fetched_at_utc"] == "2026-08-31T22:00:00Z"
    assert s["issue_source"] == "unknown"          # 绝不猜一个语义
    assert s["complete"] is True and s["missing_shards"] == []
    assert s["actual_hours"] == 5
    assert s["resolution_hours"] is None           # 未知就是未知，不假设 1h
    assert s["precip_unit"] is None
    assert len(s["payload_sha256"]) == 64


def test_stamp_never_overrides_provider_declaration():
    """provider 显式声明的值优先——盖章只补空缺。"""
    s = stamp_snapshot(_snap(issue_source="model_run", complete=False,
                             missing_shards=["day=3"], resolution_hours=3,
                             precip_unit="mm", precip_accum_window_hours=6),
                       "2026-09-01T06:00:00", "2026-08-31T22:00:00Z")
    assert s["issue_source"] == "model_run"
    assert s["complete"] is False
    assert s["resolution_hours"] == 3
    assert s["precip_accum_window_hours"] == 6


def test_invalid_issue_source_is_downgraded_to_unknown():
    """自造词表正是 §6.1 那五种语义混在同一张榜上的根源，未知值一律归一。"""
    s = stamp_snapshot(_snap(issue_source="whatever"), "t", "t")
    assert s["issue_source"] == "unknown"


def test_missing_shards_implies_incomplete():
    s = stamp_snapshot(_snap(missing_shards=["day=7"]), "t", "t")
    assert s["complete"] is False


def test_legacy_snapshots_are_treated_as_complete():
    """历史存档（无 complete 字段）必须按完整处理，否则会凭空抹掉历史样本。"""
    assert snapshot_complete({}) is True
    assert snapshot_complete({"complete": None}) is True
    assert snapshot_complete({"complete": False}) is False


# ------------------------------------------------------------------- 哈希链
def test_hash_ignores_self_reference_and_is_order_independent():
    a = _snap()
    b = {k: v for k, v in reversed(list(a.items()))}   # 键顺序不同
    assert snapshot_sha256(a) == snapshot_sha256(b)
    a2 = stamp_snapshot(a, "2026-09-01T06:00:00", "2026-08-31T22:00:00Z")
    h = a2["payload_sha256"]
    # 再盖一次（抓取时刻不同）内容哈希不变：自引用字段必须被排除，否则不可复算
    a3 = stamp_snapshot(dict(b), "2027-01-01T00:00:00", "2027-01-01T00:00:00Z")
    assert a3["payload_sha256"] == h


def test_content_change_breaks_hash():
    a = _snap()
    b = _snap()
    b["data"]["m"]["temperature_2m"] = [9, 9, 9, 9, 9]
    assert snapshot_sha256(a) != snapshot_sha256(b)


def test_canonical_bytes_is_deterministic_and_compact():
    raw = canonical_bytes(_snap())
    assert b" " not in raw and json.loads(raw)["station_id"] == "s1"


def test_merkle_root_properties():
    assert merkle_root([]) is None
    h = ["aa", "bb", "cc", "dd"]
    assert merkle_root(h) == merkle_root(list(reversed(h)))   # 顺序无关
    assert merkle_root(h) != merkle_root(["aa", "bb", "cc", "de"])
    # 奇数个叶子时最后一个与自己配对，仍得到单根
    assert merkle_root(["aa", "bb", "cc"]) is not None


def test_integrity_summary_counts_missing_fetched_at():
    snaps = [stamp_snapshot(_snap(), "2026-09-01T06:00:00", "2026-08-31T22:00:00Z"),
             _snap()]
    s = integrity_summary(snaps)
    assert s["n_snapshots"] == 2
    assert s["n_without_fetched_at"] == 1
    assert s["fetched_first"] == s["fetched_last"] == "2026-09-01T06:00:00"
    assert s["merkle_root"]


# ------------------------------------------------------- 写路径强制盖章
def test_save_forecast_snapshot_stamps_metadata(tmp_path, monkeypatch):
    """盖章发生在**唯一的快照写入口**上，provider 漏写也不可能漏字段。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    assert storage.save_forecast_snapshot("s1", "m", _snap()) is True
    got = storage.list_forecast_snapshots("s1", "m")[0]
    for k in ("meta_schema", "fetched_at_bj", "fetched_at_utc", "issue_source",
              "complete", "missing_shards", "payload_sha256"):
        assert k in got, k
    # 幂等：同 issue 不重写（哈希因此也不会漂移）
    assert storage.save_forecast_snapshot("s1", "m", _snap()) is False
    assert storage.list_forecast_snapshots("s1", "m")[0]["payload_sha256"] == \
        got["payload_sha256"]


def test_manifest_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    storage.save_manifest("2026-09", {"merkle_root": "abc", "n_snapshots": 3})
    assert storage.load_manifest("2026-09")["merkle_root"] == "abc"
    assert storage.load_manifest()["n_snapshots"] == 3        # 缺省取最新
    assert storage.load_manifest("2026-01") == {}


def test_stamped_snapshot_hash_is_recomputable(tmp_path, monkeypatch):
    """verify 命令的核心不变量：从磁盘读回后重算哈希必须与落盘值一致。"""
    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    storage.save_forecast_snapshot("s1", "m", _snap())
    got = storage.list_forecast_snapshots("s1", "m")[0]
    assert snapshot_sha256(got) == got["payload_sha256"]

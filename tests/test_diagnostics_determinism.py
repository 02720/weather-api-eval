"""诊断层的确定性回归（第六轮审查修复的钉子）。

诊断层的输出会进 meta、进报告、进 CI 判定——同一份数据两次运行得出不同
结论（哪怕只是平票众数选边不同）都是可复现性事故。这里把两处曾经非确定的
计算钉死：

1. ``fingerprint_drift`` 的众数：旧实现 ``max(set(vals), key=vals.count)``
   在平票时受 PYTHONHASHSEED 影响选边，且 O(n²)；
2. ``compute_all`` 的输入契约：obs_maps / snapshots 必须同给或同不给，
   单边混口径必须当场炸出来而不是静默算出混合结果。
"""
from __future__ import annotations

import json

import pytest

from weather_eval.report.diagnostics import compute_all, fingerprint_drift


def _snap(issue: str, **fields) -> dict:
    base = {"issue_iso": issue, "resolution_hours": 1, "precip_unit": "mm",
            "precip_accum_window_hours": None, "issue_source": "model_run",
            "source": "open_meteo", "complete": True, "missing_shards": False}
    base.update(fields)
    return base


# ---------------------------------------------------------------- 众数确定性
def test_fingerprint_mode_tie_is_deterministic_across_hash_seeds():
    """平票众数必须与哈希种子无关：两个值各出现 2 次，无论 set 迭代序如何，
    选出的都是同一个（repr 字典序最小者），changes/drift 两次运行完全一致。

    旧实现在这个场景下两次进程可能给出相反结论——健康告警间歇性误报/漏报。
    """
    # 前后半周期的 issue_source 平票（各 2 个 "a" 与 "b"）
    snaps = {("s1", "m"): [
        _snap("2026-01-01T00:00", issue_source="a"),
        _snap("2026-01-02T00:00", issue_source="b"),
        _snap("2026-01-03T00:00", issue_source="a"),
        _snap("2026-01-04T00:00", issue_source="b"),
    ]}
    results = []
    import os
    import subprocess
    import sys
    from pathlib import Path
    # pytest 的 pythonpath ini 不会传给子进程，显式指 src（仓库布局固定）
    src_dir = str(Path(__file__).resolve().parents[1] / "src")
    repo_root = str(Path(__file__).resolve().parents[1])
    for seed in ("0", "1", "42", "12345"):
        snaps_json = json.dumps(snaps[("s1", "m")], ensure_ascii=False)
        code = (
            "import json,sys;"
            "from weather_eval.report.diagnostics import fingerprint_drift;"
            f"snaps={{('s1','m'): json.loads({snaps_json!r})}};"
            "r = fingerprint_drift(None, [], snapshots=snaps);"
            "print(json.dumps(r, default=str, ensure_ascii=False))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed,
                 "PYTHONPATH": src_dir + os.pathsep
                 + os.environ.get("PYTHONPATH", "")},
            capture_output=True, text=True, check=True,
            cwd=repo_root)
        results.append(out.stdout.strip())
    assert len(set(results)) == 1, f"跨 PYTHONHASHSEED 结果不一致：{results}"


def test_fingerprint_mode_prefers_earlier_value_on_tie():
    """平票时确定性地取 repr 字典序最小者（可读、可预期，不依赖任何运行状态）。"""
    snaps = {("s1", "m"): [
        _snap("2026-01-01T00:00", issue_source="zebra"),
        _snap("2026-01-02T00:00", issue_source="apple"),
        _snap("2026-01-03T00:00", issue_source="zebra"),
        _snap("2026-01-04T00:00", issue_source="apple"),
    ]}
    out = fingerprint_drift(None, [], snapshots=snaps)
    per = out["models"]["m"]
    assert per["drift"] is False          # 两半周期各自平票 → 众数相同 → 无漂移
    changes = per["changes"]
    for ch in changes or []:
        assert ch["early"] <= ch["late"]


# ---------------------------------------------------------------- 输入契约
def test_compute_all_rejects_half_paired_inputs():
    """obs_maps / snapshots 必须同给或同不给：单边给入是口径混合，当场炸。"""
    report = {"meta": {"period_label": "2026-08",
                       "leaderboards": {"all": [{"model": "a", "score": 1.0}]}},
              "leaderboards": {"all": [{"model": "a", "score": 1.0}]}}
    with pytest.raises(ValueError, match="同时给或同时不给"):
        compute_all(report, __import__("pathlib").Path("/nonexistent"),
                    obs_maps={"s1": {}})
    with pytest.raises(ValueError, match="同时给或同时不给"):
        compute_all(report, __import__("pathlib").Path("/nonexistent"),
                    snapshots={("s1", "m"): [_snap("2026-08-01T00:00")]})

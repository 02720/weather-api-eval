"""对抗式审查修复项的回归测试（P0-1 / P1-3 / P1-4 / P1-5 / P1-6）。

这些测试的共同点：**它们验证的是"保障机制"本身，而不是指标数值**。
审查报告给出的总评是"机制设计 9/10，机制验证 5/10"——项目造了一台精密的
可审计机器，却几乎没有对这台机器本身做过验收。本文件就是那台机器的验收单：

  * 哈希链：改内容必须被抓到（含"改内容 + 改自证哈希"的更强攻击者）；
  * 趋势图：与榜单逐格同值（缺一即缺）；
  * 封存门槛：实况之后才抓回来的样本必须出榜，且被如实计数；
  * 诊断层：必须真的接进了 meta，且 Holm 的有效检验数可切换、方向正确。

每一条都写成"破坏它就会红"的形式——修好了不看守，等于没修。
"""
import copy
import json

import pytest

from weather_eval import stats as st
from weather_eval.evaluate import (
    _mean2, _mean_or_none, _note_unfrozen, _summarize_seal_lag,
    temp_score, track_cells,
)
from weather_eval.snapshot_meta import (
    check_snapshot_integrity, integrity_summary, snapshot_sha256, stamp_snapshot,
)


def _snap(issue="2026-09-01T00:00", **extra):
    s = {
        "issue_iso": issue,
        "station_id": "s1",
        "source": "src",
        "models": ["m"],
        "hourly_time": [issue, "2026-09-01T01:00"],
        "data": {"m": {"temperature_2m": [20.0, 21.0],
                       "precipitation": [0.0, 0.1]}},
        "daily_time": [],
        "daily": {},
    }
    s.update(extra)
    return s


def _stamped(**extra):
    """一份带内嵌 payload_sha256 的契约快照（模拟 2026-09-19 之后的新存档）。"""
    return stamp_snapshot(_snap(**extra), "2026-09-01T00:00:00",
                          "2026-08-31T16:00:00Z")


# ------------------------------------------------------------------ P0-1 哈希链
def test_tampering_content_is_detected():
    """改内容：重算哈希必须失配（这是修复前唯一失效的场景）。"""
    s = _stamped()
    assert integrity_summary([s])["n_hash_mismatch"] == 0

    bad = copy.deepcopy(s)
    bad["data"]["m"]["temperature_2m"][0] = 99.0
    r = integrity_summary([bad])
    assert r["n_hash_mismatch"] == 1, "改了温度却没被抓到 = 验证器形同虚设"
    assert r["hash_mismatches"][0]["recomputed"] != r["hash_mismatches"][0]["stored"]


def test_recomputed_hash_hides_tampering_but_changes_the_root():
    """诚实地钉住哈希链的**能力边界**（比审查报告的结论更进一层）。

    若篡改者改完内容后还把内嵌哈希**正确地重算一遍**，那么"重算 vs 内嵌"这条
    比对是匹配的——内容哈希在定义上就无法分辨"原始内容"与"被正确重新盖章的
    内容"。这不是缺陷，是所有内容寻址方案的共同边界。

    此时仍有一道防线：Merkle 根变了。只要清单（data/manifest）不是同一次恶意
    流程里顺手重算的，清单比对就会红。因此测试断言两件事：
      1. 内容哈希比对确实匹配（承认边界，不粉饰）；
      2. Merkle 根确实变了（清单防线仍然有效）。

    真正能堵住"改数据 + 重算哈希 + 重跑报告"这条完整攻击链的，只有**仓库外的
    不可变锚点**（把根写到别处 / 签名 / 只追加日志）。这一点应在 README 的
    威胁模型里明说，而不是佯装"任何事后改写都会让 CI 变红"。
    """
    s = _stamped()
    root_before = integrity_summary([s])["merkle_root"]

    forged = copy.deepcopy(s)
    forged["data"]["m"]["temperature_2m"][0] = 99.0
    forged["payload_sha256"] = snapshot_sha256(forged)   # 正确地重算自证哈希

    r = integrity_summary([forged])
    assert r["n_hash_mismatch"] == 0, "内容哈希不该抓到被正确重新盖章的篡改（边界）"
    assert r["merkle_root"] != root_before, "连 Merkle 根都没变 = 篡改彻底隐形"


def test_wrongly_recomputed_hash_is_still_caught():
    """篡改者若把自证哈希改成一个**错误**的值（最常见：手改、脚本写错），
    内容哈希比对立刻失配——这正是修复前完全失效的场景。"""
    s = _stamped()
    forged = copy.deepcopy(s)
    forged["data"]["m"]["temperature_2m"][0] = 99.0
    forged["payload_sha256"] = "deadbeef" * 8
    assert integrity_summary([forged])["n_hash_mismatch"] == 1


def test_merkle_root_is_based_on_recomputed_not_embedded_hash():
    """Merkle 根必须基于重算值：内嵌哈希被换成任意值也不能影响根。"""
    s = _stamped()
    root_ok = integrity_summary([s])["merkle_root"]

    forged = copy.deepcopy(s)
    forged["payload_sha256"] = "0" * 64
    assert integrity_summary([forged])["merkle_root"] == root_ok, \
        "Merkle 根仍在采信快照自带的哈希"
    assert integrity_summary([forged])["n_hash_mismatch"] == 1


def test_untouched_snapshots_produce_no_false_positive():
    """未被篡改的快照必须零误报——否则"篡改必红"会退化成"永远红"，没人看。"""
    snaps = [_stamped(issue="2026-09-01T00:00"),
             _stamped(issue="2026-09-01T06:00"),
             _stamped(issue="2026-09-01T12:00")]
    r = integrity_summary(snaps)
    assert r["n_hash_mismatch"] == 0
    assert r["n_with_embedded_hash"] == 3


def test_legacy_snapshot_without_embedded_hash_is_not_flagged():
    """历史存档没有内嵌哈希 → 无从比对，绝不判失配（新契约是纯增量）。"""
    old = _snap()          # 未经 stamp，无 payload_sha256
    r = integrity_summary([old])
    assert r["n_hash_mismatch"] == 0
    assert r["n_with_embedded_hash"] == 0
    assert r["merkle_root"] == snapshot_sha256(old)


def test_check_snapshot_integrity_matches_only_when_content_intact():
    s = _stamped()
    real, ok = check_snapshot_integrity(s)
    assert ok and real == s["payload_sha256"]

    bad = copy.deepcopy(s)
    bad["data"]["m"]["precipitation"][1] = 5.0
    _, ok2 = check_snapshot_integrity(bad)
    assert not ok2


def test_verify_command_fails_on_mismatch(tmp_path, monkeypatch):
    """`verify` 必须在失配时非零退出——这才是对"验证器"的验证。

    清单比对那一步是可以被"改完再重跑一次 report"绕过的（根会跟着一起变），
    只有"重算 vs 内嵌"这条比对不依赖任何外部产物，是唯一可靠的信号。
    """
    from weather_eval import __main__ as cli
    from weather_eval import storage

    class _Args:
        config = "config/stations.yaml"
        period = "2026-09"

    snap_ok = _stamped()
    snap_bad = copy.deepcopy(snap_ok)
    snap_bad["data"]["m"]["temperature_2m"][0] = 42.0

    calls = {}

    def _fake_preload(cfg):
        return {}, {("s1", "m"): list(calls.get("snaps", []))}

    monkeypatch.setattr(cli, "_preload_snapshots", _fake_preload)
    monkeypatch.setattr(cli, "load_config", lambda p: object())
    # load_manifest 是函数内局部导入的，必须打在来源模块上
    monkeypatch.setattr(storage, "load_manifest",
                        lambda p: {"merkle_root": "x", "n_snapshots": 1})

    calls["snaps"] = [snap_bad]
    assert cli.cmd_verify(_Args()) == 1, "快照被改过，verify 却返回 0"

    calls["snaps"] = [snap_ok]
    # 快照本身干净，但根与清单不一致 → 仍应非零退出
    assert cli.cmd_verify(_Args()) == 1

    # 快照干净且与清单一致 → 必须放行（不能把门槛做成"永远红"）
    root = integrity_summary([snap_ok])["merkle_root"]
    monkeypatch.setattr(storage, "load_manifest",
                        lambda p: {"merkle_root": root, "n_snapshots": 1})
    assert cli.cmd_verify(_Args()) == 0


# ------------------------------------------------------------- P1-5 趋势与榜单
def test_trend_overall_never_exceeds_leaderboard_discipline():
    """缺一维时，趋势的综合分必须与榜单一样是"无结论"，而不是退化成单维分。

    修复前：track_cells → (None, None, None)，而趋势图用 _mean_or_none 画出
    温度分当综合分——榜单说"无结论"、图上画 86 分，正是项目反复批判的
    "半个证据当整个用"。
    """
    t = {"acc2": 80.0, "rmse": 1.8, "mbe": 0.1, "mae98": 0.5,
         "slope": 0.98, "std_ratio": 1.0, "n": 500}
    p = {}

    ts, ps = temp_score(t), None
    assert track_cells("hourly", t, p)[0] is None          # 榜单：无结论
    assert _mean_or_none([ts, ps]) == ts                    # 旧趋势算法：会画出分
    assert _mean2(ts, ps) is None                           # 新趋势算法：同纪律
    assert _mean2(ts, ps) == track_cells("hourly", t, p)[0]


def test_mean2_and_mean_or_none_differ_only_when_a_dim_is_missing():
    assert _mean2(60.0, 80.0) == _mean_or_none([60.0, 80.0]) == 70.0
    assert _mean2(60.0, None) is None
    assert _mean_or_none([60.0, None]) == 60.0


def test_score_trend_overall_uses_mean2():
    """趋势块里的 overall 必须逐格等于 track_cells 的综合分（不变量）。"""
    from weather_eval.evaluate import _score_trend

    models = ["m"]
    track_sources = {
        "hourly": {"temp": {"m": {"1d": {"acc2": 80.0, "rmse": 1.8, "mbe": 0.1,
                                         "mae98": 0.5, "slope": 0.98,
                                         "std_ratio": 1.0, "n": 500}}},
                   "precip": {"m": {}}},     # 降水维缺
        "daily": {"temp": {"m": {}}, "precip": {"m": {}}},
    }
    trend = _score_trend(models, track_sources, 1, 1)
    assert trend["hourly"]["temp"]["m"]["1d"] is not None    # 单维曲线照旧给
    assert trend["hourly"]["overall"]["m"]["1d"] is None, \
        "降水维缺测时趋势图仍画出了综合分（与榜单分叉）"
    assert trend["hourly"]["overall"]["m"]["1d"] == \
        track_cells("hourly", track_sources["hourly"]["temp"]["m"]["1d"], {})[0]


def test_score_trend_all_matches_combined_day_board():
    """综合口径（"all"）趋势与综合天榜逐格同值（榜单与趋势图永不分叉）。

    "all" 轨与 _combined_day_boards 完全同式：同一提前天数上小时榜分数与日榜
    分数各半合成，两轨缺一即缺——任一侧出现"半截综合分"都是口径分叉。"""
    from weather_eval.evaluate import _combined_day_boards, _score_trend

    t = {"acc2": 80.0, "acc1": 60.0, "rmse": 1.8, "mae": 1.2, "mbe": 0.1,
         "r": 0.9, "slope": 0.98, "n": 500}
    p = {"ets": 0.3, "ts": 0.35, "pod": 55.0, "far": 40.0, "bias": 1.2, "n": 60}
    models = ["m"]
    track_sources = {
        "hourly": {"temp": {"m": {"1d": t, "2d": t}},
                   "precip": {"m": {"1d": p, "2d": p}}},
        "daily": {"temp": {"m": {"1d": {"max": t, "min": t}, "2d": {"max": t, "min": t}}},
                  "precip": {"m": {"1d": p, "2d": p}}},
    }
    trend = _score_trend(models, track_sources, 2, 2)
    boards = _combined_day_boards(models, track_sources, 2)
    for b in ("1d", "2d"):
        row = boards[f"all:{b}"][0]
        assert trend["all"]["overall"]["m"][b] == row["score"]
        assert trend["all"]["temp"]["m"][b] == row["temp_score"]
        assert trend["all"]["precip"]["m"][b] == row["precip_score"]
    # 单轨缺降水 → 综合分与该维同样缺（缺一即缺，与榜单同纪律）
    track_sources["hourly"]["precip"] = {"m": {}}
    trend2 = _score_trend(models, track_sources, 2, 2)
    assert trend2["all"]["overall"]["m"]["1d"] is None
    assert trend2["all"]["temp"]["m"]["1d"] is not None   # 维度分按维度各自合成


# ------------------------------------------------------------- P1-4 封存时点
def test_unfrozen_samples_are_counted_and_excludable():
    """封存门槛的统计口径：无论排不排除，"有多少/滞后多久"都必须如实记录。"""
    stats = {}
    _note_unfrozen(stats, "m", 5.0, frozen_excluded=True)
    _note_unfrozen(stats, "m", 9.0, frozen_excluded=True)
    s = _summarize_seal_lag(stats)
    assert s["available"] is True
    assert s["n_unfrozen"] == 2 and s["n_excluded"] == 2
    assert s["by_model"]["m"]["lag_median_h"] == 7.0
    assert s["by_model"]["m"]["lag_max_h"] == 9.0

    stats2 = {}
    _note_unfrozen(stats2, "m", 5.0, frozen_excluded=False)
    s2 = _summarize_seal_lag(stats2)
    assert s2["n_unfrozen"] == 1 and s2["n_excluded"] == 0


def test_seal_lag_summary_absent_when_nothing_violated():
    assert _summarize_seal_lag({})["available"] is False
    assert _summarize_seal_lag(None)["available"] is False


def test_collect_excludes_samples_fetched_after_valid_time(tmp_path, monkeypatch):
    """有效时刻早于抓取时刻的样本必须出榜，并被计入封存滞后披露。"""
    from datetime import datetime, timedelta

    from weather_eval import storage
    from weather_eval.evaluate import collect

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))

    issue = datetime(2026, 8, 1, 0, 0)
    times = [(issue + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
             for i in range(6)]
    snap = {
        "issue_iso": "2026-08-01T00:00",
        "station_id": "s1", "source": "src", "models": ["m"],
        "hourly_time": times,
        "data": {"m": {"temperature_2m": [20.0] * 6,
                       "precipitation": [0.0] * 6}},
        "daily_time": [], "daily": {},
        # 抓取发生在起报后 3 小时 → 前 3 个有效时刻已经过去了
        "fetched_at_bj": "2026-08-01T03:00:00",
    }
    storage.save_forecast_snapshot("s1", "m", snap)

    obs = {t: {"temp": 20.0, "rain": 0.0} for t in times}
    start, end = datetime(2026, 8, 1), datetime(2026, 8, 2)

    stats = {}
    strict, _ = collect(["s1"], ["m"], start, end, 1, 16,
                        obs_maps={"s1": obs}, require_frozen=True, stats=stats)
    loose, _ = collect(["s1"], ["m"], start, end, 1, 16,
                       obs_maps={"s1": obs}, require_frozen=False)

    assert len(strict) == 3, "实况之后才抓回来的样本没有被排除"
    # 宽松口径下 6 个时刻里，起报当刻（lead=0）本身就不入样，故为 5
    assert len(loose) == 5
    # 6 个时刻里 00:00 的 lead=0 本来就入不了样；01、02 落在抓取时刻之前 → 未封存
    assert stats["n_unfrozen"] == 2 and stats["n_unfrozen_excluded"] == 2
    # 留存下来的必须都是"抓取时刻 ≤ 有效时刻"的
    from weather_eval.timeutil import parse_iso
    for r in strict:
        assert parse_iso(r["valid_iso"]) >= parse_iso("2026-08-01T03:00:00")


def test_collect_keeps_snapshots_without_fetched_at(tmp_path, monkeypatch):
    """无 fetched_at 的历史存档无从判定 → 按已封存处理，绝不凭空抹掉样本。"""
    from datetime import datetime, timedelta

    from weather_eval import storage
    from weather_eval.evaluate import collect

    monkeypatch.setenv("WEATHER_EVAL_DATA_ROOT", str(tmp_path))
    issue = datetime(2026, 8, 1, 0, 0)
    times = [(issue + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M")
             for i in range(4)]
    snap = {
        "issue_iso": "2026-08-01T00:00",
        "station_id": "s1", "source": "src", "models": ["m"],
        "hourly_time": times,
        "data": {"m": {"temperature_2m": [20.0] * 4,
                       "precipitation": [0.0] * 4}},
        "daily_time": [], "daily": {},
    }
    storage.save_forecast_snapshot("s1", "m", dict(snap))
    # 盖章发生在唯一写入口里，这里把落盘后的字段去掉，模拟契约上线前的老存档
    path = tmp_path / "forecasts" / "s1" / "m" / "2026-08-01T0000.json"
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    on_disk.pop("fetched_at_bj", None)
    path.write_text(json.dumps(on_disk, ensure_ascii=False), encoding="utf-8")

    obs = {t: {"temp": 20.0, "rain": 0.0} for t in times}
    stats = {}
    got, _ = collect(["s1"], ["m"], datetime(2026, 8, 1), datetime(2026, 8, 2),
                     1, 16, obs_maps={"s1": obs}, require_frozen=True, stats=stats)
    # 4 个时刻里 00:00 的 lead=0 本来就入不了样，其余 3 个必须全部保留
    assert len(got) == 3, "老存档缺 fetched_at 字段就被整批抹掉了"
    assert stats.get("n_unfrozen", 0) == 0


# ------------------------------------------------------------------ P1-3 Holm
def test_holm_m_eff_relaxes_correction_monotonically():
    """m_eff 越小校正越松（显著的家数单调不减）——方向不能反。"""
    ps = [0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
    strict = st.holm_bonferroni(ps, 0.10)                 # m = 10
    loose = st.holm_bonferroni(ps, 0.10, m_eff=2.0)       # 折算后
    assert sum(loose) >= sum(strict), "m_eff 更小反而判得更严"


def test_holm_m_eff_never_stricter_than_standard_holm():
    """m_eff > m 时不得比标准 Holm 更严；m_eff < 1 时不得让分母归零。"""
    ps = [0.01, 0.02, 0.09]
    base = st.holm_bonferroni(ps, 0.10)
    assert st.holm_bonferroni(ps, 0.10, m_eff=100) == base
    assert st.holm_bonferroni(ps, 0.10, m_eff=0.1) == st.holm_bonferroni(ps, 0.10, m_eff=1.0)
    assert st.holm_bonferroni([], 0.10, m_eff=2.0) == []


def test_holm_handles_none_pvalues_with_m_eff():
    out = st.holm_bonferroni([0.001, None, 0.001], 0.10, m_eff=1.5)
    assert out == [True, False, True]


def test_diagnostics_module_is_wired_into_report_meta():
    """诊断层不能再是死代码：build_report 必须把它算出来并挂进 meta。

    审查 P1-3 指出 report/diagnostics.py 705 行零调用。这里用最小合成数据
    跑一次 build_report，断言 meta.diagnostics 确实存在且结构完整。
    """
    from datetime import datetime

    from weather_eval.evaluate import build_report

    cfg = {
        "temp_accuracy_limits": [1.0, 2.0, 3.0],
        "rain_threshold_mm": 1.0,
        "rain_daily_threshold_mm": 1.0,
        "min_sample": 1,
        "hourly_lead_days": 1,
        "daily_max_offset_days": 1,
        "enable_diagnostics": False,     # 真跑诊断要扫全量存档，太慢
    }
    rep = build_report(["s1"], ["m"], cfg,
                       datetime(2026, 8, 1), datetime(2026, 8, 2),
                       period_label="2026-08")
    assert "diagnostics" in rep["meta"], "诊断层没有接进 meta"
    assert rep["meta"]["diagnostics"]["available"] is False
    assert "reason" in rep["meta"]["diagnostics"]


def test_diagnostics_compute_all_is_pure_and_does_not_move_scores():
    """诊断层只读不写：诊断前后总榜逐位分数必须完全不变。"""
    from weather_eval.report import diagnostics as dg

    scores_before = [10.0, 20.0, 30.0]
    report = {
        "meta": {"period_label": "2026-08"},
        "leaderboards": {"all": [{"model": "a", "score": 10.0},
                                 {"model": "b", "score": 20.0},
                                 {"model": "c", "score": 30.0}]},
    }
    snapshot = copy.deepcopy(report)
    dg.compute_all(report, __import__("pathlib").Path("/nonexistent"))
    assert [r["score"] for r in report["leaderboards"]["all"]] == scores_before
    assert report["leaderboards"] == snapshot["leaderboards"]


# ------------------------------------------------------- P1-6 真值传输完整性
def test_truth_transport_flags_plaintext_sources():
    from weather_eval.health import truth_transport

    class S:
        def __init__(self, sid, name, url):
            self.id, self.name, self.obs_url = sid, name, url

    sts = [S("wuzhou", "梧州", "http://eia-data.com/x"),
           S("bobai", "博白", "https://secure.example/y")]
    r = truth_transport(sts)
    assert r["n_plaintext"] == 1 and r["n_total"] == 2
    assert r["plaintext_hosts"] == ["eia-data.com"]
    assert r["all_encrypted"] is False
    assert "明文" in r["verdict"]


def test_truth_transport_all_https_is_clean():
    from weather_eval.health import truth_transport

    class S:
        def __init__(self, sid, name, url):
            self.id, self.name, self.obs_url = sid, name, url

    r = truth_transport([S("a", "A", "https://x/y"), S("b", "B", "https://x/z")])
    assert r["all_encrypted"] is True and r["n_plaintext"] == 0

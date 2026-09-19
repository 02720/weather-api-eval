"""观测多源编排（obs/chain.py）的回归测试。

重点是把三条设计决定钉死：
  1. **优先级在编排层解决**——高优先级源已覆盖的小时不被低优先级源覆盖，
     绝不做"谁后写谁赢"（那会让权威值取决于抓取顺序，不可复现）；
  2. **"返回了 200"不等于"可用"**——陈旧 / 截断 / 抛错都触发降级，但被降级的源
     其已有的历史小时仍然参与合并（所以主源停摆的结果是"新旧拼接"，不是"整段换源"）；
  3. **降级必须留痕**——ChainReport 要说清谁试过、谁顶上、补了多少位。
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from weather_eval.obs.chain import (
    DEFAULT_MIN_HOURS, OBS_SOURCE_PRIORITY, ObsChain,
)
from weather_eval.timeutil import iso, now_beijing


def _recs(latest_age_h: float, n: int, *, temp: float = 20.0,
          source: str = "x", rain: float = 0.0) -> list[dict]:
    """构造 n 条逐小时观测，最新一条滞后 latest_age_h 小时。"""
    end = now_beijing().replace(minute=0, second=0, microsecond=0) \
        - timedelta(hours=latest_age_h)
    out = []
    for i in range(n):
        t = end - timedelta(hours=i)
        out.append({"time": iso(t), "source": source, "temp": temp, "rain": rain})
    return out


class _Src:
    def __init__(self, records=None, error=None, name="s"):
        self.records = records
        self.error = error
        self.name = name
        self.calls = 0

    def fetch(self, station):
        self.calls += 1
        if self.error:
            raise RuntimeError(self.error)
        return list(self.records or [])


class _Station:
    id = "s1"


def _chain(primary, backup):
    return ObsChain({"eia_data": primary, "cma_data": backup},
                    priority=OBS_SOURCE_PRIORITY)


# ------------------------------------------------------------ 主源可用时不惊动备用源
def test_healthy_primary_does_not_touch_backup():
    """备用源的应有语义：平时完全不碰它，主源一坏立刻顶上。

    这条同时是成本约束——CMA 源需要逐整点请求，不该在每轮都白跑一遍。
    """
    p = _Src(_recs(1, 24, source="wd"))
    b = _Src(_recs(0, 27, source="cma_data"))
    recs, rep = _chain(p, b).fetch(_Station())
    assert len(recs) == 24
    assert b.calls == 0, "主源健康时不应请求备用源"
    assert rep.degraded is False
    assert rep.n_filled == 0
    assert rep.served_by == ["eia_data"]


def test_priority_is_declared_and_backup_is_second():
    assert OBS_SOURCE_PRIORITY[0] == "eia_data"
    assert "cma_data" in OBS_SOURCE_PRIORITY


# ------------------------------------------------------------------ 降级路径
def test_primary_failure_falls_back_to_backup():
    p = _Src(error="页面改版，未解析到任何观测记录")
    b = _Src(_recs(0, 27, source="cma_data"))
    recs, rep = _chain(p, b).fetch(_Station())
    assert len(recs) == 27
    assert rep.degraded is True
    assert rep.n_filled == 27
    assert rep.served_by == ["cma_data"]
    assert rep.attempts[0].error and "页面改版" in rep.attempts[0].error
    assert rep.attempts[1].ok is True


def test_stale_primary_merges_old_hours_and_backup_fills_gap():
    """主源停摆：它已有的旧小时仍然有效，由备用源补齐最近几小时。

    这正是"空数据绝不伪装"之外的另一半：主源不是坏掉，是**滞后**——旧小时不能丢。
    窗口关系：主源覆盖 now-9h..now-18h（10 小时），备用源覆盖 now..now-11h（12 小时），
    重叠 3 小时归主源 → 并集 19 小时，备用源实际补位 9 小时。
    """
    p = _Src(_recs(9, 10, source="wd"))        # 最新滞后 9 小时 → 陈旧
    b = _Src(_recs(0, 12, source="cma_data"))
    recs, rep = _chain(p, b).fetch(_Station())
    times = [r["time"] for r in recs]
    assert len(recs) == 19, "按小时去重后应为并集大小"
    assert rep.attempts[0].n_used == 10
    assert rep.attempts[1].n_used == 9
    assert rep.n_filled == 9
    assert rep.degraded is True
    by_time = {r["time"]: r["source"] for r in recs}
    assert by_time[max(times)] == "cma_data", "最近的小时必须由备用源补上"
    assert by_time[min(times)] == "wd", "陈旧的老小时仍是主源提供的实况"


def test_truncated_primary_is_degraded_and_backup_fills():
    p = _Src(_recs(1, 3, source="wd"))          # 3 < min_hours → 截断
    b = _Src(_recs(0, 24, source="cma_data"))
    recs, rep = _chain(p, b).fetch(_Station())
    assert rep.degraded is True
    assert len(recs) == 24
    assert rep.n_filled == 21
    assert "门槛" in (rep.attempts[0].degraded_reason or "")


def test_higher_priority_hour_is_never_overwritten():
    """同一小时两家都有值时，以高优先级源为准（不是"后写的赢"）。"""
    p = _Src(_recs(1, 5, temp=11.0, source="wd"))
    b = _Src(_recs(1, 5, temp=99.0, source="cma_data"))   # 故意给不同的值
    recs, rep = _chain(p, b).fetch(_Station())
    assert len(recs) == 5
    assert all(r["temp"] == 11.0 for r in recs), "备用源不得覆盖主源已给的时刻"
    assert rep.n_filled == 0


def test_merge_order_is_independent_of_return_order():
    """优先级判定必须与"哪个源先返回"无关——否则权威值取决于运行顺序。"""
    p = _Src(_recs(1, 4, temp=11.0, source="wd"))
    b = _Src(_recs(1, 4, temp=99.0, source="cma_data"))
    recs_a, _ = _chain(p, b).fetch(_Station())
    recs_b, _ = _chain(p, b).fetch(_Station())
    assert [(r["time"], r["temp"]) for r in recs_a] == \
           [(r["time"], r["temp"]) for r in recs_b]


def test_all_sources_failed_raises_loudly():
    """全部源失败 → 抛错。绝不返回空列表冒充"本轮没有新观测"（那是静默失效）。"""
    with pytest.raises(RuntimeError) as ei:
        _chain(_Src(error="A 挂了"), _Src(error="B 也挂了")).fetch(_Station())
    assert "全部观测源均不可用" in str(ei.value)


def test_sources_without_usable_payload_are_dropped():
    """气温与降水都为缺测的记录对评估毫无价值，不该占据存档并让覆盖率虚高。"""
    junk = [{"time": iso(now_beijing().replace(minute=0, second=0, microsecond=0)),
             "source": "wd", "temp": None, "rain": None}]
    p = _Src(junk)
    b = _Src(_recs(0, DEFAULT_MIN_HOURS, source="cma_data"))
    recs, rep = _chain(p, b).fetch(_Station())
    assert rep.attempts[0].n_records == 0
    assert all(r["temp"] is not None or r["rain"] is not None for r in recs)
    assert rep.served_by == ["cma_data"]


# ------------------------------------------------------------------ 留痕
def test_report_serialisable_and_descriptive():
    p = _Src(_recs(1, 24, source="wd"))
    b = _Src(_recs(0, 27, source="cma_data"))
    _recs_out, rep = _chain(p, b).fetch(_Station())
    d = rep.as_dict()
    assert set(d) == {"degraded", "n_records", "n_filled", "served_by", "attempts"}
    assert d["attempts"][0]["name"] == "eia_data"
    assert d["attempts"][1]["attempted"] is False   # 未尝试也要如实记录
    assert "eia_data" in rep.describe()

    rep2 = _chain(_Src(error="boom"), _Src(_recs(0, 27))).fetch(_Station())[1]
    assert "失败" in rep2.describe()
    assert rep2.attempts[0].error == "boom"


def test_single_source_chain_assignment_form():
    """只登记一个源时，编排退化为直通（用于 --source eia_data 单一源排障）。"""
    p = _Src(_recs(1, 24, source="wd"))
    chain = ObsChain({"eia_data": p}, priority=("eia_data",))
    recs, rep = chain.fetch(_Station())
    assert len(recs) == 24
    assert rep.degraded is False
    assert len(rep.attempts) == 1

"""多观测源编排：主源优先，主源失败/陈旧/截断时自动降级到备用源。

第一性原理
----------
观测是本项目**唯一的真值来源**（README §1 三条地基之一）。此前它由单点承担
（eia-data.com 一个页面），页面一改版 → 全部评估停摆，而且下一次运行也不会自己
好起来。备用源的价值不在"更准"，而在"不断供"。

但"能抓到"和"能当真相用"是两件事。跨源合并有三个必须显式回答的问题：

**1. 谁说了算？**
优先级在**编排层**决定，绝不采用"谁后写谁赢"——那会让权威值取决于抓取顺序，
是一个不可复现的隐式状态（同一批数据、换个运行顺序就得到不同的存档）。规则：
按 `OBS_SOURCE_PRIORITY` 从高到低依次尝试；高优先级源已经给出的小时，低优先级源
**不得覆盖**，只在主源没有该小时时补位。补位条数计入 `ChainReport.n_filled`，
"这轮其实是谁顶上的"因此可考。

**2. 主源返回了 200，就等于它可用吗？**
不等于。这正是本项目已经吃过一次亏的静默失效形态：eia-data 页面改版后仍可能
正常返回 200 与一份**陈旧或截断**的观测（见 obs/eia_data.py 的 P2-1 教训——
原先的新鲜度检查还曾写在 `raise` 之后的死代码里）。因此可用性判定不只看异常：

    · 抛异常                          → 降级
    · 一条记录都没解析出来            → 降级
    · 最新观测滞后 > stale_hours      → 降级（抓取通了，但页面数据停摆）
    · 覆盖小时数 < min_hours          → 降级（窗口被截断）

被降级的源**不丢弃已有记录**：它给出的历史小时仍是有效实况，照常参与合并，只是
不再享有"说了算"的地位——由备用源补齐它缺失的时段。所以"主源停摆 8 小时"的结果是
"旧的小时来自主源 + 新的小时来自备用源"，而不是"整段换源"。

**3. 降级要留痕。**
`ChainReport` 逐源记录：试没试、成没成、各贡献了多少小时、为什么降级、最新观测
有多旧。日志与健康看板据此说话——否则等主源彻底坏掉时，没人知道备用源已经在
独自支撑多久了（那正是"静默死亡"的另一种写法）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from ..timeutil import now_beijing, parse_iso
from .base import ObsSource

logger = logging.getLogger(__name__)

# 观测源优先级（高 → 低）。主源与备用源的登记处：新增源只需改这一处。
# 为什么 eia-data 仍居首：它一次请求即返回整页 24h 逐小时，请求量最小、字段最全
# （含气压/湿度/风速/风向）；CMA 源需要逐整点请求，代价更高。备用源的角色是
# "主源不可用时顶上"，而不是"平时并行抓一遍"。
OBS_SOURCE_PRIORITY: tuple[str, ...] = ("eia_data", "cma_data")

# 最新的把小时（判定"主源是否停摆"）。与 obs/eia_data.py 的 STALE_HOURS 同口径：
# 逐小时观测在整点后数分钟内到报，滞后超过 3 小时即不是延迟而是停摆。
DEFAULT_STALE_HOURS = 3.0

# 一轮至少要有多少小时才算"窗口没被截断"。eia-data 正常返回 24 条；
# 该门槛只负责拦住"明显截断"，真正的可用性还有异常/新鲜度两条。
DEFAULT_MIN_HOURS = 6


@dataclass
class SourceAttempt:
    """单个观测源在本轮的尝试结果（诊断与留痕的唯一载体）。"""

    name: str
    attempted: bool = False
    ok: bool = False
    n_records: int = 0        # 该源本轮返回的记录数
    n_used: int = 0           # 最终进入合并结果的条数（受优先级约束）
    latest: str | None = None  # 该源最新观测时刻
    age_hours: float | None = None
    degraded_reason: str | None = None
    error: str | None = None

    def as_dict(self) -> dict:
        return {
            "name": self.name, "attempted": self.attempted, "ok": self.ok,
            "n_records": self.n_records, "n_used": self.n_used,
            "latest": self.latest, "age_hours": self.age_hours,
            "degraded_reason": self.degraded_reason, "error": self.error,
        }


@dataclass
class ChainReport:
    """多源合并结果的留痕：谁被尝试、谁顶上、补了多少位。"""

    attempts: list[SourceAttempt] = field(default_factory=list)
    n_records: int = 0
    n_filled: int = 0     # 由**非首选可用源**补位的小时数
    served_by: list[str] = field(default_factory=list)

    @property
    def degraded(self) -> bool:
        """首选源未直接可用（失败/陈旧/截断）即为降级——CI 与健康看板按此告警。"""
        if not self.attempts:
            return False
        first = self.attempts[0]
        return not (first.ok and first.degraded_reason is None)

    def as_dict(self) -> dict:
        return {
            "degraded": self.degraded,
            "n_records": self.n_records,
            "n_filled": self.n_filled,
            "served_by": self.served_by,
            "attempts": [a.as_dict() for a in self.attempts],
        }

    def describe(self) -> str:
        parts = []
        for a in self.attempts:
            if not a.attempted:
                parts.append(f"{a.name}: 未尝试")
            elif a.ok and a.degraded_reason is None:
                parts.append(f"{a.name}: 可用（{a.n_records} 条，用 {a.n_used}）")
            elif a.ok:
                parts.append(f"{a.name}: 降级（{a.degraded_reason}，{a.n_records} 条，用 {a.n_used}）")
            else:
                parts.append(f"{a.name}: 失败（{a.error or a.degraded_reason}）")
        return "；".join(parts)


def _payload_useful(rec: dict) -> bool:
    """这条记录是否携带可评估的信息。

    气温与降水都为缺测的记录对评估毫无用处（温度/降水两维都没分），却会占据
    存档位置并让"覆盖率"虚高。缺测本身不折算是底线（见 README §1），
    但"两个要素都没有"连缺测都算不上——它是空的。
    """
    return rec.get("temp") is not None or rec.get("rain") is not None


def _usability(recs: list[dict], *, stale_hours: float, min_hours: int) -> tuple[bool, str | None, str | None, float | None]:
    """判定一个源的产出是否可直接采用。

    返回 (可用?, 降级原因, 最新观测时刻, 滞后小时数)。**不可用不等于丢弃**——
    调用方仍会合并它的记录，只是不让它"说了算"。
    """
    if not recs:
        return False, "无任何记录", None, None
    try:
        latest_dt = max(parse_iso(r["time"]) for r in recs if r.get("time"))
    except (ValueError, KeyError, TypeError):
        return False, "记录时间不可解析", None, None
    latest = latest_dt.strftime("%Y-%m-%dT%H:%M")
    age = (now_beijing() - latest_dt).total_seconds() / 3600
    if age > stale_hours:
        return False, f"最新观测滞后 {age:.1f}h（阈值 {stale_hours:g}h）", latest, age
    if len(recs) < min_hours:
        return False, f"仅 {len(recs)} 小时（门槛 {min_hours}h）", latest, age
    return True, None, latest, age


class ObsChain:
    """按优先级编排多个观测源：主源优先，必要时降级补位。"""

    def __init__(
        self,
        sources: dict[str, ObsSource],
        *,
        priority: tuple[str, ...] = OBS_SOURCE_PRIORITY,
        stale_hours: float = DEFAULT_STALE_HOURS,
        min_hours: int = DEFAULT_MIN_HOURS,
    ):
        # 只保留"既登记了优先级、又真的被构造出来"的源；顺序即权威顺序
        self.sources = {name: sources[name] for name in priority if name in sources}
        self.priority = tuple(n for n in priority if n in self.sources)
        self.stale_hours = stale_hours
        self.min_hours = min_hours

    def fetch(self, station: Any) -> tuple[list[dict], ChainReport]:
        """抓取一个站的观测；返回 (记录列表, 留痕报告)。

        记录按时间倒序（最新在前），与各 ObsSource 的既有约定一致。
        全部源都失败时抛 RuntimeError——**绝不返回空列表冒充成功**：那会让上层
        把"观测断供"误当成"本轮没有新观测"，正是最该避免的静默失效。
        """
        report = ChainReport()
        merged: dict[str, dict] = {}       # time_iso -> record
        owner: dict[str, str] = {}         # time_iso -> 提供该小时的源
        served: list[str] = []

        for i, name in enumerate(self.priority):
            src = self.sources[name]
            attempt = SourceAttempt(name=name, attempted=True)
            report.attempts.append(attempt)
            try:
                recs = src.fetch(station)
            except Exception as e:  # noqa: BLE001  任一源失败都不该阻断其他源
                attempt.error = str(e)
                logger.warning("观测源 %s 在站点 %s 失败: %s", name, station.id, e)
                continue

            recs = [r for r in recs if r.get("time") and _payload_useful(r)]
            attempt.n_records = len(recs)
            ok, reason, latest, age = _usability(
                recs, stale_hours=self.stale_hours, min_hours=self.min_hours)
            attempt.ok = ok
            attempt.degraded_reason = reason
            attempt.latest = latest
            attempt.age_hours = age
            if reason:
                logger.warning("观测源 %s 在站点 %s 降级：%s（其记录仍参与合并）",
                               name, station.id, reason)

            # 合并：高优先级源已经覆盖的小时不被覆盖（优先级在编排层解决）
            n_used = 0
            for r in recs:
                t = r["time"]
                if t in owner:
                    continue
                merged[t] = r
                owner[t] = name
                n_used += 1
            attempt.n_used = n_used
            if n_used:
                served.append(name)
                if i > 0:
                    report.n_filled += n_used

            # 首选源（且可用）已采集到足够窗口 → 无需惊动备用源。
            # 这是"备用"的应有语义：平时完全不碰它，主源一坏立刻顶上。
            if ok and i == 0 and len(merged) >= self.min_hours:
                break

        if not merged:
            raise RuntimeError(
                f"站点 {station.id} 全部观测源均不可用（{report.describe()}）")

        # 未被尝试的源也要如实登记（attempted=False）。否则读者无法区分
        # "备用源不需要"与"备用源根本没登记"——而这两者的运维含义完全相反。
        seen = {a.name for a in report.attempts}
        for name in self.priority:
            if name not in seen:
                report.attempts.append(SourceAttempt(name=name, attempted=False))

        report.n_records = len(merged)
        report.served_by = served
        if report.degraded:
            logger.warning("站点 %s 观测已降级：%s", station.id, report.describe())
        else:
            logger.info("站点 %s 观测正常（%d 条）", station.id, len(merged))
        return [merged[k] for k in sorted(merged, reverse=True)], report

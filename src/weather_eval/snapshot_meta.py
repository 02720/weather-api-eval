"""快照元数据契约与哈希链（对抗式审查 P1-1、P0-6、§7.1 的落地）。

这个模块回答两个此前无法回答的问题：

1. **这份快照到底是什么时候抓的？**（P1-1）
   存档里原先只有 provider 自己声明的 `issue_iso`，没有任何抓取时刻。于是当某个
   provider 把"抓取时刻"或"页面缓存更新时间"填进 `issue_iso` 时，不但没有机器检查
   能发现，事后**也没有任何存档证据能追溯**——"预报必须在实况之前封存"这条地基
   完全依赖 provider 作者的自觉。现在每份快照落盘时都会带上 `fetched_at_bj` /
   `fetched_at_utc` 与内容哈希，这条不变量从"自我声明"变成"可事后核验"。

2. **这家被评在什么口径上？**（P0-6 / §6.1）
   `issue_iso` 在本项目里其实有五种互不等价的物理语义（真实模式轮次 / 时间轴首点 /
   数据更新时间 / 请求时刻下取整 / 未知），却在同一张排行榜上比较。契约新增
   `issue_source` 枚举 + `issue_raw`（provider 声明锚点所依据的原始字段），评估层
   据此分层披露，并对"锚点≈抓取时刻"的源加上醒目标记。

另外三个字段服务于 §6.2 / §6.4 的类型系统：`complete` / `missing_shards`（残缺快照
不再伪装成完整外壳）、`precip_unit` + `precip_accum_window_hours`（累计量口径）、
`resolution_hours`（原生分辨率——3 小时产品被平铺成逐小时后不应与原生 1 小时产品
混为一谈）。

设计原则：**纯增量、向后兼容**。字段全部可选，缺省即旧行为；stamp 只补齐不覆盖
（provider 显式声明的值优先），历史存档（没有这些字段）一律按"未知/完整"处理，
既不改变任何既有指标口径，也不凭空抹掉历史样本。
"""
from __future__ import annotations

import hashlib
import json

# 契约版本：字段语义发生变化时递增（评估层据此判断快照是否具备某项披露）
META_SCHEMA_VERSION = 2

# 起报锚点语义枚举（P0-6 / §6.1）。评估层按此分层，并据此判定"嫌疑口径"。
ISSUE_SOURCE_MODEL_RUN = "model_run"          # 真实模式轮次（UTC run 换算）
ISSUE_SOURCE_AXIS_START = "axis_start"        # 时间轴首点
ISSUE_SOURCE_DATA_UPDATED = "data_updated"    # 数据更新时间（≈抓取时刻）
ISSUE_SOURCE_REQUEST_FLOOR = "request_floor"  # 请求时刻下取整（≈抓取时刻）
ISSUE_SOURCE_UNKNOWN = "unknown"
VALID_ISSUE_SOURCES = frozenset({
    ISSUE_SOURCE_MODEL_RUN, ISSUE_SOURCE_AXIS_START, ISSUE_SOURCE_DATA_UPDATED,
    ISSUE_SOURCE_REQUEST_FLOOR, ISSUE_SOURCE_UNKNOWN,
})

# 面向读者的白话标签（页面/看板共用；契约词汇表的**唯一出处**）
ISSUE_SOURCE_LABELS = {
    ISSUE_SOURCE_MODEL_RUN: "模式轮次（真实起报时次）",
    ISSUE_SOURCE_AXIS_START: "时间轴首点（产品的起始时刻）",
    ISSUE_SOURCE_DATA_UPDATED: "数据更新时间（≈抓取时刻）",
    ISSUE_SOURCE_REQUEST_FLOOR: "请求时刻下取整（≈抓取时刻）",
    ISSUE_SOURCE_UNKNOWN: "未知（无法从响应验证）",
}

# 被评在更容易样本上的嫌疑口径：起报锚点若实质等于抓取时刻，声明时效会系统性长于
# 实际时效。这些源的名次必须带标记，不能与"真实模式轮次"的源混为一谈。
SUSPECT_ISSUE_SOURCES = frozenset({
    ISSUE_SOURCE_DATA_UPDATED, ISSUE_SOURCE_REQUEST_FLOOR, ISSUE_SOURCE_UNKNOWN,
})

# 参与哈希计算的字段名（自引用字段必须排除，否则哈希不可复算）
_HASH_EXCLUDED = ("payload_sha256", "fetched_at_bj", "fetched_at_utc", "meta_schema")


def snapshot_complete(snap: dict) -> bool:
    """快照是否完整（契约字段 complete）。

    缺字段（2026-09 之前的历史存档）一律按**完整**处理——新字段是纯增量，绝不
    因为"老存档没有这个键"就把它排除掉，那会凭空抹掉历史样本。只有显式
    complete=false 才算残缺（如 MSN 的 10 个分片只抓到部分却照常生成完整外壳）。
    """
    v = snap.get("complete")
    return True if v is None else bool(v)


def canonical_bytes(snapshot: dict) -> bytes:
    """快照的确定性序列化（排序键、紧凑分隔符、UTF-8）。

    哈希必须可跨机器复算，故排除自引用字段后按键排序序列化——dict 的插入顺序
    在不同 provider 之间不保证一致，不能作为哈希输入。
    """
    payload = {k: v for k, v in snapshot.items() if k not in _HASH_EXCLUDED}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def snapshot_sha256(snapshot: dict) -> str:
    """快照内容哈希（sha256，十六进制）。"""
    return hashlib.sha256(canonical_bytes(snapshot)).hexdigest()


def merkle_root(hashes: list[str]) -> str | None:
    """由一串叶哈希计算 Merkle 根（§7.1）。

    把当轮所有快照的哈希聚成一个根，写进报告与月度归档页：任何人拿仓库里的快照
    文件重算一遍，就能验证"这批预报确实在实况之前封存、且事后未被修改"。这是把
    "工程自律"升级为"可验证"的最小实现（成本约百行，收益是整个项目的护城河）。
    """
    if not hashes:
        return None
    level = sorted(h.lower() for h in hashes)
    while len(level) > 1:
        nxt: list[str] = []
        for i in range(0, len(level), 2):
            a = level[i]
            b = level[i + 1] if i + 1 < len(level) else level[i]
            nxt.append(hashlib.sha256((a + b).encode("ascii")).hexdigest())
        level = nxt
    return level[0]


def integrity_summary(snapshots: list[dict]) -> dict:
    """一组快照的完整性摘要：条数、Merkle 根、覆盖时间范围。"""
    hashes = []
    stamps = []
    for s in snapshots:
        h = s.get("payload_sha256")
        if not h:
            h = snapshot_sha256(s)
        hashes.append(h)
        t = s.get("fetched_at_bj")
        if t:
            stamps.append(str(t))
    return {
        "n_snapshots": len(snapshots),
        "merkle_root": merkle_root(hashes),
        "fetched_first": min(stamps) if stamps else None,
        "fetched_last": max(stamps) if stamps else None,
        "n_without_fetched_at": len(snapshots) - len(stamps),
    }


def _count_actual_hours(snapshot: dict) -> int | None:
    """逐小时时间轴上的实际点数（用于披露"这份快照覆盖多少小时"）。"""
    times = snapshot.get("hourly_time")
    return len(times) if isinstance(times, list) else None


def stamp_snapshot(snapshot: dict, fetched_bj: str, fetched_utc: str,
                   provider_defaults: dict | None = None) -> dict:
    """给快照补齐契约元数据（就地返回同一 dict，便于调用方链式使用）。

    补齐规则（**只补不覆盖**，provider 显式声明的值永远优先）：
      meta_schema      契约版本
      fetched_at_bj    抓取时刻（北京时，墙钟）
      fetched_at_utc   抓取时刻（UTC）
      issue_source     缺省 unknown（绝不猜一个语义）
      complete         缺失时按 missing_shards 推断：有缺片即 False
      missing_shards   缺省 []（显式空列表 = "我知道它完整"）
      actual_hours     逐小时点数
      resolution_hours 原生分辨率（缺省 None = 未知，不假设 1h）
      precip_unit / precip_accum_window_hours  降水口径（缺省 None = 未声明）
      payload_sha256   内容哈希（排除自引用字段后计算）
    """
    snap = snapshot
    snap.setdefault("meta_schema", META_SCHEMA_VERSION)
    snap.setdefault("fetched_at_bj", fetched_bj)
    snap.setdefault("fetched_at_utc", fetched_utc)
    if provider_defaults:
        for k, v in provider_defaults.items():
            snap.setdefault(k, v)
    src = snap.get("issue_source")
    # "声明过"必须与"有值"分开记录：盖章会给所有快照补上 issue_source（未知则填
    # unknown），因此字段**存在**不代表 provider 说过话。评估层靠这个布尔量区分
    # "provider 明确说是未知"与"这份存档根本没声明"——后者是历史存量，不该判争议。
    snap.setdefault("issue_source_declared",
                    bool(isinstance(src, str) and src in VALID_ISSUE_SOURCES))
    snap["issue_source"] = src if src in VALID_ISSUE_SOURCES else ISSUE_SOURCE_UNKNOWN
    miss = snap.get("missing_shards")
    if isinstance(miss, list):
        # 有缺片即残缺；provider 若显式声明了 complete，则以显式值为准
        snap.setdefault("complete", len(miss) == 0)
    else:
        snap["missing_shards"] = []
        snap.setdefault("complete", True)
    if snap.get("actual_hours") is None:
        n = _count_actual_hours(snap)
        if n is not None:
            snap["actual_hours"] = n
    snap.setdefault("resolution_hours", None)
    snap.setdefault("precip_unit", None)
    snap.setdefault("precip_accum_window_hours", None)
    snap["payload_sha256"] = snapshot_sha256(snap)
    return snap

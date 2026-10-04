"""报告产物的来源留痕（可复现性 I5 与可审计性 I6 的前提）。

三行代码换来的东西，是此前整个项目缺的那块地基：`report.meta` 实测只有 9 个
字段，没有任何版本留痕——连"这份报告是哪个 commit 的代码算出来的"都不可考。
README 承诺的"固定种子可复现"因此无法兑现：同一份数据、同一份种子，换了
依赖的小版本就可能给出不同的浮点结果，而读者无从分辨。

**为什么这几个字段缺一不可**（第一性原理）：可复现 = 同数据 + 同代码 + 同依赖
+ 同种子。四者缺一，"重跑一遍得到同一个数"这句话就不成立。其中：
  * code_sha   —— 锁算法（口径会随代码演进，但每次演进必须可见）
  * lock_hash  —— 锁依赖（浮点与实现细节随库版本漂移）
  * seed       —— 锁随机性（bootstrap 的抽样）
  * python / 库版本 —— 前两者的可读展开，供人眼核对，不用于判定

**"同数据"这一半曾经无人认领**（2026-10-04 补）：预报快照是写一次就不再改的
（`save_forecast_snapshot` 幂等跳过 + Merkle 根可验），代码侧于是天然可复现；但
**观测会被后续轮次回改**——第三方源修正错报是常态，`obs/cma_data.py` 每次都回看
26 小时并对其中 6 小时强制重抓，`save_obs` 就地改值并把旧值挂进 `revisions`。
于是"同一份代码重跑一遍"得到的数字会随数据落库而变，且**变化点在数据里、不在
代码里**：Golden Master 对拍会把这种数据演化误报成口径漂移（2026-10-04 那次
红灯的真因：4 站各 1 个整点的气温被回改约 +1℃，1283 个展示字段随之位移）。

`obs_input_digest` 就是补上的这半个凭证：它对"冻结窗口内、评估真正消费的那两个
要素"取指纹，于是对拍失败时可以先分清是**输入动了**还是**口径动了**——这两件事
的处置方式完全相反（前者查数据回改，后者查代码）。

**为什么不能只记 SHA**：`git rev-parse HEAD` 在 CI 的 detached checkout 里是准的，
但本地跑报告时工作区可能是脏的（改了没提交），此时 SHA 会指向一份与磁盘内容
不同的代码。故同时记 `code_dirty`：脏工作区的报告必须自认不可复现，而不是假装
干净。这是与项目"绝不静默"同一条纪律。

取不到时一律回落 "unknown" 而不是抛异常：留痕是**披露设施**，它绝不能成为
报告构建的单点故障（与诊断层同一降级哲学）。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 评估（以及诊断层）真正消费的观测要素。输入指纹只覆盖它们：
# 气压/湿度/风等要素即使被回改也不可能改变报告，把它们计进指纹只会让守卫
# 在与结论无关的地方变红——而一个会因无关改动变红的守卫，迟早被人学会忽略。
OBS_EVAL_FIELDS = ("temp", "rain")

# 锁定文件（带哈希）的路径；由 scripts/install.sh / uv 生成，可缺省
LOCK_CANDIDATES = ("requirements.lock.txt", "uv.lock")

# 报告元数据的 schema 版本：meta 结构变更时递增，让下游（页面、复核脚本）能
# 知道自己读的是哪一代结构，而不是靠"某个字段在不在"去猜
META_SCHEMA_VERSION = 2


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    got = out.stdout.strip()
    return got or None


def code_sha() -> str:
    """当前代码版本的短 SHA。CI 优先用 GITHUB_SHA（checkout 的那个 commit）。"""
    env = os.environ.get("GITHUB_SHA")
    if env:
        return env[:12]
    for cmd in (["git", "rev-parse", "--short=12", "HEAD"],
                ["git", "describe", "--always", "--dirty"]):
        got = _run(cmd)
        if got:
            return got.replace("-dirty", "")
    return "unknown"


def code_dirty() -> bool:
    """工作区是否有未提交的改动（含未跟踪的源码文件）。"""
    out = _run(["git", "status", "--porcelain", "--", "src", "config"])
    if out is None:
        return False          # 取不到就按干净处理，绝不因此把报告判成不可复现
    return bool(out.strip())


def _lock_path() -> Path | None:
    for name in LOCK_CANDIDATES:
        p = PROJECT_ROOT / name
        if p.is_file():
            return p
    return None


def lock_hash() -> str | None:
    """锁定文件的内容哈希（SHA-256，前 16 位）。无锁文件时返回 None。

    哈希而不是文件名：文件名不表达内容。锁文件一旦被改，哈希立刻变，而这个
    变化必须出现在每一份由它产出的报告里。
    """
    p = _lock_path()
    if p is None:
        return None
    try:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def lock_file() -> str | None:
    p = _lock_path()
    return p.name if p is not None else None


def _dep_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for mod, key in (("numpy", "numpy"), ("pandas", "pandas"), ("scipy", "scipy"),
                     ("pint", "pint"), ("cyeva", "cyeva")):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", None)
            out[key] = str(v) if v else "unknown"
        except Exception:                       # noqa: BLE001  取不到就记 unavailable
            out[key] = "unavailable"
    return out


# ------------------------------------------------------------------ 输入指纹
def newest_obs_hour(station_ids: Iterable[str]) -> datetime | None:
    """数据集中最新的观测时刻（naive 北京时）；无观测时返回 None。

    存在的意义：观测回改的最大回看深度是 26 小时（`obs/cma_data.py` 的
    `DEFAULT_LOOKBACK_HOURS`），比它更新的观测随时可能被下一轮改写。冻结窗口的
    终点必须落在"最新观测 − 26h"之前，输入才不再移动。
    """
    from .storage import load_obs
    from .timeutil import parse_iso

    best = None
    for sid in station_ids:
        for t_iso in load_obs(sid):
            try:
                t = parse_iso(t_iso)
            except (TypeError, ValueError):
                continue
            if best is None or t > best:
                best = t
    return best


def obs_input_digest(station_ids: Iterable[str],
                     start: str | datetime, end: str | datetime) -> dict:
    """冻结窗口内观测输入的指纹：评估结论赖以成立的那半个"同数据"。

    口径三条，每条都对应一个真实的坑：
      1. 只取窗口内 `[start, end]` 的整点——窗口外的数据与本报告无关；
      2. 只取 `OBS_EVAL_FIELDS`（气温/降水）——评估与诊断层只消费这两个要素；
      3. **不含 `source` / `revisions`**——它们是留痕不是输入：主源故障恢复后
         同一小时会换来源通道，把它算进指纹会让每次换源都被误报成"数据变了"
         （`storage._same_obs` 正是因此把来源排除在比较之外）。

    返回可 JSON 序列化的 dict：`sha256` 是全窗口汇总指纹，`hours` 是参与的小时数，
    `per_station` 让"是哪一站动了"在第一眼就能看到。

    `start` / `end` 接受 ISO 字符串或 naive 北京时 datetime——冻结脚本手里是
    datetime，测试手里是 window.json 的字符串，两边都不该被迫转换。
    """
    from .storage import load_obs
    from .timeutil import iso, parse_iso

    def _as_dt(v):
        return v if hasattr(v, "year") else parse_iso(str(v))

    lo, hi = _as_dt(start), _as_dt(end)
    lines: list[str] = []
    per_station: dict[str, int] = {}
    for sid in sorted(station_ids):
        n = 0
        for t_iso, rec in load_obs(sid).items():
            try:
                t = parse_iso(t_iso)
            except (TypeError, ValueError):
                continue
            if t < lo or t > hi:
                continue
            rec = rec or {}
            # json.dumps 的浮点表示是"最短可往返"repr，跨机器可复算
            lines.append(json.dumps([sid, t_iso] + [rec.get(f) for f in OBS_EVAL_FIELDS],
                                     ensure_ascii=False, separators=(",", ":")))
            n += 1
        per_station[sid] = n
    lines.sort()
    return {
        "sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest(),
        "hours": len(lines),
        "per_station": per_station,
        "start": iso(lo),
        "end": iso(hi),
        "fields": list(OBS_EVAL_FIELDS),
    }


def provenance(*, seed: int | None = None, extra: dict | None = None) -> dict:
    """报告的来源留痕块。放进 report.meta.provenance 并由页面页脚渲染。"""
    block = {
        "schema_version": META_SCHEMA_VERSION,
        "code_sha": code_sha(),
        "code_dirty": code_dirty(),
        "lock_file": lock_file(),
        "lock_hash": lock_hash(),
        "bootstrap_seed": seed,
        "python": platform.python_version(),
        "deps": _dep_versions(),
        "generated_by": "github-actions" if os.environ.get("GITHUB_ACTIONS") else "local",
    }
    if extra:
        block.update(extra)
    return block

"""报告产物的来源留痕（可复现性 I5 与可审计性 I6 的前提）。

三行代码换来的东西，是此前整个项目缺的那块地基：`report.meta` 实测只有 9 个
字段，没有任何版本留痕——连"这份报告是哪个 commit 的代码算出来的"都不可考。
README 承诺的"固定种子可复现"因此无法兑现：同一份数据、同一份种子，换了
依赖的小版本就可能给出不同的浮点结果，而读者无从分辨。

**为什么这三个字段缺一不可**（第一性原理）：可复现 = 同数据 + 同代码 + 同依赖
+ 同种子。四者缺一，"重跑一遍得到同一个数"这句话就不成立。其中：
  * code_sha   —— 锁算法（口径会随代码演进，但每次演进必须可见）
  * lock_hash  —— 锁依赖（浮点与实现细节随库版本漂移）
  * seed       —— 锁随机性（bootstrap 的抽样）
  * python / 库版本 —— 前两者的可读展开，供人眼核对，不用于判定

**为什么不能只记 SHA**：`git rev-parse HEAD` 在 CI 的 detached checkout 里是准的，
但本地跑报告时工作区可能是脏的（改了没提交），此时 SHA 会指向一份与磁盘内容
不同的代码。故同时记 `code_dirty`：脏工作区的报告必须自认不可复现，而不是假装
干净。这是与项目"绝不静默"同一条纪律。

取不到时一律回落 "unknown" 而不是抛异常：留痕是**披露设施**，它绝不能成为
报告构建的单点故障（与诊断层同一降级哲学）。
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

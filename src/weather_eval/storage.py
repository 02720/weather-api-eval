"""数据存档层。

所有数据以 JSON 落盘到 <项目根>/data/ 下，并在 git 中跟踪：
  data/obs/{station_id}/{YYYY-MM}.json           观测（按时间键去重合并）
  data/forecasts/{station_id}/{model}/{issue}.json  起报快照（幂等）
  data/metrics/{period}/{file}.json              评估结果（可选缓存）

写入采用"临时文件 + 原子 rename"避免半文件；观测月文件与预报快照的读-改-写
合并窗口都用文件锁（POSIX flock，锁文件 *.lock 不入 git）保护——CI 有
concurrency 组兜底，本地多进程并发抓取时无锁会丢合并更新或重复写。
读取容错：损坏文件（git 冲突残留、外部改写等导致 JSONDecodeError）告警并跳过，
不拖垮整体评估/报告。

历史归档（P3-5）：起报快照按天增长（实测约 139 份/天），全量进 git 会让仓库
持续退化。`archive_old_snapshots` 把超过保留窗口的快照原样 gzip 成 .json.gz
（文本压缩比通常 8~12×），原 .json 删除；list_forecast_snapshots 对两种扩展名
一视同仁地读取，评估口径不受归档影响。
"""
from __future__ import annotations

import fcntl
import gzip
import json
import logging
import os
import tempfile
from contextlib import contextmanager
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any

from .timeutil import now_beijing, ym, ymd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _root() -> Path:
    return Path(os.environ.get("WEATHER_EVAL_DATA_ROOT", PROJECT_ROOT / "data"))


@contextmanager
def _exclusive_lock(path: Path):
    """对目标文件的读-改-写窗口加进程间排他锁（flock，随进程退出自动释放）。

    仅覆盖"读最新 → 合并 → 写回"这一临界区；写入本身仍走原子 rename，
    因此即使锁意外失效也不会产生半文件，最多退化为丢更新（与无锁一致）。
    """
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lf.fileno(), fcntl.LOCK_UN)


def _load_json(path: Path) -> Any | None:
    """读取 JSON（自动识别 .json.gz）；损坏文件告警并返回 None（调用方按缺失处理）。"""
    try:
        if path.suffix == ".gz":
            with gzip.open(path, "rt", encoding="utf-8") as f:
                return json.load(f)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, EOFError) as e:
        logger.warning("存档文件损坏，已跳过: %s (%s)", path, e)
        return None


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        os.chmod(tmp, 0o644)   # mkstemp 默认 0600，恢复常规读权限（部署/他人可读）
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ------------------------------------------------------------------ 观测
def save_obs(station_id: str, records: list[dict]) -> int:
    """把一批观测记录合并写入该站当月文件，按时间键去重；返回新增/更新的条数。

    读-改-写全程持文件锁：两个进程并发保存同站观测时不会互相覆盖丢更新。

    **观测回改写进 revisions 数组而不是只打日志**（P2-2）：第三方观测源会修正
    早期错报，旧实现静默覆盖旧值，"当时的实况"从此不可复原——而实况是评估里
    唯一的真值来源，它一旦不可追溯，所有历史结论都失去了可复核性。revisions
    保留每次改动的时间戳与新旧值，日志照打（可见性不变），但证据不再丢失。
    """
    if not records:
        return 0
    months: dict[str, dict] = {}
    for r in records:
        months.setdefault(ym(parse_dt(r["time"])), {})[r["time"]] = r
    updated = 0
    revised_at = now_beijing().strftime("%Y-%m-%dT%H:%M:%S")
    for month, rec_map in months.items():
        path = _root() / "obs" / station_id / f"{month}.json"
        with _exclusive_lock(path):
            existing: dict = _load_json(path) or {}
            for k, v in rec_map.items():
                old = existing.get(k)
                if k not in existing or v != existing[k]:
                    updated += 1
                if old is not None and old != v:
                    logger.warning(
                        "观测回改 %s %s：temp %s→%s，rain %s→%s",
                        station_id, k, old.get("temp"), v.get("temp"),
                        old.get("rain"), v.get("rain"))
                    v = _with_revision(v, old, revised_at)
                existing[k] = v
            _atomic_write_json(path, existing)
    return updated


# revisions 数组的长度上限：回改是罕见事件，但畸形/抖动源可能反复改写同一时刻；
# 截断保留最近若干次，避免单条观测把月文件撑大
MAX_OBS_REVISIONS = 10


def _with_revision(new: dict, old: dict, ts: str) -> dict:
    """把被改写的旧值挂进新记录的 revisions 数组（不覆盖新值本身）。"""
    rec = dict(new)
    history = list(old.get("revisions") or [])
    history.append({
        "at": ts,
        "prev": {k: old.get(k) for k in ("temp", "rain", "time")}
        if any(k in old for k in ("temp", "rain")) else {k: old.get(k) for k in old
                                                          if k != "revisions"},
    })
    rec["revisions"] = history[-MAX_OBS_REVISIONS:]
    return rec


def load_obs(station_id: str, month: str | None = None) -> dict[str, dict]:
    """读取某站全部观测（指定 month 时只取该月）。返回 time_iso -> record。"""
    result: dict[str, dict] = {}
    base = _root() / "obs" / station_id
    if not base.exists():
        return result
    for p in sorted(base.glob("*.json")):
        if month and p.stem != month:
            continue
        data = _load_json(p)
        if isinstance(data, dict):
            result.update(data)
    return result


# ------------------------------------------------------------------ 预报快照
def _issue_filename(issue_iso: str) -> str:
    return issue_iso.replace(":", "") + ".json"


def save_forecast_snapshot(station_id: str, model: str, snapshot: dict) -> bool:
    """写入起报快照；若同站同模型同起报时刻已存在则跳过（幂等）。返回是否新建。

    exists 检查与写入之间不是原子的——两个进程并发保存同一快照时可能都通过
    检查、都写一遍（TOCTOU）。与观测侧同一把 flock 锁住整个临界区：内容通常
    相同、原子 rename 也不会产生半文件，实际危害有限，但观测侧已建锁机制，
    快照侧补上是零成本的正确性。

    **落盘前统一盖契约元数据**（P1-1 / P0-6）：抓取时刻、内容哈希、起报锚点语义、
    完整性标记。这是全项目唯一的快照写入口，把"每份存档都要带 fetched_at"这件事
    从各 provider 的自觉变成写路径的强制——漏盖字段在物理上不可能发生。
    """
    from .snapshot_meta import stamp_snapshot
    from .timeutil import now_beijing
    issue_iso = snapshot["issue_iso"]
    path = _root() / "forecasts" / station_id / model / _issue_filename(issue_iso)
    with _exclusive_lock(path):
        if path.exists():
            return False
        now = now_beijing()
        stamp_snapshot(
            snapshot,
            fetched_bj=now.strftime("%Y-%m-%dT%H:%M:%S"),
            fetched_utc=now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        _atomic_write_json(path, snapshot)
    return True


def list_forecast_snapshots(station_id: str, model: str) -> list[dict]:
    base = _root() / "forecasts" / station_id / model
    if not base.exists():
        return []
    out = []
    for p in sorted(_snapshot_files(base)):
        snap = _load_json(p)
        if isinstance(snap, dict):
            out.append(snap)
    return out


def _snapshot_files(base: Path) -> list[Path]:
    """该模型目录下的全部快照文件（.json 与归档 .json.gz，按文件名合并排序）。

    同名并存时（归档后残留的旧 .json 不应出现，但防御一下）普通文件优先——
    归档流程保证两者内容一致，取哪个都不影响结果。
    """
    by_stem: dict[str, Path] = {}
    for p in list(base.glob("*.json")) + list(base.glob("*.json.gz")):
        key = p.name.removesuffix(".gz")
        if key not in by_stem or p.suffix == ".json":
            by_stem[key] = p
    return sorted(by_stem.values())


def archive_old_snapshots(older_than_days: int, apply: bool = False) -> list[Path]:
    """把 issue 早于保留窗口的快照 gzip 归档（.json → .json.gz，原文件删除）。

    返回（或将）被归档的源文件列表。默认 dry-run 只列出候选不落盘；apply=True
    才真正压缩。幂等：已归档（.json.gz）与已删除的文件自然不在候选里。
    评估读取侧对两种扩展名一视同仁，归档不改变任何指标口径。
    """
    from .timeutil import now_beijing, parse_iso

    cutoff = now_beijing() - timedelta(days=older_than_days)
    changed: list[Path] = []
    forecasts_root = _root() / "forecasts"
    if not forecasts_root.exists():
        return changed
    for path in _snapshot_files_iter(forecasts_root):
        if path.suffix != ".json":
            continue
        snap = _load_json(path)
        issue = snap.get("issue_iso") if isinstance(snap, dict) else None
        if not issue:
            continue
        try:
            if parse_iso(issue) >= cutoff:
                continue
        except ValueError:
            continue
        changed.append(path)
        if apply:
            _atomic_gzip_replace(path)
    return changed


def _atomic_gzip_replace(path: Path) -> None:
    """把 path 原地替换为 path.gz，全程走"临时文件 + os.replace"（P2-1）。

    旧实现直接写目标 .gz 再 unlink 源文件：中断（磁盘满、CI 被杀）会留下半截
    .gz，而它会被下一次 `git add data` 收进仓库——一份永久损坏的"档案"。
    先写临时文件再原子改名，则中断只会留下一个临时文件（不入 git、可清理），
    源 .json 只有在 .gz 完整落盘之后才被删除。
    """
    gz_path = path.with_name(path.name + ".gz")
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".gz.tmp")
    os.close(fd)
    try:
        with open(path, "rb") as src, gzip.open(tmp, "wb", compresslevel=9) as dst:
            dst.writelines(src)
        os.chmod(tmp, 0o644)
        os.replace(tmp, gz_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    path.unlink()


def _snapshot_files_iter(forecasts_root: Path):
    for model_dir in sorted(forecasts_root.glob("*/*")):
        if not model_dir.is_dir():
            continue
        yield from _snapshot_files(model_dir)


# ------------------------------------------------------------------ 指标（可选缓存，当前实现每次重算）
def save_metrics(period: str, name: str, obj: Any) -> None:
    _atomic_write_json(_root() / "metrics" / period / f"{name}.json", obj)


# ------------------------------------------------------------------ 完整性清单
def save_manifest(period: str, obj: Any) -> Path:
    """写 data/manifest/{period}.json（哈希链清单，§7.1）。

    进 git 的目的是**公示**：归档页与月度清单都带 Merkle 根，任何人克隆仓库后
    重算一遍就能验证存档未被事后改写。放在 data/ 下随数据一起提交。
    """
    path = _root() / "manifest" / f"{period}.json"
    _atomic_write_json(path, obj)
    return path


def load_manifest(period: str | None = None) -> dict:
    """读清单；period 缺省时取最新的（文件名排序）。"""
    base = _root() / "manifest"
    if not base.exists():
        return {}
    if period is not None:
        return _load_json(base / f"{period}.json") or {}
    files = sorted(base.glob("*.json"))
    if not files:
        return {}
    return _load_json(files[-1]) or {}


def parse_dt(s: str) -> Any:
    from .timeutil import parse_iso, parse_obs_time
    try:
        return parse_iso(s)
    except ValueError:
        return parse_obs_time(s)

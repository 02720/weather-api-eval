"""数据存档层。

所有数据以 JSON 落盘到 <项目根>/data/ 下，并在 git 中跟踪：
  data/obs/{station_id}/{YYYY-MM}.json           观测（按时间键去重合并）
  data/forecasts/{station_id}/{model}/{issue}.json  起报快照（幂等，当月的热层）
  data/forecasts/{station_id}/{model}/{YYYY-MM}.json.gz  已结束月份的月度 bundle（冷层）
  data/metrics/{period}/{file}.json              评估结果（月度摘要纳入 git，见 §体积治理）
  data/manifest/{period}.json                    哈希链清单

写入采用"临时文件 + 原子 rename"避免半文件；观测月文件与预报快照的读-改-写
合并窗口都用文件锁（POSIX flock，锁文件 *.lock 不入 git）保护——CI 有
concurrency 组兜底，本地多进程并发抓取时无锁会丢合并更新或重复写。
读取容错：损坏文件（git 冲突残留、外部改写等导致 JSONDecodeError）告警并跳过，
不拖垮整体评估/报告。

--------------------------------------------------------------- 体积治理（2026-09）
**问题的第一性原理**：本项目的预报快照是**只增不减的证据流**（实测 ~170 份/天、
单份中位 18.9 KB、约 2.8 MB/天），而 git 会永久保存每一个版本。于是仓库体积
单调增长，与"能否持续自动化运行"直接冲突。两个独立的增长机制必须分开治理：

  (1) **新数据的字节数**——由布局决定。实测单份快照的 59.8% 是**同一条时间轴**
      （`hourly_time` 在 27 个模型间几乎完全相同），而逐文件 gzip 完全吃不到这份
      跨快照冗余。
  (2) **git 永久保存每个版本**——工作区压缩只能减缓、不能封顶。真正的封顶只有两条
      路：让数据离开仓库（保留期），或重写历史（人工决策，见 README）。

本模块负责 (1) 与保留期的机械部分：把**已结束的自然月**的逐份快照合并成一份
`{YYYY-MM}.json.gz`（容器格式见 `BUNDLE_MARK`）。实测收益（单站 27 模型 17.0 MB
原始 JSON）：

    逐文件 gzip      1.95 MB（8.7×）     ← 旧 `archive` 命令的口径
    月度 bundle gzip 0.70 MB（24.2×）    ← 现在：再省 66%
    文件数           4119 份 → 27 份/月/站

**冻结不变量**：bundle 一经写出即永不重写（与 `reports/monthly/` 的冻结档案同一
哲学）。这不是为了省事，而是为了让 git 侧可预测——一份永不改动的 .gz 在 git 里
只存一次，不产生任何后续 delta；而"每月重新打包一次"会让每个版本都是全新 blob，
把省下的字节又还给 git。

**安全不变量**：先写临时文件 → **解压回读并逐份比对** → 原子改名 → **最后才删除
源文件**。中途任何一步失败，源 `.json` 都原封不动（沿用 `_atomic_gzip_replace` 的
纪律：绝不允许出现"半截归档 + 源文件已删"）。

**口径不变**：读取侧（`list_forecast_snapshots`）对 bundle 与散装 `.json` 一视同仁，
并把两层展开成同一批快照。归档因此**不改变任何指标口径，也不改变 Merkle 根**——
后者由回归测试钉死（`test_bundle_preserves_merkle_root`）。
"""
from __future__ import annotations

import fcntl
import gzip
import json
import logging
import os
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .timeutil import now_beijing, ym, ymd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 月度 bundle 容器格式版本。容器是自描述的（带 BUNDLE_MARK），读取侧据此区分
# "一份快照"与"一包快照"——两种 .gz 在同一目录里共存也不会被搞混。
BUNDLE_MARK = "__bundle__"
BUNDLE_SCHEMA_VERSION = 1
BUNDLE_SUFFIX = ".json.gz"

# 观测记录里"不参与变化比较"的字段：source 记录的是这条实况由哪条链路抓回来
# （抓取通道的元数据，不是可测量的实况值），revisions 是历史数组本身。
_PROVENANCE_KEYS = frozenset({"source", "revisions"})



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
            return _read_gz(path)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError, OSError, EOFError) as e:
        logger.warning("存档文件损坏，已跳过: %s (%s)", path, e)
        return None


def _read_gz(path: Path) -> Any:
    """读 gzip 压缩的 JSON（失败时抛原始异常，由调用方决定是告警还是致命）。"""
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


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

    **"变化"的定义排除来源标记**（2026-09，随观测多源编排引入）：抓取通道
    （`source`）不是可测量的实况值。主源故障恢复后，同一小时会从备用源换成主源，
    若把来源计入比较，每一小时都会被记成一次"回改"，revisions 立刻被噪声淹没，
    真正的错报修正反而看不见了。故比较时只看要素本身（见 `_comparable`）。
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
                changed = old is None or not _same_obs(old, v)
                if changed:
                    updated += 1
                if old is not None and changed:
                    logger.warning(
                        "观测回改 %s %s：temp %s→%s，rain %s→%s",
                        station_id, k, old.get("temp"), v.get("temp"),
                        old.get("rain"), v.get("rain"))
                    v = _with_revision(v, old, revised_at)
                existing[k] = v
            _atomic_write_json(path, existing)
    return updated


def _comparable(rec: dict) -> dict:
    """观测记录的"可测量部分"：剔除来源标记与历史数组，供变化比较使用。"""
    return {k: v for k, v in rec.items() if k not in _PROVENANCE_KEYS}


def _same_obs(a: dict, b: dict) -> bool:
    """两条观测在"可测量部分"上是否等价（来源通道不同不算变化）。"""
    return _comparable(a) == _comparable(b)


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


def _eval_daily_max_offset() -> int:
    """日产品评测范围（eval.daily_max_offset_days）的读取。

    政策的单一出处是评估配置（stations.yaml 的 eval 段）；惰性导入避免
    storage→config 的模块级耦合，配置不可得时回退内置默认值——截断是
    体积优化与政策强制，绝不能因配置缺失让写路径瘫痪。
    """
    from .config import DEFAULT_EVAL, load_config
    try:
        return int(load_config().eval.get("daily_max_offset_days",
                                          DEFAULT_EVAL["daily_max_offset_days"]))
    except Exception:  # noqa: BLE001  配置文件缺失/损坏时回退内置默认
        return int(DEFAULT_EVAL["daily_max_offset_days"])


def save_forecast_snapshot(station_id: str, model: str, snapshot: dict, *,
                           daily_max_offset_days: int | None = None) -> bool:
    """写入起报快照；若同站同模型同起报时刻已存在则跳过（幂等）。返回是否新建。

    exists 检查与写入之间不是原子的——两个进程并发保存同一快照时可能都通过
    检查、都写一遍（TOCTOU）。与观测侧同一把 flock 锁住整个临界区：内容通常
    相同、原子 rename 也不会产生半文件，实际危害有限，但观测侧已建锁机制，
    快照侧补上是零成本的正确性。

    **落盘前统一盖契约元数据**（P1-1 / P0-6）：抓取时刻、内容哈希、起报锚点语义、
    完整性标记。这是全项目唯一的快照写入口，把"每份存档都要带 fetched_at"这件事
    从各 provider 的自觉变成写路径的强制——漏盖字段在物理上不可能发生。

    **落盘前截除评测范围外的日产品**（daily_max_offset_days，默认 16）：逐日预报
    块只服务按天评估，超出评测范围的日产品（实测某源 90 天）永远不会被评测，
    却要随每份快照永久占据仓库体积——截断必须在 stamp 之前完成，让内容哈希
    覆盖截断后的实际存档。评测侧的日偏移过滤（collect）是第二道闸，两者口径
    同源（同一配置项），详见 snapshot_meta.truncate_daily_block。
    """
    from .snapshot_meta import stamp_snapshot, truncate_daily_block
    from .timeutil import now_beijing
    issue_iso = snapshot["issue_iso"]
    path = _root() / "forecasts" / station_id / model / _issue_filename(issue_iso)
    with _exclusive_lock(path):
        if path.exists():
            return False
        truncate_daily_block(
            snapshot,
            daily_max_offset_days if daily_max_offset_days is not None
            else _eval_daily_max_offset())
        now = now_beijing()
        stamp_snapshot(
            snapshot,
            fetched_bj=now.strftime("%Y-%m-%dT%H:%M:%S"),
            fetched_utc=now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        _atomic_write_json(path, snapshot)
    return True


def list_forecast_snapshots(station_id: str, model: str) -> list[dict]:
    """该站该模型的全部快照（热层散装 .json 与冷层月度 bundle 透明合并）。

    两层展开成同一批快照，**按 `issue_iso` 去重**：同一份起报在两处出现时以
    **散装 .json 为准**——bundle 冻结之后才补进来的快照不可能在 bundle 里，
    它就是更新的真相（正常情况下两者不会并存；此规则是防御，不是常态）。
    """
    base = _root() / "forecasts" / station_id / model
    if not base.exists():
        return []
    out: dict[Any, dict] = {}
    anon = 0
    for p in sorted(_snapshot_files(base)):
        # _snapshot_files 已按文件名排序，且散装 .json（"2026-08-26T2100.json"）的
        # 文件名恒排在 bundle（"2026-08.json.gz"）之前，故"先到者优先 + 散装覆盖"
        # 的组合即为"散装优先"。
        for snap in _expand_snapshots(_load_json(p)):
            issue = snap.get("issue_iso")
            if issue is None:
                anon += 1
                out[f"__anon__{anon}"] = snap   # 无起报时刻的存档仍须保留，不得丢弃
            elif issue not in out or p.suffix == ".json":
                out[issue] = snap
    return [out[k] for k in out]


def _expand_snapshots(loaded: Any) -> list[dict]:
    """把一次文件读取的结果展开成快照列表（兼容"单份快照"与"月度 bundle"两种布局）。"""
    if not isinstance(loaded, dict):
        return []
    if loaded.get(BUNDLE_MARK):
        snaps = loaded.get("snapshots")
        if not isinstance(snaps, dict):
            logger.warning("bundle 容器缺少 snapshots 映射，已跳过（%s）", loaded.get("period"))
            return []
        return [v for _, v in sorted(snaps.items()) if isinstance(v, dict)]
    return [loaded]


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


# ------------------------------------------------------- 月度 bundle（体积治理）
def _looks_like_month(s: str) -> bool:
    """'2026-08' 形态的月份串（用于从文件名安全地区分月份与起报时刻）。"""
    return (len(s) == 7 and s[4] == "-" and s[:4].isdigit() and s[5:].isdigit())


def shift_month(period: str, delta: int) -> str:
    """月份串位移（'2026-09' −13 → '2025-08'）。"""
    year, month = int(period[:4]), int(period[5:7])
    total = year * 12 + (month - 1) + delta
    return f"{total // 12:04d}-{total % 12 + 1:02d}"


def compact_snapshots(*, grace_days: int = 2, retain_months: int = 13,
                      apply: bool = False, force: bool = False,
                      now: datetime | None = None) -> dict:
    """把快照从"逐份 .json（热层）"降级为"月度 bundle（冷层）"，并把超期冷层出仓。

    这是让仓库能**持续**自动化运行的那一步：不做它，`data/` 每天新增约 170 个文件、
    2.8 MB，一年后 6 万文件、约 1 GB。

    分层规则（用自然月而不是"最近 N 天"，因此**幂等、可重放、结果与运行时刻无关**）：

        当月（及宽限期内）   逐份 .json，追加写入、随时可读 —— 支撑主报告/月度冻结/verify
        已结束的自然月       一份 {YYYY-MM}.json.gz（冻结，永不重写）
        超过 retain_months   出仓（删除）—— 其结论已由冻结的月度报告与该月指标摘要固化

    **先固化、后删除**是这里的核心纪律：删除原始快照之前，调用方（CLI 的 `monthly`）
    必须已经写下该月的指标摘要。"体积变小"永远不能以"结论不可复核"为代价。

    默认 dry-run（只报告不落盘）。返回报告 dict：
      bundles_created / bundles_skipped / files_removed / bytes_before / bytes_after /
      expired / expired_bytes / errors
    """
    now = now or now_beijing()
    # 满 grace_days 天的已结束月份才冻结：月初那几轮运行仍可能补上月末最后几小时的
    # 预报与观测，抢在那之前冻结会把它们挡在归档之外。
    freezable_before = ym(now - timedelta(days=grace_days))
    expire_before = shift_month(ym(now), -abs(retain_months))

    report: dict[str, Any] = {
        "apply": bool(apply),
        "force": bool(force),
        "grace_days": grace_days,
        "retain_months": retain_months,
        "freezable_before": freezable_before,
        "expire_before": expire_before,
        "bundles_created": [],
        "bundles_skipped": [],
        "expired": [],
        "expiry_blocked": [],      # 因结论摘要缺失而拒绝出仓的 bundle（先固化后删除）
        "files_removed": 0,        # apply 时实际删除的源文件数
        "files_pending": 0,        # dry-run 时"若执行将会删除"的源文件数
        "bytes_before": 0,
        "bytes_after": 0,
        "expired_bytes": 0,
        "errors": [],
    }

    forecasts_root = _root() / "forecasts"
    if not forecasts_root.exists():
        return report

    for station_dir in sorted(p for p in forecasts_root.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in station_dir.iterdir() if p.is_dir()):
            # ---- (1) 冻结已结束月份 ----
            by_month: dict[str, list[Path]] = defaultdict(list)
            for p in sorted(model_dir.glob("*.json")):
                month = p.stem[:7]
                if _looks_like_month(month):
                    by_month[month].append(p)
            for month in sorted(by_month):
                if month >= freezable_before:
                    continue                      # 当月 / 宽限期内：保持逐份
                paths = by_month[month]
                target = model_dir / f"{month}{BUNDLE_SUFFIX}"
                if target.exists():
                    # 冻结档案永不重写。残留的散装 .json 交由读取侧的"散装优先"
                    # 规则处理，绝不把它们塞进已冻结的包（那会让包每月都变一次，
                    # 把省下来的字节原样还给 git）。
                    report["bundles_skipped"].append(str(target))
                    if apply and paths:
                        logger.warning(
                            "%s 已冻结但仍有 %d 份散装快照：按读取侧规则以散装为准，"
                            "冻结包保持不动", target.parent.name, len(paths))
                    continue
                try:
                    res = _write_month_bundle(model_dir, month, paths,
                                              station_id=station_dir.name,
                                              model=model_dir.name, apply=apply)
                except Exception as e:  # noqa: BLE001  单包失败不拖垮其余模型
                    logger.error("冻结 %s/%s %s 失败: %s",
                                 station_dir.name, model_dir.name, month, e)
                    report["errors"].append(f"{station_dir.name}/{model_dir.name}/{month}: {e}")
                    continue
                report["bundles_created"].append(res["target"])
                report["bytes_before"] += res["bytes_before"]
                report["bytes_after"] += res["bytes_after"]
                report["files_removed"] += res["files_removed"]
                report["files_pending"] += res["files_pending"]

            # ---- (2) 超期冷层出仓 ----
            for gz in sorted(model_dir.glob(f"*{BUNDLE_SUFFIX}")):
                month = gz.name[: -len(BUNDLE_SUFFIX)]
                if not _looks_like_month(month) or month >= expire_before:
                    continue
                # 先固化、后删除：该月的结论摘要不在，就不许删（见 has_period_summary）
                if not force and not has_period_summary(month):
                    report["expiry_blocked"].append(str(gz))
                    logger.error(
                        "拒绝出仓 %s：该月结论摘要缺失（data/metrics/%s/summary.json）。"
                        "原始快照一旦删除，长期结论将无法复核——先跑 monthly 固化该月，"
                        "或用 --force 明确接受这个代价", gz.name, month)
                    continue
                size = gz.stat().st_size
                report["expired"].append(str(gz))
                report["expired_bytes"] += size
                if apply:
                    try:
                        gz.unlink()
                    except OSError as e:
                        report["errors"].append(f"{gz}: {e}")
                        logger.error("出仓失败 %s: %s", gz, e)
    return report


def _write_month_bundle(model_dir: Path, month: str, paths: list[Path], *,
                        station_id: str, model: str, apply: bool = False) -> dict:
    """把某月散装快照写成一份 bundle。apply=False 时只测算收益，不落盘。

    安全顺序（任何一步失败，源 .json 都原封不动）：
        读取全部源快照 → 序列化 → 写临时文件 → **解压回读并逐份比对** →
        原子改名 → 删除源文件
    """
    snaps: dict[str, dict] = {}
    for p in paths:
        snap = _load_json(p)
        if not isinstance(snap, dict):
            raise RuntimeError(f"源快照不可读，拒绝冻结（{p.name}）")
        snaps[snap.get("issue_iso") or p.stem] = snap
    bytes_before = sum(p.stat().st_size for p in paths)
    container = {
        BUNDLE_MARK: BUNDLE_SCHEMA_VERSION,
        "station_id": station_id,
        "model": model,
        "period": month,
        "n_snapshots": len(snaps),
        "snapshots": snaps,
    }
    blob = json.dumps(container, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")
    target = model_dir / f"{month}{BUNDLE_SUFFIX}"

    if not apply:
        est = len(gzip.compress(blob, compresslevel=9))
        return {"target": str(target), "files_removed": 0, "files_pending": len(paths),
                "bytes_before": bytes_before, "bytes_after": est}

    with _exclusive_lock(target):
        if target.exists():          # 并发下另一个进程已冻结：放弃并保留源文件
            return {"target": str(target), "files_removed": 0, "files_pending": 0,
                    "bytes_before": bytes_before, "bytes_after": target.stat().st_size}
        fd, tmp = tempfile.mkstemp(dir=str(model_dir), suffix=".gz.tmp")
        os.close(fd)
        try:
            with gzip.open(tmp, "wb", compresslevel=9) as f:
                f.write(blob)
            os.chmod(tmp, 0o644)
            # 回读校验：解压后必须完整还原同一批快照，才允许进入下一步。
            # 这是"先验证、后销毁"的落点——半截归档 + 源文件已删是唯一不可恢复的失败。
            back = _read_gz(Path(tmp))
            if not isinstance(back, dict) or back.get("snapshots") != snaps:
                raise RuntimeError(f"bundle 回读校验未通过（{target.name}），源文件保持不动")
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # bundle 已完整落盘且校验通过：此时才删除源文件
    removed = 0
    for p in paths:
        try:
            p.unlink()
            removed += 1
        except OSError as e:
            logger.warning("源快照删除失败（bundle 已就绪，可安全重试）%s: %s", p, e)
    ratio = (bytes_before / target.stat().st_size) if target.stat().st_size else 0.0
    logger.info("冻结 %s/%s %s：%d 份快照 %.2f MB → %.2f MB（%.1f×）",
                station_id, model, month, len(snaps),
                bytes_before / 1e6, target.stat().st_size / 1e6, ratio)
    return {"target": str(target), "files_removed": removed, "files_pending": len(paths),
            "bytes_before": bytes_before, "bytes_after": target.stat().st_size}


def data_footprint() -> dict:
    """data/ 的体量画像（按层拆分文件数与字节数）。供体积看门狗与 compact 报告使用。

    分层口径与 `compact_snapshots` 的冷热分层一一对应，"哪一层在涨"因此可分辨：
    冷层是不可变的月度 bundle（一份只存一次），热层才是每天新增的部分。
    """
    root = _root()
    out = {
        "total_files": 0, "total_bytes": 0,
        "forecast_hot_files": 0, "forecast_hot_bytes": 0,   # 散装起报快照（当月）
        "forecast_cold_files": 0, "forecast_cold_bytes": 0,  # 月度 bundle（冻结）
        "obs_files": 0, "obs_bytes": 0,
        "manifest_files": 0, "manifest_bytes": 0,
        "other_files": 0, "other_bytes": 0,
    }
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        # 锁文件（*.lock）是 flock 机制的副产物、不入 git，且数量随 bundle 数增长
        # （每包一个）。计入的话会把"热层是否按月冻结"这个判据稀释掉。
        if p.suffix == ".lock":
            continue
        size = p.stat().st_size
        out["total_files"] += 1
        out["total_bytes"] += size
        rel = p.relative_to(root)
        top = rel.parts[0] if rel.parts else ""
        if top == "forecasts":
            if p.name.endswith(BUNDLE_SUFFIX):
                key = "forecast_cold"
            elif p.suffix == ".json":
                key = "forecast_hot"
            else:
                key = "other"
        elif top == "obs":
            key = "obs"
        elif top == "manifest":
            key = "manifest"
        else:
            key = "other"
        out[f"{key}_files"] += 1
        out[f"{key}_bytes"] += size
    return out



# ------------------------------------------------------------------ 指标（可选缓存，当前实现每次重算）
def save_metrics(period: str, name: str, obj: Any) -> None:
    _atomic_write_json(_root() / "metrics" / period / f"{name}.json", obj)


def period_summary_path(period: str) -> Path:
    """某月结论摘要的落点（data/metrics/{period}/summary.json）。"""
    return _root() / "metrics" / period / "summary.json"


def has_period_summary(period: str) -> bool:
    """该月是否已有固化的结论摘要。

    这是"先固化、后删除"的机械化落点：`compact` 出仓一个月的 bundle 之前会查这里。
    原始快照被删掉之后，长期趋势只能靠这份摘要复核；摘要不在就不许删——宁可让
    体积治理卡住并告警，也不能把"体积变小"换成"结论不可复核"。
    """
    return period_summary_path(period).is_file()


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

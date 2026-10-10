"""指标计算的进程级并行（TASK-02）：同函数、同输入、同输出，只换执行顺序。

**它为什么是安全的**：`precip_metrics` / `temp_metrics` 的每一次调用对应一个
(模型, 天桶, 轨道) 格子，输入输出确定、无共享状态、调用之间零耦合。把它们分发到
多个进程做的是"同一件事在不同的核上同时做"，**不是**换算法、换求和顺序、换
收敛判据。因此：
  * 结果按**提交顺序**回收（worker=1 与 worker=4 的输出必须逐位相同，
    由 `test_metrics_determinism_under_parallel` 锁定）；
  * 任何一个 worker 崩了就整批退回串行，绝不返回"部分并行 + 部分串行"的
    混合结果（那会让"这一轮用了哪条路径"变成不可考的事实）；
  * 小批量走串行——测试与单源对照的数据量根本不值得付进程启动的固定开销，
    而且串行路径保证了小数据场景的行为与改造前**完全一致**。

**为什么用进程而不是线程**：cyeva 内部走 pandas + pint，是纯 Python/GIL 内的
CPU 密集工作；线程池在这里拿不到任何加速。进程的代价是 pickle，故传输的是
numpy 数组（列式）而不是 dict 列表。

**为什么 worker 数默认取 min(cpu, 4)**：CI runner 通常是 2 核，取 cpu_count 就
是 2；本地 8 核机器上开 8 个进程会让 8 份 cyeva + pint 各初始化一遍，收益被
启动开销吃掉，且峰值内存翻倍（每进程一份样本副本）。4 是实测的收益/代价拐点。
可用环境变量 `WEATHER_EVAL_METRIC_WORKERS=1` 强制串行（对拍与排障用）。
"""
from __future__ import annotations

import logging
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

WORKERS_ENV = "WEATHER_EVAL_METRIC_WORKERS"
# 进程的启动方式：**显式锁定 fork**。本模块的 RSS 闸门与"fork 的写时复制成本"
# 整条论证都以 fork 为前提；Python 3.14 起默认 start method 在 Linux 上改为
# forkserver，若跟着版本漂移，闸门的含义与实测结论都会失效而不报错。
# 不支持 fork 的平台（Windows/macOS spawn）由 get_context 自行回退。
_MP_CONTEXT = "fork"
# 低于这个批量就串行：进程池的固定开销（fork + cyeva/pint 初始化）不划算
MIN_PARALLEL_JOBS = 32
MAX_DEFAULT_WORKERS = 4
# fork 的内存闸门（MB）：父进程 RSS 超过它就不开并行。
#
# 这条闸门是被实测逼出来的，不是拍脑袋：并行化的收益随"剩余 cyeva 工作量"增长，
# 而它的**成本随父进程内存增长**——`build_report` 此刻已在内存里持有约 1.9 GB 的
# 配对数据，fork 出 4 个 worker 后写时复制会把这份内存反复触碰。本机实测：
#     并行（4 worker）  114.9 s    ← 诊断层 35.2 s、collect 10.6 s（内存压力外溢）
#     串行               92.5 s    ← 诊断层 29.6 s、collect  6.8 s
# 也就是说：分级指标向量化之后，cyeva 只剩约 17 s 可并行，而 fork 的代价约 22 s，
# 净亏。等到事实层不再把全量配对驻留内存（增量 IO 落地）或数据量涨到 13 个月
# （cyeva 工作量 ∝ 样本量，届时约 100 s 可并行）时，这条闸门会自然放行。
#
# 闸门量的是**当前** RSS 而不是峰值：写时复制的成本取决于此刻驻留了多少页，
# 用峰值会让"曾经高过一次"变成永久禁令（见 _parent_rss_mb 的说明）。
FORK_RSS_LIMIT_MB = 1200.0
# `ex.map` 的等待上界（秒）：超时即整批退回串行。缺省的无限等待会让"worker
# 挂死 → 退回串行"这条兜底永远触发不了，整轮构建挂起而不是降级跑完。
MAP_TIMEOUT_S = 1200.0

def _parent_rss_mb() -> float:
    """**当前** RSS（MB）——fork 的写时复制成本只取决于此刻驻留了多少页。

    ⚠️ 此前用 `resource.getrusage().ru_maxrss`，那是**历史峰值**而非当前值：
    峰值一旦在早期被抬高（例如一次性读完 13 个月观测），此后即使 RSS 回落、
    fork 其实很便宜，`worker_count()` 也会永久判定"父进程太大"并退回串行——
    并行化在长生命周期进程（`all` 一次跑完全流程）里等于被一次性永久关闭。

    取值顺序：Linux 的 /proc/self/statm 最准且零依赖；取不到再退回 ru_maxrss
    （此时按"未知"处理更好，但保留旧行为以免在无 /proc 的环境里突然全开并行）。
    """
    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as f:
            # 第二字段 = 驻留页数；页长用 os.sysconf 取，不硬编码 4096
            pages = int(f.read().split()[1])
        return pages * (os.sysconf("SC_PAGE_SIZE") / (1024.0 * 1024.0))
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux 上报 KB；macOS 上报字节（数值量级差 1000 倍，按阈值判别）
        return rss / 1024.0 if rss > 100_000 else rss / (1024.0 * 1024.0)
    except Exception:                            # noqa: BLE001  取不到就按不限制处理
        return 0.0


def worker_count() -> int:
    """并行 worker 数；1 = 串行。

    显式环境变量优先（对拍与排障用）。否则按"fork 是否划算"自适应：
    父进程太大时，fork 的写时复制成本会超过并行省下的时间。
    """
    raw = os.environ.get(WORKERS_ENV)
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            return 1
    if _parent_rss_mb() > FORK_RSS_LIMIT_MB:
        return 1
    return max(1, min(os.cpu_count() or 1, MAX_DEFAULT_WORKERS))


def _init_worker() -> None:
    """worker 初始化：把 cyeva 与 pint 的导入成本摊到进程生命周期上。

    cyeva 首次 import 会连带初始化 pint 的单位注册表（毫秒级但每进程一次），
    放在 initializer 里而不是每次调用里，避免"每次调用付一次初始化"。
    """
    try:                                        # pragma: no cover - 环境相关
        import cyeva  # noqa: F401
    except Exception:                           # noqa: BLE001
        pass


def _run_one(job):
    """worker 侧的单次调用：函数名 + 实参，结果原样回传。"""
    name, args, kwargs = job
    from . import evaluate as _mod
    fn = getattr(_mod, name)
    return fn(*args, **kwargs)


def map_metric_calls(fn_name: str, jobs: list[tuple[tuple, dict]]) -> list:
    """按提交顺序返回每次调用的结果。

    jobs: [(args_tuple, kwargs_dict), ...]。返回的列表与 jobs **逐位同序**——
    这是并行的确定性契约：调用之间无依赖，但汇总顺序必须与串行一致，否则
    下游的浮点求和顺序会随调度漂移（§11 负面清单第 15 条）。
    """
    if not jobs:
        return []
    n = worker_count()
    if n <= 1 or len(jobs) < MIN_PARALLEL_JOBS:
        from . import evaluate as _mod
        fn = getattr(_mod, fn_name)
        return [fn(*a, **k) for a, k in jobs]

    payload = [(fn_name, a, k) for a, k in jobs]
    chunksize = max(1, len(payload) // (n * 4))
    try:
        # 显式锁定 start method，不让 Python 版本替我们决定：3.12 默认 fork（父进程
        # 2 GB + requests.Session 会被整体映射），3.14 起默认改成 forkserver，语义与
        # 开销都变了。本模块的整条论证（包括 RSS 闸门）都建立在 fork 之上。
        ctx = multiprocessing.get_context(_MP_CONTEXT)
        with ProcessPoolExecutor(max_workers=n, mp_context=ctx,
                                 initializer=_init_worker) as ex:
            # timeout 是"整批退回串行"这条兜底的**触发条件**：`ex.map` 缺省无限
            # except 永远等不到，docstring 承诺的退回就成了空话——整轮报告构建
            # 会挂在原地而不是降级跑完。上界按串行耗时的宽松倍数给（串行约 17 s，
            # 这里给 20 分钟），宁可偶尔白跑一遍串行，也不无限期挂起。
            return list(ex.map(_run_one, payload, chunksize=chunksize,
                               timeout=MAP_TIMEOUT_S))
    except Exception as exc:                    # noqa: BLE001  任何失败都整批退回串行
        logger.warning("指标并行失败（%s: %s），整批退回串行路径",
                       type(exc).__name__, exc)
        from . import evaluate as _mod
        fn = getattr(_mod, fn_name)
        return [fn(*a, **k) for a, k in jobs]

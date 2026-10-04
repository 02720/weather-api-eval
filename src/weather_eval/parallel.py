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
from concurrent.futures import ProcessPoolExecutor

logger = logging.getLogger(__name__)

WORKERS_ENV = "WEATHER_EVAL_METRIC_WORKERS"
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
FORK_RSS_LIMIT_MB = 1200.0


def _parent_rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
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
        with ProcessPoolExecutor(max_workers=n, initializer=_init_worker) as ex:
            return list(ex.map(_run_one, payload, chunksize=chunksize))
    except Exception as exc:                    # noqa: BLE001  任何失败都整批退回串行
        logger.warning("指标并行失败（%s: %s），整批退回串行路径",
                       type(exc).__name__, exc)
        from . import evaluate as _mod
        fn = getattr(_mod, fn_name)
        return [fn(*a, **k) for a, k in jobs]

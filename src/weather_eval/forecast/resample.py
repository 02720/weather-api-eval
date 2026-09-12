"""子日尺度预报要素的确定性重采样助手（风乌 / EW4ALL 共用）。

当数据源的时间分辨率粗于逐小时（风乌 3 小时采样、EW4ALL 的 3/6 小时累计……）时，
只有两类展开方式是被允许的，本模块是它们的唯一实现：

- ``interpolate_hourly``：**连续量**（气温）→ 相邻采样点之间线性插值到逐小时；
- ``spread_accumulation``：**累积量**（降水）→ 非重叠窗口平铺，把每个窗口的累计值
  均摊到窗口覆盖的每一个小时。

第一性原理（与 README「评估方法」一致，改动前先读）：

1. 插值/均摊**只允许**用在"源本身就是子日尺度产品（逐 3~6 小时）"的场合。源若只有
   **逐日**产品，绝不允许用它反推逐小时序列——那是凭空造出日内变化（base.py 契约）。
2. 累积量必须走**平铺子集**而不是"每个滑窗各自均摊"。当采样间隔 < 窗口长度时
   （如 6h 窗口每 3h 采样一次），相邻窗口相互重叠，逐窗均摊会把同一场雨重复计多次；
   取"与首个采样同相位、每窗口长度取一个端点"的平铺子集，这些窗口两两无缝拼接、
   恰好覆盖时间轴，每个小时恰属唯一窗口，于是任意完整跨度上的求和等于原始累计总量
   （日降水 BIAS 不失真）。
3. 均摊的代价是短时强降水被摊薄（对 0.1mm 晴雨阈值偏保守），属已知局限，
   各 provider 在自己的 docstring 里留档。
4. 未覆盖的小时一律给 ``None``（缺测），**绝不**外推、填 0 或沿用邻值。
"""
from __future__ import annotations

from datetime import datetime, timedelta


def interpolate_hourly(samples: list[tuple[datetime, float | None]]) \
        -> list[tuple[datetime, float | None]]:
    """把按时间升序的采样点线性插值到逐小时（首末采样之间；不外推）。

    端点缺测的区段（任一端为 None）不插值，产出 None；既有整点采样原样保留。
    """
    if len(samples) < 2:
        return list(samples)
    out: list[tuple[datetime, float | None]] = []
    for (t0, v0), (t1, v1) in zip(samples, samples[1:]):
        out.append((t0, v0))
        if v0 is None or v1 is None:
            continue
        span = int((t1 - t0).total_seconds() // 3600)
        for k in range(1, span):
            frac = k / span
            out.append((t0 + timedelta(hours=k), v0 + (v1 - v0) * frac))
    out.append(samples[-1])
    # 按小时取整并去重（采样可能落在非整点，先地板到整点）
    seen: dict[datetime, float | None] = {}
    for t, v in out:
        th = t.replace(minute=0, second=0, microsecond=0)
        seen.setdefault(th, v)
    return sorted(seen.items())


def tile_endpoints(samples: list[tuple[datetime, float | None]],
                   window_hours: int) -> list[datetime]:
    """平铺窗口端点：以首个采样为相位、每 ``window_hours`` 取一个真实存在的采样点。

    相位锚在**实际采样时刻**而非墙钟整点：端点的语义就是"该累计窗口的结束时刻"，
    由数据源定义（风乌的 tp6h 端点落在 +1/+7/+13h 这种相位上，EW4ALL 落在整 3/6 小时
    上），锚墙钟反而会把前者的窗口切错。前提是采样点均匀落在同一网格上——该前提
    由调用方用"端点间距是否恒为 window_hours"做哨兵校验（见 ew4all 的告警）。
    """
    if window_hours <= 0:
        raise ValueError("window_hours 必须为正整数小时")
    by_time: dict[datetime, float | None] = {}
    for t, v in samples:
        by_time.setdefault(t, v)  # 重复时刻保留首见（与温度插值去重口径一致）
    times = sorted(by_time)
    if not times:
        return []
    span = timedelta(hours=window_hours)
    t0 = times[0]
    return [t for t in times if (t - t0) % span == timedelta(0)]


def spread_accumulation(samples: list[tuple[datetime, float | None]],
                        hours: list[datetime],
                        window_hours: int) -> list[float | None]:
    """时长 ``window_hours`` 的**后向累计**（窗口 (t-w, t]）→ 逐小时均摊速率。

    采样间隔小于窗口长度时，窗口相互重叠，故取**相位平铺子集**：以首个采样为相位、
    每 ``window_hours`` 取一个窗口端点，这些窗口两两无缝拼接、恰好覆盖时间轴。
    每个小时 h 归属包含它的唯一平铺窗口（端点 t ∈ [h, h+w)），
    ``out[h] = 累计(t) / window_hours``。效果：

      a) 每小时恰属一个窗口，任意完整覆盖跨度上的求和 == 原始累计总量（BIAS 不失真）；
      b) 产出量为 mm/h 速率，与观测 rain@t（前 1 小时累计）的配对口径同"中科天机 mm/h
         近似"一档；
      c) 窗口时长越长，短时强降水被摊得越薄（风乌 6h > EW4ALL 的 3h/6h 视模式而定）。

    未被平铺窗口覆盖的小时（首窗之前/末窗之后）为 None。**端点缺失就是缺测**：
    某个采样点缺失时，包含它的那个窗口的总量无从得知，该窗口覆盖的 1~window_hours
    个小时整体为 None（绝不借用邻窗值或按剩余样本摊分——那是凭空造量）。
    """
    if window_hours <= 0:
        raise ValueError("window_hours 必须为正整数小时")
    by_time: dict[datetime, float | None] = {}
    for t, v in samples:
        by_time.setdefault(t, v)
    if not by_time or not hours:
        return [None] * len(hours)
    span = timedelta(hours=window_hours)
    tile = tile_endpoints(samples, window_hours)
    out: list[float | None] = []
    for h in hours:
        tk = next((t for t in tile if h <= t < h + span), None)
        v = by_time[tk] if tk is not None else None
        out.append(None if v is None else v / window_hours)
    return out

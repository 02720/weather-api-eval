"""一次性标定脚本：中国气象局公众网（weather.cma.cn）降水累计窗口的判定与复核。

背景（第一性原理）：`/api/hourly/{站号}` 的 `precipitation` 字段**未逐字声明**
累计窗口的长度与方向，而口径错一倍（把 3 小时累计当 1 小时量入库 = 雨量放大 3 倍）
不会报任何错、只会静默污染该源全部降水样本。故用两条**外部对照**判定：

  1. **逐窗同源量比（决定性）**：把本源 `precipitation(t)` 与 CMA-NDFS 的**逐小时**
     降水序列对齐，在同一批逐窗样本上求池化和。若本源是 1 小时量，Σ本源/ΣNDFS-1h
     应 ≈1；若是 3 小时累计，应 ≈3。同时给出与 NDFS **3 小时**后向累计的和比与相关
     （应 ≈1），以及三个相位的扫描（鉴别窗口方向）。
  2. 跨源分位：把该源按两种解释下的窗口总量，放进同站同窗口、存档中其余所有模式的
     窗口总量分布里看分位——同一场天气下，一个模式的雨量总量应与**其余模式的分布**
     （中位数附近）相称。（注意不能用"是否落在 min~max 带内"当判据：带的宽度常达
     两个数量级，两种解释都能落进去，没有鉴别力。）
  3. 同源日总量：与 CMA-NDFS 的同期 24 小时总量比——按 3 小时累计解释应接近 1，
     按 1 小时量解释应接近 1/3。

结论写进 README 的"评估方法·口径表"与 forecast/cma_public.py docstring 第 6 条。
运行：
  PYTHONPATH=src python scripts/calibrate_cma_public_precip.py [--days 5]

--- 2026-09-13 首次运行结论（4 站）---
  **逐窗量比（决定性）**：196 个逐窗样本，Σ本源 = 241.6mm、ΣNDFS-1h = 81.9mm，
  比值 **2.95**——恰为 3，而非 1 小时量所要求的 1；同批样本上本源与 NDFS 的 3 小时
  后向累计和比 1.01、r=0.68。相位扫描 r = 0.677/0.686/0.698（后向/居中/前向），
  三者不可分，故按国内业务惯例取后向。
  跨源分位（3h 解释 / 1h 解释）：15%/4%、8%/0%、40%/12%、8%/0%——3h 解释一致优于
  1h 解释（后者把本源推成"比中位模式干 5 倍"的系统性极值）。
  同源日总量比：0.53~0.80（随产品循环而变），按 1 小时量解释应为 ≈0.33。
  判定：`precipitation` = 后向 3 小时累计 (t−3h, t]，按 spread_accumulation(3) 展开。
  附带发现：博白站在两个口径下都显著偏干——该站的干偏差是数据源自身的特征，不是
  口径问题，已作为已知边界披露。
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from datetime import datetime, timedelta

sys.path.insert(0, "src")

import requests

from weather_eval.config import load_config
from weather_eval.forecast.cma_public import HEADERS, MODEL_NAME as CMA_PUBLIC_MODEL
from weather_eval.forecast.cma_public import extract_points
from weather_eval.storage import _root, list_forecast_snapshots

BASE_URL = "https://weather.cma.cn/api/hourly"
# CMA-NDFS 的降水快照已是"前 1 小时累计"速率，24h 求和即可与本源日累计对拍
NDFS_MODEL = "cma_ndfs"


def fetch_cma_series(station) -> dict[datetime, float]:
    """站号 → {北京时整点: precipitation}（占位哨兵 999.9 已剔除）。"""
    resp = requests.get(f"{BASE_URL}/{station.cma_id}", headers=HEADERS, timeout=30)
    resp.raise_for_status()
    points, _ = extract_points(resp.json())
    out = {}
    for t, entry in points:
        v = entry.get("precipitation")
        if isinstance(v, (int, float)) and v < 999:
            out[t] = float(v)
    return out


def ndfs_daily(station) -> dict[str, float]:
    """CMA-NDFS 快照 → {北京时自然日: 24h 累计}（取最新一份快照）。"""
    snaps = list_forecast_snapshots(station.id, NDFS_MODEL)
    if not snaps:
        return {}
    snap = snaps[-1]
    model = snap["models"][0]
    per_day: dict[str, float] = {}
    for t_iso, v in zip(snap["hourly_time"], snap["data"][model]["precipitation"]):
        if v is None:
            continue
        day = t_iso[:10]
        per_day[day] = per_day.get(day, 0.0) + float(v)
    return per_day


def archive_window_totals(station, days: list[str]) -> list[tuple[str, float]]:
    """存档中其余源在对照窗口内的降水总累计 → [(模型, 总量)]。

    这是量级对照的正确比较对象：本源某站的窗口总量应与**同站同窗口的单个模式**
    比，而不是与"所有模式加起来"比（后者会把分布带撑宽一倍以上）。
    每个模式取目录下最新一份快照（与评估侧口径一致）。
    """
    out: list[tuple[str, float]] = []
    root = _root() / "forecasts" / station.id
    if not root.exists():
        return out
    for mdir in sorted(root.iterdir()):
        if not mdir.is_dir() or mdir.name == CMA_PUBLIC_MODEL:
            continue
        files = [p for p in sorted(mdir.glob("*.json")) if p.suffix == ".json"]
        if not files:
            continue
        try:
            snap = json.loads(files[-1].read_text(encoding="utf-8"))
            model = snap["models"][0]
            vals = zip(snap["hourly_time"], snap["data"][model]["precipitation"])
        except Exception:  # noqa: BLE001
            continue
        total = sum(float(v) for t_iso, v in vals
                    if v is not None and t_iso[:10] in days)
        if total > 0:
            out.append((model, total))
    return sorted(out, key=lambda kv: kv[1], reverse=True)


def window_ratio_test(series: dict[str, dict[datetime, float]]) -> None:
    """决定性检验：逐窗把本源与 CMA-NDFS 的 1 小时 / 3 小时累计对齐求池化和。

    本源的 `precipitation(t)` 是**一个采样点的累计量**。若它是 1 小时量，则与
    NDFS 同一时刻的 1 小时值量级相同（Σ比≈1）；若是 3 小时累计，则应约为其 3 倍
    （Σ比≈3）。同一批样本上再对照 NDFS 的 3 小时后向累计（Σ比应≈1），并做相位
    扫描以看窗口方向是否可鉴别。
    """
    phases = {"后向 (t-3h, t]": (-2, -1, 0), "居中 (t-1h, t+2h]": (-1, 0, 1),
              "前向 (t, t+3h]": (0, 1, 2)}
    acc = {k: [0.0, 0.0, 0.0, 0.0, 0.0, 0] for k in list(phases) + ["1h 后向"]}
    sx = sn1 = 0.0
    for st_id, series_map in series.items():
        ndfs = ndfs_daily_series(st_id)
        if not ndfs:
            print(f"  [{st_id}] 无 CMA-NDFS 存档，跳过逐窗对照")
            continue
        for t, x in series_map.items():
            if t not in ndfs or ndfs[t] is None:
                continue
            sx += x
            sn1 += float(ndfs[t])
            for name, off in phases.items():
                if any((t + timedelta(hours=k)) not in ndfs for k in off):
                    continue
                vs = [ndfs[t + timedelta(hours=k)] for k in off]
                if any(v is None for v in vs):
                    continue
                _accumulate(acc[name], x, sum(vs))
            _accumulate(acc["1h 后向"], x, float(ndfs[t]))
    if not sn1:
        return
    print(f"逐窗池化：Σ本源 = {sx:.2f}mm   ΣNDFS-1h = {sn1:.2f}mm   "
          f"Σ本源/ΣNDFS-1h = {sx / sn1:.2f}")
    print("  判读：≈1 → 本源是 1 小时量；≈3 → 本源是 3 小时累计")
    for name, a in acc.items():
        n = a[5]
        if not n:
            continue
        print(f"    {name:<15} n={n:3d}  和比 Σ本源/Σ对照={a[0] / a[1]:.2f}  "
              f"r={_corr(a):.3f}")
    print()


def _accumulate(a: list, x: float, y: float) -> None:
    a[0] += x; a[1] += y; a[2] += x * x; a[3] += y * y; a[4] += x * y; a[5] += 1


def _corr(a: list) -> float:
    n = a[5]
    mx, my = a[0] / n, a[1] / n
    vx, vy = a[2] / n - mx * mx, a[3] / n - my * my
    cov = a[4] / n - mx * my
    return cov / (vx ** 0.5 * vy ** 0.5) if vx > 0 and vy > 0 else float("nan")


def ndfs_daily_series(station_id: str) -> dict[datetime, float | None]:
    """CMA-NDFS 最新快照的逐小时降水序列（已是"前 1 小时累计"速率）。"""
    snaps = list_forecast_snapshots(station_id, NDFS_MODEL)
    if not snaps:
        return {}
    snap = snaps[-1]
    model = snap["models"][0]
    return {datetime.strptime(t, "%Y-%m-%dT%H:%M"): v
            for t, v in zip(snap["hourly_time"], snap["data"][model]["precipitation"])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5, help="对拍的窗口天数（默认 5）")
    ap.add_argument("--start", default=None,
                    help="窗口起始日 YYYY-MM-DD（默认今天，北京时）")
    args = ap.parse_args()

    cfg = load_config()
    from weather_eval.timeutil import now_beijing
    d0 = (datetime.strptime(args.start, "%Y-%m-%d") if args.start
          else now_beijing().replace(hour=0, minute=0, second=0, microsecond=0))
    days = [(d0 + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(args.days)]

    print(f"对照窗口 {days[0]} ~ {days[-1]}（共 {args.days} 天）\n")
    fetched: dict[str, dict[datetime, float]] = {}
    for st in cfg.stations:
        if not st.cma_id:
            continue
        try:
            fetched[st.id] = fetch_cma_series(st)
        except Exception as e:  # noqa: BLE001
            print(f"[{st.id}] 抓取失败：{e}")

    window_ratio_test(fetched)

    tot_3h = tot_1h = tot_ndfs = 0.0
    pct3: list[float] = []
    pct1: list[float] = []
    logged: list[float] = []
    for st in cfg.stations:
        cma = fetched.get(st.id)
        if cma is None:
            continue
        ndfs = ndfs_daily(st)
        others = archive_window_totals(st, days)
        s3 = sum(v for t, v in cma.items() if t.strftime("%Y-%m-%d") in days)
        nd = sum(v for d, v in ndfs.items() if d in days)
        tot_3h += s3
        tot_1h += s3 / 3
        tot_ndfs += nd
        band = [v for _, v in others]
        print(f"[{st.id} / {st.cma_id}]")
        if band:
            med = statistics.median(band)
            q3 = _percentile(band, s3)
            q1 = _percentile(band, s3 / 3)
            pct3.append(q3)
            pct1.append(q1)
            logged.append(s3 / med if med else float("nan"))
            print(f"    其余 {len(band)} 个模式同窗口总量：中位 {med:.1f}mm  "
                  f"P25 {sorted(band)[len(band) // 4]:.1f}  "
                  f"P75 {sorted(band)[3 * len(band) // 4]:.1f}  "
                  f"min {min(band):.1f}  max {max(band):.1f}")
            print(f"    3h 口径 {s3:7.2f}mm → 分布分位 {q3:5.0f}%，为本源/中位 = {s3 / med:.2f}")
            print(f"    1h 口径 {s3 / 3:7.2f}mm → 分布分位 {q1:5.0f}%，为本源/中位 = "
                  f"{s3 / 3 / med:.2f}")
        else:
            print(f"    3h 口径 {s3:7.2f}mm   1h 口径 {s3 / 3:7.2f}mm（无对照模式）")
        if nd:
            print(f"    CMA-NDFS 同窗口 24h 合计 {nd:7.2f}mm → 3h 口径比值 {s3 / nd:.2f}")
    print()
    if tot_ndfs:
        print(f"同源对照合计：3h 口径 / CMA-NDFS = {tot_3h / tot_ndfs:.2f}"
              "（若本源实为 1 小时量，该比值应 ≈0.33）")
    print(f"跨源量级合计：3h 口径 {tot_3h:.1f}mm vs 1h 口径 {tot_1h:.1f}mm")
    if pct3:
        g3 = math.exp(statistics.mean(math.log(x) for x in logged))
        print(f"跨源分位：3h 口径四站分位 {[f'{p:.0f}%' for p in pct3]}（为本源/中位几何均值 {g3:.2f}）")
        print(f"          1h 口径四站分位 {[f'{p:.0f}%' for p in pct1]}"
              f"（为本源/中位几何均值 {g3 / 3:.2f}）")
        print("判读：分位接近中位、比值接近 1 的那种解释成立；把本源推成系统性极值的解释不成立。")
    return 0


def _percentile(band: list[float], x: float) -> float:
    """x 在 band 中的百分位（含等于 x 的样本）。"""
    if not band:
        return 0.0
    return 100.0 * sum(1 for v in band if v <= x) / len(band)


if __name__ == "__main__":
    raise SystemExit(main())

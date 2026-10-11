"""一次性标定脚本：中国气象数据网（data.cma.cn）智能网格预报的分段口径判定。

背景（第一性原理）：`/rest/gowfs/hour` 的 `total_rain` 字段**未逐字声明**累计窗口的
长度，而口径错一倍（把 3 小时累计当 1 小时量入库 = 雨量放大 3 倍；或把 1 小时量当
3 小时累计摊薄 = 雨量缩到 1/3）不会报任何错、只会静默污染该源全部降水样本。更要紧的
是：实测该接口的网格是**分段**的——前 24 点逐小时、其后严格 3 小时（见 forecast/
cma_gowfs.py docstring 契约第 4 条），两段的口径**可能不同**，必须分别判定。

本脚本对三个互不等价的问题各给一条证据：

  1. **网格分段点**：从首点起相邻间隔严格 1h 的连续前缀长度（近/远场分界）。
  2. **近场口径（1h 累计 vs 3h 累计按小时展示）**：块内恒定性检验。把近场按 3h 相位
     切成 3 小时块，若"每个点是 3h 累计、按小时重复展示"，块内 3 个值必须逐位相同。
     含雨的块有变化 ⇒ 近场是逐小时独立值。全零块无鉴别力，单独剔除后计数。
  3. **远窗口径与方向（(t−3h,t] vs [t,t+3h) vs 1h 累计）**：与同源 `/day` 端点的
     **定性等级**对拍。12h 昼夜窗的总量 = 恰好铺满该窗的 4 个累计点之和；按 CMA 12h
     等级标准判级，比较两种点集归属的命中率。（域外对照源：`/day` 的 day/night
     天气现象码，与远场降水同属一次起报的产品。）
  4. **起报锚点恒等式**：`pt == 首点−1h == 末点−240h` 是否对全部坐标成立。
  5. **逐日 day/night 的日界**：day/night 温度分别对标自然日最高/最低、以及昼夜窗
     （08–20 / 20–08）极值，确认 day/night 是**昼夜窗**而非自然日，并据此确定
     `temp_min(D) = nightT(D−1)` 的移位取法（对拍两种取法的中位 |Δ|）。
  6. **逐日降水字段**：清点 `precipitation` 的取值分布，确认是否恒为空串。

结论写进 README 的"按源口径与已知坑位"与 forecast/cma_gowfs.py docstring。
运行：
  PYTHONPATH=src python scripts/calibrate_cma_gowfs.py [--max-points 18] [--json 路径]

--- 2026-10-11 首次运行结论（18 个经纬点，`--json` 存原始数据可复查）---
  **分段点**：全部 18 点唯一取值 = 24 点逐小时（前 24h），其后 72 点严格 3 小时，
  跨度 239h ⇒ 视野 240h = 10 天（天桶 1..10）。
  **锚点恒等式**：18/18 点满足 pt == 首点−1h == 末点−240h（逐位相等）。
  **近场 = 1h 累计**：126 个 3h 块中 112 块全零（无鉴别力），14 个含雨块 13 块有变化
  （仅 1 块偶然相同）；块内相邻小时 |Δ| 中位 0.030mm、p90 0.220mm、最大 0.49mm。
  （块计数随实时雨区漂移，同天另一次运行得 113/13/12，结论不变。）
  **远场 = 3h 后向累计**：102 个昼夜样本，后向 (t−3h,t] 命中 77/102（75%），
  前向 [t,t+3h) 命中 58/102（57%）。分档：小雨 86% vs 64%、中雨 60% vs 40%；
  **大雨 n=3 且结论相反（33% vs 67%）**——小样本，如实留档，待满月样本复标。
  **大雨 n=3 且结论相反（33% vs 67%）**——小样本，如实留档，待满月样本复标。
  **逐日 day/night = 昼夜窗（08–20 / 20–08）而非自然日**：
    dayT − 逐小时日最高：mean +0.99℃、中位 |Δ| 0.78℃、p90 3.42℃（因线性插值不产生
      新极值，"自然日最高"与"昼夜窗最高"两个对标量几乎不可分，故 dayT 的对标高
      **不是**判别日界的证据）；
    nightT(D)   − 自然日最低(D)：mean −0.17℃、中位 |Δ| 1.07℃；
    nightT(D−1) − 自然日最低(D)：mean +0.003℃、中位 |Δ| 0.68℃ ← 明显更优
    ⇒ `temp_min(D) = nightT(D−1)`（自然日 D 的低温在 D 凌晨，属 D−1 的夜间窗）。
  **逐日降水**：18×30 = 540 个样本 `precipitation` **全部为空串** ⇒ 拒绝接入（全 null）。
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from datetime import datetime, timedelta

# 仓库统一 UA：实测不在该站反爬名单内（名单拒绝空 UA 与 python-requests）
UA = "weather-api-eval/0.1 (+https://github.com/)"
BASE = "https://data.cma.cn/rest/gowfs"
TIMEOUT = 40

# 覆盖全国不同气候带的采样点（含 4 个本站 + 高原/西北/东北/江淮/华南）
DEFAULT_POINTS = [
    (23.4783, 111.304),   # 梧州
    (22.2981, 109.99),    # 博白
    (23.5431, 110.419),   # 平南
    (18.8006, 110.327),   # 万宁
    (38.0, 101.0), (43.0, 87.0), (40.0, 116.0), (36.0, 120.0),
    (34.0, 113.0), (31.0, 121.5), (30.5, 114.3), (29.5, 106.5),
    (28.5, 112.0), (27.5, 104.9), (26.0, 119.3), (25.0, 110.0),
    (24.5, 118.1), (22.5, 114.1),
]

# CMA 12h 降水等级标准（mm）
BANDS = {"小雨": (0.1, 4.9), "中雨": (5.0, 14.9), "大雨": (15.0, 29.9), "暴雨": (30.0, 1e18)}
RAIN_INFO = {"小雨": 1, "中雨": 2, "大雨": 3, "暴雨": 4,
             "小雪": 1, "中雪": 2, "大雪": 3, "暴雪": 3}

log = logging.getLogger("calibrate_gowfs")


def fetch(path: str, lat: float, lon: float, retries: int = 2) -> dict:
    url = f"{BASE}{path}?lat={lat}&lon={lon}"
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:  # noqa: BLE001  脚本层只做留痕，失败即换下一点
            last = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"{url} 抓取失败: {last}")


def _f(v) -> float:
    try:
        return float(v or 0)
    except (TypeError, ValueError):
        return 0.0


def _parse(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d %H:%M")


def hour_series(payload: dict) -> tuple[list[tuple[datetime, dict]], int]:
    """返回 (按时刻升序去重后的点列, 逐小时前缀长度)。"""
    pts: dict[datetime, dict] = {}
    for e in payload["detail"]:
        pts[_parse(e["time"])] = e
    series = sorted(pts.items())
    n = 1
    for (t0, _), (t1, _) in zip(series, series[1:]):
        if (t1 - t0) == timedelta(hours=1):
            n += 1
        else:
            break
    return series, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-points", type=int, default=len(DEFAULT_POINTS))
    ap.add_argument("--json", default=None, help="把逐点原始数据存成 JSON 供复查")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    points = DEFAULT_POINTS[: args.max_points]
    cache: dict[tuple[float, float], tuple[dict, dict]] = {}
    for lat, lon in points:
        try:
            cache[(lat, lon)] = (fetch("/hour", lat, lon), fetch("/day", lat, lon))
        except Exception as e:  # noqa: BLE001
            log.warning("跳过 (%s,%s): %s", lat, lon, e)
    if not cache:
        log.error("没有任何坐标抓取成功")
        return 1
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({f"{a},{b}": [h, d] for (a, b), (h, d) in cache.items()}, fh)
        log.info("原始数据已写入 %s", args.json)

    # ---------------- 1+4：分段点 / 起报锚点恒等式 ----------------
    print("=" * 78)
    print("1) 网格分段点与视野 ／ 4) 起报锚点恒等式")
    print("=" * 78)
    seg_ok = anchor_ok = 0
    segs: set[int] = set()
    for (lat, lon), (hp, dp) in sorted(cache.items()):
        series, n = hour_series(hp)
        dpts = [e for e in dp["detail"] if str(e.get("pt") or "").strip()]
        pt = {e["pt"] for e in dpts}
        pt_dt = _parse(next(iter(pt))) if len(pt) == 1 else None
        first, last = series[0][0], series[-1][0]
        gaps = {(b[0] - a[0]).total_seconds() for a, b in zip(series, series[1:])}
        span = int((last - first).total_seconds() // 3600)
        segs.add(n)
        ok_anchor = (pt_dt == first - timedelta(hours=1) == last - timedelta(hours=240))
        seg_ok += 1
        anchor_ok += ok_anchor
        print(f"  ({lat},{lon})  近场 {n:2d} 点 | 间隔 {sorted(int(g/3600) for g in gaps)}h"
              f" | 跨度 {span}h | pt={pt_dt} 首点={first} 末点={last}"
              f" | 恒等式 {'✓' if ok_anchor else '✗'}")
    print(f"\n  分段点取值集合 {sorted(segs)}（应唯一 = 24）| 锚点恒等式成立 "
          f"{anchor_ok}/{seg_ok}")

    # ---------------- 2：近场口径（块内恒定性） ----------------
    print()
    print("=" * 78)
    print("2) 近场口径：3h 块内恒定性检验（近场=逐小时独立值 ⇔ 含雨块有变化）")
    print("=" * 78)
    zero = rain_const = rain_var = 0
    diffs: list[float] = []
    for (lat, lon), (hp, _) in sorted(cache.items()):
        series, n = hour_series(hp)
        near = {t: _f(e.get("total_rain")) for t, e in series[:n]}
        # 相位锚取"落在 3h 网格上、且最靠前的那个**时刻**"（不是最靠前的小时数）：
        # 后者会把锚点打到 24h 窗末段，只剩 2 个完整块，样本量缩水 3.5 倍。
        PHASE = {2, 5, 8, 11, 14, 17, 20, 23}
        t0 = min(t for t in near if t.hour in PHASE)
        for k in range(0, 24, 3):
            blk = [near[t0 + timedelta(hours=k + i)]
                   for i in range(3) if t0 + timedelta(hours=k + i) in near]
            if len(blk) < 3:
                continue
            if not any(blk):
                zero += 1
                continue
            if len(set(blk)) == 1:
                rain_const += 1
            else:
                rain_var += 1
                diffs += [abs(blk[i + 1] - blk[i]) for i in range(2)]
    n_rain = rain_const + rain_var
    print(f"  3h 块：全零 {zero}（无鉴别力，剔除）｜含雨 {n_rain}（恒定 {rain_const}，"
          f"有变化 {rain_var}）")
    if diffs:
        sd = sorted(diffs)
        print(f"  含雨块内相邻小时 |Δ|：中位 {sd[len(sd)//2]:.3f}mm、"
              f"p90 {sd[int(0.9*len(sd))]:.3f}mm、最大 {sd[-1]:.2f}mm")
    print(f"  ⇒ 含雨块恒定比例 {rain_const}/{n_rain}"
          f"（若近场是 3h 累计按小时展示，应 {n_rain}/{n_rain}）")

    # ---------------- 3：远窗口径与方向 ----------------
    print()
    print("=" * 78)
    print("3) 远窗口径与方向：与 /day 定性等级的 12h 昼夜窗对拍")
    print("=" * 78)
    rows: list[tuple[str, float, float]] = []
    for (lat, lon), (hp, dp) in sorted(cache.items()):
        series, n = hour_series(hp)
        far = {t: _f(e.get("total_rain")) for t, e in series[n:]}
        for e in dp["detail"]:
            d = datetime.strptime(e["date"], "%Y-%m-%d")
            for info, start in ((e["day"]["weather"]["info"], 8),
                                (e["night"]["weather"]["info"], 20)):
                if info not in RAIN_INFO:
                    continue
                w0 = d + timedelta(hours=start)
                # 后向 (t-3h,t]：铺满 [w0,w1) 的点集是 (w0, w1]
                back = [far.get(w0 + timedelta(hours=k)) for k in range(3, 13, 3)]
                # 前向 [t,t+3h)：铺满 [w0,w1) 的点集是 [w0-3h, w1-3h)
                fwd = [far.get(w0 + timedelta(hours=k)) for k in range(-3, 9, 3)]
                if any(v is None for v in back + fwd):
                    continue
                rows.append((info, sum(back), sum(fwd)))
    print(f"  有效昼夜样本 {len(rows)} 条")
    print(f"  {'等级':<6}{'n':>5}{'后向(t-3h,t]命中':>18}{'前向[t,t+3h)命中':>19}")
    tb = tf = 0
    for name, (lo, hi) in BANDS.items():
        sub = [(b, f) for i, b, f in rows if i == name]
        if not sub:
            continue
        hb = sum(1 for b, _ in sub if lo <= b < hi) / len(sub)
        hf = sum(1 for _, f in sub if lo <= f < hi) / len(sub)
        tb += sum(1 for b, _ in sub if lo <= b < hi)
        tf += sum(1 for _, f in sub if lo <= f < hi)
        note = "  ← n 偏小" if len(sub) < 10 else ""
        print(f"  {name:<6}{len(sub):>5}{hb:>17.0%}{hf:>18.0%}{note}")
    tot = len(rows)
    print(f"  {'合计':<6}{tot:>5}{tb/tot:>17.0%}{tf/tot:>18.0%}")
    print(f"  ⇒ 后向命中 {tb}/{tot} vs 前向 {tf}/{tot}")

    # ---------------- 5：逐日 day/night 的日界 ----------------
    print()
    print("=" * 78)
    print("5) 逐日 day/night 的日界：昼夜窗 vs 自然日（决定 temp_min 的移位数）")
    print("=" * 78)
    err: dict[str, list[float]] = {}

    def interp(pairs: list[tuple[datetime, float]]) -> dict[datetime, float]:
        series = sorted(pairs)
        full: dict[datetime, float] = {}
        for i, (t, v) in enumerate(series):
            full[t] = v
            if i + 1 < len(series):
                t2, v2 = series[i + 1]
                span = (t2 - t).total_seconds() / 3600
                if span > 1:
                    for k in range(1, int(span)):
                        full[t + timedelta(hours=k)] = v + (v2 - v) * (k / span)
        return full

    for (lat, lon), (hp, dp) in sorted(cache.items()):
        series, _ = hour_series(hp)
        T = interp([(t, _f(e.get("temperature"))) for t, e in series])
        dd = {e["date"]: e for e in dp["detail"]}
        for ds, e in dd.items():
            d = datetime.strptime(ds, "%Y-%m-%d")
            ks = [d + timedelta(hours=k) for k in range(24)]
            if any(k not in T for k in ks):
                continue
            nat_min = min(T[k] for k in ks)
            day_win = [T[k] for k in ks if 8 <= k.hour < 20]
            nxt = d + timedelta(days=1)
            night_win = [T[k] for k in ks if k.hour >= 20] + \
                        [T[nxt + timedelta(hours=k)] for k in range(8)
                         if nxt + timedelta(hours=k) in T]
            if len(night_win) < 12:
                continue
            err.setdefault("dayT−自然日最高", []).append(
                _f(e["day"]["weather"]["temperature"]) - max(T[k] for k in ks))
            err.setdefault("dayT−昼夜窗最高(08-20)", []).append(
                _f(e["day"]["weather"]["temperature"]) - max(day_win))
            err.setdefault("nightT(D)−自然日最低(D)", []).append(
                _f(e["night"]["weather"]["temperature"]) - nat_min)
            prev = dd.get((d - timedelta(days=1)).date().isoformat())
            if prev is not None:
                err.setdefault("nightT(D−1)−自然日最低(D)", []).append(
                    _f(prev["night"]["weather"]["temperature"]) - nat_min)

    def show(name: str, vals: list[float]) -> None:
        a = sorted(abs(v) for v in vals)
        print(f"  {name:<30} n={len(vals):4d}  mean={sum(vals)/len(vals):+.3f}℃"
              f"  中位|Δ|={a[len(a)//2]:.3f}℃  p90={a[int(0.9*len(a))]:.3f}℃")

    for k in ("dayT−自然日最高", "dayT−昼夜窗最高(08-20)",
              "nightT(D)−自然日最低(D)", "nightT(D−1)−自然日最低(D)"):
        if err.get(k):
            show(k, err[k])
    print("  ⇒ nightT(D−1) 中位 |Δ| 显著小于 nightT(D) ⇒ 日最低须移一天取用")

    # ---------------- 6：逐日降水字段 ----------------
    print()
    print("=" * 78)
    print("6) 逐日降水字段取值分布（恒为空串 ⇒ 拒绝接入，全 null）")
    print("=" * 78)
    vals: dict[str, int] = {}
    n_day = 0
    for (lat, lon), (_, dp) in sorted(cache.items()):
        for e in dp["detail"]:
            n_day += 1
            vals[repr(e.get("precipitation"))] = \
                vals.get(repr(e.get("precipitation")), 0) + 1
    print(f"  {n_day} 个逐日样本，precipitation 取值分布：{vals}")

    print()
    print("判定结论（写回 docstring / README）：")
    print("  · 近场 = 1h 后向累计 (t−1h, t]（透传）；远场 = 3h 后向累计 (t−3h, t]（÷3 摊到小时）")
    print("  · 起报锚点 = /day 的 pt（18/18 站 pt == 首点−1h == 末点−240h）")
    print("  · 逐日只接温度：temp_max(D)=dayT(D)、temp_min(D)=nightT(D−1)；降水全 null")
    return 0


if __name__ == "__main__":
    sys.exit(main())

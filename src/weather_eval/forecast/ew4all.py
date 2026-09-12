"""EW4ALL（Cloud-based Early Warning Supporting System）预报快照器。

数据源：中国气象局「云上早期预警支撑系统」模式预报页
（http://ew4all.wmc-bj.net/EW4ALL/predictions）背后的公开接口。**网页抓取，游客态
可用、无需任何凭据**；接口清单来自 2026-09-12 线上实测 + 前端 bundle 逆向。

  BASE = http://ew4all.wmc-bj.net/EW4ALL
  GET  {BASE}/api/modelTimeList?data_type={mode}&element={el}   # 该模式的可用起报时次
  POST {BASE}/api/raster/findByPoint                            # 单点要素序列
       body: {"mode": mode, "elements": element, "point": [[lon, lat]],
              "projection": 4326, "dataTime": <YYYYMMDDHH>, "level": 0}
       响应: {"code":200,"message":"获取成功",
              "data":[{"<element>": <val>, "Datetime": "YYYY-MM-DD HH:MM:SS", "level": 0}, ...]}

─────────────────────────────────────────────────────────────────────
关键契约（改动前必须重新实测复核）
─────────────────────────────────────────────────────────────────────
1. 模型标识：`mode` 取前端枚举值——`GDFS5KM`=CMA-NDFS（中国气象局智能数字天气预报
   系统）、`NMCFENGQING`=风清AI模式。页面上另标 `CMA_GDFS`→"CMA-NDFS"、
   `FENGQING`→"风清AI模式"，即同一产品的两套代号。**一次请求只接受一个 `elements`
   值**（实测传 "TEM,ONETPE" 或数组均返回空/422），故温度与降水各发一次请求。
2. ── 时间语义：`Datetime` 与 `dataTime` 同为 **UTC**（非北京时）。本条的判定证据
   必须保留，因为"错 8 小时"这种偏差不会报错、只会静默污染全部样本：
     a) 日变化相位：梧州站点 240h 的 TEM 日变化与 4 站 12 天观测的日变化做互相关，
        最佳平移为 +7~+8h（r=0.946/0.941），+0h 时 r 为负（−0.435）——原始时刻的
        "日最高"出现在 06~08 时、"日最低"出现在 20~23 时，只有按 UTC 解释
        （=北京时 14~16 时最高、04~07 时最低）才符合物理。
     b) 时效为整：把起报标签按 UTC 解释时，GDFS5KM 的序列末点恰为起报 +240h（10 天）、
        NMCFENGQING 恰为 +360h（15 天），与产品规格严丝合缝；按北京时解释则得到
        248h/368h 这类非整数时效。
     c) 入库后回归验证（2026-09-12，接后立即做）：按 +8h 入库的快照与 eia-data 实况
        配对，lead 1~8h 的 CMA-NDFS 温度 RMSE = 1.23°C（4 站 32 样本）；若时区错 8
        小时，同样的比对会落到"拿白天温度比夜间实况"上，RMSE 应在 4~6°C 量级。
        本条同时是"配对能对上"这一事实的证明——评估引擎按整点精确配对，
        时区错则样本数为 0 且不报任何错。
     d) 逐点时刻 = 起报标签 + 步长（GDFS5KM 首点 +1h、NMCFENGQING 首点 +6h 即 6 小时
        产品的第一个时次）。入库前一律 +8h 转北京时 naive 墙钟。
3. 起报轮次：`modelTimeList` 返回该模式近若干轮 `data_time`（"YYYYMMDDHH0000"），
   GDFS5KM 与 NMCFENGQING 实测均为 00/12 UTC 两轮。可用轮次还必须**要素齐全**：
   该源在一轮内部是分要素先后出数的（实测标注 12Z 的 GDFS5KM 在其后一两小时内
   只有 `TEM`、降水要素仍为空，下一轮 00Z 才两者齐备），故按轮次由新到旧回退，
   取第一个"温度与降水同时有数据"的轮次作为起报锚点（MAX_ISSUE_FALLBACK 轮）。
4. 各模型要素与时效（2026-09-12 实测）：

   | 模型 | mode | 温度 | 降水 | 逐小时上限 |
   |---|---|---|---|---|
   | CMA-NDFS | GDFS5KM | `TEM` 逐小时 0–72h、其后逐 3 小时 | `ONETPE`（1 小时累计）0–72h ＋ `HOURTPE`（3 小时累计）全时效 | 240h（10 天） |
   | 风清AI模式 | NMCFENGQING | `TEM` 逐 6 小时（+6h 起） | `SIXTPE`（6 小时累计，逐 6 小时） | 温度 360h（15 天）/ 降水 240h（10 天） |

5. ── 降水口径：**优先用原生 1 小时产品，其余按累计窗口展开为逐小时** ──
   该源的降水要素是一族后向累计：`ONETPE`(1h)/`HOURTPE`(3h)/`SIXTPE`(6h)/
   `TWELVETPE`(12h)/`DAYTPE`(24h)。窗口方向由"族内严格自洽"确证为**后向** (t−w, t]：
   实测 3h+3h==6h、6h+6h==12h、12h+12h==24h 全部 0.00mm 偏差（79/79 与 37/37 点），
   "自起报累计"与前向窗口都无法满足该恒等式。
   入库策略（1 小时产品优先，精度最高）：
   - **CMA-NDFS**：`ONETPE`（原生 1 小时累计）在它**实际有值的时效内直接透传**——
     与观测 `rain@t`（前 1 小时累计）口径完全一致，无需任何展开，是本源精度最高的
     降水产品；其覆盖范围之外（实测起报后 0–72h）改用 `HOURTPE`/3 平铺均摊补齐。
     交接点取"ONETPE 实际有值的最后一个小时"，**不硬编码 72h**：平台延长 ONETPE
     时效时自动跟随，快照 meta 的 `precip_1h_until` 留档每一份的实际交接点。
   - **风清AI模式**：无 1 小时产品（`ONETPE`/`HOURTPE` 实测均为空），全程 `SIXTPE`/6。
   - **非 1 小时分辨率的累计量**按 `resample.spread_accumulation` 展开：即对**累计曲线**
     线性插值后逐小时差分，数学上等价于按窗口均摊 /w，且是唯一保证"任意完整跨度求和
     == 原始累计总量"的确定性展开（与风乌 tp6h 同法）。**不采用"对累计量数值本身做斜坡
     插值"**：那会让日总量失真，而按天 24h 累计正是当前降水分的主轨道。
   **已知口径切换**：CMA-NDFS 在 ONETPE 覆盖段（约前 3 天）与之后（3 小时摊薄）之间
   存在产品分辨率切换。两者在总量上高度一致（4 站 96 个重合窗口合并总量比 0.98，
   分站 0.73~1.02），但逐窗口有 30% 的点偏差 >0.35mm，即"日总量接近、日内分布有出入"。
   按分时效榜解读时，前 3 天与之后的桶来自不同分辨率的产品——这是数据源自身的产品
   结构（短时效给了更细的产品），不是人为混用；README 口径表与"已知约束"均已披露。
6. ── `ONETPE` 与 3 小时族的既有争议（2026-09-12 当日更正，留档以防反复）──
   最初以**单站 24 个重合窗口**判定"ONETPE 与 3h/6h/12h/24h 累计族不自洽"（6h 窗口
   45% 的点偏差 >0.35mm、逐日 1.8mm vs 4.9mm），并据此拒绝使用 ONETPE、只用均摊族。
   **该结论已被推翻**：扩到 4 站 96 个重合窗口复核后——相位扫描明确 k=0（后向窗口
   (t−3h, t]）最优（MSE 0.71，其余相位 1.95~6.35；k=+1/-1 分别是 3.27/1.95），
   4 站合并总量比 0.98（ONETPE 249.6mm vs 3h 族 255.3mm，分站 0.73~1.02）：
   **同相位、同总量**，说明二者是同一物理量在不同后处理路径下的产物，30% 的逐窗口
   出入属产品间正常差异，而非"互斥的两个物理量"。原结论是小样本过度推断，已纠正；
   现以 ONETPE 作为 CMA-NDFS 短时效降水的首选产品。
   （教训：窗口级自洽性判定的样本量必须按"站 × 窗口"计，单站 20 余个窗口不足以
   区分"相位错位"与"产品差异"——两者在 MSE 上的差距会被噪声淹没。）
7. 温度：`TEM`（℃，实测已为摄氏，无需换算）。采样粗于逐小时的区段（NDFS 的
   72h 之后、风清全程）按 `resample.interpolate_hourly` 线性插值到逐小时——与风乌
   3h→1h 同法：温度场平滑、插值误差可忽略。**但插值不产生新的极值**：线性插值恒
   落在两端采样值之间，故按天评估取到的日最高/最低**恒等于该模式自身采样时刻的
   那条序列所取到的极值**，不是插值造出来的。风清的采样时刻落在北京时
   02/08/14/20 时：14 时接近日最高、02 时接近日最低，但真实日最低常在 05~06 时、
   日最高常在 15~16 时——夹在采样点之间的极值取不到，表现为**日最低系统性偏暖、
   日最高系统性偏低**。这是该源 6 小时分辨率带来的精度上限（同 MSN 的整数摄氏度），
   不是接入缺陷，但横向比较日极值指标时应知情（README"已知约束"已披露）。
8. 不做格点回显：响应不含吸附后的格点经纬度（与伏羲中期同），`grid_lat/lon` 记请求
   坐标。底层网格为 GDFS5KM 5km / NMCFENGQING 约 25km，代表点误差计入已知限制。
9. 逐日预报块（不接入）：该源没有"北京时自然日"口径的逐日极值产品；`DAYTPE` 是
   (t−24h, t] 的滑动累计、时刻锚在 00/12 UTC（=北京时 08/20 时），不是自然日窗口。
   按 `base.py` 契约（日界必须是北京时自然日），它不能作为 daily 块入库；而逐小时
   轴已插值/平铺到逐小时，按天轨道由逐小时聚合即可覆盖全时效，无需补位。

已知风险（均已在上述证据下留档）：
- 平台在标注轮次**之前**即可返回该轮数据（2026-09-12 10:55 UTC 已可取标注 12Z 的
  GDFS5KM 温度序列）。起报锚点语义因此是"平台标注轮次"，与真实模式起报时刻可能有
  小时级偏差，且可能读到"先到"的早期版本；这与存档先行原则一致（快照一经落盘不再回改）。
- 轮次内分要素先后出数（见上文第 3 条）：起报锚点取"要素齐全"的轮次，因此最新一轮
  在降水出数前不会被采纳。后果是"同一批有效时刻由相邻轮次以更长时效覆盖"，
  短时效样本的构成随之变化——这是数据源发布节奏带来的已知差异，非接入缺陷。
- 无官方契约的网页接口：页面改版或后端重构即可能失效。失效形态是"返回空 data"而非
  报错，故所有解析都必须走"空数据 → 不建快照"的守卫，绝不写入空快照
  （空快照会被同 issue 幂等锁死，正常数据永远进不来）。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import requests

from .base import ForecastProvider
from .http import request_with_retries
from .resample import interpolate_hourly, spread_accumulation, tile_endpoints

logger = logging.getLogger(__name__)

BASE_URL = "http://ew4all.wmc-bj.net/EW4ALL"
MODEL_TIME_LIST_URL = BASE_URL + "/api/modelTimeList"
FIND_BY_POINT_URL = BASE_URL + "/api/raster/findByPoint"
HEADERS = {
    "User-Agent": "weather-api-eval/0.1 (+https://github.com/)",
    "Content-Type": "application/json",
}

SOURCE = "ew4all"
TEM_ELEMENT = "TEM"
MAX_ISSUE_FALLBACK = 4      # 起报轮次回退探测次数（由新到旧）
_RUN_LABEL_LEN = 14         # "YYYYMMDDHH0000"
# ONETPE（1 小时降水）实测覆盖起报后约 72 小时；偏短即告警（交接点会提前）
ONE_TPE_EXPECTED_HOURS = 72


@dataclass(frozen=True)
class ModelSpec:
    """一个模型在 EW4ALL 上的抓取参数与时效规格。

    降水用**两级要素**：`precip_1h_element`（原生 1 小时累计，可选）在其实际覆盖的
    时效内直接透传；超出部分由 `precip_element`（后向累计）按 `precip_window_hours`
    平铺均摊补齐。无 1 小时产品的模式把 `precip_1h_element` 置 None。
    """
    model: str                  # 入库模型名（config/report 层使用）
    mode: str                   # 接口 mode 值
    precip_element: str         # 兜底的降水要素代码（后向累计，覆盖全时效）
    precip_window_hours: int    # 该要素的累计窗口长度（小时）
    precip_1h_element: str | None   # 原生 1 小时累计要素（无则 None）
    # 逐小时轴的预期点数（截断告警阈值）：= 温度末时次 − 首时次 + 1
    # （GDFS5KM: +1h → +240h 共 240 点；NMCFENGQING: +6h → +360h 共 355 点）
    expected_points: int


# 模型集合与提供方同处登记（与 __main__.py 的 SOURCE_SPECS 呼应）。
MODEL_SPECS: dict[str, ModelSpec] = {
    # CMA-NDFS：温度逐小时（0–72h）后转逐 3 小时；降水 = ONETPE（1h，约 0–72h）
    # 直接透传 + HOURTPE（3 小时累计）平铺均摊补齐其余时效
    "cma_ndfs": ModelSpec("cma_ndfs", "GDFS5KM", "HOURTPE", 3, "ONETPE", 240),
    # 风清AI模式：温度/降水均逐 6 小时；无 1 小时降水产品，全程 SIXTPE/6
    "fengqing_ai": ModelSpec("fengqing_ai", "NMCFENGQING", "SIXTPE", 6, None, 355),
}


def _num(v: Any) -> float | None:
    """数值归一：非法/NaN/inf → None，绝不伪装成 0（与其余源同口径）。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if (f == f and f not in (float("inf"), float("-inf"))) else None


def parse_dt_utc(s: str) -> datetime:
    """接口的 "YYYY-MM-DD HH:MM:SS" → naive UTC datetime。

    带非零时区偏移的串整体错 8h，显式拒绝以防契约漂移被静默误读（同风乌 parse_iso_z）。
    """
    dt = datetime.fromisoformat(s.strip())
    if dt.tzinfo is not None:
        if dt.utcoffset() != timedelta(0):
            raise ValueError(f"EW4ALL 时间串带非零时区偏移，契约应为 UTC: {s!r}")
        return dt.replace(tzinfo=None)
    return dt  # noqa: DTZ005  契约即 UTC（见模块 docstring 第 2 条）


def to_bj(dt_utc: datetime) -> datetime:
    """UTC naive → 北京时 naive 墙钟。"""
    return dt_utc + timedelta(hours=8)


def parse_run_label(s: str) -> datetime:
    """起报标签 "YYYYMMDDHH0000" → naive UTC datetime。"""
    s = s.strip()
    if len(s) != _RUN_LABEL_LEN or not s.isdigit():
        raise ValueError(f"EW4ALL 起报标签格式异常: {s!r}")
    return datetime.strptime(s[:10], "%Y%m%d%H")


def hourly_axis(first: datetime, last: datetime) -> list[datetime]:
    """[first, last] 的逐小时整点轴（含两端）。"""
    n = int((last - first).total_seconds() // 3600)
    return [first + timedelta(hours=i) for i in range(n + 1)]


def _has_values(rows: list[tuple[datetime, float | None]]) -> bool:
    """序列里至少有一个非缺测值。

    只有"行数 > 0"是不够的：契约漂移或占位期会返回一批字段缺失/NaN 的行
    （`data` 非空但值全为 null）。若把这种轮次当成"要素已就绪"采纳，就会落下一份
    全缺测的快照并被幂等键永久锁死，而**真实的、更早的完整轮次反被丢弃**。
    """
    return any(v is not None for _, v in rows)


def build_series(tem: list[tuple[datetime, float | None]],
                 prc_1h: list[tuple[datetime, float | None]],
                 prc_accum: list[tuple[datetime, float | None]],
                 accum_window_hours: int) \
        -> tuple[list[datetime], list, list, str | None]:
    """把各要素的 UTC 采样序列展开为统一的逐小时（北京时）四元组。

    - 温度：线性插值到逐小时（resample.interpolate_hourly）；首末采样之外为 None。
    - 降水：**1 小时产品优先**——在 `prc_1h` 实际有值的整点上直接透传（原生
      "前 1 小时累计"，与观测 rain@t 同口径，无需展开）；其余小时用 `prc_accum`
      的后向累计窗口平铺均摊 /w（resample.spread_accumulation）补齐。两者都未覆盖
      的小时为 None。

    返回 (轴, 温度, 降水, 1 小时产品的最后一个整点)；传入序列均为 UTC naive，
    返回的时刻为北京时。温度序列为空时由调用方先拦截。
    """
    tem_bj = [(to_bj(t), v) for t, v in tem]
    one_bj = dict((to_bj(t), v) for t, v in prc_1h)
    acc_bj = [(to_bj(t), v) for t, v in prc_accum]
    points = tem_bj + list(one_bj.items()) + acc_bj
    if not points:
        return [], [], [], None
    axis = hourly_axis(min(t for t, _ in points), max(t for t, _ in points))
    interp = dict(interpolate_hourly(tem_bj))
    temps = [interp.get(h) for h in axis]
    # 兜底轨道：全时效的累计量平铺均摊
    precips = spread_accumulation(acc_bj, axis, accum_window_hours) if acc_bj \
        else [None] * len(axis)
    # 1 小时产品覆盖到的整点直接透传（优先，精度更高）；覆盖边界之外保持摊薄值
    # 或 None——两条轨道给出的都是"前 1 小时累计"，配对口径一致，故可逐点择一。
    one_hours = {h: v for h, v in one_bj.items() if v is not None}
    for i, h in enumerate(axis):
        if h in one_hours:
            precips[i] = one_hours[h]
    return axis, temps, precips, (max(one_hours) if one_hours else None)


class Ew4allProvider(ForecastProvider):
    """EW4ALL 快照器：**按模型返回独立快照列表**。

    两个模型的起报轮次发布进度独立（实测风清的最新轮次常晚于 CMA-NDFS 出数），
    故各自解析起报锚点并各返回一份快照（各自 issue_iso 与时间轴），保证时效（lead）
    分组不被跨模式错位污染——与中科天机同属 base.py 契约的第 2 种返回形态。
    """

    def __init__(self, timeout: int | tuple = (10, 60), retries: int = 3,
                 session: requests.Session | None = None):
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        # 起报轮次列表与最终锚点是"模式级"属性，跨站点复用（4 站只探测 1 次）；
        # _probe_cache 留着探测那一站的原始行，供其落盘时直接复用
        self._runs_cache: dict[str, list[datetime]] = {}
        self._issue_cache: dict[str, datetime] = {}
        self._probe_cache: dict[tuple[str, str], tuple[list, list]] = {}

    # ------------------------------------------------------------------ 对外
    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> list[dict]:
        wanted = list(models) if models else list(MODEL_SPECS)
        unknown = [m for m in wanted if m not in MODEL_SPECS]
        if unknown:
            raise RuntimeError(
                f"EW4ALL 未登记模型 {unknown}（可选：{sorted(MODEL_SPECS)}）")
        return [self._build_snapshot(station, MODEL_SPECS[m], self._resolve_run(
            MODEL_SPECS[m], station)) for m in wanted]

    # ------------------------------------------------------------------ 内部
    def _list_runs(self, spec: ModelSpec) -> list[datetime]:
        """该模式的可用起报轮次（新→旧）。"""
        payload = self._request(MODEL_TIME_LIST_URL, params={
            "data_type": spec.mode, "element": TEM_ELEMENT})
        data = _check_payload(payload, f"起报时次列表 {spec.mode}")
        runs: list[datetime] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                runs.append(parse_run_label(str(item.get("data_time"))))
            except (TypeError, ValueError):
                continue
        if not runs:
            raise RuntimeError(
                f"EW4ALL 模式 {spec.mode} 的起报时次列表为空，契约可能已变化")
        return sorted(set(runs), reverse=True)

    def _resolve_run(self, spec: ModelSpec, probe_station: Any) -> datetime:
        """最新**要素齐全**的起报轮次：由新到旧回退探测。

        该源的轮次内部是**分要素先后出数**的：实测标注轮次 12Z 的 GDFS5KM 在其后的
        一两个小时内只出温度（`TEM` 有 128 点），降水要素 `HOURTPE`/`SIXTPE` 仍为空；
        下一轮（00Z）则两者齐备。故候选轮次必须**温度与兜底累计降水都有非缺测值**才算
        可用（只看"行数非空"不够：占位期会返回一批值全为 null 的行，见 `_has_values`）；
        1 小时降水只覆盖前 ~72h，缺了只降精度、不否决整轮。
        残缺快照一旦落盘，会被同 issue 幂等键永久锁死，该轮降水此后再无机会入档
        （本来源每日仅 2 轮，丢一轮即丢当日一半降水样本）。跳过最新轮次的代价只是
        "同一批有效时刻改由相邻轮次以更长时效覆盖"，属可接受的已知边界，
        且下一次运行仍可正常捕获该轮次（届时其降水已出数）。
        """
        cached = self._issue_cache.get(spec.model)
        if cached is not None:
            return cached
        runs = self._runs_cache.get(spec.model)
        if runs is None:
            runs = self._list_runs(spec)
            self._runs_cache[spec.model] = runs
        candidates = runs[:MAX_ISSUE_FALLBACK]
        for run in candidates:
            tem = self._fetch_series(probe_station, spec.mode, TEM_ELEMENT, run)
            if not _has_values(tem):
                logger.warning("EW4ALL %s 起报轮次 %s 温度无有效值（%d 行），尝试更早轮次",
                               spec.model, _fmt(run), len(tem))
                continue
            prc = self._fetch_series(probe_station, spec.mode, spec.precip_element, run)
            if not _has_values(prc):
                logger.warning(
                    "EW4ALL %s 起报轮次 %s 温度已出、降水要素 %s 尚未出数（%d 行），"
                    "跳过本轮（残缺快照落盘即被幂等锁死，该轮降水将永久缺失）",
                    spec.model, _fmt(run), spec.precip_element, len(prc))
                continue
            # 1 小时降水是"锦上添花"而非必需：只覆盖前 ~72h，缺了不该否决整轮
            one = self._fetch_series(probe_station, spec.mode, spec.precip_1h_element, run) \
                if spec.precip_1h_element else []
            if spec.precip_1h_element and not _has_values(one):
                logger.warning(
                    "EW4ALL %s 起报轮次 %s 的 1 小时降水 %s 无有效值，本轮降水全程用 "
                    "%s/%dh 平铺均摊兜底（精度略降，不影响成档）",
                    spec.model, _fmt(run), spec.precip_1h_element,
                    spec.precip_element, spec.precip_window_hours)
            if run != runs[0]:
                logger.info("EW4ALL %s 最新轮次 %s 要素不全，回退使用 %s",
                            spec.model, _fmt(runs[0]), _fmt(run))
            self._issue_cache[spec.model] = run
            # 探测结果顺带留给本站点的落盘复用（省最多 3 次请求，也避免"探测与落盘
            # 之间服务端状态变化"导致的竞态）
            self._probe_cache[(spec.model, probe_station.id)] = (tem, prc, one)
            return run
        raise RuntimeError(
            f"EW4ALL {spec.model} 最近 {len(candidates)} 个起报轮次（"
            f"{_fmt(candidates[-1])} ~ {_fmt(candidates[0])}）中，没有任何一轮同时"
            f"具备有效的温度 {TEM_ELEMENT} 与降水 {spec.precip_element}"
            "（契约可能已变化：要素代码或轮次发布节奏）")

    def _fetch_series(self, station: Any, mode: str, element: str,
                      run: datetime) -> list[tuple[datetime, float | None]]:
        """单点单要素序列（UTC 时刻升序）。空列表 = 该轮次/该要素无数据。"""
        payload = self._request(FIND_BY_POINT_URL, json_body={
            "mode": mode,
            "elements": element,
            "point": [[float(station.lon), float(station.lat)]],
            "projection": 4326,
            "dataTime": run.strftime("%Y%m%d%H"),
            "level": 0,
        })
        data = _check_payload(payload, f"单点要素 {mode}/{element}")
        rows: list[tuple[datetime, float | None]] = []
        missing_key = 0
        for item in data:
            if not isinstance(item, dict):
                continue
            try:
                t = parse_dt_utc(str(item.get("Datetime")))
            except (TypeError, ValueError):
                continue
            if element not in item:
                missing_key += 1
            rows.append((t, _num(item.get(element))))
        if missing_key:
            logger.warning("EW4ALL 响应有 %d 条缺少要素字段 %s（契约可能已变化）",
                           missing_key, element)
        rows.sort(key=lambda r: r[0])
        return rows

    def _build_snapshot(self, station: Any, spec: ModelSpec, run: datetime) -> dict:
        # 探测那一站的原始行直接复用（省最多 3 次请求，且消除"探测与落盘之间服务端
        # 状态变化"的竞态）；其余站点照常抓取
        cached = self._probe_cache.pop((spec.model, station.id), None)
        if cached is not None:
            tem, prc, one = cached
        else:
            tem = self._fetch_series(station, spec.mode, TEM_ELEMENT, run)
            prc = self._fetch_opt(station, spec.mode, spec.precip_element, run)
            one = self._fetch_opt(station, spec.mode, spec.precip_1h_element, run) \
                if spec.precip_1h_element else []

        issue_bj = to_bj(run)
        axis, temps, precips, one_until = build_series(
            tem, one, prc, spec.precip_window_hours)
        if not axis:
            raise RuntimeError(
                f"EW4ALL {spec.model} 站点 {station.id} 解析出 0 个有效逐小时点")
        # 空快照一旦入库会被同 issue 幂等锁死，正常数据永远进不来（同伏羲）。
        # "空"以"没有任何要素有值"为准：只要一侧有值就仍有存档价值（另一侧按缺测）。
        if all(v is None for v in temps) and all(v is None for v in precips):
            raise RuntimeError(
                f"EW4ALL {spec.model} 站点 {station.id} 起报 {_fmt(run)} 的温度与降水"
                "均无有效值，拒绝入库空快照")
        self._warn_health(station, spec, tem, prc, one, axis, temps, precips)

        # 口径留档：便于日后复核时不必再逆向接口
        expansion = [f"temperature: linear-interp-1h",
                     f"precipitation: {spec.precip_element}/{spec.precip_window_hours}h "
                     f"spread over (t-{spec.precip_window_hours}h, t]"]
        if spec.precip_1h_element:
            expansion.append(f"precipitation<= {spec.precip_1h_element} passthrough")
        snapshot = {
            "issue_iso": issue_bj.strftime("%Y-%m-%dT%H:00"),
            "station_id": station.id,
            "source": SOURCE,
            "models": [spec.model],
            # 接口不回显吸附格点（同伏羲中期）；请求坐标即代表点
            "grid_lat": float(station.lat),
            "grid_lon": float(station.lon),
            "elevation": None,
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            "api_mode": spec.mode,
            "run_label_utc": run.strftime("%Y%m%d%H"),
            # 1 小时降水实际覆盖到的最后一个整点（北京时）：口径切换点的机器可读留档，
            # 便于日后核对"哪一段是原生 1 小时产品、哪一段是 3/6 小时摊薄"
            "precip_1h_until": one_until.strftime("%Y-%m-%dT%H:00") if one_until else None,
            "expansion": "; ".join(expansion),
            "hourly_time": [h.strftime("%Y-%m-%dT%H:00") for h in axis],
            "data": {spec.model: {
                "temperature_2m": temps,
                "precipitation": precips,
            }},
        }
        logger.info("站点 %s 已抓取 EW4ALL %s 起报 %s（UTC %s），逐小时点 %d（1 小时降水覆盖至 %s）",
                    station.id, spec.model, snapshot["issue_iso"], _fmt(run), len(axis),
                    snapshot["precip_1h_until"] or "—")
        return snapshot

    def _fetch_opt(self, station: Any, mode: str, element: str | None,
                   run: datetime) -> list[tuple[datetime, float | None]]:
        """可选要素：抓取失败只降级为"该要素缺测"，绝不拖垮整份快照。

        降水（含 1 小时产品）抓取失败不得否决快照——温度是主干，快照错过起报
        即无法追补。
        """
        if not element:
            return []
        try:
            return self._fetch_series(station, mode, element, run)
        except RuntimeError as e:
            logger.warning("EW4ALL %s 站点 %s 要素 %s 抓取失败，该项计为缺测: %s",
                           mode, station.id, element, e)
            return []

    @staticmethod
    def _warn_health(station: Any, spec: ModelSpec, tem, prc, one, axis, temps,
                     precips) -> None:
        """缺测/截断/契约漂移告警：把静默的退化变成可见的日志。

        失败只影响该源、不阻断其他源，但**必须可见**——本源的失效形态多为"少了一部分
        数据"而不是报错，不告警就会在榜单上表现成"技巧变差"。
        """
        if len(axis) < spec.expected_points:
            logger.warning(
                "EW4ALL %s 站点 %s 逐小时轴仅 %d 点（预期 %d），时效可能被截断",
                spec.model, station.id, len(axis), spec.expected_points)
        if all(v is None for v in temps):
            logger.warning("EW4ALL %s 站点 %s 温度序列全部缺测，服务端契约可能已变化",
                           spec.model, station.id)
        if not _has_values(prc):
            logger.warning(
                "EW4ALL %s 站点 %s 降水要素 %s 无有效值，本快照降水计为缺测"
                "（按缺测处理，绝不折算 0）", spec.model, station.id, spec.precip_element)
        elif all(v is None for v in precips):
            logger.warning(
                "EW4ALL %s 站点 %s 降水采样有值但平铺后全为缺测（窗口相位可能失配）",
                spec.model, station.id)
        if any(v is not None and v < 0 for v in precips):
            logger.warning("EW4ALL %s 站点 %s 降水出现负值，契约可能已变化",
                           spec.model, station.id)
        # ── 1 小时降水覆盖度（CMA-NDFS 短时效降水的首选产品）──
        if spec.precip_1h_element:
            one_ok = sum(1 for t, v in one if v is not None)
            if one_ok == 0:
                logger.warning(
                    "EW4ALL %s 站点 %s 的 1 小时降水 %s 无有效值，全时效退化为 "
                    "%s/%dh 平铺均摊（短时效降水精度下降，属降级不是错）",
                    spec.model, station.id, spec.precip_1h_element,
                    spec.precip_element, spec.precip_window_hours)
            else:
                # 覆盖时长明显短于实测规格（约 72h）时告警：交接点会因此提前，
                # 更多时效落入摊薄轨道——静默缩短会被误读成"短时效技巧变差"。
                span = int((max(t for t, _ in one) - min(t for t, _ in one)).total_seconds() // 3600)
                if span + 1 < ONE_TPE_EXPECTED_HOURS:
                    logger.warning(
                        "EW4ALL %s 站点 %s 的 1 小时降水 %s 仅覆盖 %d 小时（预期约 %d），"
                        "更多时效将由 %dh 摊薄值兜底",
                        spec.model, station.id, spec.precip_1h_element, span + 1,
                        ONE_TPE_EXPECTED_HOURS, spec.precip_window_hours)
        # ── 平铺窗口完整性哨兵 ──
        # 端点间距恒为窗口长度是"平铺恰好覆盖时间轴"的前提。某个采样点缺失会让
        # 间距变成 2×窗口（该窗口的总量无从得知，其覆盖的 1~w 个小时整体为缺测，
        # 若该处同时被 1 小时产品覆盖则不受影响）——这是正确的缺测语义，
        # 但静默少掉几个小时降水会被误读成"模式少报"。
        ends = tile_endpoints(prc, spec.precip_window_hours)
        if len(ends) >= 3:
            gaps = {int((b - a).total_seconds() // 3600) for a, b in zip(ends, ends[1:])}
            if gaps != {spec.precip_window_hours}:
                logger.warning(
                    "EW4ALL %s 站点 %s 降水平铺端点间距异常 %s（预期恒为 %d 小时），"
                    "存在缺失窗口，未被 1 小时产品覆盖的小时将按缺测处理",
                    spec.model, station.id, sorted(gaps), spec.precip_window_hours)

    def _request(self, url: str, *, params: dict | None = None,
                 json_body: dict | None = None) -> Any:
        """熔断/退避统一走共享助手（4xx 立即失败、429/5xx/网络错误退避重试）。"""
        def _classify(resp: Any) -> tuple[str, Any]:
            status = getattr(resp, "status_code", None)
            if status == 200:
                try:
                    return "return", resp.json()
                except Exception as e:  # noqa: BLE001  200 但非 JSON 属确定性失败
                    raise RuntimeError(f"EW4ALL 响应非 JSON: {e}") from e
            try:
                digest = (resp.text or "")[:200]
            except Exception:  # noqa: BLE001
                digest = ""
            if isinstance(status, int) and 400 <= status < 500 and status != 429:
                return "fatal", RuntimeError(
                    f"EW4ALL 请求被拒: HTTP {status} body={digest!r}")
            return "retry", f"HTTP {status} body={digest!r}"

        return request_with_retries(
            self.session, url,
            method="POST" if json_body is not None else "GET",
            params=params, json_body=json_body, headers=HEADERS,
            timeout=self.timeout, retries=self.retries,
            source="EW4ALL", classify=_classify,
        )


def _check_payload(payload: Any, context: str) -> list:
    """统一校验响应外壳，返回 data 列表。

    该源的失败分两种：HTTP 层（4xx/5xx，由 request_with_retries 处理）与业务层
    （HTTP 200 但 `code != 200`，或 data 缺失/为空）。业务层必须显式上抛：
    "空 data" 既是"该轮次尚未出数"也是"mode/要素名写错"的唯一形态，
    静默当成缺测会把契约漂移伪装成正常数据。
    """
    if not isinstance(payload, dict):
        raise RuntimeError(f"EW4ALL {context} 响应结构异常: {str(payload)[:200]!r}")
    code = payload.get("code")
    if str(code) != "200":
        msg = payload.get("message") or payload.get("msg") or ""
        raise RuntimeError(f"EW4ALL {context} 返回错误: code={code} msg={msg!r}")
    data = payload.get("data")
    if data is None:
        raise RuntimeError(f"EW4ALL {context} 响应缺少 data 字段（契约可能已变化）")
    if not isinstance(data, list):
        raise RuntimeError(f"EW4ALL {context} 的 data 不是列表（契约可能已变化）")
    return data


def _fmt(dt: datetime) -> str:
    """起报时刻的日志表示（UTC，接口原生时钟）。"""
    return dt.strftime("%Y-%m-%dT%H:%MZ")

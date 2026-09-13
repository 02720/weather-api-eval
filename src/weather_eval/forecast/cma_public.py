"""中国气象局公众气象服务网（weather.cma.cn）逐小时预报快照器。

数据源：中国气象局对外公众服务站点 `weather.cma.cn` 的**公开 JSON 接口**
（游客态可用、无需凭据；接口契约来自 2026-09-13 线上实测，4 站 × 各自 53 个整点
逐点核对）。站点以 **WMO 站号**寻址（梧州 59265、博白 59449、平南 59255、万宁 59951）。

  GET https://weather.cma.cn/api/hourly/{station_id}
      Header: User-Agent（见契约第 1 条，必需）
  响应:
  {"msg":"success","code":0,
   "data":[                                   # 7 个"日块"，块间有重叠，见契约第 4 条
     {"date":"2026/09/13","list":[
        {"stationid":"59265",
         "publishTime":"2026/09/13 12:00",    # 产品循环的名义标签（整份响应内恒定；
                                              # **不是**起报锚点，见契约第 5 条）
         "forecastTime":"2026/09/13 17:00",   # 有效时刻（北京时，见契约第 3 条）
         "hour":9,                            # 预报时效（小时，见契约第 5 条）
         "temperature":28.5, "humidity":80.3,
         "precipitation":0.1,                 # 3 小时累计（见契约第 6 条）
         "weather":3, "text":"阵雨", ...},
        ...]},
     ... 共 7 块 ...
   ]}

─────────────────────────────────────────────────────────────────────
关键契约（2026-09-13 线上实测；非官方契约，改版即可能失效，改动前必须复核）
─────────────────────────────────────────────────────────────────────
1. ── User-Agent 有**反爬名单**，缺省 UA 会被拒 ──
   实测（同一 URL、同一时刻）：
     `weather-api-eval/0.1 (+https://github.com/)` → 200
     `Mozilla/5.0` → 200 ；`Mozilla/5.0 (…Chrome/120…)` → 200
     `python-requests/2.34.2` → **403** ；`curl/8.5.0` → **403** ；空 UA → 200
   即服务端按**已知爬虫 UA 名单**拒绝，而非"必须有 UA"。本源沿用仓库统一 UA
   （实测在名单外），并把 403 的错误信息写成"UA 可能落入反爬名单"——
   这是确定性失败（4xx），`request_with_retries` 会立即熔断、不烧退避。
2. ── 失效形态是"HTTP 200 + 空 data"而非报错 ──
   实测站号写成不存在的 `api/hourly/99999` 返回
   `{"msg":"success","code":0,"data":""}`（HTTP 200、业务码也"成功"）。
   故**空数据绝不建快照**：空快照会被同 issue 幂等键永久锁死、正常数据此后再无
   机会入档；契约漂移（字段改名/结构变化）也会以同样形态出现，必须响亮失败。
3. ── 时间语义：接口原生就是**北京时**（非 UTC）──
   `forecastTime` 形如 `2026/09/13 17:00`，无时区标注。判据是**日变化相位**：
   4 站实测温度峰值恒在 14 时、谷值恒在 05 时（如梧州 09/14 的 05/14 时为
   24.2/30.8℃）。按北京时解释恰为"午后最高、清晨最低"；按 UTC 解释则变成
   北京时 22 时最高、13 时最低，与物理不符。
   格式解析**严格匹配** `YYYY/MM/DD HH:MM`：接口若改回 ISO/带偏移形态（如 EW4ALL
   的 UTC 串），这里必须响亮失败——时区错 8 小时不会报任何错，只会让该源的
   全部样本静默归零（评估按北京时整点精确配对，错则样本数为 0）。
4. ── 网格是**严格 3 小时**（`api/hourly/` 是接口名而非分辨率）──
   实测 4 站全部 53 个整点：相邻间隔恒为 3 小时，时刻恒落在北京时
   02/05/08/11/14/17/20/23。响应被切成 7 个 `date` 日块，**块间有重叠**
   （如首块覆盖 17:00→次日 14:00，次块 08:00→次日 05:00，重叠 3 点），
   重叠点的数值实测逐点一致（4 站 0 冲突）——按 `forecastTime` 去重即可。
   实现仍保留"重复时刻数值不一致即告警"的哨兵：那意味着同一份响应里混了两个
   数据版本，绝不可静默取首见。
5. ── 起报锚点：由接口自述的 `hour` 反解；`publishTime` **不可**当锚点 ──
   `hour` 是该点的预报时效（小时）。实测 `forecastTime − hour` 在同一份响应内恒为
   同一常量（两个产品循环 × 4 站 × 53 点全部命中），故

       issue := forecastTime − hour

   这是**接口自述量之间的恒等式**，不依赖任何硬编码偏移——产品循环切换时
   issue 自动跟随。
   ── 为什么不用 `publishTime`（2026-09-13 实测反例，改动前必读）──
   连续两次抓取（相隔 16 分钟）观察到产品换循环：
     | | publishTime | hour 反解基准 | 首个有效时刻 | 末个有效时刻 |
     |---|---|---|---|---|
     | 循环 A | `2026/09/13 12:00` | 09-13 08:00 | 09-13 17:00 | 09-20 05:00 |
     | 循环 B | `2026/09/13 20:00` | 09-13 20:00 | 09-13 23:00 | 09-20 05:00 |
   `publishTime` 前进 8h、`hour` 基准前进 12h（恰为一轮 00Z→12Z），即
   **`publishTime − 起报基准` 不是常量**（先 4h、后 0h）：`publishTime` 是该站产品的
   **名义循环标签**（循环 B 的标签 20:00 甚至晚于抓取时刻 16:44），把它当锚点会让
   同一批有效时刻的 lead 系统性偏移数小时，而 lead 正是「提前 N 天」分桶的唯一依据。
   基准取 20:00（晚于抓取时刻）不是矛盾：本仓库 `forecast/ew4all.py` 已实证中国气象局
   平台会**在标注轮次到点之前就返回该轮数据**（实测 10:55 UTC 已可取标注 12Z 的序列），
   故"标签为名义轮次、数据提前可用"是本机构的既有发布形态，与存档先行原则一致
   （快照一经落盘不再回改）。
   哨兵：若各点的 `forecastTime − hour` 不是同一常量（`hour` 语义漂移），告警并退回
   `publishTime` 整点锚定（若 `publishTime` 晚于序列首点则再退回首点），同时把
   `issue_source` 写进快照 meta 供事后审计——绝不静默用错锚点。
6. ── 降水口径：`precipitation` 是**后向 3 小时累计**（窗口 (t−3h, t]）──
   接口未逐字声明窗口长度与方向，判定以**逐窗同源对照**为首要证据
   （`scripts/calibrate_cma_public_precip.py`，可复现）：
     a) **决定性命中——同源逐窗量比**：对 4 站 196 个逐窗样本，把本源
        `precipitation(t)` 与 CMA-NDFS（同为中国气象局产品，见 forecast/ew4all.py）
        的**逐小时**降水序列对齐求池化和：
        Σ本源 = 241.6mm，ΣNDFS-1h = 81.9mm，**比值 2.95**。若本源是 1 小时量，
        该比值应 ≈1；实测量比恰为 3，即本源一个采样点的量 ≈ 同源 1 小时量的 3 倍。
        同一批样本上，本源与 NDFS **3 小时**后向累计的和比为 **1.01**、r=0.68——
        两个中国气象局产品在同一场雨上量级几乎重合。
     b) 网格严格 3 小时（第 4 条）——若为 1 小时量被 3 小时采样，等于放弃 2/3 的
        降水信息，不符合该产品的输出约定。
     c) 窗口总量对照（**弱证据**，仅作旁证）：与 CMA-NDFS 的窗口总量比
        0.53~0.80（两轮产品循环的合计值；分站 0.25~0.64），按 1 小时量解释应为
        ≈0.33；跨源分位下 3h 解释亦优于 1h 解释（四站 15/8/40/8% 对 4/0/12/0%）。
        之所以弱：两个产品的起报轮次与时效并不相同，总量方差大，判别力只有 1.6 倍
        （而 a) 的判别力是 3 倍）。
     d) NWP 惯例：3 小时输出的单点要素产品按"输出间隔内的累计量"发布（WMO 通行定义）。
   入库走 `resample.spread_accumulation`（窗口 3），即对累计曲线线性插值后逐小时
   差分、数学上等价于按窗口均摊 /3，是唯一保证"任意完整跨度求和 == 原始累计总量"
   的确定性展开（与风乌 tp6h、EW4ALL 的 3h/6h 族同法，共用一个实现）。
   **窗口方向的鉴别力有限**：相位扫描（后向/居中/前向）的 r 为 0.677/0.686/0.698，
   三者几乎不可分（雨场自相关强、且都是同一场雨的 3 小时总量），故按国内业务惯例与
   观测 `rain@t`=(t−1h, t] 的对齐方式取**后向**；待满月样本后按 README「降水口径」的
   −1/0/+1h 平移标定法复核（同彩云/和风/AccuWeather 的处置原则）。
   代价是短时强降水被摊薄（对 0.1mm 晴雨阈值偏保守），属已知局限。
7. ── 缺测哨兵值 `999.9`（降水）/ `999`（天气码）/ `"9999"`（天气现象文本）──
   长时效点上（实测梧州 09/18 起）降水回填 `999.9`、天气码回填 `999`、文本回填
   `"9999"`，而**同一行的温度仍是有效值**（30.5/32.3/…）。误把 999.9 当毫米入库
   会单条摧毁降水评分，故降水按"**非物理值即缺测**"处理：`v ≥ 999` 或 `v < 0`
   一律置 None 并计数告警（真实 3 小时降水不可能达 999mm，负降水物理上不存在）。
   温度同理设 ±100℃ 的物理上界（本源实测未出现，属防御性哨兵）。
8. ── 无逐日预报块（拒绝接入，2026-09-13 留档）──
   姊妹端点 `/api/weather/view?stationid=` 确有 7 天 `daily[]`（`high`/`low`），
   但三条理由否决接入：
     a) **日界不是北京时自然日**。24 个样本量化对照：`high` 与自然日 3 小时采样
        极值吻合（|Δ| 均值 0.28℃、最大 0.70℃），而 `low` 与自然日 min 的
        |Δ| 均值 0.55℃、最大 **2.20℃**；改按"当天白天 + 当夜（20:00→次日 08:00）"
        口径则降到 0.41℃ / 1.00℃。即 `low` 用的是中国天气网"白天/夜间"日界，
        与 `base.py` 契约要求的"北京时自然日"不符（README 已明令此类日界不得照抄）。
     b) **无定量降水**：`daily[]` 只有 `dayText`/`nightText` 与天气码，没有累计
        降水量字段——按 base.py 契约不得由天气码折算 mm。
     c) **零时效增益**：逐日同样是 7 天，与逐小时等长，接进来延长不了任何时效。
   逐小时轴已插值/平铺到逐小时，按天轨道由逐小时聚合覆盖全时效，无需补位。
9. ── 不做格点回显（同 EW4ALL/伏羲中期）──
   逐小时响应不含吸附后的格点经纬度，`grid_lat/lon` 记请求坐标；底层为该站的
   单点（站点）预报产品，代表点误差计入已知限制。
10. ── 时效与视野 ──
   实测任何时刻可见 **7 天**，覆盖「提前 N 天」天桶 1..7，短于配置的
   `hourly_lead_days=16`——总榜"覆盖时效"列会自动显示 7d，跨源比较时需计入该差异
   （README"总榜"一节已披露的原理性混杂）。
   视野由**产品循环**决定而非抓取时刻：同一循环内响应稳定（实测同循环两次抓取
   相隔 7 分钟，53 个整点逐点数值完全一致），换循环时首个有效时刻随之推进
   （循环 A 首个有效时刻 17:00 → 循环 B 23:00）。可见覆盖约 6.4~6.9 天
   （循环 A 最大 lead 165h、循环 B 153h）。
   幂等语义因此干净：issue 随循环推进，同一循环内重复抓取被
   （站 × 模型 × 起报）幂等键跳过——与"存档先行、快照一经落盘不再回改"一致，
   且同一循环内响应不变，跳过不会损失任何样本。
11. ── 一次请求一份快照，无分页、无分片 ──
   与 MSN（10 次分片，需版本漂移熔断）不同，本源一次请求即取回全部时效，
   不存在跨请求的版本混合风险；版本一致性由第 4 条的重复点冲突哨兵覆盖。
12. ── 采样缺口的行为（共享助手语义，披露而非隐藏）──
   若相邻采样间隔不是 3 小时（缺一个采样点），`_warn_health` 会告警。此时：
   温度走 `resample.interpolate_hourly`，会把缺口**两端之间的逐小时线性桥接**
   （该函数只对"端点缺测"停插值，不对"网格缺口"停插值——与风乌/EW4ALL 同款
   共享行为）；降水走 `spread_accumulation`，缺口覆盖的小时因无平铺窗口而
   一律为 None。即缺口对两条轨道的影响不对称，判读该时段的温度指标时需知情。
   实测本源网格严格 3 小时（两个循环 × 4 站 × 53 点无例外），该分支属防御性。

已知风险：无官方契约的公开接口，页面/接口改版即可能失效（失败只影响该源，
不阻断其他源）；失效形态为"空 data"或"字段改名致解析为 0 点"，两者都已在实现里
转为响亮失败，绝不落空快照。
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any

import requests

from .base import ForecastProvider
from .http import request_with_retries
from .resample import interpolate_hourly, spread_accumulation
from ..timeutil import floor_to_hour

logger = logging.getLogger(__name__)

BASE_URL = "https://weather.cma.cn/api/hourly"
# 仓库统一 UA：实测不在该站的反爬名单内（名单拒绝 python-requests / curl，见 docstring 第 1 条）
HEADERS = {"User-Agent": "weather-api-eval/0.1 (+https://github.com/)"}
SOURCE = "cma_public"
MODEL_NAME = "cma_public_v1"

# 网格步长与降水窗口长度：接口名是 api/hourly 但实测分辨率严格为 3 小时（docstring 第 4/6 条）
GRID_STEP_HOURS = 3
PRECIP_WINDOW_HOURS = 3

# 非物理值上界：降水/天气码的缺测占位实测为 999.9 / 999（docstring 第 7 条）
SENTINEL_MIN = 999.0
# 地面气温的物理上界（本源实测未触发，防御性哨兵）
TEMP_ABS_MAX = 100.0

# 严格格式：接口原生北京时、无时区标注；形态一变即失败（docstring 第 3 条）
_RE_TIME = re.compile(r"^(\d{4})/(\d{1,2})/(\d{1,2}) (\d{1,2}):(\d{2})$")

# 视野健康度下限（小时）：实测约 165h，明显短于此说明长时效被截断
_MIN_EXPECTED_MAX_LEAD_HOURS = 144


class CmaPublicPayloadError(RuntimeError):
    """响应外壳/业务码/数据为空（契约漂移或站号无效）——确定性失败，重试无意义。"""


def parse_dt_bj(s: Any) -> datetime:
    """接口时间串 `YYYY/MM/DD HH:MM` → 北京时 naive datetime。

    严格匹配形态并拒绝任何带偏移的串（docstring 第 3 条）：把 UTC 串当北京时读
    会整体错 8 小时且不报任何错，只会让该源样本静默归零，故宁可响亮失败。
    """
    m = _RE_TIME.match(str(s).strip()) if s is not None else None
    if not m:
        raise ValueError(
            f"CMA 公众网时间串格式异常（契约应为 'YYYY/MM/DD HH:MM' 北京时）: {s!r}")
    return datetime(*map(int, m.groups()))


def _num(v: Any) -> float | None:
    """数值归一：非法/bool/NaN/inf → None，绝不把缺测伪装成 0。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _precip(v: Any, stats: dict[str, int]) -> float | None:
    """降水归一：非物理值（≥999 的占位、负值）一律置 None 并计数。

    `999.9` 是该源长时效的缺测占位（docstring 第 7 条），误当毫米入库会单条
    摧毁降水评分；负降水物理上不存在，也按缺测处理。
    """
    f = _num(v)
    if f is None:
        return None
    if f >= SENTINEL_MIN:
        stats["sentinel"] = stats.get("sentinel", 0) + 1
        return None
    if f < 0:
        stats["negative"] = stats.get("negative", 0) + 1
        return None
    return f


def _temp(v: Any, stats: dict[str, int]) -> float | None:
    """温度归一：超出 ±100℃ 物理上界视作占位/契约漂移。"""
    f = _num(v)
    if f is None:
        return None
    if abs(f) > TEMP_ABS_MAX:
        stats["temp_implausible"] = stats.get("temp_implausible", 0) + 1
        return None
    return f


def extract_points(payload: Any) -> tuple[list[tuple[datetime, dict]], dict[str, Any]]:
    """校验外壳并展开全部日块 → [(北京时整点, 条目)]（升序、按时刻去重）。

    去重口径：重叠日块的同刻条目实测数值一致，取首见；一旦不一致则记入
    `conflicts`（同一份响应里混了两个数据版本，必须可见，绝不静默取首见）。
    返回 (点列, 元信息)；点列为空由调用方拒绝入库。
    """
    if not isinstance(payload, dict):
        raise CmaPublicPayloadError(f"响应不是 JSON 对象: {str(payload)[:200]!r}")
    code = payload.get("code")
    if code not in (0, "0"):
        raise CmaPublicPayloadError(
            f"业务码非 0: code={code!r} msg={payload.get('msg')!r}")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        # 站号无效/契约漂移都表现为这里（HTTP 200 + data 为 ""），必须响亮失败
        raise CmaPublicPayloadError(
            f"响应 data 为空或非列表（站号无效或接口契约已变化）: {str(data)[:120]!r}")

    points: dict[datetime, dict] = {}
    conflicts = 0
    bad_time = 0
    publish: set[str] = set()
    for blk in data:
        if not isinstance(blk, dict):
            continue
        for entry in blk.get("list") or []:
            if not isinstance(entry, dict):
                continue
            try:
                t = parse_dt_bj(entry.get("forecastTime"))
            except ValueError:
                bad_time += 1
                continue
            if entry.get("publishTime") is not None:
                publish.add(str(entry["publishTime"]))
            prev = points.get(t)
            if prev is not None:
                if (prev.get("temperature"), prev.get("precipitation"), prev.get("hour")) \
                        != (entry.get("temperature"), entry.get("precipitation"),
                            entry.get("hour")):
                    conflicts += 1
                continue
            points[t] = entry
    meta = {
        # publishTime 在整份响应内应只有一个取值；多于一个即说明响应跨了两个发布版本
        "publish_raw": publish.pop() if len(publish) == 1 else None,
        "publish_variants": len(publish),
        "conflicts": conflicts,
        "bad_time": bad_time,
    }
    return sorted(points.items()), meta


def derive_issue(points: list[tuple[datetime, dict]], publish_raw: str | None,
                 meta_notes: list[str]) -> tuple[datetime, str]:
    """解析起报锚点：优先用接口自述的 `hour` 反解，退化时回退并留档。

    候选顺序（前一个不可用才降级，每次降级都写进 `meta_notes` 供审计）：

    1. `hour_offset`：`forecastTime − hour` 的**多数**取值。正常契约下全部点给出
       同一常量（docstring 第 5 条），这是接口自己声明的时效零点，不依赖任何硬编码
       偏移；少数点 `hour` 异常时取多数仍然正确（比整体退回 publishTime 更准，
       后者会系统性平移 lead）。
    2. `publish_time`：候选 1 缺失、或出现**并列**的多个基准（无法判多数）、
       或候选 1 晚于序列首点（会产生负 lead，评估侧会静默丢弃这些样本）时使用。
       同样要求不晚于序列首点。
    3. `first_point`：以上都不可用时的兜底（lead 从 0 起算）。

    返回 (issue, issue_source)，source ∈ {"hour_offset", "publish_time", "first_point"}。
    """
    bases: dict[datetime, int] = {}
    missing_hour = 0
    for t, entry in points:
        h = _num(entry.get("hour"))
        if h is None or h != int(h):
            missing_hour += 1
            continue
        b = t - timedelta(hours=int(h))
        bases[b] = bases.get(b, 0) + 1
    if missing_hour:
        meta_notes.append(f"{missing_hour} 个点缺少可解析的 hour 字段")

    first_valid = points[0][0]
    # 排序键含时刻本身，保证并列时结果可复现（不依赖 dict 插入顺序）
    ranked = sorted(bases.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(ranked) > 1:
        meta_notes.append(
            "各点 forecastTime−hour 反解出 %d 个不同起报基准（%s 出现 %d 次、"
            "%s 出现 %d 次）——hour 语义可能已漂移"
            % (len(ranked), _fmt(ranked[0][0]), ranked[0][1],
               _fmt(ranked[1][0]), ranked[1][1]))
    if ranked:
        best, top = ranked[0]
        if best > first_valid:
            meta_notes.append(
                f"hour 反解出的起报基准 {_fmt(best)} 晚于序列首点 {_fmt(first_valid)}"
                "（会产生负 lead，评估侧会丢弃这些样本），不采用")
        elif len(ranked) == 1 or top > ranked[1][1]:
            # 单一基准，或存在严格多数——两者都足以信任
            return best, "hour_offset"
        # 并列且无多数：无法判哪个基准为真，降级到 publishTime

    if publish_raw is not None:
        try:
            published = floor_to_hour(parse_dt_bj(publish_raw))
        except ValueError:
            published = None
        if published is not None and published <= first_valid:
            meta_notes.append(f"起报锚点退化为 publishTime={_fmt(published)}")
            return published, "publish_time"
        if published is not None:
            meta_notes.append(
                f"publishTime={_fmt(published)} 晚于序列首点，不能作锚点")

    if ranked:
        meta_notes.append(f"退回 hour 反解的多数基准 {_fmt(ranked[0][0])}")
        return ranked[0][0], "hour_offset"

    meta_notes.append("缺 hour 与可用 publishTime，起报锚点退化为序列首点")
    return first_valid, "first_point"


def _fmt(dt: datetime) -> str:
    """审计用的时刻表示（北京时墙钟）。"""
    return dt.strftime("%Y-%m-%dT%H:%M")


class CmaPublicProvider(ForecastProvider):
    """中国气象局公众网逐小时预报快照器：无需凭据，单模型快照 dict。"""

    def __init__(self, timeout: int | tuple = (10, 30), retries: int = 3,
                 session: requests.Session | None = None):
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()

    # ------------------------------------------------------------------ 对外
    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> dict:
        station_code = getattr(station, "cma_id", None)
        if not station_code:
            raise RuntimeError(
                f"站点 {station.id} 未配置 cma_id（WMO 站号），无法抓取 CMA 公众网预报；"
                "请在 config/stations.yaml 的该站下补 `cma_id: <站号>`")
        payload = self._request(str(station_code))
        points, meta = extract_points(payload)
        if not points:
            raise RuntimeError(
                f"CMA 公众网站点 {station.id}（站号 {station_code}）解析出 0 个有效时刻，"
                "拒绝入库空快照（空快照会被同 issue 幂等键永久锁死）")

        notes: list[str] = []
        if meta["conflicts"]:
            notes.append(f"{meta['conflicts']} 个重叠时刻在不同日块间数值不一致（疑似混版本）")
        if meta["publish_variants"] > 1:
            notes.append(
                f"响应内出现 {meta['publish_variants']} 个不同的 publishTime（疑似跨版本）")
        if meta["bad_time"]:
            notes.append(f"{meta['bad_time']} 条记录的 forecastTime 无法解析")
        issue, issue_source = derive_issue(points, meta["publish_raw"], notes)

        stats: dict[str, int] = {}
        temp_samples = [(t, _temp(e.get("temperature"), stats)) for t, e in points]
        precip_samples = [(t, _precip(e.get("precipitation"), stats)) for t, e in points]

        axis = _hourly_axis(temp_samples[0][0], temp_samples[-1][0])
        temps = dict(interpolate_hourly(temp_samples))
        precips = spread_accumulation(precip_samples, axis, PRECIP_WINDOW_HOURS)
        temp_col = [temps.get(h) for h in axis]

        # 空快照不得落盘：只要温度与降水有一侧有值就仍有存档价值（另一侧按缺测）
        if all(v is None for v in temp_col) and all(v is None for v in precips):
            raise RuntimeError(
                f"CMA 公众网站点 {station.id} 起报 {issue:%Y-%m-%dT%H:%M} 的温度与降水"
                "均无有效值，拒绝入库空快照")

        self._warn_health(station, axis, temp_col, precips, temp_samples,
                          precip_samples, stats, notes, issue)

        snapshot = {
            "issue_iso": issue.strftime("%Y-%m-%dT%H:%M"),
            "station_id": station.id,
            "source": SOURCE,
            "models": [MODEL_NAME],
            # 接口不回显吸附格点（同 EW4ALL/伏羲中期）；请求坐标即代表点
            "grid_lat": float(station.lat),
            "grid_lon": float(station.lon),
            "elevation": None,
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            "station_code": str(station_code),          # WMO 站号（本源以站号寻址）
            "publish_time": meta["publish_raw"],
            # 起报锚点的来源（hour_offset / publish_time / first_point）与推导说明，
            # 供事后审计"这份快照的 lead 是按什么口径算出来的"
            "issue_source": issue_source,
            "issue_notes": notes,
            # publishTime 相对起报基准的小时差——**不是常量**（实测同一产品线的两次
            # 循环分别为 +4h 与 +0h，见契约第 5 条），仅作审计线索，绝不可反过来当锚点
            "publish_minus_issue_hours": (
                None if meta["publish_raw"] is None else
                _hours_between(issue, meta["publish_raw"])),
            "grid_step_hours": GRID_STEP_HOURS,
            "precip_interval_hours": PRECIP_WINDOW_HOURS,
            "expansion": (
                "temperature: linear-interp-1h; "
                "precipitation: 3h back-accumulation spread over (t-3h, t]"),
            # 缺测占位/非物理值的计数：静默少掉的降水与"模式少报"必须可区分
            "missing_sentinel_precip": stats.get("sentinel", 0),
            "negative_precip": stats.get("negative", 0),
            "implausible_temp": stats.get("temp_implausible", 0),
            "hourly_time": [h.strftime("%Y-%m-%dT%H:%M") for h in axis],
            "data": {MODEL_NAME: {
                "temperature_2m": temp_col,
                "precipitation": precips,
            }},
        }
        logger.info(
            "站点 %s 已抓取 CMA 公众网起报 %s（源=%s，%d 个逐小时点，3 小时采样 %d 点，"
            "站号 %s，发布 %s）", station.id, snapshot["issue_iso"], issue_source,
            len(axis), len(points), station_code, meta["publish_raw"])
        return snapshot

    # ------------------------------------------------------------------ 内部
    def _request(self, station_code: str) -> Any:
        """抓取并校验响应外壳（退避/熔断统一走共享助手）。"""
        url = f"{BASE_URL}/{station_code}"

        def _classify(resp: Any) -> tuple[str, Any]:
            status = getattr(resp, "status_code", None)
            if status == 200:
                try:
                    return "return", resp.json()
                except Exception as e:  # noqa: BLE001  200 但非 JSON 属确定性失败
                    raise CmaPublicPayloadError(f"CMA 公众网响应非 JSON: {e}") from e
            digest = ""
            try:
                digest = (resp.text or "")[:200]
            except Exception:  # noqa: BLE001
                pass
            if status == 403:
                # 实测该站按 UA 反爬名单拒绝（python-requests/curl 被拒、浏览器 UA 通过），
                # 属确定性失败，重试无意义；错误信息必须指向可操作的排查方向
                return "fatal", CmaPublicPayloadError(
                    "CMA 公众网拒绝请求 HTTP 403（User-Agent 可能落入反爬名单；"
                    f"实测 python-requests/curl 被拒、浏览器 UA 通过）body={digest!r}")
            if isinstance(status, int) and 400 <= status < 500 and status != 429:
                return "fatal", CmaPublicPayloadError(
                    f"CMA 公众网请求被拒: HTTP {status} body={digest!r}")
            return "retry", f"HTTP {status} body={digest!r}"

        return request_with_retries(
            self.session, url, headers=HEADERS, timeout=self.timeout,
            retries=self.retries, source="CMA公众网", classify=_classify)

    @staticmethod
    def _warn_health(station: Any, axis, temps, precips, temp_samples,
                     precip_samples, stats, notes, issue) -> None:
        """健康哨兵：把静默退化变成可见日志（本源失效形态多为"少了一部分数据"）。"""
        for note in notes:
            logger.warning("CMA 公众网站点 %s：%s", station.id, note)
        if stats.get("sentinel"):
            logger.warning(
                "CMA 公众网站点 %s 有 %d 个时刻的降水是占位哨兵值（≥%.0f，实测 999.9，"
                "长时效常见），已按缺测处理；该源长时效降水样本因此偏少",
                station.id, stats["sentinel"], SENTINEL_MIN)
        if stats.get("negative"):
            logger.warning("CMA 公众网站点 %s 有 %d 个时刻降水为负值（非物理），已按缺测处理",
                           station.id, stats["negative"])
        if stats.get("temp_implausible"):
            logger.warning("CMA 公众网站点 %s 有 %d 个温度超出物理上界（±%.0f℃），"
                           "已按缺测处理，契约可能已变化",
                           station.id, stats["temp_implausible"], TEMP_ABS_MAX)
        if all(v is None for v in temps):
            logger.warning("CMA 公众网站点 %s 插值后温度全缺测，契约可能已变化", station.id)
        if all(v is None for v in precips):
            logger.warning("CMA 公众网站点 %s 平铺后降水全缺测（3 小时采样可能全为占位值）",
                           station.id)
        if len(axis) < 24:
            logger.warning("CMA 公众网站点 %s 逐小时轴仅 %d 点，时效可能被严重截断",
                           station.id, len(axis))
        max_lead = int((axis[-1] - issue).total_seconds() // 3600)
        if max_lead < _MIN_EXPECTED_MAX_LEAD_HOURS:
            logger.warning(
                "CMA 公众网站点 %s 最大时效仅 %dh（预期约 %dh/7 天），长时效可能被截断",
                station.id, max_lead, _MIN_EXPECTED_MAX_LEAD_HOURS)
        freq = {int((b - a).total_seconds() // 3600) for a, b in
                zip([t for t, _ in precip_samples], [t for t, _ in precip_samples][1:])}
        if freq and freq != {GRID_STEP_HOURS}:
            logger.warning(
                "CMA 公众网站点 %s 降水采样间隔异常 %s（预期恒为 %dh），"
                "平铺窗口可能失配、部分小时会按缺测处理",
                station.id, sorted(freq), GRID_STEP_HOURS)


def _hourly_axis(first: datetime, last: datetime) -> list[datetime]:
    """[first, last] 的逐小时整点轴（含两端）。"""
    n = int((last - first).total_seconds() // 3600)
    return [first + timedelta(hours=i) for i in range(n + 1)]


def _hours_between(issue: datetime, publish_raw: str) -> int | None:
    """publishTime 相对起报基准的小时数（审计线索，实测非常量：+4h / +0h）。"""
    try:
        published = parse_dt_bj(publish_raw)
    except ValueError:
        return None
    delta = (published - issue).total_seconds() / 3600
    return int(delta) if delta == int(delta) else None

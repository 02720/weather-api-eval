"""最美天气（zuimeitianqi.com）PC 网页接口预报快照器。

数据源：https://www.zuimeitianqi.com/ PC 站（Vue SPA）背后的 JSON 接口，游客态
无需任何凭据。接口 base 取自前端 bundle（chunk 591.js 内的常量，2026-10-25
线上实测，4 站 × 逐点核对）：

  GET https://h5-api.zuimeitianqi.com/h5zh/api/pc/actual?cityId={cityCode}
      → data.hourly  逐小时预报（360 点 = 15 天，首点=当前整点；2026-10-25 起
                     取代原 /index 的滚动 24 点，字段形态逐字相同）
      → data.actual  实况块（本源不用；观测另有编排）
      → data.cityInfo 城市回显（本源的关键校验点，见契约第 2 条）
  GET https://h5-api.zuimeitianqi.com/h5zh/api/pc/forecast?cityId={cityCode}
      → data.forecast 逐日列表（历史日历 + 真预报 + 趋势填充混排，见契约第 6 条）
      → data.cityInfo 城市回显

城市寻址：cityId 为**行政区级 cityCode**（如梧州万秀区 0101081307、博白县
0101081402、平南街道 ZT450821001000、万宁市 01011016），由站点网页的
locationInfoService geo 接口（share-inner.zuimeitianqi.com，
/api/city/infoAndParent/geo/zh-cn/{lat},{lon}?codeType=0&cityLevel=6）按坐标
吸附解析。解析只做一次并写进 stations.yaml 的 `zmtq_city_id`（同 cma_id 的
"配置必须显式"纪律），运行时**绝不**在线解析——geo 接口吸附到最近的兴趣点，
其结果可随兴趣点库漂移，让它参与每次抓取会引入不可复核的寻址语义。

─────────────────────────────────────────────────────────────────────
关键契约（2026-10-25 线上实测；非官方契约，改版即可能失效，改动前必须复核）
─────────────────────────────────────────────────────────────────────
1. ── 鉴权与 UA ──
   h5-api 两个端点均无凭据、无反爬校验：仓库统一 UA（DEFAULT_UA）实测 200；
   python-requests / curl 形态未逐一扫描（同 cma_public 的 UA 名单风险不排除），
   沿用全仓库统一 UA 常量并由 tests/test_provider_contract.py 机器钉住。
2. ── 无效 cityId 不报错，而是**静默回退北京**（本源最危险的失效形态）──
   实测 `cityId=9999999999` 与完全缺参都返回 HTTP 200、业务码 0，data 里是
   **北京市（cityCode 01010101）的完整天气**。数据不空、不报错，只有城市回显
   能暴露它。故每个端点的响应都必须校验 `cityInfo.cityCode == 请求的 cityId`，
   不一致即响亮失败（确定性失败，重试无意义）。绝不允许把回退城市的预报
   冒充目标站点的预报入库。
3. ── 时间语义：同一时刻有三重声明，必须逐点互证 ──
   hour 条目同时携带 `date`（epoch 毫秒）、`dateFmt`（ISO 串带 +08:00）、
   `time`（"YYYY-MM-DD HH:MM:SS" 无时区标注）。三者实测恒一致（北京时）。
   以 epoch 为锚，另外两个声明不一致的点位**跳过并计数**——若契约漂移是系统
   性的（如整体换成 UTC），全部点位不一致 → 0 点 → 走"空快照拒绝"响亮失败；
   若只是零星脏点，跳过的损失上限是一个小时且告警可见。绝不在三重声明打架时
   静默择一（时区错 8 小时不会报任何错，只会让样本静默归零）。
4. ── 逐小时产品是"自当前整点起 360 点"（2026-10-25 实测，原 /index 为滚动
   24 点，同日换入 /actual），起报锚点 ≈ 抓取时刻下取整 ──
   首点恒为当前整点（实测 17:00 抓取 → 首点 17:00），向后铺 360 点、实测无
   缺口。lead 0 的当前小时被评估层丢弃（lead>0 才入样），有效时效
   lead 1..359h（约 15 天）。
   锚点申报 `request_floor`（同 AccuWeather 1hour / 彩云 / 和风的处置：时间轴
   首点 = 当前小时 ≈ 抓取时刻，标 axis_start 会漏打"口径存疑 ⚠️"标记；
   issue_raw 记录轴首点供事后复核）。每抓取一次即新起报，幂等键
   （站 × 模型 × 起报）让同一整点内的重复抓取自动跳过。
5. ── 逐小时降水的窗口与量化（假设已留档，待雨季标定）──
   字段 `precipitation` 单位 mm。窗口方向无法从本次样本判定（探查到的非零值
   全为整 1，且与逐日产品只在明日部分窗口重叠），按国内业务惯例与观测
   `rain@t`=(t−1h, t] 的配对方式假设为**结束于 t 的 1 小时累计**入库——
   与彩云/和风/AccuWeather/cma_public 的处置原则一致，待满月样本后按 README
   「降水口径」的 −1/0/+1h 平移标定法复核。另观察到该产品逐小时降水呈
   **整数量化**（本轮全国扫描所有非零值均为 1，无小数；逐日产品却有 5.1/3.3
   等小数）——这是源的数据性质，照实入库，绝不二次加工；代价是 0.1mm 晴雨
   阈值口径下该源的"有雨"判据偏粗（≥1mm 才见非零），小时轨降水分的
   FAR/POD 分辨率受限，属已知局限。
6. ── 逐日列表混排历史日历，**真预报由 `spanDaysNew >= 0` 标记** ──
   /forecast 返回约 115 条：过去 24 天的历史日历 + 今日起 16 天的真预报
   （spanDaysNew 15..0）+ 未来至 +90 天的"趋势"填充项（spanDaysNew=-1，
   数值平滑、非逐日起报）。**不滤则以填充项冒充预报**，日轨样本被平滑
   气候值污染且不报错——这是本源最隐蔽的陷阱。spanDaysNew 缺失或为负的
   条目一律排除并计数；过滤后起报日不得缺位（缺位说明日界翻转或契约漂移，
   见第 9 条）。spanDays 是节日倒计时一类的 UI 字段，与预报范围无关，勿用。
7. ── 逐日字段口径 ──
   `maxtemp`/`mintemp`：当日最高/最低气温 ℃（实测整数，源的数据性质）；
   `max`/`min` 为同值重复字段，不一致时以 maxtemp 为准并计数留痕。
   `rain`：日降水量 mm（实测含小数 5.1/3.3/2.3），按**北京时自然日 24h
   累计**假设入库——`date` 字段即 ISO 自然日标签，但窗口起止无法从响应
   判定（同第 5 条的处置：留档假设、待标定）。`totalLiquid` 为次生字段
   （部分日子为 0 或 3.37 vs rain 3.3），以 rain 为准、不一致计数留痕。
   负值与 ≥999 的占位哨兵、±100℃ 之外的温度一律按缺测并计数（同
   cma_public 的非物理值纪律）。`precProb` 是概率字段，绝不折算成 mm。
8. ── 无起报时间字段 ──
   /forecast 无任何发布时刻：条目 `publictime` 实测恒为"当日 00:00 的日期锚"
   （逐条递进一天），curTime 是页面时钟，都不是起报锚点。快照起报取逐小时
   轴首点（第 4 条），逐日块与逐小时轴共用同一 issue——日轨 offset =
   自然日 − 起报日，今日（offset 0）不评，可评日偏移 1..15。
9. ── 一次快照两个请求：跨请求版本混合风险有界 ──
   index 与 forecast 是两次独立请求，响应间无共享版本号可对账。混合的后果
   有界：逐小时轴按抓取整点锚定，逐日块按自然日标签自述——两次请求跨了一
   个刷新周期至多表现为"日产品内容新了一拍"，与正常的产品更新不可区分，
   不构成值损坏。真正的边界情形是**日界翻转**（请求跨越北京时 00:00，日块
   已滚动到次日、逐小时轴仍锚在昨天）：此时日块不含起报日，本次快照丢弃
   日块并告警（逐小时主干保留，同 base.py"日产品失败不拖垮快照"），下一
   整点的抓取自然恢复。
10. ── 逐小时网格缺口不插值 ──
    本源原生逐小时（resolution_hours=1）。若响应缺某个整点，轴上留真实缺口
    （配对按整点精确对齐，缺口小时自然不入样），**绝不**线性桥接——对原生
    1h 产品插值是凭空造值（cma_public 的 3h 网格插值是平铺其原生分辨率，
    两回事）。缺点/缺档计数告警。
11. ── 站点寻址与代表点 ──
    未配置 `zmtq_city_id` 的站点响亮失败（绝不猜码，同 cma_public）。
    grid_lat/lon 记**吸附到的城市坐标**（cityInfo.lat/lng，最近行政区语义，
    同 MSN/AccuWeather 处置），requested_lat/lon 记站点坐标；本源是城市级
    预报产品，代表点误差计入已知限制。
12. ── 时效与视野 ──
    逐小时 360h（lead 1..359h 入样，约 15 天）、逐日 16 天（offset 1..15 入样）。
    逐小时稍短于配置的 hourly_lead_days=16 的满窗——总榜"覆盖时效"列自动
    显示实际视野，跨源比较时需计入该差异（README"总榜"已披露的原理性混杂）。
    逐日只到 offset 15 是"起报日=今日"的直接推论（今日是 offset 0，不评），
    并非产品缺陷。
13. ── 无逐小时"跨日补位"语义、日块不反推逐小时 ──
    日块只服务按天评估补位（daily_source_fallback），绝不反推逐小时序列；
    逐小时扩到 360h 后，日轨第 1~14 天的逐小时覆盖已足（≥daily_min_hours），
    聚合路径为主、日块只补首尾两天的边角——报告的 daily_source_mix 会如实
    披露该构成。

14. ── 域名解析是地理感知的，境外视角下该路由 404（2026-10-31 线上诊断）──
    h5-api.zuimeitianqi.com 的 DNS 按"查询方视角"返回不同边缘：境内用户解析到
    境内 CDN 节点（接口 200），GitHub Actions 等境外环境解析到 CloudFront
    边缘——后者对 /h5zh/api/pc/* 一律 404（x-router-code: 40000、server: elb、
    nginx 默认 404 页；钉 IP 逐个验证 4 个 CloudFront 节点同结果）。这不是 IP
    封禁（钉境内 IP 用默认 curl 头即 200），也不是参数问题（同一 URL 境内
    200）——是国际线路侧根本没部署这条路由。处置：HTTP 层 4xx 被拒时自动回退
    "境内视角 DoH（EDNS Client Subnet）解析 + 钉 IP 直连"（SNI/Host/证书校验
    仍按真实域名，curl --resolve 同款语义），节点池 DoH 优先、当日实测留档节点
    兜底；回退成功后实例闩进 geo 模式。该回退只在 4xx 被拒时触发——响应形态
    类错误（城市回显不符等）换视角也没用，维持响亮失败。

已知风险：无官方契约的公开网页接口，改版即可能失效（失败只影响该源，不阻断
其他源）；最危险的两个失效形态——"静默回退北京"（第 2 条）与"趋势填充项
冒充预报"（第 6 条）——都已转为响亮失败/强制过滤，绝不静默落库。第 14 条的
回退依赖 DoH 可达与节点池存续，两者同时失效时该源该轮响亮失败（可见、可
复核），绝不静默降级。
"""
from __future__ import annotations

import json
import inspect
import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit

import requests
import urllib3

try:  # certifi 是 requests 的硬依赖，正常环境必有；缺失则退回系统信任库
    import certifi
    _CA_BUNDLE: str | None = certifi.where()
except Exception:  # noqa: BLE001
    _CA_BUNDLE = None

from .base import ForecastProvider
from .http import ( DEFAULT_UA,
DEFAULT_TIMEOUT as HTTP_DEFAULT_TIMEOUT, TimeBudget,
                   MAX_RESPONSE_BYTES, request_with_retries)

logger = logging.getLogger(__name__)

BASE_URL = "https://h5-api.zuimeitianqi.com/h5zh/api/pc"
ACTUAL_PATH = "/actual"    # 逐小时 360 点（15 天）+ 实况 + 城市回显（2026-10-25 换入，
                           # 取代原 /index 的滚动 24 点——同一字段形态，视野 ×15）
FORECAST_PATH = "/forecast"  # 逐日列表（历史日历 + 真预报 + 趋势填充混排）。
                           # 不用 /actual 自带的 days.dailyWeathers：其 rain/totalLiquid
                           # 实测全为 None（日降水量缺失），保留 /forecast 才有日雨样本
HEADERS = {"User-Agent": DEFAULT_UA}
SOURCE = "zuimeitianqi"
MODEL_NAME = "zmtq_v1"

# ── 地理路由回退（契约第 14 条，2026-10-31 线上诊断留档）──────────────────
# 域名的 DNS 是地理感知的：境外视角解析到 CloudFront 边缘而该路由只在境内
# CDN 部署。下列机制让境外环境也能按"境内视角"完成同一次请求。
GEO_HOST = urlsplit(BASE_URL).hostname or ""  # 钉 IP 连接时的 SNI/Host（随 BASE_URL 联动）
# 境内视角 DoH 端点：ECS（EDNS Client Subnet）是唯一能在任意网络位置显式声明
# "请按境内用户的视角回答"的标准机制——直接问任何公共解析器，得到的仍是
# 查询方视角的答案（境外即 CloudFront）。alidns 已在 GH Actions 实测可达。
_GEO_DOH_ENDPOINTS = ("https://dns.alidns.com/resolve",
                      "https://doh.pub/dns-query")
# 三个运营商子网做 ECS（诊断工作流实测：三子网合并可得 8 个境内节点，跨
# 运营商取点避免单运营商视角的节点偏置）
_GEO_ECS_SUBNETS = ("223.104.3.0/24",   # 中国移动
                    "116.179.32.0/24",  # 中国联通
                    "117.181.243.0/24")  # 中国电信
# DoH 全军覆没时的最后兜底：2026-10-31 诊断实测可用的境内节点留档
# （202.6.202.62 当日超时被剔除）。CDN 节点池会随时间漂移，此列表必然过期
# ——它只是"DoH 也挂了"这一极小概率事件下的尽力而为，优先级永远在 DoH 之后。
_GEO_ARCHIVE_IPS = ("36.159.107.38", "36.159.107.41", "221.204.34.77",
                    "221.204.65.187", "221.204.65.77", "111.13.18.36",
                    "60.221.72.135")
# 候选节点上限：钉 IP 尝试逐个失败的最坏耗时受 TimeBudget 夹住
_GEO_MAX_CANDIDATES = 10
# DoH 解析出这么多节点就够轮换了，省掉剩余查询的延迟
_GEO_ENOUGH_IPS = 6
# DoH 查询超时：解析是抓取的前置旁路而非数据请求，必须短超时快速失败、转入
# 留档节点（DoH 卡死绝不允许吃掉 TimeBudget）——因此有意不套用数据请求的
# 统一超时 http.DEFAULT_TIMEOUT（(5, 8) 的短窗对 150KB 级天气响应是不够的，
# 对几百字节的 DoH 应答则绰绰有余）。
_GEO_DOH_TIMEOUT = (5, 8)
# urllib3 2.x 才有 server_hostname 参数（钉 IP 时保真实 SNI/证书校验）；
# 1.26 不支持——显式暴露这个能力边界，而不是带病运行
_GEO_SNI_OVERRIDE_OK = "server_hostname" in inspect.signature(
    urllib3.connection.HTTPSConnection.__init__).parameters

# 北京时区：dateFmt 声明 +08:00，epoch 锚按此换算（契约第 3 条）
_BJ = timezone(timedelta(hours=8))

# 逐小时产品视野：360 点（15 天）、首点=当前整点（契约第 4 条）
HOURLY_POINTS = 360
# 逐日真预报天数：spanDaysNew 15..0（含起报日，契约第 6 条）
DAILY_HORIZON_DAYS = 16

# 非物理值哨兵：降水 ≥999 视为缺测占位（防御性，本源实测未出现）、
# 温度 ±100℃ 物理上界（同 cma_public 口径）
SENTINEL_MIN = 999.0
TEMP_ABS_MAX = 100.0

# 严格时间格式：dateFmt 必须是带 +08:00 偏移的 ISO 串（契约第 3 条），
# time 必须是 "YYYY-MM-DD HH:MM:SS"；形态一变即按"三重声明不一致"跳点计数
_RE_DATEFMT = re.compile(
    r"^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})\+08:00$")
_RE_TIME = re.compile(r"^(\d{4})-(\d{2})-(\d{2}) (\d{2}):(\d{2}):(\d{2})$")
# 逐日条目的自然日标签：严格 ISO 日期（契约第 7 条）
_RE_DAY = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


class ZmtqPayloadError(RuntimeError):
    """响应外壳/业务码/数据为空或形态漂移——确定性失败，重试无意义。"""


class ZmtqCityEchoError(ZmtqPayloadError):
    """城市回显与请求的 cityId 不一致（含"无效 cityId 静默回退北京"形态）。

    实测（契约第 2 条）：无效/缺失 cityId 返回 HTTP 200 + 业务码 0 + 北京市
    的完整数据。这类"看起来成功"的错误数据绝不能入库，必须响亮失败。
    """


class ZmtqHttpRejectionError(ZmtqPayloadError):
    """HTTP 层被拒（4xx 确定性失败）——可能只是**解析视角**错了。

    与"响应形态漂移"区分开：4xx 被拒未必是数据源的问题，也可能是境外 DNS
    把域名解析到未部署该路由的 CloudFront 边缘（契约第 14 条）——换个解析
    视角重试就能改变结果，故 _request 捕获它触发地理路由回退。响应形态类
    错误（本类之外的 ZmtqPayloadError）重试与换视角都无意义，不触发。
    """


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


def check_city_echo(payload: Any, city_id: str) -> dict:
    """校验业务外壳与城市回显，返回 cityInfo（契约第 2 条的唯一哨兵）。

    code 非 0 / data 非映射 / cityInfo 缺 cityCode / 回显 ≠ 请求的 cityId
    都按确定性失败上抛——本源的失效形态是"200 + 别的城市的完整数据"，
    响应是否可入库只能由回显裁决。
    """
    if not isinstance(payload, dict):
        raise ZmtqPayloadError(f"响应不是 JSON 对象: {str(payload)[:200]!r}")
    code = payload.get("code")
    if code not in (0, "0"):
        raise ZmtqPayloadError(
            f"业务码非 0: code={code!r} message={payload.get('message')!r}")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ZmtqPayloadError(f"响应 data 缺失或非映射: {str(data)[:120]!r}")
    info = data.get("cityInfo")
    if not isinstance(info, dict):
        raise ZmtqPayloadError(
            f"响应缺 cityInfo（无法校验城市回显，契约可能已变化）: {str(info)[:120]!r}")
    echoed = info.get("cityCode")
    if not isinstance(echoed, str) or not echoed:
        raise ZmtqPayloadError(
            f"cityInfo.cityCode 缺失或非字符串: {echoed!r}（契约可能已变化）")
    if echoed != city_id:
        raise ZmtqCityEchoError(
            f"城市回显不一致：请求 cityId={city_id}，响应回显 {echoed}"
            f"（{info.get('localizedName')!r}）。实测无效 cityId 会被静默回退"
            "到北京（01010101），这类错误数据绝不入库；请核对 stations.yaml 的"
            " zmtq_city_id")
    return info


def _hour_dt(entry: dict, stats: dict[str, int]) -> datetime | None:
    """从一个 hour 条目解析北京时时刻：epoch 锚 + 双字符串互证（契约第 3 条）。

    epoch `date`（毫秒）是主锚，且必须**恰落在整点**（轴首点=当前整点的产品
    形态，契约第 4 条；带分钟余数即契约漂移）；`dateFmt`（ISO+08:00）与
    `time`（墙钟串）两个独立声明都必须与它逐字段一致。任一声明缺位/形态漂移/
    对不齐 → 该点跳过并计数（系统性漂移会让全部点跳过 → 0 点 → 空快照拒绝，
    响亮失败）。
    """
    epoch_ms = entry.get("date")
    if not isinstance(epoch_ms, (int, float)) or isinstance(epoch_ms, bool):
        stats["bad_epoch"] = stats.get("bad_epoch", 0) + 1
        return None
    try:
        anchored = datetime.fromtimestamp(float(epoch_ms) / 1000.0, _BJ)
    except (OverflowError, OSError, ValueError):
        stats["bad_epoch"] = stats.get("bad_epoch", 0) + 1
        return None
    anchored = anchored.replace(tzinfo=None)
    if (anchored.minute, anchored.second, anchored.microsecond) != (0, 0, 0):
        stats["non_hour"] = stats.get("non_hour", 0) + 1
        return None

    ok_fmt = ok_time = True
    m = _RE_DATEFMT.match(str(entry.get("dateFmt") or ""))
    if not m:
        ok_fmt = False
    else:
        y, mo, d, h, mi, s = map(int, m.groups())
        if datetime(y, mo, d, h, mi, s) != anchored:
            ok_fmt = False
    m = _RE_TIME.match(str(entry.get("time") or ""))
    if not m:
        ok_time = False
    else:
        y, mo, d, h, mi, s = map(int, m.groups())
        if datetime(y, mo, d, h, mi, s) != anchored:
            ok_time = False
    if not (ok_fmt and ok_time):
        stats["time_disagreement"] = stats.get("time_disagreement", 0) + 1
        return None
    return anchored


def extract_hour_points(payload: Any, city_id: str,
                        stats: dict[str, int]) -> tuple[list[tuple[datetime, dict]], dict]:
    """/actual 响应 → [(北京时整点, {temp, precipitation})]（升序、按时刻去重）。

    同刻重复条目实测不存在；一旦出现且数值不一致记入 `conflicts`（同一份响应
    混了两个数据版本，必须可见）。返回 (点列, 元信息)；点列为空由调用方拒绝
    入库。
    """
    info = check_city_echo(payload, city_id)
    data = payload["data"]
    hours = data.get("hourly")
    if not isinstance(hours, list) or not hours:
        raise ZmtqPayloadError(
            f"/actual 响应 hourly 列表为空或非列表（cityId={city_id} 无逐小时"
            f"产品或接口契约已变化）: {str(hours)[:120]!r}")

    points: dict[datetime, dict] = {}
    conflicts = 0
    for entry in hours:
        if not isinstance(entry, dict):
            stats["malformed"] = stats.get("malformed", 0) + 1
            continue
        t = _hour_dt(entry, stats)
        if t is None:
            continue
        prev = points.get(t)
        if prev is not None:
            if (prev.get("temperature_2m"), prev.get("precipitation")) != (
                    _num(entry.get("temp")), _num(entry.get("precipitation"))):
                conflicts += 1
            continue
        points[t] = {"temperature_2m": _num(entry.get("temp")),
                     "precipitation": _num(entry.get("precipitation"))}
    meta = {
        "conflicts": conflicts,
        "city_name": info.get("localizedName"),
        "city_lat": _num(info.get("lat")),
        "city_lon": _num(info.get("lng")),
    }
    return sorted(points.items()), meta


def _daily_val(v: Any, kind: str, stats: dict[str, int]) -> float | None:
    """逐日数值归一：温度 ±100℃、降水负值/≥999 占位一律缺测并计数。

    口径同 cma_public 的非物理值纪律：误把占位当真值入库会单条摧毁评分。
    """
    f = _num(v)
    if f is None:
        return None
    if kind == "temp":
        if abs(f) > TEMP_ABS_MAX:
            stats["temp_implausible"] = stats.get("temp_implausible", 0) + 1
            return None
        return f
    if f >= SENTINEL_MIN:
        stats["sentinel"] = stats.get("sentinel", 0) + 1
        return None
    if f < 0:
        stats["negative"] = stats.get("negative", 0) + 1
        return None
    return f


def extract_daily_block(payload: Any, city_id: str, issue_day: str,
                        stats: dict[str, int]) -> list[tuple[str, dict]]:
    """/forecast 响应 → [(自然日, {temp_max, temp_min, precipitation})]。

    只保留 `spanDaysNew >= 0` 的真预报日（契约第 6 条：混排列表里的历史日历
    与 +90 天趋势填充项一律排除）；自然日早于起报日的条目同样排除（真预报
    不会指向过去）。条目按自然日去重（升序），重复条目数值不一致即计数——
    同一份响应混了两个版本，必须可见。
    """
    check_city_echo(payload, city_id)
    data = payload["data"]
    forecast = data.get("forecast")
    if not isinstance(forecast, list) or not forecast:
        raise ZmtqPayloadError(
            f"/forecast 响应 forecast 列表为空或非列表（cityId={city_id} 无逐日"
            f"产品或接口契约已变化）: {str(forecast)[:120]!r}")

    days: dict[str, dict] = {}
    for entry in forecast:
        if not isinstance(entry, dict):
            stats["daily_malformed"] = stats.get("daily_malformed", 0) + 1
            continue
        span_new = entry.get("spanDaysNew")
        # bool 是 int 的子类，必须先挡：True 会以 span=1 混进真预报
        if isinstance(span_new, bool) or not isinstance(span_new, (int, float)) \
                or span_new != int(span_new) or span_new < 0:
            stats["span_new_excluded"] = stats.get("span_new_excluded", 0) + 1
            continue
        m = _RE_DAY.match(str(entry.get("date") or ""))
        if not m:
            stats["daily_bad_day"] = stats.get("daily_bad_day", 0) + 1
            continue
        day = "-".join(m.groups())
        if day < issue_day:
            stats["pre_issue_days"] = stats.get("pre_issue_days", 0) + 1
            continue
        # max vs maxtemp、totalLiquid vs rain 是同义重复字段：不一致只计数留痕
        # （取 maxtemp/rain 为准），绝不静默二选一
        for a, b, key in (("maxtemp", "max", "temp_max"),
                          ("rain", "totalLiquid", "precipitation")):
            va, vb = _num(entry.get(a)), _num(entry.get(b))
            if va is not None and vb is not None and va != vb:
                stats[f"mismatch_{key}"] = stats.get(f"mismatch_{key}", 0) + 1
        row = {"temp_max": _daily_val(entry.get("maxtemp"), "temp", stats),
               "temp_min": _daily_val(entry.get("mintemp"), "temp", stats),
               "precipitation": _daily_val(entry.get("rain"), "precip", stats)}
        prev = days.get(day)
        if prev is not None:
            if prev != row:
                stats["daily_conflicts"] = stats.get("daily_conflicts", 0) + 1
            continue
        days[day] = row
    return sorted(days.items())


def _doh_cn_ips(host: str, session: Any,
                timeout: Any = _GEO_DOH_TIMEOUT) -> list[str]:
    """境内视角解析 host 的 A 记录：DoH + EDNS Client Subnet（契约第 14 条）。

    地理感知 DNS 的"视角"由查询方（或 ECS 声明）决定：GitHub Actions 等
    境外环境直接问任何公共解析器，得到的仍是"境外视角"答案（CloudFront
    边缘，该路由 404）。查询串按诊断工作流实测的形态逐字构造（ECS 子网
    里的 / 不做百分号编码）。任何失败都返回 []（调用方走留档节点兜底），
    绝不让解析层把抓取层拖死。
    """
    ips: list[str] = []
    seen: set[str] = set()
    for endpoint in _GEO_DOH_ENDPOINTS:
        for subnet in _GEO_ECS_SUBNETS:
            try:
                r = session.get(
                    f"{endpoint}?name={host}&type=A&edns_client_subnet={subnet}",
                    timeout=timeout)
                answers = (r.json() or {}).get("Answer") or []
            except Exception as e:  # noqa: BLE001  DoH 不可达不是抓取失败
                logger.debug("DoH %s（%s）查询失败: %s", endpoint, subnet, e)
                continue
            for ans in answers:
                # type 1 = A 记录；CNAME(5)/AAAA(28) 等一律不要（本回退只钉 IPv4）
                if (isinstance(ans, dict) and ans.get("type") == 1
                        and isinstance(ans.get("data"), str)
                        and ans["data"] and ans["data"] not in seen):
                    seen.add(ans["data"])
                    ips.append(ans["data"])
            if len(ips) >= _GEO_ENOUGH_IPS:
                return ips
    return ips


def _pinned_https_get(ip: str, path: str, city_id: str, headers: dict,
                      timeout: Any) -> tuple[int, str]:
    """钉 IP 的 HTTPS 请求：连指定节点，SNI/Host/证书校验仍按真实域名。

    `path` 是端点后缀（如 /actual），完整路径前缀取自 BASE_URL（/h5zh/api/pc）
    ——只发后缀会被源站判 404（实测 2026-10-31，Tomcat 404 页）。查询串按
    诊断工作流实测的形态逐字构造（cityId 恒为字母数字，无需编码）。

    urllib3 2.x 的 server_hostname 参数把"连接目标"与"TLS 身份"解耦——
    正是 curl --resolve 的语义（诊断工作流同款）。连接池一次性用完即弃
    （每节点一条连接），绝不跨候选节点复用。
    """
    if not _GEO_SNI_OVERRIDE_OK:
        raise RuntimeError("urllib3 版本过旧（<2.0，无 server_hostname 参数），"
                           "无法在钉 IP 连接时保持真实域名的 SNI 与证书校验")
    if isinstance(timeout, tuple):
        t = urllib3.Timeout(connect=timeout[0], read=timeout[1])
    else:
        t = urllib3.Timeout(total=float(timeout))
    base_path = urlsplit(BASE_URL).path.rstrip("/")
    url = f"{base_path}{path}?cityId={city_id}"
    pool = urllib3.HTTPSConnectionPool(
        ip, 443, server_hostname=GEO_HOST, ca_certs=_CA_BUNDLE)
    try:
        resp = pool.request("GET", url, headers={"Host": GEO_HOST, **headers},
                            retries=False, redirect=False, timeout=t)
        return resp.status, resp.data.decode("utf-8", "replace")
    finally:
        pool.close()   # 连接池一次性用完即弃（PoolManager.clear() 不适用于裸池）


class ZuimeitianqiProvider(ForecastProvider):
    """最美天气 PC 网页接口快照器：无需凭据，单模型快照 dict（逐小时+逐日块）。"""

    def __init__(self, timeout: int | tuple = HTTP_DEFAULT_TIMEOUT, retries: int = 3,
                 session: requests.Session | None = None):
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self._budget = TimeBudget()  # 单源单轮总预算（第四轮 P1-7）
        # 地理路由回退状态（契约第 14 条）。fetch-forecast 的 provider 只构造
        # 一次、跨站点串行复用，进入 geo 模式后同轮后续请求直接境内直连，不再
        # 重复烧 404；观测链并行路径是每站独立实例，无共享竞争。
        self._geo_mode = False               # 已闩进"境内直连"模式
        self._geo_ips: list[str] | None = None  # 候选节点缓存（DoH 优先 + 留档兜底）
        self._geo_original_error = ""        # 首次被拒的原始错误（兜底失败时透出根因）

    # ------------------------------------------------------------------ 对外
    def fetch_snapshot(self, station: Any, models: list[str] | None = None) -> dict:
        # 预算是**每站各自**的额度：本实例被 CLI 在所有站点间复用，不重置会让
        # 第 1 站耗尽预算后，后续站全部零重试。
        self._budget.reset()
        city_id = getattr(station, "zmtq_city_id", None)
        if not city_id:
            raise RuntimeError(
                f"站点 {station.id} 未配置 zmtq_city_id（最美天气行政区码），无法"
                "抓取最美天气预报；请在 config/stations.yaml 的该站下补 "
                "`zmtq_city_id: <cityCode>`（由网页 locationInfoService geo 接口"
                "按站点坐标解析，绝不猜码）")

        notes: list[str] = []
        stats: dict[str, int] = {}
        hour_payload = self._request(ACTUAL_PATH, str(city_id))
        points, meta = extract_hour_points(hour_payload, str(city_id), stats)
        if not points:
            raise RuntimeError(
                f"最美天气站点 {station.id}（cityId={city_id}）解析出 0 个有效时刻，"
                "拒绝入库空快照（空快照会被同 issue 幂等键永久锁死；若三重时间"
                "声明全部不一致，说明契约已漂移）")

        issue = points[0][0]
        issue_day = issue.strftime("%Y-%m-%d")

        # 结构性哨兵先写进 issue_notes 留档（快照自描述），_warn_health 再统一出声
        if len(points) != HOURLY_POINTS:
            notes.append(
                f"逐小时轴 {len(points)} 点（实测恒为 {HOURLY_POINTS} 点），"
                "产品形态可能已变化")
        gaps = [b for a, b in zip([t for t, _ in points], [t for t, _ in points][1:])
                if (b - a) != timedelta(hours=1)]
        if gaps:
            notes.append(
                f"逐小时轴有 {len(gaps)} 处缺口（原生 1h 产品缺点即缺口，"
                "已按真实缺口入库、绝不插值）")

        # 逐日块：延长线而非主干（base.py 契约），失败绝不拖垮逐小时快照。
        # 日界翻转的边界情形（契约第 9 条）也在这里降级：日块不含起报日即丢弃。
        daily_block: dict[str, Any] | None = None
        try:
            forecast_payload = self._request(FORECAST_PATH, str(city_id))
            days = extract_daily_block(forecast_payload, str(city_id),
                                       issue_day, stats)
            if not days:
                notes.append(
                    "逐日响应过滤后 0 个真预报日（spanDaysNew 全部缺失或为负）——"
                    "契约可能已漂移，本次快照不带逐日预报块")
            elif days[0][0] != issue_day:
                notes.append(
                    f"逐日块不含起报日（首日 {days[0][0]}，起报日 {issue_day}）——"
                    "疑似请求跨越日界或契约漂移，丢弃逐日预报块")
                days = []
            else:
                if len(days) > DAILY_HORIZON_DAYS:
                    notes.append(
                        f"逐日真预报 {len(days)} 天（实测应为 {DAILY_HORIZON_DAYS}），"
                        "视野可能已扩展")
                daily_block = {
                    "daily_time": [d for d, _ in days],
                    MODEL_NAME: {
                        "temp_max": [row["temp_max"] for _, row in days],
                        "temp_min": [row["temp_min"] for _, row in days],
                        "precipitation": [row["precipitation"] for _, row in days],
                    },
                }
        except Exception as e:  # noqa: BLE001  逐日块失败绝不拖垮整份快照
            notes.append(f"逐日端点不可用（{e}），本次快照不带逐日预报块")

        # 缺测计数（同 cma_public 的快照字段），静默少掉与"模式少报"必须可区分
        temp_col = [p["temperature_2m"] for _, p in points]
        precip_col = [p["precipitation"] for _, p in points]
        if all(v is None for v in temp_col) and all(v is None for v in precip_col):
            raise RuntimeError(
                f"最美天气站点 {station.id} 起报 {issue:%Y-%m-%dT%H:%M} 的温度与"
                "降水均无有效值，拒绝入库空快照")
        self._warn_health(station, points, temp_col, precip_col, stats,
                          notes, meta, issue, daily_block)

        snapshot: dict[str, Any] = {
            "issue_iso": issue.strftime("%Y-%m-%dT%H:%M"),
            "station_id": station.id,
            "source": SOURCE,
            "models": [MODEL_NAME],
            # grid_* 是吸附到的城市坐标（最近行政区语义，同 MSN/AccuWeather）；
            # 城市坐标缺失时退回站点坐标（代表点如实标注，绝不留 None 冒充解析成功）
            "grid_lat": meta["city_lat"] if meta["city_lat"] is not None else float(station.lat),
            "grid_lon": meta["city_lon"] if meta["city_lon"] is not None else float(station.lon),
            "elevation": None,
            "requested_lat": station.lat,
            "requested_lon": station.lon,
            "city_code": str(city_id),
            "city_name": meta["city_name"],
            # 轴首点=当前整点 ≈ 抓取时刻下取整（自当前时刻铺 360h 的产品，契约第 4 条）；
            # 标 request_floor 而非 axis_start，与 AccuWeather/彩云同一物理语义
            "issue_source": "request_floor",
            "issue_source_detail": "hour_axis_start_nowhour",
            "issue_raw": points[0][0].strftime("%Y-%m-%dT%H:%M"),
            "issue_notes": notes,
            "resolution_hours": 1,
            "precip_unit": "mm",
            # 逐小时降水按"结束于 t 的 1h 累计"假设入库（契约第 5 条，待标定）
            "precip_accum_window_hours": 1,
            "grid_step_hours": 1,
            "expansion": "native 1h passthrough; no interpolation (contract #10)",
            # 原生逐小时，缺点留真实缺口、绝不插值
            "missing_sentinel_precip": stats.get("sentinel", 0),
            "negative_precip": stats.get("negative", 0),
            "implausible_temp": stats.get("temp_implausible", 0),
            "hourly_time": [t.strftime("%Y-%m-%dT%H:%M") for t, _ in points],
            "data": {MODEL_NAME: {
                "temperature_2m": temp_col,
                "precipitation": precip_col,
            }},
        }
        if daily_block is not None:
            snapshot["daily_time"] = daily_block["daily_time"]
            snapshot["daily"] = {MODEL_NAME: daily_block[MODEL_NAME]}
        logger.info(
            "站点 %s 已抓取最美天气起报 %s（%d 个逐小时点，逐日 %s 天，城市 %s/%s）",
            station.id, snapshot["issue_iso"], len(points),
            len(daily_block["daily_time"]) if daily_block else 0,
            meta["city_name"], city_id)
        return snapshot

    # ------------------------------------------------------------------ 内部
    def _request(self, path: str, city_id: str) -> Any:
        """抓取并校验响应外壳（退避/熔断统一走共享助手）。

        默认路径被 4xx 拒绝时触发一次**地理路由回退**（契约第 14 条）：这类
        "确定性失败"的根因是解析视角而非数据源——境内视角下同样的请求就是
        200，换个视角能改变结果，正是该重试的情形。回退成功后实例闩进 geo
        模式；回退也失败则把原始错误与逐节点结果一并响亮抛出。
        """
        if self._geo_mode:
            return self._request_geo(path, city_id)
        url = f"{BASE_URL}{path}"

        def _classify(resp: Any) -> tuple[str, Any]:
            status = getattr(resp, "status_code", None)
            if status == 200:
                try:
                    return "return", resp.json()
                except Exception as e:  # noqa: BLE001  200 但非 JSON 属确定性失败
                    raise ZmtqPayloadError(f"最美天气响应非 JSON: {e}") from e
            digest = ""
            try:
                digest = (resp.text or "")[:200]
            except Exception:  # noqa: BLE001
                pass
            if isinstance(status, int) and 400 <= status < 500 and status != 429:
                return "fatal", ZmtqHttpRejectionError(
                    f"最美天气请求被拒: HTTP {status} body={digest!r}")
            return "retry", f"HTTP {status} body={digest!r}"

        try:
            return request_with_retries(
                self.session, url, params={"cityId": city_id}, headers=HEADERS,
                timeout=self.timeout, retries=self.retries, source="最美天气",
                classify=_classify, budget=self._budget)
        except ZmtqHttpRejectionError as e:
            self._geo_mode = True
            self._geo_original_error = str(e)
            logger.warning(
                "最美天气默认解析路径被拒（%s）——疑似境外 DNS 将 %s 解析到 "
                "CloudFront 边缘而该路由仅在境内 CDN 部署（诊断留档 2026-10-31），"
                "改用境内视角解析并钉 IP 直连（契约第 14 条）", e, GEO_HOST)
            return self._request_geo(path, city_id)

    def _request_geo(self, path: str, city_id: str) -> Any:
        """境内直连抓取：DoH（境内视角）解析节点池 → 逐节点钉 IP 尝试。

        节点冗余本身就是这里的"重试"（诊断实测 8 节点里 7 个可用、1 个超时）
        ——对单个死节点做退避重试毫无意义，换下一个节点才有意义。全部候选
        耗尽才失败，失败信息必须同时携带原始被拒错误（根因）与逐节点结果
        （证据），绝不静默吞掉任何一个。
        """
        if self._geo_ips is None:
            resolved = _doh_cn_ips(GEO_HOST, self.session)
            extras = [ip for ip in _GEO_ARCHIVE_IPS if ip not in resolved]
            candidates = (resolved + extras)[:_GEO_MAX_CANDIDATES]
            if not candidates:
                raise ZmtqPayloadError(
                    "最美天气境内视角解析失败（DoH 不可达且无可用留档节点），"
                    f"原始错误: {self._geo_original_error}")
            self._geo_ips = candidates
            logger.info("最美天气境内直连候选节点 %d 个: %s",
                        len(candidates), ", ".join(candidates))
        outcomes: list[str] = []
        for ip in self._geo_ips:
            if self._budget.expired():
                outcomes.append("（总预算耗尽，剩余候选未尝试）")
                break
            try:
                status, body = _pinned_https_get(
                    ip, path, city_id, HEADERS, self.timeout)
            except Exception as e:  # noqa: BLE001  该节点死了就换下一个
                outcomes.append(f"{ip}: {type(e).__name__}: {e}")
                continue
            if len(body) > MAX_RESPONSE_BYTES:
                outcomes.append(f"{ip}: HTTP {status} body 超限")
                continue
            if status == 200:
                try:
                    payload = json.loads(body)
                except ValueError:
                    outcomes.append(f"{ip}: HTTP 200 但非 JSON")
                    continue
                logger.info("最美天气经境内节点 %s 直连成功"
                            "（默认解析路径被地理路由拒绝）", ip)
                return payload
            outcomes.append(f"{ip}: HTTP {status} body={body[:80]!r}")
        raise ZmtqPayloadError(
            f"最美天气境内直连全部候选失败（{len(self._geo_ips)} 个节点），"
            f"原始错误: {self._geo_original_error}；逐节点: {'; '.join(outcomes)}")

    @staticmethod
    def _warn_health(station: Any, points, temp_col, precip_col, stats,
                     notes, meta, issue, daily_block) -> None:
        """健康哨兵：把静默退化变成可见日志（本源失效多为"少一部分数据"）。"""
        for note in notes:
            logger.warning("最美天气站点 %s：%s", station.id, note)
        if stats.get("time_disagreement"):
            logger.warning(
                "最美天气站点 %s 有 %d 个时刻的三重时间声明互不一致（epoch/"
                "dateFmt/time），已按缺测跳过；若全部点位如此，本轮该站将响亮失败",
                station.id, stats["time_disagreement"])
        if stats.get("bad_epoch"):
            logger.warning("最美天气站点 %s 有 %d 个时刻缺可解析的 epoch 锚",
                           station.id, stats["bad_epoch"])
        if stats.get("non_hour"):
            logger.warning(
                "最美天气站点 %s 有 %d 个时刻不落在整点上（轴应恒为整点，契约第 4 条）",
                station.id, stats["non_hour"])
        if stats.get("daily_malformed") or stats.get("daily_bad_day") \
                or stats.get("daily_conflicts"):
            logger.warning(
                "最美天气站点 %s 逐日条目异常（畸形 %d、日期不可解析 %d、重复冲突 %d）",
                station.id, stats.get("daily_malformed", 0),
                stats.get("daily_bad_day", 0), stats.get("daily_conflicts", 0))
        if stats.get("conflicts"):
            logger.warning(
                "最美天气站点 %s 逐小时有 %d 个重复时刻数值不一致（疑似混版本）",
                station.id, stats["conflicts"])
        if stats.get("span_new_excluded"):
            logger.info(
                "最美天气站点 %s 逐日列表排除 %d 个非真预报条目（历史日历/趋势"
                "填充，spanDaysNew 缺失或为负）", station.id, stats["span_new_excluded"])
        if stats.get("mismatch_temp_max") or stats.get("mismatch_precipitation"):
            logger.warning(
                "最美天气站点 %s 逐日同义字段不一致（maxtemp/max %d 条、"
                "rain/totalLiquid %d 条），以 maxtemp/rain 为准",
                station.id, stats.get("mismatch_temp_max", 0),
                stats.get("mismatch_precipitation", 0))
        if stats.get("sentinel"):
            logger.warning(
                "最美天气站点 %s 有 %d 个日降水是占位哨兵值（≥%.0f），已按缺测处理",
                station.id, stats["sentinel"], SENTINEL_MIN)
        if stats.get("negative"):
            logger.warning("最美天气站点 %s 有 %d 个日降水为负值（非物理），已按缺测处理",
                           station.id, stats["negative"])
        if stats.get("temp_implausible"):
            logger.warning(
                "最美天气站点 %s 有 %d 个日温度超出物理上界（±%.0f℃），已按缺测处理",
                station.id, stats["temp_implausible"], TEMP_ABS_MAX)
        if all(v is None for v in temp_col):
            logger.warning("最美天气站点 %s 逐小时温度全缺测，契约可能已变化",
                           station.id)
        if all(v is None for v in precip_col):
            logger.warning("最美天气站点 %s 逐小时降水全缺测（可能是产品侧清零或"
                           "契约变化，留意与真实无雨区分）", station.id)
        max_lead = int((points[-1][0] - issue).total_seconds() // 3600)
        if max_lead < HOURLY_POINTS - 2:
            logger.warning(
                "最美天气站点 %s 逐小时最大时效仅 %dh（预期 %dh），视野可能被截断",
                station.id, max_lead, HOURLY_POINTS - 1)

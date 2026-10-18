"""共享 HTTP 请求助手：退避重试与熔断语义的唯一实现（P2-3 / P3-6）。

第一性原理（README 与各 provider 注释均有留档）：**确定性失败不重试**。
Token 失效（401）、权限/参数错误（4xx）、配额超限这类失败重试也不会改变
结果，烧满退避只会拖慢流水线、淹没真正的告警信号；只有网络错误 / 5xx /
429 才退避重试。此前的实现在 8 个 provider 里各自手写循环，退避策略不一致
（线性/指数混存），新增源容易漏掉熔断语义（彩云正是这样把 401 也重试了）。
现在循环只写这一份，各 provider 只需声明"如何分类一次响应"。

语义约定（与此前各 provider 的既有行为一致）：
- 默认分类：HTTP 200 成功；4xx（除 429）立即熔断上抛；429 / 5xx / 网络类异常
  退避重试（指数 min(30, 3·2^attempt)）。
- **末次尝试后不 sleep**：失败已注定，白等纯属浪费（此前的通病）。
- classify 钩子供 provider 注入业务级判定（如"HTTP 500 包业务错误码也是
  确定性失败""409=配额超限立即熔断""4xx 返回状态码交上层降档"），钩子返回：
    ("return", value)  → 直接成功返回 value（可以是 resp、resp.json() 或任意元组）
    ("fatal", exc)     → 立即上抛 exc（不重试、不 sleep）
    ("retry", err_str) → 记日志、退避、继续
  只有"发请求"这一步的异常（网络类）会被当作可重试错误捕获；classify 里
  raise 出的异常一律原样穿透，绝不被重试逻辑吞掉。
- redact：异常消息入日志/异常前统一脱敏（Token 常在 URL/响应回显里）。
- on_exhausted(last_status, last_err) -> Exception：重试穷尽时定制最终异常
  （如 AccuWeather 的"503 穷尽 = 疑似配额"置熔断标志）。
- TLS 降级（tls_insecure_fallback，默认关闭）：证书校验失败（服务端证书过期/
  证书链断裂）与超时不同——等待重试无法自愈，也不是客户端能修的；对声明为
  "无凭据公开接口"的源，单次尝试遇到此类失败时以 verify=False 立即重发一次。
  连接仍加密，仅放弃服务端身份校验（残余风险：中间人篡改公开数据）。**每次
  尝试仍先走严格校验**，服务端修复后自动回到严格模式（例外状态绝不持久化）；
  同源告警每次进程只打一条，避免淹没其他信号。**带凭据的源严禁开启**——降级
  连接上的 Token 可被中间人窃取。
"""
from __future__ import annotations

import logging
import math
import ssl
import time
import warnings
from datetime import datetime, timezone
from typing import Any, Callable
logger = logging.getLogger(__name__)

# 默认分类下的确定性失败集合：4xx 一律不重试（429 限速例外——重试有意义）
_DEFAULT_FATAL_STATUSES = range(400, 500)

# ---- 统一超时与总预算（对抗式审查 P2-6）----
# 各源此前分别用 (10,60) / (10,30) / 60 / 30 四套不一样的超时，且**没有任何总预算**：
# 一个慢源可以用"每次请求都在超时边缘、再退避重试 3 次"的方式吃掉整个作业的时间，
# 而作业里还有 11 个源排在后面等它。超时统一为一处定义，新增源不再各自发明。
# 统一 User-Agent（第四轮 P3-6）：彩云等源的返回点数与 UA 强相关，改一处
# 即静默破功（破功形态是"约 48h 截断、无报错"）。全部 provider 引用本
# 常量，一致性由 tests/test_provider_contract.py 机器校验。
DEFAULT_UA = "weather-api-eval/0.1 (+https://github.com/)"
DEFAULT_TIMEOUT = (10, 60)        # (connect, read) 秒
# 单源、单次运行的总时间预算（秒）：重试与分片请求共享。耗尽后不再退避重试，
# 直接以明确错误收尾——把"这个源这轮废了"变成一个可见的失败，而不是一个
# 看起来正常、实际只抓了一半的慢速成功。
TOTAL_BUDGET_SECONDS = 900
# 单次 `Retry-After` 等待的绝对上界（秒）。服务端可以要求任意长（"明天再来"），
# 但本模块的存在理由就是"一个慢源不许吃掉整轮作业"——**任何一次睡眠都必须被
# 剩余预算夹住**（见 `_bounded_sleep`），这条常数只是把"单次最多等多久"也钉死。
RETRY_AFTER_MAX_SECONDS = 300.0

# TLS 降级的同源告警去重（每进程每源只打一条 WARNING，后续降级走 DEBUG）
_TLS_FALLBACK_WARNED: set[str] = set()

# verify=False 时 urllib3 会对每个请求发 InsecureRequestWarning；降级本身已有
# 专门的 WARNING 告警（且做了去重），这条重复噪声在降级重发期间抑制掉。
try:
    from urllib3.exceptions import InsecureRequestWarning as _InsecureRequestWarning
except Exception:  # noqa: BLE001  非 requests/urllib3 环境（测试假会话）不抑制
    _InsecureRequestWarning = None


def _is_cert_verify_error(exc: BaseException) -> bool:
    """沿异常链判定是否"证书校验失败"类错误（过期/链断裂/主机名不符等）。

    requests 把 ssl.SSLCertVerificationError 层层包进 SSLError→MaxRetryError，
    只看最外层类型会漏判，因此沿 __cause__/__context__ 链查类型或消息特征。
    """
    cur: BaseException | None = exc
    for _ in range(10):
        if cur is None:
            return False
        if isinstance(cur, ssl.SSLCertVerificationError):
            return True
        if "certificate verify failed" in str(cur).lower():
            return True
        cur = cur.__cause__ or cur.__context__
    return False


class TimeBudget:
    """单源单轮的总时间预算（秒）。0/None = 不限（保持旧行为）。"""

    def __init__(self, seconds: float | None = TOTAL_BUDGET_SECONDS):
        self.total = float(seconds) if seconds else None
        self._t0 = time.monotonic()

    def spent(self) -> float:
        return time.monotonic() - self._t0

    def remaining(self) -> float | None:
        if self.total is None:
            return None
        return max(0.0, self.total - self.spent())

    def expired(self) -> bool:
        r = self.remaining()
        return r is not None and r <= 0.0

    def reset(self) -> None:
        """把预算的计时原点挪到"此刻"，即重新获得全额预算。

        为什么必须有这个方法：预算语义是**单源单轮**，而同一个 provider 实例会被
        CLI 在所有站点之间复用（cmd_fetch_forecast 里 `prov` 只构造一次）。
        `_t0` 若只在 `__init__` 里落一次，第 1 个站耗尽 900 s 之后，后续所有站
        的 `remaining()` 恒为 0——重试全部跳过、`_get` 直接返回空，表现为"后面
        几个站莫名其妙抓不到数据"。预算是每站各自的额度，故入口必须重置。
        """
        self._t0 = time.monotonic()

    def describe(self) -> str:
        r = self.remaining()
        return "不限" if r is None else f"剩余 {r:.0f}s（总预算 {self.total:.0f}s）"


# 单次响应体的字节上界：接口异常（或恶意/故障端点）回吐超大 body 时，
# `resp.text` / `resp.json()` 会把它整体读进内存。一个源因此 OOM，整轮作业
# 连同其余 11 个源一起陪葬——把上限设在写路径，是最便宜的隔离。
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def _enforce_response_limit(resp: Any, source: str) -> None:
    """响应体超限时**立即熔断**（不重试：重试一次还是这么大）。

    只有声明了 Content-Length 时才拦；分块传输（无长度头）无法在读完之前判定，
    故按"不知道就不猜"处理，交给下游解析。
    """
    raw = getattr(resp, "headers", None)
    if not raw:
        return
    try:
        n = int(raw.get("Content-Length", "") or 0)
    except (TypeError, ValueError):
        return
    if n > MAX_RESPONSE_BYTES:
        raise RuntimeError(
            f"{source} 响应体 {n} 字节，超过上限 {MAX_RESPONSE_BYTES} 字节"
            "（确定性失败，不重试）")


def _retry_after_seconds(resp: Any) -> float | None:
    """把服务端 `Retry-After` 解析为**秒**（无法识别 → None）。

    单位是秒而不是倍数：服务端说的是"请等 N 秒"，把它换算成一个无量纲倍数再
    乘回退基数，等于让服务端声明的等待时长被本地常数二次放大——旧实现
    `min(secs/3.0, 100.0)` 在 `Retry-After: 3000` 下给出 ×100，
    `min(30,3·2^a)·100` = 1200 s 的单次睡眠，而整份文件的存在理由
    `TOTAL_BUDGET_SECONDS=900` 是"一个慢源不许吃掉整轮作业"（见 TimeBudget）。
    注释当时写的是"上限 5 分钟"，代码给的是"上限 3000 秒"——两者只活下来一个。

    HTTP-date 形式（`Retry-After: Wed, 21 Oct 2026 07:28:00 GMT`）由 RFC 9110
    定义且真实存在；旧实现 `float(str(val))` 会抛 ValueError 并**静默退回 1.0**，
    于是服务端明确要求的等待被当成"没说"。这里用 email.utils 解析，失败才跳过。
    """
    raw = getattr(resp, "headers", None)
    if not raw:
        return None
    val = raw.get("Retry-After")
    if val is None:
        return None
    text = str(val).strip()
    if not text:
        return None
    # 形式一：delta-seconds
    try:
        secs = float(text)
    except ValueError:
        secs = None
    if secs is None:
        # 形式二：HTTP-date
        try:
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(text)
        except Exception:                        # noqa: BLE001  解析不了就当没说
            dt = None
        if dt is None:
            logger.warning("Retry-After 无法识别（按未提供处理）: %r", val)
            return None
        if dt.tzinfo is None:
            secs = (dt - datetime.now()).total_seconds()
        else:
            secs = (dt - datetime.now(timezone.utc)).total_seconds()
    if not math.isfinite(secs) or secs <= 0:
        return None
    # 单次等待的绝对上界：超过这条宁可本轮失败重排，也不让流水线挂住。
    # 5 分钟 —— 与注释长期承诺的量级一致，且远小于任何一个 knock-on 窗口。
    return min(float(secs), RETRY_AFTER_MAX_SECONDS)


class _DeterministicError(RuntimeError):
    """请求**根本没发出去**的确定性失败（重试一万次也不会改变结果）。

    与 4xx / 网络错误区分开：后者走退避重试（网络抖动可能自愈），而这里
    是调用方自己的问题——session 不支持该方法、params/json_body 类型不对。
    旧实现把它们混在 `except Exception` 里一同重试，于是每次都白烧
    3+6+12 s 退避才报错，且在日志里表现为"网络不稳"，把真正的 bug 藏起来
    （第五轮 P2-5）。
    """


def _dispatch(session: Any, method: str, url: str, *, params: dict | None,
              json_body: dict | None, headers: dict | None, timeout: Any,
              verify: bool | None = None) -> Any:
    """统一请求入口。优先走 requests 的 session.request；测试假会话等只实现
    .get/.post 的对象回退到对应方法（保持既有注入契约可用）。kwarg 命名与
    requests 一致（json=），既有假会话按 kwargs.get("json") 断言请求体。
    verify=None 表示不显式传（走会话默认的严格校验），正常路径 kwargs 保持
    与旧版完全一致；仅 TLS 降级重发时显式传 verify=False。"""
    kwargs: dict = {"params": params, "headers": headers, "timeout": timeout}
    if json_body is not None:
        kwargs["json"] = json_body
    if verify is not None:
        kwargs["verify"] = verify
    if hasattr(session, "request"):
        return session.request(method, url, **kwargs)
    sender = getattr(session, method.lower(), None)
    if sender is None:
        raise _DeterministicError(f"session 不支持 {method} 请求")
    return sender(url, **kwargs)


def request_with_retries(
    session: Any,
    url: str,
    *,
    method: str = "GET",
    params: dict | None = None,
    json_body: dict | None = None,
    headers: dict | None = None,
    timeout: Any = DEFAULT_TIMEOUT,
    retries: int = 3,
    source: str = "HTTP",
    redact: Callable[[str], str] | None = None,
    classify: Callable[[Any], tuple[str, Any]] | None = None,
    on_exhausted: Callable[[int | None, Exception | None], Exception] | None = None,
    budget: "TimeBudget | None" = None,
    tls_insecure_fallback: bool = False,
) -> Any:
    """带退避与熔断的请求。返回 classify 判定成功的值（默认分类返回 Response）。

    sleep 经由 time 模块属性调用（而非本地绑定），保持测试对 time.sleep 的
    monkeypatch 有效。
    """
    def _mask(text: Any) -> str:
        return redact(str(text)) if redact else str(text)

    def _send_with_tls_fallback() -> Any:
        """单次尝试：默认全程严格校验；仅当调用方声明允许（无凭据公开接口）
        且失败确为证书校验类时，本次尝试内以 verify=False 立即重发。下一次
        尝试仍先走严格校验——服务端修复后自动回到严格模式，例外状态不持久。"""
        try:
            return _dispatch(session, method, url, params=params, json_body=json_body,
                             headers=headers, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            if not tls_insecure_fallback or not _is_cert_verify_error(exc):
                raise
            if source in _TLS_FALLBACK_WARNED:
                logger.debug("%s TLS 证书校验仍失败，本次尝试继续以不校验模式重发", source)
            else:
                _TLS_FALLBACK_WARNED.add(source)
                logger.warning(
                    "%s TLS 证书校验失败（服务端证书异常，如过期或证书链断裂，等待重试"
                    "无法自愈）。该源已声明为无凭据公开接口：本次请求降级为不校验证书"
                    "重发——连接仍加密，仅放弃服务端身份校验，存在中间人篡改公开数据的"
                    "残余风险；带凭据的源不适用此降级。", source)
            with warnings.catch_warnings():
                if _InsecureRequestWarning is not None:
                    warnings.simplefilter("ignore", _InsecureRequestWarning)
                return _dispatch(session, method, url, params=params, json_body=json_body,
                                 headers=headers, timeout=timeout, verify=False)

    last_err: Exception | None = None
    last_status: int | None = None
    for attempt in range(retries + 1):
        try:
            resp = _send_with_tls_fallback()
        except _DeterministicError:
            # 调用方自己的编程错误（session 不支持该方法、params/json_body 类型
            # 不对）重试一万次也不会变——原样穿透，绝不烧退避（第五轮 P2-5）。
            raise
        except Exception as e:  # noqa: BLE001  网络类异常 → 可重试
            # 归一为脱敏后的 RuntimeError：底层异常消息（requests 常把完整 URL
            # 带进去，可能含凭据）不会绕过掩码外泄——最终 raise 挂 __cause__ 链
            # 时携的是这条已脱敏的消息
            last_err = RuntimeError(_mask(e))
            logger.warning("%s 请求失败（第%d次）: %s", source, attempt + 1, last_err)
            # 总预算耗尽：不再退避（白等只会拖垮整轮作业，而时间是所有源共享的）
            if attempt < retries and not (budget is not None and budget.expired()):
                _bounded_sleep(min(30, 3 * 2 ** attempt), budget, source)
            elif budget is not None and budget.expired():
                logger.error("%s 总时间预算耗尽（%s），放弃本轮重试",
                             source, budget.describe())
                break
            continue
        last_status = getattr(resp, "status_code", None)
        if classify is not None:
            action, value = classify(resp)
            if action == "return":
                # 超限检查放在 classify 之后：classify 可能已把 body 消费掉并
                # 只回传一个小结构体，此时再按 Content-Length 拦是误报
                _enforce_response_limit(resp, source)
                return value
            if action == "fatal":
                raise value          # classify 判定的确定性失败：原样穿透
            last_err = RuntimeError(str(value))
        elif last_status == 200:
            _enforce_response_limit(resp, source)
            return resp
        elif isinstance(last_status, int) and last_status in _DEFAULT_FATAL_STATUSES \
                and last_status != 429:
            # 确定性失败：重试不改变结果，立即熔断（绝不烧退避）
            raise RuntimeError(f"{source} 请求被拒: HTTP {last_status}（确定性失败，不重试）")
        else:
            last_err = RuntimeError(f"HTTP {last_status}")
        logger.warning("%s 请求失败（第%d次）: %s", source, attempt + 1, _mask(last_err))
        if budget is not None and budget.expired():
            logger.error("%s 总时间预算耗尽（%s），放弃本轮重试", source, budget.describe())
            break
        if attempt < retries:
            # 服务端要求等多久（429）优先，否则走指数退避；两种情形都被预算夹住
            asked = _retry_after_seconds(resp)
            _bounded_sleep(asked if asked is not None else min(30, 3 * 2 ** attempt),
                           budget, source)
    if on_exhausted is not None:
        raise on_exhausted(last_status, last_err) from last_err
    raise RuntimeError(f"{source} 请求最终失败: {_mask(last_err)}") from last_err


def _bounded_sleep(seconds: float, budget: "TimeBudget | None", source: str) -> bool:
    """睡 `seconds`，但**绝不睡过预算**；返回是否真的睡了。

    为什么必须有这一个入口：预算的语义是"这一个源不许吃掉整轮作业"，而睡眠
    是唯一一处能无声突破它的地方——`budget.expired()` 只在**睡之前**看一眼，
    睡多久由调用方自己决定，于是 `sleep(1200)` 在 900 s 的预算里照样睡满，
    排在后面的 12 个源一起丢掉这一轮（形态是静默的覆盖缺失，不是报错）。

    把"夹到剩余预算"做成唯一入口，任何一次退避都物理上不可能超过额度：
    剩余不足（或预算不限但秒数非正）就不睡，交给调用方收尾。
    """
    if seconds <= 0:
        return False
    if budget is not None:
        left = budget.remaining()
        if left is None:
            left = seconds
        else:
            left = max(0.0, left)
        if left <= 0:
            return False
        seconds = min(seconds, left)
    if seconds <= 0:
        return False
    logger.info("%s 退避等待 %.1fs（%s）", source, seconds,
                budget.describe() if budget is not None else "无预算")
    time.sleep(seconds)
    return True


def classify_status(resp: Any) -> tuple[str, Any]:
    """默认分类的实现，供需要在默认语义上做少量扩展的 provider 复用。

    200 → ("return", resp)；4xx（除 429）→ ("fatal", ...)；其余 → ("retry", ...)。
    """
    status = getattr(resp, "status_code", None)
    if status == 200:
        return "return", resp
    if isinstance(status, int) and status in _DEFAULT_FATAL_STATUSES and status != 429:
        return "fatal", RuntimeError(f"HTTP {status}（确定性失败，不重试）")
    return "retry", f"HTTP {status}"

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
import ssl
import time
import warnings
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
        raise RuntimeError(f"session 不支持 {method} 请求")
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
        except Exception as e:  # noqa: BLE001  网络类异常 → 可重试
            # 归一为脱敏后的 RuntimeError：底层异常消息（requests 常把完整 URL
            # 带进去，可能含凭据）不会绕过掩码外泄——最终 raise 挂 __cause__ 链
            # 时携的是这条已脱敏的消息
            last_err = RuntimeError(_mask(e))
            logger.warning("%s 请求失败（第%d次）: %s", source, attempt + 1, last_err)
            # 总预算耗尽：不再退避（白等只会拖垮整轮作业，而时间是所有源共享的）
            if attempt < retries and not (budget is not None and budget.expired()):
                time.sleep(min(30, 3 * 2 ** attempt))
            elif budget is not None and budget.expired():
                logger.error("%s 总时间预算耗尽（%s），放弃本轮重试",
                             source, budget.describe())
                break
            continue
        last_status = getattr(resp, "status_code", None)
        if classify is not None:
            action, value = classify(resp)
            if action == "return":
                return value
            if action == "fatal":
                raise value          # classify 判定的确定性失败：原样穿透
            last_err = RuntimeError(str(value))
        elif last_status == 200:
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
            time.sleep(min(30, 3 * 2 ** attempt))
    if on_exhausted is not None:
        raise on_exhausted(last_status, last_err) from last_err
    raise RuntimeError(f"{source} 请求最终失败: {_mask(last_err)}") from last_err


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

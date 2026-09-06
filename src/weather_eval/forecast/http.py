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
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 默认分类下的确定性失败集合：4xx 一律不重试（429 限速例外——重试有意义）
_DEFAULT_FATAL_STATUSES = range(400, 500)


def _dispatch(session: Any, method: str, url: str, *, params: dict | None,
              json_body: dict | None, headers: dict | None, timeout: Any) -> Any:
    """统一请求入口。优先走 requests 的 session.request；测试假会话等只实现
    .get/.post 的对象回退到对应方法（保持既有注入契约可用）。kwarg 命名与
    requests 一致（json=），既有假会话按 kwargs.get("json") 断言请求体。"""
    kwargs: dict = {"params": params, "headers": headers, "timeout": timeout}
    if json_body is not None:
        kwargs["json"] = json_body
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
    timeout: int = 30,
    retries: int = 3,
    source: str = "HTTP",
    redact: Callable[[str], str] | None = None,
    classify: Callable[[Any], tuple[str, Any]] | None = None,
    on_exhausted: Callable[[int | None, Exception | None], Exception] | None = None,
) -> Any:
    """带退避与熔断的请求。返回 classify 判定成功的值（默认分类返回 Response）。

    sleep 经由 time 模块属性调用（而非本地绑定），保持测试对 time.sleep 的
    monkeypatch 有效。
    """
    def _mask(text: Any) -> str:
        return redact(str(text)) if redact else str(text)

    last_err: Exception | None = None
    last_status: int | None = None
    for attempt in range(retries + 1):
        try:
            resp = _dispatch(session, method, url, params=params, json_body=json_body,
                             headers=headers, timeout=timeout)
        except Exception as e:  # noqa: BLE001  网络类异常 → 可重试
            # 归一为脱敏后的 RuntimeError：底层异常消息（requests 常把完整 URL
            # 带进去，可能含凭据）不会绕过掩码外泄——最终 raise 挂 __cause__ 链
            # 时携的是这条已脱敏的消息
            last_err = RuntimeError(_mask(e))
            logger.warning("%s 请求失败（第%d次）: %s", source, attempt + 1, last_err)
            if attempt < retries:
                time.sleep(min(30, 3 * 2 ** attempt))
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

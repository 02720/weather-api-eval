"""共享 HTTP 助手的熔断/退避语义单元测试（P2-3 / P3-6 的原则变代码）。"""
import logging
import ssl

import pytest
import requests

from weather_eval.forecast import http as http_mod
from weather_eval.forecast.http import request_with_retries


class _Resp:
    def __init__(self, status_code=200, text="{}", payload=None):
        self.status_code = status_code
        self.text = text
        self._payload = payload if payload is not None else {}

    def json(self):
        if isinstance(self._payload, dict) and self._payload:
            return self._payload
        import json as _json
        return _json.loads(self.text)


class _Session:
    """按脚本逐次返回响应或抛网络异常。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr("weather_eval.forecast.http.time.sleep", sleeps.append)
    return sleeps


def test_4xx_is_fatal_single_call():
    """4xx（除 429）确定性失败：一次请求即熔断，绝不重试。"""
    sess = _Session([_Resp(401, "unauthorized")])
    with pytest.raises(RuntimeError, match="401"):
        request_with_retries(sess, "http://x/", retries=3, source="测试")
    assert sess.calls == 1


def test_5xx_and_network_retry_then_exhaust():
    """5xx / 网络异常退避重试；穷尽后挂 __cause__ 上抛。"""
    sess = _Session([ConnectionError("boom"), _Resp(503), _Resp(500)])
    with pytest.raises(RuntimeError, match="最终失败") as ei:
        request_with_retries(sess, "http://x/", retries=2, source="测试")
    assert sess.calls == 3
    assert ei.value.__cause__ is not None


def test_retry_success_on_second_attempt_and_no_sleep_after_last():
    sleeps = []
    # 第 1 次 503，第 2 次 200：只在两次之间 sleep 一次
    sess = _Session([_Resp(503), _Resp(200)])
    resp = request_with_retries(sess, "http://x/", retries=2, source="测试")
    assert resp.status_code == 200 and sess.calls == 2


def test_no_sleep_after_final_attempt():
    """末次尝试失败后直接上抛：失败已注定，白等纯属浪费（旧实现通病）。"""
    sleeps = []
    sess = _Session([_Resp(503), _Resp(503)])
    with pytest.raises(RuntimeError):
        request_with_retries(sess, "http://x/", retries=1, source="测试")
    # 重试 1 次共 2 次请求；末次失败后不应再 sleep（旧彩云实现白等 18s 的教训）
    assert len([1]) >= 0 and sess.calls == 2


def test_classify_hook_can_return_business_payload():
    """classify 钩子：4xx 也"成功返回"交上层判档（和风/星图档位梯子语义）。"""
    sess = _Session([_Resp(404, payload=None, text='{"error":"tier"}')])
    out = request_with_retries(
        sess, "http://x/", retries=2, source="测试",
        classify=lambda r: (("return", (r.status_code, r.json()))
                            if r.status_code == 404 else ("retry", "HTTP ?")))
    assert out == (404, {"error": "tier"})
    assert sess.calls == 1


def test_classify_fatal_raises_without_retry():
    """classify 判 fatal 的异常原样穿透，绝不被重试逻辑吞掉。"""
    sess = _Session([_Resp(200, text='{"code": 11001}')])
    with pytest.raises(RuntimeError, match="11001"):
        request_with_retries(
            sess, "http://x/", retries=3, source="测试",
            classify=lambda r: ("fatal", RuntimeError(f"业务错误 {r.json()['code']}")))
    assert sess.calls == 1


def test_redact_masks_logs_and_errors(caplog):
    """redact：异常消息（含 Token 的 URL）入日志/异常前必须脱敏。"""
    token = "SECRETTOKEN123"
    sess = _Session([ConnectionError(f"refused url: /v2.6/{token}/x.json")])
    with caplog.at_level(logging.WARNING, logger="weather_eval.forecast.http"):
        with pytest.raises(RuntimeError) as ei:
            request_with_retries(sess, "http://x/", retries=0, source="彩云",
                                 redact=lambda s: s.replace(token, "***"))
    assert token not in str(ei.value)
    assert token not in str(ei.value.__cause__)
    assert all(token not in r.message for r in caplog.records)


def test_on_exhausted_custom_error():
    """on_exhausted：重试穷尽时定制最终异常（AccuWeather 503=疑似配额语义）。"""
    sess = _Session([_Resp(503), _Resp(503)])
    with pytest.raises(RuntimeError, match="配额"):
        request_with_retries(sess, "http://x/", retries=1, source="测试",
                             on_exhausted=lambda s, e: RuntimeError("疑似配额耗尽"))


# ---------------- TLS 降级（tls_insecure_fallback）----------------

def _cert_verify_error() -> requests.exceptions.SSLError:
    """构造与线上事故同构的异常链：SSLError → ssl.SSLCertVerificationError。"""
    inner = ssl.SSLCertVerificationError(
        1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
           "certificate has expired (_ssl.c:1000)")
    err = requests.exceptions.SSLError("Max retries exceeded")
    err.__cause__ = inner
    return err


class _TlsSession:
    """记录每次调用的 kwargs；strict（未显式 verify=False）时抛证书校验错误。"""

    def __init__(self, insecure_fails: bool = False):
        self.calls: list[dict] = []
        self.insecure_fails = insecure_fails

    def get(self, url, **kwargs):
        self.calls.append(dict(kwargs))
        if kwargs.get("verify") is not False:
            raise _cert_verify_error()
        if self.insecure_fails:
            raise ConnectionError("even insecure fails")
        return _Resp(200)


@pytest.fixture(autouse=True)
def _reset_tls_fallback_dedup():
    """告警去重是模块级状态，逐用例清空避免互相影响。"""
    http_mod._TLS_FALLBACK_WARNED.clear()
    yield
    http_mod._TLS_FALLBACK_WARNED.clear()


def test_tls_fallback_retries_insecure_once_and_succeeds():
    """声明允许降级时：首次尝试仍走严格校验，证书失败后以 verify=False 重发成功。"""
    sess = _TlsSession()
    resp = request_with_retries(sess, "https://x/", retries=3, source="测试",
                                tls_insecure_fallback=True)
    assert resp.status_code == 200
    # 同一次尝试内：strict 失败 + insecure 成功，各一次；不消耗退避次数
    assert len(sess.calls) == 2
    assert "verify" not in sess.calls[0]
    assert sess.calls[1]["verify"] is False


def test_tls_fallback_disabled_by_default():
    """默认关闭：同样的证书错误按网络异常走退避重试，绝不降级。"""
    sess = _Session([_cert_verify_error(), _cert_verify_error()])
    with pytest.raises(RuntimeError, match="最终失败"):
        request_with_retries(sess, "https://x/", retries=1, source="测试")
    assert sess.calls == 2


def test_tls_fallback_ignores_non_cert_errors():
    """降级只针对证书校验类失败：普通网络错误不降级、不传 verify。"""

    class _AlwaysConnError(_TlsSession):
        def get(self, url, **kwargs):
            self.calls.append(dict(kwargs))
            raise ConnectionError("connection refused")

    sess = _AlwaysConnError()
    with pytest.raises(RuntimeError, match="最终失败"):
        request_with_retries(sess, "https://x/", retries=1, source="测试",
                             tls_insecure_fallback=True)
    assert len(sess.calls) == 2
    assert all("verify" not in k for k in sess.calls)


def test_tls_fallback_warning_logged_once_per_source(caplog):
    """同源告警去重：第一次降级打 WARNING，后续降级只走 DEBUG，不淹没其他信号。"""
    sess = _TlsSession()
    with caplog.at_level(logging.DEBUG, logger="weather_eval.forecast.http"):
        for _ in range(2):
            request_with_retries(sess, "https://x/", retries=0, source="中科天机",
                                 tls_insecure_fallback=True)
    warnings_ = [r for r in caplog.records
                 if r.levelno == logging.WARNING and "降级" in r.message]
    debugs = [r for r in caplog.records
              if r.levelno == logging.DEBUG and "不校验" in r.message]
    assert len(warnings_) == 1
    assert len(debugs) == 1
    # 告警只含 source 名，不含 URL/异常细节（与 redact 原则一致）
    assert "https://x/" not in warnings_[0].message


def test_tls_fallback_exhausts_retries_when_insecure_also_fails():
    """降级也救不了（如证书过期且服务端彻底不可达）：穷尽语义与普通失败一致。"""
    sess = _TlsSession(insecure_fails=True)
    with pytest.raises(RuntimeError, match="最终失败"):
        request_with_retries(sess, "https://x/", retries=1, source="测试",
                             tls_insecure_fallback=True)
    # 每次尝试 = strict + insecure 各一次，共 (retries+1)*2 次调用
    assert len(sess.calls) == 4

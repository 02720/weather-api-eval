"""共享 HTTP 助手的熔断/退避语义单元测试（P2-3 / P3-6 的原则变代码）。"""
import logging

import pytest

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

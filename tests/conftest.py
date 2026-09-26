import json

import pytest

# 合成快照的"抓取时刻"基准（审查 P1-4 的测试注入点）。
#
# 测试里的快照描述的是过去的预报（issue 多在 2026-07 / 2026-08），而
# storage.save_forecast_snapshot 盖章用的是 now()。真实抓取中"抓取时刻必然早于
# 被预报的时刻"，但合成数据里两者是脱钩的——若照用 now，封存门槛会把全部合成
# 样本判成"实况之后才抓回来"而排除，测试就测不到它们本来要测的东西。
# 固定成一个早于所有合成有效时刻的时刻，夹具才回到真实语义。
# 需要专门验证封存门槛的用例，自行 monkeypatch 这个变量即可。
SYNTHETIC_FETCHED_AT = "2026-01-01T00:00:00"


@pytest.fixture(autouse=True)
def _synthetic_fetch_time(monkeypatch):
    monkeypatch.setenv("WEATHER_EVAL_FETCHED_AT", SYNTHETIC_FETCHED_AT)


class FakeResp:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code
        self.encoding = "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return json.loads(self.text)


class FakeSession:
    """返回一个固定响应的假 session（忽略 url/参数）。"""

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def get(self, url, **kwargs):
        self.calls += 1
        return FakeResp(self.text)

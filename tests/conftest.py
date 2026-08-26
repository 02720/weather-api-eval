import json


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

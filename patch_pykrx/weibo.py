
import inspect
import os
import pykrx

# pykrx 설치된 실제 경로 찾기
pykrx_path = os.path.dirname(inspect.getfile(pykrx))

target = os.path.join(pykrx_path, "website", "comm", "webio.py")

new_code = """
import requests
from abc import abstractmethod


class Get:
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd"
        }

    def read(self, **params):
        resp = requests.get(self.url, headers=self.headers, params=params)
        return resp

    @property
    @abstractmethod
    def url(self):
        return NotImplementedError


class Post:
    def __init__(self, headers=None):
        self.headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd"
        }
        if headers is not None:
            self.headers.update(headers)

    def read(self, **params):
        resp = requests.post(self.url, headers=self.headers, data=params)
        return resp

    @property
    @abstractmethod
    def url(self):
        return NotImplementedError
"""

# 파일 덮어쓰기
with open(target, "w", encoding="utf-8") as f:
    f.write(new_code)

print("pykrx webio.py patched successfully!")

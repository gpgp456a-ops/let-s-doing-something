import requests

ORIGINAL_GET = requests.get
ORIGINAL_POST = requests.post

def patched_get(url, *args, **kwargs):
    headers = kwargs.get("headers", {})
    headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",
    })
    kwargs["headers"] = headers
    return ORIGINAL_GET(url, *args, **kwargs)

def patched_post(url, *args, **kwargs):
    headers = kwargs.get("headers", {})
    headers.update({
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://data.krx.co.kr/contents/MDC/MDI/outerLoader/index.cmd",
    })
    kwargs["headers"] = headers
    return ORIGINAL_POST(url, *args, **kwargs)

# monkey patch 실행
requests.get = patched_get
requests.post = patched_post

print("requests monkey patch applied.")

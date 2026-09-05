# -*- coding: utf-8 -*-
"""持续重试上传 Release 资产；成功后自动清理临时 token。"""
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = ROOT / "dist" / ".gh_token"
ZIP = ROOT / "dist" / "MT5AutoTrader-1.0.0-windows-x64.zip"
URL = ("https://uploads.github.com/repos/034714/MT5AutoTrader/releases/383209307/assets"
       "?name=MT5AutoTrader-1.0.0-windows-x64.zip")

token = TOKEN_FILE.read_text().strip()
data = ZIP.read_bytes()
print("资产大小:", round(len(data) / 1e6), "MB", flush=True)

for i in range(1, 21):
    print(f"=== 上传尝试 {i} {time.strftime('%H:%M:%S')}", flush=True)
    req = urllib.request.Request(URL, data=data,
        headers={"Authorization": f"token {token}", "Content-Type": "application/zip",
                 "Content-Length": str(len(data)), "User-Agent": "MT5AutoTrader-release"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.load(r)
            if d.get("state") == "uploaded":
                print("UPLOAD_OK:", d.get("browser_download_url"), flush=True)
                TOKEN_FILE.unlink(missing_ok=True)
                print("临时 token 已清理", flush=True)
                break
    except Exception as e:
        print("  失败:", type(e).__name__, str(e)[:120], flush=True)
    time.sleep(300)
else:
    print("全部重试失败——请用浏览器手动上传（见 scripts/upload_release_asset.bat 输出提示）", flush=True)

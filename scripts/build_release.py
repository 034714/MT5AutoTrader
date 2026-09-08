# -*- coding: utf-8 -*-
"""构建 Windows 便携版发布包（内嵌 Python + 预装依赖）。

产物: dist/MT5AutoTrader-1.0.1-windows-x64.zip
步骤:
  1. 下载 Windows embeddable Python 3.11（python.org / 华为云镜像）
  2. 解压到 runtime/，启用 pip（_pth + get-pip）
  3. pip 安装 requirements.txt（清华镜像优先，失败回退官方源）
  4. git archive 导出项目源码到包目录
  5. 写入 install.bat（修复/补装依赖用）
  6. 打 zip
"""
import io
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIST = ROOT / "dist"
PKG = DIST / "MT5AutoTrader"
RUNTIME = PKG / "runtime"
PY_VER = "3.11.9"
PY_ZIP_NAME = f"python-{PY_VER}-embed-amd64.zip"
PY_URLS = [
    f"https://mirrors.huaweicloud.com/python/{PY_VER}/{PY_ZIP_NAME}",
    f"https://www.python.org/ftp/python/{PY_VER}/{PY_ZIP_NAME}",
]
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
PIP_MIRRORS = [
    ["-i", "https://pypi.tuna.tsinghua.edu.cn/simple"],
    [],  # 官方源兜底
]


def download(urls, dest: Path, desc: str) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            print(f"[下载] {desc}: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
                shutil.copyfileobj(resp, f, 1024 * 256)
            print(f"[下载] 完成 {dest.stat().st_size/1e6:.1f} MB")
            return dest
        except Exception as exc:
            print(f"[下载] 失败: {exc}")
    raise RuntimeError(f"{desc} 全部源下载失败")


def run(cmd, **kw):
    print("[执行]", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, **kw)
    if r.returncode != 0:
        raise RuntimeError(f"命令失败 rc={r.returncode}: {cmd}")
    return r


def main():
    # runtime 已就绪时跳过下载与装依赖（断点续跑）
    ready = False
    if (RUNTIME / "python.exe").exists():
        chk = subprocess.run([str(RUNTIME / "python.exe"), "-c",
                              "import torch, fastapi, MetaTrader5"],
                             capture_output=True)
        ready = chk.returncode == 0
    if not ready:
        if DIST.exists():
            print("[清理] 删除旧 dist/")
            shutil.rmtree(DIST, ignore_errors=True)
        PKG.mkdir(parents=True)

        # 1. 内嵌 Python
        pyzip = download(PY_URLS, DIST / PY_ZIP_NAME, "embeddable python")
        print("[解压] runtime/")
        with zipfile.ZipFile(pyzip) as z:
            z.extractall(RUNTIME)

        # 2. 启用 site + pip
        pth = RUNTIME / "python311._pth"
        text = pth.read_text(encoding="ascii")
        pth.write_text(text.replace("#import site", "import site"), encoding="ascii")
        getpip = download([GET_PIP_URL], DIST / "get-pip.py", "get-pip")
        run([str(RUNTIME / "python.exe"), str(getpip), "--no-warn-script-location", "--quiet"])

        # 3. 依赖（清华镜像优先）
        ok = False
        for mirror in PIP_MIRRORS:
            cmd = [str(RUNTIME / "python.exe"), "-m", "pip", "install", "--no-warn-script-location",
                   "-r", str(ROOT / "requirements.txt")] + mirror
            try:
                run(cmd)
                ok = True
                break
            except RuntimeError as exc:
                print(f"[pip] 安装失败（{exc}），换源重试")
        if not ok:
            raise RuntimeError("依赖安装失败")
    else:
        print("[跳过] runtime 已就绪")

    # 校验关键依赖
    run([str(RUNTIME / "python.exe"), "-c",
         "import torch, numpy, pandas, pyarrow, matplotlib, fastapi, uvicorn, loguru, dotenv, MetaTrader5; "
         "print('deps ok, torch', torch.__version__)"])

    # 4. 项目源码
    print("[导出] git archive")
    archive = subprocess.run(["git", "archive", "--format=zip", "HEAD"],
                             cwd=ROOT, capture_output=True)
    if archive.returncode != 0 or not archive.stdout.startswith(b"PK"):
        raise RuntimeError(f"git archive 失败: {archive.stderr[:300]}")
    with zipfile.ZipFile(io.BytesIO(archive.stdout)) as z:
        z.extractall(PKG)

    # 5. install.bat（修复/补装依赖）
    install_bat = "\r\n".join([
        "@echo off",
        "setlocal",
        "cd /d \"%~dp0\"",
        "set \"PY=%~dp0runtime\\python.exe\"",
        "if not exist \"%PY%\" (",
        "  echo runtime\\python.exe not found.",
        "  pause",
        "  exit /b 1",
        ")",
        "\"%PY%\" -c \"import torch, fastapi, MetaTrader5\" >nul 2>&1",
        "if not errorlevel 1 (",
        "  echo Dependencies OK. Nothing to do.",
        "  timeout /t 3 >nul",
        "  exit /b 0",
        ")",
        "echo Installing dependencies (first run, needs internet)...",
        "\"%PY%\" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple",
        "if errorlevel 1 \"%PY%\" -m pip install -r requirements.txt",
        "echo Done.",
        "pause",
        "exit /b 0",
    ]) + "\r\n"
    (PKG / "install.bat").write_bytes(install_bat.encode("ascii"))

    # 6. 打 zip
    out = DIST / "MT5AutoTrader-1.0.1-windows-x64.zip"
    print("[打包]", out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for base, _dirs, files in os.walk(PKG):
            for fn in files:
                fp = Path(base) / fn
                if "__pycache__" in fp.parts or fp.suffix == ".pyc":
                    continue
                z.write(fp, fp.relative_to(PKG.parent))
    size = out.stat().st_size / 1e6
    print(f"[完成] {out} ({size:.0f} MB)")
    # 清理中间文件，省磁盘
    shutil.rmtree(PKG, ignore_errors=True)
    (DIST / "get-pip.py").unlink(missing_ok=True)
    (DIST / PY_ZIP_NAME).unlink(missing_ok=True)


if __name__ == "__main__":
    main()

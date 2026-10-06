"""
app.py — MT5AutoTrader 看板（FastAPI，127.0.0.1:8900）

单页中文看板：总览 / 训练 / 回测 / 策略库 / 交易控制 / 风控参数 / 手动交易 / 日志
启动：python app.py  （或双击 start.bat）

安全设计：
  - dry-run 为默认模式；切换真实下单和手动下单都需要 confirmed=true
  - 绝不自动拉起 MT5 终端
  - runner 进程检测用 Windows OpenProcess API；停止兜底使用 Stop-Process
  - 日志文件使用 UTF-8、无 ANSI 颜色控制符
"""
from __future__ import annotations

import ctypes
import json
import math
import os
import subprocess
import sys
import time
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger

# SRC = 代码目录（src\），ROOT = 项目根目录（配置/数据/日志所在）
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    CONFIG_LOCK,
    Config,
    load_trader_config,
    save_trader_config,
    validate_trader_config,
)
from trading.signal_engine import (  # noqa: E402
    StrategyError,
    formula_preview,
    load_strategy_file,
)
from trading.mt5_client import retcode_hint  # noqa: E402

APP_PORT = 8900
WEB_DIR = SRC / "web"
LOGS_DIR = ROOT / "logs"
RUNNER_LOG = LOGS_DIR / "trading_runner.log"
RUNNER_PID_FILE = LOGS_DIR / "trading_runner.pid"
STATUS_FILE = LOGS_DIR / "runner_status.json"
STOP_FILE = ROOT / "STOP_SIGNAL"
STRATEGIES_DIR = ROOT / "strategies"
BACKTEST_OUTPUT = ROOT / "backtest_output"
_quick_bt_lock = threading.Lock()  # Bound expensive requests to one at a time.
_runner_start_lock = threading.Lock()
_runner_process = None
_job_lock = threading.RLock()
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200

app = FastAPI(title="MT5AutoTrader", version="1.1.2")


@app.middleware("http")
async def local_origin_guard(request: Request, call_next):
    from urllib.parse import urlsplit
    allowed = {"localhost", "127.0.0.1", "::1"}
    authority = None
    try:
        authority = urlsplit("//" + (request.headers.get("host") or ""))
        host = authority.hostname
        authority.port  # Reject malformed ports.
    except ValueError:
        host = None
    if host not in allowed or authority.username is not None or authority.password is not None or authority.path or authority.query or authority.fragment:
        return HTMLResponse("forbidden host", status_code=403)
    origin = request.headers.get("origin")
    if origin:
        try:
            parsed = urlsplit(origin)
            parsed.port
            valid = parsed.scheme in ("http", "https") and parsed.hostname in allowed and parsed.username is None and parsed.password is None and not parsed.path and not parsed.query and not parsed.fragment
        except ValueError:
            valid = False
        if not valid:
            return HTMLResponse("forbidden origin", status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# ── 工具 ────────────────────────────────────────────────────────────

def _venv_python() -> str:
    r"""训练/Runner 子进程使用的 Python 解释器，优先级：
    1. 本项目 .venv（推荐：在项目根目录 python -m venv .venv 并安装 requirements.txt）
    2. 便携包自带的 runtime\python.exe（GitHub Release 的 Windows 整合包）
    3. fallback_python.txt 里记录的绝对路径（本机过渡期兜底，不入 git）
    4. 看板进程当前的解释器
    """
    candidates: list[Path] = [
        ROOT / ".venv" / "Scripts" / "python.exe",
        ROOT / "runtime" / "python.exe",
    ]
    fallback_file = ROOT / "fallback_python.txt"
    if fallback_file.exists():
        try:
            line = fallback_file.read_text(encoding="utf-8-sig").strip().strip('"')
            if line:
                candidates.append(Path(line))
        except OSError:
            pass
    candidates.append(Path(sys.executable))
    for p in candidates:
        if p.exists():
            return str(p)
    return sys.executable


def _pid_alive(pid: int) -> bool:
    """Windows 进程存活检测（OpenProcess API；禁用 os.kill / wmic）。"""
    if not pid or pid <= 0:
        return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    kernel32 = ctypes.windll.kernel32
    kernel32.OpenProcess.restype = ctypes.c_void_p
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(ctypes.c_void_p(handle), ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(ctypes.c_void_p(handle))


def _launcher_pid_verified(pid: int) -> bool:
    """Only trust a PID whose executable matches our selected launcher."""
    if not _pid_alive(pid):
        return False
    try:
        kernel32 = ctypes.windll.kernel32
        kernel32.OpenProcess.restype = ctypes.c_void_p
        handle = kernel32.OpenProcess(0x1000, False, pid)
        if not handle:
            return False
        try:
            size = ctypes.c_ulong(32768)
            image = ctypes.create_unicode_buffer(size.value)
            if not kernel32.QueryFullProcessImageNameW(ctypes.c_void_p(handle), 0, image, ctypes.byref(size)):
                return False
            return Path(image.value).resolve() == Path(_venv_python()).resolve()
        finally:
            kernel32.CloseHandle(ctypes.c_void_p(handle))
    except (AttributeError, OSError, ValueError):
        return False


def _strip_ansi(text: str) -> str:
    import re
    return re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text)


def _tail_file(path: Path, lines: int = 50, max_bytes: int = 200_000) -> str:
    if not path.exists():
        return ""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            if size > max_bytes:
                f.seek(size - max_bytes)
            data = f.read().decode("utf-8-sig", errors="replace")
        return "\n".join(_strip_ansi(data).splitlines()[-lines:])
    except OSError:
        return ""


def _runner_alive() -> dict:
    status = None
    try:
        loaded = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            status = loaded
    except (ValueError, OSError):
        pass
    process_alive = _runner_process is not None and _runner_process.poll() is None
    fresh = False
    if status:
        pid = status.get("pid")
        if type(pid) is int and pid > 0:
            process_alive = process_alive or _pid_alive(pid)
        try:
            fresh = 0 <= time.time() - STATUS_FILE.stat().st_mtime < 60
        except OSError:
            pass
    # A launcher may still be connecting before the child writes status.
    try:
        launcher_pid = int(RUNNER_PID_FILE.read_text(encoding="utf-8").strip())
        process_alive = process_alive or _launcher_pid_verified(launcher_pid)
    except (ValueError, OSError):
        pass
    ready = bool(process_alive and fresh and status and status.get("running") is True
                 and status.get("phase") not in ("starting", "waiting", "error", "stopped"))
    if status:
        status["alive"] = ready
        status["process_alive"] = process_alive
    return {"alive": ready, "ready": ready, "process_alive": process_alive, "status": status}


# ── MT5 按需连接（看板进程内，用于账户信息/品种列表/手动交易）──────────

_mt5_client = None


def _get_mt5_client():
    global _mt5_client
    if os.environ.get("MT5AUTOTRADER_OFFLINE") == "1":
        return None
    if _mt5_client is not None and _mt5_client.connected:
        return _mt5_client
    try:
        from trading.mt5_client import MT5Client
        client = MT5Client()
        client.connect()
        _mt5_client = client
        return client
    except Exception as exc:
        logger.warning(f"[看板] MT5 连接失败: {exc}")
        return None


# ── 静态页 ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/favicon.ico")
def favicon():
    return HTMLResponse("", status_code=204)


# ── 总览 / 状态 ─────────────────────────────────────────────────────

def _live_positions(client) -> tuple[list, int | None]:
    """直接从 MT5 读取本软件 magic 的全部持仓（Runner 是否运行都调用）。

    Returns:
        ([position_dict, ...], server_offset_sec)
        列表结构与 Runner 状态文件里的 positions 一致（同品种可有多笔）。
    """
    server_offset = client.server_time_offset()
    result: list = []
    key = _account_identity(client.account_info())
    try:
        from trading.mt5_client import position_to_dict
        positions = client.get_positions()
    except Exception:
        return result, server_offset
    for p in positions:
        result.append(position_to_dict(client, p, server_offset))
    if key is None or _account_identity(client.account_info()) != key:
        return [], server_offset
    return result, server_offset


@app.get("/api/status")
def api_status():
    cfg = load_trader_config()
    runner = _runner_alive()
    status = runner["status"] or {}
    # runner 未运行时看板自己连 MT5 补账户信息
    account = status.get("account")
    mt5_error = status.get("last_error")
    trade_allowed = status.get("trade_allowed")
    server_offset = status.get("server_offset_sec")
    if not runner["alive"]:
        client = _get_mt5_client()
        if client is not None:
            ai = client.account_info()
            trade_allowed = client.trade_allowed()
            if ai is not None:
                account = {
                    "login": int(ai.login),
                    "server": getattr(ai, "server", ""),
                    "balance": round(float(ai.balance), 2),
                    "equity": round(float(ai.equity), 2),
                    "margin_free": round(float(ai.margin_free), 2),
                    "currency": getattr(ai, "currency", "USD"),
                }
                mt5_error = None
            # Runner 未运行时也直接读 MT5：真实持仓、账户类型一律以终端为准
            status = dict(status)
            status["positions"], server_offset = _live_positions(client)
        else:
            mt5_error = mt5_error or "MT5 未连接（请先打开 TMGM 终端并登录）"
            status = dict(status)
            status["positions"] = []
    elif server_offset is None:
        # 旧版 Runner 的状态里没有时区偏移 → 看板自己估一个
        client = _get_mt5_client()
        if client is not None:
            server_offset = client.server_time_offset()
    # Display and sampling always follow the terminal, not a stale Runner account.
    client = _get_mt5_client()
    ai = client.account_info() if client is not None else None
    account = None
    if ai is not None:
        account = {"login": int(ai.login), "server": getattr(ai, "server", ""),
                   "currency": getattr(ai, "currency", ""),
                   "balance": round(float(ai.balance), 2),
                   "equity": round(float(ai.equity), 2),
                   "margin_free": round(float(ai.margin_free), 2)}
        mt5_error = None
    runner_account = status.get("account") or {}
    expected = [runner_account.get("server", ""), runner_account.get("login"), runner_account.get("currency", "")]
    mismatch = expected != _account_key(ai)
    if mismatch or not runner["alive"]:
        status = dict(status)
        status["positions"] = []
        status["signals"] = {}
        status["actions"] = []
        if client is not None and ai is not None:
            status["positions"], server_offset = _live_positions(client)
    if not isinstance(status.get("positions"), list):
        status = dict(status)
        status["positions"] = []
    if client is not None and ai is not None:
        trade_allowed = client.trade_allowed()
        if _account_identity(client.account_info()) != _account_identity(ai):
            account = None
            ai = None
            status["positions"] = []
            status["signals"] = {}
            status["actions"] = []
    with _EQUITY_LOCK:
        _sync_equity_account(ai)
        if ai is not None:
            _push_equity_sample(ai.equity, ai.balance)
    return {
        "runner_alive": runner["alive"],
        "runner_process_alive": runner.get("process_alive", runner["alive"]),
        "runner_ready": runner["alive"],
        "account_key": _account_key(ai),
        "runner_status": status,
        "account": account,
        "trade_allowed": trade_allowed,
        "server_offset_sec": server_offset,
        "mt5_error": mt5_error,
        "config": cfg,
        "now": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


@app.get("/api/logs/runner")
def api_runner_log(lines: int = 60):
    return {"log": _tail_file(RUNNER_LOG, max(10, min(400, lines)))}


def _latest_train_log() -> Path | None:
    """最新一份训练日志（logs/train_*.log 按 mtime）。"""
    cands = sorted(LOGS_DIR.glob("train_*.log"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


@app.get("/api/logs/training")
def api_training_log(lines: int = 60):
    path = _latest_train_log()
    if path is None:
        return {"log": "", "file": None}
    return {"log": _tail_file(path, max(10, min(400, lines))), "file": path.name}


@app.get("/api/logs/dashboard")
def api_dashboard_log(lines: int = 60):
    return {"log": _tail_file(LOGS_DIR / "dashboard.log", max(10, min(400, lines)))}


@app.post("/api/logs/clear")
def api_logs_clear(payload: dict):
    """清空日志文件（写入方以追加模式打开，截断后下一条日志从文件头继续）。"""
    name = str(payload.get("name", "")).strip()
    path = {"runner": RUNNER_LOG, "dashboard": LOGS_DIR / "dashboard.log"}.get(name)
    if name == "training":
        raise HTTPException(400, "训练日志按次归档，不支持清空（会随新训练自动分文件）")
    if path is None:
        raise HTTPException(400, "未知的日志类型")
    try:
        path.parent.mkdir(exist_ok=True)
        path.write_text("", encoding="utf-8")
    except OSError as exc:
        raise HTTPException(500, f"清空失败: {exc}")
    logger.info(f"[看板] {name} 日志已清空")
    return {"ok": True, "message": f"{name} 日志已清空"}


@app.get("/api/logs/export")
def api_logs_export(name: str = "runner"):
    path = {"runner": RUNNER_LOG, "dashboard": LOGS_DIR / "dashboard.log"}.get(name)
    if name == "training":
        path = _latest_train_log()
    if path is None or not path.exists():
        raise HTTPException(404, "日志文件不存在")
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8",
        filename=f"MT5AutoTrader_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )


# ── 账户净值采样（内存环形缓冲，看板重启后从零开始积累）───────────────

_EQUITY_SAMPLES: list[list[float]] = []
_EQUITY_MAX_POINTS = 4320
_EQUITY_LOCK = threading.RLock()
_EQUITY_ACCOUNT = None
_BALANCE_HISTORY_CACHE = (None, 0.0, [])
_BALANCE_HISTORY_TTL = 300


def _account_identity(ai):
    if ai is None:
        return None
    return (str(getattr(ai, "server", "")), int(ai.login), str(getattr(ai, "currency", "")))


def _sync_equity_account(ai):
    global _EQUITY_ACCOUNT, _BALANCE_HISTORY_CACHE
    key = _account_identity(ai)
    if key != _EQUITY_ACCOUNT:
        _EQUITY_ACCOUNT = key
        _EQUITY_SAMPLES.clear()
        _BALANCE_HISTORY_CACHE = (None, 0.0, [])
    return key


def _push_equity_sample(equity: float, balance: float | None = None) -> None:
    now = time.time()
    if _EQUITY_SAMPLES and now - _EQUITY_SAMPLES[-1][0] < 55:
        return
    row = [now, round(float(equity), 2)]
    if balance is not None:
        row.append(round(float(balance), 2))
    _EQUITY_SAMPLES.append(row)
    del _EQUITY_SAMPLES[:-_EQUITY_MAX_POINTS]


@app.get("/api/equity/history")
def api_equity_history(days: int = 90):
    days = max(1, min(365, int(days or 90)))
    with _EQUITY_LOCK:
        client = _get_mt5_client()
        ai = client.account_info() if client is not None else None
        key = _sync_equity_account(ai)
        if key is None:
            return {"points": [], "history": [], "account_key": None, "error": "MT5 未连接或未登录"}
        history = _balance_history_points(days=days, client=client, account=ai)
        if _account_identity(client.account_info()) != key:
            _sync_equity_account(None)
            raise HTTPException(409, "账户已切换，请重新读取历史曲线")
        _push_equity_sample(ai.equity, ai.balance)
        cutoff = time.time() - days * 86400
        samples = [p for p in _EQUITY_SAMPLES if p[0] >= cutoff]
        if key != _account_identity(client.account_info()):
            _sync_equity_account(None)
            raise HTTPException(409, "账户已切换，请重新读取历史曲线")
        coverage = {"start": samples[0][0] if samples else None,
                    "end": samples[-1][0] if samples else None,
                    "count": len(samples), "requested_start": cutoff}
        return {"points": samples, "history": history, "account_key": list(key), "days": days,
                "sample_coverage": coverage, "now": time.time()}


def _balance_history_points(days: int = 90, max_points: int = 400, *, client=None, account=None) -> list:
    global _BALANCE_HISTORY_CACHE
    with _EQUITY_LOCK:
        client = client or _get_mt5_client()
        ai = account if account is not None else (client.account_info() if client is not None else None)
        key = _sync_equity_account(ai)
        if key is None:
            return []
        cache_key = (key, days, max_points, float(ai.balance))
        cached_key, cached_ts, cached = _BALANCE_HISTORY_CACHE
        now = time.time()
        if cache_key == cached_key and now - cached_ts < _BALANCE_HISTORY_TTL:
            return cached
        import MetaTrader5 as mt5
        offset = client.server_time_offset() or 0
        start = now - days * 86400
        deals = mt5.history_deals_get(
            datetime.fromtimestamp(start + offset, timezone.utc),
            datetime.fromtimestamp(now + offset, timezone.utc),
        )
        if deals is None:
            raise HTTPException(503, "MT5 历史成交读取失败，请稍后刷新")
        deltas = sorted((float(d.time) - offset,
                         float(d.profit) + float(d.commission) + float(d.swap) + float(getattr(d, "fee", 0))) for d in deals)
        balance = float(ai.balance) - sum(delta for _, delta in deltas)
        points = [[start, round(balance, 2)]]
        if deltas:
            points.append([deltas[0][0] - 1, round(balance, 2)])
            stride = max(1, len(deltas) // max_points)
            for i, (ts, delta) in enumerate(deltas):
                balance += delta
                if i % stride == 0 or i == len(deltas) - 1:
                    points.append([ts, round(balance, 2)])
        else:
            points = [[start, round(balance, 2)]]
        points.append([now, round(float(ai.balance), 2)])
        if _account_identity(client.account_info()) != key:
            _sync_equity_account(None)
            raise HTTPException(409, "读取期间账户已切换，请刷新")
        _BALANCE_HISTORY_CACHE = (cache_key, now, points)
        return points


# ── 配置读写 ────────────────────────────────────────────────────────

def _strict_bool(payload: dict, key: str, *, required=False) -> bool | None:
    if key not in payload:
        if required:
            raise HTTPException(400, f"{key} must be a boolean")
        return None
    if type(payload[key]) is not bool:
        raise HTTPException(400, f"{key} must be a boolean")
    return payload[key]


def _strict_float(payload: dict, key: str, *, required=True, positive=False, allow_zero=False):
    if key not in payload or payload[key] is None or type(payload[key]) not in (int, float):
        if required:
            raise HTTPException(400, f"{key} must be a finite number")
        return None
    try:
        value = float(payload[key])
    except (OverflowError, ValueError):
        raise HTTPException(400, f"{key} must be finite")
    if not math.isfinite(value) or (positive and (value <= 0 if not allow_zero else value < 0)):
        raise HTTPException(400, f"{key} must be a valid number")
    return value


def _account_key(ai) -> list | None:
    key = _account_identity(ai)
    return list(key) if key is not None else None


def _require_account_key(payload: dict, client) -> list[str]:
    if not isinstance(payload.get("account_key"), list) or len(payload["account_key"]) != 3 or not isinstance(payload["account_key"][0], str) or type(payload["account_key"][1]) is not int or not isinstance(payload["account_key"][2], str):
        raise HTTPException(400, "account_key must come from a fresh account confirmation")
    current = _account_key(client.account_info() if client else None)
    if current is None or payload["account_key"] != current:
        raise HTTPException(409, "MT5 账户已切换，请重新确认当前账户")
    return current


def _request_contract(fn):
    from functools import wraps
    @wraps(fn)
    def guarded(payload: dict):
        for key in ("confirmed", "check_only", "dry_run", "clear_tp"):
            _strict_bool(payload, key)
        check_only = payload.get("check_only") is True
        if "check_only" in payload and fn.__name__ != "api_mt5_pending_order":
            raise HTTPException(400, "check_only is only valid for pending order preview")
        if fn.__name__ != "api_mt5_order_check" and not check_only and payload.get("confirmed") is not True:
            raise HTTPException(400, "操作需要 confirmed=true 二次确认")
        if "ticket" in payload or fn.__name__ not in ("api_mt5_order", "api_mt5_order_check", "api_mt5_pending_order"):
            ticket = payload.get("ticket")
            if type(ticket) is not int or ticket <= 0:
                raise HTTPException(400, "ticket must be a positive integer")
        for key in ("lot", "price", "volume", "sl", "tp"):
            required = (key == "lot" and fn.__name__ in ("api_mt5_order", "api_mt5_order_check", "api_mt5_pending_order") and payload.get("direction") != "CLOSE_ALL") or (key == "price" and fn.__name__ in ("api_mt5_pending_order", "api_mt5_pending_modify", "api_position_partial")) or (key == "sl" and fn.__name__ == "api_position_sl") or (key == "tp" and fn.__name__ == "api_position_tp")
            if key in payload or required:
                if payload.get(key) is None and key in ("sl", "tp") and not required:
                    continue
                _strict_float(payload, key, positive=True, allow_zero=key in ("sl", "tp", "volume") or fn.__name__ == "api_position_partial")
        if fn.__name__ == "api_position_partial" and payload.get("price", 0) > 0:
            _strict_float(payload, "volume", positive=True)
        if fn.__name__ == "api_position_sl" and payload.get("sl", 0) <= 0:
            raise HTTPException(400, "sl must be positive")
        for key in ("symbol",):
            if fn.__name__ in ("api_mt5_order", "api_mt5_order_check", "api_mt5_pending_order") and (not isinstance(payload.get(key), str) or not payload[key].strip()):
                raise HTTPException(400, "symbol is required")
        direction_key = "side" if fn.__name__ == "api_mt5_pending_order" else "direction"
        if fn.__name__ in ("api_mt5_order", "api_mt5_order_check", "api_mt5_pending_order"):
            allowed = ("BUY", "SELL", "CLOSE_ALL") if fn.__name__ == "api_mt5_order" else ("BUY", "SELL")
            if payload.get(direction_key) not in allowed:
                raise HTTPException(400, "invalid direction")
        client = _get_mt5_client()
        if client is None:
            raise HTTPException(503, "MT5 未连接")
        _require_account_key(payload, client)
        return fn(payload)
    return guarded


def _config_transaction(fn):
    from functools import wraps
    @wraps(fn)
    def guarded(payload: dict):
        with CONFIG_LOCK:
            return fn(payload)
    return guarded


@app.get("/api/config")
def api_config():
    return load_trader_config()


@app.put("/api/config")
@_config_transaction
def api_update_config(payload: dict):
    if not isinstance(payload, dict):
        raise HTTPException(400, "payload must be an object")
    allowed = {"dry_run", "confirmed", "account_key", "risk", "min_trade_exposure", "max_lot_per_trade",
               "max_open_positions", "magic_number", "deviation_points", "signal_bars", "kline_cache_dir"}
    if set(payload) - allowed:
        raise HTTPException(400, "未知配置字段")
    cfg = load_trader_config()
    if "dry_run" in payload and type(payload["dry_run"]) is not bool:
        raise HTTPException(400, "dry_run must be a boolean")
    if "confirmed" in payload and type(payload["confirmed"]) is not bool:
        raise HTTPException(400, "confirmed must be a boolean")
    if payload.get("dry_run") is False and cfg.get("dry_run", True) is True and payload.get("confirmed") is not True:
        raise HTTPException(400, "切换真实下单需要网页二次确认")
    if payload.get("dry_run") is False:
        if payload.get("confirmed") is not True:
            raise HTTPException(400, "切换真实下单需要网页二次确认")
        _require_account_key(payload, _get_mt5_client())
    candidate = json.loads(json.dumps(cfg))
    if "dry_run" in payload:
        candidate["dry_run"] = payload["dry_run"]
    risk = payload.get("risk")
    if "risk" in payload:
        if not isinstance(risk, dict):
            raise HTTPException(400, "risk must be an object")
        candidate.setdefault("risk", {}).update(risk)
    for key in ("min_trade_exposure", "max_lot_per_trade", "max_open_positions",
                "magic_number", "deviation_points", "signal_bars", "kline_cache_dir"):
        if key in payload:
            candidate[key] = payload[key]
    try:
        validate_trader_config(candidate)
        save_trader_config(candidate)
    except (TypeError, ValueError, OSError) as exc:
        raise HTTPException(400, f"配置无效: {exc}") from exc
    return {"ok": True, "config": candidate}


# ── Runner 启停 ─────────────────────────────────────────────────────

@app.post("/api/runner/start")
def api_runner_start():
    global _runner_process
    if os.environ.get("MT5AUTOTRADER_OFFLINE") == "1":
        raise HTTPException(409, "离线看板模式不允许启动交易 Runner")
    if not _runner_start_lock.acquire(blocking=False):
        raise HTTPException(409, "runner 正在启动")
    try:
        runner = _runner_alive()
        if runner.get("process_alive", runner["alive"]):
            return {"ok": True, "message": "runner 进程已存在（可能正在等待连接）", "process_alive": True}
        STOP_FILE.unlink(missing_ok=True)
        LOGS_DIR.mkdir(exist_ok=True)
        boot = ("import sys, runpy;" f"sys.path.insert(0, {str(SRC)!r});"
                "runpy.run_module('trading.runner', run_name='__main__')")
        with open(RUNNER_LOG, "a", encoding="utf-8") as log_fh:
            proc = subprocess.Popen([_venv_python(), "-c", boot], cwd=str(SRC),
                                    stdout=log_fh, stderr=subprocess.STDOUT,
                                    creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP)
        _runner_process = proc
        RUNNER_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
        logger.info(f"[看板] runner 已启动 pid={proc.pid}")
        return {"ok": True, "pid": proc.pid, "process_alive": True, "ready": False}
    finally:
        _runner_start_lock.release()


@app.post("/api/runner/stop")
def api_runner_stop():
    STOP_FILE.write_text("STOP", encoding="utf-8")
    pid = 0
    if RUNNER_PID_FILE.exists():
        try:
            pid = int(RUNNER_PID_FILE.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            pid = 0
    # 等待进程退出（最多 15 秒）
    for _ in range(30):
        time.sleep(0.5)
        if pid and not _pid_alive(pid):
            break
        try:
            if STATUS_FILE.exists():
                st = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
                if st.get("phase") == "stopped":
                    break
        except (json.JSONDecodeError, OSError):
            pass
    if pid and _runner_process is not None and _runner_process.pid == pid and _launcher_pid_verified(pid):
        # Only terminate the verified launcher; never an unrelated reused PID.
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", "Stop-Process", "-Id", str(pid), "-Force"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        logger.warning(f"[看板] runner pid={pid} 未响应停止信号，已强制结束")
    return {"ok": True, "message": "停止指令已发送"}


# ── 策略库 ──────────────────────────────────────────────────────────

def _strategy_summary(path: Path) -> dict:
    try:
        meta = load_strategy_file(path)
        try:
            display_path = str(path.relative_to(ROOT))
        except ValueError:
            display_path = str(path)
        # 旧策略文件可能没有 formula_decoded，用 token 序列现场解码补上
        decoded = meta.get("formula_decoded") or formula_preview(meta["formula"])
        return {
            "file": display_path,
            "name": path.name,
            "symbol": meta["symbol"],
            "timeframe": meta["timeframe"],
            "best_score": meta["best_score"],
            "formula_decoded": decoded,
            "vocab_ok": True,
        }
    except StrategyError as exc:
        return {"file": str(path), "name": path.name, "error": str(exc), "vocab_ok": False}


@app.get("/api/strategies")
def api_strategies():
    items = []
    if STRATEGIES_DIR.exists():
        for p in sorted(STRATEGIES_DIR.glob("*.json")):
            items.append(_strategy_summary(p))
    cfg = load_trader_config()
    return {"strategies": items, "bindings": cfg.get("bindings", [])}


def _resolve_strategy_path(name: str) -> Path:
    """把前端传来的策略名解析成 strategies/ 下的真实文件；越界或非 json 直接拒绝。"""
    if not isinstance(name, str) or not name.strip():
        raise HTTPException(400, "缺少策略名")
    name = name.strip().replace("\\", "/")
    parts = Path(name).parts
    if Path(name).is_absolute() or any(p in (".", "..") for p in parts) or len(parts) > 2 or (len(parts) == 2 and parts[0] != "strategies"):
        raise HTTPException(400, "非法策略路径")
    name = Path(name).name
    if not name.lower().endswith(".json"):
        name += ".json"
    target = (STRATEGIES_DIR / name).resolve()
    if target.parent != STRATEGIES_DIR.resolve():
        raise HTTPException(400, "非法文件名")
    return target


@app.post("/api/strategies/delete")
@_config_transaction
def api_strategy_delete(payload: dict):
    target = _resolve_strategy_path(payload.get("name", ""))
    if not target.exists():
        raise HTTPException(404, f"文件不存在: {target.name}")
    # 同时解除绑定（配置里可能是 / 或 \ 分隔，统一比较文件名）
    cfg = load_trader_config()
    cfg["bindings"] = [
        b for b in cfg.get("bindings", [])
        if Path(str(b.get("strategy_file", "")).replace("\\", "/")).name != target.name
    ]
    save_trader_config(cfg)
    target.unlink()
    logger.info(f"[策略] 已删除 {target.name}")
    return {"ok": True, "deleted": target.name, "bindings": cfg["bindings"]}


@app.post("/api/strategies/bind")
@_config_transaction
def api_strategy_bind(payload: dict):
    strategy_file = payload.get("strategy_file")
    symbol = payload.get("symbol")
    lot = _strict_float(payload, "lot", positive=True)
    if not isinstance(symbol, str) or not symbol.strip():
        raise HTTPException(400, "symbol is required")
    path = _resolve_strategy_path(strategy_file)
    if not path.is_file():
        raise HTTPException(400, "策略文件不存在")
    strategy_file = path.relative_to(ROOT).as_posix()
    symbol = symbol.strip()
    cfg = load_trader_config()
    bindings = [b for b in cfg.get("bindings", []) if b.get("symbol") != symbol]
    bindings.append({"strategy_file": strategy_file, "symbol": symbol, "lot": lot})
    cfg["bindings"] = bindings
    try:
        save_trader_config(cfg)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    return {"ok": True, "bindings": bindings}


@app.post("/api/strategies/unbind")
@_config_transaction
def api_strategy_unbind(payload: dict):
    symbol = payload.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip():
        raise HTTPException(400, "symbol is required")
    symbol = symbol.strip()
    cfg = load_trader_config()
    cfg["bindings"] = [b for b in cfg.get("bindings", []) if b.get("symbol") != symbol]
    save_trader_config(cfg)
    return {"ok": True, "bindings": cfg["bindings"]}


@app.get("/api/mt5/symbols")
def api_mt5_symbols():
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接（请先打开 TMGM 终端并登录）")
    try:
        import MetaTrader5 as mt5
        before = _account_identity(mt5.account_info())
        if before is None:
            raise HTTPException(503, "MT5 未登录，无法读取品种")
        symbols = mt5.symbols_get()
        names = sorted({s.name for s in (symbols or []) if getattr(s, "visible", False)})
        if _account_identity(mt5.account_info()) != before:
            raise HTTPException(409, "读取品种期间 MT5 账户或服务商已切换，请重试")
        return {"symbols": names[:2000], "account_key": list(before)}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/mt5/position/partial")
@_request_contract
def api_position_partial(payload: dict):
    """设置/取消"止盈一半"计划（到价分批平仓）。

    写入覆盖文件，Runner 下个循环读取并更新台账计划：
      price > 0 且 0 < volume < 持仓手数 → 设置计划
      price <= 0（volume 忽略）          → 取消该计划的触发
    """
    ticket = int(payload.get("ticket", 0) or 0)
    price = float(payload.get("price", 0) or 0)
    volume = float(payload.get("volume", 0) or 0)
    if ticket <= 0:
        raise HTTPException(400, "ticket 无效")
    client = _get_mt5_client()
    p = _find_any_position(client, ticket) if client else None
    info = _find_status_position(ticket) if p is None else None
    if p is not None:
        lot = float(p.volume)
    elif info is not None:
        lot = float(info.get("volume", 0) or 0)
    else:
        raise HTTPException(404, f"找不到持仓 ticket={ticket}")
    if price <= 0:
        override = {"price": 0.0, "close_volume": 0.0, "ts": time.time()}
        message = "已取消该仓位的止盈一半计划"
    else:
        if lot <= 0 or volume <= 0 or volume >= lot:
            raise HTTPException(
                400, f"平仓手数必须在 0 与 {lot} 之间（全部平仓请用「平仓」按钮）")
        override = {"price": price, "close_volume": volume, "ts": time.time()}
        message = f"止盈一半已设置：到 {price} 平 {volume}手"
    override["account_key"] = list(payload["account_key"])
    path = LOGS_DIR / "sr_partial_overrides.json"
    _require_account_key(payload, client)
    p = _find_any_position(client, ticket) if client else None
    if p is not None and price > 0 and not (0 < volume < float(p.volume)):
        raise HTTPException(409, "持仓手数已变化，请重新确认分批手数")
    old_tp = float(getattr(p, "tp", 0) or 0) if p else 0
    cleared = False
    if payload.get("clear_tp") is True and p is not None and old_tp > 0:
        _require_account_key(payload, client)
        client.dry_run = False
        cleared = client.modify_sl(str(p.symbol), ticket, new_sl=float(getattr(p, "sl", 0) or 0), tp=0)
        if not cleared:
            return {"ok": False, "message": "清除原止盈失败，未写入分批止盈计划"}
    try:
        _require_account_key(payload, client)
        _atomic_override(path, ticket, override)
    except (OSError, ValueError, HTTPException) as exc:
        restored = False
        if cleared:
            try:
                _require_account_key(payload, client)
                restored = client.modify_sl(str(p.symbol), ticket, new_sl=float(getattr(p, "sl", 0) or 0), tp=old_tp)
            except Exception:
                pass
        raise HTTPException(500, f"计划写入失败；原 TP {'已恢复' if restored else '请核实'}: {exc}") from exc
    pending = not _runner_alive()["alive"]
    logger.info(f"[持仓管理] ticket={ticket} 止盈一半覆盖: {override}")
    return {"ok": True, "pending": pending,
            "message": message + ("（Runner 未就绪，计划待处理，不会立即执行）" if pending else "")}


# ── 支撑/阻力位（S/R）────────────────────────────────────────────────

@app.get("/api/sr/levels")
def api_sr_levels(symbol: str, timeframe: str = "H1", bars: int = 1000):
    """当前关键位（支撑带/压力带），供页面展示与手动交易参考。"""
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接（请先打开终端并登录）")
    symbol = symbol.strip()
    if not symbol:
        raise HTTPException(400, "缺少品种")
    from trading.sr import detect_levels
    tf = Config.get_timeframe(timeframe)
    count = max(200, min(int(bars or 1000), 5000))
    rates = client.copy_rates(symbol, tf, count)
    if rates is None or len(rates) < 60:
        raise HTTPException(400, f"K线数据不足（品种 {symbol} {timeframe}）")
    levels, info = detect_levels(symbol, rates[:-1])  # 丢弃形成中的最后一根
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "levels": levels,
        "info": info,
        "server_offset_sec": client.server_time_offset(),
        "sr_enabled": bool(load_trader_config().get("risk", {}).get("sr_enabled", True)),
    }


@app.get("/api/sr/chart")
def api_sr_chart(symbol: str, timeframe: str = "H1", bars: int = 300):
    """K线 + 关键位数据（供总览页交互式图表）。包含形成中的当前 bar。"""
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接（请先打开终端并登录）")
    symbol = symbol.strip()
    if not symbol:
        raise HTTPException(400, "缺少品种")
    key = _account_identity(client.account_info())
    if key is None:
        raise HTTPException(503, "MT5 未登录")
    from trading.sr import detect_levels
    tf = Config.get_timeframe(timeframe)
    want = max(120, min(int(bars or 300), 2000))
    rates = client.copy_rates(symbol, tf, want + 1)
    if rates is None or len(rates) < 60:
        raise HTTPException(400, f"K线数据不足（品种 {symbol} {timeframe}）")
    closed = rates[:-1]
    levels, info = detect_levels(symbol, closed)

    def row(r):
        return [int(r["time"]), round(float(r["open"]), 8), round(float(r["high"]), 8),
                round(float(r["low"]), 8), round(float(r["close"]), 8)]

    last = int(want)
    candles = [row(r) for r in closed[-last:]]
    forming = row(rates[-1]) if len(rates) else None
    offset = client.server_time_offset()
    if key != _account_identity(client.account_info()):
        raise HTTPException(409, "账户已切换，请刷新图表")
    return {
        "account_key": list(key),
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": candles,
        "forming": forming,
        "zones": levels,
        "info": info,
        "server_offset_sec": client.server_time_offset(),
        "sr_enabled": bool(load_trader_config().get("risk", {}).get("sr_enabled", True)),
    }


# ── 手动交易（二次确认）─────────────────────────────────────────────

@app.post("/api/mt5/order_check")
@_request_contract
def api_mt5_order_check(payload: dict):
    """只做 MT5 order_check，不发送订单。"""
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接")
    symbol = str(payload.get("symbol", "")).strip()
    direction = str(payload.get("direction", "")).upper()
    lot = float(payload.get("lot", 0.01) or 0.01)
    if direction not in ("BUY", "SELL"):
        raise HTTPException(400, "预检查 direction 必须是 BUY 或 SELL")
    result = client.order_check(symbol, direction, lot)
    return result


@app.post("/api/mt5/order")
@_request_contract
def api_mt5_order(payload: dict):
    if not payload.get("confirmed"):
        raise HTTPException(400, "手动下单需要二次确认")
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接")
    if client.account_info() is None:
        raise HTTPException(400, "MT5 账号信息不可用")
    symbol = str(payload.get("symbol", "")).strip()
    direction = str(payload.get("direction", "")).upper()
    lot = float(payload.get("lot", 0.01) or 0.01)
    if not symbol:
        raise HTTPException(400, "必须填写品种")
    if direction not in ("BUY", "SELL", "CLOSE_ALL"):
        raise HTTPException(400, "direction 必须是 BUY/SELL/CLOSE_ALL")
    # 手动订单明确由用户确认后执行，不受 Runner 的 dry-run 配置影响。
    _require_account_key(payload, client)
    client.dry_run = False
    if direction == "CLOSE_ALL":
        ok = client.close_symbol_all(symbol)
        return {"ok": ok, "message": "全部平仓完成" if ok else "平仓失败，请查看日志"}
    result = client.market_open(symbol, direction, lot, comment="MT5AutoTrader manual")
    if result["ok"]:
        return {"ok": True, "message": "成交"}
    hint = result.get("hint") or ""
    detail = f"失败 retcode={result['retcode']} {result['comment']}"
    return {"ok": False, "message": (detail + "｜" + hint) if hint else detail}


# ── 挂单（限价/条件单，到价后按券商规则成交；网页二次确认）────────────

@app.get("/api/mt5/pending/list")
def api_mt5_pending_list(symbol: str | None = None):
    """当前挂单（全部 magic，含手动下的）。symbol 给定时只返回该品种。"""
    client = _get_mt5_client()
    if client is None:
        return {"orders": [], "error": "MT5 未连接"}
    from trading.mt5_client import pending_to_dict
    key = _account_identity(client.account_info())
    orders = client.get_orders((symbol or "").strip() or None)
    rows = [pending_to_dict(o) for o in orders]
    if key is None or key != _account_identity(client.account_info()):
        raise HTTPException(409, "账户已切换，请重新读取挂单")
    return {"orders": rows, "account_key": list(key)}


@app.post("/api/mt5/pending/order")
@_request_contract
def api_mt5_pending_order(payload: dict):
    """下挂单。check_only=true 时只做 order_check（只读，绝不发单）。"""
    check_only = bool(payload.get("check_only"))
    if not check_only and not payload.get("confirmed"):
        raise HTTPException(400, "挂单需要二次确认")
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接")
    if client.account_info() is None:
        raise HTTPException(400, "MT5 账号信息不可用")
    symbol = str(payload.get("symbol", "")).strip()
    side = str(payload.get("side", "")).upper()
    price = float(payload.get("price", 0) or 0)
    lot = float(payload.get("lot", 0) or 0)
    sl = float(payload.get("sl", 0) or 0)
    tp = float(payload.get("tp", 0) or 0)
    if not symbol:
        raise HTTPException(400, "必须填写品种")
    if side not in ("BUY", "SELL"):
        raise HTTPException(400, "side 必须是 BUY 或 SELL")
    if price <= 0:
        raise HTTPException(400, "触发价格无效")
    if lot <= 0:
        raise HTTPException(400, "手数无效")
    _require_account_key(payload, client)
    if not check_only:
        client.dry_run = False  # 用户明确确认后的真实挂单
    result = client.pending_order(symbol, side, price, lot,
                                  sl=sl or None, tp=tp or None, check_only=check_only)
    kind = result.get("kind")
    if result["ok"]:
        if check_only:
            msg = f"检查通过：将下 {side} {kind} {lot} 手 @ {price}"
        else:
            msg = f"挂单成功：{side} {kind} {lot} 手 @ {price}"
            logger.info(f"[挂单] {msg} SL={sl or '无'} TP={tp or '无'}")
    else:
        hint = retcode_hint(result.get("retcode"))
        msg = f"失败 retcode={result.get('retcode')} {result.get('comment')}"
        if hint:
            msg += "｜" + hint
    return {"ok": result["ok"], "message": msg, "kind": kind, "retcode": result.get("retcode")}


@app.post("/api/mt5/pending/modify")
@_request_contract
def api_mt5_pending_modify(payload: dict):
    """修改挂单触发价（止损止盈随平移，未给的保持原值）。网页二次确认。"""
    if not payload.get("confirmed"):
        raise HTTPException(400, "修改挂单需要二次确认")
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接")
    ticket = int(payload.get("ticket", 0) or 0)
    price = float(payload.get("price", 0) or 0)
    sl = float(payload.get("sl", 0) or 0)
    tp = float(payload.get("tp", 0) or 0)
    if ticket <= 0 or price <= 0:
        raise HTTPException(400, "ticket 或触发价无效")
    _require_account_key(payload, client)
    client.dry_run = False  # 用户明确确认后的真实动作
    result = client.modify_order(ticket, price, sl=sl or None, tp=tp or None)
    if result["ok"]:
        logger.info(f"[挂单] ticket={ticket} 触发价改为 {price} SL={sl or '原值'} TP={tp or '原值'}")
        return {"ok": True, "message": f"挂单 ticket={ticket} 已改为 @ {price}"}
    hint = retcode_hint(result.get("retcode"))
    msg = f"修改失败 retcode={result.get('retcode')} {result.get('comment')}"
    return {"ok": False, "message": (msg + "｜" + hint) if hint else msg}


@app.post("/api/mt5/pending/cancel")
@_request_contract
def api_mt5_pending_cancel(payload: dict):
    if not payload.get("confirmed"):
        raise HTTPException(400, "撤销挂单需要二次确认")
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接")
    ticket = int(payload.get("ticket", 0) or 0)
    if ticket <= 0:
        raise HTTPException(400, "ticket 无效")
    _require_account_key(payload, client)
    client.dry_run = False  # 用户明确确认后的真实动作
    result = client.cancel_order(ticket)
    if result["ok"]:
        logger.info(f"[挂单] 已撤销 ticket={ticket}")
        return {"ok": True, "message": f"挂单 ticket={ticket} 已撤销"}
    hint = retcode_hint(result.get("retcode"))
    msg = f"撤销失败 retcode={result.get('retcode')} {result.get('comment')}"
    return {"ok": False, "message": (msg + "｜" + hint) if hint else msg}


# ── 持仓管理（平仓 / 修改止损 / 修改止盈）────────────────────────────

def _find_any_position(client, ticket: int):
    """按 ticket 查找持仓（本软件 magic 内）。找不到返回 None。"""
    for p in client.get_positions():
        if int(p.ticket) == int(ticket):
            return p
    return None


@app.post("/api/mt5/position/close")
@_request_contract
def api_position_close(payload: dict):
    if not payload.get("confirmed"):
        raise HTTPException(400, "平仓需要二次确认")
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接")
    ticket = int(payload.get("ticket", 0) or 0)
    p = _find_any_position(client, ticket)
    if p is None:
        raise HTTPException(404, f"找不到持仓 ticket={ticket}（可能已平仓）")
    _require_account_key(payload, client)
    client.dry_run = False  # 网页明确确认后的真实动作
    ok = client.close_position(str(p.symbol), ticket)
    if ok:
        logger.info(f"[持仓管理] 已平仓 ticket={ticket} {p.symbol}")
        return {"ok": True, "message": f"{p.symbol} ticket={ticket} 平仓成功"}
    return {"ok": False, "message": "平仓失败，请查看日志或重试"}


def _atomic_override(path: Path, ticket: int, override: dict) -> None:
    from config import _atomic_json_write
    with CONFIG_LOCK:
        expected = override.get("account_key")
        _require_account_key({"account_key": expected}, _get_mt5_client())
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        if not isinstance(data, dict):
            raise ValueError("override file must be an object")
        data[str(ticket)] = override
        _atomic_json_write(path, data)


def _write_sl_override(ticket: int, sl: float, applied_to_mt5: bool, account_key: list) -> None:
    path = LOGS_DIR / "sl_overrides.json"
    try:
        _atomic_override(path, ticket, {"sl": sl, "ts": time.time(),
                                      "applied_to_mt5": bool(applied_to_mt5), "account_key": list(account_key)})
    except (OSError, ValueError) as exc:
        raise HTTPException(500, f"止损覆盖写入失败（请核实终端止损）: {exc}") from exc


def _find_status_position(ticket: int) -> dict | None:
    """从 Runner 状态里找持仓（dry-run 台账和 MT5 实仓都在里面）。"""
    try:
        client = _get_mt5_client()
        key = _account_identity(client.account_info() if client else None)
        st = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        account = st.get("account") or {}
        status_key = (str(account.get("server", "")), account.get("login"), str(account.get("currency", "")))
        if key is None or key != status_key or not _runner_alive().get("process_alive"):
            return None
        return next((q for q in (st.get("positions") or [])
                     if q.get("dry_run") is True and int(q.get("ticket", 0)) == int(ticket)), None)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


@app.post("/api/mt5/position/sl")
@_request_contract
def api_position_sl(payload: dict):
    ticket = int(payload.get("ticket", 0) or 0)
    sl = float(payload.get("sl", 0) or 0)
    if sl <= 0:
        raise HTTPException(400, "止损价格无效")
    client = _get_mt5_client()
    mt5_pos = _find_any_position(client, ticket) if client else None
    if mt5_pos is not None:
        # 真实 MT5 持仓：直接修改，并写覆盖文件让 Runner 豁免自动拉回
        _require_account_key(payload, client)
        client.dry_run = False
        ok = client.modify_sl(str(mt5_pos.symbol), ticket, new_sl=sl,
                              tp=float(getattr(mt5_pos, "tp", 0) or 0))
        if ok:
            _write_sl_override(ticket, sl, applied_to_mt5=True, account_key=payload["account_key"])
            logger.info(f"[持仓管理] ticket={ticket} {mt5_pos.symbol} 止损手动改为 {sl}")
            return {"ok": True, "message": f"止损已改为 {sl}（手动管理，不再自动拉回）"}
        return {"ok": False, "message": "修改失败（可能离市价太近或市场关闭）"}
    # MT5 上没有：可能是 dry-run 模拟仓位，写覆盖文件交由 Runner 更新台账
    info = _find_status_position(ticket)
    if info is None:
        raise HTTPException(404, f"找不到持仓 ticket={ticket}")
    _write_sl_override(ticket, sl, applied_to_mt5=False, account_key=payload["account_key"])
    logger.info(f"[持仓管理] ticket={ticket} {info.get('symbol')} 止损手动改为 {sl}（模拟仓）")
    return {"ok": True, "message": f"止损已改为 {sl}（模拟仓，Runner 下圈生效）"}


@app.post("/api/mt5/position/tp")
@_request_contract
def api_position_tp(payload: dict):
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接")
    ticket = int(payload.get("ticket", 0) or 0)
    tp = float(payload.get("tp", 0) or 0)
    p = _find_any_position(client, ticket)
    if p is None:
        raise HTTPException(404, f"找不到持仓 ticket={ticket}")
    _require_account_key(payload, client)
    client.dry_run = False
    ok = client.modify_sl(str(p.symbol), ticket, new_sl=float(getattr(p, "sl", 0) or 0), tp=tp)
    if ok:
        logger.info(f"[持仓管理] ticket={ticket} {p.symbol} 止盈改为 {tp or '清除'}")
        return {"ok": True, "message": f"止盈已改为 {tp}" if tp > 0 else "止盈已清除"}
    return {"ok": False, "message": "修改失败（可能离市价太近或市场关闭）"}


# ── 历史成交 ────────────────────────────────────────────────────────

@app.get("/api/mt5/history")
def api_mt5_history(days: int = 30, scope: str = "mine"):
    """从 MT5 读取真实成交历史，按 position 归并成交易记录。

    scope: "mine"=只看本软件 magic 的成交（默认）；"all"=账户全部成交（复盘用）。
    """
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接")
    import MetaTrader5 as mt5
    from trading.history import group_history_deals
    days = max(1, min(365, int(days or 30)))
    scope = "all" if str(scope or "").lower() == "all" else "mine"
    before = _account_identity(client.account_info())
    if before is None:
        raise HTTPException(503, "MT5 未登录")
    server_offset = client.server_time_offset() or 0
    now = time.time()
    # Broker deal epochs encode server wall time; naive datetimes are
    # reinterpreted in the Windows timezone by the native MT5 bridge.
    to = datetime.fromtimestamp(now + server_offset, timezone.utc)
    frm = datetime.fromtimestamp(now - days * 86400 + server_offset, timezone.utc)
    try:
        deals = mt5.history_deals_get(frm, to)
        if deals is None:
            raise HTTPException(503, "MT5 历史读取失败")
    except Exception as exc:
        raise HTTPException(500, f"读取历史失败: {exc}")
    grouped = group_history_deals(deals, magic=None if scope == "all" else client.magic)
    if before != _account_identity(client.account_info()):
        raise HTTPException(409, "账户已切换，请重新读取历史")
    return {
        "account_key": list(before),
        "server_offset_sec": server_offset,
        "total_deals": len(deals),
        "closed": grouped["closed"],
        "open": grouped["open"],
        "days": days,
        "scope": scope,
    }


# ── 数据文件 ────────────────────────────────────────────────────────

@app.get("/api/data/files")
def api_data_files():
    cfg = load_trader_config()
    data_dir = Path(cfg.get("kline_cache_dir", r"D:\K线数据"))
    files = []
    if data_dir.exists():
        for p in sorted(data_dir.glob("*.parquet"), key=lambda x: x.stat().st_mtime, reverse=True):
            item = {
                "path": str(p),
                "name": p.name,
                "size_mb": round(p.stat().st_size / 1048576, 1),
                "mtime": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            }
            # 只读读取行数/时间范围；读取失败不影响文件列表
            try:
                import pandas as pd
                df = pd.read_parquet(p, columns=["time"])
                item["bars"] = int(len(df))
                if len(df):
                    item["start_time"] = int(df["time"].iloc[0])
                    item["end_time"] = int(df["time"].iloc[-1])
            except Exception:
                item["bars"] = None
            files.append(item)
    return {"data_dir": str(data_dir), "files": files}


# ── 训练 ────────────────────────────────────────────────────────────

def _job_transaction(fn):
    from functools import wraps
    @wraps(fn)
    def guarded(*args, **kwargs):
        with _job_lock:
            return fn(*args, **kwargs)
    return guarded


class JobManager:
    """同一时刻只允许一个训练/回测子进程。"""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.proc: subprocess.Popen | None = None
        self.log_file: Path | None = None
        self.started_at: str | None = None
        self.args: dict = {}

    @_job_transaction
    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: list[str], log_file: Path, args: dict) -> dict:
        with _job_lock:
            if self.running() or training_job.running() or backtest_job.running():
                raise HTTPException(409, "已有训练/回测任务运行中")
            log_file.parent.mkdir(exist_ok=True)
            env = dict(os.environ)
            env.setdefault("MPLBACKEND", "Agg")
            env["PYTHONUNBUFFERED"] = "1"
            if self.kind == "训练":
                (ROOT / "TRAIN_STOP").unlink(missing_ok=True)
            with open(log_file, "w", encoding="utf-8") as fh:
                self.proc = subprocess.Popen(
                    cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                    creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP, env=env,
                )
            self.log_file = log_file
            self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.args = args
            logger.info(f"[{self.kind}] 启动 pid={self.proc.pid} args={args}")
            return {"ok": True, "pid": self.proc.pid}

    @_job_transaction
    def stop(self) -> dict:
        if not self.running():
            return {"ok": True, "message": "没有在运行的任务"}
        self.proc.terminate()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        logger.info(f"[{self.kind}] 已停止")
        return {"ok": True, "message": "已停止"}

    @_job_transaction
    def status(self) -> dict:
        running = self.running()
        tail = _tail_file(self.log_file, 40) if self.log_file else ""
        progress = _parse_progress(tail)
        return {
            "running": running,
            "log_file": str(self.log_file or ""),
            "log_tail": tail,
            "started_at": self.started_at,
            "args": self.args,
            "progress": progress,
            "exit_code": None if running else (self.proc.returncode if self.proc else None),
        }


def _parse_progress(tail: str) -> dict:
    """从训练日志尾部解析 [N/total] 步进度。"""
    import re
    result = {"step": None, "total": None, "best": None}
    for line in reversed(tail.splitlines()):
        m = re.search(r"\[(\d+)/(\d+)\]", line)
        if m:
            result["step"] = int(m.group(1))
            result["total"] = int(m.group(2))
        mb = re.search(r"[Bb]est[=:\s]+([0-9.]+)", line)
        if mb and result["best"] is None:
            try:
                result["best"] = float(mb.group(1))
            except ValueError:
                pass
        if result["step"] is not None and result["best"] is not None:
            break
    return result


training_job = JobManager("训练")
backtest_job = JobManager("回测")


@app.post("/api/training/start")
def api_training_start(payload: dict):
    with _job_lock:
        if training_job.running() or backtest_job.running():
            raise HTTPException(409, "已有训练/回测任务运行中")
    direct_mt5 = bool(payload.get("direct_mt5"))
    from_scratch = bool(payload.get("from_scratch"))
    steps = int(payload.get("steps", 0) or 0)
    timeframe = str(payload.get("timeframe", "H1") or "H1").upper()
    islands = int(payload.get("islands", 0) or 0)
    resume_file = str(payload.get("resume_file", "") or "").strip()
    if resume_file:
        candidate = Path(resume_file).resolve()
        if candidate.parent != (ROOT / "checkpoints").resolve() or candidate.suffix != ".pt" or not candidate.is_file():
            raise HTTPException(400, "续训文件必须位于 checkpoints/*.pt")
        resume_file = str(candidate)
    if timeframe not in Config._TF_SECONDS:
        raise HTTPException(400, "timeframe 无效")
    if direct_mt5:
        if os.environ.get("MT5AUTOTRADER_OFFLINE") == "1":
            raise HTTPException(409, "离线看板模式不允许从 MT5 获取训练数据")
        symbol = str(payload.get("symbol", "")).strip()
        bars = int(payload.get("bars", 6000) or 6000)
        if not symbol:
            raise HTTPException(400, "MT5 直连训练需要填写品种")
        if bars < 800:
            raise HTTPException(400, "MT5 直连训练至少获取 800 根 K线")
        cmd = [_venv_python(), "-u", "src/mt5_train.py", "--symbol", symbol,
               "--bars", str(bars), "--timeframe", timeframe]
        if steps > 0:
            cmd.extend(["--steps", str(steps)])
        if islands > 1:
            cmd.extend(["--islands", str(islands)])
        if resume_file:
            cmd.extend(["--resume-file", resume_file])
        if from_scratch:
            cmd.append("--from-scratch")
        data_file = "（由 MT5 直连获取）"
        log_name = f"{symbol}_{timeframe}"
    else:
        data_file = str(payload.get("data_file", "")).strip()
        if not data_file or not Path(data_file).exists():
            raise HTTPException(400, "数据文件不存在")
        if islands > 1:
            cmd = [_venv_python(), "-u", "src/train_island.py",
                   "--data-file", data_file, "--islands", str(islands)]
        else:
            cmd = [_venv_python(), "-u", "src/train_file.py", "--data-file", data_file]
        if from_scratch:
            cmd.append("--from-scratch")
        if steps > 0:
            cmd.extend(["--steps", str(steps)])
        if resume_file:
            cmd.extend(["--resume-file", resume_file])
        log_name = Path(data_file).stem
    if not log_name or any(c in log_name for c in "/\\:") or log_name in (".", ".."):
        raise HTTPException(400, "非法训练日志标签")
    log_file = LOGS_DIR / f"train_{log_name}_{int(time.time())}.log"
    result = training_job.start(cmd, log_file, {
        "data_file": data_file, "direct_mt5": direct_mt5,
        "from_scratch": from_scratch, "steps": steps, "timeframe": timeframe,
        "islands": islands, "resume_file": resume_file,
    })
    result["log"] = str(log_file)
    return result


@app.get("/api/training/checkpoints")
def api_training_checkpoints(data_file: str = ""):
    """列出某数据文件（品种+周期）可用的检查点，供前端可视化选择续训起点。

    返回单引擎 ckpt_ 与岛模式 island_ckpt_ 两类，各带步数、最优分、时间、大小。
    """
    data_file = str(data_file or "").strip()
    if not data_file:
        raise HTTPException(400, "缺少 data_file")
    try:
        from data_pipeline.parquet_manager import inspect_parquet_file
        info = inspect_parquet_file(data_file)
        symbol, tf = info["symbol"], info["timeframe"]
    except Exception as exc:
        raise HTTPException(400, f"数据文件无法解析: {exc}")
    tag = f"{symbol}_{tf}" if tf else symbol
    ck_dir = ROOT / "checkpoints"
    singles: list[dict] = []
    islands: list[dict] = []
    if ck_dir.exists():
        import re as _re
        for p in sorted(ck_dir.glob("*.pt")):
            m = _re.match(r"(island_)?ckpt_(.+?)_step_(\d+)\.pt$", p.name)
            if not m:
                continue
            is_island = bool(m.group(1))
            entry_tag = m.group(2)
            # 同时列出：本品种本周期（新命名）、本品种旧命名（无周期）、
            # 以及本品种其它周期（方便用户看清/删除），按 symbol 前缀匹配
            if not (entry_tag == tag or entry_tag == symbol
                    or entry_tag.startswith(symbol + "_")):
                continue
            step = int(m.group(3))
            try:
                st = p.stat()
                entry = {
                    "file": str(p), "name": p.name, "step": step, "tag": entry_tag,
                    "current": entry_tag == tag,
                    "mtime": datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M"),
                    "size_mb": round(st.st_size / 1048576, 1),
                }
            except OSError:
                continue
            (islands if is_island else singles).append(entry)
    singles.sort(key=lambda x: (x["tag"] != tag, x["step"]))
    islands.sort(key=lambda x: (x["tag"] != tag, x["step"]))
    return {"tag": tag, "symbol": symbol, "timeframe": tf,
            "single": singles, "island": islands}


@app.post("/api/training/checkpoint/delete")
def api_delete_checkpoint(payload: dict):
    """删除指定检查点文件（仅限 checkpoints/ 下的 .pt），给用户清理占用。"""
    name = str(payload.get("name", "") or "").strip()
    if not name:
        raise HTTPException(400, "缺少检查点名")
    ck_dir = (ROOT / "checkpoints").resolve()
    target = (ck_dir / Path(name).name).resolve()
    if Path(name).name != name or target.parent != ck_dir or target.suffix != ".pt":
        raise HTTPException(400, "非法检查点路径")
    if not target.exists():
        return {"ok": True, "message": "文件已不存在", "deleted": name}
    try:
        size = target.stat().st_size
        target.unlink()
    except OSError as exc:
        raise HTTPException(500, f"删除失败: {exc}")
    logger.info(f"[看板] 已删除检查点 {target.name}（{size/1048576:.1f}MB）")
    return {"ok": True, "message": f"已删除 {target.name}",
            "freed_mb": round(size / 1048576, 1)}


@app.post("/api/training/stop")
@_job_transaction
def api_training_stop():
    """安全停止：写 TRAIN_STOP 信号 → 引擎存完检查点/策略后自己退出；
    超时未退才强杀兜底。这样「停止训练」不再丢进度。"""
    stop_flag = ROOT / "TRAIN_STOP"
    try:
        stop_flag.write_text("STOP", encoding="utf-8")
    except OSError as exc:
        raise HTTPException(500, f"写停止信号失败: {exc}")
    if not training_job.running():
        stop_flag.unlink(missing_ok=True)
        return {"ok": True, "message": "没有在运行的任务"}
    for _ in range(40):  # 最多等 40 秒安全退出
        time.sleep(1)
        if not training_job.running():
            stop_flag.unlink(missing_ok=True)
            logger.info("[看板] 训练已安全停止（进度已保存）")
            return {"ok": True, "message": "已安全停止，进度已保存（检查点/策略/曲线）"}
    result = training_job.stop()  # 超时兜底强杀
    stop_flag.unlink(missing_ok=True)
    return result


@app.get("/api/training/status")
def api_training_status():
    st = training_job.status()
    # 信息面板：把「正在训练什么」解析出来（品种/周期/引擎/本次新增步数/起点）
    args = training_job.args or {}
    info = {"symbol": None, "timeframe": None, "engine": "single",
            "additional_steps": args.get("steps", 0), "from_scratch": args.get("from_scratch", False)}
    data_file = str(args.get("data_file") or "")
    if args.get("direct_mt5"):
        info["symbol"] = str(args.get("symbol") or "") or None
        info["timeframe"] = args.get("timeframe")
    elif data_file.endswith(".parquet") and Path(data_file).exists():
        try:
            from data_pipeline.parquet_manager import parse_parquet_filename
            sym, tf = parse_parquet_filename(data_file)
            info["symbol"], info["timeframe"] = sym, tf
        except Exception:
            pass
    if int(args.get("islands", 0) or 0) > 1:
        info["engine"] = f"island×{args['islands']}"
    st["info"] = info
    # 训练产物实时状态：检查点占用 + 策略文件
    ck_dir = ROOT / "checkpoints"
    ck_files = list(ck_dir.glob("*.pt")) if ck_dir.exists() else []
    ck_mb = round(sum(f.stat().st_size for f in ck_files) / 1048576, 1)
    st["artifacts"] = {
        "checkpoints": len(ck_files),
        "checkpoints_mb": ck_mb,
        "strategies": len(list((ROOT / "strategies").glob("*.json"))),
    }
    return st


def _training_history_dir() -> Path:
    return ROOT / "training_history"


@app.get("/api/training/curve")
def api_training_curve(symbol: str = ""):
    """Read only ROOT/training_history/training_history_tag.json."""
    if symbol and (not isinstance(symbol, str) or any(c in symbol for c in "/\\:") or symbol in (".", "..")):
        raise HTTPException(400, "非法历史标签")
    history_dir = _training_history_dir()
    if symbol:
        candidates = [history_dir / f"training_history_{symbol}.json"]
    else:
        arg_file = str(training_job.args.get("data_file") or "")
        candidates = []
        if arg_file and arg_file.endswith(".parquet"):
            try:
                from data_pipeline.parquet_manager import parse_parquet_filename
                sym, tf = parse_parquet_filename(arg_file)
                tag = f"{sym}_{tf}" if tf else sym
            except (ValueError, OSError):
                tag = Path(arg_file).stem
            island = int(training_job.args.get("islands", 0) or 0) > 1
            if island:
                candidates.append(history_dir / f"training_history_{tag}_island.json")
            candidates.append(history_dir / f"training_history_{tag}.json")
        candidates.extend(sorted(history_dir.glob("training_history_*.json"),
                                 key=lambda p: p.stat().st_mtime, reverse=True))
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            steps = data.get("step") or []
            best = data.get("best_score") or []
            points = []
            for step, score in zip(steps, best):
                try:
                    step = int(step)
                    score = float(score)
                except (TypeError, ValueError, OverflowError):
                    continue
                if not (math.isfinite(step) and math.isfinite(score)):
                    continue
                points.append([step, round(score, 4)])
            return {"symbol": path.stem.replace("training_history_", ""),
                    "points": points, "best": points[-1][1] if points else None,
                    "file": path.name,
                    "skipped_invalid": min(len(steps), len(best)) - len(points)}
        except (json.JSONDecodeError, OSError, TypeError, ValueError, OverflowError):
            continue
    return {"symbol": symbol or None, "points": [], "best": None, "file": None}


# ── 回测 ────────────────────────────────────────────────────────────


def _quick_mt5_api():
    # Do not initialize here: MT5.initialize may launch a closed terminal.
    # Dashboard status owns its normal connection; quick backtest only reads it.
    try:
        import MetaTrader5 as mt5
    except ImportError as exc:
        raise HTTPException(503, "MetaTrader5 包不可用，无法读取真实行情") from exc
    if mt5.terminal_info() is None:
        raise HTTPException(503, "MT5 尚未连接；请打开并登录终端，在总览确认连接后重试")
    return mt5


@app.get("/api/backtest/quick/options")
def api_quick_backtest_options(strategy_file: str, symbol: str = ""):
    from config import DEFAULT_TRADER_CONFIG
    from trading.quick_backtest import QuickBacktestError, get_options, read_config, resolve_strategy
    try:
        resolve_strategy(ROOT, strategy_file)  # Reject unsafe paths before touching MT5.
        cfg = read_config(ROOT, DEFAULT_TRADER_CONFIG)
        return get_options(ROOT, strategy_file, cfg, _quick_mt5_api(), symbol=symbol or None)
    except QuickBacktestError as exc:
        raise HTTPException(exc.status, str(exc)) from exc


@app.post("/api/backtest/quick")
def api_quick_backtest(payload: dict):
    from config import DEFAULT_TRADER_CONFIG
    from trading.quick_backtest import (QuickBacktestError, read_config, resolve_strategy,
                                       run_quick_backtest, validate_request)
    try:
        validate_request(payload)
        resolve_strategy(ROOT, payload.get("strategy_file"))
    except QuickBacktestError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    if not _quick_bt_lock.acquire(blocking=False):
        raise HTTPException(409, "已有快速回测正在运行，请稍后重试")
    try:
        cfg = read_config(ROOT, DEFAULT_TRADER_CONFIG)
        mt5 = _quick_mt5_api()
        offset = _mt5_client.server_time_offset() if _mt5_client is not None and _mt5_client.connected else None
        return run_quick_backtest(ROOT, payload, cfg, mt5, server_offset_sec=offset)
    except QuickBacktestError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning(f"[快速回测] 失败: {exc}")
        raise HTTPException(503, f"快速回测失败，未产生结果: {exc}") from exc
    finally:
        _quick_bt_lock.release()


@app.post("/api/backtest/start")
def api_backtest_start(payload: dict):
    strategy_file = str(_resolve_strategy_path(payload.get("strategy_file", "")).relative_to(ROOT))
    if not (ROOT / strategy_file).is_file():
        raise HTTPException(400, "策略文件不存在")
    cmd = [_venv_python(), "-u", "src/run_backtest.py",
           "--strategy-file", strategy_file]
    data_file = str(payload.get("data_file", "") or "").strip()
    if not data_file:
        # 策略没记录数据文件时，按 品种+周期 到数据目录自动寻找同名 parquet
        try:
            meta = load_strategy_file(ROOT / strategy_file)
            sym, tf = meta["symbol"], meta["timeframe"]
            cache_dir = Path(load_trader_config().get("kline_cache_dir", r"D:\K线数据"))
            candidate = cache_dir / f"{sym}_{tf}.parquet"
            if candidate.exists():
                data_file = str(candidate)
        except Exception:
            data_file = ""
    if data_file:
        if not Path(data_file).exists():
            raise HTTPException(400, f"数据文件不存在: {data_file}")
        cmd.extend(["--data-file", data_file])
    commission = float(payload.get("commission", 0.02) or 0.02)
    slippage = float(payload.get("slippage", 0.01) or 0.01)
    cmd.extend(["--commission", str(commission), "--slippage", str(slippage)])
    log_file = LOGS_DIR / f"backtest_{int(time.time())}.log"
    result = backtest_job.start(cmd, log_file, {
        "strategy_file": strategy_file, "data_file": data_file,
        "commission": commission, "slippage": slippage,
    })
    result["log"] = str(log_file)
    return result


@app.post("/api/backtest/stop")
def api_backtest_stop():
    return backtest_job.stop()


@app.get("/api/backtest/status")
def api_backtest_status():
    return backtest_job.status()


@app.get("/api/backtest/report")
def api_backtest_report():
    report = BACKTEST_OUTPUT / "multi_factor_report.json"
    equity = BACKTEST_OUTPUT / "equity_curve.json"
    out = {"report": None, "equity": None, "charts": []}
    if report.exists():
        try:
            out["report"] = json.loads(report.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    if equity.exists():
        try:
            data = json.loads(equity.read_text(encoding="utf-8"))
            out["equity"] = data.get("portfolio", data)
        except (json.JSONDecodeError, OSError):
            pass
    if BACKTEST_OUTPUT.exists():
        out["charts"] = [p.name for p in sorted(BACKTEST_OUTPUT.glob("*.png"))]
    return out


@app.get("/api/backtest/chart/{name}")
def api_backtest_chart(name: str):
    target = (BACKTEST_OUTPUT / name).resolve()
    if target.parent != BACKTEST_OUTPUT.resolve() or target.suffix != ".png":
        raise HTTPException(400, "非法文件名")
    if not target.exists():
        raise HTTPException(404, "图表不存在")
    return FileResponse(target, media_type="image/png")


# ── 启动 ────────────────────────────────────────────────────────────

def _autostart_dry_runner() -> None:
    if os.environ.get("MT5AUTOTRADER_OFFLINE") == "1":
        return
    try:
        with CONFIG_LOCK:
            if _runner_alive().get("process_alive"):
                return
            cfg = load_trader_config()
            if cfg.get("dry_run") is not True or not cfg.get("bindings"):
                return
            api_runner_start()
    except Exception as exc:
        logger.warning(f"[看板] dry-run 自动启动失败，可在交易控制页重试: {exc}")


def main() -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    STRATEGIES_DIR.mkdir(exist_ok=True)
    logger.remove()
    logger.add(
        LOGS_DIR / "dashboard.log",
        rotation="5 MB",
        retention=5,
        encoding="utf-8",
        colorize=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )
    _autostart_dry_runner()
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=APP_PORT, log_level="warning")


if __name__ == "__main__":
    main()

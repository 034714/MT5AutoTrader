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
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from loguru import logger

# SRC = 代码目录（src\），ROOT = 项目根目录（配置/数据/日志所在）
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    Config,
    load_trader_config,
    save_trader_config,
)
from trading.signal_engine import (  # noqa: E402
    StrategyError,
    formula_preview,
    load_strategy_file,
)

APP_PORT = 8900
WEB_DIR = SRC / "web"
LOGS_DIR = ROOT / "logs"
RUNNER_LOG = LOGS_DIR / "trading_runner.log"
RUNNER_PID_FILE = LOGS_DIR / "trading_runner.pid"
STATUS_FILE = LOGS_DIR / "runner_status.json"
STOP_FILE = ROOT / "STOP_SIGNAL"
STRATEGIES_DIR = ROOT / "strategies"
BACKTEST_OUTPUT = ROOT / "backtest_output"
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_PROCESS_GROUP = 0x00000200

app = FastAPI(title="MT5AutoTrader", version="1.0.0")


@app.middleware("http")
async def no_cache(request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    return response


# ── 工具 ────────────────────────────────────────────────────────────

def _venv_python() -> str:
    """训练/Runner 子进程使用的 Python 解释器，优先级：
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
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
    if not handle:
        return False
    try:
        exit_code = ctypes.c_ulong()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


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
    """runner 存活状态：优先看状态文件新鲜度 + PID 检测。"""
    if not STATUS_FILE.exists():
        return {"alive": False, "status": None}
    try:
        status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"alive": False, "status": None}
    pid = int(status.get("pid", 0) or 0)
    alive = _pid_alive(pid)
    fresh = False
    try:
        age = time.time() - STATUS_FILE.stat().st_mtime
        fresh = age < 60  # 状态文件 10 秒内会刷新，60 秒未刷新视为僵死
    except OSError:
        pass
    status["alive"] = alive and fresh and bool(status.get("running"))
    return {"alive": status["alive"], "status": status}


# ── MT5 按需连接（看板进程内，用于账户信息/品种列表/手动交易）──────────

_mt5_client = None


def _get_mt5_client():
    global _mt5_client
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
    try:
        from trading.mt5_client import position_to_dict
        positions = client.get_positions()
    except Exception:
        return result, server_offset
    for p in positions:
        result.append(position_to_dict(client, p, server_offset))
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
            status["positions"] = {}
    elif server_offset is None:
        # 旧版 Runner 的状态里没有时区偏移 → 看板自己估一个
        client = _get_mt5_client()
        if client is not None:
            server_offset = client.server_time_offset()
    # 账户净值采样（供总览页折线图）：每次轮询记一个点，内存保留最近 6 小时
    if account is not None and account.get("equity") is not None:
        _push_equity_sample(account["equity"])
    return {
        "runner_alive": runner["alive"],
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


@app.get("/api/logs/dashboard")
def api_dashboard_log(lines: int = 60):
    return {"log": _tail_file(LOGS_DIR / "dashboard.log", max(10, min(400, lines)))}


@app.post("/api/logs/clear")
def api_logs_clear(payload: dict):
    """清空日志文件（写入方以追加模式打开，截断后下一条日志从文件头继续）。"""
    name = str(payload.get("name", "")).strip()
    path = {"runner": RUNNER_LOG, "dashboard": LOGS_DIR / "dashboard.log"}.get(name)
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
    if path is None or not path.exists():
        raise HTTPException(404, "日志文件不存在")
    return FileResponse(
        path,
        media_type="text/plain; charset=utf-8",
        filename=f"MT5AutoTrader_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
    )


# ── 账户净值采样（内存环形缓冲，看板重启后从零开始积累）───────────────

_EQUITY_SAMPLES: list[list[float]] = []   # [unix_ts, equity]
_EQUITY_MAX_POINTS = 4320                 # 3 天 × 每 60 秒一点


def _push_equity_sample(equity: float) -> None:
    now = time.time()
    if _EQUITY_SAMPLES and now - _EQUITY_SAMPLES[-1][0] < 55:
        return  # 每分钟记一个点即可（页面图表也是 60 秒读一次）
    _EQUITY_SAMPLES.append([now, round(float(equity), 2)])
    if len(_EQUITY_SAMPLES) > _EQUITY_MAX_POINTS:
        del _EQUITY_SAMPLES[: len(_EQUITY_SAMPLES) - _EQUITY_MAX_POINTS]


@app.get("/api/equity/history")
def api_equity_history():
    return {"points": _EQUITY_SAMPLES, "history": _balance_history_points()}


# 历史余额推算结果缓存：(时刻, 点列表)。页面每 60 秒轮询本接口，
# 流水重读很费，5 分钟刷新一次足够（余额历史本来就是粗粒度）。
_BALANCE_HISTORY_CACHE: tuple[float, list] = (0.0, [])
_BALANCE_HISTORY_TTL = 300


def _balance_history_points(days: int = 90, max_points: int = 400) -> list:
    """从 MT5 成交流水推算账户余额历史（粗粒度），供净值图回填。

    MT5 不保存历史净值，只能按「每笔成交的盈亏+手续费+库存费+出入金」
    从当前余额反推：起点余额 = 当前余额 - 窗口内全部变动，再逐笔累加。
    返回 [[本机unix秒, balance], ...]，失败返回 []。
    """
    global _BALANCE_HISTORY_CACHE
    now = time.time()
    cached_ts, cached = _BALANCE_HISTORY_CACHE
    if now - cached_ts < _BALANCE_HISTORY_TTL:
        return cached
    points: list = []
    try:
        client = _get_mt5_client()
        ai = client.account_info() if client is not None else None
        if client is not None and ai is not None:
            import MetaTrader5 as mt5
            to = datetime.now() + timedelta(days=1)
            frm = datetime.now() - timedelta(days=days)
            deals = mt5.history_deals_get(frm, to) or []
            offset = client.server_time_offset() or 0
            deltas: list[tuple[float, float]] = []
            total = 0.0
            for d in deals:
                delta = float(d.profit) + float(d.commission) + float(d.swap)
                total += delta
                deltas.append((float(d.time) - offset, delta))
            if deltas:
                bal = float(ai.balance) - total
                stride = max(1, len(deltas) // max_points)
                for i, (ts, delta) in enumerate(deltas):
                    bal += delta
                    if i % stride == 0 or i == len(deltas) - 1:
                        points.append([ts, round(bal, 2)])
    except Exception as exc:
        logger.warning(f"[净值图] 历史余额推算失败（只用在线采样）: {exc}")
        points = []
    _BALANCE_HISTORY_CACHE = (now, points)
    return points


# ── 配置读写 ────────────────────────────────────────────────────────

@app.get("/api/config")
def api_config():
    return load_trader_config()


@app.put("/api/config")
def api_update_config(payload: dict):
    cfg = load_trader_config()
    # 切到真实下单必须明确确认；回到 dry-run 随时允许
    if "dry_run" in payload and bool(payload["dry_run"]) is False \
            and bool(cfg.get("dry_run", True)) is True:
        if not payload.get("confirmed"):
            raise HTTPException(400, "切换真实下单需要网页二次确认")
    if "dry_run" in payload:
        cfg["dry_run"] = bool(payload["dry_run"])
    risk = payload.get("risk")
    if isinstance(risk, dict):
        cfg.setdefault("risk", {}).update(risk)
    for key in ("min_trade_exposure", "max_lot_per_trade", "max_open_positions",
                "magic_number", "deviation_points", "signal_bars", "kline_cache_dir"):
        if key in payload:
            cfg[key] = payload[key]
    save_trader_config(cfg)
    return {"ok": True, "config": cfg}


# ── Runner 启停 ─────────────────────────────────────────────────────

@app.post("/api/runner/start")
def api_runner_start():
    alive = _runner_alive()["alive"]
    if alive:
        return {"ok": True, "message": "runner 已在运行"}
    try:
        STOP_FILE.unlink()
    except OSError:
        pass
    LOGS_DIR.mkdir(exist_ok=True)
    log_fh = open(RUNNER_LOG, "a", encoding="utf-8")
    # 嵌入式 Python（runtime\python.exe 带 python311._pth）处于隔离模式：
    # 不会把 cwd 加进 sys.path，"python -m trading.runner" 会报 No module named
    # 'trading'。这里显式把 src 注入 sys.path 后再以 __main__ 运行。
    boot = (
        "import sys, runpy;"
        f"sys.path.insert(0, r'{SRC}');"
        "runpy.run_module('trading.runner', run_name='__main__')"
    )
    proc = subprocess.Popen(
        [_venv_python(), "-c", boot],
        cwd=str(SRC),
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP,
    )
    RUNNER_PID_FILE.write_text(str(proc.pid), encoding="utf-8")
    logger.info(f"[看板] runner 已启动 pid={proc.pid}")
    return {"ok": True, "pid": proc.pid}


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
    if pid and _pid_alive(pid):
        # Windows 下不要用 os.kill(pid, 0)；这里使用 PowerShell Stop-Process 兜底。
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
    name = str(name or "").strip().replace("\\", "/")
    if not name:
        raise HTTPException(400, "缺少策略名")
    # 允许传 "best_X"、"best_X.json"、"strategies/best_X.json" 三种写法
    name = Path(name).name
    if not name.lower().endswith(".json"):
        name += ".json"
    target = (STRATEGIES_DIR / name).resolve()
    if target.parent != STRATEGIES_DIR.resolve():
        raise HTTPException(400, "非法文件名")
    return target


@app.post("/api/strategies/delete")
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
def api_strategy_bind(payload: dict):
    strategy_file = str(payload.get("strategy_file", "")).strip()
    symbol = str(payload.get("symbol", "")).strip()
    lot = float(payload.get("lot", 0.01) or 0.01)
    if not strategy_file or not symbol:
        raise HTTPException(400, "需要 strategy_file 和 symbol")
    if not (ROOT / strategy_file).exists():
        raise HTTPException(400, f"策略文件不存在: {strategy_file}")
    if lot <= 0:
        raise HTTPException(400, "手数必须大于 0")
    cfg = load_trader_config()
    bindings = [b for b in cfg.get("bindings", []) if b.get("symbol") != symbol]
    bindings.append({"strategy_file": strategy_file, "symbol": symbol, "lot": lot})
    cfg["bindings"] = bindings
    save_trader_config(cfg)
    return {"ok": True, "bindings": bindings}


@app.post("/api/strategies/unbind")
def api_strategy_unbind(payload: dict):
    symbol = str(payload.get("symbol", "")).strip()
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
        symbols = mt5.symbols_get()
        names = sorted({s.name for s in (symbols or [])})
        return {"symbols": names[:2000]}
    except Exception as exc:
        raise HTTPException(500, str(exc))


@app.post("/api/mt5/position/partial")
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
    # 仓位信息优先取 Runner 状态（dry-run 台账和 MT5 实仓都在里面）
    info = None
    try:
        st = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        info = next((q for q in (st.get("positions") or [])
                     if int(q.get("ticket", 0)) == ticket), None)
    except (json.JSONDecodeError, OSError):
        info = None
    if info is None:
        client = _get_mt5_client()
        p = _find_any_position(client, ticket) if client else None
        if p is None:
            raise HTTPException(404, f"找不到持仓 ticket={ticket}")
        lot = float(p.volume)
    else:
        lot = float(info.get("volume", 0) or 0)
    if price <= 0:
        override = {"price": 0.0, "close_volume": 0.0, "ts": time.time()}
        message = "已取消该仓位的止盈一半计划"
    else:
        if lot <= 0 or volume <= 0 or volume >= lot:
            raise HTTPException(
                400, f"平仓手数必须在 0 与 {lot} 之间（全部平仓请用「平仓」按钮）")
        override = {"price": price, "close_volume": volume, "ts": time.time()}
        message = f"止盈一半已设置：到 {price} 平 {volume}手"
    path = LOGS_DIR / "sr_partial_overrides.json"
    try:
        data = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        data[str(ticket)] = override
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(500, f"写入失败: {exc}")
    logger.info(f"[持仓管理] ticket={ticket} 止盈一半覆盖: {override}")
    return {"ok": True, "message": message}


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
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "candles": [row(r) for r in closed[-last:]],
        "forming": row(rates[-1]) if len(rates) else None,
        "zones": levels,
        "info": info,
        "server_offset_sec": client.server_time_offset(),
        "sr_enabled": bool(load_trader_config().get("risk", {}).get("sr_enabled", True)),
    }


# ── 手动交易（二次确认）─────────────────────────────────────────────

@app.post("/api/mt5/order_check")
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


# ── 持仓管理（平仓 / 修改止损 / 修改止盈）────────────────────────────

def _find_any_position(client, ticket: int):
    """按 ticket 查找持仓（本软件 magic 内）。找不到返回 None。"""
    for p in client.get_positions():
        if int(p.ticket) == int(ticket):
            return p
    return None


@app.post("/api/mt5/position/close")
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
    client.dry_run = False  # 网页明确确认后的真实动作
    ok = client.close_position(str(p.symbol), ticket)
    if ok:
        logger.info(f"[持仓管理] 已平仓 ticket={ticket} {p.symbol}")
        return {"ok": True, "message": f"{p.symbol} ticket={ticket} 平仓成功"}
    return {"ok": False, "message": "平仓失败，请查看日志或重试"}


def _write_sl_override(ticket: int, sl: float, applied_to_mt5: bool) -> None:
    """把手动止损写入覆盖文件，Runner 下圈应用并豁免自动拉回。"""
    path = LOGS_DIR / "sl_overrides.json"
    try:
        data = {}
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
        data[str(int(ticket))] = {"sl": sl, "ts": time.time(),
                                  "applied_to_mt5": bool(applied_to_mt5)}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(f"[持仓管理] 写入手动止损覆盖失败: {exc}")


def _find_status_position(ticket: int) -> dict | None:
    """从 Runner 状态里找持仓（dry-run 台账和 MT5 实仓都在里面）。"""
    try:
        st = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        return next((q for q in (st.get("positions") or [])
                     if int(q.get("ticket", 0)) == int(ticket)), None)
    except (json.JSONDecodeError, OSError, ValueError):
        return None


@app.post("/api/mt5/position/sl")
def api_position_sl(payload: dict):
    ticket = int(payload.get("ticket", 0) or 0)
    sl = float(payload.get("sl", 0) or 0)
    if sl <= 0:
        raise HTTPException(400, "止损价格无效")
    client = _get_mt5_client()
    mt5_pos = _find_any_position(client, ticket) if client else None
    if mt5_pos is not None:
        # 真实 MT5 持仓：直接修改，并写覆盖文件让 Runner 豁免自动拉回
        client.dry_run = False
        ok = client.modify_sl(str(mt5_pos.symbol), ticket, new_sl=sl,
                              tp=float(getattr(mt5_pos, "tp", 0) or 0))
        if ok:
            _write_sl_override(ticket, sl, applied_to_mt5=True)
            logger.info(f"[持仓管理] ticket={ticket} {mt5_pos.symbol} 止损手动改为 {sl}")
            return {"ok": True, "message": f"止损已改为 {sl}（手动管理，不再自动拉回）"}
        return {"ok": False, "message": "修改失败（可能离市价太近或市场关闭）"}
    # MT5 上没有：可能是 dry-run 模拟仓位，写覆盖文件交由 Runner 更新台账
    info = _find_status_position(ticket)
    if info is None:
        raise HTTPException(404, f"找不到持仓 ticket={ticket}")
    _write_sl_override(ticket, sl, applied_to_mt5=False)
    logger.info(f"[持仓管理] ticket={ticket} {info.get('symbol')} 止损手动改为 {sl}（模拟仓）")
    return {"ok": True, "message": f"止损已改为 {sl}（模拟仓，Runner 下圈生效）"}


@app.post("/api/mt5/position/tp")
def api_position_tp(payload: dict):
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接")
    ticket = int(payload.get("ticket", 0) or 0)
    tp = float(payload.get("tp", 0) or 0)
    p = _find_any_position(client, ticket)
    if p is None:
        raise HTTPException(404, f"找不到持仓 ticket={ticket}")
    client.dry_run = False
    ok = client.modify_sl(str(p.symbol), ticket, new_sl=float(getattr(p, "sl", 0) or 0), tp=tp)
    if ok:
        logger.info(f"[持仓管理] ticket={ticket} {p.symbol} 止盈改为 {tp or '清除'}")
        return {"ok": True, "message": f"止盈已改为 {tp}" if tp > 0 else "止盈已清除"}
    return {"ok": False, "message": "修改失败（可能离市价太近或市场关闭）"}


# ── 历史成交 ────────────────────────────────────────────────────────

@app.get("/api/mt5/history")
def api_mt5_history(days: int = 30):
    """从 MT5 读取真实成交历史，按 position 归并成交易记录。"""
    client = _get_mt5_client()
    if client is None:
        raise HTTPException(400, "MT5 未连接")
    import MetaTrader5 as mt5
    from trading.history import group_history_deals
    days = max(1, min(365, int(days or 30)))
    to = datetime.now() + timedelta(days=1)
    frm = datetime.now() - timedelta(days=days)
    try:
        deals = mt5.history_deals_get(frm, to) or []
    except Exception as exc:
        raise HTTPException(500, f"读取历史失败: {exc}")
    grouped = group_history_deals(deals, magic=client.magic)
    server_offset = client.server_time_offset()
    return {
        "server_offset_sec": server_offset,
        "total_deals": len(deals),
        "closed": grouped["closed"],
        "open": grouped["open"],
        "days": days,
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


@app.post("/api/data/export_mt5")
def api_export_mt5(payload: dict):
    """从 MT5 导出指定周期 K 线到数据目录（供训练用）。"""
    symbol = str(payload.get("symbol", "")).strip()
    bars = int(payload.get("bars", 20000) or 20000)
    timeframe = str(payload.get("timeframe", "H1") or "H1").upper()
    if not symbol:
        raise HTTPException(400, "缺少 symbol")
    cfg = load_trader_config()
    out_dir = cfg.get("kline_cache_dir", r"D:\K线数据")
    code = (
        "import sys; sys.path.insert(0, r'{src}');"
        "import export_mt5_data as m;"
        "ok = m.export_symbol('{sym}', {bars}, '{tf}', r'{out}');"
        "sys.exit(0 if ok else 1)"
    ).format(src=SRC, sym=symbol, bars=bars, tf=timeframe, out=out_dir)
    log_file = LOGS_DIR / f"export_{symbol}_{int(time.time())}.log"
    with open(log_file, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(
            [_venv_python(), "-c", code], cwd=str(ROOT),
            stdout=fh, stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW,
        )
    return {"ok": True, "pid": proc.pid, "log": str(log_file),
            "message": "导出已在后台启动，稍后刷新数据文件列表"}


# ── 训练 ────────────────────────────────────────────────────────────

class JobManager:
    """同一时刻只允许一个训练/回测子进程。"""

    def __init__(self, kind: str) -> None:
        self.kind = kind
        self.proc: subprocess.Popen | None = None
        self.log_file: Path | None = None
        self.started_at: str | None = None
        self.args: dict = {}

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, cmd: list[str], log_file: Path, args: dict) -> dict:
        if self.running():
            raise HTTPException(400, f"{self.kind} 已在运行中")
        log_file.parent.mkdir(exist_ok=True)
        fh = open(log_file, "w", encoding="utf-8")
        env = dict(os.environ)
        env.setdefault("MPLBACKEND", "Agg")
        env["PYTHONUNBUFFERED"] = "1"
        self.proc = subprocess.Popen(
            cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
            creationflags=CREATE_NO_WINDOW | CREATE_NEW_PROCESS_GROUP, env=env,
        )
        self.log_file = log_file
        self.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.args = args
        logger.info(f"[{self.kind}] 启动 pid={self.proc.pid} args={args}")
        return {"ok": True, "pid": self.proc.pid}

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
    direct_mt5 = bool(payload.get("direct_mt5"))
    from_scratch = bool(payload.get("from_scratch"))
    steps = int(payload.get("steps", 0) or 0)
    timeframe = str(payload.get("timeframe", "H1") or "H1").upper()
    if direct_mt5:
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
        if from_scratch:
            cmd.append("--from-scratch")
        data_file = "（由 MT5 直连获取）"
        log_name = f"{symbol}_{timeframe}"
    else:
        data_file = str(payload.get("data_file", "")).strip()
        if not data_file or not Path(data_file).exists():
            raise HTTPException(400, "数据文件不存在")
        cmd = [_venv_python(), "-u", "src/train_file.py", "--data-file", data_file]
        if from_scratch:
            cmd.append("--from-scratch")
        if steps > 0:
            cmd.extend(["--steps", str(steps)])
        log_name = Path(data_file).stem
    log_file = LOGS_DIR / f"train_{log_name}_{int(time.time())}.log"
    result = training_job.start(cmd, log_file, {
        "data_file": data_file, "direct_mt5": direct_mt5,
        "from_scratch": from_scratch, "steps": steps, "timeframe": timeframe,
    })
    result["log"] = str(log_file)
    return result


@app.post("/api/training/stop")
def api_training_stop():
    return training_job.stop()


@app.get("/api/training/status")
def api_training_status():
    return training_job.status()


@app.get("/api/training/curve")
def api_training_curve(symbol: str = ""):
    """训练分数曲线：读取 training_history_{symbol}.json。

    symbol 留空时优先取当前训练任务的数据文件品种，否则取最新的一份历史。
    """
    if symbol:
        candidates = [ROOT / f"training_history_{symbol}.json"]
    else:
        arg_file = str(training_job.args.get("data_file") or "")
        candidates = []
        if arg_file and arg_file.endswith(".parquet"):
            candidates.append(ROOT / f"training_history_{Path(arg_file).stem}.json")
        candidates.extend(sorted(
            ROOT.glob("training_history_*.json"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        ))
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            steps = data.get("step") or []
            best = data.get("best_score") or []
            n = min(len(steps), len(best))
            points = [[int(steps[i]), round(float(best[i]), 4)] for i in range(n)]
            return {"symbol": path.stem.replace("training_history_", ""),
                    "points": points, "best": points[-1][1] if points else None,
                    "file": path.name}
        except (json.JSONDecodeError, OSError, ValueError):
            continue
    return {"symbol": symbol or None, "points": [], "best": None, "file": None}


# ── 回测 ────────────────────────────────────────────────────────────

@app.post("/api/backtest/start")
def api_backtest_start(payload: dict):
    strategy_file = str(payload.get("strategy_file", "")).strip()
    if not strategy_file or not (ROOT / strategy_file).exists():
        raise HTTPException(400, "策略文件不存在")
    cmd = [_venv_python(), "-u", "src/run_backtest.py",
           "--strategy-file", strategy_file]
    data_file = str(payload.get("data_file", "")).strip()
    if data_file:
        if not Path(data_file).exists():
            raise HTTPException(400, "数据文件不存在")
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
    if not str(target).startswith(str(BACKTEST_OUTPUT.resolve())) or target.suffix != ".png":
        raise HTTPException(400, "非法文件名")
    if not target.exists():
        raise HTTPException(404, "图表不存在")
    return FileResponse(target, media_type="image/png")


# ── 启动 ────────────────────────────────────────────────────────────

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
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=APP_PORT, log_level="warning")


if __name__ == "__main__":
    main()

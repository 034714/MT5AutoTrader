"""
trading/runner.py — 自动交易主循环（独立进程，由看板启停）

两个节奏合在一个循环里（默认每 5~10 秒一圈）：
  1. 信号循环：发现新收盘 H1 K线 → 计算策略信号 → 对账执行
     （同向 HOLD 不重复开仓；反向先平旧仓、确认成功才开反向仓；
      平仓失败禁止开新仓并记录）
  2. 实时风控循环：每圈读取 bid/ask，执行初始止损 + 阶梯保本

状态持久化 portfolio_state.json：
  - dry-run 模式下本地模拟仓位即真实状态，绝不会被 MT5 同步误删
  - live 模式下与 MT5 持仓对账（magic 过滤），止损/手动平仓后状态自动校正
  - 每个 ticket 记录已锁定利润，重启后阶梯保本不回退

看板通过以下文件与本进程通信：
  - logs/runner_status.json   心跳与状态（看板读）
  - STOP_SIGNAL               停止信号（看板写 "STOP"，本进程回写 "STOPPED" 后退出）
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from loguru import logger

# SRC = 代码目录（src\），ROOT = 项目根目录（日志/状态/台账所在）
SRC = Path(__file__).resolve().parent.parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

from config import Config, load_trader_config  # noqa: E402
from strategy_manager.signal import reconcile_action  # noqa: E402
from trading.mt5_client import MT5Client  # noqa: E402
from trading.risk import (  # noqa: E402
    LOCK_NONE,
    RiskManager,
    RiskParams,
    initial_stop_price,
    ladder_lock,
    lock_price,
    profit_pct,
)
from trading.signal_engine import (  # noqa: E402
    DIR_LONG,
    DIR_SHORT,
    compute_signal,
    load_strategy_file,
    rates_to_raw_dict,
    check_vocab_version,
)

STATUS_FILE = ROOT / "logs" / "runner_status.json"
STATE_FILE = ROOT / Config.PORTFOLIO_FILE
STOP_FILE = ROOT / Config.STOP_SIGNAL
# 看板写入的"止盈一半"手动覆盖（ticket → {price, close_volume, ts}），每圈读取一次
PARTIAL_OVERRIDES_FILE = ROOT / "logs" / "sr_partial_overrides.json"
# 看板写入的"手动止损"覆盖（ticket → {sl, ts, applied_to_mt5}）：手动设置后
# 风控不再把该仓位的止损拉回初始安全位（阶梯保本仍生效）
SL_OVERRIDES_FILE = ROOT / "logs" / "sl_overrides.json"
DIRECTION_TO_INT = {"LONG": 1, "SHORT": -1, "FLAT": 0}


# ── 状态（仓位台账）─────────────────────────────────────────────────

class PositionBook:
    """仓位台账：dry-run 时即真实状态；live 时与 MT5 对账。"""

    def __init__(self, mode: str) -> None:
        self.mode = mode  # "dry" | "live"
        self.state: dict = {"mode": mode, "next_ticket": 9_000_000_001,
                            "positions": {}, "tickets": {}, "actions": []}
        self._load()

    def _load(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("mode") == self.mode:
                self.state = data
                self.state.setdefault("positions", {})
                self.state.setdefault("tickets", {})
                self.state.setdefault("actions", [])
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(f"[状态] portfolio_state.json 读取失败: {exc}")

    def save(self) -> None:
        try:
            tmp = STATE_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(STATE_FILE)
        except OSError as exc:
            logger.error(f"[状态] portfolio_state.json 写入失败: {exc}")

    # ── 仓位操作 ────────────────────────────────────────────────

    def get_position(self, symbol: str) -> dict | None:
        return self.state["positions"].get(symbol)

    def set_position(self, symbol: str, info: dict) -> None:
        self.state["positions"][symbol] = info
        self.state["tickets"].setdefault(str(info["ticket"]), {"locked": LOCK_NONE})

    def clear_symbol(self, symbol: str) -> None:
        info = self.state["positions"].pop(symbol, None)
        if info:
            self.state["tickets"].pop(str(info.get("ticket")), None)

    def locked_for(self, ticket: int) -> float:
        entry = self.state["tickets"].get(str(int(ticket)))
        if entry is None:
            return LOCK_NONE
        return float(entry.get("locked", LOCK_NONE))

    def set_locked(self, ticket: int, locked: float) -> None:
        self.state["tickets"].setdefault(str(int(ticket)), {"locked": LOCK_NONE})
        self.state["tickets"][str(int(ticket))]["locked"] = float(locked)

    def next_ticket(self) -> int:
        ticket = int(self.state.get("next_ticket", 9_000_000_001))
        self.state["next_ticket"] = ticket + 1
        return ticket

    def open_count(self) -> int:
        return len(self.state["positions"])

    def record_action(self, text: str) -> None:
        entry = {"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "action": text}
        self.state["actions"].append(entry)
        self.state["actions"] = self.state["actions"][-200:]
        logger.info(f"[动作] {text}")

    def mark_manual_sl(self, ticket: int) -> None:
        """标记该仓位止损为手动管理（风控不再自动拉回）。"""
        self.state["tickets"].setdefault(str(int(ticket)), {"locked": LOCK_NONE})
        self.state["tickets"][str(int(ticket))]["manual_sl"] = True

    def manual_sl_tickets(self) -> set[int]:
        return {int(t) for t, v in self.state["tickets"].items() if v.get("manual_sl")}


# ── 主运行器 ────────────────────────────────────────────────────────

class TradingRunner:
    def __init__(self) -> None:
        self.client = MT5Client()
        self.risk = RiskManager(self.client)
        self.book: PositionBook | None = None
        self.bindings: list[dict] = []
        self.strategies: dict[str, dict] = {}       # strategy_file → 校验后的策略
        self._strategy_mtime: dict[str, float | None] = {}  # 策略文件改动检测（热更新）
        self._last_bar_time: dict[str, int] = {}    # symbol → 已处理的最后收盘 bar 时间
        self._stop = False
        self._loop_count = 0
        self._last_error: str | None = None
        self._started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self._last_signal_info: dict[str, dict] = {}
        self._signal_bars: int = 1000
        self._startup_dry_run: bool = True
        self._server_offset: int | None = None
        self._ov_mtime: float | None = None
        self._ov_cache: dict | None = None
        self._ov_applied_ts: dict[int, float] = {}
        self._sl_ov_mtime: float | None = None
        self._sl_ov_cache: dict | None = None
        self._sl_ov_applied_ts: dict[int, float] = {}

    # ── 启停 ────────────────────────────────────────────────────

    def stop(self) -> None:
        self._stop = True

    def _check_stop_signal(self) -> bool:
        try:
            if STOP_FILE.exists():
                text = STOP_FILE.read_text(encoding="utf-8", errors="ignore").strip()
                if text == "STOP":
                    STOP_FILE.write_text("STOPPED", encoding="utf-8")
                    return True
        except OSError:
            pass
        return False

    # ── 策略加载 ────────────────────────────────────────────────

    def _refresh_bindings(self) -> None:
        cfg = load_trader_config()
        self.bindings = [b for b in cfg.get("bindings", []) if b.get("symbol")]
        for binding in self.bindings:
            path = binding.get("strategy_file", "")
            if not path:
                continue
            file_path = ROOT / path
            try:
                mtime = file_path.stat().st_mtime
            except OSError:
                mtime = None
            # 文件没变化（含上次加载失败的情况）→ 跳过；
            # 文件被重新训练覆盖后自动热更新，无需重启 Runner。
            if path in self.strategies and self._strategy_mtime.get(path) == mtime:
                continue
            try:
                strategy = load_strategy_file(file_path)
            except Exception as exc:
                logger.error(f"[策略] 加载失败 {path}: {exc}")
                self.strategies[path] = {"error": str(exc)}
                self._strategy_mtime[path] = mtime
                continue
            if not check_vocab_version(strategy["vocab_version"]):
                logger.error(
                    f"[策略] {path} vocab_version={strategy['vocab_version']} "
                    f"与本引擎词表不一致，已跳过"
                )
                self.strategies[path] = {"error": "vocab_version 不一致"}
                self._strategy_mtime[path] = mtime
                continue
            first_load = path not in self.strategies
            self.strategies[path] = strategy
            self._strategy_mtime[path] = mtime
            action = "已加载" if first_load else "检测到重新训练，已热更新"
            logger.info(
                f"[策略] {action} {path} score={strategy['best_score']} "
                f"{strategy['formula_decoded']}"
            )

    # ── 仓位视图 ────────────────────────────────────────────────

    def _sync_live_positions(self) -> None:
        """live 模式：与 MT5 持仓对账（magic 已过滤）。

        - MT5 有而台账无 → 台账补记（如手动开的同 magic 仓）
        - 台账有而 MT5 无 → 已被止损/手动平仓，清台账并记录
        - 同步时顺带刷新现价/浮盈/止损/止盈，供看板展示
        """
        for symbol in list(self.book.state["positions"].keys()):
            info = self.book.get_position(symbol)
            if info is None:
                continue
            live = [p for p in self.client.get_positions(symbol)
                    if int(p.ticket) == int(info.get("ticket", 0))]
            if not live:
                self.book.record_action(
                    f"{symbol} 持仓 ticket={info.get('ticket')} 已不存在（止损/手动平仓），清除台账"
                )
                self.book.clear_symbol(symbol)
                continue
            p = live[0]
            from trading.mt5_client import position_to_dict
            view = position_to_dict(self.client, p, self._server_offset)
            info.update({
                "volume": view["volume"],
                "open_price": view["open_price"],
                "sl": view["sl"],
                "tp": view["tp"],
                "direction": view["direction"],
                "current_price": view["current_price"],
                "profit": view["profit"],
                "profit_pct": view["profit_pct"],
            })

    def _current_direction(self, symbol: str) -> int:
        """当前净方向：dry 看台账；live 看 MT5 实际持仓。"""
        if self.book.mode == "dry":
            info = self.book.get_position(symbol)
            if info is None:
                return 0
            return 1 if info["direction"] == "BUY" else -1
        positions = self.client.get_positions(symbol)
        has_buy = any(p.type == 0 for p in positions)
        has_sell = any(p.type == 1 for p in positions)
        if has_buy and has_sell:
            return 2  # 对冲状态：需要先全部平掉
        if has_buy:
            return 1
        if has_sell:
            return -1
        return 0

    # ── 平仓 / 开仓 ─────────────────────────────────────────────

    def _close_symbol(self, symbol: str) -> bool:
        """平掉该品种全部仓位并确认。成功返回 True。"""
        if self.book.mode == "dry":
            info = self.book.get_position(symbol)
            if info is None:
                return True
            ok = self.client.close_position(symbol, int(info["ticket"]))
            if ok:
                self.book.clear_symbol(symbol)
                self.book.record_action(f"{symbol} dry-run 平仓 {info['direction']} 成功")
            return ok
        # live
        for attempt in range(3):
            positions = self.client.get_positions(symbol)
            if not positions:
                self.book.clear_symbol(symbol)
                return True
            ok = self.client.close_symbol_all(symbol)
            if ok:
                time.sleep(0.5)
                if not self.client.get_positions(symbol):
                    self.book.clear_symbol(symbol)
                    self.book.record_action(f"{symbol} 平仓成功")
                    return True
            logger.warning(f"[平仓] {symbol} 第 {attempt+1} 次尝试未完全平掉，重试")
            time.sleep(0.5)
        remaining = self.client.get_positions(symbol)
        if not remaining:
            self.book.clear_symbol(symbol)
            return True
        self.book.record_action(f"{symbol} 平仓失败（仍有 {len(remaining)} 笔持仓）")
        return False

    def _tf_for_strategy(self, strategy_path: str) -> tuple[int, int, str]:
        """取该策略训练周期的 (MT5常量, 秒数, 字符串)；未知按 H1。"""
        strategy = self.strategies.get(strategy_path) or {}
        tf_str = str(strategy.get("timeframe") or "H1").upper()
        return Config.get_timeframe(tf_str), Config.timeframe_seconds(tf_str), tf_str

    def _open_position(self, symbol: str, direction: str, binding: dict,
                       strategy_path: str) -> bool:
        """开仓（带初始止损）。成功返回 True 并写台账。"""
        lot = self._normalize_lot(symbol, float(binding.get("lot", 0.01)))
        if lot <= 0:
            self.book.record_action(f"{symbol} 手数无效，放弃开仓")
            return False
        # 最大持仓数限制
        max_pos = int(getattr(Config, "MAX_OPEN_POSITIONS", 3) or 0)
        if max_pos > 0:
            held = {s for s, info in self.book.state["positions"].items()
                    if info.get("direction")}
            held.discard(symbol)  # 反手场景：先平后开，旧仓已清
            if len(held) >= max_pos and symbol not in held:
                self.book.record_action(
                    f"{symbol}已达最大持仓数限制 {max_pos}，放弃开仓"
                )
                return False

        sl = None
        tp = None
        # 拿参考价算初始止损
        tick = self.client.get_tick(symbol)
        if tick is None:
            self.book.record_action(f"{symbol} 无法获取报价，放弃开仓")
            return False
        ref_price = tick["ask"] if direction == "BUY" else tick["bid"]
        sl = initial_stop_price(direction, ref_price, self.risk.params.stop_loss_pct)

        # 支撑/阻力优化（加性规则：固定止损始终保留，S/R 只会更紧或提供止盈）
        sr_summary = ""
        sr_partial = None
        try:
            from trading.sr import SRParams, sl_tp_for_trade
            sr_params = SRParams.from_config()
            if sr_params.enabled:
                tf_const, _tf_secs, _tf_str = self._tf_for_strategy(strategy_path)
                rates = self.client.copy_rates(symbol, tf_const, self._signal_bars)
                if rates is not None and len(rates) >= 120:
                    plan = sl_tp_for_trade(
                        symbol, direction, ref_price, rates[:-1], sr_params,
                        fixed_sl=sl,
                    )
                    if plan["sl"]:
                        sl = plan["sl"]
                    tp = plan["tp"]
                    tp = self._respect_tp_level(symbol, direction, tick, tp)
                    partial = plan.get("partial")
                    if partial and partial.get("price"):
                        ok_split, close_vol, remaining = self._split_lot(
                            symbol, lot, float(partial["fraction"]))
                        if ok_split:
                            sr_partial = {
                                "price": float(partial["price"]),
                                "fraction": float(partial["fraction"]),
                                "close_volume": close_vol,
                                "remaining": remaining,
                                "done": False,
                            }
                        else:
                            logger.info(
                                f"[S/R] {symbol} 手数 {lot} 无法拆出止盈一半"
                                f"（最小手数限制），本次不启用止盈一半"
                            )
                    sl_zone, tp_zone = plan.get("sl_zone"), plan.get("tp_zone")
                    parts = [f"S/R止损源={plan['sl_source']}"]
                    if sl_zone:
                        parts.append(
                            f"止损位{sl_zone['center']:.5f}（{sl_zone['kind']}，"
                            f"历史触及{sl_zone['n_events']}次）"
                        )
                    if sr_partial:
                        parts.append(f"止盈一半={sr_partial['price']:.5f}（平{close_vol}手）")
                    if tp:
                        parts.append(f"全部止盈={tp:.5f}")
                        if tp_zone:
                            parts.append(f"（前方{tp_zone['kind']} {tp_zone['center']:.5f}）")
                    sr_summary = " ".join(parts)
                    logger.info(f"[S/R] {symbol} {direction} {sr_summary}")
        except Exception as exc:
            logger.warning(f"[S/R] {symbol} 计算异常，使用固定止损: {exc}")
            sl = initial_stop_price(direction, ref_price, self.risk.params.stop_loss_pct)
            tp = None

        result = self.client.market_open(symbol, direction, lot, sl=sl, tp=tp)
        if not result["ok"]:
            hint = result.get("hint") or ""
            self.book.record_action(
                f"{symbol} 开仓失败 {direction} {lot}手 retcode={result['retcode']}"
                + (f"（{hint}）" if hint else "")
            )
            return False

        if self.book.mode == "dry":
            fill_price = ref_price
            ticket = self.book.next_ticket()
            self.book.set_position(symbol, {
                "ticket": ticket, "direction": direction, "volume": lot,
                "open_price": fill_price, "open_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "sl": sl, "tp": tp or 0.0, "sr_partial": sr_partial,
                "locked": LOCK_NONE, "strategy": strategy_path,
                "dry_run": True,
            })
            self.book.record_action(
                f"{symbol} [DRY-RUN] 开仓 {direction} {lot}手 @≈{fill_price:.5f} "
                f"初始SL={sl:.5f}" + (f" TP={tp:.5f}" if tp else "")
            )
            return True

        # live：重新读取真实持仓，记录真实 ticket / 成交价
        time.sleep(0.5)
        positions = self.client.get_positions(symbol)
        if not positions:
            # 用 order result 的 order id 再找一次
            found = False
            for p in self.client.get_positions():
                if result.get("order") and int(p.ticket) == int(result["order"]):
                    positions = [p]
                    found = True
                    break
            if not found:
                self.book.record_action(f"{symbol} 开仓成功但未找到持仓记录，请人工检查")
                return False
        p = positions[0]
        direction_live = "BUY" if p.type == 0 else "SELL"
        info = {
            "ticket": int(p.ticket), "direction": direction_live,
            "volume": float(p.volume), "open_price": float(p.price_open),
            "open_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sl": float(getattr(p, "sl", 0.0) or 0.0),
            "tp": float(getattr(p, "tp", 0.0) or 0.0),
            "sr_partial": sr_partial,
            "locked": LOCK_NONE,
            "strategy": strategy_path, "dry_run": False,
        }
        # 开仓请求带了初始止损但可能被拒单重试；若无 SL，这里补记由风控循环兜底
        self.book.set_position(symbol, info)
        self.book.record_action(
            f"{symbol} 开仓 {direction_live} {info['volume']}手 "
            f"@{info['open_price']:.5f} ticket={info['ticket']} SL={info['sl'] or sl:.5f}"
            + (f" TP={info['tp'] or tp:.5f}" if (info['tp'] or tp) else "")
            + (" " + sr_summary if sr_summary else "")
        )
        return True

    def _respect_tp_level(self, symbol: str, direction: str, tick: dict,
                          tp: float | None) -> float | None:
        """止盈价必须离市价至少 stops_level；不满足时放弃止盈（宁可不设，不让整单被拒）。"""
        if not tp:
            return None
        stops_points = self.client.stops_level_points(symbol)
        point = self.client.point(symbol)
        if stops_points <= 0 or point <= 0:
            return tp
        min_dist = stops_points * point
        if direction == "BUY" and tp - tick["bid"] < min_dist:
            return None
        if direction == "SELL" and tick["ask"] - tp < min_dist:
            return None
        return tp

    def _split_lot(self, symbol: str, lot: float,
                   frac: float) -> tuple[bool, float, float]:
        """把手数按 frac 拆成（平掉量, 剩余量）；不足最小手数时拆不动。"""
        step = self.client.volume_step(symbol) or 0.01
        vmin = self.client.volume_min(symbol) or step
        close_vol = math.floor(lot * frac / step + 1e-9) * step
        close_vol = round(close_vol, 8)
        remaining = round(lot - close_vol, 8)
        if close_vol < vmin - 1e-9 or remaining < vmin - 1e-9:
            return False, 0.0, lot
        return True, close_vol, remaining

    def _load_partial_overrides(self) -> dict:
        """读看板写入的止盈一半覆盖（按文件 mtime 缓存）。"""
        try:
            if not PARTIAL_OVERRIDES_FILE.exists():
                return {}
            mtime = PARTIAL_OVERRIDES_FILE.stat().st_mtime
            if self._ov_mtime == mtime and self._ov_cache is not None:
                return self._ov_cache
            data = json.loads(PARTIAL_OVERRIDES_FILE.read_text(encoding="utf-8"))
            self._ov_mtime = mtime
            self._ov_cache = data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            self._ov_cache = {}
        return self._ov_cache

    def _apply_partial_overrides(self, overrides: dict) -> None:
        """把看板的止盈一半手动设置应用到台账（同一时间戳只应用一次）。"""
        for key, ov in overrides.items():
            if not isinstance(ov, dict):
                continue
            try:
                ticket = int(key)
                ts = float(ov.get("ts", 0) or 0)
            except (ValueError, TypeError):
                continue
            if self._ov_applied_ts.get(ticket, -1.0) >= ts:
                continue
            match = [(sym, inf) for sym, inf in self.book.state["positions"].items()
                     if int(inf.get("ticket", 0)) == ticket]
            if not match:
                # 仓位可能在别的循环才被纳入台账，下一圈再试
                continue
            symbol, info = match[0]
            price = float(ov.get("price", 0) or 0)
            close_vol = float(ov.get("close_volume", 0) or 0)
            lot = float(info.get("volume") or 0)
            if price <= 0:
                info["sr_partial"] = {"price": 0.0, "done": True,
                                      "skipped": "override_cleared"}
                self.book.record_action(f"{symbol} 已取消止盈一半计划（ticket={ticket}）")
            else:
                step = self.client.volume_step(symbol) or 0.01
                vmin = self.client.volume_min(symbol) or step
                cv2 = round(math.floor(close_vol / step + 1e-9) * step, 8)
                remaining = round(lot - cv2, 8)
                if lot <= 0 or cv2 < vmin - 1e-9 or remaining < vmin - 1e-9:
                    info["sr_partial"] = {"price": 0.0, "done": True,
                                          "skipped": "override_invalid_lot"}
                    self.book.record_action(
                        f"{symbol} 止盈一半设置无效：手数 {close_vol} 无法从 {lot}手 拆出"
                        f"（ticket={ticket}）")
                else:
                    info["sr_partial"] = {
                        "price": price, "fraction": round(cv2 / lot, 4) if lot else 0,
                        "close_volume": cv2, "remaining": remaining,
                        "done": False, "manual": True,
                    }
                    self.book.record_action(
                        f"{symbol} 止盈一半已手动设置：到 {price:.5f} 平 {cv2}手"
                        f"（剩 {remaining}手，ticket={ticket}）")
            self._ov_applied_ts[ticket] = ts

    def _load_sl_overrides(self) -> dict:
        """读看板写入的手动止损覆盖（按文件 mtime 缓存）。"""
        try:
            if not SL_OVERRIDES_FILE.exists():
                return {}
            mtime = SL_OVERRIDES_FILE.stat().st_mtime
            if getattr(self, "_sl_ov_mtime", None) == mtime and \
                    getattr(self, "_sl_ov_cache", None) is not None:
                return self._sl_ov_cache
            data = json.loads(SL_OVERRIDES_FILE.read_text(encoding="utf-8"))
            self._sl_ov_mtime = mtime
            self._sl_ov_cache = data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            self._sl_ov_cache = {}
        return self._sl_ov_cache

    def _apply_sl_overrides(self, overrides: dict) -> None:
        """应用看板的手动止损：更新台账、标记手动管理；MT5 的修改由看板完成。"""
        applied = getattr(self, "_sl_ov_applied_ts", None)
        if applied is None:
            applied = self._sl_ov_applied_ts = {}
        for key, ov in overrides.items():
            if not isinstance(ov, dict):
                continue
            try:
                ticket = int(key)
                ts = float(ov.get("ts", 0) or 0)
                sl = float(ov.get("sl", 0) or 0)
            except (ValueError, TypeError):
                continue
            if self._sl_ov_applied_ts.get(ticket, -1.0) >= ts:
                continue
            match = [(sym, inf) for sym, inf in self.book.state["positions"].items()
                     if int(inf.get("ticket", 0)) == ticket]
            if not match:
                continue
            symbol, info = match[0]
            info["sl"] = sl
            self.book.mark_manual_sl(ticket)
            self._sl_ov_applied_ts[ticket] = ts
            self.book.record_action(
                f"{symbol} 手动止损已生效：{sl:.5f}（ticket={ticket}，"
                f"风控不再自动拉回；阶梯保本仍生效）"
            )

    def _check_partial_take_profits(self) -> None:
        """关键位前"止盈一半"（dry 与 live 通用）。

        live 模式下 MT5 上所有本软件 magic 的持仓都会被管理（含网页手动开的），
        台账里没有的计划自动补算一次并记入台账，重启不重复触发。
        """
        from trading.sr import SRParams
        params = SRParams.from_config()
        # 手动覆盖优先于自动开关：用户在网页上明确设置的到价平仓始终生效
        overrides = self._load_partial_overrides()
        if overrides:
            self._apply_partial_overrides(overrides)
        if not params.enabled or not params.partial_enabled:
            return
        targets: list[tuple[str, dict]] = []
        if self.book.mode == "live":
            for p in self.client.get_positions():
                sym = str(p.symbol)
                info = None
                for s, inf in self.book.state["positions"].items():
                    if int(inf.get("ticket", 0)) == int(p.ticket):
                        info = inf
                        break
                if info is None:
                    info = {
                        "ticket": int(p.ticket),
                        "direction": "BUY" if p.type == 0 else "SELL",
                        "volume": float(p.volume),
                        "open_price": float(p.price_open),
                        "sl": float(getattr(p, "sl", 0.0) or 0.0),
                        "tp": float(getattr(p, "tp", 0.0) or 0.0),
                        "locked": LOCK_NONE, "dry_run": False, "manual": True,
                    }
                    self.book.set_position(sym, info)
                    self.book.record_action(
                        f"{sym} 纳入管理：MT5 持仓 ticket={int(p.ticket)}（含关键位止盈一半）"
                    )
                targets.append((sym, info))
        else:
            targets = [(s, inf) for s, inf in list(self.book.state["positions"].items())]

        for symbol, info in targets:
            try:
                plan = info.get("sr_partial")
                if plan is None:
                    plan = self._backfill_partial_plan(symbol, info, params)
                if not plan or plan.get("done"):
                    continue
                trigger = float(plan.get("price") or 0)
                if trigger <= 0:
                    continue
                tick = self.client.get_tick(symbol)
                if tick is None:
                    continue
                direction = info["direction"]
                hit = tick["bid"] >= trigger if direction == "BUY" \
                    else tick["ask"] <= trigger
                if not hit:
                    continue
                close_vol = float(plan.get("close_volume") or 0)
                remaining = float(plan.get("remaining") or 0)
                ok = self.client.close_position(symbol, int(info["ticket"]),
                                                volume=close_vol)
                if not ok:
                    continue          # 下一圈重试
                if self.book.mode == "dry" and remaining > 0:
                    info["volume"] = remaining
                plan["done"] = True
                self.book.record_action(
                    f"{symbol} 关键位前止盈一半：平 {close_vol}手（剩 {remaining}手）"
                    f" @≈{trigger:.5f}"
                )
            except Exception as exc:
                logger.error(f"[S/R] {symbol} 止盈一半处理异常: {exc}")

    def _backfill_partial_plan(self, symbol: str, info: dict, params) -> dict:
        """存量仓位补一份"止盈一半"计划（只算一次，结果写入台账）。"""
        plan: dict = {"price": 0.0, "done": True, "skipped": "init"}
        try:
            tf_const, _tf_secs, _tf_str = self._tf_for_strategy(info.get("strategy", ""))
            rates = self.client.copy_rates(symbol, tf_const, self._signal_bars)
            lot = float(info.get("volume") or 0)
            if rates is None or len(rates) < 120 or lot <= 0:
                info["sr_partial"] = plan
                return plan
            from trading.sr import sl_tp_for_trade
            p = sl_tp_for_trade(symbol, info["direction"],
                                float(info.get("open_price") or 0),
                                rates[:-1], params, fixed_sl=0.0)
            partial = p.get("partial")
            ok_split, close_vol, remaining = self._split_lot(
                symbol, lot, float(partial.get("fraction", 0.5) or 0.5))
            if partial and partial.get("price") and ok_split:
                plan = {
                    "price": float(partial["price"]),
                    "fraction": float(partial.get("fraction", 0.5)),
                    "close_volume": close_vol, "remaining": remaining,
                    "done": False, "backfilled": True,
                }
                logger.info(
                    f"[S/R] {symbol} 补算止盈一半计划：{plan['price']:.5f}"
                    f"（平 {close_vol}手）"
                )
            info["sr_partial"] = plan
        except Exception as exc:
            logger.warning(f"[S/R] {symbol} 补算止盈一半失败: {exc}")
            info["sr_partial"] = plan
        return plan

    def _check_dry_take_profits(self) -> None:
        """dry-run：bid/ask 触及台账止盈价 → 模拟平仓并记录。"""
        for symbol, info in list(self.book.state["positions"].items()):
            tp = float(info.get("tp") or 0.0)
            if tp <= 0:
                continue
            tick = self.client.get_tick(symbol)
            if tick is None:
                continue
            direction = info["direction"]
            hit = tick["bid"] >= tp if direction == "BUY" else tick["ask"] <= tp
            if not hit:
                continue
            open_price = float(info["open_price"])
            profit = profit_pct(direction, open_price, tick["bid"], tick["ask"])
            ok = self.client.close_position(symbol, int(info["ticket"]))
            if ok:
                self.book.clear_symbol(symbol)
                self.book.record_action(
                    f"{symbol} [DRY-RUN] 触发止盈平仓 @≈{tp:.5f}（{profit*100:+.2f}%）"
                )

    def _normalize_lot(self, symbol: str, lot: float) -> float:
        step = self.client.volume_step(symbol) or 0.01
        lot = max(step, round(lot / step) * step)
        max_lot = float(getattr(Config, "MAX_LOT_PER_TRADE", 1.0) or 1.0)
        return round(min(lot, max_lot), 2)

    # ── 信号 → 对账 ─────────────────────────────────────────────

    def _reconcile(self, binding: dict, direction: str, strength: float) -> None:
        symbol = binding["symbol"]
        strategy_path = binding.get("strategy_file", "")
        target = DIRECTION_TO_INT.get(direction, 0)
        current = self._current_direction(symbol)
        action = reconcile_action(current if current in (-1, 0, 1) else 0, target)

        if current == 2:
            # 对冲残留：先全部平掉（本轮不开新仓，下根K线再对账）
            self.book.record_action(f"{symbol} 检测到双向持仓，先全部平掉")
            self._close_symbol(symbol)
            return

        if action == "HOLD":
            return

        if action == "CLOSE":
            self.book.record_action(
                f"{symbol} 信号 {direction}（强度{strength:.2f}）→ 平仓"
            )
            self._close_symbol(symbol)
            return

        if action in ("OPEN_LONG", "OPEN_SHORT"):
            want = "BUY" if action == "OPEN_LONG" else "SELL"
            # 防御：开仓前确保没有残留仓位
            if self._current_direction(symbol) != 0:
                if not self._close_symbol(symbol):
                    self.book.record_action(f"{symbol} 开仓前清理旧仓失败，禁止开新仓")
                    return
            self.book.record_action(
                f"{symbol} 信号 {direction}（强度{strength:.2f}）→ 开{'多' if want=='BUY' else '空'}"
            )
            self._open_position(symbol, want, binding, strategy_path)
            return

        if action in ("REVERSE_TO_LONG", "REVERSE_TO_SHORT"):
            want = "BUY" if action == "REVERSE_TO_LONG" else "SELL"
            self.book.record_action(
                f"{symbol} 信号反向 {direction}（强度{strength:.2f}）→ 先平旧仓再反手"
            )
            if not self._close_symbol(symbol):
                self.book.record_action(f"{symbol} 反手失败：平旧仓未成功，禁止开反向仓")
                return
            self._open_position(symbol, want, binding, strategy_path)
            return

    def _process_symbol(self, binding: dict, force: bool = False) -> None:
        """对单个绑定品种：检查新收盘K线 → 算信号 → 对账。

        最后一根 bar 是形成中的当前 bar，信号一律丢弃它、只用已收盘 bar，
        与训练数据（历史收盘K线）保持一致。
        """
        symbol = binding["symbol"]
        strategy_path = binding.get("strategy_file", "")
        strategy = self.strategies.get(strategy_path)
        if strategy is None:
            return
        if "error" in strategy:
            self._last_signal_info[symbol] = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "state": "error", "message": strategy["error"],
            }
            return

        # 按策略自身训练周期拉取 K 线（H1 策略用 H1，M30 策略用 M30），
        # 不再固定用 Config.TIMEFRAME，避免不同周期策略用错周期算信号。
        tf_str = str(strategy.get("timeframe") or "H1").upper()
        tf_const = Config.get_timeframe(tf_str)
        tf_secs = Config.timeframe_seconds(tf_str)

        rates = self.client.copy_rates(symbol, tf_const, self._signal_bars)
        if rates is None or len(rates) < 2:
            self._last_signal_info[symbol] = {
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "state": "error", "message": "K线拉取失败（品种不存在或未连接）",
            }
            return

        closed = rates[:-1]  # 丢弃形成中的最后一根
        last_closed_time = int(closed["time"][-1])
        if not force and self._last_bar_time.get(symbol) == last_closed_time:
            return  # 没有新收盘K线

        raw_dict = rates_to_raw_dict(closed)
        signal = compute_signal(
            [strategy["formula"]], raw_dict,
            min_trade_exposure=float(Config.MIN_TRADE_EXPOSURE),
        )
        self._last_bar_time[symbol] = last_closed_time
        signal_entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # MT5 的 bar time 是「开盘时间」；H1 的 13:00 那根要到 14:00 才收盘。
            # 这里换算成收盘时间显示，避免总览页看起来「永远晚一小时」。
            "bar_time": last_closed_time + tf_secs,
            "bar_open_time": last_closed_time,
            "timeframe": tf_str,
            "direction": signal.get("direction", "FLAT"),
            "strength": signal.get("strength", 0.0),
            "position": signal.get("position", 0.0),
            "state": signal.get("state"),
            "message": signal.get("message", ""),
            "strategy": strategy_path,
        }
        self._last_signal_info[symbol] = signal_entry
        logger.info(
            f"[信号] {symbol} {tf_str} 收盘bar={last_closed_time + tf_secs} "
            f"{signal.get('direction')} "
            f"强度={signal.get('strength')} ({signal.get('state')} {signal.get('message','')})"
        )

        if signal.get("state") != "ok":
            return
        self.book.save()  # 对账前先落盘
        self._reconcile(binding, signal["direction"], float(signal.get("strength", 0.0)))

    # ── 实时风控 ────────────────────────────────────────────────

    def _monitor_positions(self) -> None:
        if not self.risk.params.enable_price_monitor:
            return
        # 关键位前止盈一半（两种模式都做，排在整单止盈之前）
        self._check_partial_take_profits()
        if self.book.mode == "dry":
            # dry-run 模拟止盈：MT5 服务器不会帮本地台账执行 TP，这里手动触发
            self._check_dry_take_profits()
            targets = []
            for symbol, info in list(self.book.state["positions"].items()):
                targets.append((symbol, int(info["ticket"]), info["direction"],
                                float(info["open_price"]), float(info.get("sl", 0.0)),
                                self.book.locked_for(info["ticket"]), True))
        else:
            targets = []
            for p in self.client.get_positions():
                symbol = str(p.symbol)
                targets.append((symbol, int(p.ticket),
                                "BUY" if p.type == 0 else "SELL",
                                float(p.price_open), float(getattr(p, "sl", 0.0) or 0.0),
                                self.book.locked_for(p.ticket), False))
        for symbol, ticket, direction, open_price, sl, locked, is_dry in targets:
            try:
                moved, new_locked, reason = self.risk.protect_position(
                    symbol, ticket, direction, open_price, sl, locked
                )
            except Exception as exc:
                logger.error(f"[风控] {symbol} ticket={ticket} 处理异常: {exc}")
                continue
            if moved:
                self.book.set_locked(ticket, new_locked)
            elif new_locked != locked and reason is None:
                self.book.set_locked(ticket, new_locked)

        # dry-run 模拟仓的 sl / profit_pct 字段同步（供看板展示）
        if self.book.mode == "dry":
            for symbol, info in self.book.state["positions"].items():
                tick = self.client.get_tick(symbol)
                if tick is None:
                    continue
                direction = info["direction"]
                open_price = float(info["open_price"])
                profit = profit_pct(direction, open_price, tick["bid"], tick["ask"])
                locked_now = self.book.locked_for(info["ticket"])
                # dry-run 的 modify_sl 不会有 MT5 持仓对象可回读，
                # 因此依据台账里的已锁定档位持续重建页面显示的止损价。
                if locked_now > LOCK_NONE:
                    info["sl"] = lock_price(direction, open_price, locked_now)
                elif not info.get("sl"):
                    info["sl"] = initial_stop_price(
                        direction, open_price, self.risk.params.stop_loss_pct)
                info["profit_pct"] = round(profit * 100, 2)

    # ── 状态上报 ────────────────────────────────────────────────

    def _write_status(self, phase: str = "running") -> None:
        account = None
        ai = self.client.account_info()
        if ai is not None:
            account = {
                "login": int(ai.login),
                "server": getattr(ai, "server", ""),
                "balance": round(float(ai.balance), 2),
                "equity": round(float(ai.equity), 2),
                "margin_free": round(float(ai.margin_free), 2),
                "currency": getattr(ai, "currency", "USD"),
            }
        cfg = load_trader_config()
        server_offset = getattr(self, "_server_offset", None)
        # live 模式下展示 MT5 上全部本软件持仓（含手动开的），dry 模式展示台账；
        # 都是列表：同一品种可能有多笔持仓（手动+策略），按 ticket 逐笔列出
        if self.book and self.book.mode == "live" and self.client.connected:
            from trading.mt5_client import position_to_dict
            live_list: list = []
            for p in self.client.get_positions():
                info = position_to_dict(self.client, p, server_offset)
                book_info = self.book.get_position(info["symbol"]) if self.book else None
                if book_info and int(book_info.get("ticket", 0)) == int(p.ticket) \
                        and book_info.get("sr_partial"):
                    info["sr_partial"] = book_info["sr_partial"]
                live_list.append(info)
            positions_view = live_list
        else:
            positions_view = [
                dict(info, symbol=sym)
                for sym, info in (self.book.state.get("positions", {}).items()
                                  if self.book else [])
            ]
        status = {
            "running": phase == "running",
            "phase": phase,
            "pid": os.getpid(),
            "mode": "dry" if cfg.get("dry_run", True) else "live",
            "dry_run": bool(cfg.get("dry_run", True)),
            "started_at": self._started_at,
            "last_loop": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "loop_count": self._loop_count,
            "connected": self.client.connected,
            "trade_allowed": self.client.trade_allowed(),
            "server_offset_sec": server_offset,
            "account": account,
            "signals": self._last_signal_info,
            "positions": positions_view,
            "actions": self.book.state.get("actions", [])[-30:] if self.book else [],
            "bindings": [
                {"symbol": b.get("symbol"), "strategy_file": b.get("strategy_file"),
                 "lot": b.get("lot")}
                for b in self.bindings
            ],
            "last_error": self._last_error,
        }
        try:
            STATUS_FILE.parent.mkdir(exist_ok=True)
            tmp = STATUS_FILE.with_suffix(".tmp")
            tmp.write_text(json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8")
            tmp.replace(STATUS_FILE)
        except OSError:
            pass

    # ── 主循环 ──────────────────────────────────────────────────

    def run(self) -> None:
        logger.info(f"[runner] 启动 pid={os.getpid()} dry_run={Config.DRY_RUN}")
        self._startup_dry_run = bool(Config.DRY_RUN)
        self.book = PositionBook("dry" if self._startup_dry_run else "live")
        self._refresh_bindings()
        self._write_status("starting")

        # 连接（终端未运行时不拉起，只报错等待）
        try:
            self.client.connect()
            self._last_error = None
        except (ConnectionError, RuntimeError) as exc:
            self._last_error = str(exc)
            logger.error(f"[runner] MT5 连接失败: {exc}")
            self._write_status("waiting_mt5")

        while not self._stop:
            loop_t0 = time.time()
            self._loop_count += 1
            try:
                if self._check_stop_signal():
                    logger.info("[runner] 收到停止信号，退出")
                    break

                # 热更新配置与绑定
                Config.reload()
                self.client.dry_run = Config.DRY_RUN
                if bool(Config.DRY_RUN) != self._startup_dry_run:
                    self._last_error = "运行模式已修改，请停止并重新启动 Runner 使其生效"
                    logger.warning(f"[runner] {self._last_error}")
                    self._write_status("mode_changed")
                    break
                self.book.save()
                self._refresh_bindings()
                self.risk.refresh_params()
                # 手动止损覆盖：应用到台账，并让风控豁免这些仓位的自动拉回
                sl_overrides = self._load_sl_overrides()
                if sl_overrides:
                    self._apply_sl_overrides(sl_overrides)
                self.risk.params.manual_sl_tickets = self.book.manual_sl_tickets()
                self._signal_bars = max(800, int(load_trader_config().get("signal_bars", 1000)))

                # 连接维护
                if not self.client.connected or self.client.account_info() is None:
                    if not self.client.ensure_connected():
                        self._last_error = "MT5 未连接（请先打开 TMGM 终端并登录）"
                        self._write_status("waiting_mt5")
                        self._sleep_loop(loop_t0)
                        continue
                self._last_error = None
                self._server_offset = self.client.server_time_offset()

                # 同步 live 持仓 → 台账
                if self.book.mode == "live":
                    self._sync_live_positions()

                # 信号循环：新K线才对账
                for binding in self.bindings:
                    self._process_symbol(binding)

                # 实时风控循环：每圈执行（5~10秒）
                self._monitor_positions()

                self.book.save()
                self._write_status("running")
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.exception("[runner] 循环异常")
                self._write_status("error")
            self._sleep_loop(loop_t0)

        self.client.disconnect()
        self._write_status("stopped")
        logger.info("[runner] 已退出")

    def _sleep_loop(self, loop_t0: float) -> None:
        interval = max(1, int(self.risk.params.price_monitor_interval))
        elapsed = time.time() - loop_t0
        remaining = interval - elapsed
        # 分段睡眠，便于快速响应停止信号
        deadline = time.time() + max(0.0, remaining)
        while time.time() < deadline and not self._stop:
            if self._check_stop_signal():
                self._stop = True
                break
            time.sleep(min(1.0, max(0.05, deadline - time.time())))


def main() -> None:
    from config import TRADER_CONFIG_FILE
    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    # 子进程的 stdout/stderr 会被看板重定向到同一个日志文件。
    # 移除 loguru 默认彩色 stderr sink，避免同一条日志写两遍并混入 ANSI 控制符。
    logger.remove()
    logger.add(
        log_dir / "trading_runner.log",
        rotation="5 MB",
        retention=10,
        encoding="utf-8",
        colorize=False,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        enqueue=False,
    )
    # 初始模式以 trader_config.json 为准
    cfg = load_trader_config()
    Config.DRY_RUN = bool(cfg.get("dry_run", True))
    runner = TradingRunner()
    try:
        runner.run()
    except KeyboardInterrupt:
        runner.stop()
        runner._write_status("stopped")


if __name__ == "__main__":
    main()

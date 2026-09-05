"""
trading/mt5_client.py — MT5 连接与订单执行

特性：
  - 连接依赖终端已打开并登录（绝不自动拉起 MT5 终端）
  - 市价开仓（可带初始止损）、按 ticket 平仓、全部平仓
  - 填充模式自动重试：IOC → FOK → RETURN
  - 修改止损（TRADE_ACTION_SLTP），处理券商最小止损距离
  - dry_run：所有下单动作只记日志并返回合成成功
  - magic number 过滤
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from loguru import logger

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:  # pragma: no cover - 测试环境
    _MT5_AVAILABLE = False
    mt5 = None

from config import Config


RETCODE_HINTS = {
    10027: "MT5 终端未启用算法交易：请点亮 MT5 工具栏的“算法交易/AutoTrading”按钮",
    10030: "券商不支持该填充模式（软件会自动按品种 filling_mode 重试）",
    10019: "资金不足",
    10016: "止损/止盈价格无效（可能离市价太近）",
    10018: "市场已关闭",
    10014: "手数无效",
    10013: "请求参数无效",
    10006: "订单被拒绝",
}


def retcode_hint(retcode) -> str:
    """把 MT5 retcode 翻译成中文提示；未知码返回空字符串。"""
    try:
        return RETCODE_HINTS.get(int(retcode), "")
    except (TypeError, ValueError):
        return ""


def position_to_dict(client: "MT5Client", p, server_offset: int | None = None) -> dict:
    """把 MT5 持仓对象转成看板友好的 dict（含现价、浮动盈亏、开仓时间）。"""
    symbol = str(p.symbol)
    direction = "BUY" if p.type == 0 else "SELL"
    open_price = float(p.price_open)
    volume = float(p.volume)
    sl = float(getattr(p, "sl", 0.0) or 0.0)
    tp = float(getattr(p, "tp", 0.0) or 0.0)
    profit = float(getattr(p, "profit", 0.0) or 0.0)
    tick = client.get_tick(symbol)
    current = None
    profit_pct = None
    if tick is not None and open_price > 0:
        current = tick["bid"] if direction == "BUY" else tick["ask"]
        pct = (current - open_price) / open_price
        profit_pct = round((pct if direction == "BUY" else -pct) * 100, 2)
    open_time_server = int(getattr(p, "time", 0) or 0)
    # 服务器伪时间戳 → 真实 UTC 之后的本地时间字符串
    if open_time_server > 0:
        real_utc = open_time_server - (server_offset or 0)
        open_time = datetime.fromtimestamp(real_utc).strftime("%Y-%m-%d %H:%M:%S")
    else:
        open_time = ""
    return {
        "ticket": int(p.ticket),
        "symbol": symbol,
        "direction": direction,
        "volume": volume,
        "open_price": open_price,
        "current_price": current,
        "sl": sl,
        "tp": tp,
        "profit": round(profit, 2),
        "profit_pct": profit_pct,
        "open_time": open_time,
        "dry_run": False,
    }


class MT5Client:
    """MT5 订单执行与行情访问。dry_run=True 时下单类动作只记日志。"""

    def __init__(self, magic: int | None = None, dry_run: bool | None = None) -> None:
        self.magic = int(magic if magic is not None else Config.MAGIC_NUMBER)
        self.dry_run = bool(Config.DRY_RUN if dry_run is None else dry_run)
        self._connected = False

    # ── 连接 ─────────────────────────────────────────────────────

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        """初始化 MT5 连接。终端未运行/未登录时抛 ConnectionError，绝不自动启动终端。"""
        if not _MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 包不可用")
        if not mt5.initialize():
            raise ConnectionError(
                f"MT5 初始化失败: {mt5.last_error()}（请先打开 TMGM 终端并登录）"
            )
        ai = mt5.account_info()
        if ai is None:
            mt5.shutdown()
            raise ConnectionError(f"MT5 未登录账号: {mt5.last_error()}")
        self._connected = True
        logger.info(
            f"[MT5Client] 已连接 {mt5.terminal_info().name}，"
            f"账号 {ai.login}，dry_run={self.dry_run}"
        )

    def disconnect(self) -> None:
        if _MT5_AVAILABLE and self._connected:
            try:
                mt5.shutdown()
            except Exception:  # pragma: no cover
                pass
        self._connected = False

    def ensure_connected(self) -> bool:
        """断线时自动重连（终端被关闭后重新打开的场景）。"""
        if self._connected and mt5.account_info() is not None:
            return True
        logger.warning("[MT5Client] 连接失效，尝试重连...")
        self._connected = False
        try:
            self.connect()
            return True
        except (ConnectionError, RuntimeError) as exc:
            logger.error(f"[MT5Client] 重连失败: {exc}")
            return False

    # ── 账户 / 行情 ──────────────────────────────────────────────

    def account_info(self):
        if not self._connected:
            return None
        return mt5.account_info()

    def terminal_info(self):
        if not self._connected:
            return None
        return mt5.terminal_info()

    def trade_allowed(self) -> bool:
        """MT5 终端是否允许程序下单（工具栏“算法交易”开关）。"""
        ti = self.terminal_info()
        if ti is None:
            return False
        return bool(getattr(ti, "trade_allowed", False))

    def symbol_info(self, symbol: str):
        if not self._connected:
            return None
        return mt5.symbol_info(symbol)

    def ensure_symbol_selected(self, symbol: str) -> bool:
        info = self.symbol_info(symbol)
        if info is not None and info.visible:
            return True
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"[MT5Client] 品种 {symbol} 不可用: {mt5.last_error()}")
            return False
        return True

    def get_tick(self, symbol: str) -> dict | None:
        """返回 {"bid","ask","time"}；失败返回 None。"""
        if not self._connected:
            return None
        try:
            tick = mt5.symbol_info_tick(symbol)
        except Exception:  # pragma: no cover
            return None
        if tick is None or tick.bid <= 0 or tick.ask <= 0:
            return None
        return {"bid": float(tick.bid), "ask": float(tick.ask), "time": int(tick.time)}

    def server_time_offset(self) -> int | None:
        """估算 MT5 服务器时区偏移（秒）。

        MT5 的 K 线/成交时间是"服务器墙上时钟"编码成的伪 Unix 时间戳，
        不等于真实 UTC。用最新 tick 时间和本机 UTC 时间之差即可估算偏移。
        返回 None 表示暂时拿不到报价。
        """
        if not self._connected:
            return None
        for symbol in ("ETHUSD_", "BTCUSD_", "EURUSD", "GBPUSD", "XAUUSD"):
            tick = self.get_tick(symbol)
            if tick is not None:
                offset = tick["time"] - time.time()
                # 对齐到 30 分钟； Broker 时区几乎都是整小时或半点
                return int(round(offset / 1800.0) * 1800)
        return None

    def copy_rates(self, symbol: str, timeframe: int, count: int):
        """拉取 K 线（从最新往回 count 根），返回 MT5 structured array 或 None。"""
        if not self._connected:
            return None
        try:
            return mt5.copy_rates_from_pos(symbol, timeframe, 0, int(count))
        except Exception:
            return None

    # ── 持仓 ─────────────────────────────────────────────────────

    def get_positions(self, symbol: str | None = None, magic: int | None = None) -> list:
        """返回持仓 namedtuple 列表，按 magic 过滤。"""
        if not self._connected:
            return []
        try:
            if symbol is not None:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
        except Exception:  # pragma: no cover
            return []
        if positions is None:
            return []
        want_magic = self.magic if magic is None else magic
        return [p for p in positions if getattr(p, "magic", None) == want_magic]

    # ── 下单 ─────────────────────────────────────────────────────

    def market_open(
        self,
        symbol: str,
        direction: str,
        lot: float,
        sl: float | None = None,
        tp: float | None = None,
        comment: str = "MT5AutoTrader",
    ) -> dict[str, Any]:
        """市价开仓。direction: 'BUY'/'SELL'。返回 {"ok": bool, "retcode", "comment", "order"}。"""
        lot = float(lot)
        if lot <= 0:
            return {"ok": False, "retcode": None, "comment": f"手数无效 lot={lot}", "order": 0}
        if self.dry_run:
            logger.info(
                f"[DRY-RUN] 开仓 {symbol} {direction} {lot} 手 SL={sl} TP={tp}"
            )
            return {"ok": True, "retcode": 10009, "comment": "dry-run", "order": 0}
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"[MT5Client] {symbol}: 无法获取报价: {mt5.last_error()}")
            return {"ok": False, "retcode": None, "comment": "no tick", "order": 0}
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        price = tick.ask if direction == "BUY" else tick.bid
        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": float(price),
            "deviation": int(getattr(Config, "DEVIATION_POINTS", 20)),
            "magic": self.magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        if sl is not None and sl > 0:
            request["sl"] = float(sl)
        if tp is not None and tp > 0:
            request["tp"] = float(tp)
        return self._send_request(request)

    def order_check(self, symbol: str, direction: str, lot: float) -> dict[str, Any]:
        """调用 MT5 order_check 进行下单前检查，不发送订单。"""
        if not self._connected:
            return {"ok": False, "message": "MT5 未连接"}
        lot = float(lot)
        if direction not in ("BUY", "SELL") or lot <= 0:
            return {"ok": False, "message": "品种、方向或手数无效"}
        if not self.ensure_symbol_selected(symbol):
            return {"ok": False, "message": f"品种不可用: {symbol}"}
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"ok": False, "message": f"无法获取报价: {mt5.last_error()}"}
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": float(tick.ask if direction == "BUY" else tick.bid),
            "deviation": int(getattr(Config, "DEVIATION_POINTS", 20)),
            "magic": self.magic,
            "comment": "MT5AutoTrader order check",
            "type_time": mt5.ORDER_TIME_GTC,
        }
        last = None
        attempts = []
        for filling in self._fill_modes(symbol):
            request["type_filling"] = filling
            try:
                result = mt5.order_check(dict(request))
            except Exception as exc:
                return {"ok": False, "message": f"order_check 异常: {exc}"}
            if result is None:
                attempts.append({"filling": filling, "retcode": None, "comment": str(mt5.last_error())})
                continue
            retcode = int(getattr(result, "retcode", -1))
            comment = str(getattr(result, "comment", ""))
            attempts.append({"filling": filling, "retcode": retcode, "comment": comment})
            last = result
            if retcode not in (mt5.TRADE_RETCODE_INVALID_FILL, 10030):
                break
        if last is None:
            return {"ok": False, "retcode": None, "message": str(mt5.last_error()), "attempts": attempts}
        retcode = int(getattr(last, "retcode", -1))
        return {
            "ok": retcode in (0, 10009),
            "retcode": retcode,
            "comment": str(getattr(last, "comment", "")),
            "balance": getattr(last, "balance", None),
            "equity": getattr(last, "equity", None),
            "margin": getattr(last, "margin", None),
            "free_margin": getattr(last, "margin_free", None),
            "attempts": attempts,
            "request": {"symbol": symbol, "direction": direction, "volume": lot},
        }

    def close_position(self, symbol: str, ticket: int,
                       volume: float | None = None) -> bool:
        """按 ticket 平仓。volume 给定时只平该手数（部分平仓），否则全部平掉。"""
        if self.dry_run:
            logger.info(f"[DRY-RUN] 平仓 {symbol} ticket={ticket} vol={volume or '全部'}")
            return True
        position = self._find_ticket(ticket)
        if position is None:
            logger.warning(f"[MT5Client] 未找到 ticket={ticket} 的持仓（可能已平）")
            return not self._ticket_exists(ticket)  # 已不存在视为成功
        close_volume = float(volume) if (volume is not None and volume > 0) \
            else float(position.volume)
        if close_volume > float(position.volume) + 1e-9:
            close_volume = float(position.volume)
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"[MT5Client] {symbol}: 无法获取报价: {mt5.last_error()}")
            return False
        close_type = mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY
        price = tick.bid if position.type == 0 else tick.ask
        result = self._send_request(
            {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": close_volume,
                "type": close_type,
                "position": int(position.ticket),
                "price": float(price),
                "deviation": 20,
                "magic": self.magic,
                "comment": "MT5AutoTrader close",
                "type_time": mt5.ORDER_TIME_GTC,
            }
        )
        return bool(result["ok"])

    def close_symbol_all(self, symbol: str) -> bool:
        """平掉该品种（magic 过滤）的全部持仓；全部成功才返回 True。"""
        positions = self.get_positions(symbol)
        if not positions:
            return True
        ok = True
        for p in positions:
            if not self.close_position(symbol, int(p.ticket)):
                ok = False
            time.sleep(0.1)
        return ok

    def modify_sl(self, symbol: str, ticket: int, new_sl: float | None = None,
                  tp: float | None = None) -> bool:
        """修改持仓止损/止盈（至少给一个）。dry_run 时只记日志。"""
        if self.dry_run:
            logger.info(f"[DRY-RUN] 修改止损 {symbol} ticket={ticket} SL→{new_sl} TP→{tp}")
            return True
        request: dict[str, Any] = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": int(ticket),
        }
        if new_sl is not None and new_sl > 0:
            request["sl"] = float(new_sl)
        if tp is not None and tp > 0:
            request["tp"] = float(tp)
        result = self._send_request(request)
        return bool(result["ok"])

    def stops_level_points(self, symbol: str) -> int:
        """券商最小止损距离（点）。symbol_info 不可用时返回 0。"""
        info = self.symbol_info(symbol)
        if info is None:
            return 0
        return int(getattr(info, "trade_stops_level", 0) or 0)

    def point(self, symbol: str) -> float:
        info = self.symbol_info(symbol)
        if info is None:
            return 0.0
        return float(getattr(info, "point", 0.0) or 0.0)

    def volume_step(self, symbol: str) -> float:
        info = self.symbol_info(symbol)
        if info is None:
            return 0.01
        return float(getattr(info, "volume_step", 0.01) or 0.01)

    def volume_min(self, symbol: str) -> float:
        info = self.symbol_info(symbol)
        if info is None:
            return 0.01
        return float(getattr(info, "volume_min", 0.01) or 0.01)

    # ── 内部 ─────────────────────────────────────────────────────

    def _find_ticket(self, ticket: int):
        for p in self.get_positions():
            if int(p.ticket) == int(ticket):
                return p
        return None

    def _ticket_exists(self, ticket: int) -> bool:
        return self._find_ticket(ticket) is not None

    def _fill_modes(self, symbol: str | None = None) -> list[int]:
        """返回可尝试的填充模式；优先使用品种的 filling_mode。"""
        if not _MT5_AVAILABLE:
            return [0]
        modes: list[int] = []
        info = mt5.symbol_info(symbol) if symbol else None
        flags = int(getattr(info, "filling_mode", 0) or 0) if info is not None else 0
        # MT5 的 filling_mode 是位掩码：FOK=1、IOC=2；RETURN 通常由服务器默认支持。
        if flags & 1:
            modes.append(getattr(mt5, "ORDER_FILLING_FOK", 0))
        if flags & 2:
            modes.append(getattr(mt5, "ORDER_FILLING_IOC", 1))
        modes.extend([
            getattr(mt5, "ORDER_FILLING_FOK", 0),
            getattr(mt5, "ORDER_FILLING_IOC", 1),
            getattr(mt5, "ORDER_FILLING_RETURN", 2),
        ])
        return list(dict.fromkeys(modes))

    def _send_request(self, request: dict) -> dict[str, Any]:
        """发送订单，填充模式自动重试。返回 {"ok","retcode","comment","order"}。"""
        last_retcode = None
        last_comment = ""
        last_order = 0
        for filling in self._fill_modes(request.get("symbol")):
            req = dict(request)
            req["type_filling"] = filling
            try:
                result = mt5.order_send(req)
            except Exception as exc:  # pragma: no cover
                logger.error(f"[MT5Client] order_send 异常: {exc}")
                return {"ok": False, "retcode": None, "comment": str(exc), "order": 0}
            if result is None:
                last_retcode, last_comment = mt5.last_error()
                continue
            last_retcode = result.retcode
            last_comment = result.comment
            last_order = int(getattr(result, "order", 0) or 0)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                logger.success(
                    f"[MT5Client] 订单成交 ticket={last_order} {request.get('symbol')} "
                    f"vol={request.get('volume')} sl={request.get('sl')}"
                )
                return {"ok": True, "retcode": result.retcode, "comment": "", "hint": "", "order": last_order}
            if result.retcode not in (mt5.TRADE_RETCODE_INVALID_FILL, 10030):
                break  # 非填充模式错误，重试无意义
        logger.error(
            f"[MT5Client] 订单失败: {request.get('symbol')} vol={request.get('volume')} "
            f"retcode={last_retcode} comment={last_comment} {retcode_hint(last_retcode)}"
        )
        return {
            "ok": False,
            "retcode": last_retcode,
            "comment": str(last_comment),
            "hint": retcode_hint(last_retcode),
            "order": last_order,
        }

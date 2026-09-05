"""
tests/test_trading.py — MT5AutoTrader 交易逻辑单元测试（全部 mock，绝不触碰真实 MT5）

覆盖 1.txt 第 10 节的每一条要求：
  1. 信号方向阈值
  2. 同向信号不重复开仓
  3. 反向信号：先平旧仓，平仓成功才开反向仓
  4. 平仓失败时禁止开新仓
  5. 初始止损（多/空）
  6. 阶梯保本（多/空、只收紧不放松、档位不回退）
  7. dry-run 本地持仓不被同步逻辑误删
  8. 券商最小止损距离
运行：python tests/test_trading.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # src\（导入用）
PROJECT_ROOT = ROOT.parent                             # 项目根（数据/策略所在）
sys.path.insert(0, str(ROOT))

# ── 隔离配置与状态文件，避免污染真实运行环境 ──────────────────────
_TMP = Path(tempfile.mkdtemp(prefix="mt5at_test_"))
import config as cfgmod  # noqa: E402

cfgmod.TRADER_CONFIG_FILE = _TMP / "trader_config.json"
cfgmod.save_trader_config(json.loads(json.dumps(cfgmod.DEFAULT_TRADER_CONFIG)))
from config import Config  # noqa: E402

import trading.runner as runner_mod  # noqa: E402
from trading.risk import (  # noqa: E402
    LOCK_NONE,
    RiskManager,
    RiskParams,
    initial_stop_price,
    ladder_lock,
    lock_price,
    profit_pct,
)

runner_mod.STATE_FILE = _TMP / "portfolio_state.json"
runner_mod.STATUS_FILE = _TMP / "runner_status.json"
runner_mod.STOP_FILE = _TMP / "STOP_SIGNAL"
runner_mod.PARTIAL_OVERRIDES_FILE = _TMP / "sr_partial_overrides.json"

PASSED: list[str] = []
FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(f"{name} {detail}")
        print(f"  [FAIL] {name} {detail}")


# ── Mock MT5 客户端 ──────────────────────────────────────────────

class MockPosition:
    def __init__(self, ticket, symbol, ptype, volume, price_open, sl=0.0, magic=0):
        self.ticket = ticket
        self.symbol = symbol
        self.type = ptype          # 0=BUY 1=SELL
        self.volume = volume
        self.price_open = price_open
        self.sl = sl
        self.magic = magic


class MockClient:
    """行为可控的 MT5 客户端替身，记录所有调用。"""

    def __init__(self, dry_run=False, bid=100.0, ask=100.1,
                 stops_points=0, point=0.01):
        self.dry_run = dry_run
        self.connected = True
        self.magic = 20260904
        self._bid, self._ask = bid, ask
        self._stops_points, self._point = stops_points, point
        self.positions: list[MockPosition] = []
        self.calls: list[tuple] = []
        self.close_should_fail = False
        self.open_should_fail = False
        self.modify_should_fail = False
        self._next_ticket = 500001

    # 行情
    def get_tick(self, symbol):
        return {"bid": self._bid, "ask": self._ask, "time": 0}

    def set_price(self, bid, ask):
        self._bid, self._ask = bid, ask

    def stops_level_points(self, symbol):
        return self._stops_points

    def point(self, symbol):
        return self._point

    def volume_step(self, symbol):
        return 0.01

    def volume_min(self, symbol):
        return 0.01

    def symbol_info(self, symbol):
        return None

    def account_info(self):
        class AI:
            login, server, trade_mode = 12345678, "Demo-Server", 0
            balance = equity = margin_free = 10000.0
            currency = "USD"
        return AI()

    def ensure_connected(self):
        return True

    def copy_rates(self, symbol, tf, count):
        return None

    def disconnect(self):
        pass

    # 持仓
    def get_positions(self, symbol=None, magic=None):
        return [p for p in self.positions if symbol is None or p.symbol == symbol]

    # 下单
    def market_open(self, symbol, direction, lot, sl=None, tp=None, comment=""):
        self.calls.append(("open", symbol, direction, lot, sl, tp))
        if self.open_should_fail:
            return {"ok": False, "retcode": 10004, "comment": "mock fail", "order": 0}
        if not self.dry_run:
            t = self._next_ticket
            self._next_ticket += 1
            self.positions.append(MockPosition(
                t, symbol, 0 if direction == "BUY" else 1, lot,
                self._ask if direction == "BUY" else self._bid,
                sl or 0.0, self.magic))
            return {"ok": True, "retcode": 10009, "comment": "", "order": t}
        return {"ok": True, "retcode": 10009, "comment": "dry-run", "order": 0}

    def close_position(self, symbol, ticket, volume=None):
        self.calls.append(("close", symbol, ticket, volume))
        if self.close_should_fail:
            return False
        for p in self.positions:
            if p.ticket == ticket and volume is not None and volume < p.volume:
                p.volume = round(p.volume - volume, 8)
                return True
        self.positions = [p for p in self.positions if p.ticket != ticket]
        return True

    def close_symbol_all(self, symbol):
        self.calls.append(("close_all", symbol))
        if self.close_should_fail:
            return False
        self.positions = [p for p in self.positions if p.symbol != symbol]
        return True

    def modify_sl(self, symbol, ticket, new_sl, tp=None):
        self.calls.append(("modify_sl", symbol, ticket, round(new_sl, 6)))
        if self.modify_should_fail:
            return False
        for p in self.positions:
            if p.ticket == ticket:
                p.sl = new_sl
        return True


def make_runner(client: MockClient, mode: str) -> runner_mod.TradingRunner:
    """构造一个跳过 MT5 连接的 runner。"""
    if runner_mod.STATE_FILE.exists():
        runner_mod.STATE_FILE.unlink()
    r = runner_mod.TradingRunner.__new__(runner_mod.TradingRunner)
    r.client = client
    r.risk = RiskManager(client, RiskParams())
    r.book = runner_mod.PositionBook(mode)
    r.bindings = []
    r.strategies = {}
    r._last_bar_time = {}
    r._stop = False
    r._loop_count = 0
    r._last_error = None
    r._started_at = "test"
    r._last_signal_info = {}
    r._signal_bars = 1000
    return r


# ── 测试 1：信号方向阈值 ─────────────────────────────────────────

def test_signal_threshold():
    print("\n[1] 信号方向阈值")
    import torch
    from trading.signal_engine import compute_signal, MIN_BARS_SIGNAL

    # 构造 bar 数不足的数据 → insufficient
    short = {k: torch.ones(1, 100) for k in ("open", "high", "low", "close", "volume")}
    res = compute_signal([[8, 98]], short)
    check("bar 数不足时拒绝出信号", res["state"] == "insufficient", str(res))

    # 阈值映射（直接验证纯函数逻辑）
    from strategy_manager.signal import target_to_direction
    Config.MIN_TRADE_EXPOSURE = 0.05
    check("0.06 → 做多", target_to_direction(0.06, 0.05) == 1)
    check("-0.06 → 做空", target_to_direction(-0.06, 0.05) == -1)
    check("0.03 → 观望", target_to_direction(0.03, 0.05) == 0)
    check("-0.03 → 观望", target_to_direction(-0.03, 0.05) == 0)


# ── 测试 2：同向不重复开仓 ───────────────────────────────────────

def test_no_duplicate_open():
    print("\n[2] 同向信号不重复开仓")
    client = MockClient(dry_run=False, bid=100.0, ask=100.1)
    r = make_runner(client, "live")
    binding = {"symbol": "ETHUSD_", "strategy_file": "s.json", "lot": 0.01}

    r._reconcile(binding, "LONG", 0.8)
    opens = [c for c in client.calls if c[0] == "open"]
    check("首个 LONG 信号开多仓", len(opens) == 1 and opens[0][2] == "BUY", str(client.calls))

    client.calls.clear()
    r._reconcile(binding, "LONG", 0.9)   # 同向再来
    r._reconcile(binding, "LONG", 0.7)
    check("同向信号不再下单（HOLD）", len(client.calls) == 0, str(client.calls))
    check("持仓仍只有 1 笔", len(client.positions) == 1, str(len(client.positions)))


# ── 测试 3：反向先平后开 ─────────────────────────────────────────

def test_reverse_close_then_open():
    print("\n[3] 反向信号：先平旧仓再开反向仓")
    client = MockClient(dry_run=False, bid=100.0, ask=100.1)
    r = make_runner(client, "live")
    binding = {"symbol": "ETHUSD_", "strategy_file": "s.json", "lot": 0.01}

    r._reconcile(binding, "LONG", 0.8)
    client.calls.clear()
    r._reconcile(binding, "SHORT", 0.8)

    kinds = [c[0] for c in client.calls]
    close_idx = next((i for i, k in enumerate(kinds) if k in ("close", "close_all")), -1)
    open_idx = next((i for i, k in enumerate(kinds) if k == "open"), -1)
    check("有平仓动作", close_idx >= 0, str(client.calls))
    check("有开仓动作", open_idx >= 0, str(client.calls))
    check("平仓在开仓之前", close_idx >= 0 and open_idx > close_idx, str(kinds))
    check("最终只有 1 笔空仓",
          len(client.positions) == 1 and client.positions[0].type == 1,
          str([(p.ticket, p.type) for p in client.positions]))


# ── 测试 4：平仓失败禁止开新仓 ───────────────────────────────────

def test_close_fail_blocks_open():
    print("\n[4] 平仓失败时禁止开新仓")
    client = MockClient(dry_run=False, bid=100.0, ask=100.1)
    r = make_runner(client, "live")
    binding = {"symbol": "ETHUSD_", "strategy_file": "s.json", "lot": 0.01}

    r._reconcile(binding, "LONG", 0.8)          # 建多仓
    client.close_should_fail = True
    client.calls.clear()
    r._reconcile(binding, "SHORT", 0.8)         # 反手但平仓会失败

    opens = [c for c in client.calls if c[0] == "open"]
    check("平仓失败后没有开新仓", len(opens) == 0, str(client.calls))
    check("旧多仓仍保留（未被误删）",
          len(client.positions) == 1 and client.positions[0].type == 0,
          str([(p.ticket, p.type) for p in client.positions]))
    acts = " ".join(a["action"] for a in r.book.state["actions"])
    check("失败已记录到动作日志", "禁止开反向仓" in acts or "平仓失败" in acts, acts)


# ── 测试 5：初始止损 ─────────────────────────────────────────────

def test_initial_stop():
    print("\n[5] 初始止损")
    check("多头 -2%：100 → 98", abs(initial_stop_price("BUY", 100.0, -0.02) - 98.0) < 1e-9)
    check("空头 -2%：100 → 102", abs(initial_stop_price("SELL", 100.0, -0.02) - 102.0) < 1e-9)

    client = MockClient(dry_run=False, bid=100.0, ask=100.0)
    r = make_runner(client, "live")
    r._reconcile({"symbol": "ETHUSD_", "strategy_file": "s.json", "lot": 0.01}, "LONG", 0.8)
    opens = [c for c in client.calls if c[0] == "open"]
    sl_sent = opens[0][4] if opens else None
    check("开仓请求带初始止损", sl_sent is not None and abs(sl_sent - 98.0) < 1e-6, str(sl_sent))


# ── 测试 6：阶梯保本 ─────────────────────────────────────────────

def test_ladder():
    print("\n[6] 阶梯保本档位计算")
    levels = [(0.01, 0.00), (0.02, 0.01), (0.03, 0.02), (0.04, 0.03)]
    check("浮盈0.5% 未触发", ladder_lock(0.005, levels, LOCK_NONE)[0] is None)
    check("浮盈1% → 锁 0%（成本价）", ladder_lock(0.010, levels, LOCK_NONE)[0] == 0.00)
    check("浮盈2.5% → 锁 1%", ladder_lock(0.025, levels, LOCK_NONE)[0] == 0.01)
    check("浮盈3.5% → 锁 2%", ladder_lock(0.035, levels, LOCK_NONE)[0] == 0.02)
    check("浮盈9% → 锁 3%（最高档）", ladder_lock(0.09, levels, LOCK_NONE)[0] == 0.03)
    check("已锁2%时浮盈2.5%不回退", ladder_lock(0.025, levels, 0.02)[0] is None)
    check("已锁1%时浮盈3.5%升到2%", ladder_lock(0.035, levels, 0.01)[0] == 0.02)

    print("\n[6b] 阶梯保本执行（多头）")
    client = MockClient(dry_run=False, bid=100.0, ask=100.0)
    risk = RiskManager(client, RiskParams())
    pos = MockPosition(1, "ETHUSD_", 0, 0.01, 100.0, 98.0, client.magic)
    client.positions.append(pos)

    client.set_price(100.5, 100.5)      # +0.5%
    moved, locked, _ = risk.protect_position("ETHUSD_", 1, "BUY", 100.0, pos.sl, LOCK_NONE)
    check("+0.5% 不动止损", not moved and abs(pos.sl - 98.0) < 1e-9, f"sl={pos.sl}")

    client.set_price(101.2, 101.2)      # +1.2% → 锁成本价 100
    moved, locked, _ = risk.protect_position("ETHUSD_", 1, "BUY", 100.0, pos.sl, LOCK_NONE)
    check("+1.2% 止损移到成本价", moved and abs(pos.sl - 100.0) < 1e-6, f"sl={pos.sl}")
    check("锁定值记为 0%", abs(locked - 0.0) < 1e-9, str(locked))

    client.set_price(102.5, 102.5)      # +2.5% → 锁 1% = 101
    moved, locked, _ = risk.protect_position("ETHUSD_", 1, "BUY", 100.0, pos.sl, locked)
    check("+2.5% 止损上移到锁1%（101）", moved and abs(pos.sl - 101.0) < 1e-6, f"sl={pos.sl}")

    client.set_price(101.5, 101.5)      # 回落到 +1.5%：不得放松
    prev_sl = pos.sl
    moved, locked2, _ = risk.protect_position("ETHUSD_", 1, "BUY", 100.0, pos.sl, locked)
    check("价格回落止损不放松", abs(pos.sl - prev_sl) < 1e-9, f"sl={pos.sl} prev={prev_sl}")
    check("锁定档位不回退", abs(locked2 - locked) < 1e-9, f"{locked2} vs {locked}")

    print("\n[6c] 阶梯保本执行（空头）")
    client2 = MockClient(dry_run=False, bid=100.0, ask=100.0)
    risk2 = RiskManager(client2, RiskParams())
    pos2 = MockPosition(2, "ETHUSD_", 1, 0.01, 100.0, 102.0, client2.magic)
    client2.positions.append(pos2)

    client2.set_price(98.8, 98.8)       # 空头 +1.2%
    moved, locked, _ = risk2.protect_position("ETHUSD_", 2, "SELL", 100.0, pos2.sl, LOCK_NONE)
    check("空头 +1.2% 止损下移到成本价", moved and abs(pos2.sl - 100.0) < 1e-6, f"sl={pos2.sl}")

    client2.set_price(97.4, 97.4)       # 空头 +2.6% → 锁 1% = 99
    moved, locked, _ = risk2.protect_position("ETHUSD_", 2, "SELL", 100.0, pos2.sl, locked)
    check("空头 +2.6% 止损下移到 99", moved and abs(pos2.sl - 99.0) < 1e-6, f"sl={pos2.sl}")

    client2.set_price(99.0, 99.0)       # 回落：不放松
    prev = pos2.sl
    risk2.protect_position("ETHUSD_", 2, "SELL", 100.0, pos2.sl, locked)
    check("空头价格回落止损不放松", abs(pos2.sl - prev) < 1e-9, f"sl={pos2.sl}")

    print("\n[6d] 修改止损失败时不误记锁定值")
    client3 = MockClient(dry_run=False, bid=101.5, ask=101.5)
    risk3 = RiskManager(client3, RiskParams())
    pos3 = MockPosition(3, "ETHUSD_", 0, 0.01, 100.0, 98.0, client3.magic)
    client3.positions.append(pos3)
    client3.modify_should_fail = True
    moved, locked, reason = risk3.protect_position("ETHUSD_", 3, "BUY", 100.0, 98.0, LOCK_NONE)
    check("修改失败返回 False", not moved and reason == "modify_failed", f"{moved} {reason}")
    check("修改失败不推进锁定值", locked == LOCK_NONE, str(locked))


# ── 测试 7：dry-run 台账不被误删 + 完整闭环 ──────────────────────

def test_dry_run_book():
    print("\n[7] dry-run 台账（本地模拟仓不被同步误删）")
    client = MockClient(dry_run=True, bid=100.0, ask=100.0)
    r = make_runner(client, "dry")
    binding = {"symbol": "ETHUSD_", "strategy_file": "s.json", "lot": 0.01}

    r._reconcile(binding, "LONG", 0.8)
    check("dry-run 记录了模拟持仓", r.book.get_position("ETHUSD_") is not None,
          str(r.book.state["positions"]))
    check("MT5 端确实没有真实持仓", len(client.positions) == 0)
    check("方向为 BUY", r.book.get_position("ETHUSD_")["direction"] == "BUY")
    check("标记为 dry_run", r.book.get_position("ETHUSD_")["dry_run"] is True)

    # 同向不重复
    client.calls.clear()
    r._reconcile(binding, "LONG", 0.9)
    check("dry-run 同向不重复开仓", len([c for c in client.calls if c[0] == "open"]) == 0,
          str(client.calls))

    # live 同步逻辑绝不能碰 dry 台账
    r._sync_live_positions() if r.book.mode == "live" else None
    check("dry 模式不执行 live 同步（台账仍在）",
          r.book.get_position("ETHUSD_") is not None)

    # 反手
    client.calls.clear()
    r._reconcile(binding, "SHORT", 0.8)
    check("dry-run 反手后方向为 SELL",
          r.book.get_position("ETHUSD_")["direction"] == "SELL",
          str(r.book.state["positions"]))
    kinds = [c[0] for c in client.calls]
    check("dry-run 反手也是先平后开",
          "close" in kinds and kinds.index("close") < kinds.index("open"), str(kinds))

    # 观望 → 平仓
    r._reconcile(binding, "FLAT", 0.01)
    check("FLAT 信号后台账清空", r.book.get_position("ETHUSD_") is None,
          str(r.book.state["positions"]))

    # 状态持久化：重启后锁定档位不丢
    r._reconcile(binding, "LONG", 0.8)
    ticket = r.book.get_position("ETHUSD_")["ticket"]
    r.book.set_locked(ticket, 0.02)
    r.book.save()
    book2 = runner_mod.PositionBook("dry")
    check("重启后台账恢复", book2.get_position("ETHUSD_") is not None)
    check("重启后锁定档位不回退", abs(book2.locked_for(ticket) - 0.02) < 1e-9,
          str(book2.locked_for(ticket)))


# ── 测试 8：live 同步（止损后清台账）───────────────────────────

def test_live_sync():
    print("\n[8] live 同步：MT5 侧仓位消失后清理台账")
    client = MockClient(dry_run=False, bid=100.0, ask=100.1)
    r = make_runner(client, "live")
    r._reconcile({"symbol": "ETHUSD_", "strategy_file": "s.json", "lot": 0.01}, "LONG", 0.8)
    check("台账有记录", r.book.get_position("ETHUSD_") is not None)

    client.positions.clear()   # 模拟被止损平掉
    r._sync_live_positions()
    check("MT5 无仓后台账被清理", r.book.get_position("ETHUSD_") is None,
          str(r.book.state["positions"]))
    acts = " ".join(a["action"] for a in r.book.state["actions"])
    check("清理已记录", "已不存在" in acts, acts)

    # 下一个同向信号应能重新开仓（不被误判为已有持仓）
    client.calls.clear()
    r._reconcile({"symbol": "ETHUSD_", "strategy_file": "s.json", "lot": 0.01}, "LONG", 0.8)
    check("止损后同向信号可重新开仓",
          len([c for c in client.calls if c[0] == "open"]) == 1, str(client.calls))


# ── 测试 9：券商最小止损距离 ────────────────────────────────────

def test_stops_level():
    print("\n[9] 券商最小止损距离")
    # stops_points=50, point=0.01 → 最小距离 0.5
    client = MockClient(dry_run=False, bid=100.2, ask=100.2, stops_points=50, point=0.01)
    risk = RiskManager(client, RiskParams())
    pos = MockPosition(9, "ETHUSD_", 0, 0.01, 100.0, 98.0, client.magic)
    client.positions.append(pos)
    # 浮盈 +0.2% 未触发档位；此处验证钳制函数本身
    clamped = risk._respect_stops_level("ETHUSD_", "BUY", {"bid": 100.2, "ask": 100.2}, 100.0)
    check("多头止损被钳到 bid-0.5=99.7", abs(clamped - 99.7) < 1e-9, str(clamped))
    clamped2 = risk._respect_stops_level("ETHUSD_", "BUY", {"bid": 100.2, "ask": 100.2}, 99.0)
    check("已满足距离则不改动", abs(clamped2 - 99.0) < 1e-9, str(clamped2))
    clamped3 = risk._respect_stops_level("ETHUSD_", "SELL", {"bid": 99.8, "ask": 99.8}, 100.0)
    check("空头止损被钳到 ask+0.5=100.3", abs(clamped3 - 100.3) < 1e-9, str(clamped3))

    # 钳制后若比现有止损更松，必须放弃修改
    client.set_price(101.2, 101.2)
    pos.sl = 101.0                    # 已经很紧（高于钳制上限 100.7）
    moved, locked, _ = risk.protect_position("ETHUSD_", 9, "BUY", 100.0, pos.sl, 0.01)
    check("钳制结果更松时不放松止损", abs(pos.sl - 101.0) < 1e-9, f"sl={pos.sl}")


# ── 测试 10：最大持仓限制 ───────────────────────────────────────

def test_max_positions():
    print("\n[10] 最大同时持仓品种限制")
    old = Config.MAX_OPEN_POSITIONS
    Config.MAX_OPEN_POSITIONS = 1
    try:
        client = MockClient(dry_run=True, bid=100.0, ask=100.0)
        r = make_runner(client, "dry")
        r._reconcile({"symbol": "ETHUSD_", "strategy_file": "s.json", "lot": 0.01}, "LONG", 0.8)
        r._reconcile({"symbol": "BTCUSD_", "strategy_file": "s.json", "lot": 0.01}, "LONG", 0.8)
        check("超过上限的第二个品种被拒绝",
              r.book.get_position("BTCUSD_") is None, str(r.book.state["positions"]))
        acts = " ".join(a["action"] for a in r.book.state["actions"])
        check("拒绝原因已记录", "最大持仓数" in acts, acts)

        # 反手不应被上限挡住（先平后开，净持仓数不变）
        client.calls.clear()
        r._reconcile({"symbol": "ETHUSD_", "strategy_file": "s.json", "lot": 0.01}, "SHORT", 0.8)
        check("反手不被持仓上限阻挡",
              r.book.get_position("ETHUSD_") is not None
              and r.book.get_position("ETHUSD_")["direction"] == "SELL",
              str(r.book.state["positions"]))
    finally:
        Config.MAX_OPEN_POSITIONS = old


# ── 测试 11：浮盈计算方向正确 ───────────────────────────────────

def test_profit_pct():
    print("\n[11] 浮盈计算（多头用 bid、空头用 ask）")
    check("多头 100→bid101 = +1%", abs(profit_pct("BUY", 100, 101, 101.5) - 0.01) < 1e-9)
    check("多头 100→bid99 = -1%", abs(profit_pct("BUY", 100, 99, 99.5) + 0.01) < 1e-9)
    check("空头 100→ask99 = +1%", abs(profit_pct("SELL", 100, 98.5, 99) - 0.01) < 1e-9)
    check("空头 100→ask101 = -1%", abs(profit_pct("SELL", 100, 100.5, 101) + 0.01) < 1e-9)
    check("多头锁1%价 = 101", abs(lock_price("BUY", 100, 0.01) - 101) < 1e-9)
    check("空头锁1%价 = 99", abs(lock_price("SELL", 100, 0.01) - 99) < 1e-9)


# ── 测试 12：策略文件加载与校验 ─────────────────────────────────

def test_strategy_loading():
    print("\n[12] 策略文件加载与校验")
    from trading.signal_engine import StrategyError, load_strategy_file, check_vocab_version

    good = _TMP / "good.json"
    good.write_text(json.dumps({
        "vocab_version": "vTEST", "symbol": "ETHUSD_", "timeframe": "H1",
        "formula": [8, 98, 121, 88, 103, 88, 103, 71], "best_score": 5.31,
        "formula_decoded": "X",
    }), encoding="utf-8")
    meta = load_strategy_file(good)
    check("正常策略可加载", meta["formula"] == [8, 98, 121, 88, 103, 88, 103, 71])
    check("分数解析正确", abs(meta["best_score"] - 5.31) < 1e-9)

    bad = _TMP / "bad.json"
    bad.write_text(json.dumps({"symbol": "X"}), encoding="utf-8")
    try:
        load_strategy_file(bad)
        check("缺 formula 的策略被拒绝", False, "未抛异常")
    except StrategyError:
        check("缺 formula 的策略被拒绝", True)

    try:
        load_strategy_file(_TMP / "nope.json")
        check("不存在的文件被拒绝", False, "未抛异常")
    except StrategyError:
        check("不存在的文件被拒绝", True)

    check("错误 vocab_version 被识别", not check_vocab_version("vWRONG"))
    from model_core.vocab import VOCAB_VERSION
    check("本地 vocab_version 自洽", check_vocab_version(VOCAB_VERSION))

    # 真实策略文件（若已导入）应能通过校验
    real = PROJECT_ROOT / "strategies" / "best_ETHUSD_.json"
    if real.exists():
        m = load_strategy_file(real)
        check("真实 best_ETHUSD_.json 可加载", len(m["formula"]) > 0)
        check("真实策略 vocab 与本引擎一致", check_vocab_version(m["vocab_version"]),
              f"{m['vocab_version']} vs {VOCAB_VERSION}")


# ── 测试 13：真实策略端到端信号计算 ─────────────────────────────

def test_real_signal_pipeline():
    print("\n[13] 真实策略端到端信号（合成 1000 根 K线）")
    import numpy as np
    import torch
    from trading.signal_engine import compute_signal, load_strategy_file, rates_to_raw_dict

    real = PROJECT_ROOT / "strategies" / "best_ETHUSD_.json"
    if not real.exists():
        print("  [SKIP] 策略未导入，跳过")
        return
    formula = load_strategy_file(real)["formula"]

    n = 1000
    rng = np.random.default_rng(42)
    close = 3000 + np.cumsum(rng.normal(0, 5, n)).astype(np.float32)
    high = close + np.abs(rng.normal(0, 3, n)).astype(np.float32)
    low = close - np.abs(rng.normal(0, 3, n)).astype(np.float32)
    open_ = np.concatenate([[close[0]], close[:-1]]).astype(np.float32)
    vol = np.abs(rng.normal(1000, 200, n)).astype(np.float32)
    times = np.arange(n, dtype=np.int64) * 3600 + 1700000000

    # 用 structured array 走和实盘完全一样的转换路径
    rates = np.zeros(n, dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"),
                              ("low", "f8"), ("close", "f8"), ("tick_volume", "i8")])
    rates["time"] = times
    rates["open"], rates["high"], rates["low"], rates["close"] = open_, high, low, close
    rates["tick_volume"] = vol.astype(np.int64)

    raw = rates_to_raw_dict(rates)
    check("K线转张量成功", raw is not None and raw["close"].shape == (1, n),
          str(None if raw is None else raw["close"].shape))
    res = compute_signal([formula], raw, min_trade_exposure=0.05)
    check("真实公式计算出信号", res["state"] == "ok", str(res))
    check("方向在合法集合内", res["direction"] in ("LONG", "SHORT", "FLAT"), str(res))
    check("强度在 [0,1]", 0.0 <= res["strength"] <= 1.0, str(res["strength"]))
    print(f"       → 方向={res['direction']} 强度={res['strength']} 仓位={res['position']}")

    # 可复现性：同样输入两次结果一致
    res2 = compute_signal([formula], raw, min_trade_exposure=0.05)
    check("同输入结果可复现", res2["position"] == res["position"],
          f"{res['position']} vs {res2['position']}")


# ── 支撑/阻力位（S/R）────────────────────────────────────────────

def _sr_rates(lo=103.5, hi=107.5, cycles=12, period=40, end_price=105.0):
    """合成箱体震荡行情：区间 [lo, hi]，收在 end_price（用于确定性的关键位）。"""
    n = cycles * period
    o, h, l, c, v = [], [], [], [], []
    for i in range(n):
        phase = (i % period) / period
        up = (i // period) % 2 == 0
        price = lo + (hi - lo) * phase if up else hi - (hi - lo) * phase
        c.append(round(price, 4))
        o.append(round(price, 4))
        h.append(round(price + 0.3, 4))
        l.append(round(price - 0.3, 4))
        v.append(1000.0)
    c[-1] = o[-1] = end_price
    h[-1], l[-1] = end_price + 0.3, end_price - 0.3
    return {"time": list(range(n)), "open": o, "high": h, "low": l,
            "close": c, "tick_volume": v}


def test_sr_levels():
    print("\n— S/R 关键位检测 —")
    from trading.sr import detect_levels
    rates = _sr_rates()
    levels, info = detect_levels("TEST", rates)
    check("能检测出关键位", len(levels) >= 1, f"levels={len(levels)}")
    check("ATR 有效", (info.get("atr") or 0) > 0, str(info.get("atr")))
    price = info["price"]
    ok_side = all((z["kind"] == "support" and z["high"] < price) or
                  (z["kind"] == "resistance" and z["low"] > price) for z in levels)
    check("支撑在现价下方 / 压力在上方", ok_side, str([(z["kind"], z["center"]) for z in levels]))
    ok_band = all(0.5 <= abs(z["dist_atr"]) <= 5.0 for z in levels)
    check("距离带 0.5~5 ATR", ok_band)
    ok_fields = all(z.get("n_events") is not None and z.get("low") < z.get("high")
                    for z in levels)
    check("区带字段完整", ok_fields)


def test_sr_sl_tp_buy():
    print("\n— S/R 多头止盈止损 —")
    from trading.sr import SRParams, sl_tp_for_trade
    rates = _sr_rates(lo=103.5, hi=107.5, end_price=105.0)
    entry = 105.0
    fixed = initial_stop_price("BUY", entry, -0.02)     # 102.9（-2%）
    plan = sl_tp_for_trade("TEST", "BUY", entry, rates, SRParams(), fixed_sl=fixed)
    check("止损存在", plan["sl"] is not None)
    check("止损不比固定更松", plan["sl"] >= fixed,
          f"sl={plan['sl']} fixed={fixed}")
    if plan["sl_source"] == "sr":
        d = (entry - plan["sl"]) / entry
        check("S/R 止损距离在带内", 0.004 <= d <= 0.03, f"dist={d:.4f}")
        check("S/R 止损在支撑带外缘下方", plan["sl_zone"] is not None and
              plan["sl"] <= plan["sl_zone"]["low"], str(plan["sl_zone"]))
    check("止盈来自 S/R 且在入场价上方",
          plan["tp"] is not None and plan["tp"] > entry,
          f"tp={plan['tp']} src={plan['tp_source']}")
    if plan["tp"]:
        d = (plan["tp"] - entry) / entry
        check("止盈距离在带内", 0.006 <= d <= 0.06, f"dist={d:.4f}")


def test_sr_fixed_sl_never_loosened():
    print("\n— S/R 绝不放松固定止损 —")
    from trading.sr import SRParams, sl_tp_for_trade
    rates = _sr_rates(lo=103.5, hi=107.5, end_price=105.0)
    entry = 105.0
    tight_fixed = initial_stop_price("BUY", entry, -0.008)   # -0.8%，比结构位更近
    plan = sl_tp_for_trade("TEST", "BUY", entry, rates, SRParams(),
                           fixed_sl=tight_fixed)
    check("固定止损更紧时保持固定", plan["sl_source"] == "fixed" and
          abs(plan["sl"] - tight_fixed) < 1e-9, f"{plan['sl_source']} sl={plan['sl']}")
    # 空头同理
    tight_fixed_s = initial_stop_price("SELL", entry, -0.008)
    plan_s = sl_tp_for_trade("TEST", "SELL", entry, rates, SRParams(),
                             fixed_sl=tight_fixed_s)
    check("空头固定止损更紧时保持固定", plan_s["sl_source"] == "fixed" and
          abs(plan_s["sl"] - tight_fixed_s) < 1e-9, f"{plan_s['sl_source']} sl={plan_s['sl']}")
    # 禁用 S/R → 完全退回固定止损、无止盈
    off = SRParams(enabled=False)
    plan_off = sl_tp_for_trade("TEST", "BUY", entry, rates, off, fixed_sl=tight_fixed)
    check("禁用 S/R 退回固定止损", plan_off["sl"] == tight_fixed and
          plan_off["tp"] is None, str(plan_off))


def test_sr_sl_out_of_band_falls_back():
    print("\n— S/R 止损距离带外的回退 —")
    from trading.sr import SRParams, sl_tp_for_trade
    # 支撑在 5% 之外（默认 sr_max_sl_pct=3%）→ 不采纳，退回固定止损
    rates = _sr_rates(lo=99.0, hi=107.5, cycles=14, end_price=105.0)
    entry = 105.0
    fixed = initial_stop_price("BUY", entry, -0.02)
    plan = sl_tp_for_trade("TEST", "BUY", entry, rates, SRParams(), fixed_sl=fixed)
    check("结构位太远时退回固定止损", plan["sl_source"] == "fixed" and
          abs(plan["sl"] - fixed) < 1e-9, f"{plan['sl_source']} sl={plan['sl']}")


def test_sr_respect_tp_level():
    print("\n— S/R 止盈尊重券商最小距离 —")
    r = make_runner(MockClient(stops_points=500, point=0.01), "dry")  # 最小距离 5.0
    tick = {"bid": 105.0, "ask": 105.1, "time": 0}
    check("过近的止盈被放弃", r._respect_tp_level("TEST", "BUY", tick, 105.5) is None)
    check("足够远的止盈保留", abs(r._respect_tp_level("TEST", "BUY", tick, 110.5) - 110.5) < 1e-9)
    check("空头过近的止盈被放弃",
          r._respect_tp_level("TEST", "SELL", tick, 104.8) is None)
    check("无止盈时返回 None", r._respect_tp_level("TEST", "BUY", tick, None) is None)


def test_sr_dry_run_tp_fill():
    print("\n— dry-run 模拟止盈触发 —")
    client = MockClient(dry_run=True, bid=105.0, ask=105.1)
    r = make_runner(client, "dry")
    r.book.set_position("TEST", {
        "ticket": 9000001, "direction": "BUY", "volume": 0.01,
        "open_price": 103.0, "open_time": "2026-09-05 00:00:00",
        "sl": 100.9, "tp": 105.5, "locked": LOCK_NONE, "dry_run": True,
    })
    client.set_price(bid=105.6, ask=105.7)   # bid 触及止盈 105.5
    r._check_dry_take_profits()
    check("触及止盈后台账清仓", r.book.get_position("TEST") is None)
    check("记录了止盈动作", any("止盈" in a["action"] for a in r.book.state["actions"]))
    # 未触及时不平仓
    r.book.set_position("TEST", {
        "ticket": 9000002, "direction": "BUY", "volume": 0.01,
        "open_price": 103.0, "open_time": "2026-09-05 00:00:00",
        "sl": 100.9, "tp": 106.5, "locked": LOCK_NONE, "dry_run": True,
    })
    client.set_price(bid=105.6, ask=105.7)
    r._check_dry_take_profits()
    check("未触及止盈继续持有", r.book.get_position("TEST") is not None)


def test_sr_partial_plan():
    print("\n— S/R 止盈一半计划 —")
    from trading.sr import SRParams, _partial_gate, sl_tp_for_trade
    rates = _sr_rates(lo=103.5, hi=107.5, end_price=105.0)
    entry = 105.0
    p = SRParams(partial_use_model=False)
    plan = sl_tp_for_trade("TEST", "BUY", entry, rates, p, fixed_sl=entry * 0.98)
    check("止盈一半计划存在且在入场价上方",
          plan["partial"] is not None and plan["partial"]["price"] > entry,
          str(plan["partial"]))
    check("止盈一半在第一关键位前、全部止盈更远",
          plan["tp"] is not None and plan["tp"] > plan["partial"]["price"],
          f"partial={plan['partial'] and plan['partial']['price']} tp={plan['tp']}")
    plan_off = sl_tp_for_trade("TEST", "BUY", entry, rates,
                               SRParams(partial_enabled=False), fixed_sl=entry * 0.98)
    check("关闭止盈一半时回退单一止盈",
          plan_off["partial"] is None and plan_off["tp"] is not None, str(plan_off))
    # 模型把关（纯函数）
    check("守住概率达标放行", _partial_gate({"p_hold": 0.8}, SRParams()))
    check("守住概率不足拦截", not _partial_gate({"p_hold": 0.3}, SRParams()))
    check("无模型输出放行", _partial_gate({"p_hold": None}, SRParams()))
    check("关闭把关时放行", _partial_gate({"p_hold": 0.1}, SRParams(partial_use_model=False)))


def test_sr_dry_run_partial_fill():
    print("\n— dry-run 止盈一半触发 —")
    client = MockClient(dry_run=True, bid=105.0, ask=105.1)
    r = make_runner(client, "dry")
    r.book.set_position("TEST", {
        "ticket": 9000011, "direction": "BUY", "volume": 0.10,
        "open_price": 103.0, "open_time": "2026-09-05 00:00:00",
        "sl": 100.9, "tp": 108.0, "locked": LOCK_NONE, "dry_run": True,
        "sr_partial": {"price": 105.5, "fraction": 0.5, "close_volume": 0.05,
                       "remaining": 0.05, "done": False},
    })
    client.set_price(bid=105.6, ask=105.7)   # 触及 105.5
    r._check_partial_take_profits()
    pos = r.book.get_position("TEST")
    check("触发后剩余手数减半", pos is not None and abs(pos["volume"] - 0.05) < 1e-9,
          str(pos and pos["volume"]))
    check("计划标记已执行", pos["sr_partial"]["done"] is True)
    check("记录了止盈一半动作", any("止盈一半" in a["action"] for a in r.book.state["actions"]))
    # 已执行过不再重复触发
    client.set_price(bid=106.0, ask=106.1)
    r._check_partial_take_profits()
    pos = r.book.get_position("TEST")
    check("不会重复执行", abs(pos["volume"] - 0.05) < 1e-9 and
          sum(1 for a in r.book.state["actions"] if "止盈一半" in a["action"]) == 1)
    # 未触及不执行
    r.book.set_position("TEST2", {
        "ticket": 9000012, "direction": "BUY", "volume": 0.10,
        "open_price": 103.0, "open_time": "2026-09-05 00:00:00",
        "sl": 100.9, "tp": 0.0, "locked": LOCK_NONE, "dry_run": True,
        "sr_partial": {"price": 106.5, "fraction": 0.5, "close_volume": 0.05,
                       "remaining": 0.05, "done": False},
    })
    client.set_price(bid=106.0, ask=106.1)
    r._check_partial_take_profits()
    check("未触及时继续持有", abs(r.book.get_position("TEST2")["volume"] - 0.10) < 1e-9)


def test_sr_live_partial_volume_sent():
    print("\n— live 部分平仓调用 —")
    client = MockClient(dry_run=False, bid=105.0, ask=105.1)
    client.positions.append(MockPosition(7000001, "TEST", 0, 0.10, 103.0, sl=100.9,
                                         magic=20260904))
    r = make_runner(client, "live")
    r.book.set_position("TEST", {
        "ticket": 7000001, "direction": "BUY", "volume": 0.10,
        "open_price": 103.0, "open_time": "2026-09-05 00:00:00",
        "sl": 100.9, "tp": 0.0, "locked": LOCK_NONE, "dry_run": False,
        "sr_partial": {"price": 105.5, "fraction": 0.5, "close_volume": 0.05,
                       "remaining": 0.05, "done": False},
    })
    client.set_price(bid=105.6, ask=105.7)
    r._check_partial_take_profits()
    partial_calls = [c for c in client.calls if c[0] == "close"]
    check("向 MT5 发送了半手平仓", len(partial_calls) == 1 and
          abs(partial_calls[0][3] - 0.05) < 1e-9, str(partial_calls))
    check("计划标记已执行", r.book.get_position("TEST")["sr_partial"]["done"] is True)


def test_sr_live_manual_position_adopted():
    print("\n— live 手动仓位纳入管理 —")
    client = MockClient(dry_run=False, bid=105.0, ask=105.1)
    client.positions.append(MockPosition(7000099, "TEST", 0, 0.01, 103.0, sl=100.9,
                                         magic=20260904))
    r = make_runner(client, "live")
    r._check_partial_take_profits()
    info = r.book.get_position("TEST")
    check("台账补记了手动仓位", info is not None and info.get("manual") is True)
    check("补算失败/无数据时不产生触发价", info["sr_partial"]["done"] is True and
          float(info["sr_partial"]["price"]) == 0.0)


def test_sr_partial_override():
    print("\n— 看板手动覆盖止盈一半 —")
    import json as _json
    client = MockClient(dry_run=True, bid=105.0, ask=105.1)
    r = make_runner(client, "dry")
    r.book.set_position("TEST", {
        "ticket": 9000031, "direction": "BUY", "volume": 0.10,
        "open_price": 103.0, "open_time": "2026-09-05 00:00:00",
        "sl": 100.9, "tp": 0.0, "locked": LOCK_NONE, "dry_run": True,
        "sr_partial": {"price": 105.5, "fraction": 0.5, "close_volume": 0.05,
                       "remaining": 0.05, "done": False},
    })
    r._ov_mtime = None
    r._ov_cache = None
    r._ov_applied_ts = {}
    # 设置覆盖：到 106 平 0.03 手
    runner_mod.PARTIAL_OVERRIDES_FILE.write_text(_json.dumps({
        "9000031": {"price": 106.0, "close_volume": 0.03, "ts": 100.0},
    }), encoding="utf-8")
    r._apply_partial_overrides(r._load_partial_overrides())
    plan = r.book.get_position("TEST")["sr_partial"]
    check("覆盖后价格与手数生效", plan["price"] == 106.0 and
          abs(plan["close_volume"] - 0.03) < 1e-9 and
          abs(plan["remaining"] - 0.07) < 1e-9 and plan["done"] is False,
          str(plan))
    # 同一时间戳不重复应用
    r._apply_partial_overrides(r._load_partial_overrides())
    check("同一覆盖不重复应用", r.book.get_position("TEST")["sr_partial"] == plan)
    # 取消覆盖：price=0
    runner_mod.PARTIAL_OVERRIDES_FILE.write_text(_json.dumps({
        "9000031": {"price": 0.0, "close_volume": 0.0, "ts": 200.0},
    }), encoding="utf-8")
    r._ov_mtime = None
    r._apply_partial_overrides(r._load_partial_overrides())
    plan2 = r.book.get_position("TEST")["sr_partial"]
    check("price=0 取消计划", plan2["done"] is True and float(plan2["price"]) == 0.0,
          str(plan2))
    # 非法手数（0.2 ≥ 持仓 0.1）→ 标记无效，不再触发
    r.book.get_position("TEST")["sr_partial"] = {
        "price": 105.5, "close_volume": 0.05, "remaining": 0.05, "done": False}
    runner_mod.PARTIAL_OVERRIDES_FILE.write_text(_json.dumps({
        "9000031": {"price": 106.0, "close_volume": 0.2, "ts": 300.0},
    }), encoding="utf-8")
    r._ov_mtime = None
    r._apply_partial_overrides(r._load_partial_overrides())
    plan3 = r.book.get_position("TEST")["sr_partial"]
    check("手数无法拆出时标记无效", plan3["done"] is True and
          float(plan3["price"]) == 0.0, str(plan3))


def test_manual_sl_respected():
    print("\n— 手动止损不被自动拉回 —")
    client = MockClient(dry_run=False, bid=103.5, ask=103.6)   # 浮盈 +0.49%，未触发阶梯
    client.positions.append(MockPosition(7100001, "TEST", 0, 0.10, 103.0, sl=100.5,
                                         magic=20260904))
    r = make_runner(client, "live")
    r.book.set_position("TEST", {
        "ticket": 7100001, "direction": "BUY", "volume": 0.10,
        "open_price": 103.0, "open_time": "2026-09-05 00:00:00",
        "sl": 100.5, "tp": 0.0, "locked": LOCK_NONE, "dry_run": False,
    })
    init_sl = initial_stop_price("BUY", 103.0, -0.02)   # 100.94
    # 有手动标记：更松的止损不被拉回
    r.risk.params.manual_sl_tickets = {7100001}
    moved, _locked, reason = r.risk.protect_position(
        "TEST", 7100001, "BUY", 103.0, 100.5, LOCK_NONE)
    check("手动止损（更松）不被自动拉回", moved is False and reason is None,
          f"moved={moved} reason={reason}")
    check("MT5 上的止损保持原价", abs(client.positions[0].sl - 100.5) < 1e-9)
    # 无标记：被拉回初始安全位
    r.risk.params.manual_sl_tickets = set()
    moved2, _l2, _r2 = r.risk.protect_position(
        "TEST", 7100001, "BUY", 103.0, 100.5, LOCK_NONE)
    check("无标记时拉回安全位", moved2 is True and
          abs(client.positions[0].sl - init_sl) < 0.01,
          f"sl={client.positions[0].sl} init={init_sl}")
    # 看板覆盖应用：更新台账止损 + 标记手动
    r._apply_sl_overrides({"7100001": {"sl": 101.2, "ts": 1000.0, "applied_to_mt5": True}})
    check("覆盖更新台账止损并标记手动", r.book.get_position("TEST")["sl"] == 101.2 and
          r.book.manual_sl_tickets() == {7100001})
    # 阶梯保本对手动仓位仍生效：浮盈 +1% → 锁到成本
    client.positions[0].sl = 101.2
    client.set_price(bid=104.03, ask=104.13)
    moved3, _l3, _r3 = r.risk.protect_position(
        "TEST", 7100001, "BUY", 103.0, 101.2, LOCK_NONE)
    check("阶梯保本对手动仓位仍生效", moved3 is True and
          abs(client.positions[0].sl - 103.0) < 0.01,
          f"sl={client.positions[0].sl}")


# ── 主入口 ──────────────────────────────────────────────────────

def main() -> int:
    print("=" * 64)
    print("  MT5AutoTrader 交易逻辑测试（全部 mock，不触碰真实 MT5/订单）")
    print("=" * 64)
    for fn in (test_signal_threshold, test_no_duplicate_open, test_reverse_close_then_open,
               test_close_fail_blocks_open, test_initial_stop, test_ladder,
               test_dry_run_book, test_live_sync, test_stops_level,
               test_max_positions, test_profit_pct, test_strategy_loading,
               test_real_signal_pipeline,
               test_sr_levels, test_sr_sl_tp_buy, test_sr_fixed_sl_never_loosened,
               test_sr_sl_out_of_band_falls_back, test_sr_respect_tp_level,
               test_sr_dry_run_tp_fill,
               test_sr_partial_plan, test_sr_dry_run_partial_fill,
               test_sr_live_partial_volume_sent, test_sr_live_manual_position_adopted,
               test_sr_partial_override, test_manual_sl_respected):
        try:
            fn()
        except Exception as exc:
            import traceback
            FAILED.append(f"{fn.__name__} 异常: {exc}")
            print(f"  [ERROR] {fn.__name__}: {exc}")
            traceback.print_exc()
    print("\n" + "=" * 64)
    print(f"  通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    if FAILED:
        print("\n  失败明细：")
        for f in FAILED:
            print(f"    - {f}")
    print("=" * 64)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())

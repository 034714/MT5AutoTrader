"""
trading/risk.py — 实时风控（初始止损 + 阶梯保本止损）

规则（与需求 1.txt 第 10 节一致）：
  - 开仓即设初始止损（默认 -2%）
  - 实时监控浮盈（多头用 bid、空头用 ask 相对开仓价）
  - 阶梯保本：浮盈 1% → 止损移到成本价；2% → 锁 1%；3% → 锁 2%；4% → 锁 3%
    （档位可在 trader_config.json / 看板中修改）
  - 止损只收紧、绝不放松；多空两向都支持
  - 每个 ticket 记录已锁定的最高利润，重启后不回退
  - 尊重券商最小止损距离（trade_stops_level），钳制后仍不得放松
  - dry-run 时只记日志并更新本地模拟状态
"""
from __future__ import annotations

from dataclasses import dataclass, field
from loguru import logger

from trading.mt5_client import MT5Client

# 「尚未锁定任何利润」的哨兵值（低于任何档位，含 0% 保本档）
LOCK_NONE = -1.0


@dataclass
class RiskParams:
    enable_price_monitor: bool = True
    price_monitor_interval: int = 10          # 秒（5~10）
    stop_loss_pct: float = -0.02              # 初始止损（负数）
    breakeven_levels: list[tuple[float, float]] = field(
        default_factory=lambda: [(0.01, 0.00), (0.02, 0.01), (0.03, 0.02), (0.04, 0.03)]
    )
    # 用户在网页/图表上手动设置过止损的 ticket：不再自动拉回初始止损（安全网豁免）
    manual_sl_tickets: set[int] = field(default_factory=set)

    @classmethod
    def from_config(cls) -> "RiskParams":
        from config import Config
        return cls(
            enable_price_monitor=bool(Config.ENABLE_PRICE_MONITOR),
            price_monitor_interval=int(Config.PRICE_MONITOR_INTERVAL),
            stop_loss_pct=float(Config.STOP_LOSS_PCT),
            breakeven_levels=[(float(t), float(l)) for t, l in Config.BREAKEVEN_LEVELS],
        )


def initial_stop_price(direction: str, open_price: float, stop_loss_pct: float) -> float:
    """初始止损价。多头：open*(1+pct)（pct=-0.02 → open*0.98）；空头：open*(1-pct)。"""
    if direction == "BUY":
        return open_price * (1.0 + stop_loss_pct)
    return open_price * (1.0 - stop_loss_pct)


def profit_pct(direction: str, open_price: float, bid: float, ask: float) -> float:
    """当前浮盈比例。多头按 bid（可卖出价）、空头按 ask（可买回价）。"""
    if open_price <= 0:
        return 0.0
    if direction == "BUY":
        return (bid - open_price) / open_price
    return (open_price - ask) / open_price


def ladder_lock(
    profit: float,
    levels: list[tuple[float, float]],
    already_locked: float = LOCK_NONE,
) -> tuple[float | None, int]:
    """根据当前浮盈返回应锁定的利润档位。

    Args:
        profit: 当前浮盈比例（如 0.023 = 2.3%）。
        levels: [(触发浮盈, 锁定利润), ...]，任意顺序。
        already_locked: 该仓位已锁定的最高利润（LOCK_NONE = 仅初始止损）。

    Returns:
        (目标锁定利润 or None, 命中的最高档位索引)。
        目标为 None 表示没有可新触发的档位。
    """
    ordered = sorted(levels, key=lambda x: float(x[0]))
    best_lock: float | None = None
    best_idx = -1
    for i, (trigger, lock) in enumerate(ordered):
        if profit >= float(trigger) and (
            best_lock is None or float(lock) > best_lock
        ):
            best_lock = float(lock)
            best_idx = i
    if best_lock is None or best_lock <= already_locked:
        return None, best_idx
    return best_lock, best_idx


def lock_price(direction: str, open_price: float, lock_pct: float) -> float:
    """锁定 lock_pct 利润对应的止损价。"""
    if direction == "BUY":
        return open_price * (1.0 + lock_pct)
    return open_price * (1.0 - lock_pct)


def implied_lock_pct(direction: str, open_price: float, sl_price: float) -> float:
    """由止损价反推实际锁定的利润比例（用于如实记录状态）。"""
    if open_price <= 0 or sl_price <= 0:
        return LOCK_NONE
    pct = (sl_price - open_price) / open_price
    return pct if direction == "BUY" else -pct


class RiskManager:
    """对单一持仓执行初始止损 + 阶梯保本。由 runner 的监控循环驱动。"""

    def __init__(self, client: MT5Client, params: RiskParams | None = None) -> None:
        self.client = client
        self.params = params or RiskParams.from_config()

    def refresh_params(self) -> None:
        """每个监控 tick 前调用：从 trader_config.json 热更新参数。"""
        from config import Config
        Config.reload()
        self.params = RiskParams.from_config()

    # ── 主入口 ───────────────────────────────────────────────────

    def protect_position(
        self,
        symbol: str,
        ticket: int,
        direction: str,
        open_price: float,
        current_sl: float,
        already_locked: float,
    ) -> tuple[bool, float, str | None]:
        """对一个持仓执行保护（初始止损兜底 + 阶梯保本）。

        Args:
            symbol/ticket/direction: 持仓标识，direction 'BUY'/'SELL'。
            open_price: 开仓价。
            current_sl: 当前止损价（0 = 无止损）。
            already_locked: 已锁定利润（LOCK_NONE = 尚未锁利）。

        Returns:
            (是否执行了修改, 新的已锁定利润, 失败原因 or None)
        """
        if not self.params.enable_price_monitor:
            return False, already_locked, None

        tick = self.client.get_tick(symbol)
        if tick is None:
            return False, already_locked, "no_tick"

        profit = profit_pct(direction, open_price, tick["bid"], tick["ask"])

        # ── 1. 计算目标止损价（取「最紧」的需求）────────────────
        desired: float | None = None

        # 1a. 无止损或止损比初始止损更松 → 补/移到初始止损（安全网）。
        #     用户手动设置过止损的仓位不受此约束（拖远也尊重），阶梯保本仍生效。
        if ticket not in self.params.manual_sl_tickets:
            init_sl = initial_stop_price(direction, open_price, self.params.stop_loss_pct)
            if current_sl <= 0 or not self._tighter_or_equal(direction, current_sl, init_sl):
                desired = init_sl

        # 1b. 阶梯保本：达到触发档位且高于已锁定档位
        lock, _idx = ladder_lock(profit, self.params.breakeven_levels, already_locked)
        if lock is not None:
            lp = lock_price(direction, open_price, lock)
            if desired is None or self._tighter_or_equal(direction, lp, desired):
                desired = lp

        if desired is None:
            return False, already_locked, None

        # ── 2. 只收紧不放松 ─────────────────────────────────
        if current_sl > 0:
            if self._tighter_or_equal(direction, current_sl, desired) and \
                    not self._same_price(current_sl, desired):
                return False, already_locked, None  # 当前更紧 → 不动
            if self._same_price(current_sl, desired):
                # 止损已到位（如重启后状态丢失），如实补记锁定值
                return False, implied_lock_pct(direction, open_price, current_sl), None

        # ── 3. 券商最小止损距离约束 ─────────────────────────
        final_sl = self._respect_stops_level(symbol, direction, tick, desired)
        if final_sl is None:
            return False, already_locked, "stops_level"
        # 钳制后仍不得比当前止损更松
        if current_sl > 0 and self._tighter_or_equal(direction, current_sl, final_sl) \
                and not self._same_price(current_sl, final_sl):
            return False, already_locked, None

        if self._same_price(current_sl, final_sl):
            return False, implied_lock_pct(direction, open_price, current_sl), None

        ok = self.client.modify_sl(symbol, ticket, final_sl)
        if ok:
            achieved = implied_lock_pct(direction, open_price, final_sl)
            logger.info(
                f"[风控] {symbol} ticket={ticket} {direction} 浮盈={profit*100:.2f}% "
                f"止损移动 → {final_sl:.5f}（约锁定 {achieved*100:.2f}%）"
            )
            return True, max(already_locked, achieved), None
        return False, already_locked, "modify_failed"

    # ── 工具 ─────────────────────────────────────────────────────

    @staticmethod
    def _tighter_or_equal(direction: str, sl_a: float, sl_b: float) -> bool:
        """sl_a 是否至少和 sl_b 一样紧（多头更高更紧，空头更低更紧）。"""
        if direction == "BUY":
            return sl_a >= sl_b
        return sl_a <= sl_b

    @staticmethod
    def _same_price(a: float, b: float, rel_eps: float = 1e-6) -> bool:
        if a <= 0 or b <= 0:
            return a == b
        return abs(a - b) <= abs(b) * rel_eps

    def _respect_stops_level(
        self, symbol: str, direction: str, tick: dict, desired: float
    ) -> float | None:
        """确保证损价离市价足够远；过近时向外退到最小允许距离。

        返回修正后的止损价；无法满足时返回 None。
        注意：调用方会再校验钳制结果不比当前止损更松。
        """
        client = self.client
        stops_points = client.stops_level_points(symbol)
        point = client.point(symbol)
        if stops_points <= 0 or point <= 0:
            return desired
        min_dist = stops_points * point
        if direction == "BUY":
            max_allowed = tick["bid"] - min_dist   # 多头止损必须 ≤ bid - min_dist
            return desired if desired <= max_allowed else max_allowed
        min_allowed = tick["ask"] + min_dist       # 空头止损必须 ≥ ask + min_dist
        return desired if desired >= min_allowed else min_allowed

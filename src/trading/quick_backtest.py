"""Read-only, fixed-lot backtest on MT5's latest CLOSED bars.

This module never initializes/shuts down MT5, selects symbols, sends orders,
starts a runner, or writes files. The caller supplies an already connected API.
The offline CLI is deliberately not involved.
"""
from __future__ import annotations

import json
import math
import re
import time
from decimal import Decimal, ROUND_FLOOR
from pathlib import Path

MIN_BARS = 500
MAX_BARS = 50_000
DEFAULT_BARS = 2000
MAX_FORMULA_TOKENS = 128
MAX_SECONDS = 180
TIMEFRAMES = {"M1": 60, "M5": 300, "M15": 900, "M30": 1800,
              "H1": 3600, "H4": 14400, "D1": 86400}
LIMITATIONS = [
    "固定手数、单仓位信号回测；不模拟实盘 SL/TP、阶梯止损、支撑阻力、部分止盈、风控、保证金或再入场冷却。",
    "使用已收盘 K 线，信号下一根开盘成交；最后一根收盘强制平仓，未使用当前未收盘 K 线。",
    "MT5 K 线按 Bid 价格建模；历史 bar spread 是点差近似，不是开盘/收盘真实逐笔 Bid/Ask。",
    "滑点与每手单边佣金是用户假设，不是券商实收费用；默认均为 0，隔夜费也未计入。",
    "order_calc_profit 使用终端当前合约规则及当前汇率换算账户货币，并非历史汇率盈亏。",
    "最大回撤基于逐 bar 收盘可平仓净权益（含浮盈浮亏和预计平仓成本），不含 bar 内极值或账户其他仓位。",
    "不模拟流动性、成交失败或资金限制；结果不是实际成交记录，也不保证未来收益。",
    "单策略单品种特征，预热历史有限；与训练组合及实盘滚动窗口结果可能不同。",
    "快速回测始终使用新版因果特征与归一化；旧策略部署时保留旧语义，结果可能与其运行信号不同。",
]


class QuickBacktestError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


def number(value, name, low, high):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QuickBacktestError(f"{name} 必须是数字")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise QuickBacktestError(f"{name} 必须在 {low}..{high} 范围内")
    return value


def _symbol(value, name="symbol", *, required=True):
    if value is None and not required:
        return None
    if not isinstance(value, str):
        raise QuickBacktestError(f"{name} 必须是文本")
    value = value.strip()
    if not value or len(value) > 128 or re.search(r"[\x00-\x1f]", value):
        raise QuickBacktestError(f"{name} 无效")
    return value


def _effective_meta(meta, symbol=None):
    """Keep the strategy intact while using an explicit terminal symbol for one simulation."""
    strategy_symbol = meta["symbol"]
    effective = _symbol(symbol, "symbol", required=False) or strategy_symbol
    return dict(meta, symbol=effective, strategy_symbol=strategy_symbol)


def validate_request(payload):
    if not isinstance(payload, dict):
        raise QuickBacktestError("请求必须是 JSON 对象")
    unknown = set(payload) - {"strategy_file", "symbol", "bars", "lot", "slippage_points", "commission_per_lot_side"}
    if unknown:
        raise QuickBacktestError("未知参数: " + ", ".join(sorted(unknown)))
    _symbol(payload.get("symbol"), "symbol", required=False)
    bars = payload.get("bars")
    if isinstance(bars, bool) or not isinstance(bars, int) or not MIN_BARS <= bars <= MAX_BARS:
        raise QuickBacktestError(f"bars 必须是 {MIN_BARS}..{MAX_BARS} 的整数")
    lot = number(payload.get("lot"), "lot", 1e-8, 1e6)
    slip = number(payload.get("slippage_points", 0), "slippage_points", 0, 10000)
    commission = number(payload.get("commission_per_lot_side", 0), "commission_per_lot_side", 0, 100000)
    return bars, lot, slip, commission


def read_config(root: Path, defaults: dict):
    """Unlike load_trader_config, never creates a missing config."""
    cfg = dict(defaults)
    path = root / "trader_config.json"
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(saved, dict):
                raise ValueError("配置不是对象")
            cfg.update(saved)
        except (OSError, ValueError) as exc:
            raise QuickBacktestError(f"无法读取当前配置: {exc}") from exc
    return cfg


def resolve_strategy(root: Path, name):
    if not isinstance(name, str) or not name.strip():
        raise QuickBacktestError("缺少 strategy_file")
    name = name.strip().replace("\\", "/")
    # Accept the library's strategies/foo.json or basename; reject traversal,
    # absolute paths, nested paths, drive/ADS syntax and Windows special names.
    if name.startswith("strategies/"):
        name = name[len("strategies/"):]
    if "/" in name or ":" in name or not name.endswith(".json") or name in (".json", "..json"):
        raise QuickBacktestError("策略必须是 strategies 目录内的 JSON 文件")
    base = (root / "strategies").resolve()
    path = (base / name).resolve()
    if path.parent != base:
        raise QuickBacktestError("策略路径越界")
    if not path.is_file():
        raise QuickBacktestError("策略文件不存在", 404)
    if path.stat().st_size > 1_000_000:
        raise QuickBacktestError("策略文件过大")
    try:
        meta = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        raise QuickBacktestError(f"策略 JSON 无效: {exc}") from exc
    if not isinstance(meta, dict):
        raise QuickBacktestError("策略 JSON 必须是对象")
    formula = meta.get("formula", meta.get("formula_tokens"))
    if (not isinstance(formula, list) or not 1 <= len(formula) <= MAX_FORMULA_TOKENS
            or any(type(t) is not int or t < 0 for t in formula)):
        raise QuickBacktestError("策略公式必须是非负整数 token 数组（1..128 项）")
    symbol = meta.get("symbol")
    if not isinstance(symbol, str) or not symbol.strip() or len(symbol) > 128 or re.search(r"[\x00-\x1f]", symbol):
        raise QuickBacktestError("策略缺少有效 symbol")
    tf = meta.get("timeframe", "H1")
    if not isinstance(tf, str) or tf not in TIMEFRAMES:
        raise QuickBacktestError("策略 timeframe 不支持")
    semantics = meta.get("feature_semantics_version", "legacy-v1")
    if semantics not in ("legacy-v1", "causal-features-v2"):
        raise QuickBacktestError("策略特征语义版本不支持", 422)
    return {"strategy_file": "strategies/" + path.name, "symbol": _symbol(symbol, "策略 symbol"),
            "timeframe": tf, "formula": formula, "vocab_version": meta.get("vocab_version", ""),
            "feature_semantics_version": semantics}


def _account_identity(account):
    if account is None:
        return None
    return (str(getattr(account, "server", "")), getattr(account, "login", None),
            str(getattr(account, "currency", "")))


def _terminal_context(mt5, meta):
    try:
        terminal = mt5.terminal_info()
        account = mt5.account_info()
        info = mt5.symbol_info(meta["symbol"])
    except Exception as exc:
        raise QuickBacktestError(f"无法读取 MT5: {exc}", 503) from exc
    if terminal is None or not getattr(terminal, "connected", False) or account is None:
        raise QuickBacktestError("MT5 未连接或未登录；请先打开终端并登录", 503)
    currency = getattr(account, "currency", "")
    if not currency:
        raise QuickBacktestError("MT5 未返回账户货币", 503)
    if info is None:
        raise QuickBacktestError("MT5 中不存在所选回测品种", 422)
    if not getattr(info, "visible", False):
        raise QuickBacktestError("请先在 MT5 市场报价中显示所选回测品种；回测不会修改品种选择", 422)
    return account, info


def _lot_options(cfg, meta, info):
    minimum = number(getattr(info, "volume_min", None), "MT5 volume_min", 1e-8, 1e6)
    maximum = number(getattr(info, "volume_max", None), "MT5 volume_max", minimum, 1e6)
    step = number(getattr(info, "volume_step", None), "MT5 volume_step", 1e-8, 1e6)
    cap = number(cfg.get("max_lot_per_trade"), "配置 max_lot_per_trade", 1e-8, 1e6)
    maximum = min(maximum, cap)
    if maximum < minimum:
        raise QuickBacktestError("配置手数上限低于品种最小手数")
    maximum = float((Decimal(str(maximum)) / Decimal(str(step))).to_integral_value(rounding=ROUND_FLOOR) * Decimal(str(step)))
    default, source = 0.1, "默认首选手数（0.1）"
    for binding in cfg.get("bindings", []):
        if not isinstance(binding, dict):
            continue
        same_strategy = str(binding.get("strategy_file", "")).replace("\\", "/").split("/")[-1] == meta["strategy_file"].split("/")[-1]
        binding_symbol = binding.get("symbol")
        same_symbol = binding_symbol is None or str(binding_symbol).strip() == meta["symbol"]
        if same_strategy and same_symbol:
            default = number(binding.get("lot", cap), "策略绑定 lot", 1e-8, 1e6)
            source = "config.bindings.lot"
            break
    default = min(maximum, max(minimum, default))
    default = float((Decimal(str(default)) / Decimal(str(step))).to_integral_value(rounding=ROUND_FLOOR) * Decimal(str(step)))
    if default < minimum or maximum < minimum:
        raise QuickBacktestError("品种手数步长与可用上下限不兼容")
    return {"min": minimum, "max": maximum, "step": step, "default": default, "source": source}


def _warmup(cfg):
    value = cfg.get("signal_bars", 1000)
    if isinstance(value, bool) or not isinstance(value, int) or not 800 <= value <= 10000:
        raise QuickBacktestError("配置 signal_bars 必须是 800..10000 的整数")
    return value


def get_options(root, strategy_file, cfg, mt5, *, symbol=None):
    strategy = resolve_strategy(root, strategy_file)
    meta = _effective_meta(strategy, symbol)
    account, info = _terminal_context(mt5, meta)
    return {k: meta[k] for k in ("strategy_file", "symbol", "timeframe", "strategy_symbol")} | {
        "currency": account.currency, "bars": {"min": MIN_BARS, "max": MAX_BARS, "default": DEFAULT_BARS},
        "lot": _lot_options(cfg, meta, info), "warmup_bars": _warmup(cfg),
        "costs": {"spread_mode": "bar_spread", "slippage_points": 0, "commission_per_lot_side": 0},
        "limitations": list(LIMITATIONS)}


def compute_positions(rates, meta):
    """Use the shared causal VM, never the deployed legacy global gate."""
    try:
        import numpy as np
        import torch
        from model_core.vm import StackVM
        from model_core.vocab import FORMULA_VOCAB, VOCAB_VERSION
        from model_core.features import MT5FeatureEngineer
        from trading.signal_engine import rates_to_raw_dict
    except ImportError as exc:
        raise QuickBacktestError(f"信号计算依赖不可用: {exc}", 503) from exc
    if meta["vocab_version"] and meta["vocab_version"] != VOCAB_VERSION:
        raise QuickBacktestError("策略词表版本与当前引擎不匹配", 422)
    if any(t >= len(FORMULA_VOCAB.token_names) for t in meta["formula"]):
        raise QuickBacktestError("策略含未知 token", 422)

    try:
        with torch.inference_mode():
            raw = rates_to_raw_dict(rates)
            if raw is None:
                raise ValueError("K线字段无效")
            features = MT5FeatureEngineer.compute_features(raw)
            factor = StackVM().execute(meta["formula"], features)
            if factor is None or tuple(factor.shape) != (1, len(rates)) or not torch.isfinite(factor).all():
                raise ValueError("公式没有有效输出")
            result = torch.tanh(factor[0]).cpu().numpy().astype(np.float64)
        return result
    except QuickBacktestError:
        raise
    except Exception as exc:
        raise QuickBacktestError(f"策略信号计算失败: {exc}", 422) from exc


def _validate_rates(rates, required):
    import numpy as np
    if rates is None or len(rates) != required:
        got = 0 if rates is None else len(rates)
        raise QuickBacktestError(f"MT5 已收盘历史不足：需要 {required} 根（含预热），实际 {got}；请在终端加载更多历史", 422)
    names = rates.dtype.names or ()
    if any(k not in names for k in ("time", "open", "high", "low", "close", "tick_volume", "spread")):
        raise QuickBacktestError("MT5 K线缺少 OHLC/时间/成交量/历史点差字段", 422)
    for key in ("open", "high", "low", "close", "tick_volume", "spread", "time"):
        if not np.isfinite(rates[key]).all() or (rates[key] < 0).any():
            raise QuickBacktestError(f"MT5 K线 {key} 含无效数值", 422)
    if ((rates["low"] <= 0).any() or (rates["time"] <= 0).any()
            or (np.diff(rates["time"]) <= 0).any()
            or (rates["low"] > np.minimum(rates["open"], rates["close"])).any()
            or (rates["high"] < np.maximum(rates["open"], rates["close"])).any()):
        raise QuickBacktestError("MT5 K线价格或时间顺序无效", 422)


def run_quick_backtest(root, payload, cfg, mt5, *, signal_fn=None, server_offset_sec=None):
    started = time.monotonic()
    bars, lot, slippage, commission_rate = validate_request(payload)
    strategy = resolve_strategy(root, payload.get("strategy_file"))
    meta = _effective_meta(strategy, payload.get("symbol"))
    account, info = _terminal_context(mt5, meta)
    lots = _lot_options(cfg, meta, info)
    number(lot, "lot", lots["min"], lots["max"])
    units = lot / lots["step"]
    if not math.isclose(units, round(units), abs_tol=1e-7, rel_tol=0):
        raise QuickBacktestError(f"lot 必须符合步长 {lots['step']}")
    warmup = _warmup(cfg)
    threshold = number(cfg.get("min_trade_exposure", 0.05), "min_trade_exposure", 1e-8, 1)
    point = number(getattr(info, "point", None), "MT5 point", 1e-12, 1e6)
    timeframe = getattr(mt5, "TIMEFRAME_" + meta["timeframe"], None)
    if timeframe is None:
        raise QuickBacktestError("MT5 不支持策略周期", 422)
    try:
        # Position zero is forming. Fetch extra *older* bars solely for warmup.
        rates = mt5.copy_rates_from_pos(meta["symbol"], timeframe, 1, bars + warmup)
    except Exception as exc:
        raise QuickBacktestError(f"MT5 历史读取失败: {exc}", 503) from exc
    _validate_rates(rates, bars + warmup)
    signals = (signal_fn or compute_positions)(rates, meta)
    import numpy as np
    signals = np.asarray(signals, dtype=float)
    if signals.shape != (len(rates),) or not np.isfinite(signals).all():
        raise QuickBacktestError("信号序列无效", 422)
    directions = np.where(signals >= threshold, 1, np.where(signals <= -threshold, -1, 0))
    action = {1: mt5.ORDER_TYPE_BUY, -1: mt5.ORDER_TYPE_SELL}

    def check_time():
        if time.monotonic() - started > MAX_SECONDS:
            raise QuickBacktestError("快速回测超时，请减少 K 线数量", 503)

    def profit(side, entry, exit_price):
        check_time()
        if not (entry > 0 and exit_price > 0):
            raise QuickBacktestError("成本假设导致成交价格非正数", 422)
        try:
            value = mt5.order_calc_profit(action[side], meta["symbol"], lot, float(entry), float(exit_price))
        except Exception as exc:
            raise QuickBacktestError(f"MT5 order_calc_profit 失败: {exc}", 503) from exc
        if value is None or not math.isfinite(float(value)):
            raise QuickBacktestError("MT5 order_calc_profit 无法计算账户货币盈亏（检查合约/换汇报价）", 503)
        return float(value)

    check_time()
    # Probe even for an all-flat strategy: unavailable conversion must not yield
    # a fabricated successful report. This API calculates, never places orders.
    profit(1, float(rates["open"][warmup]), float(rates["open"][warmup]))
    trade_list, curve = [], []
    realized = peak = max_dd = closed_peak = closed_dd = 0.0
    current = None
    commission_trade = commission_rate * lot * 2

    def valuation(position, bid, spread):
        side = position["side"]
        exit_spread = bid + (spread if side == -1 else 0)
        exit_price = exit_spread - side * slippage * point
        gross = profit(side, position["bid"], bid)
        with_spread = profit(side, position["spread_entry"], exit_spread)
        after_slip = profit(side, position["entry_price"], exit_price)
        return {"exit_price": exit_price, "gross_profit": gross,
                "spread_cost": gross - with_spread, "slippage_cost": with_spread - after_slip,
                "commission": commission_trade, "swap": 0.0,
                "net_profit": after_slip - commission_trade}

    def close(position, bid, spread, ts, forced):
        nonlocal realized, closed_peak, closed_dd
        values = valuation(position, bid, spread)
        realized += values["net_profit"]
        closed_peak = max(closed_peak, realized)
        closed_dd = max(closed_dd, closed_peak - realized)
        trade_list.append({"side": "LONG" if position["side"] == 1 else "SHORT", "lot": lot,
                           "signal_time": position["signal_time"], "entry_time": position["time"],
                           "exit_time": ts, "entry_price": position["entry_price"],
                           "forced_close": forced, **values})

    for i in range(warmup, len(rates)):
        check_time()
        row = rates[i]
        desired = int(directions[i - 1])  # Never use this bar's close to enter at its open.
        bid, spread, ts = float(row["open"]), float(row["spread"]) * point, int(row["time"])
        if current is not None and desired != current["side"]:
            close(current, bid, spread, ts, False)
            current = None
        if current is None and desired:
            spread_entry = bid + (spread if desired == 1 else 0)
            current = {"side": desired, "bid": bid, "spread_entry": spread_entry,
                       "entry_price": spread_entry + desired * slippage * point,
                       "time": ts, "signal_time": int(rates["time"][i - 1])}
        if i == len(rates) - 1 and current is not None:
            close(current, float(row["close"]), spread, ts + TIMEFRAMES[meta["timeframe"]], True)
            current = None
        floating = valuation(current, float(row["close"]), spread)["net_profit"] if current else 0.0
        equity = realized + floating
        peak = max(peak, equity)
        drawdown = peak - equity
        max_dd = max(max_dd, drawdown)
        curve.append({"time": ts, "equity": equity, "realized": realized, "drawdown": drawdown})

    summary = {key: sum(t[key] for t in trade_list) for key in
               ("gross_profit", "spread_cost", "slippage_cost", "commission", "swap", "net_profit")}
    summary.update({"total_cost": summary["spread_cost"] + summary["slippage_cost"] + summary["commission"],
                    "max_drawdown": max_dd, "closed_trade_max_drawdown": closed_dd,
                    "trades": len(trade_list), "wins": sum(t["net_profit"] > 0 for t in trade_list),
                    "losses": sum(t["net_profit"] < 0 for t in trade_list)})
    summary["win_rate"] = summary["wins"] / len(trade_list) if trade_list else 0.0
    # Refuse mixed-account results if the user changed accounts mid-request.
    after = mt5.account_info()
    if _account_identity(after) != _account_identity(account):
        raise QuickBacktestError("回测期间 MT5 账户或服务商已切换、或已断开，请重试", 503)
    return {"ok": True, **{k: meta[k] for k in ("strategy_file", "symbol", "timeframe", "strategy_symbol")},
            "currency": account.currency, "bars_requested": bars, "bars_used": bars,
            "warmup_bars": warmup, "lot": lot, "server_offset_sec": server_offset_sec,
            "summary": summary, "equity_curve": curve, "trades": trade_list,
            "assumptions": {"execution": "signal_close_to_next_open", "final_exit": "last_closed_bar_close",
                            "spread_mode": "bar_spread", "slippage_points": slippage,
                            "commission_per_lot_side": commission_rate, "commission_currency": account.currency,
                            "swap": 0, "signal_threshold": threshold, "equity_start": 0,
                            "equity_time": "bar_open_timestamp_for_close_mark", "drawdown_basis": "bar_close_liquidation_equity",
                            "gross_profit_definition": "signed_pnl_before_spread_slippage_commission",
                            "currency_conversion": "MT5_current_contract_and_FX", "point": point,
                            "zero_spread_bars": int(np.count_nonzero(rates['spread'][warmup:] == 0)),
                            "normalization": "causal_prefix_constant_gate_then_rolling_500",
                            "normalization_warmup": "bars_1_to_499_neutral_bar_500_first_eligible",
                            "feature_semantics_version": "causal-features-v2",
                            "strategy_feature_semantics_version": meta["feature_semantics_version"],
                            "strategy_semantics_differ": meta["feature_semantics_version"] != "causal-features-v2",
                            "pnl_semantics": "fixed_lot_bid_ask_account_currency_not_training_proxy"},
            "limitations": list(LIMITATIONS)}

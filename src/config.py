"""
config.py — MT5AutoTrader 统一配置（项目根目录）

三层配置来源（优先级从高到低）：
  1. trader_config.json   —— 网页看板读写的主配置（品种绑定/手数/风控/dry-run）
  2. .env                 —— MT5 登录凭证（MT5_LOGIN / MT5_PASSWORD / MT5_SERVER）
  3. 本文件默认值         —— 兜底

训练子系统（model_core / data_pipeline / train_file.py / run_backtest.py）
从这里读取 Config 的公共字段，字段名与 AlphaMaster 保持一致。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

try:
    import MetaTrader5 as mt5
    _MT5_AVAILABLE = True
except ImportError:  # 无 MT5 的测试环境用整数占位常量（与真实 MT5 值一致）
    _MT5_AVAILABLE = False

    class _MT5Stub:
        TIMEFRAME_M1 = 1
        TIMEFRAME_M5 = 5
        TIMEFRAME_M15 = 15
        TIMEFRAME_M30 = 30
        TIMEFRAME_H1 = 16385
        TIMEFRAME_H4 = 16388
        TIMEFRAME_D1 = 16408

    mt5 = _MT5Stub()

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
except ImportError:
    pass

# SRC_DIR = 代码目录（src\），ROOT_DIR = 项目根目录（数据/配置/日志所在）
SRC_DIR = Path(__file__).resolve().parent
ROOT_DIR = SRC_DIR.parent
TRADER_CONFIG_FILE = ROOT_DIR / "trader_config.json"

# trader_config.json 的默认内容（首次运行时生成）
DEFAULT_TRADER_CONFIG: dict = {
    # ── 运行模式 ─────────────────────────────────────────────
    # True = dry-run：信号照算、动作只记日志，不发真实订单（默认安全）
    # 切换为 False 时看板会要求确认且校验 MT5 账号是模拟盘
    "dry_run": True,
    # ── 策略绑定：训练出的策略手动导入 strategies/ 后在这里绑定品种 ──
    # [{"strategy_file": "strategies/best_ETHUSD_.json", "symbol": "ETHUSD_", "lot": 0.01}]
    "bindings": [],
    # ── 实时风控（1.txt 第 10 节要求）─────────────────────────
    "risk": {
        "enable_price_monitor": True,
        "price_monitor_interval": 10,     # 秒，5~10
        "stop_loss_pct": -0.02,           # 初始止损 -2%
        # 阶梯保本：[触发浮盈, 锁定利润]，浮盈达到触发值后止损移到锁定位
        "breakeven_levels": [
            [0.01, 0.00],
            [0.02, 0.01],
            [0.03, 0.02],
            [0.04, 0.03],
        ],
        # ── 支撑/阻力位优化（V3 融合算法，见 src/trading/srlab/）────
        # 开仓时用支撑/阻力位优化止损止盈；固定止损 stop_loss_pct 始终保留，
        # 最终止损取「固定止损」与「S/R 止损」中更近（更保守）的一个。
        "sr_enabled": True,
        "sr_buffer_atr": 0.25,    # 止损放在区带边缘外再让出 0.25*ATR
        "sr_tp_buffer_atr": 0.10, # 止盈放在对面区带前 0.10*ATR（抢在停滞区前离场）
        "sr_min_sl_pct": 0.004,   # S/R 止损距开仓价最近 0.4%（再近视为噪音）
        "sr_max_sl_pct": 0.03,    # 超过 3% 的止损位不采纳（用固定止损）
        "sr_min_tp_pct": 0.006,   # 止盈至少距开仓价 0.6%
        "sr_max_tp_pct": 0.06,    # 超过 6% 不设止盈（交给阶梯保本管理）
        # 关键位前"止盈一半"：到达对面关键位前自动平掉一部分仓位锁利，
        # 剩余仓位继续持有（交给阶梯保本/更远的关键位管理）。
        "sr_partial_enabled": True,
        "sr_partial_fraction": 0.5,   # 平掉的比例
        "sr_partial_use_model": True, # 由概率模型把关：该关键位守住概率足够高才执行
        "sr_partial_min_phold": 0.55, # 守住概率门槛
    },
    # ── 交易限制 ─────────────────────────────────────────────
    "min_trade_exposure": 0.05,           # |tanh(factor)| 低于此值视为空仓
    "max_lot_per_trade": 1.0,
    "max_open_positions": 3,
    "magic_number": 20260904,
    "deviation_points": 20,
    # ── 数据 ────────────────────────────────────────────────
    "signal_bars": 1000,                  # 信号计算拉取的 H1 K 线数（≥800）
    "kline_cache_dir": r"D:\K线数据",
}


def load_trader_config() -> dict:
    """读取 trader_config.json；不存在或缺字段时用默认值补齐并写回。"""
    cfg = json.loads(json.dumps(DEFAULT_TRADER_CONFIG))  # deep copy
    if TRADER_CONFIG_FILE.exists():
        try:
            data = json.loads(TRADER_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for key, val in data.items():
                    if key == "risk" and isinstance(val, dict):
                        cfg["risk"].update(val)
                    else:
                        cfg[key] = val
        except (json.JSONDecodeError, OSError):
            pass  # 配置损坏时退回默认值，不让交易进程崩掉
    else:
        try:
            save_trader_config(cfg)
        except OSError:
            pass
    return cfg


def save_trader_config(cfg: dict) -> None:
    TRADER_CONFIG_FILE.write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


_TRADER = load_trader_config()
_RISK = _TRADER.get("risk", {})


class Config:
    # ── MT5 连接凭证（.env）─────────────────────────────────
    MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0") or 0)
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER = os.getenv("MT5_SERVER", "")

    # ── 品种与周期（训练子系统需要）──────────────────────────
    SYMBOLS = ["ETHUSD_", "BTCUSD_"]
    TIMEFRAME = mt5.TIMEFRAME_H1
    BARS_COUNT = 10_000_000
    MIN_BARS = 300
    DATA_REFRESH_INTERVAL = 300
    KLINE_CACHE_DIR = _TRADER.get("kline_cache_dir", r"D:\K线数据")

    # ── 模型参数（训练实际使用 model_core.config.ModelConfig）──
    INPUT_DIM = 20
    DEVICE = "cpu"

    # ── 回测/训练成本 ───────────────────────────────────────
    COST_RATE = 0.0003

    # ── 信号阈值 ────────────────────────────────────────────
    MIN_TRADE_EXPOSURE = float(_TRADER.get("min_trade_exposure", 0.05))

    # ── 风控 ────────────────────────────────────────────────
    STOP_LOSS_PCT = float(_RISK.get("stop_loss_pct", -0.02))
    TAKE_PROFIT_PCT = 0.04
    MAX_OPEN_POSITIONS = int(_TRADER.get("max_open_positions", 3))
    MAX_LOT_PER_TRADE = float(_TRADER.get("max_lot_per_trade", 1.0))

    # ── 文件路径 ────────────────────────────────────────────
    STRATEGY_FILE = "best_mt5_strategy.json"
    CHECKPOINT_DIR = "checkpoints"
    PORTFOLIO_FILE = "portfolio_state.json"
    STOP_SIGNAL = "STOP_SIGNAL"

    # ── Magic Number / 下单 ─────────────────────────────────
    MAGIC_NUMBER = int(_TRADER.get("magic_number", 20260904))
    DEVIATION_POINTS = int(_TRADER.get("deviation_points", 20))
    DRY_RUN = bool(_TRADER.get("dry_run", True))

    # ── 实时风控（runner 每个循环会重新读取 trader_config.json 热更新）──
    ENABLE_PRICE_MONITOR = bool(_RISK.get("enable_price_monitor", True))
    PRICE_MONITOR_INTERVAL = max(5, min(10, int(_RISK.get("price_monitor_interval", 10))))
    BREAKEVEN_LEVELS = tuple(
        (float(t), float(lock))
        for t, lock in _RISK.get(
            "breakeven_levels",
            DEFAULT_TRADER_CONFIG["risk"]["breakeven_levels"],
        )
    )

    @classmethod
    def reload(cls) -> None:
        """热更新：从 trader_config.json 重新加载（runner 每个循环调用）。"""
        global _TRADER
        _TRADER = load_trader_config()
        risk = _TRADER.get("risk", {})
        cls.MIN_TRADE_EXPOSURE = float(_TRADER.get("min_trade_exposure", 0.05))
        cls.STOP_LOSS_PCT = float(risk.get("stop_loss_pct", -0.02))
        cls.MAX_OPEN_POSITIONS = int(_TRADER.get("max_open_positions", 3))
        cls.MAX_LOT_PER_TRADE = float(_TRADER.get("max_lot_per_trade", 1.0))
        cls.MAGIC_NUMBER = int(_TRADER.get("magic_number", cls.MAGIC_NUMBER))
        cls.DEVIATION_POINTS = int(_TRADER.get("deviation_points", cls.DEVIATION_POINTS))
        cls.DRY_RUN = bool(_TRADER.get("dry_run", True))
        cls.ENABLE_PRICE_MONITOR = bool(risk.get("enable_price_monitor", True))
        cls.PRICE_MONITOR_INTERVAL = max(
            5, min(10, int(risk.get("price_monitor_interval", 10)))
        )
        cls.BREAKEVEN_LEVELS = tuple(
            (float(t), float(lock))
            for t, lock in risk.get(
                "breakeven_levels",
                DEFAULT_TRADER_CONFIG["risk"]["breakeven_levels"],
            )
        )
        cls.KLINE_CACHE_DIR = _TRADER.get("kline_cache_dir", cls.KLINE_CACHE_DIR)

    @classmethod
    def get_timeframe(cls, tf_str: str) -> int:
        """'H1'/'M15' 等字符串 → MT5 时间框常量。"""
        mapping = {
            "M1": mt5.TIMEFRAME_M1,
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "M30": mt5.TIMEFRAME_M30,
            "H1": mt5.TIMEFRAME_H1,
            "H4": mt5.TIMEFRAME_H4,
            "D1": mt5.TIMEFRAME_D1,
        }
        return mapping.get(str(tf_str).upper(), mt5.TIMEFRAME_H1)

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
import math
import os
import tempfile
import threading
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
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
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
        "reentry_cooldown_sec": 60,   # 全平后同品种自动再入场冷却；0 可显式关闭
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


CONFIG_LOCK = threading.RLock()
_LAST_VALID: dict | None = None


def _finite(value, name: str, low=None, high=None, *, integer=False) -> float:
    try:
        finite = type(value) in (int, float) and math.isfinite(value)
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        raise ValueError(f"{name} must be a finite number")
    if integer and type(value) is not int:
        raise ValueError(f"{name} must be an integer")
    if (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is outside the allowed range")
    return value


def validate_trader_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict) or type(cfg.get("dry_run")) is not bool:
        raise ValueError("dry_run must be a boolean")
    for key, low, high, integer in (
        ("min_trade_exposure", 0, 1, False),
        ("max_lot_per_trade", 1e-12, 100000, False),
        ("max_open_positions", 1, 100000, True),
        ("magic_number", 0, 2147483647, True),
        ("deviation_points", 0, 100000, True),
        ("signal_bars", 800, 10000000, True),
    ):
        _finite(cfg.get(key), key, low, high, integer=integer)
    if not isinstance(cfg.get("kline_cache_dir"), str) or not cfg["kline_cache_dir"].strip():
        raise ValueError("kline_cache_dir must be a nonempty path")
    risk = cfg.get("risk")
    if not isinstance(risk, dict) or set(risk) - set(DEFAULT_TRADER_CONFIG["risk"]):
        raise ValueError("invalid risk keys")
    for key, default in DEFAULT_TRADER_CONFIG["risk"].items():
        value = risk.get(key)
        if type(default) is bool:
            if type(value) is not bool:
                raise ValueError(f"{key} must be a boolean")
        elif key != "breakeven_levels":
            low, high = 0, 100
            if key == "stop_loss_pct": low, high = -1, -1e-12
            elif key == "price_monitor_interval": low, high = 5, 10
            elif key == "reentry_cooldown_sec": low, high = 0, 86400000
            elif key == "sr_partial_fraction": low, high = 1e-12, 1 - 1e-12
            elif key.endswith("_pct") or key == "sr_partial_min_phold": high = 1
            _finite(value, key, low, high, integer=type(default) is int)
    levels = risk.get("breakeven_levels")
    if not isinstance(levels, list):
        raise ValueError("breakeven_levels must be a list")
    previous = -1
    for pair in levels:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError("each breakeven level must have two numbers")
        trigger = _finite(pair[0], "trigger", 1e-12, 1)
        lock = _finite(pair[1], "lock", 0, 1)
        if trigger <= previous or lock >= trigger:
            raise ValueError("breakeven levels must increase and lock below trigger")
        previous = trigger
    for prefix in ("sl", "tp"):
        if risk[f"sr_min_{prefix}_pct"] > risk[f"sr_max_{prefix}_pct"]:
            raise ValueError(f"sr_min_{prefix}_pct exceeds maximum")
    bindings = cfg.get("bindings")
    if not isinstance(bindings, list):
        raise ValueError("bindings must be a list")
    symbols = set()
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("invalid binding")
        for key in ("symbol", "strategy_file"):
            if not isinstance(binding.get(key), str) or not binding[key].strip():
                raise ValueError(f"binding {key} is required")
        path = Path(binding["strategy_file"].replace("\\", "/"))
        if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "strategies" or path.suffix != ".json":
            raise ValueError("binding must use a strategies/*.json path")
        _finite(binding.get("lot"), "lot", 1e-12, cfg["max_lot_per_trade"])
        if binding["symbol"] in symbols:
            raise ValueError("duplicate binding symbol")
        symbols.add(binding["symbol"])
    return cfg


def _atomic_json_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, allow_nan=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    finally:
        Path(tmp_name).unlink(missing_ok=True)


def load_trader_config() -> dict:
    global _LAST_VALID
    with CONFIG_LOCK:
        cfg = json.loads(json.dumps(DEFAULT_TRADER_CONFIG))
        if TRADER_CONFIG_FILE.exists():
            try:
                data = json.loads(TRADER_CONFIG_FILE.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("config must be an object")
                for key, val in data.items():
                    if key == "risk" and isinstance(val, dict):
                        cfg["risk"].update(val)
                    else:
                        cfg[key] = val
                validate_trader_config(cfg)
            except (json.JSONDecodeError, OSError, TypeError, ValueError):
                return json.loads(json.dumps(_LAST_VALID or DEFAULT_TRADER_CONFIG))
        else:
            try:
                save_trader_config(cfg)
            except (OSError, TypeError, ValueError):
                pass
        _LAST_VALID = json.loads(json.dumps(cfg))
        return cfg


def save_trader_config(cfg: dict) -> None:
    global _LAST_VALID
    with CONFIG_LOCK:
        validate_trader_config(cfg)
        _atomic_json_write(TRADER_CONFIG_FILE, cfg)
        _LAST_VALID = json.loads(json.dumps(cfg))


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
    ROOT_DIR = ROOT_DIR
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

    # 各周期一根 K 线的秒数，用于把「开盘时间」换算成「收盘时间」显示
    _TF_SECONDS: dict[str, int] = {
        "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
        "H1": 3600, "H4": 14400, "D1": 86400,
    }

    @classmethod
    def timeframe_seconds(cls, tf_str: str) -> int:
        """'H1'/'M30' 等 → 该周期一根 K 线的秒数（未知按 H1）。"""
        return cls._TF_SECONDS.get(str(tf_str).upper(), 3600)

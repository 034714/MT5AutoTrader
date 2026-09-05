"""从 MT5 导出品种 K线到 parquet（输出到项目数据目录）。

用法:
    python export_mt5_data.py ETHUSD_ 4000
    python export_mt5_data.py BTCUSD_ XAUUSD EURUSD
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import MetaTrader5 as mt5

# 输出目录：优先项目 config 的数据目录，默认 D:\K线数据（与既有数据同处）
try:
    from config import Config as _Config
    OUT_DIR = Path(_Config.KLINE_CACHE_DIR)
except Exception:
    OUT_DIR = Path(r"D:\K线数据")
TIMEFRAME = mt5.TIMEFRAME_H1
_DEFAULT_BARS = 4000
_COLUMNS = ["time", "open", "high", "low", "close", "tick_volume"]


def export_symbol(symbol: str, bars: int, timeframe: str = "H1",
                  out_dir: Path | str | None = None) -> Path | None:
    """导出指定周期的 K 线；文件名 {symbol}_{timeframe}.parquet。"""
    tf = str(timeframe or "H1").upper()
    tf_const = None
    try:
        from config import Config as _Config
        tf_const = _Config.get_timeframe(tf)
    except Exception:
        tf_const = {"M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
                    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1,
                    "H4": mt5.TIMEFRAME_H4, "D1": mt5.TIMEFRAME_D1}.get(tf)
    if tf_const is None:
        print(f"[跳过] {symbol}: 不支持的周期 {tf}")
        return None
    out = Path(out_dir) if out_dir else OUT_DIR

    if mt5.symbol_info(symbol) is None:
        print(f"[跳过] {symbol}: MT5 不认识该品种")
        return None
    if not mt5.symbol_select(symbol, True):
        time.sleep(0.2)

    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars)
    if rates is None or len(rates) == 0:
        print(f"[失败] {symbol}: {mt5.last_error()}")
        return None

    df = pd.DataFrame(rates)[_COLUMNS].copy()
    # time 已经是 Unix 秒 int64（服务器时间），与项目管线一致
    df["time"] = df["time"].astype("int64")
    df["tick_volume"] = df["tick_volume"].astype("int64")

    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{symbol.replace('.', '_')}_{tf}.parquet"
    df.to_parquet(path, index=False)
    t0 = datetime.fromtimestamp(int(df["time"].iloc[0]), tz=timezone.utc).strftime("%Y-%m-%d")
    t1 = datetime.fromtimestamp(int(df["time"].iloc[-1]), tz=timezone.utc).strftime("%Y-%m-%d")
    print(f"[完成] {path.name}: {len(df):,} 根, {t0} ~ {t1}")
    return path


def main() -> None:
    if not mt5.initialize():
        print(f"MT5 连接失败: {mt5.last_error()}（请先打开 TMGM 终端并登录）")
        sys.exit(1)
    ai = mt5.account_info()
    print(f"已连接: {mt5.terminal_info().name}, 账号 {ai.login} ({'演示' if ai.trade_mode == 0 else '实盘'})")

    args = [a for a in sys.argv[1:] if a]
    if not args:
        print("用法: python export_mt5_data.py <品种> [根数]  |  多品种: python export_mt5_data.py ETHUSD_ BTCUSD_")
        sys.exit(1)

    if args[-1].isdigit() and len(args) > 1:
        bars = int(args[-1])
        symbols = args[:-1]
    else:
        bars = _DEFAULT_BARS
        symbols = args

    for sym in symbols:
        export_symbol(sym, bars)

    mt5.shutdown()


if __name__ == "__main__":
    main()

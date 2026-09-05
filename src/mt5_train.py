"""
mt5_train.py — 从 MT5 读取 K 线并立即训练（周期可选）

这个入口把"获取 MT5 数据"和"离线训练"串成一个用户无需理解 Parquet 的流程。
数据仍会保存为 Parquet，便于复用、回测和审计；训练本身不再连接 MT5。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import MetaTrader5 as mt5
import pandas as pd

from config import Config
from train_file import train_from_file

_SUPPORTED_TF = ("M5", "M15", "M30", "H1", "H4", "D1")


def fetch_mt5(symbol: str, bars: int, timeframe: str, output_dir: Path) -> Path:
    tf = str(timeframe or "H1").upper()
    if tf not in _SUPPORTED_TF:
        raise RuntimeError(f"不支持的周期: {tf}（可选 {'/'.join(_SUPPORTED_TF)}）")
    if not mt5.initialize():
        raise RuntimeError(f"MT5 连接失败: {mt5.last_error()}（请先打开 MT5 终端并登录）")
    try:
        if mt5.symbol_info(symbol) is None:
            raise RuntimeError(f"MT5 不认识品种: {symbol}")
        mt5.symbol_select(symbol, True)
        rates = mt5.copy_rates_from_pos(symbol, Config.get_timeframe(tf), 0, int(bars))
        if rates is None or len(rates) == 0:
            raise RuntimeError(f"读取 {symbol} {tf} K线失败: {mt5.last_error()}")
        columns = ["time", "open", "high", "low", "close", "tick_volume"]
        frame = pd.DataFrame(rates)
        frame = frame[[c for c in columns if c in frame.columns]].copy()
        frame["time"] = frame["time"].astype("int64")
        if "tick_volume" in frame.columns:
            frame["tick_volume"] = frame["tick_volume"].astype("int64")
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{symbol}_{tf}.parquet"
        frame.to_parquet(path, index=False)
        print(f"[MT5] 已读取 {len(frame):,} 根 {symbol} {tf} K线")
        print(f"[MT5] 数据已保存: {path}")
        return path
    finally:
        mt5.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--bars", type=int, default=6000)
    parser.add_argument("--timeframe", default="H1")
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()
    if args.bars < 800:
        parser.error("--bars 至少需要 800")
    if args.steps > 0:
        from model_core.config import ModelConfig
        ModelConfig.TRAIN_STEPS = args.steps
    try:
        data_path = fetch_mt5(args.symbol, args.bars, args.timeframe,
                              Path(Config.KLINE_CACHE_DIR))
        result = train_from_file(str(data_path), from_scratch=args.from_scratch)
        return 0 if result is not None else 1
    except Exception as exc:
        print(f"[错误] MT5 直连训练失败: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

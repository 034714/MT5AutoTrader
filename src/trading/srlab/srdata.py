# -*- coding: utf-8 -*-
"""ATR / tick 估算（内嵌自上游 srlab/data.py 的纯 numpy 部分）。"""
from __future__ import annotations

import numpy as np


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    prev_c = np.concatenate([[close[0]], close[:-1]])
    return np.maximum(high - low,
                      np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))


def atr_series(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    """
    因果 ATR（滚动均值）。第 i 个元素仅使用 0..i 的数据，可安全用于 t 时刻决策。
    前 period 根用累计均值填充，不产生 NaN。
    """
    tr = true_range(high, low, close)
    n = len(tr)
    out = np.empty(n, dtype=np.float64)
    if n == 0:
        return out
    csum = np.cumsum(tr)
    for i in range(min(period, n)):
        out[i] = csum[i] / (i + 1)
    if n > period:
        roll = (csum[period:] - np.concatenate([[0.0], csum[:n - period - 1]]))
        out[period:] = roll / period
    return out


def tick_size_guess(close: np.ndarray) -> float:
    """
    估计最小价格变动单位，仅用于给分箱宽度设下限，避免箱宽小于可成交精度。
    """
    px = float(np.nanmedian(close))
    if px <= 0:
        return 0.01
    if px < 1000:
        return 0.01
    return 10 ** (np.floor(np.log10(px)) - 4)

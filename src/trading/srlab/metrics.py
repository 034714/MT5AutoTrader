# -*- coding: utf-8 -*-
"""Wilson 置信区间（内嵌自上游 srlab/metrics.py，本项目只需这一个函数）。"""
from __future__ import annotations

from typing import Tuple

import numpy as np


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float, float]:
    """Wilson 得分区间。返回 (点估计, 下界, 上界)，n=0 时全 nan"""
    if n <= 0:
        return (np.nan, np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))

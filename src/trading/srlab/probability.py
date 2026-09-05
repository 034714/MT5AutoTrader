# -*- coding: utf-8 -*-
"""
双概率模型：触及概率 + 守住概率（内嵌自上游 srlab/probability.py 的线上预测部分，
去掉 pandas 依赖的训练辅助函数）。

为什么是两个概率：
    ① 触及概率  未来 horizon 根内，价格会不会走到这个位？（实测 AUC 0.73~0.77）
    ② 守住概率  走到之后，会不会反弹（而不是跌穿）？（AUC 0.59~0.61 但校准良好）
一个"守住概率 85% 但触及概率 8%"的位实际用不上，所以两个都要看。

排序分（edge）故意排除距离和带宽，因为那是在检验"算法有没有本事定位"；
而"预测某个位会不会有效"是另一个任务，距离是合法且最强的预测因子。
两个口径不冲突，服务的问题不同。

模型：逻辑回归（自实现，标准化 + L2，不引入 sklearn 依赖）。
models/prob_models.json 缺失或特征列表不符时 ok=False，调用方应优雅降级。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# 特征列表（顺序即模型系数顺序，改动必须重训）
FEATURES: Tuple[str, ...] = (
    "dist",      # |距现价| / ATR —— 触及概率的主驱动
    "dist_sq",   # 距离平方，捕捉非线性
    "width",     # 带宽 / ATR
    "logev",     # log1p(历史触及事件数)
    "stale",     # 陈旧度 = 1 - 近期新鲜度
    "vp",        # 成交量剖面强度（只进概率模型，不进排序分）
    "close_p",   # 收盘价堆积强度
    "atr_pct",   # ATR / 价格，波动率水平
)

TARGETS = ("touch", "hold")
KINDS = ("support", "resistance")


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def features_from_level(dist_atr: float, width_atr: float, n_events: float,
                        stale: float, vp: float, close_p: float,
                        atr: float, price: float) -> np.ndarray:
    """线上单个关键位的特征向量（顺序必须与 FEATURES 一致）"""
    ad = abs(float(dist_atr))
    atr_pct = 0.02
    if price and np.isfinite(price) and price > 0:
        atr_pct = float(np.clip(atr / price, 0.0, 0.2))
    return np.array([
        ad,
        ad * ad,
        float(width_atr),
        float(np.log1p(max(n_events, 0.0))),
        float(stale),
        float(vp),
        float(close_p),
        atr_pct,
    ], dtype=np.float64)


# ============================================================
# 模型容器
# ============================================================
@dataclass
class LogitModel:
    """标准化 + 逻辑回归系数。predict 输入原始特征矩阵。"""
    mean: np.ndarray
    std: np.ndarray
    coef: np.ndarray          # 含截距，长度 = len(FEATURES) + 1
    base_rate: float
    metrics: Dict[str, float] = field(default_factory=dict)

    def predict(self, F: np.ndarray) -> np.ndarray:
        F = np.atleast_2d(np.asarray(F, dtype=np.float64))
        Z = (F - self.mean) / self.std
        X = np.column_stack([np.ones(len(Z)), Z])
        return _sigmoid(X @ self.coef)

    def predict_one(self, f: np.ndarray) -> float:
        return float(self.predict(f.reshape(1, -1))[0])

    @staticmethod
    def from_dict(d: dict) -> "LogitModel":
        return LogitModel(
            mean=np.asarray(d["mean"], dtype=np.float64),
            std=np.asarray(d["std"], dtype=np.float64),
            coef=np.asarray(d["coef"], dtype=np.float64),
            base_rate=float(d.get("base_rate", 0.75)),
            metrics=d.get("metrics", {}),
        )


# ============================================================
# 模型集合（touch/hold × support/resistance）
# ============================================================
class ProbabilityModels:
    """
    线上使用的模型集合。缺文件时 ok=False，调用方应优雅降级（不显示概率）。
    """

    def __init__(self, path: str):
        self.path = path
        self.ok = False
        self.meta: dict = {}
        self._m: Dict[Tuple[str, str], LogitModel] = {}
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                js = json.load(fh)
            if js.get("features") != list(FEATURES):
                # 特征定义变了，旧模型不能用（系数顺序会错位）
                return
            self.meta = js.get("meta", {})
            for tgt in TARGETS:
                for kind in KINDS:
                    d = (js.get("models", {}).get(tgt, {}) or {}).get(kind)
                    if d:
                        self._m[(tgt, kind)] = LogitModel.from_dict(d)
            self.ok = len(self._m) > 0
        except Exception:
            self.ok = False

    def get(self, target: str, kind: str) -> Optional[LogitModel]:
        return self._m.get((target, kind))

    def predict(self, target: str, kind: str, f: np.ndarray) -> Optional[float]:
        m = self.get(target, kind)
        if m is None:
            return None
        p = m.predict_one(f)
        return p if np.isfinite(p) else None

    def metrics(self, target: str, kind: str) -> dict:
        m = self.get(target, kind)
        return dict(m.metrics) if m else {}

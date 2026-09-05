"""
trading/signal_engine.py — 策略加载与信号计算

与 AlphaMaster 训练/回测走完全相同的计算链：
  MT5 K线 → MT5FeatureEngineer.compute_features → StackVM.execute(formula)
  → tanh(最新bar因子值) → 多策略平均 → ≥+thr 做多 / ≤-thr 做空 / 否则观望

策略 JSON 格式（AlphaMaster train_file.py 产出）：
  {"vocab_version", "symbol", "formula": [int...], "formula_decoded",
   "best_score", "timeframe", "data_file", ...}
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model_core.features import MT5FeatureEngineer
from model_core.vm import StackVM

# 实盘最小 bar 数：特征 warm-up(~360) + 滚动归一化窗口(500)
MIN_BARS_SIGNAL = 800
DIR_LONG = "LONG"
DIR_SHORT = "SHORT"
DIR_FLAT = "FLAT"

_VM = StackVM()


class StrategyError(Exception):
    """策略文件无效。"""


def load_strategy_file(path: str | Path) -> dict[str, Any]:
    """读取并校验策略 JSON。校验失败抛 StrategyError。

    返回 {"path","symbol","timeframe","formula","formula_decoded","best_score","vocab_version"}
    """
    p = Path(path)
    if not p.exists():
        raise StrategyError(f"策略文件不存在: {p}")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise StrategyError(f"策略文件读取失败: {exc}") from exc
    formula = data.get("formula", data.get("formula_tokens"))
    if not isinstance(formula, list) or not formula:
        raise StrategyError("策略缺少 formula 字段")
    try:
        formula = [int(t) for t in formula]
    except (TypeError, ValueError) as exc:
        raise StrategyError("formula 含非整数 token") from exc
    score = data.get("best_score", data.get("train_best_score"))
    if score is not None:
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = None
    return {
        "path": str(p),
        "name": p.stem,
        "symbol": str(data.get("symbol", "")),
        "timeframe": str(data.get("timeframe", "H1")),
        "formula": formula,
        "formula_decoded": data.get("formula_decoded", ""),
        "best_score": score,
        "vocab_version": str(data.get("vocab_version", "")),
    }


def check_vocab_version(vocab_version: str) -> bool:
    """校验策略的 vocab_version 与本引擎词表一致。"""
    try:
        from model_core.vocab import VOCAB_VERSION
    except ImportError:  # pragma: no cover
        return True
    return vocab_version == VOCAB_VERSION


def rates_to_raw_dict(rates) -> dict[str, torch.Tensor] | None:
    """MT5 copy_rates structured array → raw_dict 张量（与训练侧 data_manager 一致）。

    volume 用 tick_volume（训练管线同样以 tick_volume 作为 volume）。
    """
    if rates is None or len(rates) == 0:
        return None
    names = rates.dtype.names
    if names is None or "close" not in names:
        return None
    raw: dict[str, torch.Tensor] = {}
    for field in ("open", "high", "low", "close"):
        if field not in names:
            return None
        raw[field] = torch.tensor(
            np.asarray(rates[field], dtype=np.float32), dtype=torch.float32
        ).unsqueeze(0)
    if "tick_volume" in names:
        vol = np.asarray(rates["tick_volume"], dtype=np.float32)
    elif "volume" in names:
        vol = np.asarray(rates["volume"], dtype=np.float32)
    else:
        vol = np.ones(len(rates), dtype=np.float32)
    raw["volume"] = torch.tensor(vol, dtype=torch.float32).unsqueeze(0)
    if "time" in names:
        # MT5 structured array 字段可能是非标准 stride；先 copy 成连续数组，
        # 避免 torch.tensor 在 Windows/不同 MT5 版本上拒绝该字段。
        times = np.array(rates["time"], dtype=np.int64, copy=True)
        raw["time"] = torch.tensor(times, dtype=torch.int64).unsqueeze(0)
    return raw


def compute_signal(
    formulas: list[list[int]],
    raw_dict: dict[str, torch.Tensor],
    min_trade_exposure: float = 0.05,
) -> dict[str, Any]:
    """在最新已收盘 bar 上计算信号（多策略取 tanh 后平均）。

    Returns:
        {"state","direction","strength","position","bars_used","message"}
    """
    close = raw_dict.get("close")
    if close is None or close.ndim != 2:
        return {"state": "error", "direction": DIR_FLAT, "strength": 0.0,
                "position": 0.0, "bars_used": 0, "message": "行情数据格式无效"}
    n_bars = int(close.shape[1])
    if n_bars < MIN_BARS_SIGNAL:
        return {"state": "insufficient", "direction": DIR_FLAT, "strength": 0.0,
                "position": 0.0, "bars_used": n_bars,
                "message": f"历史 bar 不足（{n_bars}/{MIN_BARS_SIGNAL}）"}

    try:
        feats = MT5FeatureEngineer.compute_features(raw_dict)
    except Exception as exc:
        return {"state": "error", "direction": DIR_FLAT, "strength": 0.0,
                "position": 0.0, "bars_used": n_bars, "message": f"特征计算失败: {exc}"}

    positions: list[float] = []
    for formula in formulas:
        try:
            factor = _VM.execute([int(t) for t in formula], feats)
        except Exception as exc:
            return {"state": "error", "direction": DIR_FLAT, "strength": 0.0,
                    "position": 0.0, "bars_used": n_bars, "message": f"公式执行失败: {exc}"}
        if factor is None or factor.ndim != 2 or factor.shape[1] == 0:
            continue
        value = float(factor[0, -1])
        if not math.isfinite(value):
            continue
        positions.append(math.tanh(value))

    if not positions:
        return {"state": "error", "direction": DIR_FLAT, "strength": 0.0,
                "position": 0.0, "bars_used": n_bars, "message": "公式无有效输出"}

    position = sum(positions) / len(positions)
    strength = abs(position)
    if position >= min_trade_exposure:
        direction = DIR_LONG
    elif position <= -min_trade_exposure:
        direction = DIR_SHORT
    else:
        direction = DIR_FLAT
    return {
        "state": "ok",
        "direction": direction,
        "strength": round(strength, 4),
        "position": round(position, 4),
        "bars_used": n_bars,
        "message": "",
    }


def formula_preview(formula: list[int]) -> str:
    """把 token 序列解码成可读公式（用于看板展示）。"""
    try:
        from model_core.vocab import FORMULA_VOCAB
        names = FORMULA_VOCAB.token_names
        return " → ".join(names[int(t)] for t in formula if 0 <= int(t) < len(names))
    except Exception:
        return str(formula)


def vocab_fingerprint() -> str:
    """本地词表指纹（用于诊断）。"""
    try:
        from model_core.vocab import VOCAB_VERSION
        return VOCAB_VERSION
    except Exception:
        return hashlib.sha256(b"unknown").hexdigest()[:12]

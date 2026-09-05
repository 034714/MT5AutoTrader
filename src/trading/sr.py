"""
trading/sr.py — 支撑/阻力位（S/R）交易适配层

上游算法（src/trading/srlab/，V3 融合：成交量分布 + 极值结构 + ATR 归一化）
负责"关键位在哪、多强"；本模块把它接到交易上：

  1. detect_levels()    —— 给看板/日志输出当前关键位（距离带 0.5~5 ATR，每侧限 3 个）
  2. sl_tp_for_trade()  —— 开仓时的止盈止损优化（加性规则，绝不放松风控）：
       止损 = 取「固定初始止损」与「S/R 结构位止损」中更近（更保守）的一个；
              S/R 止损藏在区带外缘再让出 sr_buffer_atr*ATR，
              距开仓价必须在 [sr_min_sl_pct, sr_max_sl_pct] 内，否则弃用该区带
       止盈一半 = 到最近对面关键位前沿（回撤 sr_tp_buffer_atr*ATR）先平掉
              一部分仓位锁利；概率模型把关（守住概率 ≥ partial_min_phold 才执行，
              弱位可能直接突破就让它继续跑）；全部止盈顺延到更远的关键位
       全部止盈 = 未启用止盈一半时停在最近关键位前；启用后顺延到次近关键位
     固定止损（stop_loss_pct）始终参与比较，S/R 永远不会把止损改得更松。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from loguru import logger

from trading.srlab.base import Ctx
from trading.srlab.detectors import V3Fusion, V3Params
from trading.srlab.pivots import zigzag
from trading.srlab.probability import ProbabilityModels, features_from_level
from trading.srlab.srdata import atr_series, tick_size_guess

# 与上游 sr_engine 一致的距离带：更近的位视为噪音，更远的位交易上无意义
MIN_DIST_ATR = 0.5
MAX_DIST_ATR = 5.0

_MODELS_DIR = Path(__file__).resolve().parent / "srlab" / "models"

_detector: Optional[V3Fusion] = None
_prob_models: Optional[ProbabilityModels] = None


def _get_detector() -> V3Fusion:
    global _detector
    if _detector is None:
        _detector = V3Fusion(p=V3Params())
    return _detector


def _get_prob_models() -> ProbabilityModels:
    global _prob_models
    if _prob_models is None:
        _prob_models = ProbabilityModels(str(_MODELS_DIR / "prob_models.json"))
    return _prob_models


@dataclass
class SRParams:
    """S/R 止盈止损参数（trader_config.json 的 risk 段，每圈热更新）。"""
    enabled: bool = True
    buffer_atr: float = 0.25      # 止损藏在区带外缘之外的距离（×ATR）
    tp_buffer_atr: float = 0.10   # 止盈提前到区带之前的距离（×ATR）
    min_sl_pct: float = 0.004     # S/R 止损离开仓价的最小距离
    max_sl_pct: float = 0.03      # 超过该距离的止损位不采纳
    min_tp_pct: float = 0.006     # 止盈离开盘价的最小距离
    max_tp_pct: float = 0.06      # 超过该距离不设止盈
    # 关键位前"止盈一半"：到最近对面关键位前平掉一部分仓位锁利，剩余继续持有
    partial_enabled: bool = True
    partial_fraction: float = 0.5
    partial_use_model: bool = True   # 由概率模型把关：该位守住概率够高才执行
    partial_min_phold: float = 0.55

    @classmethod
    def from_config(cls) -> "SRParams":
        from config import load_trader_config
        risk = load_trader_config().get("risk", {})
        f = lambda key, default: float(risk.get(key, default) or default)  # noqa: E731
        min_sl = max(0.0005, f("sr_min_sl_pct", 0.004))
        min_tp = max(0.0005, f("sr_min_tp_pct", 0.006))
        return cls(
            enabled=bool(risk.get("sr_enabled", True)),
            buffer_atr=max(0.05, f("sr_buffer_atr", 0.25)),
            tp_buffer_atr=max(0.0, f("sr_tp_buffer_atr", 0.10)),
            min_sl_pct=min_sl,
            max_sl_pct=max(min_sl * 1.5, f("sr_max_sl_pct", 0.03)),
            min_tp_pct=min_tp,
            max_tp_pct=max(min_tp * 1.5, f("sr_max_tp_pct", 0.06)),
            partial_enabled=bool(risk.get("sr_partial_enabled", True)),
            partial_fraction=min(0.9, max(0.1, f("sr_partial_fraction", 0.5))),
            partial_use_model=bool(risk.get("sr_partial_use_model", True)),
            partial_min_phold=min(0.95, max(0.05, f("sr_partial_min_phold", 0.55))),
        )


def _col(rates: Any, name: str) -> np.ndarray:
    """MT5 结构化数组 / dict 两种来源都支持，统一成 float64。"""
    try:
        arr = rates[name]
    except (KeyError, IndexError, TypeError):
        arr = rates.get(name) if isinstance(rates, dict) else None
    if arr is None:
        raise ValueError(f"K线缺少字段 {name}")
    return np.array(arr, dtype=np.float64, copy=True)


def _level_json(z, atr: float, price: float) -> dict:
    m = z.meta
    n_ev = int(m.get("n_events", 0))
    p_touch = p_hold = None
    probs = _get_prob_models()
    if probs.ok and price > 0:
        fv = features_from_level(
            dist_atr=z.dist_atr, width_atr=z.width_atr, n_events=n_ev,
            stale=float(m.get("stale", 0.0)), vp=float(m.get("vp", 0.5)),
            close_p=float(m.get("close", 0.5)), atr=atr, price=price)
        p_touch = probs.predict("touch", z.kind, fv)
        p_hold = probs.predict("hold", z.kind, fv)
    p_eff = p_touch * p_hold if (p_touch is not None and p_hold is not None) else None
    label = "支撑带" if z.kind == "support" else "压力带"
    return {
        "kind": z.kind,
        "label": label,
        "center": round(z.center, 6),
        "low": round(z.low, 6),
        "high": round(z.high, 6),
        "dist_atr": round(z.dist_atr, 3),
        "dist_pct": round(z.dist_pct, 3),
        "width_atr": round(z.width_atr, 3),
        "edge_score": round(float(z.score), 4),
        "n_events": n_ev,
        "stale": round(float(m.get("stale", 0.0)), 3),
        "p_stall": round(float(m.get("p_stall", 0.0)), 3),
        "p_touch": round(p_touch, 4) if p_touch is not None else None,
        "p_hold": round(p_hold, 4) if p_hold is not None else None,
        "p_effective": round(p_eff, 4) if p_eff is not None else None,
    }


def detect_levels(symbol: str, rates: Any, max_per_side: int = 3) -> tuple[list[dict], dict]:
    """检测关键位。rates 必须是已收盘 K 线（不含形成中的最后一根）。

    返回 (levels, info)；数据不足时返回 ([], {"error": ...})。
    """
    high = _col(rates, "high")
    low = _col(rates, "low")
    close = _col(rates, "close")
    open_ = _col(rates, "open")
    volume = _col(rates, "tick_volume") if _has_field(rates, "tick_volume") \
        else _col(rates, "volume")
    n = len(close)
    info: dict = {"symbol": symbol, "bars": n}
    if n < 60:
        return [], {"error": f"数据不足：{n} 条", **info}

    atr_arr = atr_series(high, low, close, 14)
    t = n - 1
    atr = max(float(atr_arr[t]), 1e-12)
    price = float(close[t])
    piv = zigzag(high, low, atr_arr, k_atr=2.0)
    ctx = Ctx(
        code=symbol, t=t, open_=open_, high=high, low=low, close=close,
        volume=volume, atr=atr, tick=float(tick_size_guess(close)),
        atr_arr=atr_arr, pivots=piv.view_at(t, max_lookback=600),
    )

    levels = _get_detector().detect(ctx)
    kept: dict[str, list] = {"support": [], "resistance": []}
    for z in levels:
        if z.kind == "support" and z.high >= price:
            continue
        if z.kind == "resistance" and z.low <= price:
            continue
        d = abs(z.center - price) / atr
        if d < MIN_DIST_ATR or d > MAX_DIST_ATR:
            continue
        kept[z.kind].append(z)
    out: list[dict] = []
    for k in ("support", "resistance"):
        kept[k].sort(key=lambda z: -z.score)
        out.extend(kept[k][:max_per_side])
    out.sort(key=lambda z: z.center)  # 按价格从低到高，便于阅读
    return [_level_json(z, atr, price) for z in out], {
        **info, "atr": round(atr, 8), "price": round(price, 8),
        "atr_pct": round(atr / price * 100, 4) if price > 0 else None,
        "prob_ready": _get_prob_models().ok,
    }


def _has_field(rates: Any, name: str) -> bool:
    if isinstance(rates, dict):
        return name in rates
    try:
        rates[name]
        return True
    except Exception:
        return False


def _candidate_zones(levels: list[dict], side: str, entry: float) -> list[dict]:
    """排序后的候选区带：支撑在 entry 下方 / 阻力在上方，由近到远。"""
    out = []
    for z in levels:
        if z["kind"] != side:
            continue
        if side == "support" and z["high"] >= entry:
            continue
        if side == "resistance" and z["low"] <= entry:
            continue
        out.append(z)
    out.sort(key=lambda z: abs(z["center"] - entry))
    return out


def sl_tp_for_trade(symbol: str, direction: str, entry: float, rates: Any,
                    params: SRParams, fixed_sl: float = 0.0) -> dict:
    """开仓止盈止损计划。

    Args:
        direction: 'BUY'/'SELL'。
        entry: 参考开仓价（多头用 ask、空头用 bid）。
        rates: 已收盘 K 线（不含形成中的最后一根）。
        fixed_sl: 固定初始止损价（>0 时参与"取更紧"比较；S/R 不会比它更松）。

    Returns:
        {"sl", "tp", "sl_source": "sr"|"fixed"|"none", "tp_source": "sr"|"none",
         "sl_zone"/"tp_zone": 区带摘要 or None, "partial": 止盈一半计划 or None,
         "levels", "info"}
    """
    result: dict = {"sl": None, "tp": None, "sl_source": "none",
                    "tp_source": "none", "sl_zone": None, "tp_zone": None,
                    "partial": None, "levels": None, "info": None}
    if not params.enabled or entry <= 0:
        if fixed_sl > 0:
            result["sl"], result["sl_source"] = fixed_sl, "fixed"
        return result
    try:
        levels, info = detect_levels(symbol, rates)
    except Exception as exc:
        logger.warning(f"[S/R] {symbol} 关键位检测失败，退回固定止损: {exc}")
        if fixed_sl > 0:
            result["sl"], result["sl_source"] = fixed_sl, "fixed"
        return result
    result["levels"], result["info"] = levels, info
    if not levels or info.get("error"):
        if fixed_sl > 0:
            result["sl"], result["sl_source"] = fixed_sl, "fixed"
        return result
    atr = max(float(info["atr"]), 1e-12)

    def zone_summary(z: dict) -> dict:
        return {k: z[k] for k in ("kind", "center", "low", "high",
                                  "dist_atr", "n_events", "p_hold", "p_stall")}

    # ── 止损：结构位外缘 ± buffer，距离带 [min_sl_pct, max_sl_pct] ──
    sl_side = "support" if direction == "BUY" else "resistance"
    sr_sl = None
    for z in _candidate_zones(levels, sl_side, entry):
        if direction == "BUY":
            raw = z["low"] - params.buffer_atr * atr
            dist = (entry - raw) / entry if raw < entry else -1.0
        else:
            raw = z["high"] + params.buffer_atr * atr
            dist = (raw - entry) / entry if raw > entry else -1.0
        if dist < 0:
            continue
        if dist < params.min_sl_pct:
            continue          # 太近：噪音止损，换更远的区带
        if dist > params.max_sl_pct:
            break             # 更远的区带只会更远
        sr_sl = raw
        result["sl_zone"] = zone_summary(z)
        break

    # 取「固定」与「S/R」中更紧（离开盘价更近）的一个
    if sr_sl is not None and (fixed_sl <= 0 or
                              (direction == "BUY" and sr_sl > fixed_sl) or
                              (direction == "SELL" and sr_sl < fixed_sl)):
        result["sl"], result["sl_source"] = sr_sl, "sr"
    elif fixed_sl > 0:
        result["sl"], result["sl_source"] = fixed_sl, "fixed"

    # ── 止盈：对面关键位前沿再回撤 tp_buffer，距离带 [min_tp_pct, max_tp_pct] ──
    tp_side = "resistance" if direction == "BUY" else "support"
    tp_candidates: list[tuple[dict, float]] = []   # (区带, 前沿价)
    for z in _candidate_zones(levels, tp_side, entry):
        if direction == "BUY":
            raw = z["low"] - params.tp_buffer_atr * atr
            dist = (raw - entry) / entry if raw > entry else -1.0
        else:
            raw = z["high"] + params.tp_buffer_atr * atr
            dist = (entry - raw) / entry if raw < entry else -1.0
        if dist < 0:
            continue
        if dist < params.min_tp_pct:
            continue
        if dist > params.max_tp_pct:
            break             # 太远的目标交给阶梯保本管理
        tp_candidates.append((z, raw))

    # "止盈一半"：到最近的关键位前先平掉一部分锁利（由概率模型把关），
    # 全部止盈顺延到更远的关键位；没有更远的位就不设全止盈，剩余交给阶梯保本。
    if params.partial_enabled and tp_candidates and \
            _partial_gate(tp_candidates[0][0], params):
        z1, raw1 = tp_candidates[0]
        result["partial"] = {
            "price": raw1, "fraction": params.partial_fraction,
            "zone": zone_summary(z1),
        }
        if len(tp_candidates) > 1:
            z2, raw2 = tp_candidates[1]
            result["tp"], result["tp_source"] = raw2, "sr"
            result["tp_zone"] = zone_summary(z2)
    elif tp_candidates:
        z1, raw1 = tp_candidates[0]
        result["tp"], result["tp_source"] = raw1, "sr"
        result["tp_zone"] = zone_summary(z1)
    return result


def _partial_gate(zone: dict, params: SRParams) -> bool:
    """止盈一半的把关：不开模型时直接执行；开模型时要求该关键位的
    历史守住概率达到门槛（强位大概率反弹，适合先落袋一半；
    弱位可能直接被突破，让剩余仓位继续跑）。模型不可用时放行。"""
    if not params.partial_use_model:
        return True
    p_hold = zone.get("p_hold")
    if p_hold is None:
        return True
    return float(p_hold) >= params.partial_min_phold

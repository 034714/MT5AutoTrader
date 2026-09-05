"""
trading/history.py — 把 MT5 成交记录归并成可读的交易历史

MT5 的 history_deals_get 返回逐笔成交（deal）。一笔交易（position）可能由
多笔成交组成：1 笔入场 + N 笔出场（部分平仓时）。按 position_id 归并后，
可以像券商历史报表一样展示每笔交易的开仓/平仓/盈亏。

注意：deal.time 是"服务器墙上时钟"伪时间戳，调用方负责用
MT5Client.server_time_offset() 换算后再展示。
"""
from __future__ import annotations

from typing import Any

# MetaTrader5 常量（与 mt5 模块一致，避免在无 MT5 环境导入失败）
DEAL_TYPE_BUY = 0
DEAL_TYPE_SELL = 1
DEAL_ENTRY_IN = 0
DEAL_ENTRY_OUT = 1
DEAL_ENTRY_INOUT = 2
DEAL_ENTRY_OUT_BY = 3


def group_history_deals(deals, magic: int | None = None) -> dict[str, list[dict]]:
    """把 deal 列表按 position_id 归并。

    Args:
        deals: mt5.history_deals_get() 的返回值（namedtuple 列表）。
        magic: 只保留该 magic 的记录；None 表示不过滤。

    Returns:
        {"closed": [...], "open": [...]}，各自按时间倒序。
        closed = 已有出场成交的交易；open = 只有入场（当前持仓中）。
    """
    groups: dict[int, list] = {}
    for d in deals or []:
        if magic is not None and getattr(d, "magic", None) != magic:
            continue
        groups.setdefault(int(getattr(d, "position_id", 0) or 0), []).append(d)

    closed: list[dict] = []
    open_list: list[dict] = []
    for pid, ds in groups.items():
        if pid == 0:
            continue  # 入金/出金等非交易记录
        ds.sort(key=lambda d: int(getattr(d, "time", 0)))
        entries = [d for d in ds if int(getattr(d, "entry", -1)) in (DEAL_ENTRY_IN, DEAL_ENTRY_INOUT)]
        exits = [d for d in ds if int(getattr(d, "entry", -1)) in (DEAL_ENTRY_OUT, DEAL_ENTRY_OUT_BY)]
        if not entries:
            continue
        entry = entries[0]
        open_volume = sum(float(d.volume) for d in entries)
        close_volume = sum(float(d.volume) for d in exits)
        profit = sum(float(d.profit) + float(d.commission) + float(d.swap) for d in ds)
        direction = "BUY" if int(getattr(entry, "type", 0)) == DEAL_TYPE_BUY else "SELL"

        close_price = None
        close_time = None
        if exits:
            total_vol = close_volume or 1.0
            close_price = sum(float(d.price) * float(d.volume) for d in exits) / total_vol
            close_time = int(getattr(exits[-1], "time", 0))

        record = {
            "position_id": int(pid),
            "symbol": str(getattr(entry, "symbol", "")),
            "direction": direction,
            "volume": round(open_volume, 2),
            "open_price": float(entry.price),
            "open_time": int(getattr(entry, "time", 0)),
            "closed": bool(exits),
            "close_volume": round(close_volume, 2),
            "close_price": round(close_price, 5) if close_price else None,
            "close_time": close_time,
            "profit": round(profit, 2),
            "comment": str(getattr(entry, "comment", "") or ""),
        }
        if exits:
            closed.append(record)
        else:
            open_list.append(record)

    closed.sort(key=lambda r: r["close_time"] or 0, reverse=True)
    open_list.sort(key=lambda r: r["open_time"], reverse=True)
    return {"closed": closed, "open": open_list}

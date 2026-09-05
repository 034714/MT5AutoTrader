"""
backtest_viz/report.py — 回测统计摘要报告
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .engine import SymbolResult, Trade


def _ts(ts: int) -> str:
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)


class BacktestReport:
    """生成回测统计报告（控制台打印 + JSON 保存）。"""

    def __init__(self, formula: list[int], formula_readable: str = ""):
        self.formula          = formula
        self.formula_readable = formula_readable

    def print_summary(self, results: list[SymbolResult]) -> None:
        """打印所有品种的汇总报告。"""
        line = "─" * 72
        print(f"\n{'═'*72}")
        print(f"  AlphaGPT 回测报告")
        print(f"  公式：{self.formula}")
        if self.formula_readable:
            print(f"  解读：{self.formula_readable}")
        print(f"{'═'*72}")

        all_pnl = 0.0
        for r in results:
            print(f"\n  品种：{r.symbol}")
            print(line)
            print(f"  {'Sortino Ratio':20s}: {r.sortino:+.4f}")
            print(f"  {'总 PnL (log ret)':20s}: {r.total_return:+.6f}")
            pl = r.profit_loss_ratio
            print(f"  {'盈亏比':20s}: {pl:.4f}" if pl is not None else f"  {'盈亏比':20s}: —")
            print(f"  {'总交易笔数':20s}: {r.n_trades}")
            print(f"  {'胜率':20s}: {r.win_rate:.1%}")
            print(f"  {'平均持仓 (bars)':20s}: {r.avg_hold_bars:.1f}")
            all_pnl += r.total_return

            # 盈利最好和最差的前 5 笔
            if r.trades:
                sorted_trades = sorted(r.trades, key=lambda t: t.pnl, reverse=True)
                print(f"\n  Top 5 盈利交易：")
                for t in sorted_trades[:5]:
                    ep = f"{t.exit_price:.5f}" if t.exit_price is not None else "—"
                    et = _ts(t.exit_time) if t.exit_time is not None else "open"
                    print(
                        f"    [{'+' if t.direction==1 else '-'}] "
                        f"进 {_ts(t.entry_time)} @ {t.entry_price:.5f}  "
                        f"出 {et} @ {ep}  PnL={t.pnl:+.6f}"
                    )
                print(f"\n  Top 5 亏损交易：")
                for t in sorted_trades[-5:][::-1]:
                    ep = f"{t.exit_price:.5f}" if t.exit_price is not None else "—"
                    et = _ts(t.exit_time) if t.exit_time is not None else "open"
                    print(
                        f"    [{'+' if t.direction==1 else '-'}] "
                        f"进 {_ts(t.entry_time)} @ {t.entry_price:.5f}  "
                        f"出 {et} @ {ep}  PnL={t.pnl:+.6f}"
                    )

        print(f"\n  {'全品种总 PnL':20s}: {all_pnl:+.6f}")
        print(f"{'═'*72}\n")

    def save_json(
        self,
        results: list[SymbolResult],
        path:    str = "backtest_output/report.json",
    ) -> str:
        """将回测结果保存为 JSON 文件（供外部分析使用）。"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        data = {
            "formula":          self.formula,
            "formula_readable": self.formula_readable,
            "generated_at":     datetime.utcnow().isoformat() + "Z",
            "symbols":          [],
        }

        for r in results:
            trades_data = []
            for t in r.trades:
                trades_data.append({
                    "direction":   "long" if t.direction == 1 else "short",
                    "entry_bar":   t.entry_bar,
                    "entry_time":  _ts(t.entry_time),
                    "entry_price": round(t.entry_price, 6),
                    "exit_bar":    t.exit_bar,
                    "exit_time":   _ts(t.exit_time) if t.exit_time else None,
                    "exit_price":  round(t.exit_price, 6) if t.exit_price else None,
                    "pnl":         round(t.pnl, 8),
                    "cum_pnl":     round(t.cum_pnl, 8),
                })

            data["symbols"].append({
                "symbol":        r.symbol,
                "sortino":       round(r.sortino, 4),
                "total_return":  round(r.total_return, 8),
                "profit_loss_ratio": (
                    round(r.profit_loss_ratio, 4)
                    if r.profit_loss_ratio is not None else None
                ),
                "n_trades":      r.n_trades,
                "win_rate":      round(r.win_rate, 4),
                "avg_hold_bars": round(r.avg_hold_bars, 2),
                "trades":        trades_data,
            })

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"  JSON 报告已保存 → {path}")
        return path

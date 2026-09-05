"""
backtest_viz/chart.py — K 线图 + 出入场标注 + PnL 子图

依赖 matplotlib（见 requirements.txt）。
在 matplotlib 无法显示 GUI 时自动切换到 Agg 后端，直接保存为 PNG/HTML。
"""
from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ── matplotlib 后端自动选择 ─────────────────────────────────────────────
try:
    import matplotlib
    _DISPLAY = os.environ.get("DISPLAY") or os.name == "nt"   # Windows 有 GUI
    if not _DISPLAY:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from matplotlib.ticker import MaxNLocator
    _MPL_OK = True
except ImportError:
    _MPL_OK = False

from .engine import SymbolResult, Trade


def _ts_to_label(ts: int, fmt: str = "%m-%d %H:%M") -> str:
    """Unix 秒 → 可读字符串（UTC）"""
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime(fmt)
    except Exception:
        return str(ts)


class BacktestChart:
    """可视化回测图表生成器。

    为每个品种生成一张包含以下子图的综合图：
      ① K 线图（蜡烛图 + 出入场三角标记 + 仓位背景色）
      ② 因子 / 信号强度折线
      ③ 逐 bar PnL 直方图
      ④ 累计 PnL 曲线

    另可为每笔交易生成独立缩放图（plot_trade_zoom），
    只展示入场前后局部K线，入场/出场价格一目了然。

    用法：
        chart = BacktestChart()
        chart.plot(result, save_path="out/XAUUSD.png")
        chart.plot_all_trade_zooms(result, output_dir="out/")
    """

    # 颜色方案
    _LONG_ENTRY_COLOR  = "#26a69a"   # 绿色：开多
    _SHORT_ENTRY_COLOR = "#ef5350"   # 红色：开空
    _EXIT_COLOR        = "#ffa726"   # 橙色：平仓
    _LONG_BG           = "#e8f5e9"   # 浅绿背景：多头持仓期
    _SHORT_BG          = "#ffebee"   # 浅红背景：空头持仓期

    def __init__(
        self,
        figsize:    tuple[int, int] = (22, 14),
        max_bars:   int             = 120,       # 全局图最多显示的 bar 数，120根更清晰
        dpi:        int             = 120,
    ):
        if not _MPL_OK:
            raise ImportError(
                "matplotlib 未安装。请运行: pip install matplotlib"
            )
        self.figsize  = figsize
        self.max_bars = max_bars
        self.dpi      = dpi

    # ─────────────────────────────────────────────────────────────────────
    # 公开接口
    # ─────────────────────────────────────────────────────────────────────

    def plot(
        self,
        result:    SymbolResult,
        save_path: Optional[str] = None,
        show:      bool          = False,
        title_suffix: str        = "",
    ) -> Optional[str]:
        """为单个品种生成完整图表。

        Args:
            result:       BacktestEngine.run() 返回的 SymbolResult。
            save_path:    保存路径（.png / .svg）；None 则不保存。
            show:         是否调用 plt.show()（交互环境下使用）。
            title_suffix: 附加到标题的额外说明。

        Returns:
            实际保存路径字符串，未保存时返回 None。
        """
        T   = len(result.times)
        # 显示最后 max_bars 个 bar
        start_idx = max(0, T - self.max_bars)
        sl = slice(start_idx, T)

        fig = plt.figure(figsize=self.figsize, dpi=self.dpi)
        gs  = GridSpec(
            4, 1, figure=fig,
            height_ratios=[4, 1.2, 1.2, 1.5],
            hspace=0.08,
        )

        ax_candle = fig.add_subplot(gs[0])
        ax_factor = fig.add_subplot(gs[1], sharex=ax_candle)
        ax_pnl    = fig.add_subplot(gs[2], sharex=ax_candle)
        ax_cum    = fig.add_subplot(gs[3], sharex=ax_candle)

        x = np.arange(T)[sl]

        # ── ① K 线图 ─────────────────────────────────────────────────
        self._draw_candles(ax_candle, result, sl, x)
        self._draw_trade_markers(ax_candle, result, start_idx, T)
        self._draw_position_background(ax_candle, result, start_idx, T, x)
        self._draw_trade_labels(ax_candle, result, start_idx, T)

        ax_candle.set_ylabel("Price", fontsize=9)
        ax_candle.legend(
            handles=self._legend_handles(), loc="upper left", fontsize=8,
            framealpha=0.7,
        )
        ax_candle.grid(alpha=0.3)
        pl_txt = (
            f"{result.profit_loss_ratio:.3f}"
            if result.profit_loss_ratio is not None
            else "—"
        )
        ax_candle.set_title(
            f"{result.symbol}  |  "
            f"Sortino={result.sortino:.2f}  "
            f"TotalRet={result.total_return:.4f}  "
            f"Trades={result.n_trades}  "
            f"WinRate={result.win_rate:.1%}  "
            f"PLRatio={pl_txt}  "
            f"AvgHold={result.avg_hold_bars:.1f}bars"
            + (f"  |  {title_suffix}" if title_suffix else ""),
            fontsize=10, pad=6,
        )

        # ── ② 因子强度 ────────────────────────────────────────────────
        ax_factor.plot(x, result.signal[sl], color="#7e57c2", linewidth=0.8,
                       label="signal (tanh)")
        ax_factor.axhline(0, color="gray", linewidth=0.6, linestyle="--")
        ax_factor.fill_between(x, result.signal[sl], 0,
                               where=result.signal[sl] > 0,
                               alpha=0.15, color=self._LONG_ENTRY_COLOR)
        ax_factor.fill_between(x, result.signal[sl], 0,
                               where=result.signal[sl] < 0,
                               alpha=0.15, color=self._SHORT_ENTRY_COLOR)
        ax_factor.set_ylabel("Signal", fontsize=8)
        ax_factor.legend(fontsize=7, loc="upper left")
        ax_factor.grid(alpha=0.25)

        # ── ③ 逐 bar PnL 直方图 ──────────────────────────────────────
        pnl_sl = result.pnl[sl]
        colors_bar = [
            self._LONG_ENTRY_COLOR if v >= 0 else self._SHORT_ENTRY_COLOR
            for v in pnl_sl
        ]
        ax_pnl.bar(x, pnl_sl, color=colors_bar, width=0.8, alpha=0.7)
        ax_pnl.axhline(0, color="gray", linewidth=0.6)
        ax_pnl.set_ylabel("Bar PnL", fontsize=8)
        ax_pnl.grid(alpha=0.25)

        # ── ④ 累计 PnL 曲线 ──────────────────────────────────────────
        cum_sl = result.cum_pnl[sl]
        ax_cum.plot(x, cum_sl, color="#1565c0", linewidth=1.2, label="Cum PnL")
        ax_cum.fill_between(x, cum_sl, 0,
                            where=cum_sl >= 0, alpha=0.12,
                            color=self._LONG_ENTRY_COLOR)
        ax_cum.fill_between(x, cum_sl, 0,
                            where=cum_sl < 0, alpha=0.12,
                            color=self._SHORT_ENTRY_COLOR)
        ax_cum.axhline(0, color="gray", linewidth=0.6)
        ax_cum.set_ylabel("Cum PnL", fontsize=8)
        ax_cum.legend(fontsize=7, loc="upper left")
        ax_cum.grid(alpha=0.25)

        # ── X 轴刻度（时间标签）─────────────────────────────────────
        self._set_time_ticks(ax_cum, result.times[sl], x, n_ticks=10)
        plt.setp(ax_candle.get_xticklabels(), visible=False)
        plt.setp(ax_factor.get_xticklabels(), visible=False)
        plt.setp(ax_pnl.get_xticklabels(), visible=False)
        ax_cum.tick_params(axis="x", labelsize=7, rotation=30)

        # ── 保存 / 展示 ──────────────────────────────────────────────
        saved_path: Optional[str] = None
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
            saved_path = save_path
            print(f"  图表已保存 → {save_path}")

        if show:
            plt.show()

        plt.close(fig)
        return saved_path

    def plot_all(
        self,
        results:    list[SymbolResult],
        output_dir: str  = "backtest_output",
        show:       bool = False,
        title_suffix: str = "",
    ) -> list[str]:
        """为所有品种批量生成图表。

        Returns:
            已保存的文件路径列表。
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved = []
        for r in results:
            path = str(Path(output_dir) / f"{r.symbol}.png")
            p = self.plot(r, save_path=path, show=show,
                          title_suffix=title_suffix)
            if p:
                saved.append(p)
        return saved

    def plot_trade_zoom(
        self,
        result:       SymbolResult,
        trade_idx:    int,
        pre_bars:     int            = 20,   # 入场前显示的K线数
        post_bars:    int            = 10,   # 出场后显示的K线数
        save_path:    Optional[str]  = None,
        show:         bool           = False,
    ) -> Optional[str]:
        """为单笔交易生成局部缩放K线图。

        只显示入场前 pre_bars 根 + 持仓期 + 出场后 post_bars 根，
        清晰标注入场/出场价格水平线和具体价格数值。

        Args:
            result:    SymbolResult 回测结果。
            trade_idx: result.trades 中的交易索引（0-based）。
            pre_bars:  入场前显示的K线数量，默认20根。
            post_bars: 出场后显示的K线数量，默认10根。
            save_path: 保存路径；None 则不保存。
            show:      是否调用 plt.show()。

        Returns:
            实际保存路径，未保存时返回 None。
        """
        if trade_idx >= len(result.trades):
            raise IndexError(
                f"trade_idx={trade_idx} 超出范围（共 {len(result.trades)} 笔交易）"
            )

        trade = result.trades[trade_idx]
        T     = len(result.times)

        # 实际成交在信号 bar 的下一根（与 target_ret 时间对齐）
        entry_exec_bar = min(trade.entry_bar + 1, T - 1)
        exit_exec_bar  = min(trade.exit_bar + 1, T - 1) if trade.exit_bar is not None else None

        # 计算显示窗口（围绕实际成交 bar 展开）
        win_start = max(0, entry_exec_bar - pre_bars)
        ref_end   = exit_exec_bar if exit_exec_bar is not None else entry_exec_bar
        win_end   = min(T, ref_end + post_bars + 1)
        sl  = slice(win_start, win_end)
        x   = np.arange(win_end - win_start)

        # 入场/出场在 x 轴的位置（用实际成交 bar）
        entry_xi = entry_exec_bar - win_start
        exit_xi  = (exit_exec_bar - win_start) if exit_exec_bar is not None else None

        fig, (ax_candle, ax_cum) = plt.subplots(
            2, 1, figsize=(14, 8), dpi=self.dpi,
            gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        )

        # ── K 线图 ───────────────────────────────────────────────────
        self._draw_candles(ax_candle, result, sl, x)
        self._draw_position_background(ax_candle, result, win_start, win_end, x)

        # 入场标记（三角）
        lo = result.low[sl]
        hi = result.high[sl]
        cl = result.close[sl]
        offset = (hi.max() - lo.min()) * 0.008

        if trade.direction == 1:
            ax_candle.plot(
                entry_xi, lo[entry_xi] - offset,
                marker="^", color=self._LONG_ENTRY_COLOR,
                markersize=14, zorder=6, markeredgewidth=1,
                markeredgecolor="white",
            )
        else:
            ax_candle.plot(
                entry_xi, hi[entry_xi] + offset,
                marker="v", color=self._SHORT_ENTRY_COLOR,
                markersize=14, zorder=6, markeredgewidth=1,
                markeredgecolor="white",
            )

        # 出场标记（菱形）
        if exit_xi is not None:
            ax_candle.plot(
                exit_xi, cl[exit_xi],
                marker="D", color=self._EXIT_COLOR,
                markersize=10, zorder=6, markeredgewidth=1,
                markeredgecolor="white",
            )

        # 入场价格水平虚线
        entry_price = trade.entry_price
        ax_candle.axhline(
            entry_price, color=self._LONG_ENTRY_COLOR if trade.direction == 1
            else self._SHORT_ENTRY_COLOR,
            linewidth=1.2, linestyle="--", alpha=0.8, zorder=4,
        )
        ax_candle.annotate(
            f"Entry  {entry_price:.5f}",
            xy=(x[-1], entry_price),
            xytext=(-4, 4), textcoords="offset points",
            fontsize=8, color=self._LONG_ENTRY_COLOR if trade.direction == 1
            else self._SHORT_ENTRY_COLOR,
            ha="right", fontweight="bold",
        )

        # 出场价格水平虚线
        if trade.exit_price is not None and exit_xi is not None:
            ax_candle.axhline(
                trade.exit_price, color=self._EXIT_COLOR,
                linewidth=1.2, linestyle="--", alpha=0.8, zorder=4,
            )
            ax_candle.annotate(
                f"Exit  {trade.exit_price:.5f}",
                xy=(x[-1], trade.exit_price),
                xytext=(-4, -8), textcoords="offset points",
                fontsize=8, color=self._EXIT_COLOR,
                ha="right", fontweight="bold",
            )
            # 入场→出场垂直价差连线
            ax_candle.annotate(
                "",
                xy=(exit_xi, trade.exit_price),
                xytext=(entry_xi, entry_price),
                arrowprops=dict(
                    arrowstyle="->",
                    color="#9e9e9e",
                    lw=1.2,
                    connectionstyle="arc3,rad=0.15",
                ),
                zorder=3,
            )

        # PnL 标注框
        pnl_color = "#1b5e20" if trade.pnl > 0 else "#b71c1c"
        direction_str = "Long ▲" if trade.direction == 1 else "Short ▼"
        hold_bars = (trade.exit_bar - trade.entry_bar) if trade.exit_bar is not None else 0
        entry_time_str = _ts_to_label(trade.entry_time, "%Y-%m-%d %H:%M")
        exit_time_str  = (
            _ts_to_label(trade.exit_time, "%Y-%m-%d %H:%M")
            if trade.exit_time else "open"
        )
        exit_price_str = f"{trade.exit_price:.5f}" if trade.exit_price else "-"
        info_text = (
            f"{direction_str}  PnL: {trade.pnl:+.5f}\n"
            f"Entry: {entry_time_str} @ {entry_price:.5f}\n"
            f"Exit : {exit_time_str} @ {exit_price_str}\n"
            f"Hold : {hold_bars} bars"
        )
        ax_candle.text(
            0.01, 0.97, info_text,
            transform=ax_candle.transAxes,
            fontsize=8.5, verticalalignment="top",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white", alpha=0.85,
                edgecolor=pnl_color, linewidth=1.5,
            ),
            color=pnl_color,
        )

        # 标题
        trade_no = trade_idx + 1
        ax_candle.set_title(
            f"{result.symbol}  Trade {trade_no}/{result.n_trades}  "
            f"{'Long' if trade.direction == 1 else 'Short'}  "
            f"PnL={trade.pnl:+.5f}  WinRate={result.win_rate:.1%}",
            fontsize=10, pad=6,
        )
        ax_candle.set_ylabel("Price", fontsize=9)
        ax_candle.grid(alpha=0.3)
        ax_candle.legend(
            handles=self._legend_handles(), loc="upper right",
            fontsize=7, framealpha=0.7,
        )

        # ── 累计 PnL 曲线（全局，标注当前交易位置）─────────────────
        cum_all = result.cum_pnl
        x_all   = np.arange(T)
        ax_cum.plot(x_all, cum_all, color="#1565c0", linewidth=1.0, label="Cum PnL")
        ax_cum.fill_between(
            x_all, cum_all, 0,
            where=cum_all >= 0, alpha=0.10, color=self._LONG_ENTRY_COLOR,
        )
        ax_cum.fill_between(
            x_all, cum_all, 0,
            where=cum_all < 0, alpha=0.10, color=self._SHORT_ENTRY_COLOR,
        )
        # 标注当前交易在全局 PnL 上的位置
        if trade.exit_bar is not None:
            ax_cum.axvspan(
                trade.entry_bar, trade.exit_bar,
                alpha=0.25,
                color=self._LONG_BG if trade.direction == 1 else self._SHORT_BG,
            )
        ax_cum.axhline(0, color="gray", linewidth=0.6)
        ax_cum.set_ylabel("Cum PnL", fontsize=8)
        ax_cum.legend(fontsize=7, loc="upper left")
        ax_cum.grid(alpha=0.25)
        self._set_time_ticks(ax_cum, result.times, x_all, n_ticks=8)
        ax_cum.tick_params(axis="x", labelsize=7, rotation=30)

        # X 轴时间刻度（局部 K 线图）
        self._set_time_ticks(ax_candle, result.times[sl], x, n_ticks=6)
        ax_candle.tick_params(axis="x", labelsize=7, rotation=20)

        # 保存 / 展示
        saved_path: Optional[str] = None
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, bbox_inches="tight")
            saved_path = save_path
            print(f"  缩放图已保存 → {save_path}")

        if show:
            plt.show()

        plt.close(fig)
        return saved_path

    def plot_all_trade_zooms(
        self,
        result:       SymbolResult,
        output_dir:   str  = "backtest_output",
        pre_bars:     int  = 20,
        post_bars:    int  = 10,
        max_trades:   int  = 30,    # 最多生成多少张缩放图（避免文件爆炸）
        show:         bool = False,
    ) -> list[str]:
        """为一个品种的所有交易批量生成缩放图。

        Args:
            result:     SymbolResult 回测结果。
            output_dir: 输出目录。
            pre_bars:   入场前显示K线数量。
            post_bars:  出场后显示K线数量。
            max_trades: 最多生成张数（按 |PnL| 降序选 top-N）。
            show:       是否调用 plt.show()。

        Returns:
            已保存的文件路径列表。
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        saved = []

        trades = result.trades
        if not trades:
            print(f"  {result.symbol} 无交易记录，跳过缩放图生成。")
            return saved

        # 选取最重要的 max_trades 笔（按 |PnL| 降序）
        indexed = sorted(
            enumerate(trades),
            key=lambda kv: abs(kv[1].pnl),
            reverse=True,
        )[:max_trades]
        # 按时间顺序重新排序，编号更直观
        indexed.sort(key=lambda kv: kv[1].entry_bar)

        for rank, (orig_idx, _) in enumerate(indexed, start=1):
            fname = f"{result.symbol}_trade{rank}_zoom.png"
            path  = str(Path(output_dir) / fname)
            p = self.plot_trade_zoom(
                result, orig_idx,
                pre_bars=pre_bars, post_bars=post_bars,
                save_path=path, show=show,
            )
            if p:
                saved.append(p)

        return saved

    # ─────────────────────────────────────────────────────────────────────
    # 内部绘图方法
    # ─────────────────────────────────────────────────────────────────────

    def _draw_candles(
        self,
        ax:     "plt.Axes",
        result: SymbolResult,
        sl:     slice,
        x:      np.ndarray,
    ) -> None:
        """绘制蜡烛图（用矩形 + 线段模拟，不依赖 mplfinance）"""
        op = result.open[sl]
        hi = result.high[sl]
        lo = result.low[sl]
        cl = result.close[sl]

        bar_w = 0.6
        for i, xi in enumerate(x):
            is_bull = cl[i] >= op[i]
            color   = self._LONG_ENTRY_COLOR if is_bull else self._SHORT_ENTRY_COLOR
            # 上下影线
            ax.plot([xi, xi], [lo[i], hi[i]], color=color, linewidth=0.7)
            # 实体
            body_lo = min(op[i], cl[i])
            body_hi = max(op[i], cl[i])
            body_h  = max(body_hi - body_lo, (hi[i] - lo[i]) * 0.01)
            rect = mpatches.Rectangle(
                (xi - bar_w / 2, body_lo), bar_w, body_h,
                linewidth=0, facecolor=color, alpha=0.85,
            )
            ax.add_patch(rect)

        ax.set_xlim(x[0] - 1, x[-1] + 1)
        ax.set_ylim(result.low[sl].min() * 0.9995,
                    result.high[sl].max() * 1.0005)

    def _draw_position_background(
        self,
        ax:        "plt.Axes",
        result:    SymbolResult,
        start_idx: int,
        T:         int,
        x:         np.ndarray,
    ) -> None:
        """在持仓期间填充背景色（多头绿/空头红）。

        背景从实际成交 bar 开始（信号 bar + 1）。
        """
        for trade in result.trades:
            entry_exec = min(trade.entry_bar + 1, T - 1)
            exit_exec  = min(trade.exit_bar + 1, T - 1) if trade.exit_bar is not None else T - 1

            # 转换为 x 轴坐标（相对于 start_idx）
            x_entry = max(entry_exec - start_idx, 0)
            x_exit  = min(exit_exec  - start_idx, len(x) - 1)

            if x_entry >= len(x) or x_exit < 0:
                continue

            color = self._LONG_BG if trade.direction == 1 else self._SHORT_BG
            ax.axvspan(x[x_entry], x[x_exit], alpha=0.25, color=color, linewidth=0)

    def _draw_trade_markers(
        self,
        ax:        "plt.Axes",
        result:    SymbolResult,
        start_idx: int,
        T:         int,
    ) -> None:
        """绘制开平仓三角标记。

        标记打在实际成交 bar（信号 bar + 1），与 entry_price/exit_price 对齐。

        多头开仓：绿色向上三角（▲），标注在 low 下方
        空头开仓：红色向下三角（▼），标注在 high 上方
        平仓/反手：橙色菱形（◆），标注在 close 附近
        """
        lo = result.low
        hi = result.high
        cl = result.close
        total = T

        for trade in result.trades:
            # 实际成交 bar = 信号 bar + 1
            eb = min(trade.entry_bar + 1, total - 1)
            xb = min(trade.exit_bar + 1, total - 1) if trade.exit_bar is not None else None

            if eb >= start_idx:
                xi = eb - start_idx
                offset = (hi[eb] - lo[eb]) * 0.3 + (hi[eb] - lo[eb]) * 0.05
                if trade.direction == 1:
                    ax.plot(xi, lo[eb] - offset,
                            marker="^", color=self._LONG_ENTRY_COLOR,
                            markersize=8, zorder=5, markeredgewidth=0.5,
                            markeredgecolor="white")
                else:
                    ax.plot(xi, hi[eb] + offset,
                            marker="v", color=self._SHORT_ENTRY_COLOR,
                            markersize=8, zorder=5, markeredgewidth=0.5,
                            markeredgecolor="white")

            if xb is not None and xb >= start_idx:
                xi = xb - start_idx
                ax.plot(xi, cl[xb],
                        marker="D", color=self._EXIT_COLOR,
                        markersize=6, zorder=5, markeredgewidth=0.5,
                        markeredgecolor="white")

    def _draw_trade_labels(
        self,
        ax:        "plt.Axes",
        result:    SymbolResult,
        start_idx: int,
        T:         int,
    ) -> None:
        """在每笔交易标注 PnL 数值（仅盈亏超过阈值时显示，避免文字过密）。"""
        hi = result.high
        lo = result.low
        price_range = hi.max() - lo.min()
        threshold   = price_range * 0.001

        visible = [
            t for t in result.trades
            if abs(t.pnl) > threshold
            and t.exit_bar is not None
            and min(t.exit_bar + 1, T - 1) >= start_idx
        ]
        if len(visible) > 20:
            visible = sorted(visible, key=lambda t: abs(t.pnl), reverse=True)[:20]

        for trade in visible:
            if trade.exit_bar is None:
                continue
            # 标注在实际成交（出场）的 bar 上
            xb = min(trade.exit_bar + 1, T - 1)
            xi    = xb - start_idx
            price = result.close[xb]
            label = f"{trade.pnl:+.4f}"
            color = "#1b5e20" if trade.pnl > 0 else "#b71c1c"
            ax.annotate(
                label,
                xy=(xi, price),
                xytext=(0, 12 if trade.pnl > 0 else -16),
                textcoords="offset points",
                fontsize=6, color=color,
                ha="center",
                bbox=dict(
                    boxstyle="round,pad=0.1", fc="white",
                    alpha=0.6, edgecolor="none",
                ),
            )

    @staticmethod
    def _set_time_ticks(
        ax:      "plt.Axes",
        times:   np.ndarray,
        x:       np.ndarray,
        n_ticks: int = 10,
    ) -> None:
        """设置 X 轴时间刻度标签。"""
        step  = max(1, len(x) // n_ticks)
        ticks = x[::step]
        labels = [_ts_to_label(int(times[i]), fmt="%y-%m-%d\n%H:%M")
                  for i in range(0, len(times), step)]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels[:len(ticks)])

    def _legend_handles(self) -> list:
        """构建图例 handles"""
        return [
            mpatches.Patch(color=self._LONG_ENTRY_COLOR, label="▲ Long entry"),
            mpatches.Patch(color=self._SHORT_ENTRY_COLOR, label="▼ Short entry"),
            mpatches.Patch(color=self._EXIT_COLOR,        label="◆ Exit"),
            mpatches.Patch(color=self._LONG_BG,  alpha=0.4, label="Long position"),
            mpatches.Patch(color=self._SHORT_BG, alpha=0.4, label="Short position"),
        ]

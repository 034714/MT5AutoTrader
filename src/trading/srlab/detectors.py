# -*- coding: utf-8 -*-
"""
检测器实现（内嵌自上游 srlab/detectors.py，保留 V2/V3 融合算法，
去掉 scipy 依赖的 V1 基线与对照/消融变体——那些是研究工具，线上不需要）。

V3 的每一处改动都对应上游 scripts/factor_analysis.py 在 test 集 53,823 个
关键位上的实测结论（不是拍脑袋调参）：

 证据          实测结论                                        V3 的处置
 ------------  ----------------------------------------------  ----------------------
 m_n_events    support hold 69.6%->76.8% (p<1e-7)，且在**每个    升为主证据，
               距离档内**都单调；唯一 hold 与 fwd_ret 同向上升    用 log1p(n) 直接入模
               的证据（远档 fwd_ret 0.67%->1.46%）
 m_vp          hold +6.8pp 但 fwd_ret **-0.91pp**              改为只进 p_stall，
 m_close       hold +6.6pp 但 fwd_ret **-0.84pp**              不进 edge 排序
               => 重仓筹码区= "价格会停"，不等于"反弹能赚"
 m_recency     resistance 越新鲜越弱: 84.2%->70.5% (p<1e-34)     **符号反转**，
               V2 里 w_recency=+0.35，方向完全错了                改用 stale=1-recency
 m_pivot       support p=0.069 不显著，fwd_ret spread 为负；      权重从 1.6 降到 0.3
               消融显示去掉它 hold 反而 +1.3pp                    （定位仍保留）
 m_round       support p=0.70；resistance 显著为**负**(p=2e-4)   直接删除
 width_atr     hold +10.3pp 而 fwd_ret -0.39pp                  区宽固定，不参与排序
 dist_atr      resistance hold 68.4%->84.9% (+16.5pp)           不进排序（否则只是
               距离是最强主效应                                  在复现 placebo）

最重要的结构性改动：把一个分数拆成两个
    p_stall  —— 价格在此停下的倾向（vp/close/宽度/距离驱动）
                用途：放止损、设目标位
    edge     —— 在此进场的期望收益（n_events/stale/低vp 驱动）
                用途：排序、选信号
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .base import Ctx, Level, annotate, nms
from .metrics import wilson
from .profile import (
    bin_centers, decay_weights, find_peaks_simple, gaussian_blur,
    make_bins, point_profile, robust_z, volume_profile,
)


# ============================================================
# V2: 多证据融合（V3 的基类）
# ============================================================
@dataclass
class V2Params:
    # --- 窗口 ---
    lookback: int = 250          # 成交量剖面回看长度
    hist_lookback: int = 750     # 历史触及统计回看长度
    pivot_lookback: int = 500    # 枢轴回看长度
    # --- 分箱 ---
    bin_atr: float = 0.25        # 箱宽 = bin_atr * ATR（下限 tick）
    # --- 时间衰减 ---
    half_life: float = 60.0      # 成交量剖面半衰期（bar）
    pivot_half_life: float = 120.0
    # --- 证据核宽 ---
    close_kernel_atr: float = 0.30
    pivot_kernel_atr: float = 0.35
    # --- 融合权重 ---
    w_vp: float = 1.0            # 成交量剖面
    w_close: float = 0.8         # 收盘价堆积
    w_pivot: float = 1.6         # 枢轴聚集（V3 中仅用于"定位候选"，不参与排序）
    w_hold: float = 1.0          # 历史守住率（Wilson 下界）
    w_round: float = 0.25        # 整数关口（实测无信息，V3 已弃用）
    w_recency: float = 0.35      # 近期相关性（实测方向相反，V3 改用 stale）
    # --- 区宽 ---
    width_min_atr: float = 0.30
    width_max_atr: float = 1.20
    # --- 输出 ---
    nms_gap_atr: float = 0.90
    max_out: int = 10            # 上游多给，由 walkforward 统一按侧限流
    max_dist_atr: float = 6.0    # 超过此距离的关键位不输出（交易上无意义）
    prefilter_k: int = 8         # 每侧先按候选面得分保留 k 个，再算昂贵的历史统计
    # --- 历史触及统计 ---
    touch_gap: int = 5           # 事件去重冷却期（bar）
    touch_hold_bars: int = 10
    touch_break_atr: float = 0.5
    touch_target_atr: float = 1.0
    min_touch_events: int = 2


@dataclass
class V2Fusion:
    p: V2Params = field(default_factory=V2Params)
    name: str = "V2_fusion"
    calibrator: Optional[object] = None   # 可选：外部校准器

    # ---------- 主流程 ----------
    def detect(self, ctx: Ctx) -> List[Level]:
        p = self.p
        atr = max(ctx.atr, 1e-12)
        price = ctx.price
        w = min(p.lookback, ctx.n)
        if w < 30:
            return []

        h, l, c, v = ctx.high[-w:], ctx.low[-w:], ctx.close[-w:], ctx.volume[-w:]

        # --- 分箱：ATR 尺度，跨标的可比 ---
        bin_w = max(ctx.tick, p.bin_atr * atr)
        lo_px = float(l.min()) - bin_w
        hi_px = float(h.max()) + bin_w
        p0, bin_w, nb = make_bins(lo_px, hi_px, bin_w)
        if nb < 8:
            return []
        px = bin_centers(p0, bin_w, nb)

        # --- 证据 1: 体积守恒的成交量剖面 + 时间衰减 ---
        dw = decay_weights(w, p.half_life)
        vp = volume_profile(h, l, v * dw, p0, bin_w, nb)
        vp = gaussian_blur(vp, max(0.5, p.close_kernel_atr * atr / bin_w))
        vp_z = robust_z(vp)

        # --- 证据 2: 收盘价堆积（市场"认可"的价格，比全区间更聚焦）---
        cp = point_profile(c, v * dw, p0, bin_w, nb,
                           kernel_bins=max(0.5, p.close_kernel_atr * atr / bin_w))
        cp_z = robust_z(cp)

        # --- 证据 3: 枢轴聚集（真正发生过反转的价格）---
        pv_sup, pv_res = self._pivot_profiles(ctx, p0, bin_w, nb, atr)

        # --- 组合出候选面（支撑面/阻力面分开：枢轴有方向）---
        base = p.w_vp * vp_z + p.w_close * cp_z
        surf_sup = base + p.w_pivot * pv_sup
        surf_res = base + p.w_pivot * pv_res

        min_d = max(1, int(round(p.nms_gap_atr * atr / bin_w)))
        cands: List[Tuple[int, str, float]] = []
        for surf, kind in ((surf_sup, "support"), (surf_res, "resistance")):
            pk = find_peaks_simple(surf, min_distance=min_d)
            side: List[Tuple[int, str, float]] = []
            for i in pk:
                center = float(px[i])
                # 方向过滤：支撑必须在现价下方，阻力必须在上方
                if kind == "support" and center >= price:
                    continue
                if kind == "resistance" and center <= price:
                    continue
                if abs(center - price) / atr > p.max_dist_atr:
                    continue
                side.append((int(i), kind, float(surf[i])))
            # 预筛：只对候选面得分最高的 k 个算昂贵的历史触及统计
            side.sort(key=lambda x: -x[2])
            cands.extend(side[:p.prefilter_k])

        if not cands:
            return []

        # --- 逐候选：定区宽 -> 历史守住率 -> 整数关口 -> 近期相关性 -> 打分 ---
        out: List[Level] = []
        hist_lo = max(0, ctx.n - p.hist_lookback)
        for i, kind, sval in cands:
            center = float(px[i])
            surf = surf_sup if kind == "support" else surf_res
            zlo, zhi = self._zone_bounds(surf, px, i, atr, bin_w)

            hold_lb, n_ev, n_hold = self._touch_stats(
                ctx, zlo, zhi, kind, hist_lo)
            round_b = self._round_bonus(center, atr)
            recency = self._recency(ctx, zlo, zhi, hist_lo)

            score = (sval
                     + p.w_hold * hold_lb
                     + p.w_round * round_b
                     + p.w_recency * recency)

            out.append(Level(
                center=center, low=zlo, high=zhi,
                score=float(score), kind=kind,
                meta={
                    "surf": float(sval),
                    "vp": float(vp_z[i]),
                    "close": float(cp_z[i]),
                    "pivot": float((pv_sup if kind == "support" else pv_res)[i]),
                    "hold_lb": float(hold_lb),
                    "n_events": float(n_ev),
                    "n_hold": float(n_hold),
                    "round": float(round_b),
                    "recency": float(recency),
                },
            ))

        out = nms(out, p.nms_gap_atr, atr, p.max_out)
        out = annotate(out, price, atr)

        if self.calibrator is not None:
            feats = np.array([z.score for z in out], dtype=np.float64)
            probs = self.calibrator.predict(feats)
            for z, pr in zip(out, probs):
                z.confidence = float(pr) if np.isfinite(pr) else 0.0
        return out

    # ---------- 证据 3: 枢轴剖面 ----------
    def _pivot_profiles(self, ctx: Ctx, p0: float, bin_w: float, nb: int,
                        atr: float) -> Tuple[np.ndarray, np.ndarray]:
        p = self.p
        zeros = np.zeros(nb, dtype=np.float64)
        pvv = ctx.pivots
        if pvv is None or len(pvv) == 0:
            return zeros, zeros
        keep = pvv.idx >= (ctx.n - 1 - p.pivot_lookback)
        idx, pxs, kd, leg = pvv.idx[keep], pvv.price[keep], pvv.kind[keep], pvv.leg_atr[keep]
        if len(idx) == 0:
            return zeros, zeros

        age = (ctx.n - 1) - idx
        rec = np.exp(-np.log(2.0) * age / max(p.pivot_half_life, 1e-9))
        # 显著性：形成该枢轴的行程越大，越值得当关键位
        sig = np.clip(leg, 0.5, 8.0)
        wts = rec * sig

        kb = max(0.5, p.pivot_kernel_atr * atr / bin_w)
        lo_m, hi_m = kd == -1, kd == 1
        sup = point_profile(pxs[lo_m], wts[lo_m], p0, bin_w, nb, kernel_bins=kb)
        res = point_profile(pxs[hi_m], wts[hi_m], p0, bin_w, nb, kernel_bins=kb)
        # 前低同时是潜在支撑，前高同时是潜在阻力；但破位后角色互换，
        # 因此两个面都保留对方 40% 的贡献（role reversal）
        return robust_z(sup + 0.4 * res), robust_z(res + 0.4 * sup)

    # ---------- 区宽：由证据的局部离散度决定，并用 ATR 夹逼 ----------
    def _zone_bounds(self, surf: np.ndarray, px: np.ndarray, i: int,
                     atr: float, bin_w: float) -> Tuple[float, float]:
        p = self.p
        nb = len(surf)
        thr = surf[i] - 0.5 * (surf[i] - np.median(surf))  # 半高（相对中位数）
        li = i
        while li > 0 and surf[li - 1] >= thr:
            li -= 1
        ri = i
        while ri < nb - 1 and surf[ri + 1] >= thr:
            ri += 1
        half = max((px[ri] - px[li]) / 2.0, 0.0)
        half = float(np.clip(half, p.width_min_atr * atr / 2, p.width_max_atr * atr / 2))
        c = float(px[i])
        return c - half, c + half

    # ---------- 证据 4: 历史触及统计（因果，事件去重，ATR 阈值）----------
    def _touch_stats(self, ctx: Ctx, zlo: float, zhi: float, kind: str,
                     hist_lo: int) -> Tuple[float, int, int]:
        """
        返回 (Wilson 下界守住率, 事件数, 守住数)

          - 事件去重：同一次进入区间只算一次，冷却 touch_gap 根
          - 方向校验：支撑要求上一根收盘在区上方（自上而下测试）
          - 阈值 ATR 化：突破 = 收盘越过区外 break_atr*ATR(当时)
          - 样本不足时返回 Wilson 下界（自动惩罚"只测过 1 次"的位）
        """
        p = self.p
        hi_end = ctx.n - 1
        if hi_end - hist_lo < 30:
            return (0.0, 0, 0)

        h = ctx.high[hist_lo:hi_end]      # 不含当前 bar
        l = ctx.low[hist_lo:hi_end]
        c = ctx.close[hist_lo:hi_end]
        n = len(c)
        if n < 20:
            return (0.0, 0, 0)

        inz = (l <= zhi) & (h >= zlo)
        if not inz.any():
            return (0.0, 0, 0)

        prev_c = np.concatenate([[c[0]], c[:-1]])
        if kind == "support":
            valid = inz & (prev_c > zhi)
        else:
            valid = inz & (prev_c < zlo)
        cand = np.nonzero(valid)[0]
        if len(cand) == 0:
            return (0.0, 0, 0)

        # 事件去重
        events = []
        last = -10 ** 9
        for j in cand:
            if j - last >= p.touch_gap:
                events.append(int(j))
                last = int(j)

        # ATR 序列（因果：用当时的 ATR）
        atr_hist = ctx_atr_slice(ctx, hist_lo, hi_end)

        hold = brk = 0
        for j in events:
            e2 = min(j + p.touch_hold_bars, n - 1)
            if e2 <= j:
                continue
            a = max(atr_hist[j], 1e-12)
            cc, hh, ll = c[j:e2 + 1], h[j:e2 + 1], l[j:e2 + 1]
            if kind == "support":
                bi = np.nonzero(cc < zlo - p.touch_break_atr * a)[0]
                gi = np.nonzero(hh >= zhi + p.touch_target_atr * a)[0]
            else:
                bi = np.nonzero(cc > zhi + p.touch_break_atr * a)[0]
                gi = np.nonzero(ll <= zlo - p.touch_target_atr * a)[0]
            b = int(bi[0]) if len(bi) else 10 ** 9
            g = int(gi[0]) if len(gi) else 10 ** 9
            if b == g == 10 ** 9:
                continue
            if g <= b:
                hold += 1
            else:
                brk += 1

        tot = hold + brk
        if tot < p.min_touch_events:
            # 样本不足：给一个保守的中性下界，而不是 0 或虚高比例
            return (0.25, tot, hold)
        _, lb, _ = wilson(hold, tot)
        return (float(lb), tot, hold)

    # ---------- 证据 6: 近期相关性 ----------
    def _recency(self, ctx: Ctx, zlo: float, zhi: float, hist_lo: int) -> float:
        h = ctx.high[hist_lo:]
        l = ctx.low[hist_lo:]
        inz = (l <= zhi) & (h >= zlo)
        w = np.nonzero(inz)[0]
        if len(w) == 0:
            return 0.0
        age = len(inz) - 1 - int(w[-1])
        return float(math.exp(-math.log(2.0) * age / 120.0))


def ctx_atr_slice(ctx: Ctx, lo: int, hi: int) -> np.ndarray:
    """
    历史 ATR 切片。优先用预计算好的因果 ATR 序列，
    缺失时才现算（仍只用 <= t 的数据，保持因果）。
    """
    if ctx.atr_arr is not None:
        return ctx.atr_arr[lo:hi]
    from .srdata import atr_series
    arr = getattr(ctx, "_atr_cache", None)
    if arr is None:
        arr = atr_series(ctx.high, ctx.low, ctx.close, 14)
        setattr(ctx, "_atr_cache", arr)
    return arr[lo:hi]


# ============================================================
# V3: 由单因子分析驱动的重设计
# ============================================================
@dataclass
class V3Params(V2Params):
    """
    排序权重的最终取值来自上游 scripts/score_search.py 的候选评分搜索
    （在 train/test 两个互斥股票池上各 ~38,000 个已触及关键位）。
    结论：s1（事件数 + 陈旧度）两个因子达到甚至超过 5 因子版本，
    且 train/test 高度一致，因此砍到 2 个因子；
    成交量剖面守住率差最大但收益率差为负 —— 移出排序，作为 p_stall 单独输出。
    """
    # 排序用（edge 方向）—— 只保留两个实测稳定的因子
    w_events: float = 1.0        # log1p(历史触及事件数)
    w_stale: float = 0.8         # (1 - recency)：越久没被碰过越强（实测方向与直觉相反）
    w_hold_wilson: float = 0.0   # train/test 符号翻转，归零
    w_vp_edge: float = 0.0       # 移出排序，改由 p_stall 输出
    w_pivot_edge: float = 0.0    # support 上无信息，归零
    min_events_for_wilson: int = 3
    # p_stall 方向（不参与排序，仅输出给用户放止损/目标）
    s_vp: float = 1.0
    s_close: float = 0.8
    s_width: float = 0.3
    # 候选补齐
    fill_quota_per_side: int = 3


@dataclass
class V3Fusion(V2Fusion):
    p: V3Params = field(default_factory=V3Params)
    name: str = "V3_fusion"

    def detect(self, ctx: Ctx) -> List[Level]:
        p = self.p
        atr = max(ctx.atr, 1e-12)
        price = ctx.price
        w = min(p.lookback, ctx.n)
        if w < 30:
            return []

        h, l, c, v = ctx.high[-w:], ctx.low[-w:], ctx.close[-w:], ctx.volume[-w:]
        bin_w = max(ctx.tick, p.bin_atr * atr)
        p0, bin_w, nb = make_bins(float(l.min()) - bin_w, float(h.max()) + bin_w, bin_w)
        if nb < 8:
            return []
        px = bin_centers(p0, bin_w, nb)

        dw = decay_weights(w, p.half_life)
        vp = robust_z(gaussian_blur(volume_profile(h, l, v * dw, p0, bin_w, nb),
                                    max(0.5, p.close_kernel_atr * atr / bin_w)))
        cp = robust_z(point_profile(c, v * dw, p0, bin_w, nb,
                                    kernel_bins=max(0.5, p.close_kernel_atr * atr / bin_w)))
        pv_sup, pv_res = self._pivot_profiles(ctx, p0, bin_w, nb, atr)

        # 候选面：用于**定位**（哪里可能是关键位），不用于最终排序
        loc_sup = p.w_vp * vp + p.w_close * cp + p.w_pivot * pv_sup
        loc_res = p.w_vp * vp + p.w_close * cp + p.w_pivot * pv_res

        min_d = max(1, int(round(p.nms_gap_atr * atr / bin_w)))
        hist_lo = max(0, ctx.n - p.hist_lookback)
        out: List[Level] = []

        for loc, pvz, kind in ((loc_sup, pv_sup, "support"),
                              (loc_res, pv_res, "resistance")):
            idxs = self._candidates(loc, px, price, kind, atr, min_d,
                                    p.fill_quota_per_side, p.prefilter_k)
            for i in idxs:
                center = float(px[i])
                zlo, zhi = self._zone_bounds(loc, px, i, atr, bin_w)
                hold_lb, n_ev, n_hold = self._touch_stats(ctx, zlo, zhi, kind, hist_lo)
                recency = self._recency(ctx, zlo, zhi, hist_lo)
                stale = 1.0 - recency

                # --- edge 分：只放实测 fwd_ret 同向的证据 ---
                ev_term = math.log1p(max(n_ev, 0))
                wil = hold_lb if n_ev >= p.min_events_for_wilson else 0.0
                edge = (p.w_events * ev_term
                        + p.w_stale * stale
                        + p.w_hold_wilson * wil
                        + p.w_vp_edge * float(vp[i])
                        + p.w_pivot_edge * float(pvz[i]))

                # --- p_stall 分：价格在此停下的倾向（供止损/目标位使用）---
                stall = (p.s_vp * float(vp[i]) + p.s_close * float(cp[i])
                         + p.s_width * ((zhi - zlo) / atr))

                out.append(Level(
                    center=center, low=zlo, high=zhi,
                    score=float(edge), kind=kind,
                    meta={"vp": float(vp[i]), "close": float(cp[i]),
                          "pivot": float(pvz[i]), "loc": float(loc[i]),
                          "hold_lb": float(hold_lb), "n_events": float(n_ev),
                          "n_hold": float(n_hold), "stale": float(stale),
                          "p_stall": float(stall), "edge": float(edge)},
                ))

        out = nms(out, p.nms_gap_atr, atr, p.max_out)
        out = annotate(out, price, atr)
        if self.calibrator is not None:
            probs = self.calibrator.predict(np.array([z.score for z in out]))
            for z, pr in zip(out, probs):
                z.confidence = float(pr) if np.isfinite(pr) else 0.0
        return out

    def _candidates(self, loc: np.ndarray, px: np.ndarray, price: float,
                    kind: str, atr: float, min_d: int,
                    quota: int, cap: int) -> List[int]:
        """
        候选生成 + 配额补齐。

        峰值不足时，用满足间距约束的次高点补齐到 quota，
        保证每侧输出数量与"傻"关键位对照组同口径。
        """
        p = self.p

        def ok(i: int) -> bool:
            cc = float(px[i])
            if kind == "support" and cc >= price:
                return False
            if kind == "resistance" and cc <= price:
                return False
            return abs(cc - price) / atr <= p.max_dist_atr

        peaks = [int(i) for i in find_peaks_simple(loc, min_distance=min_d) if ok(i)]
        peaks.sort(key=lambda i: -loc[i])
        chosen = peaks[:cap]

        if len(chosen) < quota:
            order = np.argsort(loc)[::-1]
            for i in order:
                i = int(i)
                if len(chosen) >= quota:
                    break
                if not ok(i) or i in chosen:
                    continue
                if all(abs(i - j) >= min_d for j in chosen):
                    chosen.append(i)
        return chosen

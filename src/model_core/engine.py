import copy
import heapq
import json
import math
import os
import pathlib
import pickle
import random
import uuid
import sys
import time

import torch

# ── 并行评估线程配置（必须在第一个张量操作前设置）──────────────────────────
# torch.set_num_interop_threads 一旦有任何张量 op 就锁定，所以紧跟 import torch。
# 策略：与 MT5_AlphaGPT 一致——保持 intra_threads=full cores，让 PyTorch 内部
# 多线程处理大张量；同时开 8 workers 并行评估不同公式。
# PyTorch 的 intra-op 线程在执行期间会释放 GIL，允许多个 worker 真正并行。
_PHYS_CORES = os.cpu_count() or 4
_EVAL_WORKERS = min(_PHYS_CORES, 8)
_INTRA = _PHYS_CORES  # 保持满线程，不做 phys // workers
try:
    torch.set_num_interop_threads(_EVAL_WORKERS)
except RuntimeError:
    pass  # 已被锁定
torch.set_num_threads(_INTRA)

import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from .config import ModelConfig
from .alphagpt import AlphaGPT, NewtonSchulzLowRankDecay, StableRankMonitor
from .vm import StackVM
from .backtest import MT5Backtest, estimate_periods_per_year
from .vocab import FORMULA_VOCAB, VOCAB_VERSION, VocabVersionMismatchError  # task 12.2
from utils.training_artifacts import atomic_json_write, atomic_write, checkpoint_dir, checkpoint_step, history_path, json_safe, root_path, strategy_path
from utils.training_context import data_fingerprint, rng_state, restore_rng

# P3：冠军在场时间稳健性校验所需
try:
    from strategy_manager.signal import compute_target_positions_stateless
except ImportError:
    # 兼容无 strategy_manager 的测试环境
    def compute_target_positions_stateless(factors):  # type: ignore
        import torch as _torch
        return _torch.sign(_torch.tanh(factors))

try:
    from config import Config as _RootConfig
    _STRATEGY_FILE = str(strategy_path())
    _CHECKPOINT_DIR = checkpoint_dir()
except ImportError:
    _STRATEGY_FILE = "best_mt5_strategy.json"
    _CHECKPOINT_DIR = pathlib.Path("checkpoints")


def _current_scoring_version() -> str:
    from . import backtest
    return str(backtest.SCORING_VERSION)


def _feature_semantics_version() -> str:
    try:
        from . import features
        return str(getattr(features, "FEATURE_SEMANTICS_VERSION", "legacy"))
    except ImportError:
        return "legacy"


def _strategy_file_for_symbol(symbol: str | None, timeframe: str | None = None) -> str:
    """返回该品种+周期对应的策略文件路径。

    带周期时用 strategies/best_{symbol}_{tf}.json（如 best_BTCUSD__H1.json），
    不同周期互不覆盖；不带周期时退回 best_{symbol}.json；
    多品种/未指定品种时回退到默认路径。
    """
    return str(strategy_path(symbol, timeframe))


class CheckpointPreflightError(ValueError):
    """Known incompatibility or safe decoding failure, before engine mutation."""


class CheckpointIdentityError(CheckpointPreflightError):
    """Checkpoint identity is missing or differs from the requested training."""


class CheckpointRestoreError(RuntimeError):
    """Restoration failed; this engine must not be reused for fresh training."""


STRATEGY_SAVE_REASONS = {
    "no_champion": "本轮没有有效新冠军（无公式或分数非有限）",
    "legacy_scoring": "现有策略为旧评分口径，分数不可比",
    "feature_semantics": "特征语义版本不同，分数不可比",
    "context": "训练上下文不同（数据/评估口径），分数不可比",
    "unreadable": "现有策略文件无法解析",
    "score": "现有策略分数不低于本轮结果",
    "unchanged": "本轮冠军已保存在策略文件中",
}


def save_strategy_artifact(path, payload) -> str | None:
    """尝试把策略写入 path；成功返回 None，被保护时返回原因代码（STRATEGY_SAVE_REASONS）。

    保护规则不变：旧口径/不同特征语义/不同上下文/分数不更高时绝不覆盖主文件。
    """
    try:
        score = float(payload.get("best_score"))
    except (TypeError, ValueError):
        return "no_champion"
    if not payload.get("formula") or not math.isfinite(score):
        return "no_champion"
    path = pathlib.Path(path)
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "unreadable"
        if not isinstance(existing, dict):
            return "unreadable"
        if existing.get("scoring_version") != _current_scoring_version():
            return "legacy_scoring"
        if existing.get("feature_semantics_version") != payload.get("feature_semantics_version"):
            return "feature_semantics"
        if existing.get("training_context") != payload.get("training_context"):
            return "context"
        try:
            old_score = float(existing["best_score"])
        except (KeyError, TypeError, ValueError):
            return "unreadable"
        if (math.isfinite(old_score) and old_score == score
                and existing.get("formula") == payload.get("formula")):
            return "unchanged"
        if not math.isfinite(old_score) or old_score >= score:
            return "score"
    atomic_json_write(path, payload)
    return None


def _candidate_strategy_file(symbol: str | None, timeframe: str | None = None,
                             stamp: str | None = None) -> str:
    """候选策略路径：主文件受保护时本轮新策略另存的位置，永不覆盖主文件。"""
    base = pathlib.Path(_strategy_file_for_symbol(symbol, timeframe))
    # No fixed shared candidate: callers updating a run pass its stable UUID.
    suffix = stamp or uuid.uuid4().hex
    return str(base.with_name(f"{base.stem}_候选_{suffix}{base.suffix}"))


def write_candidate_strategy(save_path, payload, stamp: str | None = None) -> str | None:
    """把被保护的新策略另存为 strategies/ 下的候选文件；返回路径，失败返回 None。

    候选文件带 candidate_of 字段指向被保护的主文件；策略库可正常展示、绑定，
    由用户对比验收后手动替换主文件。"""
    try:
        data = dict(payload)
        data["candidate_of"] = pathlib.Path(save_path).name
        cand = _candidate_strategy_file(data.get("symbol"), data.get("timeframe"), stamp)
        atomic_json_write(cand, json_safe(data))
        return cand
    except Exception as exc:  # noqa: BLE001
        try:
            tqdm.write(f"[警告] 候选策略另存失败: {exc}")
        except Exception:
            pass
        return None


def _fallback_data_file_for_symbol(symbol: str) -> tuple[str | None, str | None]:
    """Read web_settings.json last_data_file when strategy JSON lacks data_file."""
    settings_path = root_path("web_settings.json")
    if not settings_path.exists():
        return None, None
    try:
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        last = str(settings.get("last_data_file") or "").strip()
    except (json.JSONDecodeError, OSError):
        return None, None
    if not last:
        return None, None
    p = pathlib.Path(last)
    if not p.exists():
        return None, None
    try:
        from data_pipeline.parquet_manager import inspect_parquet_file

        info = inspect_parquet_file(str(p.resolve()))
    except Exception:
        return str(p.resolve()), None
    if info.get("symbol") != symbol:
        return None, None
    return str(p.resolve()), info.get("timeframe")


# ─────────────────────────────────────────────────────────────────────────────
# Walk-Forward 折叠构建
# ─────────────────────────────────────────────────────────────────────────────

def _build_walk_forward_folds(T: int, n_folds: int = 5, gap: int = 20) -> list[dict]:
    """Rolling train/validation blocks with an explicitly reserved purge gap.

    Reduce the number of blocks for short data, never silently erase the gap.
    Later training windows may reuse earlier validation data: these folds are
    search validation, NOT an untouched final test set.
    """
    if T < 4 or n_folds < 2 or gap < 0:
        raise ValueError("Walk-forward needs T >= 4, n_folds >= 2 and gap >= 0")
    while n_folds > 2 and (T - gap * (n_folds - 1)) // n_folds < 2:
        n_folds -= 1
    fold_size = (T - gap * (n_folds - 1)) // n_folds
    if fold_size < 2:
        raise ValueError("Not enough bars for separate train/validation blocks and WF_GAP")
    folds = []
    for k in range(1, n_folds):
        train_start = (k - 1) * (fold_size + gap)
        train_end = train_start + fold_size
        val_start = train_end + gap
        folds.append({"train_start": train_start, "train_end": train_end,
                      "val_start": val_start, "val_end": val_start + fold_size,
                      "gap": gap})
    return folds


def _repetition_penalty(formula: list[int]) -> float:
    if not formula:
        return 0.0
    penalty, count = 0.0, 1
    for i in range(1, len(formula)):
        if formula[i] == formula[i - 1]:
            count += 1
            if count >= 2:
                penalty += 0.3
        else:
            count = 1
    return penalty


# ─────────────────────────────────────────────────────────────────────────────
# ConstrainedSampler — 保证 100% 合法公式
# ─────────────────────────────────────────────────────────────────────────────

class ConstrainedSampler:
    def __init__(self, vocab_size: int, feat_offset: int, arity_map: dict[int, int],
                 positive_only_ids: set[int] | None = None):
        self.vocab_size  = vocab_size
        self.feat_offset = feat_offset
        self.arity_map   = arity_map
        self.delta: dict[int, int] = {}
        for tid in range(vocab_size):
            if tid < feat_offset:
                self.delta[tid] = 1
            else:
                a = arity_map.get(tid, 1)
                self.delta[tid] = 1 - a
        # 恒正算子 token id 集合（用于算子链约束）
        self.positive_only_ids = positive_only_ids or set()
        # 构建感染传播/恢复算子 id 集合
        from .vm import INFECTED_PROPAGATING_OPS, SIGN_RESTORE_OPS
        from .ops import OPS_CONFIG as _ops
        self.infected_propagating_ids = set()
        self.sign_restore_ids = set()
        for i, cfg in enumerate(_ops):
            tid = i + feat_offset
            if cfg[0] in INFECTED_PROPAGATING_OPS:
                self.infected_propagating_ids.add(tid)
            if cfg[0] in SIGN_RESTORE_OPS:
                self.sign_restore_ids.add(tid)

    def valid_mask(self, stack_depth: int, step_idx: int,
                   total_steps: int, device: torch.device,
                   prev_token: int | None = None,
                   infected_chain_len: int = 0) -> torch.Tensor:
        remaining = total_steps - step_idx
        if remaining < 1:
            raise ValueError("No token slots remain")
        mask = torch.ones(self.vocab_size, dtype=torch.bool, device=device)
        structural = mask.clone()
        max_reduction = max(0, -min(self.delta.values()))
        for tid in range(self.vocab_size):
            arity = 0 if tid < self.feat_offset else self.arity_map.get(tid, 1)
            new_depth = stack_depth + self.delta[tid]
            min_future = new_depth - (remaining - 1) * max_reduction
            max_future = new_depth + remaining - 1
            if stack_depth < arity or new_depth < 1 or not min_future <= 1 <= max_future:
                mask[tid] = structural[tid] = False
                continue
            # ── 算子链约束（感染模型）──────────────────────────────
            # 如果已感染且感染链 >= 2，禁止再使用传播算子
            # （允许恢复算子和非传播算子如 ADD/SUB/MUL）
            if infected_chain_len >= 2 and tid in self.infected_propagating_ids:
                mask[tid] = False
            if infected_chain_len >= 3 and (tid in self.infected_propagating_ids
                                            or tid in self.positive_only_ids):
                mask[tid] = False
        if not structural.any():
            # 结构上完全走死（不应发生，arith_bounds + arity 需 ≥1 可达）
            raise ValueError("No structurally valid token: increase MAX_FORMULA_LEN")
        if not mask.any():
            # Relax only the heuristic, never arity or final-stack constraints.
            mask = structural
        return mask

    def apply_mask_to_logits(self, logits: torch.Tensor, stack_depths: list[int],
                              step_idx: int, total_steps: int,
                              prev_tokens: list[int | None] | None = None,
                              infected_chain_lens: list[int] | None = None) -> torch.Tensor:
        masked = logits.clone()
        device = logits.device
        for b, depth in enumerate(stack_depths):
            prev_t = prev_tokens[b] if prev_tokens else None
            icl = infected_chain_lens[b] if infected_chain_lens else 0
            vmask = self.valid_mask(depth, step_idx, total_steps, device,
                                    prev_token=prev_t, infected_chain_len=icl)
            masked[b][~vmask] = -1e9
        return masked

    def update_infection(self, token: int, infected_chain_len: int) -> int:
        """更新感染链长度，返回新的感染链长度。"""
        if token in self.positive_only_ids:
            return infected_chain_len + 1
        elif token in self.sign_restore_ids:
            return 0
        elif token in self.infected_propagating_ids:
            if infected_chain_len > 0:
                return infected_chain_len + 1
            return 0
        return infected_chain_len  # 非传播/非恢复算子，不改变状态


# ─────────────────────────────────────────────────────────────────────────────
# AlphaEngine — __init__ 与静态辅助方法
# ─────────────────────────────────────────────────────────────────────────────

class AlphaEngine:
    def __init__(self, data_manager=None, use_lord_regularization=True,
                 lord_decay_rate=1e-3, lord_num_iterations=5, n_folds: int = 5,
                 target_symbol: str | None = None):
        self.data_manager  = data_manager
        self.n_folds       = n_folds
        self.target_symbol = target_symbol   # None = 多品种模式，str = 单品种模式
        self.timeframe: str | None = None    # 单品种训练周期（train_file 设置）
        self.model   = AlphaGPT().to(ModelConfig.DEVICE)
        self.opt     = torch.optim.AdamW(self.model.parameters(), lr=1e-3)

        self.use_lord = use_lord_regularization
        if self.use_lord:
            self.lord_opt = NewtonSchulzLowRankDecay(
                self.model.named_parameters(),
                decay_rate=lord_decay_rate,
                num_iterations=lord_num_iterations,
                target_keywords=["attention", "qk_norm"],
            )
            self.rank_monitor = StableRankMonitor(
                self.model, target_keywords=["in_proj", "out_proj", "qk_norm"]
            )
        else:
            self.lord_opt = None
            self.rank_monitor = None

        self.vm = StackVM()
        self.bt = MT5Backtest()

        from .vocab import FORMULA_VOCAB as _v
        self.sampler = ConstrainedSampler(
            vocab_size=_v.size, feat_offset=_v.operator_offset,
            arity_map=self.vm.arity_map,
            positive_only_ids=self.vm.positive_only_ids
        )

        # Not checkpointed: a resumed/new engine owns a new bindable candidate.
        self.candidate_run_id = uuid.uuid4().hex
        self.best_score   = -float('inf')
        self.best_formula = None
        self._best_snapshot: dict | None = None

        self.training_history = {
            'step': [], 'avg_reward': [], 'best_score': [], 'val_score': [], 'stable_rank': []
        }
        self._restart_count      = 0
        # 多岛训练时各岛共用 target_symbol 会互写同一个训练曲线文件；
        # history_tag 非空时优先用它命名（如 "BTCUSD__isl2"）
        self.history_tag: str | None = None
        # 关闭后不写检查点（岛模式的岛不可续训，且 3 岛×每 20 步一存太占盘）
        self.save_checkpoints = True
        self.factor_pool: list[tuple[float, int, torch.Tensor]] = []
        self._factor_pool_counter = 0

        # Elite Replay pool: (val_score, counter, formula_tokens, birth_step)
        self._elite_pool: list[tuple[float, int, list[int], int]] = []
        self._elite_counter = 0

        # 自适应噪声：记录 best 刷新步数
        self._best_update_step = 0
        self._stagnation_steps = 0
        # 连续无进步的停滞窗口数（自动停止用）
        self._stag_windows_no_gain = 0
        # 自动停止后供岛模式跳过该岛的后续阶段
        self.stopped_early = False
        # 看板「停止训练」信号触发的安全停止
        self.user_stopped = False
        self.completed_steps = 0
        self._last_stag_restart_step = None
        self._low_entropy_streak = 0
        self._previous_init_dist = None
        self._data_fingerprint = data_fingerprint(data_manager) if data_manager is not None else None

        # Fix 3: EMA reward baseline
        self._reward_ema: float | None = None
        self._reward_ema_step: int = 0

        # ── 并行评估线程池（CPU 利用率优化）──────────────────────────────
        self._eval_pool = None
        self._eval_workers = 1
        if ModelConfig.PARALLEL_EVAL:
            self._init_parallel_eval()

    # ── 文件命名：品种 + 周期，避免不同周期互相覆盖 ────────────────────────

    def _file_tag(self) -> str | None:
        """单品种训练时的文件名标签：有周期则 '{symbol}_{tf}'，否则 '{symbol}'。"""
        if not self.target_symbol:
            return None
        if self.timeframe:
            return f"{self.target_symbol}_{self.timeframe}"
        return self.target_symbol

    # ── 并行评估初始化 ──────────────────────────────────────────────────────

    def _init_parallel_eval(self):
        """初始化 ThreadPoolExecutor 用于并行公式评估。

        PyTorch CPU 算子会释放 GIL，多个 worker 线程可以真正并行执行
        vm.execute / bt.evaluate_fold 等纯张量计算。

        线程配置（与 MT5_AlphaGPT 一致）：
        - intra_threads = physical_cores（保持满，让大张量操作快）
        - inter_threads = workers（允许多 worker 并行调度）
        PyTorch 的 intra-op 线程池会自适应负载，不会真的 8×8=64 全跑满。
        """
        from concurrent.futures import ThreadPoolExecutor

        phys = _PHYS_CORES
        workers = ModelConfig.EVAL_WORKERS
        if workers <= 0:
            workers = min(phys, 8)
        intra = ModelConfig.EVAL_INTRA_THREADS
        if intra <= 0:
            intra = phys  # 保持满线程
        # 线程已在模块 import 时设置，这里只确认
        try:
            torch.set_num_threads(intra)
        except RuntimeError:
            pass

        self._eval_workers = workers
        self._eval_pool = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="alpha-eval",
        )
        print(f"[ParallelEval] workers={workers} intra_threads={intra} "
              f"(physical_cores={phys})  pool={'ON' if workers > 1 else 'OFF'}",
              flush=True)

    # ── 单条公式评估任务（线程安全）───────────────────────────────────────

    def _eval_formula_task(
        self,
        idx: int,
        fml: list[int],
        feat: torch.Tensor,
        t_ret: torch.Tensor,
        folds: list[dict],
        use_wf: bool,
        factor_pool_snapshot: list,
    ) -> dict:
        """评估单条公式（线程安全，可被多个 worker 并发调用）。

        所有共享对象（self.vm, self.bt）在评估路径上都是只读的；
        factor_pool 通过 snapshot 传入只读快照。
        """
        try:
            with torch.no_grad():
                res = self.vm.execute(fml, feat)

            if res is None:
                return {'idx': idx, 'status': 'none', 'reward': -5.0,
                        'val_score': -5.0, 'fml': fml}
            # target_ret has two sentinel tail values because no forward return
            # exists there.  Keep those bars for feature generation, but never
            # let them enter scoring, IC, or net-profit eligibility.
            valid_t = min(res.shape[1], t_ret.shape[1]) - 2
            if valid_t < 2:
                raise ValueError("Need at least two observed forward-return bars")
            res = res[:, :valid_t]
            t_ret = t_ret[:, :valid_t]
            if res.std() < 1e-4:
                return {'idx': idx, 'status': 'const', 'reward': -2.0,
                        'val_score': -2.0, 'fml': fml}

            with torch.no_grad():
                if use_wf:
                    fold_tr, fold_vl, fold_ic = [], [], []
                    for fold in folds:
                        tr_sc, vl_sc = self.bt.evaluate_fold(
                            res, t_ret,
                            fold["train_start"], fold["train_end"],
                            fold["val_start"],   fold["val_end"],
                        )
                        ic_m, _ = AlphaEngine._compute_ic(
                            res[:, fold["train_start"]:fold["train_end"]],
                            t_ret[:, fold["train_start"]:fold["train_end"]],
                        )
                        tr_adj = AlphaEngine._apply_ic_gate(tr_sc, ic_m)
                        fold_tr.append(ModelConfig.REWARD_ALPHA * tr_adj)
                        ic_v, _ = AlphaEngine._compute_ic(
                            res[:, fold["val_start"]:fold["val_end"]],
                            t_ret[:, fold["val_start"]:fold["val_end"]],
                        )
                        vl_adj = AlphaEngine._apply_ic_gate(vl_sc, ic_v)
                        fold_vl.append(vl_adj)
                        fold_ic.append(ic_m.item())
                    train_score = torch.stack(fold_tr).mean()
                    val_score = torch.stack(fold_vl).mean()
                    ic_i = sum(fold_ic) / len(fold_ic)
                else:
                    T_total = res.shape[1]
                    if T_total < 4:
                        raise ValueError("Need separate train and validation samples")
                    split_pt = min(T_total - 2, max(2, int(T_total * 0.8)))
                    train_score, vl_sc = self.bt.evaluate_fold(
                        res, t_ret, 0, split_pt, split_pt, T_total,
                    )
                    ic_m0, _ = AlphaEngine._compute_ic(res[:, :split_pt], t_ret[:, :split_pt])
                    train_score = AlphaEngine._apply_ic_gate(
                        ModelConfig.REWARD_ALPHA * train_score, ic_m0
                    )
                    ic_v0, _ = AlphaEngine._compute_ic(res[:, split_pt:], t_ret[:, split_pt:])
                    val_score = AlphaEngine._apply_ic_gate(vl_sc, ic_v0)
                    ic_i = ic_m0.item()
                ic_full, ic_stab_full = AlphaEngine._compute_ic(res, t_ret)

            # 惩罚（纯函数，线程安全）
            reward = train_score
            val_score_out = val_score

            # 重复惩罚
            rp = _repetition_penalty(fml)
            if rp > 0:
                reward = reward - rp
                val_score_out = val_score_out - rp

            # 相关性惩罚（用 step 起始快照）
            if use_wf:
                _corr_slice = (folds[0]["train_start"], folds[0]["train_end"])
            else:
                _corr_slice = (0, split_pt)
            reward = self._apply_corr_penalty(reward, res, _corr_slice, factor_pool_snapshot)
            # Ranking must be stationary: pool membership is an exploration
            # penalty only, otherwise an unchanged champion loses score on replay.

            position = compute_target_positions_stateless(res)
            prev = torch.roll(position, 1, dims=1)
            prev[:, 0] = 0
            net_pnl = position * t_ret - (position - prev).abs() * self.bt.cost_rate
            val_pnl = (torch.cat([net_pnl[:, f['val_start']:f['val_end']] for f in folds], dim=1)
                       if use_wf else net_pnl[:, split_pt:])
            # Auxiliary metrics may be positive despite a net loss. Such
            # candidates may teach the policy, but must never become champions.
            eligible = bool(torch.isfinite(net_pnl).all() and net_pnl.mean() > 0
                            and val_pnl.mean() > 0)
            return {
                'idx': idx, 'status': 'ok',
                'eligible': eligible, 'net_mean': net_pnl.mean().item(),
                'validation_net_mean': val_pnl.mean().item(),
                'reward': reward.item() if isinstance(reward, torch.Tensor) else float(reward),
                'val_score': val_score_out.item() if isinstance(val_score_out, torch.Tensor) else float(val_score_out),
                'ic_full': ic_full.item(), 'ic_stab': ic_stab_full.item(),
                'ic_i': ic_i, 'res': res, 'fml': fml,
            }
        except Exception as e:
            return {'idx': idx, 'status': 'error', 'reward': -5.0,
                    'val_score': -5.0, 'fml': fml,
                    'error': f'{type(e).__name__}: {e}'}

    # ── IC computation ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_ic(factor: torch.Tensor, target_ret: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor]:
        """时序 IC（factor[t] vs 已前移的 target_ret[t]）的均值与稳定性。

        对 5 品种宇宙，时序 IC 比横截面 IC 统计意义更强。
        """
        N, T = factor.shape
        if T < 2:
            z = torch.zeros(1, device=factor.device)
            return z, z

        ic_list = []
        for n in range(N):
            # target_ret[t] already represents open[t+1] -> open[t+2].
            x  = factor[n]
            y  = target_ret[n]
            xm = x - x.mean()
            ym = y - y.mean()
            sx = (xm ** 2).mean().sqrt()
            sy = (ym ** 2).mean().sqrt()
            if sx < 1e-6 or sy < 1e-6:
                continue
            ic = (xm * ym).mean() / (sx * sy + 1e-8)
            ic_list.append(ic)

        if not ic_list:
            z = torch.zeros(1, device=factor.device)
            return z, z

        ic_tensor = torch.stack(ic_list)
        ic_mean   = ic_tensor.mean()
        ic_stab   = (ic_mean / (ic_tensor.std(unbiased=False) + 1e-6)
                     if ic_tensor.numel() >= 2
                     else torch.zeros(1, device=factor.device))
        return ic_mean, ic_stab

    # ── IC gate: direction-based, dimension-agnostic ──────────────────────────

    @staticmethod
    def _apply_ic_gate(reward: torch.Tensor, ic_mean) -> torch.Tensor:
        """Positive IC may boost positive scores; negative IC never improves losses."""
        ic_val = ic_mean.item() if isinstance(ic_mean, torch.Tensor) else float(ic_mean)
        t = ModelConfig.IC_GATE_THRESH
        if ic_val > t:
            return torch.where(reward > 0, reward * ModelConfig.IC_GATE_MULT, reward)
        elif ic_val < -t:
            return reward - reward.abs() * (1.0 - ModelConfig.IC_NEG_MULT)
        return reward


    # ── Elite pool ────────────────────────────────────────────────────────────

    @staticmethod
    def _dedup_elite_pool(
        pool: list[tuple[float, int, list[int], int]]
    ) -> list[tuple[float, int, list[int], int]]:
        """对精英池去重：相同 tokens 只保留得分最高的一条，重建最小堆。"""
        best: dict[str, tuple[float, int, list[int], int]] = {}
        for sc, cnt, toks, birth in pool:
            key = str(toks)
            if key not in best or sc > best[key][0]:
                best[key] = (sc, cnt, toks, birth)
        deduped = list(best.values())
        heapq.heapify(deduped)
        return deduped

    def _update_elite_pool(self, val_score: float, formula: list[int], step: int = 0) -> None:
        """维护精英公式池（最小堆，Top-ELITE_POOL_SIZE 个历史最优公式，自动去重）。

        去重逻辑：若 formula 已在池中，只在新得分更高时原地更新，不插入重复副本。
        这防止了单一公式垄断 elite pool，保持多样性。
        新增：记录 birth_step 用于 elite decay。
        """
        k = ModelConfig.ELITE_POOL_SIZE

        # 检查是否已有相同公式
        for idx, (sc, cnt, toks, birth) in enumerate(self._elite_pool):
            if toks == formula:
                if val_score <= sc:
                    return  # 已有更高分的相同公式，不更新
                # 分数更高：从堆中移除旧条目，插入新条目
                self._elite_pool[idx] = self._elite_pool[-1]
                self._elite_pool.pop()
                heapq.heapify(self._elite_pool)  # O(k)，k≤20，可接受
                break

        entry = (val_score, self._elite_counter, list(formula), step)
        self._elite_counter += 1
        if len(self._elite_pool) < k:
            heapq.heappush(self._elite_pool, entry)
        elif val_score > self._elite_pool[0][0]:
            heapq.heapreplace(self._elite_pool, entry)

    # ── Factor pool ───────────────────────────────────────────────────────────

    def _update_factor_pool(self, val_score: float, factor: torch.Tensor) -> None:
        k     = ModelConfig.FACTOR_TOP_K
        f_gpu = factor.detach()
        entry = (val_score, self._factor_pool_counter, f_gpu)
        self._factor_pool_counter += 1
        if len(self.factor_pool) < k:
            heapq.heappush(self.factor_pool, entry)
        elif val_score > self.factor_pool[0][0]:
            heapq.heapreplace(self.factor_pool, entry)

    def _apply_corr_penalty(
        self,
        reward: torch.Tensor,
        factor: torch.Tensor,
        train_slice: tuple[int, int] | None = None,
        pool_snapshot: list | None = None,
    ) -> torch.Tensor:
        """相关性惩罚：与因子池中已有因子的相关性超过阈值则惩罚 reward。

        P1-6 修复：相关性只在 train 切片上计算，避免含 val 段数据泄漏。
        train_slice=None 时回退到整段（向后兼容）。
        """
        pool = self.factor_pool if pool_snapshot is None else pool_snapshot
        if not pool:
            return reward
        # P1-6: 相关性只在 train 切片上计算，避免 val 信息泄漏
        if train_slice is not None:
            s, e = train_slice
            f = factor.detach()[:, s:e]
        else:
            f = factor.detach()
        f_flat = f.reshape(-1).float()
        if f_flat.std() < 1e-4:
            return reward
        # 因子池中的历史因子也按相同切片取（若形状一致）
        pool_vecs_list = []
        for _, _cnt, pf in pool:
            if pf.shape != factor.shape:
                continue
            pf_t = pf.detach()
            if train_slice is not None and pf_t.shape[1] >= factor.shape[1]:
                pf_t = pf_t[:, s:e]
            pool_vecs_list.append(pf_t.reshape(-1).float())
        if not pool_vecs_list:
            return reward
        pool_vecs = torch.stack(pool_vecs_list, dim=0)
        f_c  = f_flat - f_flat.mean()
        p_c  = pool_vecs - pool_vecs.mean(dim=1, keepdim=True)
        cov  = (p_c * f_c).sum(dim=1)
        sx   = f_c.norm() + 1e-8
        sy   = p_c.norm(dim=1) + 1e-8
        corr = (cov / (sx * sy)).abs()
        if (corr > ModelConfig.CORR_THRESHOLD).any():
            reward = reward - reward.abs() * (1.0 - ModelConfig.CORR_PENALTY)
        return reward

    def _distribution_stats(self, prev_dist=None):
        """计算模型初始位置（zero prefix）token 分布的细化指标，用于判断 H 不变时
        分布是否真的在变化。
        """
        vocab_size = FORMULA_VOCAB.size
        with torch.no_grad():
            inp = torch.zeros((1, 1), dtype=torch.long,
                              device=ModelConfig.DEVICE)
            logits, _, _ = self.model(inp)
            logits = self.sampler.apply_mask_to_logits(
                logits, [0], 0, ModelConfig.MAX_FORMULA_LEN
            )
            dist = F.softmax(logits, dim=-1).squeeze(0)
            ent = -(dist * torch.log(dist + 1e-12)).sum().item()
            log_v = math.log(vocab_size)
            kl_uniform = log_v - ent
            top1 = dist.max().item()
            top5 = dist.topk(5, dim=-1).values.sum().item()
            eff_vocab = math.exp(ent)
            prob_std = dist.std(unbiased=False).item()
            kl_prev = 0.0
            if prev_dist is not None:
                kl_prev = (
                    dist * (torch.log(dist + 1e-12) -
                            torch.log(prev_dist.to(dist.device) + 1e-12))
                ).sum().item()
        return {
            'dist': dist.cpu(),
            'entropy': ent,
            'kl_uniform': kl_uniform,
            'top1_prob': top1,
            'top5_prob': top5,
            'eff_vocab': eff_vocab,
            'prob_std': prob_std,
            'kl_prev': kl_prev,
        }

    # ── Main training loop ────────────────────────────────────────────────────

    def train(self, start_step: int = 0, end_step: int | None = None,
              migration_hook=None, verbose_header: bool = True,
              run_end_step: int | None = None, run_start_step: int | None = None):
        if self.data_manager is None:
            raise RuntimeError("AlphaEngine requires a data_manager.")

        if end_step is None:
            end_step = ModelConfig.TRAIN_STEPS
        run_end_step = end_step if run_end_step is None else run_end_step
        run_start_step = start_step if run_start_step is None else run_start_step
        self.completed_steps = start_step
        self.user_stopped = False

        if verbose_header:
            print("开始 Alpha 因子挖掘训练" +
                  ("（含 LoRD 正则化）..." if self.use_lord else "..."))
            print(f"   策略熵: 坍塌阈值={ModelConfig.ENTROPY_COLLAPSE_THRESH}  "
                  f"系数上限={ModelConfig.ENTROPY_COEFF_MAX}  "
                  f"连续坍塌步数={ModelConfig.ENTROPY_COLLAPSE_STEPS}")
            print(f"   精英回放: 比例={ModelConfig.ELITE_REPLAY_FRAC}  "
                  f"池大小={ModelConfig.ELITE_POOL_SIZE}")
            print(f"   IC门控: 阈值±{ModelConfig.IC_GATE_THRESH}  "
                  f"正向×{ModelConfig.IC_GATE_MULT}  负向×{ModelConfig.IC_NEG_MULT}")
            print(f"   最大重启: {ModelConfig.MAX_RESTARTS}  "
                  f"噪声={ModelConfig.RESTART_NOISE}")
            if ModelConfig.STAG_HARD_RESTART:
                print(f"   停滞硬重启: 每 {ModelConfig.STAGNATION_WINDOW} 步未刷新最优即强扰动"
                      + (f"；连续 {ModelConfig.STAG_AUTO_STOP_WINDOWS} 个窗口无进步自动结束"
                         if ModelConfig.STAG_AUTO_STOP_WINDOWS > 0 else ""))

        T     = self.data_manager.target_ret.shape[1] - 2
        folds = _build_walk_forward_folds(T, self.n_folds,
                                          gap=getattr(ModelConfig, 'WF_GAP', 20))
        # Even a single short-data fold must retain its reserved purge gap.
        use_wf = bool(folds)
        if verbose_header:
            if use_wf:
                print(f"   滚动验证: {len(folds)} 折  共 {T} 根K线")
                for k, f in enumerate(folds):
                    print(f"  第{k+1}折: 训练[{f['train_start']},{f['train_end']}) "
                          f"间隔={f['gap']} 验证[{f['val_start']},{f['val_end']})")
            else:
                print(f"   退化为全量评估（共 {T} 根K线）")

        # 因果安全：features.py 的 _robust_norm 已改为滚动因果实现
        # 每个 t 的归一化参数只用 [t-w+1..t]，walk-forward 折叠切片无泄露
        feat  = self.data_manager.feat_tensor.to(ModelConfig.DEVICE)
        t_ret = self.data_manager.target_ret.to(ModelConfig.DEVICE)

        # 数据驱动年化因子：按训练数据的实际时间戳估计每年 bar 数，
        # 替代 MT5Backtest 默认的 H1=6240。A 股日线/15min、加密日线等
        # 非 H1 周期不再被按 H1 年化（否则 Sharpe/年化收益被放大数倍）。
        _dm_raw = getattr(self.data_manager, "raw_dict", None) or {}
        _dm_time = _dm_raw.get("time", None)
        if _dm_time is not None:
            try:
                _ppy = estimate_periods_per_year(_dm_time)
                if _ppy != self.bt.periods_per_year:
                    if verbose_header:
                        print(f"   年化因子: {_ppy} bar/年（按数据周期自动估计）")
                    self.bt.periods_per_year = _ppy
            except Exception:
                pass  # 估计失败则保留默认 6240

        bs      = ModelConfig.BATCH_SIZE
        n_elite = max(1, int(bs * ModelConfig.ELITE_REPLAY_FRAC))
        n_new   = bs - n_elite

        remaining = end_step - start_step
        if remaining <= 0:
            print(f"[训练] 起始步 {start_step} 已达目标步 {end_step}，无需继续训练。")
            return

        # 非交互/重定向输出时关闭 tqdm 进度条，避免进度条刷屏把自定义日志淹掉。
        # tqdm.write 仍然可用，详细 step 日志会继续输出。
        pbar               = tqdm(range(start_step, end_step),
                                  total=end_step,
                                  initial=start_step,
                                  disable=not sys.stderr.isatty(),
                                  leave=False,
                                  mininterval=5.0)
        low_entropy_streak = self._low_entropy_streak
        prev_init_dist = self._previous_init_dist
        # 停止信号文件（看板「停止训练」写入；由训练入口脚本在启动时清理，
        # 引擎只读不删——岛模式下后跑的岛不能把信号吞掉）
        stop_file = root_path("TRAIN_STOP")

        for step in pbar:
            self._low_entropy_streak = low_entropy_streak
            self._previous_init_dist = prev_init_dist
            if (self.save_checkpoints and step > start_step
                    and step % max(1, int(getattr(ModelConfig, "CHECKPOINT_EVERY", 10))) == 0):
                self.save_checkpoint(self.completed_steps)
            if stop_file.exists():
                tqdm.write("[收到停止信号] 正在保存已完成进度后退出…")
                self.user_stopped = True
                break
            # ── Part A: Sample n_new new formulas ────────────────────
            inp_new = torch.zeros((n_new, 1), dtype=torch.long,
                                  device=ModelConfig.DEVICE)
            lp_new, tok_new, ent_new = [], [], []
            sd_new = [0] * n_new
            prev_tokens_new: list[int | None] = [None] * n_new
            infected_chain_new: list[int] = [0] * n_new

            for si in range(ModelConfig.MAX_FORMULA_LEN):
                lg, _, _ = self.model(inp_new)
                lg = self.sampler.apply_mask_to_logits(lg, sd_new, si,
                                                       ModelConfig.MAX_FORMULA_LEN,
                                                       prev_tokens=prev_tokens_new,
                                                       infected_chain_lens=infected_chain_new)
                d  = Categorical(logits=lg)
                a  = d.sample()
                lp_new.append(d.log_prob(a))
                tok_new.append(a)
                ent_new.append(d.entropy())
                inp_new = torch.cat([inp_new, a.unsqueeze(1)], dim=1)
                for b in range(n_new):
                    sd_new[b] += self.sampler.delta[a[b].item()]
                    prev_tokens_new[b] = a[b].item()
                    infected_chain_new[b] = self.sampler.update_infection(
                        a[b].item(), infected_chain_new[b])

            seqs_new = torch.stack(tok_new, dim=1)


            # ── Part B: Elite Replay ─────────────────────────────────
            elite_formulas: list[list[int]] = []
            if self._elite_pool and n_elite > 0:
                ps = []
                pt = []
                weights = []
                for sc, cnt, toks, birth in self._elite_pool:
                    age = max(0, step - birth)
                    decay = 1.0
                    if ModelConfig.ELITE_DECAY:
                        half = max(1, ModelConfig.ELITE_DECAY_HALF_LIFE)
                        decay = 0.5 ** (age / half)
                    ps.append(sc)
                    pt.append(toks)
                    weights.append(decay)
                ps_min  = min(ps)
                ps_max  = max(ps)
                # 软温度采样：避免最高分公式垄断
                # 先归一到 [0,1]，再除以温度 T=0.5 后做 softmax
                # T<1 → 高分公式仍被偏好，但不再独占
                if ps_max > ps_min:
                    normalized = [(s - ps_min) / (ps_max - ps_min + 1e-8) for s in ps]
                else:
                    normalized = [1.0] * len(ps)
                temp = 0.5
                exp_s = [weights[i] * (2.0 ** (normalized[i] / temp)) for i in range(len(ps))]
                exp_sum = sum(exp_s)
                probs = [e / exp_sum for e in exp_s]
                idx_e   = random.choices(range(len(self._elite_pool)),
                                         weights=probs, k=n_elite)
                elite_formulas = [pt[i] for i in idx_e]

                # 详细日志：Elite Replay 衰减状态（每 100 步打印一次）
                if step % 100 == 0:
                    avg_decay = sum(weights) / len(weights)
                    max_age = max(max(0, step - birth) for _, _, _, birth in self._elite_pool)
                    age_list = sorted([max(0, step - birth) for _, _, _, birth in self._elite_pool])
                    tqdm.write(
                        f"[精英回放 @ 第{step}步] 池大小={len(self._elite_pool)} "
                        f"平均衰减={avg_decay:.3f} 最大龄期={max_age} 龄期列表={age_list} "
                        f"抽样分数=[{', '.join(f'{ps[i]:.3f}' for i in idx_e[:3])}...]"
                    )
            else:
                elite_formulas = seqs_new[:n_elite].tolist()

            lp_elite, ent_elite = [], []
            if elite_formulas:
                ne     = len(elite_formulas)
                inp_e  = torch.zeros((ne, 1), dtype=torch.long,
                                     device=ModelConfig.DEVICE)
                sd_e   = [0] * ne
                prev_tokens_elite: list[int | None] = [None] * ne
                infected_chain_elite: list[int] = [0] * ne
                tok_e_t = torch.tensor(elite_formulas, dtype=torch.long,
                                       device=ModelConfig.DEVICE)
                for si in range(ModelConfig.MAX_FORMULA_LEN):
                    lg_e, _, _ = self.model(inp_e)
                    lg_e = self.sampler.apply_mask_to_logits(
                        lg_e, sd_e, si, ModelConfig.MAX_FORMULA_LEN,
                        prev_tokens=prev_tokens_elite,
                        infected_chain_lens=infected_chain_elite
                    )
                    d_e  = Categorical(logits=lg_e)
                    tk   = tok_e_t[:, si]
                    lp_elite.append(d_e.log_prob(tk))
                    ent_elite.append(d_e.entropy())
                    inp_e = torch.cat([inp_e, tk.unsqueeze(1)], dim=1)
                    for b in range(ne):
                        sd_e[b] += self.sampler.delta[tk[b].item()]
                        prev_tokens_elite[b] = tk[b].item()
                        infected_chain_elite[b] = self.sampler.update_infection(
                        tk[b].item(), infected_chain_elite[b])


            # ── Part C: Evaluate all formulas (并行评估) ────────────────
            all_fmls = seqs_new.tolist() + elite_formulas
            tot      = len(all_fmls)
            rewards    = torch.zeros(tot, device=ModelConfig.DEVICE)
            val_scores = torch.zeros(tot, device=ModelConfig.DEVICE)

            ok_cnt = none_cnt = const_cnt = 0
            step_max_val = -float('inf');  step_best_f = None
            bic, bis, bsor = [], [], []

            # factor_pool 快照：所有 worker 看到同一份只读视图
            factor_pool_snapshot = list(self.factor_pool)

            # 并行提交所有公式评估任务
            if self._eval_pool is not None and self._eval_workers > 1 and tot > 1:
                from concurrent.futures import ThreadPoolExecutor
                futures = [
                    self._eval_pool.submit(
                        self._eval_formula_task, i, fml, feat, t_ret,
                        folds, use_wf, factor_pool_snapshot,
                    )
                    for i, fml in enumerate(all_fmls)
                ]
                results_by_idx: dict[int, dict] = {}
                for fut in futures:
                    r = fut.result()
                    results_by_idx[r['idx']] = r
                results = [results_by_idx[i] for i in range(tot)]
            else:
                # 串行回退
                results = [
                    self._eval_formula_task(
                        i, fml, feat, t_ret,
                        folds, use_wf, factor_pool_snapshot,
                    )
                    for i, fml in enumerate(all_fmls)
                ]

            # ── 串行后处理：写入 rewards/val_scores，更新冠军/池 ─────────
            for r in results:
                i = r['idx']
                status = r.get('status', 'error')
                rewards[i] = r['reward']
                val_scores[i] = r['val_score']

                if status == 'none':
                    none_cnt += 1
                    bic.append(0.0); bis.append(0.0); bsor.append(r['val_score'])
                    continue
                if status == 'const':
                    const_cnt += 1
                    bic.append(0.0); bis.append(0.0); bsor.append(r['val_score'])
                    continue
                if status == 'error':
                    none_cnt += 1
                    bic.append(0.0); bis.append(0.0); bsor.append(r['val_score'])
                    continue

                ok_cnt += 1
                bic.append(r['ic_full']); bis.append(r['ic_stab']); bsor.append(r['val_score'])
                fml = r['fml']
                res = r['res']
                ic_i = r.get('ic_i', 0.0)
                final_val = r['val_score']

                if final_val > step_max_val:
                    step_max_val = final_val; step_best_f = fml

                if r['eligible'] and math.isfinite(final_val) and final_val > self.best_score:
                    # OOS 泛化门控：val_score / train_score < 0.5 说明过拟合
                    train_val = r['reward']
                    if train_val > 0.5 and final_val < train_val * 0.5:
                        tqdm.write(
                            f"[过拟合跳过 @ 第{step}步] 验证={final_val:.3f} "
                            f"训练={train_val:.3f} 比值={final_val/train_val:.2f} | 样本外表现过差"
                        )
                        pass
                    else:
                        pos_check = compute_target_positions_stateless(res)
                        exposure = pos_check.abs().mean().item()
                        if exposure < 0.05:
                            tqdm.write(
                                f"[稀疏跳过 @ 第{step}步] 验证={final_val:.3f} "
                                f"IC={ic_i:.4f} 暴露度={exposure:.1%} | 仓位过稀疏，不更新最优"
                            )
                            pass
                        else:
                            old_best = self.best_score
                            self.best_score   = final_val
                            self.best_formula = fml
                            self._best_snapshot = copy.deepcopy(self.model.state_dict())
                            self._best_update_step = step
                            self._stagnation_steps = 0
                            self._stag_windows_no_gain = 0
                            self._update_factor_pool(final_val, res)
                            self._save_strategy_live()
                            tqdm.write(
                                f"[!] 新最优 @ 第{step}步: 验证={final_val:.3f} "
                                f"(原 {old_best:.3f}，+{final_val-old_best:.3f}) "
                                f"IC={ic_i:.4f} 暴露度={exposure:.1%} | "
                                f"{fml}\n    {self._decode_formula(fml)}"
                            )
                self._update_elite_pool(final_val, fml, step)


            # ── Part D: REINFORCE gradient update ────────────────────
            # Fix 3: EMA baseline 替代 batch mean，避免全负 batch 的相对优选问题
            batch_mean = rewards.mean().item()
            batch_std  = rewards.std().clamp(min=0.1)
            if ModelConfig.REWARD_EMA_BASELINE and self._reward_ema_step >= ModelConfig.REWARD_EMA_WARMUP:
                baseline = self._reward_ema
                adv = (rewards - baseline) / (batch_std + 1e-5)
            else:
                adv = (rewards - batch_mean) / (batch_std + 1e-5)
            # 更新 EMA
            if self._reward_ema is None:
                self._reward_ema = batch_mean
            else:
                self._reward_ema = ModelConfig.REWARD_EMA_DECAY * self._reward_ema + (1.0 - ModelConfig.REWARD_EMA_DECAY) * batch_mean
            self._reward_ema_step += 1
            adv_new   = adv[:n_new]
            adv_elite = adv[n_new:]

            policy_loss = torch.zeros(1, device=ModelConfig.DEVICE)
            for ti in range(len(lp_new)):
                policy_loss += (-lp_new[ti] * adv_new).mean()
            if lp_elite and adv_elite.shape[0] > 0:
                for ti in range(len(lp_elite)):
                    lpe = lp_elite[ti]
                    if lpe.shape[0] == adv_elite.shape[0]:
                        policy_loss += (-lpe * adv_elite
                                        * ModelConfig.ELITE_REWARD_SCALE).mean()

            if ent_new:
                mean_ent_new = torch.stack(ent_new).mean()
            else:
                mean_ent_new = torch.zeros(1, device=ModelConfig.DEVICE)
            if ent_elite:
                mean_ent_elite = torch.stack(ent_elite).mean()
                mean_ent = (
                    mean_ent_new * n_new + mean_ent_elite * n_elite
                ) / (n_new + n_elite)
            else:
                mean_ent = mean_ent_new
            ent_val   = mean_ent.item()
            ent_coeff = ModelConfig.ENTROPY_COEFF_MAX / (
                (1.0 + ent_val) ** ModelConfig.ENTROPY_COEFF_POWER
            )
            loss = policy_loss - ent_coeff * mean_ent
            # 熵下限：把 floor_gap 变成对 mean_ent 的真实梯度惩罚，
            # 低于阈值时对 H 产生持续向上的推力，而不是加一个常数项。
            if ModelConfig.ENTROPY_FLOOR and ent_val < ModelConfig.ENTROPY_FLOOR_THRESH:
                loss = loss + ModelConfig.ENTROPY_FLOOR_LAMBDA * (
                    ModelConfig.ENTROPY_FLOOR_THRESH - mean_ent
                )

            self.opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.opt.step()
            if self.use_lord:
                self.lord_opt.step()
            self.completed_steps = step + 1

            # ── Part D2: 分布细化指标 ────────────────────────────────
            dst = self._distribution_stats(prev_init_dist)
            prev_init_dist = dst['dist']
            with torch.no_grad():
                uniq_tokens = seqs_new.unique().numel()
                uniq_fmls   = torch.unique(seqs_new, dim=0).shape[0]
                fml_div     = uniq_fmls / max(1, n_new)

            # ── Part E: Logging & history & checkpoint ───────────────
            avg_rew = rewards.mean().item()
            avg_val = val_scores.mean().item()
            bim  = sum(bic)  / len(bic)  if bic  else 0.0
            bis_ = sum(bis)  / len(bis)  if bis  else 0.0
            bsor_= sum(bsor) / len(bsor) if bsor else 0.0

            self._stagnation_steps = step - self._best_update_step
            tqdm.write(
                f"[{step+1}/{end_step}] "
                f"新公式={n_new} 精英={n_elite} | "
                f"有效={ok_cnt} 无效={none_cnt} 常数={const_cnt} | "
                f"奖励={avg_rew:.3f} 验证={avg_val:.3f} | "
                f"IC={bim:.4f} | 熵={ent_val:.3f}(系数={ent_coeff:.3f}) | "
                f"最优={self.best_score:.3f} 停滞={self._stagnation_steps} "
                f"精英池={len(self._elite_pool)} 重启={self._restart_count}"
            )
            tqdm.write(
                f"   分布: 初始熵={dst['entropy']:.3f} KL均匀={dst['kl_uniform']:.3f} "
                f"KL上步={dst['kl_prev']:.4f} 最高概率={dst['top1_prob']:.3f} "
                f"前五概率={dst['top5_prob']:.3f} 有效词汇={dst['eff_vocab']:.2f} "
                f"标准差={dst['prob_std']:.4f} | "
                f"本批: 唯一符号={uniq_tokens}/{FORMULA_VOCAB.size} "
                f"唯一公式={uniq_fmls}/{n_new} 多样性={fml_div:.2f}"
            )
            pbar.set_postfix({
                '验证': f"{avg_val:.3f}", '最优': f"{self.best_score:.3f}",
                '熵':   f"{ent_val:.2f}", 'IC':   f"{bim:.4f}",
                '停滞': f"{self._stagnation_steps}",
                '初始熵':  f"{dst['entropy']:.2f}",
                'KL上步': f"{dst['kl_prev']:.3f}",
            })

            if self.use_lord and step % 10 == 0:
                sr = self.rank_monitor.compute()
                self.training_history['stable_rank'].append(sr)

            self.training_history['step'].append(step)
            self.training_history['avg_reward'].append(avg_rew)
            self.training_history['val_score'].append(avg_val)
            self.training_history['best_score'].append(self.best_score)
            self.training_history.setdefault('entropy', []).append(ent_val)
            self.training_history.setdefault('ic_mean', []).append(bim)
            self.training_history.setdefault('ic_stability', []).append(bis_)
            self.training_history.setdefault('sortino', []).append(bsor_)
            self.training_history.setdefault('elite_pool_size', []).append(
                len(self._elite_pool))
            self.training_history.setdefault('init_entropy', []).append(dst['entropy'])
            self.training_history.setdefault('kl_uniform', []).append(dst['kl_uniform'])
            self.training_history.setdefault('kl_prev', []).append(dst['kl_prev'])
            self.training_history.setdefault('top1_prob', []).append(dst['top1_prob'])
            self.training_history.setdefault('eff_vocab', []).append(dst['eff_vocab'])
            self.training_history.setdefault('batch_uniq_tokens', []).append(uniq_tokens)
            self.training_history.setdefault('batch_uniq_fmls', []).append(uniq_fmls)
            self.training_history.setdefault('batch_fml_div', []).append(fml_div)

            if self.best_formula is not None and self.target_symbol:
                self._save_strategy_live()

            self._save_training_history_live()

            # ── Part F: Migration hook（多岛训练时交换精英）────────────
            if migration_hook is not None and (step + 1) % ModelConfig.MIGRATION_INTERVAL == 0:
                tqdm.write(f"[迁移钩子 @ 第{step+1}步] 调用已注册钩子")
                migration_hook(self, step + 1)

            # ── Part G2: 停滞硬重启（与熵无关，2026-09-07）────────────
            # 熵坍塌检测只覆盖 H < COLLAPSE_THRESH；实测有熵停在阈值上方
            # （如 1.05）但分布已死的"假健康"状态（有效词汇≈1），卡数百步。
            # 每满首个阈值后按 STAGNATION_WINDOW 间隔未刷新最优，就从
            # best_snapshot 强扰动。首个阈值比常规窗口小，避免短训练全程不介入。
            if ModelConfig.STAG_HARD_RESTART:
                stag = step - self._best_update_step
                first = max(1, int(getattr(ModelConfig, "STAG_HARD_RESTART_FIRST", ModelConfig.STAGNATION_WINDOW)))
                interval = max(1, int(getattr(
                    ModelConfig, "STAG_HARD_RESTART_INTERVAL", ModelConfig.STAGNATION_WINDOW)))
                # 短续训（如"再练 200 步"）：首次阈值若大于本次步数，整段都不会介入。
                # 按本次步数压缩阈值，让它来得及扰动、重启后还有余量继续学习。
                run_len = max(1, run_end_step - run_start_step)
                first_effective = min(first, max(1, run_len // 3))
                anchor = max(self._best_update_step, run_start_step)
                if self._last_stag_restart_step is not None and self._last_stag_restart_step >= anchor:
                    due_at = self._last_stag_restart_step + interval
                else:
                    due_at = anchor + first_effective
                due = self.completed_steps >= due_at
                # This is the overall run end, not an island migration boundary.
                reserve = max(1, min(30, run_len // 3))
                if run_end_step - self.completed_steps < reserve:
                    due = False
                if due:
                    self._last_stag_restart_step = self.completed_steps
                    self._stag_windows_no_gain += 1
                    # 多次无效训练自动停止：连续 N 个停滞窗口仍无进步，
                    # 提前结束并保留当前最优（策略在循环外的收尾段落保存）
                    if (
                        ModelConfig.STAG_AUTO_STOP_WINDOWS > 0
                        and self._stag_windows_no_gain >= ModelConfig.STAG_AUTO_STOP_WINDOWS
                    ):
                        tqdm.write(
                            f"[自动停止 @ 第{step}步] 连续 {self._stag_windows_no_gain} 个停滞窗口"
                            f"（约 {stag} 步）未刷新最优，提前结束训练，保留当前最优 "
                            f"{self.best_score:.3f}"
                        )
                        self.stopped_early = True
                        break
                    self._restart_count += 1
                    noise = min(ModelConfig.NOISE_MAX, ModelConfig.RESTART_NOISE * 2.0)
                    if self._restart_count > ModelConfig.MAX_RESTARTS:
                        self._restart_count = ModelConfig.MAX_RESTARTS
                        tqdm.write(
                            f"[停滞重启上限] 第{step}步仍无提升，已达到 {ModelConfig.MAX_RESTARTS} 次；"
                            "后续继续累计停滞窗口，达到阈值自动结束。"
                        )
                        continue
                    if self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                    with torch.no_grad():
                        for p in self.model.parameters():
                            p.add_(torch.randn_like(p) * noise)
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                    tqdm.write(
                        f"[停滞重启 {self._restart_count}/{ModelConfig.MAX_RESTARTS} "
                        f"@ 第{step}步] 已 {stag} 步未刷新最优，从最优快照强扰动全参数 "
                        f"噪声={noise:.3f} 熵={ent_val:.3f}"
                    )

            # ── Part G: Entropy collapse detection & restart ─────────
            if ent_val < ModelConfig.ENTROPY_COLLAPSE_THRESH:
                low_entropy_streak += 1
            else:
                low_entropy_streak  = 0

            if low_entropy_streak >= ModelConfig.ENTROPY_COLLAPSE_STEPS:
                # ── 自适应噪声：根据 stagnation 调整 ─────────────────────
                self._stagnation_steps = step - self._best_update_step
                stagnation_ratio = self._stagnation_steps / max(1, ModelConfig.STAGNATION_WINDOW)
                base_noise = ModelConfig.RESTART_NOISE
                if ModelConfig.ADAPTIVE_NOISE:
                    raw_noise = base_noise + ModelConfig.NOISE_BOOST_FACTOR * 0.1 * min(stagnation_ratio, 3.0)
                    noise = max(ModelConfig.NOISE_MIN, min(ModelConfig.NOISE_MAX, raw_noise))
                else:
                    noise = base_noise

                max_r = ModelConfig.MAX_RESTARTS
                if self._restart_count < max_r:
                    self._restart_count  += 1
                    low_entropy_streak    = 0

                    # Fix 2: 每 N 次重启做一次完全随机初始化，逃离 best_snapshot 吸引子
                    # 深度坍塌 (H < 0.3) 时强制 full reset，不给 best_snapshot 恢复的机会
                    do_full_reset = (
                        self._restart_count % ModelConfig.FULL_RESET_EVERY == 0
                        or ent_val < 0.3
                    )

                    if do_full_reset:
                        # 完全重新初始化模型参数
                        for layer in self.model.modules():
                            if hasattr(layer, 'reset_parameters'):
                                layer.reset_parameters()
                        tqdm.write(
                            f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                            f"模式=完全重置（脱离最优快照吸引子） "
                            f"停滞={self._stagnation_steps} "
                            f"熵={ent_val:.3f}"
                        )
                    elif self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                        with torch.no_grad():
                            if ModelConfig.PARTIAL_RESET:
                                perturbed_layers = []
                                for nm, p in self.model.named_parameters():
                                    if any(k in nm for k in ModelConfig.PARTIAL_RESET_LAYERS):
                                        p.add_(torch.randn_like(p) * noise)
                                        perturbed_layers.append(nm)
                            else:
                                perturbed_layers = []
                                for nm, p in self.model.named_parameters():
                                    if 'ffn' in nm or 'attention' in nm or nm.startswith('blocks'):
                                        p.add_(torch.randn_like(p) * noise)
                                        perturbed_layers.append(nm)
                        tqdm.write(
                            f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                            f"模式={'部分层' if ModelConfig.PARTIAL_RESET else 'FFN/注意力'} "
                            f"噪声={noise:.4f}(基准={base_noise:.3f}，比率={stagnation_ratio:.2f}) "
                            f"停滞={self._stagnation_steps} "
                            f"熵={ent_val:.3f} "
                            f"扰动层数={len(perturbed_layers)}"
                        )
                    else:
                        with torch.no_grad():
                            for p in self.model.parameters():
                                p.add_(torch.randn_like(p) * noise)
                        tqdm.write(
                            f"[重启 {self._restart_count}/{max_r} @ 第{step}步] "
                            f"模式=全参数 "
                            f"噪声={noise:.4f}(基准={base_noise:.3f}，比率={stagnation_ratio:.2f}) "
                            f"停滞={self._stagnation_steps} "
                            f"熵={ent_val:.3f} | 无最优快照"
                        )
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                else:
                    # 训练时间不敏感：超过重启上限后不再 Early Stop 终止，
                    # 改为「全参数强扰动 + 重置流计数」继续探索，直到跑满 TRAIN_STEPS。
                    # 从 best_snapshot 恢复（若有）以保住已发现的最优结构，再加大扰动。
                    low_entropy_streak = 0
                    hard_noise = min(ModelConfig.NOISE_MAX, noise * 2.0)
                    if self._best_snapshot is not None:
                        self.model.load_state_dict(self._best_snapshot)
                    with torch.no_grad():
                        for p in self.model.parameters():
                            p.add_(torch.randn_like(p) * hard_noise)
                    self.opt = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
                    tqdm.write(
                        f"[强重启 @ 第{step}步] 已达最大重启次数={max_r} "
                        f"熵={ent_val:.3f} 强噪声={hard_noise:.4f} "
                        f"继续训练，不提前停止"
                    )

        self._low_entropy_streak = low_entropy_streak
        self._previous_init_dist = prev_init_dist
        self._save_training_history_live()
        if self.save_checkpoints:
            self.save_checkpoint(self.completed_steps)
        # ── End of training ──────────────────────────────────────────
        # 仅当跑满最终步时才保存最终 strategy 和历史；岛模式的岛没有
        # target_symbol，策略由 IslandAlphaEngine 统一保存
        if end_step == ModelConfig.TRAIN_STEPS and self.target_symbol:
            if self.best_formula is not None:
                self._save_strategy_live()

            sym_tag = f"[{self.target_symbol}] " if self.target_symbol else ""
            self.training_history.pop('_low_entropy_streak', None)
            self._save_training_history_live()

            print(f"\n[完成] {sym_tag}训练结束！")
            print(f"  最优验证分数 : {self.best_score:.4f}")
            print(f"  最优公式令牌 : {self.best_formula}")
            print(f"  可读公式     : {self._decode_formula(self.best_formula)}")
            print(f"  精英池大小   : {len(self._elite_pool)}")
            print(f"  精英衰减     : 启用={ModelConfig.ELITE_DECAY}，半衰期={ModelConfig.ELITE_DECAY_HALF_LIFE}")
            print(f"  自适应噪声   : 启用={ModelConfig.ADAPTIVE_NOISE}，范围=[{ModelConfig.NOISE_MIN}, {ModelConfig.NOISE_MAX}]")
            print(f"  部分层重置   : 启用={ModelConfig.PARTIAL_RESET}，层={ModelConfig.PARTIAL_RESET_LAYERS}")
            print(f"  重启次数     : {self._restart_count}")
            print(f"  策略目标路径 : {_strategy_file_for_symbol(self.target_symbol, self.timeframe)}（受保护时新结果另存候选文件）")


    # ── 实时保存最优公式（防进程意外退出丢失）────────────────────────────────
    def _save_training_history_live(self) -> None:
        """周期性写入训练曲线 JSON，供 Web UI 实时展示。

        P1-3 修复：原子写入（tmp + os.replace），避免 Ctrl+C / OOM 打断写入
        导致 history 文件损坏。异常打印告警而非静默吞掉。
        """
        if not self.target_symbol and not self.history_tag:
            return
        try:
            sym = self.history_tag or self._file_tag()
            hist_path = history_path(sym)
            payload = {
                k: v for k, v in self.training_history.items()
                if k != "_low_entropy_streak"
            }
            atomic_json_write(hist_path, json_safe(payload))
        except Exception as exc:  # noqa: BLE001
            # 静默吞掉会掩盖磁盘满/权限错误，至少打印告警
            try:
                tqdm.write(f"[警告] 训练历史保存失败: {exc}")
            except Exception:
                pass

    def _live_strategy_payload(self) -> dict:
        """构造当前最优公式的策略 JSON（供主文件与候选文件共用）。"""
        from .vocab import VOCAB_VERSION
        save_path = pathlib.Path(_strategy_file_for_symbol(self.target_symbol, self.timeframe))
        existing: dict = {}
        if save_path.exists():
            try:
                raw = json.loads(save_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    existing = raw
            except Exception:
                existing = {}
        strategy_data = {
            "scoring_version": _current_scoring_version(),
            "feature_semantics_version": _feature_semantics_version(),
            "vocab_version": VOCAB_VERSION,
            "symbol": self.target_symbol,
            "timeframe": self.timeframe,
            "training_context": self.training_context(),
            "formula": self.best_formula,
            "formula_decoded": self._decode_formula(self.best_formula),
            "best_score": self.best_score,
        }
        # 保留训练数据路径等元数据，避免 live 保存把 data_file 冲掉
        for key in ("timeframe", "data_file", "mode", "train_steps"):
            val = getattr(self, key, None)
            if val is None:
                val = existing.get(key)
            if val is not None:
                strategy_data[key] = val
        if not strategy_data.get("data_file") and self.target_symbol:
            data_file, tf = _fallback_data_file_for_symbol(self.target_symbol)
            if data_file:
                strategy_data["data_file"] = data_file
            if tf and not strategy_data.get("timeframe"):
                strategy_data["timeframe"] = tf
            if data_file and not strategy_data.get("mode"):
                strategy_data["mode"] = "parquet_file"
        return strategy_data

    def _save_strategy_live(self) -> None:
        """每次 best_formula 更新时立即保存 strategy json。
        即使训练中途进程被杀（OOM/终端回收/Ctrl+C），也能保留最新最优公式。

        主文件受保护时，每个 engine/run 使用独立候选路径并持续更新；
        可绑定的候选不会自动删除。原子写入（tmp + os.replace），异常打印告警。
        """
        # 岛模式的岛没有 target_symbol（策略由 IslandAlphaEngine 统一保存），
        # 不能让它写通用策略文件
        if self.best_formula is None or not self.target_symbol or not math.isfinite(self.best_score):
            return
        try:
            save_path = _strategy_file_for_symbol(self.target_symbol, self.timeframe)
            name = pathlib.Path(save_path).name
            payload = self._live_strategy_payload()
            # This guard checks scoring/features/context BEFORE comparing scores.
            reason = save_strategy_artifact(save_path, payload)
            if reason is not None and reason not in ("no_champion", "score", "unchanged"):
                cand = write_candidate_strategy(save_path, payload, stamp=self.candidate_run_id)
                # Throttle only the notice, never writes or retries after a failure.
                if cand and not getattr(self, '_context_block_warned', False):
                    tqdm.write(f"[策略保护] 未覆盖 {name}（{STRATEGY_SAVE_REASONS.get(reason, reason)}）；"
                               f"最新结果实时另存至 {pathlib.Path(cand).name}（本轮持续更新同一文件，结束时不再另存）")
                    self._context_block_warned = True
            elif reason == "score" and not getattr(self, '_strategy_downgrade_warned', False):
                tqdm.write(f"[续训保护] 既有策略 {name} 在相同上下文下分数不低于本轮结果，不覆盖策略文件。")
                self._strategy_downgrade_warned = True
        except Exception as exc:  # noqa: BLE001
            # 静默吞掉会让用户误以为策略已保存，实则没有
            try:
                tqdm.write(f"[警告] 策略保存失败: {exc}")
            except Exception:
                pass

    # ── Checkpoint save / load ────────────────────────────────────────────────

    def training_context(self) -> dict:
        from config import Config
        from . import features, vm
        identity = getattr(self, "context_symbol", None) or self.target_symbol
        if identity is None:
            identity = getattr(self.data_manager, "symbol", None)
        return {
            "symbol": identity,
            "timeframe": self.timeframe or getattr(self.data_manager, "timeframe", None),
            "data_fingerprint": self._data_fingerprint,
            "evaluation": {
                "scoring_version": _current_scoring_version(),
                "target_return_policy": "observed-open-t+1-to-t+2-v1",
                "feature_semantics_version": _feature_semantics_version(),
                "feature_names": list(features.FEATURE_NAMES),
                "normalization_version": getattr(vm, "NORMALIZATION_VERSION", "legacy"),
                "normalization_window": getattr(vm, "NORMALIZATION_WINDOW", 500),
                "vocab_version": VOCAB_VERSION,
                "n_folds": self.n_folds,
                "cost_rate": float(self.bt.cost_rate),
                "periods_per_year": self._evaluation_periods_per_year(),
                "min_trade_exposure": float(Config.MIN_TRADE_EXPOSURE),
                "parameters": {key: getattr(ModelConfig, key) for key in (
                    "WF_GAP", "REWARD_MODE", "REWARD_ALPHA", "IC_GATE_THRESH",
                    "IC_GATE_MULT", "IC_NEG_MULT", "BETA_NEUTRAL_PENALTY",
                    "HALF_CONSISTENCY_BONUS", "BETA_NEUTRAL_THRESH",
                    "BETA_NEUTRAL_LIGHT_THRESH", "MAX_FORMULA_LEN")},
            },
        }

    def _evaluation_periods_per_year(self):
        raw = getattr(self.data_manager, "raw_dict", None) or {}
        if raw.get("time") is not None:
            return estimate_periods_per_year(raw["time"])
        return self.bt.periods_per_year

    def validate_checkpoint_context(self, ckpt: dict) -> bool:
        """Reject wrong identity before mutation; True means weight-only hotstart."""
        artifact_version = ckpt.get("vocab_version")
        if artifact_version is None:
            raise VocabVersionMismatchError("检查点缺少词表版本，需重新训练")
        FORMULA_VOCAB.verify(artifact_version)
        current = self.training_context()
        saved = ckpt.get("training_context")
        if saved is None:
            if current["symbol"] is not None or current["timeframe"] is not None:
                raise CheckpointIdentityError("检查点缺少品种/周期身份，无法确认续训对象")
            return True
        for key in ("symbol", "timeframe"):
            if saved.get(key) != current.get(key):
                raise CheckpointIdentityError(f"检查点 {key} mismatch：与当前品种/周期不一致")
        return (saved != current or ckpt.get("scoring_version") != _current_scoring_version())

    def _clear_rankings(self, reset_stops=False):
        self.best_score, self.best_formula, self._best_snapshot = -float("inf"), None, None
        self.factor_pool, self._elite_pool = [], []
        self._factor_pool_counter = self._elite_counter = 0
        self._reward_ema, self._reward_ema_step = None, 0
        self.training_history = {key: [] for key in self.training_history}
        self._low_entropy_streak, self._previous_init_dist = 0, None
        if reset_stops:
            self._restart_count = self._stag_windows_no_gain = self._stagnation_steps = 0
            self._best_update_step = self.completed_steps
            self._last_stag_restart_step = None
            self.stopped_early = self.user_stopped = False

    def checkpoint_state(self, step: int) -> dict:
        """返回可供单引擎和岛模式复用的完整续训状态。"""
        return {
            "step":                 step,
            "scoring_version":      _current_scoring_version(),
            "feature_semantics_version": _feature_semantics_version(),
            "training_context":     self.training_context(),
            "rng_state":            rng_state(),
            "low_entropy_streak":   self._low_entropy_streak,
            "previous_init_dist":   self._previous_init_dist,
            "last_stag_restart_step": self._last_stag_restart_step,
            "vocab_version":        VOCAB_VERSION,
            "model_state_dict":     self.model.state_dict(),
            "optimizer_state_dict": self.opt.state_dict(),
            "best_score":           self.best_score,
            "best_formula":         self.best_formula,
            "best_snapshot":        self._best_snapshot,
            "factor_pool":          self.factor_pool,
            "factor_pool_counter":  self._factor_pool_counter,
            "elite_pool":           self._elite_pool,
            "elite_counter":        self._elite_counter,
            "restart_count":        self._restart_count,
            "best_update_step":     self._best_update_step,
            "stagnation_steps":     self._stagnation_steps,
            "stag_windows_no_gain": self._stag_windows_no_gain,
            "stopped_early":        self.stopped_early,
            "reward_ema":           self._reward_ema,
            "reward_ema_step":      self._reward_ema_step,
            "training_history":     {
                k: v for k, v in self.training_history.items()
                if k != "_low_entropy_streak"
            },
        }

    def restore_checkpoint_state(self, ckpt: dict) -> int:
        """恢复 checkpoint_state() 写入的训练状态，返回已完成步数。"""
        hotstart = self.validate_checkpoint_context(ckpt)

        self.model.load_state_dict(ckpt["model_state_dict"], strict=True)
        self.opt.load_state_dict(ckpt["optimizer_state_dict"])
        self.best_score = ckpt.get("best_score", -float("inf"))
        self.best_formula = ckpt.get("best_formula")
        self._best_snapshot = ckpt.get("best_snapshot")
        self.factor_pool = ckpt.get("factor_pool", [])
        self._factor_pool_counter = ckpt.get("factor_pool_counter", 0)
        self._elite_pool = self._dedup_elite_pool(ckpt.get("elite_pool", []))
        self._elite_counter = ckpt.get("elite_counter", 0)
        self._restart_count = ckpt.get("restart_count", 0)
        self._best_update_step = ckpt.get("best_update_step", 0)
        self._stagnation_steps = ckpt.get("stagnation_steps", 0)
        self._stag_windows_no_gain = ckpt.get("stag_windows_no_gain", 0)
        self.stopped_early = bool(ckpt.get("stopped_early", False))
        self._reward_ema = ckpt.get("reward_ema")
        self._reward_ema_step = ckpt.get("reward_ema_step", 0)
        for k, v in ckpt.get("training_history", {}).items():
            self.training_history[k] = v
        self.completed_steps = int(ckpt.get("step", 0))
        self._low_entropy_streak = int(ckpt.get("low_entropy_streak", 0))
        self._previous_init_dist = ckpt.get("previous_init_dist")
        self._last_stag_restart_step = ckpt.get("last_stag_restart_step")
        self.user_stopped = False
        if hotstart:
            # Preserve weights, optimizer and completed-step accounting, but
            # refreshed bars/evaluation cannot inherit scores or hard stops.
            refresh = ckpt.get("training_context") != self.training_context()
            self._clear_rankings(reset_stops=refresh)
            print("[Hotstart] Preserved learned weights; cleared stale rankings/pools/history")
        else:
            restore_rng(ckpt.get("rng_state"))
        self._hotstarted = hotstart
        return self.completed_steps

    def save_checkpoint(self, step: int, path: str | None = None) -> str:
        ckpt_dir = checkpoint_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # 检查点文件名带周期（ckpt_{symbol}_{tf}_step_*.pt），不同周期互不覆盖；
        # 同时作为清理前缀，保证只清理本品种本周期的检查点。
        tag = self._file_tag()
        sym_tag = f"_{tag}" if tag else ""
        if path is None:
            path = str(ckpt_dir / f"ckpt{sym_tag}_step_{step:04d}.pt")
        ckpt = self.checkpoint_state(step)
        path = str(root_path(path)) if not pathlib.Path(path).is_absolute() else str(path)
        atomic_write(path, lambda tmp: torch.save(ckpt, tmp))
        # 只保留同品种最近 KEEP_CHECKPOINTS 个检查点（每个约 170MB），
        # 防止 checkpoints/ 无限膨胀；刚写入的这条是最新的一定保留。
        try:
            keep = max(1, int(getattr(ModelConfig, "KEEP_CHECKPOINTS", 3)))
            prefix = f"ckpt{sym_tag}_step_"
            olds = sorted(
                ckpt_dir.glob(f"{prefix}*.pt"),
                key=checkpoint_step,
            )
            for old in olds[:-keep]:
                old.unlink(missing_ok=True)
        except OSError as exc:
            tqdm.write(f"[警告] 清理旧检查点失败: {exc}")
        return path

    def load_checkpoint(self, path: str) -> int:
        # Only failures known to precede mutation may trigger automatic fallback.
        try:
            ckpt = torch.load(path, map_location=ModelConfig.DEVICE, weights_only=True)
        except (pickle.UnpicklingError, EOFError, UnicodeDecodeError):
            raise CheckpointPreflightError("检查点无法安全解码（损坏或含不支持的对象）") from None
        except RuntimeError as exc:
            if any(marker in str(exc) for marker in (
                    "PytorchStreamReader failed reading zip archive",
                    "invalid header or archive is corrupted", "Invalid magic number")):
                raise CheckpointPreflightError("检查点存档损坏，无法安全解码") from None
            raise
        if not isinstance(ckpt, dict) or "islands" in ckpt:
            raise ValueError("指定文件不是单引擎检查点，已保留原文件")
        try:
            self.validate_checkpoint_context(ckpt)
        except VocabVersionMismatchError:
            raise CheckpointPreflightError("检查点词表版本缺失或不兼容") from None

        try:
            completed = self.restore_checkpoint_state(ckpt)
        except Exception as exc:
            # load_state_dict/optimizer/history/RNG may already have mutated us.
            # Never let even a ValueError here masquerade as safe prevalidation.
            raise CheckpointRestoreError(
                f"检查点状态恢复失败（{type(exc).__name__}），训练已中止；保留原文件，请勿复用此引擎"
            ) from None
        tqdm.write(f"[检查点] 已从 {path} 恢复。"
                   f" 当前步={completed}  最优={self.best_score:.4f}"
                   f"  精英池={len(self._elite_pool)}（去重后）")
        return completed

    # ── Decode formula tokens to readable string ──────────────────────────────

    def _decode_formula(self, tokens: list[int] | None) -> str:
        if tokens is None:
            return "无"
        from .vocab import FORMULA_VOCAB
        names = FORMULA_VOCAB.token_names
        return " -> ".join(names[t] if 0 <= t < len(names) else f"?{t}"
                           for t in tokens)

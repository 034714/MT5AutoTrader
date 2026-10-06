"""
model_core/island_engine.py — 多起点并行训练（Island Model）

同时维护 N 个独立的 AlphaEngine（island），每个 island 独立探索不同的
公式空间区域。每隔 migration_interval 步，把所有 island 的 elite 公式
汇总，取 Top-K 注入到其他 island 的 elite pool 中，实现"精英迁移"。

注意：这里的"并行"是算法层面的多群体演化，不是 Python multiprocessing。
CPU 训练下串行轮流训练每个 island 一个小阶段效率更高，且输出不混乱。
"""
import copy
import os
import random
from pathlib import Path

import torch

from .config import ModelConfig
from .engine import AlphaEngine, _current_scoring_version
from utils.training_artifacts import atomic_json_write, atomic_write, checkpoint_dir, checkpoint_step, history_path, json_safe, root_path
from utils.training_context import rng_state, restore_rng


class IslandAlphaEngine:
    """管理多个 AlphaEngine 组成 island population。"""

    def __init__(self, data_manager, n_islands: int | None = None,
                 migration_interval: int | None = None,
                 migration_top_k: int | None = None,
                 base_seed: int | None = None):
        self.data_manager = data_manager
        self.n_islands = n_islands or ModelConfig.N_ISLANDS
        self.migration_interval = migration_interval or ModelConfig.MIGRATION_INTERVAL
        self.migration_top_k = migration_top_k or ModelConfig.MIGRATION_TOP_K

        self.islands: list[AlphaEngine] = []
        seed0 = base_seed if base_seed is not None else 2026
        for i in range(self.n_islands):
            # Seed before construction: model, AdamW, LoRD and rank monitor must
            # all refer to the same parameters (never replace the model alone).
            torch.manual_seed(seed0 + i * 17)
            isl = AlphaEngine(data_manager=data_manager)
            self.islands.append(isl)

        self.global_best_score = -float('inf')
        self.global_best_formula = None
        self.global_best_island = -1
        self._step = 0
        self._phase_interrupted = False
        self.checkpoint_tag: str | None = None
        self.global_history = {"step": [], "best_score": []}

    def tag_islands(self, symbol: str, timeframe=None, data_file=None, mode=None):
        """入口脚本调用：为每个岛设置独立的训练曲线文件名与元数据。

        文件名标签 = 品种_周期（与单引擎命名一致），不同周期互不覆盖；
        各岛训练曲线存 training_history_{tag}__islN.json，互不覆盖；
        刻意不设 target_symbol——岛不写策略文件、不写检查点，
        策略由本类在训练结束后统一保存。
        """
        tag = f"{symbol}_{timeframe}" if timeframe else symbol
        self.checkpoint_tag = tag
        for i, isl in enumerate(self.islands):
            isl.history_tag = f"{tag}__isl{i + 1}"
            isl.context_symbol = symbol
            # 岛模式由管理器写复合检查点，单岛不写自己的检查点，避免互相覆盖
            isl.save_checkpoints = False
            if timeframe is not None:
                isl.timeframe = timeframe
            if data_file is not None:
                isl.data_file = data_file
            if mode is not None:
                isl.mode = mode

    def save_checkpoint(self, step: int, path: str | None = None) -> str:
        """Save a completed phase or an interrupted phase with per-island counts."""
        ckpt_dir = checkpoint_dir()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tag = self.checkpoint_tag or "unknown"
        if path is None:
            path = str(ckpt_dir / f"island_ckpt_{tag}_step_{step:04d}.pt")
        payload = {
            "kind": "island",
            "scoring_version": _current_scoring_version(),
            "training_context": self.islands[0].training_context(),
            "rng_state": rng_state(),
            "step": step,
            "phase_interrupted": self._phase_interrupted,
            "vocab_version": self.islands[0].checkpoint_state(step)["vocab_version"],
            "n_islands": self.n_islands,
            "migration_interval": self.migration_interval,
            "migration_top_k": self.migration_top_k,
            "global_best_score": self.global_best_score,
            "global_best_formula": self.global_best_formula,
            "global_best_island": self.global_best_island,
            "global_history": self.global_history,
            "islands": [isl.checkpoint_state(isl.completed_steps) for isl in self.islands],
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
        }
        path = str(root_path(path)) if not Path(path).is_absolute() else str(path)
        atomic_write(path, lambda tmp: torch.save(payload, tmp))
        keep = max(1, int(getattr(ModelConfig, "KEEP_CHECKPOINTS", 3)))
        prefix = f"island_ckpt_{tag}_step_"
        files = sorted((ckpt_dir / p for p in os.listdir(ckpt_dir)
                        if p.startswith(prefix) and p.endswith(".pt")),
                       key=checkpoint_step)
        for old in files[:-keep]:
            old.unlink(missing_ok=True)
        print(f"[岛检查点] → {path}（步数={step}，保留最近 {keep} 个）")
        return path

    def load_checkpoint(self, path: str) -> int:
        """恢复复合岛检查点，要求岛数和迁移设置一致。"""
        ckpt = torch.load(path, map_location=ModelConfig.DEVICE, weights_only=True)
        if ckpt.get("kind") != "island":
            raise ValueError(f"不是岛模式检查点: {path}")
        if int(ckpt.get("n_islands", 0)) != self.n_islands:
            raise ValueError("检查点岛数与本次训练不一致")
        if int(ckpt.get("migration_interval", 0)) != self.migration_interval:
            raise ValueError("检查点迁移间隔与本次训练不一致")
        self.islands[0].validate_checkpoint_context(ckpt)
        states = ckpt.get("islands", [])
        if len(states) != self.n_islands:
            raise ValueError("Checkpoint island state count mismatch")
        if int(ckpt.get("migration_top_k", 0)) != self.migration_top_k:
            raise ValueError("Checkpoint migration top-k mismatch")
        hotstart = any(isl.validate_checkpoint_context(state) for isl, state in zip(self.islands, states))
        # Validate all identities first: a later wrong island cannot partially load.
        for isl, state in zip(self.islands, states):
            isl.validate_checkpoint_context(state)
        for isl, state in zip(self.islands, states):
            isl.restore_checkpoint_state(state)
        self.global_best_score = ckpt.get("global_best_score", -float("inf"))
        self.global_best_formula = ckpt.get("global_best_formula")
        self.global_best_island = int(ckpt.get("global_best_island", -1))
        if not hotstart:
            restore_rng(ckpt.get("rng_state"))
        self.global_history = ckpt.get("global_history", {"step": [], "best_score": []})
        if hotstart or ckpt.get("scoring_version") != _current_scoring_version():
            self.global_best_score = -float('inf')
            self.global_best_formula = None
            self.global_best_island = -1
            self.global_history = {"step": [], "best_score": []}
            self._update_global_best()
        step = int(ckpt.get("step", 0))
        self._step = step
        self._phase_interrupted = bool(ckpt.get("phase_interrupted", False))
        print(f"[岛检查点] 已恢复 {path}：步数={step}，全局最优={self.global_best_score:.4f}")
        return step

    def _migrate_elites(self, step: int):
        """在所有 islands 之间交换 Top-K elite 公式。"""
        # 收集所有 island 的 elite
        all_elites = []
        for isl in self.islands:
            all_elites.extend(isl._elite_pool)

        if len(all_elites) < 2:
            return

        # 去重：相同公式保留最高分和最新 birth_step
        best_by_formula = {}
        for sc, cnt, toks, birth in all_elites:
            key = str(toks)
            if key not in best_by_formula or sc > best_by_formula[key][0]:
                best_by_formula[key] = (sc, cnt, toks, birth)

        # 按得分排序取 Top-K
        sorted_elites = sorted(
            best_by_formula.values(), key=lambda x: x[0], reverse=True
        )
        top_elites = sorted_elites[:self.migration_top_k]

        # 注入到每个 island（替换低分 elite）
        injected = 0
        for isl in self.islands:
            for sc, cnt, toks, birth in top_elites:
                # 避免注入 island 已存在的公式（_update_elite_pool 会处理去重）
                isl._update_elite_pool(sc, list(toks), step)
                injected += 1

        print(f"\n[Island Migration @ step {step}] "
              f"collected {len(all_elites)} elites, "
              f"deduped to {len(best_by_formula)}, "
              f"injected {injected} top elites across {self.n_islands} islands\n")

    def _update_global_best(self):
        """从所有 island 中更新全局最优。"""
        for i, isl in enumerate(self.islands):
            if isl.best_score > self.global_best_score:
                self.global_best_score = isl.best_score
                self.global_best_formula = isl.best_formula
                self.global_best_island = i

    def train(self, start_step: int = 0):
        """主训练循环：每个 island 轮流训练一个阶段，然后迁移 elite。"""
        total_steps = ModelConfig.TRAIN_STEPS
        if start_step >= total_steps:
            if self._phase_interrupted:
                raise ValueError("Interrupted island phase requires a target above its phase-start step")
            print(f"[岛训练] 起始步 {start_step} 已达目标步 {total_steps}，无需继续训练。")
            return
        if self._phase_interrupted and any(isl.completed_steps > total_steps for isl in self.islands):
            raise ValueError("Requested target is below an interrupted island completed step")
        for isl in self.islands:
            isl.user_stopped = False
        # 检查点是在所有岛完成、迁移和同步后保存的，即使目标步数不是
        # migration_interval 的整数倍也可以安全续训；下一阶段从当前步继续。
        interval = max(1, int(self.migration_interval))
        n_phases = max(1, (total_steps - start_step + interval - 1) // interval)
        phase_no = start_step // interval + 1

        print(f"\n{'='*60}")
        print(f"  Island Alpha Training")
        print(f"  islands={self.n_islands}  migration_every={self.migration_interval}")
        print(f"  total_steps={total_steps}  remaining_phases={n_phases}")
        print(f"{'='*60}\n")

        start = start_step
        while start < total_steps:
            end = min(((start // interval) + 1) * interval, total_steps)
            phase_label = f"第{phase_no}阶段"

            for i, isl in enumerate(self.islands):
                if isl.stopped_early:
                    print(f"\n>>> {phase_label} — Island {i+1}/{self.n_islands} "
                          "已因长期无改进停止，跳过")
            active = [(i, isl) for i, isl in enumerate(self.islands)
                      if not isl.stopped_early]
            if not active:
                print("[岛训练] 所有岛均因长期无改进停止，结束训练。")
                break
            for i, _ in active:
                print(f"\n>>> {phase_label} — Island {i+1}/{self.n_islands} "
                      f"steps [{start}:{end}]")

            def _run_phase(isl: AlphaEngine) -> None:
                # 每岛独立训练一个阶段（岛间互不共享可变状态，可安全并行）
                isl.train(start_step=max(start, isl.completed_steps), end_step=end,
                          migration_hook=None, verbose_header=False,
                          run_end_step=total_steps, run_start_step=start_step)

            if ModelConfig.ISLAND_PARALLEL and len(active) > 1:
                # 岛级并行：PyTorch CPU 算子释放 GIL，整岛粒度可真并行；
                # 三个岛同时跑，阶段墙钟时间 ≈ 最慢一岛，而非三岛之和。
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=len(active)) as pool:
                    futures = [pool.submit(_run_phase, isl) for _, isl in active]
                    for fut in futures:
                        fut.result()  # 任一岛异常则向上抛，终止训练
            else:
                for _, isl in active:
                    _run_phase(isl)
            self._update_global_best()

            if any(getattr(isl, "user_stopped", False) for isl in self.islands):
                self._step = start
                self._phase_interrupted = True
                self.save_checkpoint(start)
                self._save_history()
                print("[岛训练] Stopped safely; per-island completed steps preserved")
                return

            # 阶段结束：迁移 elite
            self._migrate_elites(end)

            # 同步全局最优到每个 island 的 best_snapshot
            # 这样下次 restart 时可以从全局最优恢复，而非局部最优
            # P1-2 修复：同时同步 _best_snapshot，否则 restart 时加载的仍是
            # island 局部 snapshot，而非全局最优——与注释承诺不符。
            best_isl_idx = self.global_best_island
            if best_isl_idx >= 0 and best_isl_idx < len(self.islands):
                best_isl = self.islands[best_isl_idx]
                for isl in self.islands:
                    if self.global_best_score > isl.best_score:
                        isl.best_score = self.global_best_score
                        isl.best_formula = copy.deepcopy(self.global_best_formula)
                        # 同步模型 snapshot，restart 时能从全局最优恢复
                        if best_isl._best_snapshot is not None:
                            isl._best_snapshot = copy.deepcopy(best_isl._best_snapshot)

            # 只有所有岛完成、迁移和全局最优同步都落定后才保存，续训状态一致。
            self._step = end
            self._phase_interrupted = False
            self.global_history["step"].append(end)
            self.global_history["best_score"].append(self.global_best_score)
            self.save_checkpoint(end)
            self._save_history()
            start = end
            phase_no += 1

        # Named strategies are saved by train_island_from_file through the shared guard.
        self._update_global_best()
        self._save_history()

    def _save_history(self):
        if self.checkpoint_tag:
            atomic_json_write(history_path(f"{self.checkpoint_tag}_island"), json_safe(self.global_history))

    def get_global_best(self):
        return self.global_best_formula, self.global_best_score

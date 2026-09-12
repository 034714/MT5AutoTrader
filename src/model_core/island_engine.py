"""
model_core/island_engine.py — 多起点并行训练（Island Model）

同时维护 N 个独立的 AlphaEngine（island），每个 island 独立探索不同的
公式空间区域。每隔 migration_interval 步，把所有 island 的 elite 公式
汇总，取 Top-K 注入到其他 island 的 elite pool 中，实现"精英迁移"。

注意：这里的"并行"是算法层面的多群体演化，不是 Python multiprocessing。
CPU 训练下串行轮流训练每个 island 一个小阶段效率更高，且输出不混乱。
"""
import copy
import heapq
import json
import os
import random
from pathlib import Path

import torch

from .config import ModelConfig
from .engine import AlphaEngine


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
            isl = AlphaEngine(data_manager=data_manager)
            # 给每个 island 不同的随机初始化，增加多样性
            torch.manual_seed(seed0 + i * 17)
            isl.model = isl.model.__class__().to(ModelConfig.DEVICE)
            isl.opt = torch.optim.AdamW(isl.model.parameters(), lr=1e-3)
            self.islands.append(isl)

        self.global_best_score = -float('inf')
        self.global_best_formula = None
        self.global_best_island = -1
        self._step = 0
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
            # 岛模式由管理器写复合检查点，单岛不写自己的检查点，避免互相覆盖
            isl.save_checkpoints = False
            if timeframe is not None:
                isl.timeframe = timeframe
            if data_file is not None:
                isl.data_file = data_file
            if mode is not None:
                isl.mode = mode

    def save_checkpoint(self, step: int, path: str | None = None) -> str:
        """在完整迁移阶段结束后保存所有岛的复合状态。"""
        ckpt_dir = Path("checkpoints")
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        tag = self.checkpoint_tag or "unknown"
        if path is None:
            path = str(ckpt_dir / f"island_ckpt_{tag}_step_{step:04d}.pt")
        payload = {
            "kind": "island",
            "step": step,
            "vocab_version": self.islands[0].checkpoint_state(step)["vocab_version"],
            "n_islands": self.n_islands,
            "migration_interval": self.migration_interval,
            "migration_top_k": self.migration_top_k,
            "global_best_score": self.global_best_score,
            "global_best_formula": self.global_best_formula,
            "global_best_island": self.global_best_island,
            "global_history": self.global_history,
            "islands": [isl.checkpoint_state(step) for isl in self.islands],
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": random.getstate(),
        }
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        keep = max(1, int(getattr(ModelConfig, "KEEP_CHECKPOINTS", 3)))
        prefix = f"island_ckpt_{tag}_step_"
        files = sorted((ckpt_dir / p for p in os.listdir(ckpt_dir)
                        if p.startswith(prefix) and p.endswith(".pt")),
                       key=lambda p: p.stat().st_mtime)
        for old in files[:-keep]:
            old.unlink(missing_ok=True)
        print(f"[岛检查点] → {path}（步数={step}，保留最近 {keep} 个）")
        return path

    def load_checkpoint(self, path: str) -> int:
        """恢复复合岛检查点，要求岛数和迁移设置一致。"""
        ckpt = torch.load(path, map_location=ModelConfig.DEVICE)
        if ckpt.get("kind") != "island":
            raise ValueError(f"不是岛模式检查点: {path}")
        if int(ckpt.get("n_islands", 0)) != self.n_islands:
            raise ValueError("检查点岛数与本次训练不一致")
        if int(ckpt.get("migration_interval", 0)) != self.migration_interval:
            raise ValueError("检查点迁移间隔与本次训练不一致")
        for isl, state in zip(self.islands, ckpt.get("islands", [])):
            isl.restore_checkpoint_state(state)
        self.global_best_score = ckpt.get("global_best_score", -float("inf"))
        self.global_best_formula = ckpt.get("global_best_formula")
        self.global_best_island = int(ckpt.get("global_best_island", -1))
        if ckpt.get("torch_rng_state") is not None:
            torch.set_rng_state(ckpt["torch_rng_state"])
        if ckpt.get("python_rng_state") is not None:
            random.setstate(ckpt["python_rng_state"])
        self.global_history = ckpt.get("global_history", {"step": [], "best_score": []})
        step = int(ckpt.get("step", 0))
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
            print(f"[岛训练] 起始步 {start_step} 已达目标步 {total_steps}，无需继续训练。")
            return
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

            active = [(i, isl) for i, isl in enumerate(self.islands)
                      if not isl.stopped_early]
            for i, isl in enumerate(self.islands):
                if isl.stopped_early:
                    print(f"\n>>> {phase_label} — Island {i+1}/{self.n_islands} "
                          "已因长期无改进停止，跳过")
            for i, _ in active:
                print(f"\n>>> {phase_label} — Island {i+1}/{self.n_islands} "
                      f"steps [{start}:{end}]")

            def _run_phase(isl: AlphaEngine) -> None:
                # 每岛独立训练一个阶段（岛间互不共享可变状态，可安全并行）
                isl.train(start_step=start, end_step=end,
                          migration_hook=None, verbose_header=False)

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
            self.global_history["step"].append(end)
            self.global_history["best_score"].append(self.global_best_score)
            self.save_checkpoint(end)
            start = end
            phase_no += 1

        # 最终保存全局最优
        self._update_global_best()
        if self.global_best_formula is not None:
            from .vocab import VOCAB_VERSION
            strategy_data = {
                "vocab_version": VOCAB_VERSION,
                "formula": self.global_best_formula,
                "best_score": self.global_best_score,
                "island_engine": True,
                "n_islands": self.n_islands,
            }
            save_path = Path("strategies") / "best_island_strategy.json"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as fp:
                json.dump(strategy_data, fp, indent=2)
            print(f"\n✓ Island training completed!")
            print(f"  Global best score : {self.global_best_score:.4f}")
            print(f"  From island       : {self.global_best_island + 1}")
            print(f"  Formula           : {self.global_best_formula}")
            sample_island = self.islands[self.global_best_island] if self.global_best_island >= 0 else self.islands[0]
            readable = sample_island._decode_formula(self.global_best_formula)
            print(f"  Readable          : {readable}")
            print(f"  Saved to          : {save_path}")

    def get_global_best(self):
        return self.global_best_formula, self.global_best_score

"""
train_island.py — 岛模式训练（多起点并行 + 精英迁移）

用法:
    python train_island.py --data-file D:\\K线数据\\BTCUSD__H1.parquet [--islands 3] [--steps 3000]

与 train_file.py 的单引擎区别：同时维护 N 个独立种群各自探索，每隔
MIGRATION_INTERVAL 步互换精英公式，专治单引擎"过早收敛→反复重启"的
停滞问题。代价：耗时约为单引擎的 N 倍（串行轮流训练）。

岛模式为多群体搜索，支持从最近一次完整迁移阶段续训；已有 best_{symbol}.json 会作为最终保存保护，
低分新结果不会覆盖旧策略。
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.train_logging import configure_train_stdio

configure_train_stdio()

from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
from model_core.config import ModelConfig
from model_core.island_engine import IslandAlphaEngine
from train_file import _save_strategy


def train_island_from_file(
    data_file: str, *, n_islands: int | None = None, from_scratch: bool = False,
) -> IslandAlphaEngine | None:
    info = inspect_parquet_file(data_file)
    symbol = info["symbol"]
    timeframe = info["timeframe"]

    print(f"\n{'='*60}")
    print(f"  Island 岛模式训练 — {info['filename']}")
    print(f"{'='*60}")
    print(f"  品种: {symbol}")
    print(f"  周期: {timeframe}")
    print(f"  岛数: {n_islands or ModelConfig.N_ISLANDS}")
    print(f"  文件: {Path(data_file).resolve()}")
    print(f"  总步数: {ModelConfig.TRAIN_STEPS}（每岛各跑满）")
    print(f"  迁移: 每 {ModelConfig.MIGRATION_INTERVAL} 步互换精英 Top-{ModelConfig.MIGRATION_TOP_K}")
    print(f"{'='*60}")

    try:
        mgr = ParquetDataManager(data_file)
        mgr.load()
        T = mgr.raw_dict["open"].shape[1]
        print(f"  数据加载成功，共 {T} 根K线")
    except Exception as e:
        print(f"  [错误] 数据加载失败: {e}")
        return None

    itrain = IslandAlphaEngine(data_manager=mgr, n_islands=n_islands)
    itrain.tag_islands(
        symbol,
        timeframe=timeframe,
        data_file=str(Path(data_file).resolve()),
        mode="parquet_file",
    )

    # 岛模式检查点仅在完整迁移阶段结束后写入；自动续训从最近阶段恢复。
    ckpt_dir = pathlib.Path("checkpoints")
    ckpt_pattern = f"island_ckpt_{symbol}_step_*.pt"
    ckpts = sorted(ckpt_dir.glob(ckpt_pattern)) if ckpt_dir.exists() else []
    # 岛模式使用独立的全局曲线文件，不覆盖单引擎训练历史。
    island_history_path = pathlib.Path(f"training_history_{symbol}_island.json")
    start_step = 0
    if from_scratch:
        for p in ckpts:
            p.unlink(missing_ok=True)
        for p in pathlib.Path(".").glob(f"training_history_{symbol}__isl*.json"):
            p.unlink(missing_ok=True)
        island_history_path.unlink(missing_ok=True)
        print(f"  [从头训练] 已清除 {len(ckpts)} 个岛检查点和岛训练曲线")
    elif ckpts:
        try:
            start_step = itrain.load_checkpoint(str(ckpts[-1]))
            print(f"  [岛续训] 从 {ckpts[-1]} 恢复，起始步={start_step}")
        except Exception as exc:
            print(f"  [警告] 岛检查点加载失败: {exc}，将从头开始")

    # 岛内不设置旧策略分数下限：否则界面从第 1 步起永远显示旧高分，
    # 无法判断新一轮搜索有没有真实进展。旧策略保护只在最终 _save_strategy
    # 时执行，低分新结果绝不会覆盖磁盘上的已有策略。
    strat_path = pathlib.Path("strategies") / f"best_{symbol}.json"
    if strat_path.exists():
        try:
            old_score = json.loads(strat_path.read_text(encoding="utf-8")).get("best_score")
            if old_score is not None:
                print(f"  [旧策略保护] 当前已有最优 {float(old_score):.4f}；岛内独立搜索，"
                      "仅在最终结果更高时覆盖")
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass

    t0 = time.time()
    itrain.train(start_step=start_step)
    elapsed = time.time() - t0
    # 给训练页提供岛模式的全局最佳曲线（按迁移阶段一条点），不混入单岛曲线。
    if itrain.global_history.get("step"):
        island_history_path.write_text(
            json.dumps({
                "step": itrain.global_history["step"],
                "best_score": itrain.global_history["best_score"],
            }, ensure_ascii=False), encoding="utf-8",
        )

    # island_engine 内部会写一个固定名 best_island_strategy.json（多品种会互相
    # 覆盖且无消费方）；真正的策略由下面按品种保存，删掉避免误导
    try:
        pathlib.Path("strategies", "best_island_strategy.json").unlink(missing_ok=True)
    except OSError:
        pass

    # 用全局最优所在的岛引擎对象保存 best_{symbol}.json
    # （复用 train_file._save_strategy 的"磁盘更优不覆盖"守卫）
    best_idx = itrain.global_best_island if itrain.global_best_island >= 0 else 0
    best_isl = itrain.islands[best_idx]
    if best_isl.best_formula is None:
        print("  [警告] 所有岛均未找到有效公式，跳过策略保存")
        return itrain
    best_isl.target_symbol = symbol  # 仅为通过 _save_strategy 的属性访问
    _save_strategy(best_isl, symbol, timeframe, data_file)
    best_isl.target_symbol = None
    print(f"\n<<< [{symbol}] 岛模式训练完成: 全局最优分数={itrain.global_best_score:.4f}，"
          f"来自岛 {best_idx + 1}，耗时 {elapsed/3600:.2f} 小时")
    return itrain


if __name__ == "__main__":
    ModelConfig.REWARD_MODE = "ftmo"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--islands", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--from-scratch", action="store_true")
    args = parser.parse_args()

    if args.steps > 0:
        ModelConfig.TRAIN_STEPS = args.steps
        print(f"[参数] 训练步数 = {ModelConfig.TRAIN_STEPS}")
    n_islands = args.islands if args.islands > 0 else None
    if n_islands is None and ModelConfig.N_ISLANDS <= 1:
        n_islands = 3  # 单岛没有意义，默认 3
    eng = train_island_from_file(
        args.data_file, n_islands=n_islands, from_scratch=args.from_scratch,
    )
    if eng is None:
        sys.exit(1)

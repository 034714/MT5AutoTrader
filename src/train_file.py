"""
train_file.py — 从单个 Parquet K 线文件训练

用法:
    python train_file.py --data-file D:\\K线数据\\AAPL_H1.parquet

文件名格式: {品种}_{周期}.parquet，例如 AAPL_H1.parquet、US30.cash_H1.parquet
"""
from __future__ import annotations

import glob as _glob
import json
import pathlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.train_logging import configure_train_stdio

configure_train_stdio()

from config import Config
from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
from model_core.config import ModelConfig
from model_core.engine import AlphaEngine
from model_core.vocab import VOCAB_VERSION


def train_from_file(data_file: str, *, from_scratch: bool = False) -> AlphaEngine | None:
    info = inspect_parquet_file(data_file)
    symbol = info["symbol"]
    timeframe = info["timeframe"]

    print(f"\n{'='*60}")
    print(f"  AlphaGPT 文件训练 — {info['filename']}")
    print(f"{'='*60}")
    print(f"  品种: {symbol}")
    print(f"  周期: {timeframe}")
    print(f"  数据: 强制离线 Parquet（不连接 MT5）")
    print(f"  文件: {Path(data_file).resolve()}")
    print(f"  训练步数: {ModelConfig.TRAIN_STEPS}")
    print(f"  K线数: {info['bars']}")
    print(f"  模式: {'重新训练（从头）' if from_scratch else '自动续训'}")
    print(f"{'='*60}")

    try:
        mgr = ParquetDataManager(data_file)
        mgr.load()
        T = mgr.raw_dict["open"].shape[1]
        print(f"  数据加载成功，共 {T} 根K线")
    except Exception as e:
        print(f"  [错误] 数据加载失败: {e}")
        return None

    engine = AlphaEngine(data_manager=mgr, target_symbol=symbol)
    engine.timeframe = timeframe
    engine.data_file = str(Path(data_file).resolve())
    engine.mode = "parquet_file"
    engine.train_steps = ModelConfig.TRAIN_STEPS

    ckpt_pattern = str(pathlib.Path("checkpoints") / f"ckpt_{symbol}_step_*.pt")
    ckpt_files = sorted(_glob.glob(ckpt_pattern))
    start_step = 0

    if from_scratch:
        removed = 0
        for p in ckpt_files:
            try:
                pathlib.Path(p).unlink(missing_ok=True)
                removed += 1
            except OSError as e:
                print(f"  [警告] 无法删除检查点 {p}: {e}")
        hist_path = pathlib.Path(f"training_history_{symbol}.json")
        if hist_path.exists():
            try:
                hist_path.unlink()
            except OSError:
                pass
        print(f"  [重新训练] 已清除 {removed} 个检查点，从第 0 步开始")
        # 保留已有最优策略作为分数下限，避免开局弱公式覆盖 strategies/best_*.json
        _seed_best_from_strategy(engine, symbol)
        ckpt_files = []
    elif ckpt_files:
        latest = ckpt_files[-1]
        try:
            start_step = engine.load_checkpoint(latest)
            print(f"  [续训] 从 {latest} 恢复，起始步={start_step}")
        except Exception as e:
            print(f"  [警告] 检查点加载失败: {e}，将从头开始")

    if start_step >= ModelConfig.TRAIN_STEPS:
        print(f"  [完成] {symbol} 已完成全部 {ModelConfig.TRAIN_STEPS} 步，跳过训练")
        _save_strategy(engine, symbol, timeframe, data_file)
        return engine

    if start_step == 0 and not from_scratch:
        hist_path = pathlib.Path(f"training_history_{symbol}.json")
        if hist_path.exists():
            hist_path.unlink()
        print("  [新训] 从第 0 步开始")

    if start_step > 0:
        engine._save_training_history_live()

    engine.train(start_step=start_step)
    _save_strategy(engine, symbol, timeframe, data_file)
    return engine


def _seed_best_from_strategy(engine: AlphaEngine, symbol: str) -> None:
    """把已有 best_{symbol}.json 当作重新训练的分数下限。"""
    path = pathlib.Path("strategies") / f"best_{symbol}.json"
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [警告] 读取已有策略失败: {e}")
        return
    formula = data.get("formula")
    score = data.get("best_score")
    if not formula or score is None:
        return
    try:
        engine.best_formula = [int(t) for t in formula]
        engine.best_score = float(score)
        print(f"  [重新训练] 保留已有最优分数下限={engine.best_score:.4f}，仅更好时才会覆盖策略文件")
    except (TypeError, ValueError) as e:
        print(f"  [警告] 已有策略无法用作下限: {e}")


def _save_strategy(engine: AlphaEngine, symbol: str, timeframe: str, data_file: str) -> None:
    path = pathlib.Path("strategies") / f"best_{symbol}.json"
    path.parent.mkdir(exist_ok=True)
    # 若磁盘上已有更高分，不要用更弱结果覆盖
    if path.exists() and engine.best_formula is not None:
        try:
            old = json.loads(path.read_text(encoding="utf-8"))
            old_score = old.get("best_score")
            if old_score is not None and float(old_score) > float(engine.best_score):
                print(
                    f"  [策略] 保留磁盘更优结果 {float(old_score):.4f} "
                    f"> 本次 {float(engine.best_score):.4f}，未覆盖 {path}"
                )
                merged = dict(old)
                for key, val in (
                    ("timeframe", timeframe),
                    ("data_file", str(Path(data_file).resolve())),
                    ("mode", "parquet_file"),
                    ("train_steps", ModelConfig.TRAIN_STEPS),
                ):
                    if val is not None and not merged.get(key):
                        merged[key] = val
                if merged != old:
                    path.write_text(
                        json.dumps(merged, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    print(f"  [策略] 已补全数据路径等元数据: {path}")
                return
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    data = {
        "vocab_version": VOCAB_VERSION,
        "symbol": symbol,
        "timeframe": timeframe,
        "data_file": str(Path(data_file).resolve()),
        "mode": "parquet_file",
        "formula": engine.best_formula,
        "formula_decoded": engine._decode_formula(engine.best_formula)
        if engine.best_formula
        else None,
        "best_score": engine.best_score,
        "train_steps": ModelConfig.TRAIN_STEPS,
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  策略已保存: {path}")


if __name__ == "__main__":
    ModelConfig.REWARD_MODE = "ftmo"

    if "--data-file" not in sys.argv:
        print("用法: python train_file.py --data-file PATH\\TO\\SYMBOL_TF.parquet [--from-scratch]")
        print("示例: python train_file.py --data-file D:\\K线数据\\AAPL_H1.parquet")
        sys.exit(1)

    idx = sys.argv.index("--data-file")
    if idx + 1 >= len(sys.argv):
        print("错误: --data-file 后需要文件路径")
        sys.exit(1)

    data_file = sys.argv[idx + 1]
    from_scratch = "--from-scratch" in sys.argv
    if "--steps" in sys.argv:
        si = sys.argv.index("--steps")
        if si + 1 < len(sys.argv):
            try:
                ModelConfig.TRAIN_STEPS = int(sys.argv[si + 1])
                print(f"[参数] 训练步数 = {ModelConfig.TRAIN_STEPS}")
            except ValueError:
                print("[参数] --steps 非法，使用默认步数")
    t0 = time.time()
    eng = train_from_file(data_file, from_scratch=from_scratch)
    elapsed = time.time() - t0

    if eng:
        sym = eng.target_symbol or "?"
        print(f"\n<<< [{sym}] 训练完成: 最优分数={eng.best_score:.4f}，耗时 {elapsed/3600:.2f} 小时")
        if eng.best_formula:
            print(f"    {eng._decode_formula(eng.best_formula)}")
    else:
        print("\n<<< 训练失败")
        sys.exit(1)

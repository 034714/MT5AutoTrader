"""
train_file.py — 从单个 Parquet K 线文件训练

用法:
    python train_file.py --data-file D:\\K线数据\\AAPL_H1.parquet
    python train_file.py --data-file ... --steps 200            # 本次再跑 200 步
    python train_file.py --data-file ... --resume-file checkpoints\\ckpt_X_step_0200.pt
    python train_file.py --data-file ... --from-scratch          # 清除检查点、换随机种子从头搜索

文件名格式: {品种}_{周期}.parquet，例如 AAPL_H1.parquet、BTCUSD__H1.parquet
产物文件名带周期（best_{品种}_{周期}.json / ckpt_{品种}_{周期}_step_*.pt /
training_history_{品种}_{周期}.json），不同周期互不覆盖。
"""
from __future__ import annotations

import glob as _glob
import json
import pathlib
import random
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


def file_tag(symbol: str, timeframe: str | None) -> str:
    """品种+周期的文件名标签；无周期时退回纯品种（兼容旧产物）。"""
    return f"{symbol}_{timeframe}" if timeframe else symbol


def _seed_rng(seed: int | None) -> int:
    """设定随机种子并返回实际使用的种子（从头训练时打乱轨迹）。"""
    if seed is None:
        seed = random.SystemRandom().randrange(1, 2**31 - 1)
    import torch
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % (2**32))
    except Exception:
        pass
    return int(seed)


def _latest_ckpt(tag: str) -> Path | None:
    files = sorted(_glob.glob(str(pathlib.Path("checkpoints") / f"ckpt_{tag}_step_*.pt")))
    return Path(files[-1]) if files else None


def train_from_file(
    data_file: str, *, from_scratch: bool = False, additional_steps: int = 0,
    resume_file: str | None = None, seed: int | None = None,
) -> AlphaEngine | None:
    info = inspect_parquet_file(data_file)
    symbol = info["symbol"]
    timeframe = info["timeframe"]
    tag = file_tag(symbol, timeframe)

    print(f"\n{'='*60}")
    print(f"  AlphaGPT 文件训练 — {info['filename']}")
    print(f"{'='*60}")
    print(f"  品种: {symbol}")
    print(f"  周期: {timeframe}")
    print(f"  数据: 强制离线 Parquet（不连接 MT5）")
    print(f"  文件: {Path(data_file).resolve()}")
    if additional_steps > 0:
        print(f"  本次新增步数: {additional_steps}")
    else:
        print(f"  目标总步数: {ModelConfig.TRAIN_STEPS}（未指定新增步数）")
    print(f"  K线数: {info['bars']}")
    print(f"  模式: {'重新训练（从头，换随机种子）' if from_scratch else '自动续训'}")
    print(f"{'='*60}")

    try:
        mgr = ParquetDataManager(data_file)
        mgr.load()
        T = mgr.raw_dict["open"].shape[1]
        print(f"  数据加载成功，共 {T} 根K线")
    except Exception as e:
        print(f"  [错误] 数据加载失败: {e}")
        return None

    # 从头训练：换随机种子，避免每次踩同一条搜索轨迹
    if from_scratch:
        used = _seed_rng(seed)
        print(f"  [随机种子] {used}（从头训练已打乱探索轨迹）")

    engine = AlphaEngine(data_manager=mgr, target_symbol=symbol)
    engine.timeframe = timeframe
    engine.data_file = str(Path(data_file).resolve())
    engine.mode = "parquet_file"

    start_step = 0
    if from_scratch:
        removed = 0
        for p in _glob.glob(str(pathlib.Path("checkpoints") / f"ckpt_{tag}_step_*.pt")):
            try:
                pathlib.Path(p).unlink(missing_ok=True)
                removed += 1
            except OSError as e:
                print(f"  [警告] 无法删除检查点 {p}: {e}")
        hist_path = pathlib.Path(f"training_history_{tag}.json")
        if hist_path.exists():
            try:
                hist_path.unlink()
            except OSError:
                pass
        print(f"  [重新训练] 已清除 {removed} 个检查点，从第 0 步开始")
        # 保留已有最优策略作为分数下限，避免开局弱公式覆盖 strategies/best_*.json
        _seed_best_from_strategy(engine, symbol, timeframe)
    elif resume_file:
        rp = pathlib.Path(resume_file)
        if rp.exists():
            try:
                start_step = engine.load_checkpoint(str(rp))
                print(f"  [续训] 从指定检查点 {rp.name} 恢复，起始步={start_step}")
            except Exception as e:
                print(f"  [警告] 检查点加载失败: {e}，将从头开始")
        else:
            print(f"  [警告] 指定检查点不存在: {resume_file}，将自动寻找最新检查点")
            latest = _latest_ckpt(tag)
            if latest:
                start_step = engine.load_checkpoint(str(latest))
                print(f"  [续训] 从 {latest} 恢复，起始步={start_step}")
    else:
        latest = _latest_ckpt(tag)
        if latest:
            try:
                start_step = engine.load_checkpoint(str(latest))
                print(f"  [续训] 从 {latest} 恢复，起始步={start_step}")
            except Exception as e:
                print(f"  [警告] 检查点加载失败: {e}，将从头开始")

    # 目标总步数：指定了「本次新增步数」则 start+新增；否则沿用默认总步数
    if additional_steps > 0:
        target = start_step + int(additional_steps)
    else:
        target = max(ModelConfig.TRAIN_STEPS, start_step)
    ModelConfig.TRAIN_STEPS = target
    engine.train_steps = target

    if start_step >= target:
        print(f"  [完成] {tag} 已达目标步数 {target}（当前 {start_step}），无需再训；"
              f"想继续请填「本次新增步数」或从头训练")
        _save_strategy(engine, symbol, timeframe, data_file)
        return engine

    if start_step == 0 and not from_scratch:
        hist_path = pathlib.Path(f"training_history_{tag}.json")
        if hist_path.exists():
            hist_path.unlink()
        print("  [新训] 从第 0 步开始")

    if start_step > 0:
        engine._save_training_history_live()

    engine.train(start_step=start_step)
    _save_strategy(engine, symbol, timeframe, data_file)
    return engine


def _seed_best_from_strategy(engine: AlphaEngine, symbol: str, timeframe: str | None = None) -> None:
    """把已有 best_{symbol}_{tf}.json（或旧版 best_{symbol}.json）当作重新训练的分数下限。"""
    path = pathlib.Path("strategies") / f"best_{file_tag(symbol, timeframe)}.json"
    if not path.exists() and timeframe:
        path = pathlib.Path("strategies") / f"best_{symbol}.json"  # 兼容旧命名
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
    path = pathlib.Path("strategies") / f"best_{file_tag(symbol, timeframe)}.json"
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
        print("用法: python train_file.py --data-file PATH\\TO\\SYMBOL_TF.parquet "
              "[--steps N] [--from-scratch] [--resume-file ckpt.pt]")
        print("示例: python train_file.py --data-file D:\\K线数据\\AAPL_H1.parquet --steps 200")
        sys.exit(1)

    idx = sys.argv.index("--data-file")
    if idx + 1 >= len(sys.argv):
        print("错误: --data-file 后需要文件路径")
        sys.exit(1)

    data_file = sys.argv[idx + 1]
    from_scratch = "--from-scratch" in sys.argv
    additional_steps = 0
    if "--steps" in sys.argv:
        si = sys.argv.index("--steps")
        if si + 1 < len(sys.argv):
            try:
                additional_steps = int(sys.argv[si + 1])
                print(f"[参数] 本次新增步数 = {additional_steps}")
            except ValueError:
                print("[参数] --steps 非法，忽略")
    resume_file = None
    if "--resume-file" in sys.argv:
        ri = sys.argv.index("--resume-file")
        if ri + 1 < len(sys.argv):
            resume_file = sys.argv[ri + 1]

    t0 = time.time()
    eng = train_from_file(data_file, from_scratch=from_scratch,
                          additional_steps=additional_steps, resume_file=resume_file)
    elapsed = time.time() - t0

    if eng:
        sym = eng.target_symbol or "?"
        print(f"\n<<< [{sym}] 训练完成: 最优分数={eng.best_score:.4f}，耗时 {elapsed/3600:.2f} 小时")
        if eng.best_formula:
            print(f"    {eng._decode_formula(eng.best_formula)}")
    else:
        print("\n<<< 训练失败")
        sys.exit(1)

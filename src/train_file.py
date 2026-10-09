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

续训规则: 自动从最新检查点恢复；仅预校验身份/词表不兼容或安全解码失败
自动移入 checkpoints/incompatible/ 下的唯一目录，继续尝试更早检查点；
全部被安全拒绝时从第 0 步开始。状态恢复失败则中止并保留原文件。
显式选择错误品种/周期的检查点会拒绝，原文件不归档且不中途改为新训。
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.train_logging import configure_train_stdio

configure_train_stdio()

from config import Config
from data_pipeline.parquet_manager import ParquetDataManager, inspect_parquet_file
from model_core.config import ModelConfig
from model_core.engine import (
    STRATEGY_SAVE_REASONS,
    AlphaEngine,
    CheckpointIdentityError,
    CheckpointPreflightError,
    CheckpointRestoreError,
    _current_scoring_version,
    _feature_semantics_version,
    save_strategy_artifact,
    write_candidate_strategy,
)
from model_core.vocab import VOCAB_VERSION, VocabVersionMismatchError
from utils.training_artifacts import (
    checkpoint_dir,
    checkpoint_step,
    history_path,
    root_path,
    strategy_path,
)


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


def _tag_checkpoints(tag: str) -> list[Path]:
    """本品种本周期全部检查点，按步数从新到旧。"""
    ck_dir = checkpoint_dir()
    if not ck_dir.exists():
        return []
    return sorted(ck_dir.glob(f"ckpt_{tag}_step_*.pt"), key=checkpoint_step, reverse=True)


def _archive_checkpoint(path: Path) -> str | None:
    """把无法续训的检查点移入 checkpoints/incompatible/（不删除，便于人工恢复）。"""
    try:
        archive_root = checkpoint_dir() / "incompatible"
        archive_root.mkdir(parents=True, exist_ok=True)
        # An atomically allocated directory prevents overwrites even when the
        # same checkpoint name is rejected repeatedly or concurrently.
        dest_dir = Path(tempfile.mkdtemp(prefix="checkpoint_", dir=archive_root))
        dest = dest_dir / path.name
        pathlib.Path(path).rename(dest)
        return str(dest)
    except OSError as exc:
        print(f"  [警告] 归档不兼容检查点失败（保留原位）{path.name}: {exc}")
        return None


def _resume_or_fresh(engine: AlphaEngine, tag: str, resume_file: str | None) -> int:
    """自动续训：从新到旧逐个尝试检查点。

    仅「改动引擎状态之前」就能判定的不兼容（身份/词表/安全解码失败）才自动
    归档到 checkpoints/incompatible/ 并继续试更早的；全部被安全拒绝时返回 0
    （从头训练）。状态恢复中途失败说明引擎可能已被部分改写，必须中止整个
    训练而不是继续用被污染的引擎；原文件保留供人工处理。
    """
    explicit_candidate = None
    if resume_file:
        if pathlib.Path(resume_file).name.startswith("island_"):
            print("  [提示] 岛模式已停用：岛检查点无法用于单引擎续训，从第 0 步开始")
            return 0
        rp = pathlib.Path(resume_file)
        if not rp.is_absolute():
            rp = root_path(rp)
        if rp.exists():
            candidates = [rp]
            explicit_candidate = rp
        else:
            print(f"  [警告] 指定检查点不存在: {resume_file}，自动改用本品种最新检查点")
            candidates = _tag_checkpoints(tag)
    else:
        candidates = _tag_checkpoints(tag)
    archived = 0
    for cand in candidates:
        try:
            start_step = engine.load_checkpoint(str(cand))
        except CheckpointRestoreError as exc:
            print(f"  [错误] {exc}")
            raise
        except (CheckpointPreflightError, VocabVersionMismatchError) as exc:
            if cand == explicit_candidate and isinstance(exc, CheckpointIdentityError):
                # A user's explicit checkpoint may belong to another valid run.
                # Reject the selection without relocating it or starting fresh.
                raise
            dest = _archive_checkpoint(cand)
            archived += bool(dest)
            print(f"  [续训] 检查点 {cand.name} 不兼容：{exc}")
            print(f"         已移入 {dest or '归档失败（保留原位）'}，继续尝试更早的检查点")
            continue
        except ValueError as exc:
            # 明确拒绝但文件本身有效（如非单引擎检查点）：保留原文件，跳过
            print(f"  [续训] 跳过 {cand.name}：{exc}")
            continue
        except Exception as exc:  # noqa: BLE001
            # 未知读取失败不做自动归档：保留原位并跳过，避免误移可用文件
            print(f"  [警告] 检查点 {cand.name} 读取失败，已保留原位并跳过：{type(exc).__name__}: {exc}")
            continue
        print(f"  [续训] 从 {cand.name} 恢复，起始步={start_step}")
        return start_step
    if candidates:
        print(f"  [从头训练] 没有可续训的检查点（已归档 {archived} 个不兼容检查点到"
              " checkpoints/incompatible/），从第 0 步开始")
    else:
        print("  [新训] 没有历史检查点，从第 0 步开始")
    return 0


def train_from_file(
    data_file: str, *, from_scratch: bool = False, additional_steps: int = 0,
    resume_file: str | None = None, seed: int | None = None,
) -> AlphaEngine | None:
    # 清掉旧的「停止训练」信号（引擎只读不删，避免岛模式后跑的岛吞掉信号）
    stop_flag = root_path("TRAIN_STOP")
    try:
        stop_flag.unlink(missing_ok=True)
    except OSError:
        pass

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
        for p in checkpoint_dir().glob(f"ckpt_{tag}_step_*.pt"):
            try:
                pathlib.Path(p).unlink(missing_ok=True)
                removed += 1
            except OSError as e:
                print(f"  [警告] 无法删除检查点 {p}: {e}")
        hist_path = history_path(tag)
        if hist_path.exists():
            try:
                hist_path.unlink()
            except OSError:
                pass
        print(f"  [重新训练] 已清除 {removed} 个检查点，从第 0 步开始")
        # 保留已有最优策略作为分数下限，避免开局弱公式覆盖 strategies/best_*.json
        _seed_best_from_strategy(engine, symbol, timeframe)
    else:
        start_step = _resume_or_fresh(engine, tag, resume_file)

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
        hist_path = history_path(tag)
        if hist_path.exists():
            hist_path.unlink()
        print("  [新训] 从第 0 步开始")

    if start_step > 0:
        engine._save_training_history_live()

    engine.train(start_step=start_step)
    _save_strategy(engine, symbol, timeframe, data_file)
    return engine


def _seed_best_from_strategy(engine: AlphaEngine, symbol: str, timeframe: str | None = None) -> None:
    """Seed only a finite champion scored on this exact data/evaluation context."""
    path = strategy_path(symbol, timeframe)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [警告] 读取已有策略失败: {e}")
        return
    if (data.get("scoring_version") != _current_scoring_version()
            or data.get("feature_semantics_version") != _feature_semantics_version()
            or data.get("training_context") != engine.training_context()):
        print(f"  [评分隔离] 不继承旧策略分数；保留原文件 {path}")
        return
    formula = data.get("formula")
    score = data.get("best_score")
    if not formula or score is None:
        return
    try:
        import math
        if not math.isfinite(float(score)):
            return
        engine.best_formula = [int(t) for t in formula]
        engine.best_score = float(score)
        print(f"  [重新训练] 保留已有最优分数下限={engine.best_score:.4f}，仅更好时才会覆盖策略文件")
    except (TypeError, ValueError) as e:
        print(f"  [警告] 已有策略无法用作下限: {e}")


def _save_strategy(engine: AlphaEngine, symbol: str, timeframe: str, data_file: str) -> None:
    path = strategy_path(symbol, timeframe)
    context = engine.training_context()
    for key, expected in (("symbol", symbol), ("timeframe", timeframe)):
        if context.get(key) is not None and context[key] != expected:
            raise ValueError(f"Strategy {key} conflicts with training context")
        context[key] = expected
    data = {
        "scoring_version": _current_scoring_version(),
        "feature_semantics_version": _feature_semantics_version(),
        "vocab_version": VOCAB_VERSION,
        "symbol": symbol,
        "timeframe": timeframe,
        "training_context": context,
        "data_file": str(Path(data_file).resolve()),
        "mode": "parquet_file",
        "formula": engine.best_formula,
        "formula_decoded": engine._decode_formula(engine.best_formula),
        "best_score": engine.best_score,
        "train_steps": ModelConfig.TRAIN_STEPS,
    }
    reason = save_strategy_artifact(path, data)
    if reason is None:
        print(f"  策略已保存: {path}")
    elif reason == "no_champion":
        print("  [策略保护] 本轮没有有效新冠军（无公式或分数非有限），未写策略文件")
    elif reason == "unchanged":
        print(f"  [策略已保存] 本轮冠军已在 {pathlib.Path(path).name}，无需重复写入")
    elif reason == "score":
        try:
            old = float(json.loads(pathlib.Path(path).read_text(encoding="utf-8")).get("best_score"))
        except (OSError, ValueError, TypeError):
            old = float("nan")
        print(f"  [策略保护] 已有策略分数不低于本轮结果（{old:.4f} ≥ {engine.best_score:.4f}），"
              "未覆盖；本轮结果仅在检查点中")
    else:
        # 候选在训练中就可被绑定，最终保存必须保持路径稳定。
        cand = write_candidate_strategy(path, data, stamp=engine.candidate_run_id)
        reason_text = STRATEGY_SAVE_REASONS.get(reason, reason)
        print(f"  [策略保护] 未覆盖 {pathlib.Path(path).name}（{reason_text}）")
        if cand:
            print(f"  [策略保护] 本轮新策略已另存（每轮一个候选文件）: {pathlib.Path(cand).as_posix()}")
            print("            请在「策略库」对比验收后手动替换：删除或改名旧策略文件，再把候选文件改回 best_ 名称")
        else:
            print("  [策略保护] 候选另存失败，本轮结果仅在检查点中")


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

"""Offline training regressions. Run with runtime/python.exe; never connects to MT5.
All training writes are confined to a TemporaryDirectory; no user artifacts are loaded.
"""
import contextlib
import io
import math
import os
from pathlib import Path
import random
import numpy as np
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from model_core import engine as engine_mod
from model_core.engine import AlphaEngine, _build_walk_forward_folds
from model_core.island_engine import IslandAlphaEngine
from model_core.config import ModelConfig
from model_core.backtest import MT5Backtest
from model_core.vocab import FORMULA_VOCAB


class TrainingTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        tmp = self.stack.enter_context(tempfile.TemporaryDirectory(prefix="mt5_training_test_"))
        self.tmp = Path(tmp)
        cwd = os.getcwd()
        os.chdir(tmp)
        self.stack.callback(os.chdir, cwd)
        from config import Config
        self.stack.enter_context(patch.object(Config, "ROOT_DIR", self.tmp))
        self.stack.enter_context(patch.object(Config, "MIN_TRADE_EXPOSURE", 0.05))
        self.stack.enter_context(patch.object(Config, "COST_RATE", 0.0003))
        self.stack.enter_context(patch.multiple(
            ModelConfig, TRAIN_STEPS=2, BATCH_SIZE=4, MAX_FORMULA_LEN=4,
            PARALLEL_EVAL=False, ISLAND_PARALLEL=False,
        ))
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.stack.callback(torch.set_num_threads, threads)
        torch.manual_seed(123)
        random.seed(123)
        self.data = SimpleNamespace(
            feat_tensor=torch.randn(1, FORMULA_VOCAB.feature_count, 160),
            target_ret=torch.randn(1, 160) * 0.001,
            raw_dict={"time": torch.arange(160).reshape(1, -1) * 3600 + 1700000000},
        )

    def test_net_losing_candidate_not_eligible_despite_positive_score(self):
        eng = AlphaEngine(self.data)
        x = torch.tensor([[1.0, -1.0] * 80])
        returns = -x * 0.001
        with patch.object(eng.vm, "execute", return_value=x), \
             patch.object(eng.bt, "evaluate_fold", return_value=(torch.tensor(10.0), torch.tensor(10.0))):
            result = eng._eval_formula_task(0, [0], x, returns, [], False, [])
        self.assertEqual(result['status'], 'ok', result)
        self.assertGreater(result['val_score'], 0)
        self.assertLess(result['net_mean'], 0)
        self.assertFalse(result['eligible'])

    def test_training_folds_use_only_observed_forward_returns(self):
        eng = AlphaEngine(self.data, use_lord_regularization=False)
        eng.save_checkpoints = False
        with patch.object(engine_mod, "_build_walk_forward_folds", wraps=_build_walk_forward_folds) as build, \
             patch.object(eng, "_eval_formula_task", side_effect=lambda idx, fml, *args:
                 {"idx": idx, "status": "none", "reward": -5., "val_score": -5., "fml": fml}), \
             contextlib.redirect_stdout(io.StringIO()):
            eng.train(end_step=1, verbose_header=False)
        self.assertEqual(build.call_args.args[0], 158)

    def test_old_tail_evaluation_context_cannot_restore_rankings(self):
        eng = AlphaEngine(self.data, use_lord_regularization=False)
        eng.best_formula, eng.best_score = [0], 99.
        state = eng.checkpoint_state(1)
        self.assertEqual(state["training_context"]["evaluation"]["target_return_policy"],
                         "observed-open-t+1-to-t+2-v1")
        state["training_context"]["evaluation"].pop("target_return_policy")
        restored = AlphaEngine(self.data, use_lord_regularization=False)
        with contextlib.redirect_stdout(io.StringIO()):
            restored.restore_checkpoint_state(state)
        self.assertIsNone(restored.best_formula)
        self.assertEqual(restored.best_score, -float("inf"))

    def test_entropy_floor_has_gradient_in_actual_training(self):
        # Neutralize the separate entropy bonus and compare the actual training
        # loop at the same seed. Below-floor loss must change policy parameters.
        states = []
        for enabled in (False, True):
            torch.manual_seed(88)
            eng = AlphaEngine(self.data, use_lord_regularization=False)
            eng.save_checkpoints = False
            with patch.multiple(ModelConfig, ENTROPY_FLOOR=enabled,
                                ENTROPY_FLOOR_THRESH=10.0, ENTROPY_COEFF_MAX=0.0), \
                 contextlib.redirect_stdout(io.StringIO()):
                eng.train(end_step=1, verbose_header=False)
            states.append(eng.model.mtp_head.head.weight.detach().clone())
        self.assertGreater((states[0] - states[1]).abs().max().item(), 1e-6)

    def test_ic_alignment_and_corr_penalty(self):
        # A perfectly aligned alternating predictor becomes -1 under the old
        # extra target shift; current engine and backtest must agree on +1.
        x = torch.tensor([[1.0, -1.0] * 20])
        ic, _ = AlphaEngine._compute_ic(x, x)
        self.assertAlmostEqual(ic.item(), 1.0, places=6)
        self.assertGreater(MT5Backtest()._ts_ic_stability(x, x), 0)
        eng = AlphaEngine.__new__(AlphaEngine)
        eng.factor_pool = [(1.0, 0, x)]
        with patch.object(ModelConfig, "CORR_PENALTY", 0.8):
            self.assertAlmostEqual(eng._apply_corr_penalty(torch.tensor(-5.0), x).item(), -6.0)
            self.assertAlmostEqual(eng._apply_corr_penalty(torch.tensor(5.0), x).item(), 4.0)
            self.assertEqual(eng._apply_corr_penalty(torch.tensor(-5.0), x, pool_snapshot=[]).item(), -5.0)

    def test_wf_gap_and_sampler_structure(self):
        folds = _build_walk_forward_folds(1000, 5, 20)
        self.assertEqual(len(folds), 4)
        for f in folds:
            self.assertEqual(f["val_start"] - f["train_end"], 20)
            self.assertLessEqual(f["val_end"], 1000)
        with self.assertRaises(ValueError):
            _build_walk_forward_folds(10, 5, 20)
        eng = AlphaEngine(self.data)
        sampler = eng.sampler
        for length in (4, 8, 14):
            for _ in range(30):
                depth = 0
                for step in range(length):
                    mask = sampler.valid_mask(depth, step, length, torch.device("cpu"), infected_chain_len=3)
                    ids = mask.nonzero().flatten().tolist()
                    for tid in ids:
                        arity = 0 if tid < sampler.feat_offset else sampler.arity_map[tid]
                        self.assertGreaterEqual(depth, arity)
                    tid = random.choice(ids)
                    depth += sampler.delta[tid]
                self.assertEqual(depth, 1)

    def test_entry_save_and_legacy_island_round_trip(self):
        import json
        # Avoid replacing test output streams on import; no training is mocked.
        with patch("utils.train_logging.configure_train_stdio"):
            import train_file
        from model_core.backtest import SCORING_VERSION
        eng = AlphaEngine(self.data)
        eng.best_formula, eng.best_score = [0], 1.0
        with contextlib.redirect_stdout(io.StringIO()):
            train_file._save_strategy(eng, "SYNTHETIC", "H1", "synthetic.parquet")
        path = self.tmp / "strategies" / "best_SYNTHETIC_H1.json"
        data = json.loads(path.read_text())
        self.assertEqual(data["scoring_version"], SCORING_VERSION)
        FORMULA_VOCAB.verify(data["vocab_version"])
        data.pop("scoring_version")
        original = json.dumps(data).encode()
        path.write_bytes(original)
        with contextlib.redirect_stdout(io.StringIO()):
            train_file._save_strategy(eng, "SYNTHETIC", "H1", "synthetic.parquet")
            eng.best_formula, eng.best_score = None, -float("inf")
            train_file._seed_best_from_strategy(eng, "SYNTHETIC", "H1")
        self.assertEqual(path.read_bytes(), original)
        self.assertIsNone(eng.best_formula)
        island = IslandAlphaEngine(self.data, n_islands=1, migration_interval=1)
        with contextlib.redirect_stdout(io.StringIO()):
            cp = island.save_checkpoint(3)
        payload = torch.load(cp, weights_only=False)
        payload.pop("scoring_version")
        payload["global_best_score"], payload["global_best_formula"] = 999.0, [0]
        payload["islands"][0].pop("scoring_version")
        payload["islands"][0]["stopped_early"] = True
        torch.save(payload, cp)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(island.load_checkpoint(cp), 3)
        self.assertIsNone(island.global_best_formula)
        self.assertEqual(island.global_history["step"], [])
        self.assertTrue(island.islands[0].stopped_early)
        self.assertEqual(island._step, 3)

    def test_strategy_version_and_legacy_bytes_preserved(self):
        import json
        from model_core.backtest import SCORING_VERSION
        eng = AlphaEngine(self.data, target_symbol="SYNTHETIC")
        eng.best_formula, eng.best_score = [0], 1.0
        eng._save_strategy_live()
        path = self.tmp / "strategies" / "best_SYNTHETIC.json"
        saved = json.loads(path.read_text())
        self.assertEqual(saved["scoring_version"], SCORING_VERSION)
        self.assertIn("vocab_version", saved)  # vocabulary compatibility is still required
        saved.pop("scoring_version")
        original = json.dumps(saved).encode()
        path.write_bytes(original)
        eng.best_score = 999.0
        with contextlib.redirect_stdout(io.StringIO()):
            eng._save_strategy_live()
        self.assertEqual(path.read_bytes(), original)

    def test_resume_never_overwrites_higher_scored_strategy(self):
        """续训保护：磁盘上既有策略分数更高时，绝不被更低的检查点结果覆盖。"""
        import json
        eng = AlphaEngine(self.data, target_symbol="SYNTHETIC")
        path = self.tmp / "strategies" / "best_SYNTHETIC.json"
        path.parent.mkdir(parents=True)
        from model_core.engine import _current_scoring_version, _feature_semantics_version
        path.write_text(json.dumps({"scoring_version": _current_scoring_version(),
                                    "feature_semantics_version": _feature_semantics_version(),
                                    "training_context": eng.training_context(),
                                    "vocab_version": "x", "best_score": 2.1796}))
        eng.best_formula, eng.best_score = [0], 1.9938
        with contextlib.redirect_stdout(io.StringIO()) as out:
            eng._save_strategy_live()
        self.assertEqual(float(json.loads(path.read_text())["best_score"]), 2.1796)
        self.assertIn("续训保护", out.getvalue())
        # 本轮分数更高时仍正常覆盖
        eng.best_score = 2.5
        with contextlib.redirect_stdout(io.StringIO()):
            eng._save_strategy_live()
        self.assertEqual(float(json.loads(path.read_text())["best_score"]), 2.5)

    def test_legacy_checkpoint_drops_rankings_not_weights_or_hard_stops(self):
        eng = AlphaEngine(self.data)
        eng.best_score = 999.0
        eng.best_formula = [0]
        eng._elite_pool = [(999.0, 0, [0], 2)]
        eng._reward_ema = 999.0
        eng.stopped_early = True
        eng._restart_count = 3
        eng._stag_windows_no_gain = 2
        state = eng.checkpoint_state(17)
        state.pop("scoring_version")
        restored = AlphaEngine(self.data)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(restored.restore_checkpoint_state(state), 17)
        self.assertEqual(restored.best_score, -float("inf"))
        self.assertIsNone(restored.best_formula)
        self.assertEqual(restored._elite_pool, [])
        self.assertIsNone(restored._reward_ema)
        self.assertTrue(restored.stopped_early)
        self.assertEqual(restored._restart_count, 3)
        self.assertEqual(restored._stag_windows_no_gain, 2)
        for n, p in restored.model.named_parameters():
            self.assertTrue(torch.equal(p, state["model_state_dict"][n]))

    def test_ic_gate_sign_exact(self):
        with patch.multiple(ModelConfig, IC_GATE_THRESH=0.01, IC_GATE_MULT=1.15, IC_NEG_MULT=0.75):
            self.assertAlmostEqual(AlphaEngine._apply_ic_gate(torch.tensor(-4.0), -0.5).item(), -5.0)
            self.assertAlmostEqual(AlphaEngine._apply_ic_gate(torch.tensor(4.0), -0.5).item(), 3.0)
            self.assertAlmostEqual(AlphaEngine._apply_ic_gate(torch.tensor(-4.0), 0.5).item(), -4.0)

    def test_fallback_uses_validation_output_and_training_slice(self):
        # No Transformer/optimizer needed: exercise the actual evaluator with
        # controlled fold outputs and neutral IC, so expected values are exact.
        eng = AlphaEngine.__new__(AlphaEngine)
        factors = torch.arange(20, dtype=torch.float32).reshape(1, -1)
        eng.vm = SimpleNamespace(execute=lambda *_: factors)
        eng.factor_pool = []
        calls = []
        def evaluate_fold(f, r, ts, te, vs, ve):
            calls.append((ts, te, vs, ve))
            return torch.tensor(12.0), torch.tensor(-7.0)
        eng.bt = SimpleNamespace(evaluate_fold=evaluate_fold, cost_rate=0.0003,
                                 evaluate=lambda *_: (torch.tensor(99.0), 0.0))
        with patch.object(AlphaEngine, "_compute_ic", return_value=(torch.tensor(0.0), torch.tensor(0.0))), \
             patch.object(ModelConfig, "REWARD_ALPHA", 1.0):
            result = eng._eval_formula_task(0, [0], factors, torch.zeros_like(factors), [], False, [])
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(result["reward"], 12.0)
        self.assertEqual(result["val_score"], -7.0)
        self.assertEqual(calls, [(0, 14, 14, 18)])

    def test_oos_gate_cannot_reward_a_loss(self):
        bt = MT5Backtest(cost_rate=0.0)
        factors = torch.ones(1, 40)
        returns = -torch.ones(1, 40) * 0.01
        # A positive auxiliary score must not label net-losing OOS as good;
        # an already negative objective must never be moved toward zero.
        for base in (-4.0, 4.0):
            with patch.object(bt, "_multi_objective", return_value=torch.tensor(base)):
                _, validation = bt.evaluate_fold(factors, returns, 0, 20, 20, 40)
                public_score, mean_oos = bt.evaluate(factors, {}, returns)
            self.assertLessEqual(validation.item(), min(base, 0.0))
            self.assertLessEqual(public_score.item(), min(base, 0.0))
            self.assertLess(mean_oos, 0.0)

    def test_island_real_train_resume_and_all_stopped(self):
        islands = IslandAlphaEngine(self.data, n_islands=2, migration_interval=1, base_seed=17)
        islands.tag_islands("SYNTHETIC", "H1")
        before = []
        for isl in islands.islands:
            params = dict(isl.model.named_parameters())
            self.assertIs(isl.rank_monitor.model, isl.model)
            self.assertTrue(isl.lord_opt.params_to_decay)
            for name, param in isl.lord_opt.params_to_decay:
                self.assertIs(params[name], param)
            self.assertEqual({id(p) for p in isl.model.parameters()},
                             {id(p) for g in isl.opt.param_groups for p in g["params"]})
            before.append({n: p.detach().clone() for n, p in params.items()})
        self.assertFalse(torch.equal(before[0]["token_emb.weight"], before[1]["token_emb.weight"]))
        with contextlib.redirect_stdout(io.StringIO()), patch.object(ModelConfig, "TRAIN_STEPS", 1):
            islands.train()
        self.assertEqual(islands._step, 1)
        for i, isl in enumerate(islands.islands):
            self.assertEqual(isl.training_history["step"], [0])
            self.assertTrue(all(torch.isfinite(p).all() for p in isl.model.parameters()))
            self.assertTrue(any(not torch.equal(p, before[i][n]) for n, p in isl.model.named_parameters()))
            self.assertTrue(math.isfinite(isl.training_history["avg_reward"][0]))
        checkpoint = next((self.tmp / "checkpoints").glob("island_ckpt_*.pt"))
        resumed = IslandAlphaEngine(self.data, n_islands=2, migration_interval=1, base_seed=99)
        resumed.tag_islands("SYNTHETIC", "H1")
        with contextlib.redirect_stdout(io.StringIO()):
            step = resumed.load_checkpoint(str(checkpoint))
            resumed.train(start_step=step)
        self.assertEqual(resumed._step, 2)
        for isl in resumed.islands:
            self.assertEqual(isl.training_history["step"], [0, 1])
            self.assertTrue(all(torch.isfinite(p).all() for p in isl.model.parameters()))
            isl.stopped_early = True
        with contextlib.redirect_stdout(io.StringIO()), patch.object(ModelConfig, "TRAIN_STEPS", 4):
            resumed.train(start_step=2)
        self.assertEqual(resumed._step, 2)
        self.assertTrue((self.tmp / "training_history" / "training_history_SYNTHETIC_H1__isl1.json").exists())


class TrainingDataBoundaryTests(unittest.TestCase):
    @staticmethod
    def frame(indices, base=100.):
        import pandas as pd
        price = np.asarray(indices, dtype=float) + base
        return pd.DataFrame({"time": 1700000000 + np.asarray(indices) * 3600,
                             "open": price, "high": price + 1, "low": price - 1,
                             "close": price, "volume": np.ones(len(indices))})

    def test_parquet_retains_price_tail_for_last_observed_label(self):
        from config import Config
        from data_pipeline.parquet_manager import ParquetDataManager
        frame = self.frame(range(8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "TEST_H1.parquet"
            frame.to_parquet(path)
            manager = ParquetDataManager(path)
            with patch.object(Config, "MIN_BARS", 8):
                manager.load()
            self.assertEqual(manager.raw_dict["open"].shape, (1, 8))
            torch.testing.assert_close(manager.target_ret[:, :-2],
                torch.log(manager.raw_dict["open"][:, 2:] / manager.raw_dict["open"][:, 1:-1]))
            self.assertGreater(manager.target_ret[0, -3].item(), 0)
            self.assertEqual(torch.count_nonzero(manager.target_ret[:, -2:]).item(), 0)

    def test_tail_factors_do_not_change_any_evaluator_metrics(self):
        from data_pipeline.data_manager import MT5DataManager
        price = torch.tensor([[100., 100.] + [100. + i for i in range(1, 21)]])
        returns = MT5DataManager._compute_target_ret(price)
        factors = torch.tensor([[.2, .8] * 10 + [0., 0.]])
        eng = AlphaEngine.__new__(AlphaEngine)
        eng.factor_pool = []
        eng.bt = MT5Backtest(cost_rate=.001, periods_per_year=1)
        folds = _build_walk_forward_folds(20, 2, 0)
        for use_wf in (False, True):
            results = []
            for tail in ([0., 0.], [10000., -10000.], [float("nan"), float("inf")]):
                changed = factors.clone()
                changed[:, -2:] = torch.tensor(tail)
                eng.vm = SimpleNamespace(execute=lambda *_: changed)
                result = eng._eval_formula_task(0, [0], changed[:, None, :], returns,
                                               folds if use_wf else [], use_wf, [])
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(result["res"].shape, (1, 20))
                results.append(result)
            for key in ("reward", "val_score", "ic_full", "ic_stab", "ic_i",
                        "net_mean", "validation_net_mean", "eligible"):
                self.assertEqual(results[0][key], results[1][key], key)
                self.assertEqual(results[0][key], results[2][key], key)
            self.assertTrue(results[0]["eligible"])
            position = engine_mod.compute_target_positions_stateless(factors[:, :-2])
            previous = torch.zeros_like(position)
            previous[:, 1:] = position[:, :-1]
            expected = position * returns[:, :-2] - (position - previous).abs() * eng.bt.cost_rate
            self.assertAlmostEqual(results[0]["net_mean"], expected.mean().item())

    def test_tail_variation_does_not_rescue_constant_candidate(self):
        factors = torch.tensor([[0.] * 20 + [10., -10.]])
        eng = AlphaEngine.__new__(AlphaEngine)
        eng.vm = SimpleNamespace(execute=lambda *_: factors)
        result = eng._eval_formula_task(0, [0], factors[:, None, :],
                                       torch.zeros_like(factors), [], False, [])
        self.assertEqual(result["status"], "const", result)

    def test_union_fallback_trims_leading_unknown_prices_without_backfill(self):
        from config import Config
        from data_pipeline.data_manager import MT5DataManager
        manager = MT5DataManager(None)
        manager._symbols = ["EARLY", "LATE"]
        frames = {"EARLY": self.frame([0, 2, 4, 6, 8]),
                  "LATE": self.frame([3, 5, 7, 9, 11], base=200.)}
        with patch.object(Config, "MIN_BARS", 5):
            aligned = manager._align_timelines(frames)
        expected_index = [1700000000 + i * 3600 for i in range(3, 10)] + [1700000000 + 11 * 3600]
        self.assertEqual(aligned["EARLY"].index.tolist(), expected_index)
        self.assertEqual(aligned["LATE"].index.tolist(), expected_index)
        self.assertEqual(aligned["EARLY"].iloc[0]["open"], 102.)
        self.assertEqual(aligned["LATE"].iloc[1]["open"], 203.)
        raw = manager._build_raw_dict(aligned)
        self.assertTrue(torch.all(raw["open"] > 0))
        self.assertTrue(torch.isfinite(manager._compute_target_ret(raw["open"])).all())
        # Future prices must not alter the causally forward-filled prefix.
        frames["LATE"].loc[frames["LATE"]["time"] > expected_index[1], "open"] = 999.
        with patch.object(Config, "MIN_BARS", 5):
            changed = manager._align_timelines(frames)
        np.testing.assert_array_equal(aligned["LATE"]["open"].iloc[:2], changed["LATE"]["open"].iloc[:2])

    def test_union_fallback_rejects_insufficient_causal_history(self):
        from config import Config
        from data_pipeline.data_manager import MT5DataManager
        frames = {"EARLY": self.frame([0, 1, 2, 3, 4]), "LATE": self.frame([10, 11])}
        with patch.object(Config, "MIN_BARS", 5):
            with self.assertRaisesRegex(ValueError, "causal.*bars"):
                MT5DataManager(None)._align_timelines(frames)


if __name__ == "__main__":
    unittest.main(verbosity=2)

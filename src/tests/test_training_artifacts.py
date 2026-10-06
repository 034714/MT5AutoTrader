"""Focused training artifact and data-ingress regressions."""
import contextlib
import tempfile
import unittest
import copy
import io
import json
import os
import random
import torch
import numpy as np
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from data_pipeline.parquet_manager import validate_training_frame
from mt5_train import fetch_mt5
from utils.training_artifacts import atomic_json_write, atomic_write, checkpoint_step, latest_checkpoint


class TrainingArtifactTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="training_artifacts_")))
        self.stack.enter_context(patch.object(Config, "ROOT_DIR", self.tmp))

    def test_atomic_json_preserves_previous_bytes_on_writer_failure(self):
        path = self.tmp / "config.json"
        atomic_json_write(path, {"version": 1})
        before = path.read_bytes()
        with self.assertRaises(RuntimeError):
            atomic_write(path, lambda _: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertEqual(path.read_bytes(), before)
        self.assertFalse(list(self.tmp.glob(".config.json.*.tmp")))

    def test_checkpoint_steps_are_numeric(self):
        checkpoint_dir = self.tmp / "checkpoints"
        checkpoint_dir.mkdir()
        for name in ("ckpt_X_step_9.pt", "ckpt_X_step_10.pt", "ckpt_X_step_100.pt"):
            (checkpoint_dir / name).write_bytes(b"x")
        with patch("utils.training_artifacts.checkpoint_dir", return_value=checkpoint_dir):
            self.assertEqual(checkpoint_step(checkpoint_dir / "ckpt_X_step_100.pt"), 100)
            self.assertEqual(latest_checkpoint("ckpt_X_step_*.pt").name, "ckpt_X_step_100.pt")

    def test_validation_deduplicates_before_minimum_and_checks_ohlc(self):
        frame = pd.DataFrame({
            "time": [1_700_000_000, 1_700_000_000, 1_700_003_600],
            "open": [10.0, 10.1, 10.2], "high": [10.2, 10.3, 10.4],
            "low": [9.8, 9.9, 10.0], "close": [10.1, 10.2, 10.3],
            "tick_volume": [1, 2, 3],
        })
        with patch.object(Config, "MIN_BARS", 2):
            result = validate_training_frame(frame)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["time"].tolist(), [1_700_000_000, 1_700_003_600])
        bad = frame.copy()
        bad.loc[1, "high"] = float("nan")
        with patch.object(Config, "MIN_BARS", 2):
            with self.assertRaises(ValueError):
                validate_training_frame(bad)

    def test_cache_writer_failure_preserves_existing_parquet(self):
        import mt5_train
        rates = [{"time": 1700000000 + i * 3600, "open": 10., "high": 11.,
                  "low": 9., "close": 10., "tick_volume": 5} for i in range(3)]
        fake = Mock()
        fake.initialize.return_value = True
        fake.symbol_info.return_value = object()
        fake.copy_rates_from_pos.return_value = rates
        path = self.tmp / "TEST_H1.parquet"
        path.write_bytes(b"previous complete cache")
        def fail(frame, path, **kwargs):
            Path(path).write_bytes(b"partial")
            raise OSError("disk full")
        with patch.object(mt5_train, "mt5", fake), patch.object(Config, "MIN_BARS", 2), \
                patch.object(Config, "get_timeframe", return_value=1), \
                patch.object(pd.DataFrame, "to_parquet", fail):
            with self.assertRaisesRegex(OSError, "disk full"):
                fetch_mt5("TEST", 3, "H1", self.tmp)
        self.assertEqual(path.read_bytes(), b"previous complete cache")
        self.assertEqual(list(self.tmp.iterdir()), [path])
        fake.shutdown.assert_called_once()

    def test_inspection_and_load_share_post_dedup_validation(self):
        from data_pipeline.parquet_manager import inspect_parquet_file, ParquetDataManager
        frame = pd.DataFrame({"time": [1700000000, 1700000000, 1700003600],
            "open": [10., 10., 10.], "high": [11., 11., 11.],
            "low": [9., 9., 9.], "close": [10., 10., 10.], "volume": [1., 1., 1.]})
        path = self.tmp / "TEST_H1.parquet"
        frame.to_parquet(path)
        for check in (lambda: inspect_parquet_file(path), lambda: ParquetDataManager(path).load()):
            with patch.object(Config, "MIN_BARS", 3):
                with self.assertRaisesRegex(ValueError, "unique bars"):
                    check()
        frame.loc[2, "low"] = 12.
        frame.to_parquet(path)
        with patch.object(Config, "MIN_BARS", 2):
            with self.assertRaisesRegex(ValueError, "bounds"):
                inspect_parquet_file(path)
        with self.assertRaises(ValueError):
            atomic_json_write(self.tmp / "bad.json", {"score": float("nan")})
        self.assertFalse((self.tmp / "bad.json").exists())

    def test_mt5_fetch_starts_at_closed_bar_position_one(self):
        rates = [{"time": 1_700_000_000 + i * 3600, "open": 10.0,
                  "high": 10.2, "low": 9.8, "close": 10.1,
                  "tick_volume": 10} for i in range(3)]
        copy_rates = Mock(return_value=rates)
        mt5_mock = SimpleNamespace(
            initialize=lambda: True,
            last_error=lambda: "",
            symbol_info=lambda _: object(),
            symbol_select=lambda *_: True,
            copy_rates_from_pos=copy_rates,
            shutdown=lambda: None,
        )
        with patch("mt5_train.mt5", mt5_mock), patch.object(Config, "MIN_BARS", 2), \
                patch.object(Config, "get_timeframe", return_value=1):
            path = fetch_mt5("TEST", 3, "H1", self.tmp)
        copy_rates.assert_called_once_with("TEST", 1, 1, 3)
        self.assertEqual(len(pd.read_parquet(path)), 3)


class CheckpointContextTests(unittest.TestCase):
    def setUp(self):
        from model_core.config import ModelConfig
        from model_core.vocab import FORMULA_VOCAB
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.tmp = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="training_context_")))
        self.stack.enter_context(patch.object(Config, "ROOT_DIR", self.tmp))
        self.stack.enter_context(patch.multiple(ModelConfig, BATCH_SIZE=4, MAX_FORMULA_LEN=4,
                                              TRAIN_STEPS=6, PARALLEL_EVAL=False,
                                              ISLAND_PARALLEL=False, STAG_HARD_RESTART=False))
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.stack.callback(torch.set_num_threads, threads)
        self.data = SimpleNamespace(symbol="SYNTH", timeframe="H1",
            feat_tensor=torch.randn(1, FORMULA_VOCAB.feature_count, 160),
            target_ret=torch.randn(1, 160) * .001,
            raw_dict={"time": torch.arange(160).reshape(1, -1) * 3600 + 1700000000})

    def engine(self, data=None, symbol="SYNTH", timeframe="H1"):
        from model_core.engine import AlphaEngine
        engine = AlphaEngine(data or self.data, target_symbol=symbol, use_lord_regularization=False)
        engine.timeframe = timeframe
        return engine

    def test_context_same_data_full_restore_changed_content_hotstart_wrong_identity_no_mutation(self):
        source = self.engine()
        source.best_formula, source.best_score = [0], 3.
        source._elite_pool = [(3., 0, [0], 1)]
        source.factor_pool = [(3., 0, torch.ones(1, 160))]
        source.training_history["step"] = [0]
        source.stopped_early = True
        state = copy.deepcopy(source.checkpoint_state(8))
        restored = self.engine()
        self.assertEqual(restored.restore_checkpoint_state(state), 8)
        self.assertEqual(restored.best_score, 3.)
        self.assertEqual(restored.training_history["step"], [0])
        changed = copy.deepcopy(self.data)
        changed.target_ret[0, 0] += .01
        refreshed = self.engine(changed)
        refreshed.restore_checkpoint_state(state)
        self.assertEqual(refreshed.completed_steps, 8)
        self.assertIsNone(refreshed.best_formula)
        self.assertFalse(refreshed.factor_pool or refreshed._elite_pool)
        self.assertFalse(refreshed.training_history["step"])
        self.assertFalse(refreshed.stopped_early)
        for key, value in refreshed.model.state_dict().items():
            self.assertTrue(torch.equal(value, state["model_state_dict"][key]))
        for bad in (self.engine(symbol="OTHER"), self.engine(timeframe="M5")):
            before = copy.deepcopy(bad.model.state_dict())
            with self.assertRaisesRegex(ValueError, "mismatch"):
                bad.restore_checkpoint_state(state)
            for key, value in before.items():
                self.assertTrue(torch.equal(value, bad.model.state_dict()[key]))

    def test_weights_only_rng_roundtrip_and_gadget_rejected(self):
        source = self.engine()
        path = source.save_checkpoint(2)
        payload = torch.load(path, weights_only=True)
        self.assertIsInstance(payload["rng_state"]["numpy"][1], torch.Tensor)
        expected = (torch.rand(4), random.random(), np.random.rand(4))
        restored = self.engine()
        restored.load_checkpoint(path)
        actual = (torch.rand(4), random.random(), np.random.rand(4))
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertEqual(expected[1], actual[1])
        np.testing.assert_array_equal(expected[2], actual[2])
        malicious = self.tmp / "gadget.pt"
        torch.save({"untrusted": Exception("gadget")}, malicious)
        with self.assertRaises(Exception):
            restored.load_checkpoint(str(malicious))

    def test_root_anchor_histories_and_stop_completed_count(self):
        from utils.training_artifacts import history_path, strategy_path
        engine = self.engine()
        cwd = os.getcwd()
        other = self.tmp / "othercwd"
        other.mkdir()
        os.chdir(other)
        self.stack.callback(os.chdir, cwd)
        engine.best_formula, engine.best_score = [0], 1.
        engine._save_strategy_live()
        engine._save_training_history_live()
        self.assertTrue(strategy_path("SYNTH", "H1").exists())
        self.assertTrue(history_path("SYNTH_H1").exists())
        self.assertFalse(list(other.iterdir()))
        (self.tmp / "TRAIN_STOP").write_text("stop")
        with contextlib.redirect_stdout(io.StringIO()):
            engine.train(start_step=4, end_step=6, verbose_header=False)
        self.assertEqual(engine.completed_steps, 4)
        cp = self.tmp / "checkpoints" / "ckpt_SYNTH_H1_step_0004.pt"
        self.assertEqual(torch.load(cp, weights_only=True)["step"], 4)
        self.assertEqual(engine.training_history["step"], [])

    def test_final_and_early_stop_checkpoint_short_run_restart_is_not_modulo(self):
        from model_core.config import ModelConfig
        engine = self.engine()
        with contextlib.redirect_stdout(io.StringIO()), patch.multiple(
                ModelConfig, STAG_HARD_RESTART=True, STAG_HARD_RESTART_FIRST=150,
                STAG_HARD_RESTART_INTERVAL=2, STAG_AUTO_STOP_WINDOWS=1), \
                patch.object(engine, "_eval_formula_task", side_effect=lambda idx, fml, *args:
                    {"idx": idx, "status": "none", "reward": -5., "val_score": -5., "fml": fml}):
            engine.train(start_step=1001, end_step=1007, verbose_header=False)
        self.assertTrue(engine.stopped_early)
        self.assertEqual(engine.completed_steps, 1003)
        cp = self.tmp / "checkpoints" / "ckpt_SYNTH_H1_step_1003.pt"
        state = torch.load(cp, weights_only=True)
        self.assertTrue(state["stopped_early"])
        self.assertEqual(state["step"], 1003)

    def test_same_data_split_run_matches_uninterrupted_parameters(self):
        from model_core.config import ModelConfig
        torch.manual_seed(222)
        random.seed(222)
        full = self.engine()
        full.save_checkpoints = False
        with contextlib.redirect_stdout(io.StringIO()), patch.object(ModelConfig, "ENTROPY_COLLAPSE_STEPS", 1):
            full.train(end_step=2, verbose_header=False)
        expected = copy.deepcopy(full.model.state_dict())
        torch.manual_seed(222)
        random.seed(222)
        split = self.engine()
        split.save_checkpoints = False
        with contextlib.redirect_stdout(io.StringIO()), patch.object(ModelConfig, "ENTROPY_COLLAPSE_STEPS", 1):
            split.train(end_step=1, run_end_step=2, verbose_header=False)
            state = copy.deepcopy(split.checkpoint_state(1))
            resumed = self.engine()
            resumed.restore_checkpoint_state(state)
            resumed.save_checkpoints = False
            resumed.train(start_step=1, end_step=2, verbose_header=False)
        for key, value in expected.items():
            self.assertTrue(torch.equal(value, resumed.model.state_dict()[key]), key)
        self.assertEqual(full.training_history, resumed.training_history)

    def test_island_phase_stop_and_resume_preserves_partial_counts(self):
        from model_core.config import ModelConfig
        from model_core.island_engine import IslandAlphaEngine
        islands = IslandAlphaEngine(self.data, n_islands=2, migration_interval=4)
        islands.tag_islands("SYNTH", "H1")
        stop = self.tmp / "TRAIN_STOP"
        calls = []
        def first_phase(**kwargs):
            calls.append(kwargs)
            islands.islands[0].completed_steps = 2
            islands.islands[0].user_stopped = True
            stop.write_text("stop")
        with patch.object(islands.islands[0], "train", side_effect=first_phase), \
                patch.object(ModelConfig, "TRAIN_STEPS", 4), contextlib.redirect_stdout(io.StringIO()):
            islands.train()
        self.assertEqual(calls[0]["run_end_step"], 4)
        cp = self.tmp / "checkpoints" / "island_ckpt_SYNTH_H1_step_0000.pt"
        payload = torch.load(cp, weights_only=True)
        self.assertTrue(payload["phase_interrupted"])
        self.assertEqual([s["step"] for s in payload["islands"]], [2, 0])
        resumed = IslandAlphaEngine(self.data, n_islands=2, migration_interval=4)
        resumed.tag_islands("SYNTH", "H1")
        with contextlib.redirect_stdout(io.StringIO()):
            step = resumed.load_checkpoint(str(cp))
        stop.unlink()
        with patch.object(ModelConfig, "TRAIN_STEPS", 4), contextlib.redirect_stdout(io.StringIO()):
            resumed.train(start_step=step)
        self.assertEqual(resumed._step, 4)
        self.assertEqual(resumed.islands[0].training_history["step"], [2, 3])
        self.assertEqual(resumed.islands[1].training_history["step"], [0, 1, 2, 3])
        self.assertFalse((self.tmp / "strategies" / "best_island_strategy.json").exists())

    def test_island_context_validation_precedes_all_mutation(self):
        from model_core.island_engine import IslandAlphaEngine
        source = IslandAlphaEngine(self.data, n_islands=2, migration_interval=2)
        source.tag_islands("SYNTH", "H1")
        cp = source.save_checkpoint(0)
        state = torch.load(cp, weights_only=True)
        state["islands"][1]["training_context"]["symbol"] = "WRONG"
        torch.save(state, cp)
        other = IslandAlphaEngine(self.data, n_islands=2, migration_interval=2)
        other.tag_islands("SYNTH", "H1")
        before = [copy.deepcopy(isl.model.state_dict()) for isl in other.islands]
        with self.assertRaisesRegex(ValueError, "symbol mismatch"):
            other.load_checkpoint(cp)
        for idx, isl in enumerate(other.islands):
            for key, value in before[idx].items():
                self.assertTrue(torch.equal(value, isl.model.state_dict()[key]))

    def test_strategy_all_save_paths_protect_nonfinite_context_and_highscore(self):
        from train_file import _save_strategy
        from utils.training_artifacts import strategy_path
        source = self.engine()
        source.best_formula, source.best_score = [0], 3.
        _save_strategy(source, "SYNTH", "H1", "synthetic.parquet")
        path = strategy_path("SYNTH", "H1")
        before = path.read_bytes()
        for score in (2., float("inf"), float("nan")):
            source.best_score = score
            _save_strategy(source, "SYNTH", "H1", "changed.parquet")
            source._save_strategy_live()
            self.assertEqual(path.read_bytes(), before)
        source.best_score = 4.
        _save_strategy(source, "SYNTH", "H1", "synthetic.parquet")
        self.assertEqual(json.loads(path.read_text())["best_score"], 4.)
        before = path.read_bytes()
        other = self.engine(copy.deepcopy(self.data))
        other._data_fingerprint = "refreshed"
        other.best_formula, other.best_score = [0], 99.
        _save_strategy(other, "SYNTH", "H1", "synthetic.parquet")
        self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)

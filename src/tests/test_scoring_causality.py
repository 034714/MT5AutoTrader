"""Focused offline numerical and causal-prefix regression tests.

Run: python src/tests/test_scoring_causality.py
"""
import math
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch

from config import Config
from model_core.backtest import MT5Backtest, SCORING_VERSION
from model_core.features import (
    MT5FeatureEngineer as FE, FEATURE_REGISTRY, FEATURE_SEMANTICS_VERSION,
    LEGACY_FEATURE_SEMANTICS_VERSION, _FEATURE_SEMANTICS,
)
from model_core.vm import StackVM, LEGACY_NORMALIZATION_VERSION
from backtest_viz.engine import BacktestEngine
from trading import quick_backtest as qb
from test_quick_backtest import rates_fixture, ReadOnlyMT5
import json
import tempfile


def raw_fixture(length=750, symbols=1):
    rng = np.random.default_rng(110)
    close = torch.tensor(100 + np.cumsum(rng.normal(0, .2, (symbols, length)), axis=1)).float()
    return {"open": close - .1, "high": close + 1, "low": close - 1,
            "close": close, "volume": torch.tensor(rng.uniform(50, 150, (symbols, length))).float()}


class CausalityTests(unittest.TestCase):
    def test_vm_warmup_499_500_501(self):
        x = torch.arange(501, dtype=torch.float64).reshape(1, -1)
        a = StackVM._normalize_output(x[:, :499])
        b = StackVM._normalize_output(x[:, :500])
        c = StackVM._normalize_output(x)
        self.assertEqual(torch.count_nonzero(a).item(), 0)
        self.assertEqual(torch.count_nonzero(b[:, :499]).item(), 0)
        expected = (x[0, 499] - x[0, :500].mean()) / x[0, :500].std()
        self.assertAlmostEqual(b[0, 499].item(), expected.item(), places=12)
        torch.testing.assert_close(b, c[:, :500], rtol=0, atol=0)
        self.assertAlmostEqual(c[0, 500].item(), expected.item(), places=12)

    def test_vm_constant_prefix_before_future_variation(self):
        x = torch.ones(1, 800)
        x[:, 700:] = torch.arange(100).float()
        prefix = StackVM._normalize_output(x[:, :700])
        full = StackVM._normalize_output(x)
        torch.testing.assert_close(prefix, full[:, :700], rtol=0, atol=0)
        self.assertEqual(prefix[0, 498].item(), 0)
        self.assertEqual(prefix[0, 499].item(), 1)

    def test_vm_multi_symbol_constant_future_cannot_change_prefix(self):
        x = torch.full((3, 800), 7.)
        x[:, 700:] = torch.tensor([1., 2., 5.]).reshape(3, 1)
        a = StackVM._normalize_output(x[:, :700])
        b = StackVM._normalize_output(x)
        torch.testing.assert_close(a, b[:, :700], rtol=0, atol=0)
        self.assertEqual(torch.count_nonzero(a).item(), 0)

    def test_vm_random_prefix_and_legacy_opt_in(self):
        x = torch.randn(1, 850, generator=torch.Generator().manual_seed(12))
        for length in (1, 499, 500, 501, 700):
            torch.testing.assert_close(StackVM._normalize_output(x[:, :length]),
                                       StackVM._normalize_output(x)[:, :length], rtol=0, atol=0)
        legacy = StackVM(normalization_version=LEGACY_NORMALIZATION_VERSION)
        torch.testing.assert_close(legacy.execute([0], x[:, None, :]), StackVM._normalize_output_legacy(x))
        with self.assertRaises(ValueError):
            StackVM(normalization_version="unknown")

    def test_features_every_registered_feature_prefix(self):
        raw = raw_fixture(750, 2)
        full = FE.compute_features(raw)
        for length in (199, 200, 350, 501):
            prefix = FE.compute_features({k: v[:, :length] for k, v in raw.items()})
            torch.testing.assert_close(prefix, full[:, :, :length], rtol=1e-5, atol=1e-5)

    def test_ema_fixed_kernel_prefix_and_legacy_branch(self):
        x = raw_fixture(750)["close"]
        torch.testing.assert_close(FE._ema_simple(x[:, :100], 26),
                                   FE._ema_simple(x, 26)[:, :100], rtol=0, atol=0)
        token = _FEATURE_SEMANTICS.set(LEGACY_FEATURE_SEMANTICS_VERSION)
        try:
            torch.testing.assert_close(FE._ema_simple(x[:, :100], 26),
                                       FE._ema_simple(x[:, :100], 26, exact=True), rtol=0, atol=0)
        finally:
            _FEATURE_SEMANTICS.reset(token)

    def test_willr_negative_endpoints(self):
        high = torch.full((1, 20), 110.)
        low = torch.full((1, 20), 90.)
        for price, expected in ((110., 0.), (100., -.5), (90., -1.)):
            actual = FE._willr(torch.full((1, 20), price), high, low)
            self.assertAlmostEqual(actual[0, -1].item(), expected)

    def test_feature_legacy_willr_and_unknown_version(self):
        raw = raw_fixture(250)
        latest = FE.compute_features(raw)
        legacy = FE.compute_features(raw, semantics_version=LEGACY_FEATURE_SEMANTICS_VERSION)
        index = list(FEATURE_REGISTRY.feature_names).index("WILLR_14")
        self.assertTrue((latest[:, index, :] < 0).any())
        self.assertEqual(torch.count_nonzero(legacy[:, index, :]).item(), 0)
        self.assertEqual(FEATURE_SEMANTICS_VERSION, "causal-features-v2")
        self.assertEqual(SCORING_VERSION, "causal-portfolio-v3")
        with self.assertRaises(ValueError):
            FE.compute_features(raw, semantics_version="unknown")

    def test_quick_reports_latest_vs_deployed_legacy_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "strategies").mkdir()
            strategy = root / "strategies" / "test.json"
            metadata = {"symbol": "TEST", "timeframe": "H1", "formula": [0]}
            strategy.write_text(json.dumps(metadata))
            cfg = {"signal_bars": 800, "min_trade_exposure": .05, "max_lot_per_trade": 1.}
            payload = {"strategy_file": "test.json", "bars": 500, "lot": .2}
            result = qb.run_quick_backtest(root, payload, cfg, ReadOnlyMT5(rates_fixture()),
                                           signal_fn=lambda rates, meta: np.zeros(len(rates)))
            self.assertTrue(result["assumptions"]["strategy_semantics_differ"])
            self.assertEqual(result["assumptions"]["strategy_feature_semantics_version"], "legacy-v1")
            self.assertEqual(result["assumptions"]["feature_semantics_version"], FEATURE_SEMANTICS_VERSION)
            self.assertIn("fixed_lot", result["assumptions"]["pnl_semantics"])
            metadata["feature_semantics_version"] = "unknown"
            strategy.write_text(json.dumps(metadata))
            with self.assertRaises(qb.QuickBacktestError):
                qb.resolve_strategy(root, "test.json")

    def test_quick_uses_core_vm_all_prefix_bars(self):
        rates = rates_fixture(800)
        x = torch.arange(800).float().reshape(1, 1, -1)
        meta = {"formula": [0], "vocab_version": ""}
        with patch.object(FE, "compute_features", return_value=x):
            actual = qb.compute_positions(rates, meta)
        expected = torch.tanh(StackVM().execute([0], x)[0]).numpy()
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
        self.assertEqual(np.count_nonzero(actual[:499]), 0)


class NumericalScoringTests(unittest.TestCase):
    def setUp(self):
        self.threshold = patch.object(Config, "MIN_TRADE_EXPOSURE", .05)
        self.threshold.start()
        self.addCleanup(self.threshold.stop)

    def test_calmar_equal_weight_time_path(self):
        scorer = MT5Backtest(cost_rate=0, periods_per_year=1)
        pnl = torch.tensor([[.3, -.4, .2], [-.1, .2, .1]], dtype=torch.float64)
        portfolio = pnl.mean(dim=0)
        path = torch.cat([torch.zeros(1, dtype=pnl.dtype), portfolio.cumsum(0)])
        expected = portfolio.mean() / (path.cummax(0).values - path).max()
        self.assertAlmostEqual(scorer._calmar(pnl).item(), expected.item(), places=12)
        self.assertAlmostEqual(scorer._calmar(pnl.flip(0)).item(), expected.item(), places=12)

    def test_calmar_first_loss_drawdown_from_zero(self):
        scorer = MT5Backtest(cost_rate=0, periods_per_year=1)
        self.assertAlmostEqual(scorer._calmar(torch.tensor([-.2, -.1], dtype=torch.float64)).item(), -.5)

    def test_turnover_quality_continuous_direction_runs(self):
        scorer = MT5Backtest(cost_rate=0)
        continuous = torch.tensor([[.2, .8, .3, 0, -.2, -.8, .1, .5]])
        discrete = torch.tensor([[1., 1., 1., 0., -1., -1., 1., 1.]])
        self.assertAlmostEqual(scorer._turnover_quality(continuous), scorer._turnover_quality(discrete))
        dirs, starts = scorer._direction_runs(continuous)
        self.assertEqual(starts.sum().item(), 3)
        self.assertEqual((dirs != 0).sum().item(), 7)
        self.assertEqual(scorer._turnover_quality(torch.zeros(1, 50)), -2)

    def test_direction_threshold_shared_config(self):
        from strategy_manager.signal import target_to_direction
        with patch.object(Config, "MIN_TRADE_EXPOSURE", .1):
            pos = torch.tensor([[.099, .1, -.099, -.1]], dtype=torch.float64)
            dirs, starts = MT5Backtest._direction_runs(pos)
            self.assertEqual(dirs.tolist()[0], [target_to_direction(v) for v in pos.tolist()[0]])
            self.assertEqual(starts.sum().item(), 2)

    def test_direction_threshold_07_equal_and_just_below(self):
        from strategy_manager.signal import target_to_direction, compute_target_positions
        below = math.nextafter(.7, 0.)
        pos = torch.tensor([[below, .7, -below, -.7]], dtype=torch.float64)
        with patch.object(Config, "MIN_TRADE_EXPOSURE", .7):
            dirs, starts = MT5Backtest._direction_runs(pos)
            self.assertEqual(dirs.tolist(), [[0, 1, 0, -1]])
            self.assertEqual(starts.sum().item(), 2)
            self.assertEqual(dirs.tolist()[0], [target_to_direction(v) for v in pos.tolist()[0]])
            with patch("strategy_manager.signal.torch.tanh", return_value=pos):
                gated = compute_target_positions(torch.zeros_like(pos))
            torch.testing.assert_close(gated, torch.tensor([[0., .7, 0., -.7]], dtype=torch.float64),
                                       rtol=0, atol=0)

    def test_offline_tail_close_cost_and_trade_reconciliation(self):
        engine = BacktestEngine([0], cost_rate=.01, periods_per_year=1)
        raw = {k: v[0] for k, v in raw_fixture(10).items()}
        raw["open"] = torch.full((10,), 100.)
        factor = torch.tensor([1., 1., -1., -1., 0., 1., 1., 1., 3., -3.])
        result = engine._backtest_symbol("TEST", raw, factor)
        self.assertTrue(np.all(result.position[-2:] == 0))
        expected_close = abs(math.tanh(1.)) * .01
        self.assertAlmostEqual(result.final_close_cost, expected_close, places=7)
        self.assertAlmostEqual(result.pnl[-2], -expected_close, places=7)
        self.assertEqual(result.pnl[-1], 0)
        self.assertAlmostEqual(sum(t.pnl for t in result.trades), result.total_return, places=7)
        self.assertAlmostEqual(result.max_drawdown, -result.total_return, places=7)
        self.assertEqual(result.valid_return_bars, 8)
        self.assertIn("proxy", result.pnl_semantics)

    def test_offline_no_valid_return_does_not_open(self):
        engine = BacktestEngine([0], cost_rate=.01)
        for length in (1, 2):
            raw = {k: v[0] for k, v in raw_fixture(length).items()}
            result = engine._backtest_symbol("TEST", raw, torch.ones(length))
            self.assertEqual(result.trades, [])
            self.assertEqual(result.total_return, 0)
            self.assertEqual(result.final_close_cost, 0)

    def test_offline_default_cost_and_neutral_band(self):
        with patch.object(Config, "COST_RATE", .007), patch.object(Config, "MIN_TRADE_EXPOSURE", .1):
            engine = BacktestEngine([0])
            self.assertEqual(engine.cost_rate, .007)
            raw = {k: v[0] for k, v in raw_fixture(10).items()}
            result = engine._backtest_symbol("TEST", raw, torch.full((10,), .01))
            self.assertEqual(result.trades, [])
            self.assertEqual(result.total_return, 0)

    def test_forward_target_alignment_and_tail(self):
        from data_pipeline.data_manager import MT5DataManager
        price = torch.tensor([[100., 101., 103., 99., 105.]])
        actual = MT5DataManager._compute_target_ret(price)
        torch.testing.assert_close(actual[:, :-2], torch.log(price[:, 2:] / price[:, 1:-1]))
        self.assertTrue(torch.all(actual[:, -2:] == 0))


if __name__ == "__main__":
    unittest.main()

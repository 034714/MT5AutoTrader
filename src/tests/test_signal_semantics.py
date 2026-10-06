"""Strategy-version dispatch without a terminal or native MT5 calls."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trading import signal_engine as signals
from model_core.features import FEATURE_SEMANTICS_VERSION
from model_core.vm import NORMALIZATION_VERSION


class SignalSemanticsTests(unittest.TestCase):
    def test_strategy_versions_and_strict_tokens(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "strategy.json"
            for version in ("legacy-v1", FEATURE_SEMANTICS_VERSION):
                path.write_text(json.dumps({"formula": [0], "feature_semantics_version": version}))
                self.assertEqual(signals.load_strategy_file(path)["feature_semantics_version"], version)
            path.write_text(json.dumps({"formula": [0]}))
            self.assertEqual(signals.load_strategy_file(path)["feature_semantics_version"], "legacy-v1")
            for payload in ([1], {"formula": [True]}, {"formula": [0.2]},
                            {"formula": [-1]}, {"formula": [0], "feature_semantics_version": "unknown"}):
                path.write_text(json.dumps(payload))
                with self.assertRaises(signals.StrategyError):
                    signals.load_strategy_file(path)

    def test_signal_selects_feature_and_vm_versions(self):
        raw = {key: torch.ones(1, 800) for key in ("open", "high", "low", "close", "volume")}
        for semantics, normalization in (("legacy-v1", "legacy-v1"),
                                         (FEATURE_SEMANTICS_VERSION, NORMALIZATION_VERSION)):
            with patch.object(signals.MT5FeatureEngineer, "compute_features", return_value=torch.ones(1, 800, 1)) as features, \
                    patch.object(signals, "StackVM") as vm:
                vm.return_value.execute.return_value = torch.ones(1, 800)
                result = signals.compute_signal([[0]], raw, semantics_version=semantics)
                self.assertEqual(result["state"], "ok")
                self.assertEqual(features.call_args.kwargs["semantics_version"], semantics)
                vm.assert_called_once_with(normalization_version=normalization)
                vm.return_value.execute.assert_called_once()

    def test_default_threshold_equal_and_just_below(self):
        raw = {"close": torch.ones(1, 800)}
        for threshold in (None, .05, .7, .8):
            expected = .7 if threshold is None else threshold
            below = math.nextafter(expected, 0.)
            for position, direction in ((expected, "LONG"), (-expected, "SHORT"),
                                        (below, "FLAT"), (-below, "FLAT")):
                with self.subTest(threshold=threshold, position=position), \
                        patch.object(signals.MT5FeatureEngineer, "compute_features"), \
                        patch.object(signals, "StackVM") as vm, \
                        patch.object(signals.math, "tanh", return_value=position):
                    vm.return_value.execute.return_value = torch.ones(1, 800)
                    args = () if threshold is None else (threshold,)
                    result = signals.compute_signal([[0]], raw, *args)
                    self.assertEqual(result["state"], "ok")
                    self.assertEqual(result["direction"], direction)

    def test_shared_threshold_fallback_without_config(self):
        from strategy_manager import signal as shared
        self.assertEqual(shared.MIN_TRADE_EXPOSURE, .7)
        for config in (None, SimpleNamespace(Config=SimpleNamespace())):
            with self.subTest(config=config), patch.dict(sys.modules, {"config": config}):
                self.assertEqual(shared._min_trade_exposure(), .7)
                for position, expected in ((.7, 1), (-.7, -1),
                                           (math.nextafter(.7, 0.), 0),
                                           (-math.nextafter(.7, 0.), 0)):
                    self.assertEqual(shared.target_to_direction(position), expected)

    def test_config_defaults_and_explicit_thresholds_are_isolated(self):
        import config
        self.assertEqual(config.DEFAULT_TRADER_CONFIG["min_trade_exposure"], .7)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(config, "TRADER_CONFIG_FILE", Path(tmp) / "trader_config.json"), \
                patch.object(config, "_LAST_VALID", None), \
                patch.object(config, "_TRADER", dict(config._TRADER)), \
                patch.multiple(config.Config, **{name: value for name, value in vars(config.Config).items()
                                               if name.isupper()}):
            config.Config.reload()
            self.assertEqual(config.Config.MIN_TRADE_EXPOSURE, .7)
            for value in (.05, .8):
                config.TRADER_CONFIG_FILE.write_text(json.dumps({"min_trade_exposure": value}))
                config.Config.reload()
                self.assertEqual(config.Config.MIN_TRADE_EXPOSURE, value)
            config.TRADER_CONFIG_FILE.write_text("{}")
            config.Config.reload()
            self.assertEqual(config.Config.MIN_TRADE_EXPOSURE, .7)
            with patch.object(config, "load_trader_config", return_value={}):
                config.Config.reload()
                self.assertEqual(config.Config.MIN_TRADE_EXPOSURE, .7)
            config.TRADER_CONFIG_FILE.write_text("invalid json")
            with patch.object(config, "_LAST_VALID", None):
                self.assertEqual(config.load_trader_config()["min_trade_exposure"], .7)

    def test_invalid_threshold_fails_before_features(self):
        raw = {"close": torch.ones(1, 800)}
        with patch.object(signals.MT5FeatureEngineer, "compute_features") as features:
            for value in (float("nan"), float("inf"), -0.1, 0, True):
                self.assertEqual(signals.compute_signal([[0]], raw, value)["state"], "error")
            features.assert_not_called()


if __name__ == "__main__":
    unittest.main()

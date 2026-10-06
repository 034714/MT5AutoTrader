"""Strategy-version dispatch without a terminal or native MT5 calls."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
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

    def test_invalid_threshold_fails_before_features(self):
        raw = {"close": torch.ones(1, 800)}
        with patch.object(signals.MT5FeatureEngineer, "compute_features") as features:
            for value in (float("nan"), float("inf"), -0.1, 0, True):
                self.assertEqual(signals.compute_signal([[0]], raw, value)["state"], "error")
            features.assert_not_called()


if __name__ == "__main__":
    unittest.main()

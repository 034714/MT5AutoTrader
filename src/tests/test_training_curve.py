"""Regression coverage for the dashboard training-curve endpoint."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class TrainingCurveTests(unittest.TestCase):
    def test_nonfinite_initial_scores_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "training_history_TEST__M5.json").write_text(json.dumps({
                "step": [0, 1, 2, 3, "bad"],
                "best_score": [float("-inf"), float("nan"), -0.5, 1.25, 2.0],
            }), encoding="utf-8")
            with patch.object(app, "ROOT", root), patch.object(app.training_job, "args", {}):
                result = app.api_training_curve()
        self.assertEqual(result["file"], "training_history_TEST__M5.json")
        self.assertEqual(result["points"], [[2, -0.5], [3, 1.25]])
        self.assertEqual(result["best"], 1.25)
        self.assertEqual(result["skipped_invalid"], 3)

    def test_all_invalid_scores_returns_empty_curve_not_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "training_history_TEST.json").write_text(json.dumps({
                "step": [0, 1], "best_score": [float("-inf"), float("nan")],
            }), encoding="utf-8")
            with patch.object(app, "ROOT", root), patch.object(app.training_job, "args", {}):
                result = app.api_training_curve()
        self.assertEqual(result["points"], [])
        self.assertEqual(result["skipped_invalid"], 2)


if __name__ == "__main__":
    unittest.main()

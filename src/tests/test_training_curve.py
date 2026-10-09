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
            (root / "training_history").mkdir()
            (root / "training_history" / "training_history_TEST__M5.json").write_text(json.dumps({
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
            (root / "training_history").mkdir()
            (root / "training_history" / "training_history_TEST.json").write_text(json.dumps({
                "step": [0, 1], "best_score": [float("-inf"), float("nan")],
            }), encoding="utf-8")
            with patch.object(app, "ROOT", root), patch.object(app.training_job, "args", {}):
                result = app.api_training_curve()
        self.assertEqual(result["points"], [])
        self.assertEqual(result["skipped_invalid"], 2)

    def test_active_job_never_falls_back_to_another_symbols_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "training_history"
            history.mkdir()
            (history / "training_history_OTHER_H1.json").write_text(
                json.dumps({"step": [1], "best_score": [99.]}), encoding="utf-8")
            for args in ({"direct_mt5": True, "symbol": "TEST_", "timeframe": "M15"},
                         {"data_file": str(root / "TEST__M15.parquet")}):
                with self.subTest(args=args), patch.object(app, "ROOT", root), \
                        patch.object(app.training_job, "args", args):
                    self.assertEqual(app.api_training_curve()["points"], [])
                    (history / "training_history_TEST__M15.json").write_text(
                        json.dumps({"step": [3], "best_score": [-.2]}), encoding="utf-8")
                    result = app.api_training_curve()
                    self.assertEqual(result["points"], [[3, -.2]])
                    self.assertEqual(result["symbol"], "TEST__M15")
                    (history / "training_history_TEST__M15.json").unlink()

    def test_file_curve_accepts_case_insensitive_parquet_suffix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            history = root / "training_history"
            history.mkdir()
            curve = history / "training_history_TEST__H1.json"
            curve.write_text(json.dumps({"step": [3], "best_score": [-.2]}), encoding="utf-8")
            for suffix in (".parquet", ".PARQUET", ".Parquet"):
                args = {"data_file": str(root / f"TEST__60min{suffix}")}
                with self.subTest(suffix=suffix), patch.object(app, "ROOT", root), \
                        patch.object(app.training_job, "args", args):
                    result = app.api_training_curve()
                self.assertEqual(result["points"], [[3, -.2]])
                self.assertEqual(result["symbol"], "TEST__H1")
                self.assertEqual(result["file"], curve.name)

    def test_direct_training_start_records_symbol_without_launching_process(self):
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(app.training_job, "running", return_value=False), \
                patch.object(app.backtest_job, "running", return_value=False), \
                patch.object(app.training_job, "start", return_value={"ok": True}) as start, \
                patch.dict(app.os.environ, {"MT5AUTOTRADER_OFFLINE": "0"}):
            app.api_training_start({"direct_mt5": True, "symbol": "TEST_",
                                    "timeframe": "M15", "bars": 800})
            args = start.call_args.args[2]
            self.assertEqual(args["symbol"], "TEST_")
            with patch.object(app, "ROOT", Path(tmp)), \
                    patch.object(app.training_job, "args", args), \
                    patch.object(app.training_job, "status", return_value={}):
                self.assertEqual(app.api_training_status()["info"]["symbol"], "TEST_")


if __name__ == "__main__":
    unittest.main()

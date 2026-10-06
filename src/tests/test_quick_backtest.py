"""Dedicated read-only quick-backtest tests. No terminal or server is started.
Run: python src/tests/test_quick_backtest.py
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from trading import quick_backtest as qb


class ReadOnlyMT5:
    TIMEFRAME_H1 = 991
    ORDER_TYPE_BUY = 81  # Intentionally not real constants.
    ORDER_TYPE_SELL = 82

    def __init__(self, rates):
        self.rates = rates
        self.calls = []
        self.account = SimpleNamespace(login=7, currency="EUR", server="BrokerA")
        self.info = SimpleNamespace(visible=True, point=0.1, volume_min=0.1,
                                    volume_max=10., volume_step=0.1)
        self.symbols = {"TEST"}

    def terminal_info(self):
        return SimpleNamespace(connected=True)

    def account_info(self):
        return self.account

    def symbol_info(self, symbol):
        return self.info if symbol in self.symbols else None

    def copy_rates_from_pos(self, *args):
        self.calls.append(args)
        return self.rates

    def order_calc_profit(self, kind, symbol, lot, entry, exit_price):
        assert kind in (81, 82)
        assert symbol in self.symbols
        # Contract/FX conversion deliberately not just price delta * lot.
        return (exit_price - entry) * lot * 10 * (1 if kind == 81 else -1)

    def __getattr__(self, name):
        raise AssertionError("Forbidden or unexpected MT5 API: " + name)


def rates_fixture(n=1300):
    dtype = [(k, "i8" if k in ("time", "tick_volume", "spread") else "f8")
             for k in ("time", "open", "high", "low", "close", "tick_volume", "spread")]
    rates = np.zeros(n, dtype=dtype)
    rates["time"] = 1700000000 + np.arange(n) * 3600
    rates["open"] = rates["close"] = 100.
    rates["high"], rates["low"] = 101., 99.
    rates["spread"], rates["tick_volume"] = 2, 100
    return rates


class QuickBacktestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "strategies").mkdir()
        self.strategy = self.root / "strategies" / "test.json"
        self.strategy.write_text(json.dumps({"symbol": "TEST", "timeframe": "H1", "formula": [0]}))
        self.cfg = {"signal_bars": 800, "min_trade_exposure": .05, "max_lot_per_trade": 1.,
                    "bindings": [{"strategy_file": "strategies\\test.json", "lot": .3}]}
        self.payload = {"strategy_file": "strategies/test.json", "bars": 500, "lot": .2}
        self.mt5 = ReadOnlyMT5(rates_fixture())

    def tearDown(self):
        self.temp.cleanup()

    def run_bt(self, signals=None, **payload):
        if signals is None:
            signals = np.ones(1300)
        return qb.run_quick_backtest(self.root, self.payload | payload, self.cfg, self.mt5,
                                    signal_fn=lambda rates, meta: signals)

    def test_long_next_open_force_close_costs_account_currency(self):
        self.mt5.rates["close"][-1] = 110
        self.mt5.rates["high"][-1] = 111
        result = self.run_bt(slippage_points=1, commission_per_lot_side=3)
        self.assertEqual(self.mt5.calls, [("TEST", 991, 1, 1300)])
        self.assertEqual(result["currency"], "EUR")
        self.assertEqual(result["bars_used"], 500)
        self.assertEqual(len(result["equity_curve"]), 500)
        trade = result["trades"][0]
        self.assertEqual(trade["entry_time"], int(self.mt5.rates["time"][800]))
        self.assertEqual(trade["signal_time"], int(self.mt5.rates["time"][799]))
        self.assertTrue(trade["forced_close"])
        self.assertAlmostEqual(trade["entry_price"], 100.3)
        self.assertAlmostEqual(trade["exit_price"], 109.9)
        summary = result["summary"]
        self.assertAlmostEqual(summary["gross_profit"], 20)
        self.assertAlmostEqual(summary["spread_cost"], .4)
        self.assertAlmostEqual(summary["slippage_cost"], .4)
        self.assertAlmostEqual(summary["commission"], 1.2)
        self.assertAlmostEqual(summary["net_profit"], 18)
        self.assertAlmostEqual(result["equity_curve"][-1]["equity"], 18)
        self.assertEqual(summary["win_rate"], 1)

    def test_short_bid_ask_and_signal_reversal(self):
        signals = np.ones(1300)
        signals[800:] = -1
        result = self.run_bt(signals)
        self.assertEqual([t["side"] for t in result["trades"]], ["LONG", "SHORT"])
        self.assertEqual(result["trades"][0]["exit_time"], int(self.mt5.rates["time"][801]))
        self.assertEqual(result["trades"][1]["entry_time"], int(self.mt5.rates["time"][801]))
        self.assertAlmostEqual(result["trades"][1]["exit_price"], 100.2)
        self.assertAlmostEqual(result["summary"]["spread_cost"], .8)

    def test_floating_drawdown_not_closed_trade_drawdown(self):
        self.mt5.rates["spread"] = 0
        self.mt5.rates["close"][850] = 50
        self.mt5.rates["low"][850] = 49
        self.mt5.rates["close"][-1] = 110
        self.mt5.rates["high"][-1] = 111
        result = self.run_bt()
        self.assertAlmostEqual(result["summary"]["max_drawdown"], 100)
        self.assertEqual(result["summary"]["closed_trade_max_drawdown"], 0)
        self.assertEqual(result["summary"]["trades"], 1)

    def test_flat_and_no_same_bar_signal_fill(self):
        signals = np.zeros(1300)
        signals[-1] = 1
        result = self.run_bt(signals)
        self.assertEqual(result["trades"], [])
        self.assertEqual(result["summary"]["net_profit"], 0)
        self.assertEqual(result["summary"]["win_rate"], 0)

    def test_threshold_default_and_explicit_boundaries(self):
        for threshold in (None, .05, .7, .8):
            if threshold is None:
                self.cfg.pop("min_trade_exposure", None)
                expected = .7
            else:
                self.cfg["min_trade_exposure"] = threshold
                expected = threshold
            below = np.nextafter(expected, 0.)
            for position, side in ((expected, "LONG"), (-expected, "SHORT"),
                                   (below, None), (-below, None)):
                with self.subTest(threshold=threshold, position=position):
                    result = self.run_bt(np.full(1300, position))
                    self.assertEqual(result["assumptions"]["signal_threshold"], expected)
                    self.assertEqual([t["side"] for t in result["trades"]],
                                     [] if side is None else [side])
        self.assertFalse((self.root / "trader_config.json").exists())

    def test_invalid_input_and_lot(self):
        for payload in ({"bars": True}, {"bars": 499}, {"bars": 50001}, {"bars": 500.0},
                        {"lot": "0.2"}, {"lot": float("nan")}, {"lot": True}, {"lot": .15},
                        {"lot": 2}, {"slippage_points": -1}, {"commission_per_lot_side": float("inf")},
                        {"symbol": "OTHER"}, {"timeframe": "M1"}):
            with self.subTest(payload=payload), self.assertRaises(qb.QuickBacktestError):
                self.run_bt(**payload)

    def test_strategy_path_and_metadata_security(self):
        for path in ("../test.json", "strategies/../test.json", "/test.json", "D:/test.json",
                     "strategies/a/test.json", "test.json:stream", "test.txt", None):
            with self.subTest(path=path), self.assertRaises(qb.QuickBacktestError):
                qb.resolve_strategy(self.root, path)
        for data in ([], {"symbol": "TEST", "formula": [-1]}, {"symbol": "TEST", "formula": [True]},
                     {"formula": [0]}, {"symbol": "TEST", "formula": [0], "timeframe": "INVALID"}):
            self.strategy.write_text(json.dumps(data))
            with self.assertRaises(qb.QuickBacktestError):
                qb.resolve_strategy(self.root, "test.json")

    def test_defaults_from_config_no_writes(self):
        options = qb.get_options(self.root, "test.json", self.cfg, self.mt5)
        self.assertEqual(options["lot"]["default"], .3)
        self.assertEqual(options["lot"]["source"], "config.bindings.lot")
        self.cfg["bindings"] = []
        self.cfg["max_lot_per_trade"] = .7
        self.assertEqual(qb.get_options(self.root, "test.json", self.cfg, self.mt5)["lot"]["default"], .1)
        self.assertEqual(qb.read_config(self.root, self.cfg), self.cfg)
        self.run_bt()
        self.assertFalse((self.root / "trader_config.json").exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["strategies"])

    def test_unbound_default_prefers_01_clamped_to_broker_config_and_step(self):
        # No binding: default 0.1, clamped to [volume_min, min(volume_max, cap)].
        self.cfg["bindings"] = []
        self.cfg["max_lot_per_trade"] = 1.
        self.assertEqual(qb.get_options(self.root, "test.json", self.cfg, self.mt5)["lot"]["default"], .1)
        self.assertEqual(qb.get_options(self.root, "test.json", self.cfg, self.mt5)["lot"]["source"],
                         "默认首选手数（0.1）")
        # Config cap below 0.1 wins.
        self.cfg["max_lot_per_trade"] = .05
        self.mt5.info.volume_min = .01
        self.mt5.info.volume_step = .01
        self.assertEqual(qb.get_options(self.root, "test.json", self.cfg, self.mt5)["lot"]["default"], .05)
        # Broker minimum above 0.1 wins.
        self.cfg["max_lot_per_trade"] = 1.
        self.mt5.info.volume_min = .5
        self.assertEqual(qb.get_options(self.root, "test.json", self.cfg, self.mt5)["lot"]["default"], .5)
        # 0.1 snapped down to the broker volume_step grid.
        self.mt5.info.volume_min = .01
        self.mt5.info.volume_step = .04
        self.assertEqual(qb.get_options(self.root, "test.json", self.cfg, self.mt5)["lot"]["default"], .08)
        # Matching binding still beats the 0.1 fallback.
        self.mt5.info.volume_step = .1
        self.cfg["bindings"] = [{"strategy_file": "strategies/test.json", "lot": .3}]
        options = qb.get_options(self.root, "test.json", self.cfg, self.mt5)
        self.assertEqual(options["lot"]["default"], .3)
        self.assertEqual(options["lot"]["source"], "config.bindings.lot")

    def test_symbol_override_uses_current_terminal_symbol_not_old_binding(self):
        # A strategy from another broker may name a symbol absent in this terminal.
        self.strategy.write_text(json.dumps({"symbol": "OLD_BROKER", "timeframe": "H1", "formula": [0]}))
        self.cfg["bindings"] = [{"strategy_file": "strategies/test.json", "symbol": "OLD_BROKER", "lot": .3}]
        self.mt5.symbols = {"CURRENT"}
        with self.assertRaises(qb.QuickBacktestError):
            qb.get_options(self.root, "test.json", self.cfg, self.mt5)
        options = qb.get_options(self.root, "test.json", self.cfg, self.mt5, symbol="CURRENT")
        self.assertEqual(options["symbol"], "CURRENT")
        self.assertEqual(options["strategy_symbol"], "OLD_BROKER")
        self.assertEqual(options["lot"]["default"], .1)
        self.assertEqual(options["lot"]["source"], "默认首选手数（0.1）")
        result = self.run_bt(symbol="CURRENT")
        self.assertEqual(self.mt5.calls[-1], ("CURRENT", 991, 1, 1300))
        self.assertEqual(result["symbol"], "CURRENT")
        self.assertEqual(result["strategy_symbol"], "OLD_BROKER")

    def test_symbol_override_validation_and_matching_binding(self):
        for symbol in ("", "  ", "bad\nname", 3, "x" * 129):
            with self.subTest(symbol=symbol), self.assertRaises(qb.QuickBacktestError):
                self.run_bt(symbol=symbol)
        self.cfg["bindings"] = [{"strategy_file": "strategies/test.json", "symbol": "TEST", "lot": .3}]
        self.assertEqual(qb.get_options(self.root, "test.json", self.cfg, self.mt5, symbol="TEST")["lot"]["default"], .3)

    def test_missing_data_and_bad_data_fail_not_synthesize(self):
        for rates in (None, self.mt5.rates[:-1], self.mt5.rates[::-1]):
            self.mt5.rates = rates
            with self.assertRaises(qb.QuickBacktestError):
                self.run_bt()
            self.mt5.rates = rates_fixture()
        self.mt5.rates["close"][5] = float("nan")
        with self.assertRaises(qb.QuickBacktestError):
            self.run_bt()

    def test_unavailable_profit_even_with_no_trades(self):
        with patch.object(self.mt5, "order_calc_profit", return_value=None):
            with self.assertRaisesRegex(qb.QuickBacktestError, "order_calc_profit"):
                self.run_bt(np.zeros(1300))

    def test_disconnected_and_hidden_symbol(self):
        with patch.object(self.mt5, "terminal_info", return_value=SimpleNamespace(connected=False)):
            with self.assertRaises(qb.QuickBacktestError):
                self.run_bt()
        self.mt5.info.visible = False
        with self.assertRaises(qb.QuickBacktestError):
            self.run_bt()

    def test_real_engine_prefix_invariance_and_invalid_formula(self):
        # Actual engine, CPU-only synthetic OHLC, never real MT5. Future mutation
        # must not change earlier factors or use future constants in normalization.
        rates = rates_fixture(1350)
        rng = np.random.default_rng(99)
        rates["close"] = 100 + np.cumsum(rng.normal(0, .1, len(rates)))
        rates["open"] = rates["close"] - .01
        rates["high"], rates["low"] = rates["close"] + 1, rates["close"] - 1
        meta = {"formula": [0], "vocab_version": ""}
        prefix = qb.compute_positions(rates[:1300], meta)
        full = qb.compute_positions(rates, meta)
        np.testing.assert_allclose(prefix[1000:], full[1000:1300], atol=1e-5, rtol=1e-5)
        with self.assertRaises(qb.QuickBacktestError):
            qb.compute_positions(rates, {"formula": [999999], "vocab_version": ""})

    def test_real_engine_constant_prefix_cannot_see_future_variation(self):
        import torch
        from model_core.features import MT5FeatureEngineer
        rates = rates_fixture(1350)
        feats = torch.ones(1, 1, 1350)
        feats[:, :, 1300:] = 7
        meta = {"formula": [0], "vocab_version": ""}
        with patch.object(MT5FeatureEngineer, "compute_features", return_value=feats[:, :, :1300]):
            before = qb.compute_positions(rates[:1300], meta)
        with patch.object(MT5FeatureEngineer, "compute_features", return_value=feats):
            after = qb.compute_positions(rates, meta)
        np.testing.assert_array_equal(before, after[:1300])
        self.assertAlmostEqual(before[-1], np.tanh(1), places=6)

    def test_missing_dependency_reports_error(self):
        with patch.dict(sys.modules, {"torch": None}):
            with self.assertRaisesRegex(qb.QuickBacktestError, "依赖不可用"):
                qb.compute_positions(self.mt5.rates, {"formula": [0], "vocab_version": ""})

    def test_account_switch_and_timeout_fail_without_result(self):
        accounts = [self.mt5.account, SimpleNamespace(login=8, currency="EUR", server="BrokerA")]
        with patch.object(self.mt5, "account_info", side_effect=accounts):
            with self.assertRaisesRegex(qb.QuickBacktestError, "账户或服务商已切换"):
                self.run_bt()
        providers = [self.mt5.account, SimpleNamespace(login=7, currency="EUR", server="BrokerB")]
        with patch.object(self.mt5, "account_info", side_effect=providers):
            with self.assertRaisesRegex(qb.QuickBacktestError, "账户或服务商已切换"):
                self.run_bt()
        with patch.object(qb, "MAX_SECONDS", -1):
            with self.assertRaisesRegex(qb.QuickBacktestError, "超时"):
                self.run_bt()

    def test_app_symbol_options_only_returns_visible_terminal_symbols(self):
        import app
        hidden = SimpleNamespace(name="HIDDEN", visible=False)
        visible = SimpleNamespace(name="VISIBLE", visible=True)
        fake_mt5 = SimpleNamespace(symbols_get=lambda: [hidden, visible],
                                   account_info=lambda: SimpleNamespace(server="BrokerA", login=7, currency="EUR"))
        with patch.object(app, "_get_mt5_client", return_value=object()), patch.dict(sys.modules, {"MetaTrader5": fake_mt5}):
            self.assertEqual(app.api_mt5_symbols(), {"symbols": ["VISIBLE"], "account_key": ["BrokerA", 7, "EUR"]})

    def test_app_symbol_options_rejects_account_switch(self):
        import app
        accounts = [SimpleNamespace(server="BrokerA", login=7, currency="EUR"),
                    SimpleNamespace(server="BrokerB", login=7, currency="EUR")]
        fake_mt5 = SimpleNamespace(symbols_get=lambda: [], account_info=lambda: accounts.pop(0))
        with patch.object(app, "_get_mt5_client", return_value=object()), patch.dict(sys.modules, {"MetaTrader5": fake_mt5}):
            with self.assertRaises(app.HTTPException) as caught:
                app.api_mt5_symbols()
        self.assertEqual(caught.exception.status_code, 409)

    def test_app_endpoints_no_server_no_side_effects(self):
        import app
        from fastapi import HTTPException
        with patch.object(app, "ROOT", self.root), patch.object(app, "_quick_mt5_api", return_value=self.mt5), \
             patch.object(app, "_mt5_client", None), patch.object(qb, "compute_positions", return_value=np.ones(1300)), \
             patch.object(app, "save_trader_config", side_effect=AssertionError("config write")), \
             patch.object(app, "load_trader_config", side_effect=AssertionError("write-capable config read")), \
             patch.object(app.subprocess, "Popen", side_effect=AssertionError("process spawn")):
            # App defaults warmup is 1000, serve exactly that amount of history.
            self.mt5.rates = rates_fixture(1500)
            with patch.object(qb, "compute_positions", return_value=np.zeros(1500)):
                options = app.api_quick_backtest_options("test.json")
                self.assertEqual(options["currency"], "EUR")
                result = app.api_quick_backtest(self.payload)
                self.assertTrue(result["ok"])
            app._quick_bt_lock.acquire()
            try:
                with self.assertRaises(HTTPException) as ctx:
                    app.api_quick_backtest(self.payload)
                self.assertEqual(ctx.exception.status_code, 409)
            finally:
                app._quick_bt_lock.release()
            with self.assertRaises(HTTPException) as ctx:
                app.api_quick_backtest(self.payload | {"bars": 1})
            self.assertEqual(ctx.exception.status_code, 400)
            with patch.object(qb, "run_quick_backtest", side_effect=qb.QuickBacktestError("missing", 503)):
                with self.assertRaises(HTTPException):
                    app.api_quick_backtest(self.payload)
            self.assertFalse(app._quick_bt_lock.locked())


if __name__ == "__main__":
    unittest.main()

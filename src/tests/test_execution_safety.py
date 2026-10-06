"""Mock-only execution regressions. Never initialize or call native MT5."""
import atexit
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_trading as base

atexit.register(shutil.rmtree, base._TMP, ignore_errors=True)
client_mod = base.client_mod
runner_mod = base.runner_mod


def result(code=10009, volume=.1, order=77):
    return NS(retcode=code, comment="mock", order=order, deal=88, volume=volume, price=100)


class ExecutionClientTests(unittest.TestCase):
    def setUp(self):
        self.api = NS(TRADE_ACTION_DEAL=1, TRADE_ACTION_PENDING=5,
                      TRADE_ACTION_SLTP=6, TRADE_RETCODE_DONE=10009,
                      TRADE_RETCODE_DONE_PARTIAL=10010, TRADE_RETCODE_PLACED=10008,
                      TRADE_RETCODE_INVALID_FILL=10030, ORDER_TYPE_BUY=0,
                      ORDER_TYPE_SELL=1, ORDER_TIME_GTC=0,
                      order_send=Mock(), last_error=Mock(return_value=(-1, "lost")),
                      positions_get=Mock(), symbol_info_tick=Mock(return_value=NS(bid=100, ask=101)))
        self.patch = patch.object(client_mod, "mt5", self.api)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.client = client_mod.MT5Client(magic=7, dry_run=False)
        self.client._connected = True
        self.client._fill_modes = lambda symbol: [0, 1, 2]
        self.pos = NS(ticket=77, symbol="TEST", magic=7, type=0, volume=.1,
                      price_open=100, sl=98, tp=110)

    def test_none_and_exception_never_resend(self):
        for response in (None, RuntimeError("transport lost")):
            with self.subTest(response=response):
                self.api.order_send.reset_mock()
                self.api.order_send.side_effect = [response, result()]
                out = self.client._send_request({"action": 1, "symbol": "TEST", "volume": .1})
                self.assertEqual(out["status"], "unknown")
                self.assertTrue(out["unknown"])
                self.assertEqual(self.api.order_send.call_count, 1)

    def test_only_invalid_fill_retries(self):
        self.api.order_send.side_effect = [result(10030), result()]
        out = self.client._send_request({"action": 1, "symbol": "TEST"})
        self.assertEqual(self.api.order_send.call_count, 2)
        self.assertEqual(out["status"], "completed")
        self.api.order_send.reset_mock()
        self.api.order_send.side_effect = [result(10006), result()]
        self.assertFalse(self.client._send_request({"action": 1})["ok"])
        self.assertEqual(self.api.order_send.call_count, 1)

    def test_action_status_and_actual_execution_fields(self):
        for action, code, status in ((5, 10008, "accepted"), (5, 10009, "accepted"),
                                     (1, 10008, "accepted"), (1, 10010, "partial"),
                                     (1, 10009, "completed"), (6, 10009, "completed")):
            self.api.order_send.return_value = result(code, .03)
            out = self.client._send_request({"action": action, "symbol": "TEST", "volume": .1})
            self.assertEqual(out["status"], status)
            self.assertEqual((out["order"], out["deal"], out["volume"]), (77, 88, .03))

    def test_failed_query_is_not_absent_success(self):
        self.api.positions_get.return_value = None
        self.assertFalse(self.client.close_position("TEST", 77))
        self.assertFalse(self.client.close_symbol_all("TEST"))
        self.api.order_send.assert_not_called()

    def test_full_close_partial_fill_is_not_full_success(self):
        remaining = NS(**vars(self.pos))
        remaining.volume = .07
        self.api.positions_get.side_effect = [(self.pos,), (remaining,)]
        self.api.order_send.return_value = result(10010, .03)
        self.assertFalse(self.client.close_position("TEST", 77))
        self.assertEqual(self.client.last_execution["status"], "partial")
        self.assertEqual(self.client.last_execution["remaining_volume"], .07)

    def test_partial_close_verifies_remaining(self):
        remaining = NS(**vars(self.pos))
        remaining.volume = .05
        self.api.positions_get.side_effect = [(self.pos,), (remaining,)]
        self.api.order_send.return_value = result(10009, .05)
        self.assertTrue(self.client.close_position("TEST", 77, volume=.05))
        self.assertTrue(self.client.last_execution["partial"])
        self.assertEqual(self.client.last_execution["volume"], .05)

    def test_accepted_close_needs_authoritative_verification(self):
        self.api.positions_get.side_effect = [(self.pos,), None]
        self.api.order_send.return_value = result()
        self.assertFalse(self.client.close_position("TEST", 77))
        self.assertTrue(self.client.last_execution["unknown"])
        self.api.positions_get.side_effect = [(self.pos,), ()]
        self.assertTrue(self.client.close_position("TEST", 77))

    def test_sl_preserves_tp_and_explicit_zero_clears(self):
        self.api.positions_get.return_value = (self.pos,)
        self.api.order_send.return_value = result()
        self.assertTrue(self.client.modify_sl("TEST", 77, 99))
        self.assertEqual(self.api.order_send.call_args.args[0]["tp"], 110)
        self.assertTrue(self.client.modify_sl("TEST", 77, tp=0))
        req = self.api.order_send.call_args.args[0]
        self.assertEqual((req["sl"], req["tp"]), (98, 0))


class ExecutionRunnerTests(unittest.TestCase):
    def setUp(self):
        self.sleep = patch.object(runner_mod.time, "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)
        self.r = base.make_runner(base.MockClient(), "live")
        self.binding = {"symbol": "TEST", "lot": .1, "strategy_file": "s.json"}
        self.r._ov_applied_ts = {}
        self.r._load_partial_overrides = lambda: {}

    def test_unobserved_and_unknown_open_persist_barrier(self):
        for outcome in ({"ok": False, "status": "unknown", "unknown": True, "retcode": None},
                        {"ok": True, "status": "accepted", "retcode": 10008, "order": 77}):
            self.r.book.state["pending_execution"].clear()
            self.r.client.market_open = Mock(return_value=outcome)
            self.assertFalse(self.r._open_position("TEST", "BUY", self.binding, "s.json"))
            self.r.book = runner_mod.PositionBook("live")
            self.r._reconcile(self.binding, "LONG", .8)
            self.assertEqual(self.r.client.market_open.call_count, 1)
            self.assertIn("TEST", self.r.book.state["pending_execution"])

    def test_open_matches_result_not_first_position(self):
        def open_order(*args, **kwargs):
            self.r.client.positions = [base.MockPosition(66, "TEST", 0, .1, 100),
                                       base.MockPosition(77, "TEST", 0, .03, 101)]
            return {"ok": True, "status": "completed", "retcode": 10009, "order": 77}
        self.r.client.market_open = open_order
        self.assertTrue(self.r._open_position("TEST", "BUY", self.binding, "s.json"))
        self.assertEqual(self.r.book.get_position("TEST")["ticket"], 77)
        self.assertEqual(self.r.book.get_position("TEST")["volume"], .03)

    def test_unknown_close_does_not_resend_or_reverse(self):
        self.r.client.positions = [base.MockPosition(77, "TEST", 0, .1, 100)]
        self.r.client.close_symbol_all = Mock(return_value=False)
        self.r.client.last_execution = {"unknown": True, "status": "unknown"}
        self.assertFalse(self.r._close_symbol("TEST"))
        self.r._reconcile(self.binding, "SHORT", .8)
        self.assertEqual(self.r.client.close_symbol_all.call_count, 1)
        self.assertFalse(any(c[0] == "open" for c in self.r.client.calls))

    def test_same_bar_failure_retries_then_success_and_hold_consume(self):
        import numpy as np
        rates = np.array([(1,), (2,), (3,)], dtype=[("time", "i8")])
        self.r.strategies = {"s.json": {"formula": [1], "timeframe": "H1"}}
        self.r.client.copy_rates = lambda *a: rates
        with patch.object(runner_mod, "rates_to_raw_dict", return_value={}), \
             patch.object(runner_mod, "compute_signal", return_value={"state": "ok", "direction": "LONG"}), \
             patch.object(self.r, "_reconcile", side_effect=[False, True]) as reconcile:
            self.r._process_symbol(self.binding)
            self.assertNotIn("TEST", self.r._last_bar_time)
            self.r._process_symbol(self.binding)
            self.r._process_symbol(self.binding)
            self.assertEqual(reconcile.call_count, 2)
            self.assertEqual(self.r._last_bar_time["TEST"], 2)
        self.r.client.positions = [base.MockPosition(77, "TEST", 0, .1, 100)]
        self.assertTrue(self.r._reconcile(self.binding, "LONG", .8))

    def test_account_switch_stops_open_close_and_protection(self):
        self.r.book.set_position("TEST", {"ticket": 77, "direction": "BUY", "volume": .1})
        self.r.client.account_info = lambda: NS(login=222, server="Other", currency="USD")
        self.assertFalse(self.r._open_position("TEST", "BUY", self.binding, ""))
        self.assertFalse(self.r._close_symbol("TEST"))
        self.r._monitor_positions()
        self.assertTrue(self.r._stop)
        self.assertEqual(self.r.client.calls, [])
        self.assertIsNotNone(self.r.book.get_position("TEST"))

    def test_manual_plans_disabled_sr_multi_ticket_and_no_repeat(self):
        from trading.sr import SRParams
        self.r.client.positions = [base.MockPosition(t, "TEST", 0, .1, 100) for t in (77, 78)]
        account_key = self.r._current_account_identity()
        self.r._load_partial_overrides = lambda: {
            str(t): {"price": 99, "close_volume": .05, "ts": 1,
                     "account_key": account_key} for t in (77, 78)}
        with patch("trading.sr.SRParams.from_config", return_value=SRParams(enabled=False, partial_enabled=False)):
            self.r._check_partial_take_profits()
            self.r._check_partial_take_profits()
        calls = [c for c in self.r.client.calls if c[0] == "close"]
        self.assertEqual({c[2] for c in calls}, {77, 78})
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(self.r._management_entries()), 2)
        self.assertTrue(all(info["sr_partial"]["done"] for _, info in self.r._management_entries()))
        self.assertEqual(self.r.book.cooldown_start("TEST"), 0)

    def test_overrides_require_current_account_even_for_reused_ticket(self):
        self.r.book.set_position("TEST", {"ticket": 77, "direction": "BUY",
                                           "volume": .1, "sl": 98})
        for key in (None, ["Other", 222, "USD"],
                    ["Demo-Server", 12345678, "EUR"]):
            with self.subTest(account_key=key):
                override = {"account_key": key, "ts": 1, "price": 99,
                            "close_volume": .05, "sl": 99,
                            "applied_to_mt5": True}
                self.r._apply_partial_overrides({"77": override})
                self.r._apply_sl_overrides({"77": override})
                info = self.r.book.get_position("TEST")
                self.assertNotIn("sr_partial", info)
                self.assertEqual(info["sl"], 98)
                self.assertFalse(self.r.book.manual_sl_tickets())
        self.assertFalse(self.r.client.calls)

    def test_nonfinite_risk_inputs_never_modify(self):
        for field, value in (("open_price", float("nan")), ("current_sl", float("inf")),
                             ("already_locked", float("nan"))):
            args = dict(symbol="TEST", ticket=77, direction="BUY", open_price=100,
                        current_sl=98, already_locked=-1)
            args[field] = value
            self.assertEqual(self.r.risk.protect_position(**args)[2], "invalid_input")
        self.r.client.set_price(float("inf"), 101)
        self.assertEqual(self.r.risk.protect_position("TEST", 77, "BUY", 100, 98, -1)[2], "invalid_tick")
        self.assertFalse(self.r.client.calls)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Isolated cooldown regression tests; no terminal connection or real orders."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_trading as base

runner_mod = base.runner_mod


class ReentryCooldownTests(unittest.TestCase):
    def setUp(self):
        self.clock = patch.object(runner_mod.time, "time", return_value=1000)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.sleep = patch.object(runner_mod.time, "sleep")
        self.sleep.start()
        self.addCleanup(self.sleep.stop)

    def test_controlled_dry_close_persists_and_direct_open_is_blocked(self):
        r = base.make_runner(base.MockClient(dry_run=True), "dry")
        binding = {"symbol": "BTCUSD_", "lot": .01}
        self.assertTrue(r._open_position("BTCUSD_", "BUY", binding, ""))
        self.assertTrue(r._close_symbol("BTCUSD_"))
        self.assertEqual(runner_mod.PositionBook("dry").cooldown_start("BTCUSD_"), 1000)
        self.assertFalse(r._open_position("BTCUSD_", "SELL", binding, ""))
        self.assertTrue(r._close_symbol("BTCUSD_"))
        self.assertEqual(r.book.cooldown_start("BTCUSD_"), 1000)

    def test_dry_tp_persists_and_partial_does_not_start_cooldown(self):
        r = base.make_runner(base.MockClient(dry_run=True, bid=110, ask=110), "dry")
        r.book.set_position("BTCUSD_", {"ticket": 1, "direction": "BUY",
            "volume": .1, "open_price": 100, "sl": 98, "tp": 105})
        r._check_dry_take_profits()
        self.assertEqual(runner_mod.PositionBook("dry").cooldown_start("BTCUSD_"), 1000)
        from trading.history import latest_full_close_time
        ds = [base._deal(1, 0, 0, .1, r.client.magic, time=900),
              base._deal(1, 1, 1, .05, 0, time=990)]
        self.assertIsNone(latest_full_close_time(ds, "BTCUSD_", r.client.magic))

    def test_history_newer_close_updates_old_cooldown_without_refreshing(self):
        r = base.make_runner(base.MockClient(), "live")
        r.book.mark_full_close("BTCUSD_", 800)
        ds = [base._deal(1, 0, 0, .1, r.client.magic, time=100),
              base._deal(1, 1, 1, .1, 0, time=990)]
        r.client.history_deals_get = lambda *a, **k: ds
        self.assertTrue(r._backfill_cooldowns_from_history())
        self.assertEqual(r.book.cooldown_start("BTCUSD_"), 990)
        r._backfill_cooldowns_from_history()
        self.assertEqual(r.book.cooldown_start("BTCUSD_"), 990)
        self.assertEqual(runner_mod.PositionBook("live").cooldown_start("BTCUSD_"), 990)
        ds[0].magic = 999
        r.book.state["cooldowns"].clear()
        r._backfill_cooldowns_from_history()
        self.assertEqual(r.book.cooldown_start("BTCUSD_"), 0)

    def test_other_position_same_symbol_prevents_full_close(self):
        r = base.make_runner(base.MockClient(), "live")
        r.book.set_position("BTCUSD_", {"ticket": 1, "direction": "BUY"})
        r.client.positions = [base.MockPosition(2, "BTCUSD_", 1, .1, 100)]
        r._sync_live_positions()
        self.assertEqual(r.book.cooldown_start("BTCUSD_"), 0)
        ds = [base._deal(1, 0, 0, .1, r.client.magic, time=900),
              base._deal(1, 1, 1, .1, 0, time=990)]
        r.client.history_deals_get = lambda *a, **k: ds
        r._backfill_cooldowns_from_history()
        self.assertEqual(r.book.cooldown_start("BTCUSD_"), 0)

    def test_history_unavailable_blocks_entry_not_close(self):
        r = base.make_runner(base.MockClient(), "live")
        r.client.history_deals_get = lambda *a, **k: None
        self.assertFalse(r._open_position("BTCUSD_", "BUY", {"lot": .01}, ""))
        self.assertFalse(any(c[0] == "open" for c in r.client.calls))
        r.client.positions = [base.MockPosition(1, "BTCUSD_", 0, .1, 100)]
        self.assertTrue(r._close_symbol("BTCUSD_"))
        self.assertEqual(r.book.cooldown_start("BTCUSD_"), 1000)

    def test_mt5_read_helpers_use_range_and_complete_position_history(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        api = Mock()
        api.DEAL_ENTRY_OUT = 1
        api.DEAL_ENTRY_OUT_BY = 3
        exit_deal = base._deal(12, 1, 1, .1, 0, time=990)
        entry = base._deal(12, 0, 0, .1, 7, time=1)
        api.history_deals_get.side_effect = [[exit_deal], [entry, exit_deal]]
        client = base.client_mod.MT5Client.__new__(base.client_mod.MT5Client)
        client._connected = True
        client.magic = 7
        with patch.object(base.client_mod, "mt5", api):
            self.assertEqual(client.history_deals_get(800, 1001), [entry, exit_deal])
            self.assertEqual(api.history_deals_get.call_args_list[0].args, (800, 1001))
            self.assertEqual(api.history_deals_get.call_args_list[1].kwargs, {"position": 12})
            api.positions_get.return_value = None
            with self.assertRaises(ConnectionError):
                client.get_positions(strict=True)
            api.positions_get.return_value = ()
            self.assertEqual(client.get_positions(strict=True), [])
            api.order_send.assert_not_called()

    def test_external_close_after_restart_blocks_both_directions(self):
        client = base.MockClient(dry_run=False)
        r = base.make_runner(client, "live")
        r._server_offset = 10800
        r.bindings = [{"symbol": "BTCUSD_", "lot": 0.01}]
        r.book.set_position("BTCUSD_", {
            "ticket": 12, "direction": "BUY", "volume": 0.1,
            "open_price": 100, "sl": 98,
        })
        r.book.save()
        r.book = runner_mod.PositionBook("live")
        deals = [base._deal(12, 0, 0, .1, client.magic, time=10000),
                 base._deal(12, 1, 1, .1, 0, time=11790)]
        client.history_deals_get = lambda *args, **kwargs: deals
        with patch.object(runner_mod.time, "time", return_value=1000), \
             patch.object(runner_mod.time, "sleep"), \
             patch.object(runner_mod, "load_trader_config", return_value={"risk": {}}):
            r._sync_live_positions()
            self.assertEqual(r.book.cooldown_start("BTCUSD_"), 990)
            restarted = runner_mod.PositionBook("live")
            self.assertEqual(restarted.cooldown_start("BTCUSD_"), 990,
                             "External close must persist before the next signal")
            for direction in ("LONG", "SHORT"):
                r._reconcile(r.bindings[0], direction, .8)
            self.assertFalse(any(call[0] == "open" for call in client.calls))


if __name__ == "__main__":
    unittest.main(verbosity=2)

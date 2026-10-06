"""Mock-only dashboard API contract tests; no MT5, orders, or child processes."""
import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class MockClient:
    def __init__(self, login=7, server="BrokerA", currency="USD"):
        self.connected = True
        self.dry_run = True
        self.ai = SimpleNamespace(login=login, server=server, currency=currency,
                                  balance=100.0, equity=101.0, margin_free=90.0)
        self.calls = []

    def account_info(self):
        return self.ai

    def get_positions(self):
        self.calls.append("positions")
        return []

    def server_time_offset(self):
        return 0

    def trade_allowed(self):
        return True


class DashboardContractTests(unittest.TestCase):
    def setUp(self):
        self.client = MockClient()
        self.key = ["BrokerA", 7, "USD"]
        self.patches = [patch.object(app, "_get_mt5_client", return_value=self.client),
                        patch.object(app, "_find_any_position", return_value=None)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()

    def test_offline_mode_never_connects_or_starts_mt5_jobs(self):
        self.patches[0].stop()
        with patch.dict(app.os.environ, {"MT5AUTOTRADER_OFFLINE": "1"}), \
             patch("trading.mt5_client.MT5Client") as client, \
             patch.object(app, "subprocess") as sub, \
             patch.object(app.training_job, "running", return_value=False), \
             patch.object(app.backtest_job, "running", return_value=False):
            self.assertIsNone(app._get_mt5_client())
            with self.assertRaises(app.HTTPException) as runner_error:
                app.api_runner_start()
            self.assertEqual(runner_error.exception.status_code, 409)
            with self.assertRaises(app.HTTPException) as train_error:
                app.api_training_start({"direct_mt5": True, "symbol": "TEST"})
            self.assertEqual(train_error.exception.status_code, 409)
            client.assert_not_called()
            sub.Popen.assert_not_called()

    def test_history_uses_aware_server_epoch_window(self):
        now = 1791296700.0
        self.client.magic = 20260904
        self.client.server_time_offset = lambda: 10800
        deal = SimpleNamespace(position_id=62417586, time=1791296130,
                               entry=0, magic=20260904, type=1, volume=.1,
                               price=31245.48, profit=0., commission=-.1,
                               swap=0., symbol="NQ100_", comment="mock")
        exit_deal = SimpleNamespace(**vars(deal))
        exit_deal.time = 1791296410
        exit_deal.entry = 1
        exit_deal.type = 0
        exit_deal.price = 31244.41
        exit_deal.profit = .11
        exit_deal.commission = 0.
        calls = []
        def history(frm, to):
            calls.append((frm, to))
            return (deal, exit_deal) if frm.tzinfo is app.timezone.utc and to.tzinfo is app.timezone.utc else ()
        with patch.dict(sys.modules, {"MetaTrader5": SimpleNamespace(history_deals_get=history)}), \
             patch.object(app.time, "time", return_value=now):
            result = app.api_mt5_history(days=7)
        frm, to = calls[0]
        self.assertEqual(to.timestamp(), now + 10800)
        self.assertEqual(frm.timestamp(), now - 7 * 86400 + 10800)
        self.assertEqual(result["total_deals"], 2)
        self.assertEqual(len(result["closed"]), 1)
        self.assertEqual(result["closed"][0]["position_id"], 62417586)
        self.assertEqual(result["closed"][0]["profit"], .01)

    def test_invalid_config_types_do_not_call_save(self):
        with patch.object(app, "save_trader_config") as save:
            with self.assertRaises(app.HTTPException):
                app.api_update_config({"dry_run": "false"})
            with self.assertRaises(app.HTTPException):
                app.api_update_config({"risk": {"price_monitor_interval": 0}})
            save.assert_not_called()

    def test_trading_confirmation_and_account_key_are_strict(self):
        with self.assertRaises(app.HTTPException):
            app.api_mt5_order({"confirmed": 1, "symbol": "ETHUSD_", "direction": "BUY", "lot": 0.1})
        with self.assertRaises(app.HTTPException):
            app.api_mt5_order({"confirmed": True, "account_key": ["BrokerA", 7, "EUR"],
                               "symbol": "ETHUSD_", "direction": "BUY", "lot": 0.1})
        self.assertFalse(self.client.calls)

    def test_zero_lot_is_rejected_without_client_call(self):
        with self.assertRaises(app.HTTPException):
            app.api_mt5_order({"confirmed": True, "account_key": self.key,
                               "symbol": "ETHUSD_", "direction": "BUY", "lot": 0})
        self.assertFalse(self.client.calls)

    def test_host_and_origin_guard(self):
        async def call_chain():
            called = []
            async def next_call(request):
                called.append(True)
                return SimpleNamespace(headers={})
            bad = SimpleNamespace(headers={"host": "evil.example"})
            result = await app.local_origin_guard(bad, next_call)
            self.assertEqual(result.status_code, 403)
            self.assertFalse(called)
            good = SimpleNamespace(headers={"host": "[::1]:8900", "origin": "http://localhost:8900"})
            await app.local_origin_guard(good, next_call)
            self.assertTrue(called)
        asyncio.run(call_chain())

    def test_startup_preserves_existing_runner_and_requires_bound_dry_mode(self):
        for offline, alive, dry, bindings, expected in (
            (False, True, False, [{"symbol": "TEST"}], 0),
            (False, False, False, [{"symbol": "TEST"}], 0),
            (False, False, True, [], 0),
            (True, False, True, [{"symbol": "TEST"}], 0),
            (False, False, True, [{"symbol": "TEST"}], 1),
        ):
            with self.subTest(offline=offline, alive=alive, dry=dry, bindings=bindings), \
                 patch.dict(app.os.environ, {"MT5AUTOTRADER_OFFLINE": "1" if offline else "0"}), \
                 patch.object(app, "_runner_alive", return_value={"process_alive": alive}), \
                 patch.object(app, "load_trader_config", return_value={"dry_run": dry, "bindings": bindings}), \
                 patch.object(app, "api_runner_start") as start:
                app._autostart_dry_runner()
                self.assertEqual(start.call_count, expected)

    def test_runner_start_uses_busy_lock_without_spawning(self):
        acquired = app._runner_start_lock.acquire(blocking=False)
        self.assertTrue(acquired)
        try:
            with patch.object(app, "subprocess") as sub:
                with self.assertRaises(app.HTTPException) as caught:
                    app.api_runner_start()
                self.assertEqual(caught.exception.status_code, 409)
                sub.Popen.assert_not_called()
        finally:
            app._runner_start_lock.release()

    def test_status_positions_is_list_and_account_switch_clears_runner_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            status = root / "runner_status.json"
            status.write_text(json.dumps({"pid": 1, "running": True,
                                          "account": {"server": "Other", "login": 9, "currency": "USD"},
                                          "positions": {"ETHUSD_": {"ticket": 1}}, "signals": {"x": 1}}), encoding="utf-8")
            with patch.object(app, "STATUS_FILE", status), patch.object(app, "RUNNER_PID_FILE", root / "pid"), patch.object(app, "_pid_alive", return_value=True):
                result = app.api_status()
            self.assertIsInstance(result["runner_status"]["positions"], list)
            self.assertEqual(result["runner_status"]["positions"], [])
            self.assertEqual(result["account_key"], self.key)

    def test_http_routes_validate_json_and_foreign_origin_without_mt5(self):
        from fastapi.testclient import TestClient
        with TestClient(app.app, base_url="http://127.0.0.1:8900") as http, patch.object(app, "_get_mt5_client") as client:
            for path, payload in (("/api/mt5/order", {"confirmed": "true"}),
                                  ("/api/mt5/position/sl", {"confirmed": True, "ticket": 1, "sl": "12"}),
                                  ("/api/mt5/position/tp", {"confirmed": False, "ticket": 1, "tp": 0}),
                                  ("/api/mt5/position/partial", {"confirmed": True, "ticket": 1, "price": 12, "volume": 0}),
                                  ("/api/mt5/pending/order", {"check_only": "true"})):
                self.assertEqual(http.post(path, json=payload).status_code, 400)
            self.assertEqual(http.post("/api/mt5/order", json={}, headers={"Origin": "https://evil.example"}).status_code, 403)
            self.assertEqual(http.get("/api/config", headers={"Host": "evil.example:8900"}).status_code, 403)
            client.assert_not_called()

    def test_check_only_cannot_bypass_confirmation_on_live_actions(self):
        for fn, values in ((app.api_position_sl, {"sl": 12}),
                           (app.api_position_tp, {"tp": 0}),
                           (app.api_position_close, {})):
            with self.subTest(fn=fn.__name__), patch.object(app, "_get_mt5_client") as client:
                with self.assertRaises(app.HTTPException):
                    fn({"ticket": 1, "check_only": True, "account_key": self.key, **values})
                client.assert_not_called()

    def test_nonfinite_and_untyped_requests_rejected_before_client_access(self):
        base = {"confirmed": True, "account_key": self.key, "symbol": "ETHUSD_", "direction": "BUY", "lot": 0.1}
        for value in (None, True, "0.1", 0, -1, float("nan"), float("inf"), 10 ** 400):
            with self.subTest(value=repr(value)), patch.object(app, "_get_mt5_client") as client:
                with self.assertRaises(app.HTTPException):
                    app.api_mt5_order({**base, "lot": value})
                client.assert_not_called()

    def test_runner_waiting_is_alive_but_not_ready_and_blocks_start(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            status = root / "runner_status.json"
            status.write_text(json.dumps({"pid": 123, "phase": "waiting", "running": False}), encoding="utf-8")
            with patch.object(app, "STATUS_FILE", status), patch.object(app, "RUNNER_PID_FILE", root / "pid"), patch.object(app, "_runner_process", None), patch.object(app, "_pid_alive", return_value=True), patch.object(app.subprocess, "Popen") as spawn:
                state = app._runner_alive()
                self.assertTrue(state["process_alive"])
                self.assertFalse(state["ready"])
                self.assertTrue(app.api_runner_start()["ok"])
                spawn.assert_not_called()

    def test_job_busy_preserves_stop_flag_and_spawn_failure_closes_handle(self):
        from unittest.mock import Mock
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            flag = root / "TRAIN_STOP"
            flag.write_text("STOP", encoding="utf-8")
            with patch.object(app, "ROOT", root), patch.object(app.training_job, "proc", Mock(poll=lambda: None)), patch.object(app.subprocess, "Popen") as spawn:
                with self.assertRaises(app.HTTPException):
                    app.api_training_start({})
                self.assertTrue(flag.exists())
                spawn.assert_not_called()
            handles = []
            def fail(*args, **kwargs):
                handles.append(kwargs["stdout"])
                raise OSError("mock failure")
            job = app.JobManager("测试")
            with patch.object(app, "ROOT", root), patch.object(app.training_job, "proc", None), patch.object(app.backtest_job, "proc", None), patch.object(app.subprocess, "Popen", side_effect=fail):
                with self.assertRaises(OSError):
                    job.start(["mock"], root / "mock.log", {})
            self.assertEqual(len(handles), 1)
            self.assertTrue(handles[0].closed)

    def test_partial_clear_failure_and_override_failure_restore_tp(self):
        from unittest.mock import Mock
        pos = SimpleNamespace(ticket=1, symbol="ETHUSD_", volume=1.0, sl=5.0, tp=15.0)
        payload = {"confirmed": True, "account_key": self.key, "ticket": 1,
                   "price": 12.0, "volume": 0.5, "clear_tp": True}
        self.client.modify_sl = Mock(return_value=False)
        with patch.object(app, "_find_status_position", return_value=None), patch.object(app, "_find_any_position", return_value=pos), patch.object(app, "_atomic_override") as write:
            result = app.api_position_partial(payload)
            self.assertFalse(result["ok"])
            write.assert_not_called()
            self.client.modify_sl = Mock(return_value=True)
            write.side_effect = OSError("mock disk failure")
            with self.assertRaises(app.HTTPException):
                app.api_position_partial(payload)
            self.assertEqual([call.kwargs["tp"] for call in self.client.modify_sl.call_args_list], [0, 15.0])

    def test_config_rmw_concurrency_preserves_independent_updates(self):
        import config
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as td, patch.object(config, "TRADER_CONFIG_FILE", Path(td) / "trader_config.json"), patch.object(config, "_LAST_VALID", None):
            config.save_trader_config(json.loads(json.dumps(config.DEFAULT_TRADER_CONFIG)))
            payloads = [{"risk": {"sr_buffer_atr": 0.7}},
                        {"risk": {"sr_tp_buffer_atr": 0.3}},
                        {"max_open_positions": 4}, {"min_trade_exposure": 0.2}]
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(app.api_update_config, payloads))
            self.assertTrue(all(r["ok"] for r in results))
            saved = config.load_trader_config()
            self.assertEqual(saved["risk"]["sr_buffer_atr"], 0.7)
            self.assertEqual(saved["risk"]["sr_tp_buffer_atr"], 0.3)
            self.assertEqual(saved["max_open_positions"], 4)
            self.assertEqual(saved["min_trade_exposure"], 0.2)

    def test_invalid_config_file_preserves_last_valid_risk(self):
        import config
        with tempfile.TemporaryDirectory() as td, patch.object(config, "TRADER_CONFIG_FILE", Path(td) / "trader_config.json"), patch.object(config, "_LAST_VALID", None):
            cfg = config.load_trader_config()
            cfg["risk"]["stop_loss_pct"] = -0.04
            config.save_trader_config(cfg)
            config.TRADER_CONFIG_FILE.write_text('{"risk":{"stop_loss_pct":null}}', encoding="utf-8")
            self.assertEqual(config.load_trader_config()["risk"]["stop_loss_pct"], -0.04)
            self.assertFalse(list(Path(td).glob("*.tmp")))

    def test_override_records_expected_account_and_partial_is_pending_without_runner(self):
        pos = SimpleNamespace(ticket=1, symbol="ETHUSD_", volume=1.0, sl=5.0, tp=0.0)
        payload = {"confirmed": True, "account_key": self.key, "ticket": 1,
                   "price": 12.0, "volume": 0.5, "clear_tp": True}
        with tempfile.TemporaryDirectory() as td, patch.object(app, "LOGS_DIR", Path(td)), patch.object(app, "_find_any_position", return_value=pos), patch.object(app, "_runner_alive", return_value={"alive": False}):
            result = app.api_position_partial(payload)
            self.assertTrue(result["pending"])
            data = json.loads((Path(td) / "sr_partial_overrides.json").read_text(encoding="utf-8"))
            self.assertEqual(data["1"]["account_key"], self.key)
            app._write_sl_override(1, 6, True, self.key)
            data = json.loads((Path(td) / "sl_overrides.json").read_text(encoding="utf-8"))
            self.assertEqual(data["1"]["account_key"], self.key)
            with self.assertRaises(app.HTTPException):
                app._atomic_override(Path(td) / "sl_overrides.json", 1, {"sl": 9, "account_key": ["Other", 7, "USD"]})

    def test_equity_anchors_and_sample_coverage(self):
        app._sync_equity_account(None)
        mt5 = SimpleNamespace(history_deals_get=lambda *args: [])
        with patch.dict(sys.modules, {"MetaTrader5": mt5}):
            result = app.api_equity_history(days=7)
        self.assertAlmostEqual(result["history"][-1][0] - result["history"][0][0], 7 * 86400, delta=2)
        self.assertEqual(result["history"][-1][1], 100.0)
        self.assertEqual(result["sample_coverage"]["count"], 1)
        app._sync_equity_account(None)

    def test_disconnected_outputs_have_list_shapes(self):
        with patch.object(app, "_get_mt5_client", return_value=None), patch.object(app, "_runner_alive", return_value={"alive": False, "status": None}):
            self.assertEqual(app.api_status()["runner_status"]["positions"], [])
            self.assertEqual(app.api_mt5_pending_list()["orders"], [])
            self.assertEqual(app.api_equity_history()["points"], [])

    def test_curve_reads_only_training_history_directory(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "training_history_X.json").write_text(json.dumps({"step": [1], "best_score": [9]}), encoding="utf-8")
            hdir = root / "training_history"
            hdir.mkdir()
            (hdir / "training_history_Y.json").write_text(json.dumps({"step": [2], "best_score": [8]}), encoding="utf-8")
            with patch.object(app, "ROOT", root), patch.object(app.training_job, "args", {}):
                result = app.api_training_curve()
            self.assertEqual(result["file"], "training_history_Y.json")


if __name__ == "__main__":
    unittest.main()

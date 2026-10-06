"""Account-switch API tests without a terminal, orders, or application server."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app


class EquityAccountTests(unittest.TestCase):
    def setUp(self):
        app._sync_equity_account(None)
        self.ai = SimpleNamespace(server='BrokerA', login=7, currency='USD', balance=100., equity=101.)
        self.client = SimpleNamespace(account_info=lambda: self.ai, server_time_offset=lambda: 0)
        self.deals = [SimpleNamespace(time=100, profit=10., commission=0., swap=0., fee=0.)]
        self.mt5 = SimpleNamespace(history_deals_get=lambda *args: self.deals)
        self.patches = [patch.object(app, '_get_mt5_client', return_value=self.client),
                        patch.dict(sys.modules, {'MetaTrader5': self.mt5})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        app._sync_equity_account(None)

    def test_switch_server_same_login_invalidates_cache_and_samples(self):
        a = app.api_equity_history()
        self.ai = SimpleNamespace(server='BrokerB', login=7, currency='USD', balance=900., equity=901.)
        self.deals = [SimpleNamespace(time=200, profit=50., commission=0., swap=0., fee=0.)]
        b = app.api_equity_history()
        self.assertEqual(a['history'][-1][1], 100.)
        self.assertEqual(b['history'][-1][1], 900.)
        self.assertEqual(b['points'][0][1], 901.)
        self.assertEqual(len(b['points']), 1)
        self.assertNotEqual(a['account_key'], b['account_key'])

    def test_login_currency_and_balance_refresh(self):
        app.api_equity_history()
        for name, value in [('login', 8), ('currency', 'EUR')]:
            setattr(self.ai, name, value)
            self.assertEqual(app.api_equity_history()['account_key'], list(app._account_identity(self.ai)))
            self.assertEqual(len(app._EQUITY_SAMPLES), 1)
        self.ai.balance = 105.
        self.assertEqual(app.api_equity_history()['history'][-1][1], 105.)

    def test_days_param_limits_history_window(self):
        calls = []
        def record(*args):
            calls.append(args)
            return self.deals
        self.mt5.history_deals_get = record
        app.api_equity_history(days=7)
        self.assertEqual(len(calls), 1)
        frm, to = calls[0]
        self.assertIs(frm.tzinfo, app.timezone.utc)
        self.assertIs(to.tzinfo, app.timezone.utc)
        self.assertGreaterEqual((to - frm).days, 7)
        self.assertLessEqual((to - frm).days, 9)  # +1 day 边沿
        self.assertEqual(app.api_equity_history(days=900)["days"], 365)
        self.assertEqual(app.api_equity_history(days=0)["days"], 90)  # 0=未传，回退默认

    def test_disconnected_clears_previous_account(self):
        app.api_equity_history()
        self.ai = None
        result = app.api_equity_history()
        self.assertEqual(result['points'], [])
        self.assertEqual(result['history'], [])

    def test_history_failure_not_cached_and_account_change_rejected(self):
        self.deals = None
        with self.assertRaises(app.HTTPException):
            app.api_equity_history()
        self.assertEqual(app._BALANCE_HISTORY_CACHE[0], None)
        self.deals = []
        def switch(*args):
            self.ai.login = 88
            return []
        self.mt5.history_deals_get = switch
        with self.assertRaises(app.HTTPException) as caught:
            app.api_equity_history()
        self.assertEqual(caught.exception.status_code, 409)


if __name__ == '__main__':
    unittest.main()

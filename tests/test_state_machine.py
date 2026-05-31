"""Unit tests for state machine (make_rec, config constraints)."""
import pytest
import pandas as pd
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from state_machine import make_rec
from config import load_config


class TestExitCooldown:
    def test_cooldown_config(self):
        cfg = load_config()
        assert cfg['exit_cooldown_days'] == 5


class TestPullbackTimeout:
    def test_timeout_config(self):
        cfg = load_config()
        assert cfg['pullback_max_days'] == 15


class TestNewStockLock:
    def test_new_stock_days(self):
        cfg = load_config()
        assert cfg['new_stock_days'] == 490


class TestBearGate:
    def test_bear_gate_config(self):
        cfg = load_config()
        assert cfg['bear_gate_wma40_roc30'] == -0.10

    def test_bear_gate_blocks_sos(self):
        from indicators import is_bear_market
        cfg = load_config()
        rows = [dict(fwd_close=10, wma40=100 - i * 5) for i in range(10)]
        wdf = pd.DataFrame(rows)
        assert is_bear_market(wdf, 8, cfg) == True


class TestMakeRec:
    def test_basic_record(self):
        row = pd.Series(dict(
            date=pd.Timestamp('2026-05-29'),
            fwd_close=10.5,
            fwd_high=11.0,
            fwd_low=10.0,
            fwd_open=10.2,
            ma10=10.0,
            ma20=9.5,
            ma60=8.5,
            volume=1000,
            vol_ma30=800,
        ))
        rec = make_rec(row, 'TRENDING', 'STRONG', 5, 0, 0, None, 10.5,
                        '', False, False, 'STRONG', False)
        assert rec['state'] == 'TRENDING'
        assert rec['substate'] == 'STRONG'
        assert rec['days_in_state'] == 5

    def test_pullback_dd_pct(self):
        row = pd.Series(dict(
            date=pd.Timestamp('2026-05-29'),
            fwd_close=9.0,
            fwd_high=9.5,
            fwd_low=8.8,
            fwd_open=9.2,
            ma10=10.0,
            ma20=9.5,
            ma60=8.5,
            volume=1000,
            vol_ma30=800,
        ))
        rec = make_rec(row, 'PULLBACK', '', 10, 4, 10, 10.0, None,
                        '', False, False, '', False)
        assert rec['pullback_peak'] == 10.0
        assert rec['pullback_dd_pct'] == -10.0

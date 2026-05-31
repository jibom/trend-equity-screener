"""Unit tests for indicators module."""
import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from indicators import (
    check_align,
    check_tolerance,
    weekly_short_dispersion,
    weekly_wma40_roc30,
    daily_ma10_slope,
    daily_short_align,
    is_bear_market,
    compute_daily_mas,
    compute_weekly_mas,
)


# ── daily_short_align ──

class TestDailyShortAlign:
    def test_aligned(self):
        row = pd.Series(dict(ma10=10, ma20=9, ma50=8))
        assert daily_short_align(row) == True

    def test_not_aligned(self):
        row = pd.Series(dict(ma10=10, ma20=11, ma50=8))
        assert daily_short_align(row) == False

    def test_nan(self):
        row = pd.Series(dict(ma10=10, ma20=np.nan, ma50=8))
        assert daily_short_align(row) is False


# ── check_align ──

class TestCheckAlign:
    def test_aligned(self):
        row = pd.Series(dict(wma40=100, wma50=95, wma60=90, wma70=85))
        assert check_align(row, [40, 50, 60, 70]) is True

    def test_not_aligned(self):
        row = pd.Series(dict(wma40=100, wma50=105, wma60=90, wma70=85))
        assert check_align(row, [40, 50, 60, 70]) is False

    def test_nan(self):
        row = pd.Series(dict(wma40=100, wma50=np.nan, wma60=90, wma70=85))
        assert check_align(row, [40, 50, 60, 70]) is False


# ── weekly_short_dispersion ──

class TestWeeklyShortDispersion:
    def test_tight(self):
        row = pd.Series(dict(wma10=10.1, wma20=10.0, wma30=9.9, wma40=9.8))
        disp = weekly_short_dispersion(row, (10, 20, 30, 40))
        assert disp is not None and disp < 0.08

    def test_wide(self):
        row = pd.Series(dict(wma10=12, wma20=10, wma30=8, wma40=6))
        disp = weekly_short_dispersion(row, (10, 20, 30, 40))
        assert disp is not None and disp > 0.08

    def test_nan_returns_none(self):
        row = pd.Series(dict(wma10=10, wma20=np.nan, wma30=9, wma40=8))
        assert weekly_short_dispersion(row, (10, 20, 30, 40)) is None


# ── weekly_wma40_roc30 ──

class TestWeeklyWma40Roc30:
    def test_rising(self):
        rows = [dict(fwd_close=10 + i, wma40=90 + i) for i in range(10)]
        wdf = pd.DataFrame(rows)
        roc = weekly_wma40_roc30(wdf, 8)
        assert roc is not None and roc > 0

    def test_falling(self):
        rows = [dict(fwd_close=10, wma40=100 - i * 2) for i in range(10)]
        wdf = pd.DataFrame(rows)
        roc = weekly_wma40_roc30(wdf, 8)
        assert roc is not None and roc < 0

    def test_too_short(self):
        wdf = pd.DataFrame([dict(fwd_close=10, wma40=90)] * 4)
        assert weekly_wma40_roc30(wdf, 3) is None


# ── daily_ma10_slope ──

class TestDailyMa10Slope10d:
    def test_rising(self):
        df = pd.DataFrame(dict(ma10=[10 + i * 0.5 for i in range(20)]))
        slope = daily_ma10_slope(df, 15)
        assert slope is not None and slope > 0

    def test_too_short(self):
        df = pd.DataFrame(dict(ma10=[10, 11]))
        assert daily_ma10_slope(df, 1) is None


# ── is_bear_market ──

class TestIsBearMarket:
    @pytest.fixture
    def cfg(self):
        from config import load_config
        return load_config()

    def test_bear(self, cfg):
        rows = [dict(fwd_close=10, wma40=100 - i * 5) for i in range(10)]
        wdf = pd.DataFrame(rows)
        assert is_bear_market(wdf, 8, cfg) == True

    def test_not_bear(self, cfg):
        rows = [dict(fwd_close=10, wma40=90 + i * 0.5) for i in range(10)]
        wdf = pd.DataFrame(rows)
        assert is_bear_market(wdf, 8, cfg) == False

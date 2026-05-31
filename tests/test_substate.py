"""Unit tests for substate determination."""
import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from substate import trending_substate
from config import load_config


@pytest.fixture
def cfg():
    return load_config()


class TestTrendingSubstate:
    def test_new_stock_returns_new(self, cfg):
        """New stock (<490 days) with aligned MAs → NEW."""
        daily_df = pd.DataFrame(dict(
            ma10=[10], ma20=[9], ma50=[8], fwd_close=[10.5], ma60=[7]
        ))
        weekly_df = pd.DataFrame([dict(fwd_close=10, wma40=9)])
        result = trending_substate(daily_df, 0, weekly_df, 0, 100, cfg)
        assert result == 'NEW'

    def test_new_stock_not_aligned_returns_empty(self, cfg):
        """New stock but MAs not aligned → ''."""
        daily_df = pd.DataFrame(dict(
            ma10=[9], ma20=[10], ma50=[8], fwd_close=[10.5], ma60=[7]
        ))
        weekly_df = pd.DataFrame([dict(fwd_close=10, wma40=9)])
        result = trending_substate(daily_df, 0, weekly_df, 0, 100, cfg)
        assert result == ''

    def test_strong_overrides_mid(self, cfg):
        """Long + short aligned + tolerance → STRONG."""
        daily_df = pd.DataFrame(dict(
            ma10=[10], ma20=[9], ma50=[8], fwd_close=[10.5], ma60=[7]
        ))
        weekly_rows = [
            dict(fwd_close=10 + i, wma10=5 + i, wma20=4.5 + i,
                 wma30=4 + i, wma40=3.5 + i, wma50=3 + i, wma60=2.5 + i, wma70=2 + i)
            for i in range(5)
        ]
        weekly_df = pd.DataFrame(weekly_rows)
        result = trending_substate(daily_df, 0, weekly_df, 4, 500, cfg)
        assert result == 'STRONG'

    def test_invalid_week_idx_returns_empty(self, cfg):
        """Invalid week_idx → ''."""
        daily_df = pd.DataFrame(dict(
            ma10=[10], ma20=[9], ma50=[8], fwd_close=[10.5], ma60=[7]
        ))
        weekly_df = pd.DataFrame([dict(fwd_close=10)])
        result = trending_substate(daily_df, 0, weekly_df, -1, 500, cfg)
        assert result == ''

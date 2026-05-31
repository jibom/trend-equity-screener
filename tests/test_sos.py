"""Unit tests for SOS classification (v5.3)."""
import pytest
import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from sos import classify_sos
from config import load_config


@pytest.fixture
def cfg():
    return load_config()


class TestClassifySos:
    def test_sos_a_new_high_vol_body(self, cfg):
        """New high + volume + big green body → SOS-A."""
        row = pd.Series(dict(
            volume=2000, vol_ma30=1000,
            fwd_close=11.0, fwd_open=10.0, fwd_high=11.2, fwd_low=9.7
        ))
        assert classify_sos(row, cfg, is_new_high=True) == 'SOS-A'

    def test_sos_b_vol_body_no_new_high(self, cfg):
        """Volume + big green body (no new high) → SOS-B."""
        row = pd.Series(dict(
            volume=2000, vol_ma30=1000,
            fwd_close=11.0, fwd_open=10.0, fwd_high=11.2, fwd_low=9.7
        ))
        assert classify_sos(row, cfg, is_new_high=False) == 'SOS-B'

    def test_sos_c_new_high_weak(self, cfg):
        """New high but no volume or big body → SOS-C."""
        row = pd.Series(dict(
            volume=800, vol_ma30=1000,
            fwd_close=11.0, fwd_open=10.5, fwd_high=11.5, fwd_low=10.0
        ))
        assert classify_sos(row, cfg, is_new_high=True) == 'SOS-C'

    def test_no_signal(self, cfg):
        """No new high, no volume, no big body → ''."""
        row = pd.Series(dict(
            volume=800, vol_ma30=1000,
            fwd_close=10.1, fwd_open=10.0, fwd_high=10.5, fwd_low=9.5
        ))
        assert classify_sos(row, cfg, is_new_high=False) == ''

    def test_no_vol_ma(self, cfg):
        """Missing vol_ma30 → ''."""
        row = pd.Series(dict(
            volume=2000, vol_ma30=np.nan,
            fwd_close=11.0, fwd_open=10.0, fwd_high=11.2, fwd_low=9.7
        ))
        assert classify_sos(row, cfg, is_new_high=True) == ''

    def test_zero_range(self, cfg):
        """Zero price range → ''."""
        row = pd.Series(dict(
            volume=2000, vol_ma30=1000,
            fwd_close=10.0, fwd_open=10.0, fwd_high=10.0, fwd_low=10.0
        ))
        assert classify_sos(row, cfg, is_new_high=True) == ''

    def test_sos_a_overrides_b(self, cfg):
        """Same conditions with new high → SOS-A, without → SOS-B."""
        row = pd.Series(dict(
            volume=2000, vol_ma30=1000,
            fwd_close=11.0, fwd_open=10.0, fwd_high=11.2, fwd_low=9.7
        ))
        assert classify_sos(row, cfg, is_new_high=True) == 'SOS-A'
        assert classify_sos(row, cfg, is_new_high=False) == 'SOS-B'

"""SOS (Sign of Strength) classification for Wyckoff v5.3.

v5.3 SOS redefinition (aligned with Wyckoff 4 buy points):
  SOS-A: new_high + volume_up + big_green_body  → Type 4 JOC (Jump Over Creek)
  SOS-B: volume_up + big_green_body (no new high) → Type 1/2 Spring / Test
  SOS-C: new_high + weak volume                  → 弱势新高

close_loc threshold lowered: 0.70 → 0.60
"""
from __future__ import annotations

import pandas as pd


def classify_sos(row: pd.Series, cfg: dict, is_new_high: bool = False) -> str:
    """Classify a single day's SOS signal (v5.3).

    Args:
        row: Daily data row with volume, vol_ma30, fwd_close/open/high/low.
        cfg: Config dict.
        is_new_high: Whether today is a 60-day new high.

    Returns 'SOS-A', 'SOS-B', 'SOS-C', or ''.
    """
    if pd.isna(row.get('vol_ma30')) or row['vol_ma30'] == 0:
        return ''

    vol_up = row['volume'] >= cfg['sos_volume_multiple'] * row['vol_ma30']
    rng = row['fwd_high'] - row['fwd_low']
    if rng == 0:
        return ''

    body = abs(row['fwd_close'] - row['fwd_open']) / rng
    close_loc = (row['fwd_close'] - row['fwd_low']) / rng
    big_green = (body >= cfg['sos_big_body_pct']
                 and close_loc >= cfg['sos_close_loc']
                 and row['fwd_close'] > row['fwd_open'])

    vol_and_body = vol_up and big_green

    if vol_and_body and is_new_high:
        return 'SOS-A'
    if vol_and_body and not is_new_high:
        return 'SOS-B'
    if is_new_high and not vol_and_body:
        return 'SOS-C'
    return ''

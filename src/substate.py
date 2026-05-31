"""TRENDING substate determination for Wyckoff v5.2.

Substates: STRONG / MID / EARLY / NEW / ''

Priority: NEW (overrides all for new stocks) > STRONG > MID > EARLY
"""
from __future__ import annotations

import pandas as pd

from indicators import (
    check_align,
    check_tolerance,
    weekly_short_dispersion,
    weekly_wma40_roc30,
    daily_short_align,
    daily_ma10_slope,
    no_weekly_cross_down,
)


def trending_substate(daily_df: pd.DataFrame, i: int,
                      weekly_df: pd.DataFrame, week_idx: int,
                      days_listed: int, cfg: dict) -> str:
    """Determine TRENDING substate for day *i*.

    Returns one of: 'STRONG', 'MID', 'EARLY', 'NEW', ''
    """
    row = daily_df.iloc[i]

    # New stock (<490 days): always NEW if aligned, else ''
    if days_listed < cfg['new_stock_days']:
        mas = cfg['new_stock_mas']
        vals = [row.get(f'ma{m}') for m in mas]
        if all(pd.notna(v) for v in vals):
            aligned = all(vals[j] > vals[j + 1] for j in range(len(vals) - 1))
            if aligned and row['fwd_close'] > row.get('ma20', 0):
                return 'NEW'
        return ''

    if week_idx < 0 or week_idx >= len(weekly_df):
        return ''

    weekly_row = weekly_df.iloc[week_idx]

    long_align = check_align(weekly_row, cfg['weekly_long_mas'])
    short_align = check_align(weekly_row, cfg['weekly_short_mas'])
    tolerance_ok = check_tolerance(weekly_df, week_idx, cfg)

    # STRONG: long aligned + short aligned + tolerance
    if long_align and short_align and tolerance_ok:
        return 'STRONG'

    # MID: long aligned but short not aligned
    # + slow MAs trending up (WMA40 RoC > 0)
    # + no fast-slow cross-down (e.g. 5W > 10W > 20W)
    if long_align and not short_align:
        wma40_roc = weekly_wma40_roc30(weekly_df, week_idx)
        slow_trending = wma40_roc is not None and wma40_roc > 0
        fast_ma = cfg.get('weekly_fast_ma', 5)
        no_cross = (no_weekly_cross_down(weekly_row, fast_ma, 10) and
                    no_weekly_cross_down(weekly_row, 10, 20))
        if slow_trending and no_cross:
            return 'MID'

    # EARLY: daily short aligned + MA10 slope + weekly entangled + not falling
    daily_ok = daily_short_align(row)
    slope = daily_ma10_slope(daily_df, i)
    slope_ok = slope is not None and slope > cfg['early_ma10_slope10d']
    disp = weekly_short_dispersion(weekly_row, cfg['weekly_short_mas'])
    disp_ok = disp is not None and disp <= cfg['early_entangle_disp']
    roc = weekly_wma40_roc30(weekly_df, week_idx)
    not_falling = roc is not None and roc >= cfg['early_wma40_roc30']
    if daily_ok and slope_ok and disp_ok and not_falling:
        return 'EARLY'

    return ''

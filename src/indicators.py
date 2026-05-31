"""Technical indicator calculations for Wyckoff v5.2.

All functions are pure: they take DataFrames/Series and return values.
No data fetching or state mutation.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


# ── Daily indicators ──────────────────────────────────────────────

def compute_daily_mas(df: pd.DataFrame, windows=(5, 10, 20, 30, 50, 60, 100, 200)) -> pd.DataFrame:
    """Compute daily moving averages on fwd_close and vol_ma30."""
    df = df.copy()
    for w in windows:
        df[f'ma{w}'] = df['fwd_close'].rolling(w).mean()
    df['vol_ma30'] = df['volume'].rolling(30).mean()
    return df


def compute_ma60_slope(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """Compute MA60 slope as RoC over *lookback* days."""
    df = df.copy()
    if 'ma60' not in df.columns:
        df['ma60'] = df['fwd_close'].rolling(60).mean()
    df['ma60_slope5'] = df['ma60'] / df['ma60'].shift(lookback) - 1
    return df


def daily_short_align(row: pd.Series) -> bool:
    """Daily MA10 > MA20 > MA50."""
    vals = [row.get('ma10'), row.get('ma20'), row.get('ma50')]
    if any(pd.isna(v) for v in vals):
        return False
    return vals[0] > vals[1] > vals[2]


def daily_ma10_slope(df: pd.DataFrame, i: int, lookback: int = 10) -> float | None:
    """Daily MA10 10-day slope (RoC)."""
    if i < lookback:
        return None
    cur = df.iloc[i].get('ma10')
    past = df.iloc[i - lookback].get('ma10')
    if pd.isna(cur) or pd.isna(past) or past == 0:
        return None
    return cur / past - 1


# ── Weekly indicators ─────────────────────────────────────────────

def compute_weekly_mas(daily_df: pd.DataFrame,
                       mas=(10, 20, 30, 40, 50, 60, 70)) -> pd.DataFrame:
    """Resample daily to weekly (Fri) and compute MAs on fwd_close."""
    w = daily_df.set_index('date')[['fwd_close']].resample('W-FRI').last().dropna()
    for m in mas:
        w[f'wma{m}'] = w['fwd_close'].rolling(m).mean()
    return w


def check_align(row: pd.Series, mas: list[int]) -> bool:
    """Check if weekly MAs are in strict descending order (bullish alignment)."""
    vals = [row.get(f'wma{m}') for m in mas]
    if any(pd.isna(v) for v in vals):
        return False
    return all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def check_tolerance(weekly_df: pd.DataFrame, week_idx: int, cfg: dict) -> bool:
    """Tolerance check: within the last N weeks, close < wma20 at most K times."""
    if week_idx < 0:
        return False
    window = cfg['tolerance_window_weeks']
    start = max(0, week_idx - window + 1)
    recent = weekly_df.iloc[start:week_idx + 1]
    if 'wma20' not in recent.columns or recent['wma20'].isna().any():
        return False
    breaks = (recent['fwd_close'] < recent['wma20']).sum()
    return breaks <= cfg['tolerance_max_breaks']


def weekly_short_dispersion(row: pd.Series, mas=(10, 20, 30, 40)) -> float | None:
    """Weekly short-MA dispersion: (max - min) / median."""
    vals = [row.get(f'wma{m}') for m in mas]
    if any(pd.isna(v) for v in vals):
        return None
    vals = sorted(vals)
    median = vals[len(vals) // 2]
    if median == 0:
        return None
    return (max(vals) - min(vals)) / median


def is_entangled(weekly_df: pd.DataFrame, week_idx: int,
                 mas=(10, 20, 30, 40), max_disp: float = 0.08) -> bool:
    """Check if weekly short MAs are entangled (dispersion ≤ threshold)."""
    if week_idx < 0 or week_idx >= len(weekly_df):
        return False
    disp = weekly_short_dispersion(weekly_df.iloc[week_idx], mas)
    if disp is None:
        return False
    return disp <= max_disp


def weekly_wma40_roc30(weekly_df: pd.DataFrame, week_idx: int,
                       lookback_weeks: int = 6) -> float | None:
    """Weekly WMA40 RoC over ~30 trading days (6 weeks)."""
    if week_idx < lookback_weeks:
        return None
    cur = weekly_df.iloc[week_idx].get('wma40')
    past = weekly_df.iloc[week_idx - lookback_weeks].get('wma40')
    if pd.isna(cur) or pd.isna(past) or past == 0:
        return None
    return cur / past - 1


def is_bear_market(weekly_df: pd.DataFrame, week_idx: int, cfg: dict) -> bool:
    """Bear gate: WMA40 30-day RoC < threshold (default -10%)."""
    roc = weekly_wma40_roc30(weekly_df, week_idx)
    if roc is None:
        return False
    return roc < cfg['bear_gate_wma40_roc30']


def no_weekly_cross_down(weekly_row: pd.Series, fast_ma: int, slow_ma: int) -> bool:
    """Check that fast weekly MA is above slow weekly MA (no cross-down).

    Returns True if no cross-down (bullish) or if values are unavailable.
    """
    fast_val = weekly_row.get(f'wma{fast_ma}')
    slow_val = weekly_row.get(f'wma{slow_ma}')
    if pd.isna(fast_val) or pd.isna(slow_val):
        return True
    return fast_val > slow_val

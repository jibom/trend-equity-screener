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


# ── KDJ ────────────────────────────────────────────────────────────

def compute_kdj(df: pd.DataFrame, n: int = 9,
                high_col: str = 'fwd_high',
                low_col: str = 'fwd_low',
                close_col: str = 'fwd_close') -> pd.DataFrame:
    """Compute KDJ indicator.

    RSV(n) = (Close - Low_n) / (High_n - Low_n) * 100
    K = 2/3 * K_prev + 1/3 * RSV   (initial K = 50)
    D = 2/3 * D_prev + 1/3 * K     (initial D = 50)
    J = 3*K - 2*D

    Adds columns: k, d, j, kd_golden_cross, kd_death_cross
    """
    df = df.copy()
    low_n = df[low_col].rolling(n, min_periods=1).min()
    high_n = df[high_col].rolling(n, min_periods=1).max()
    denom = high_n - low_n
    rsv = pd.Series(50.0, index=df.index, dtype=float)
    valid = denom > 0
    rsv[valid] = ((df.loc[valid, close_col] - low_n[valid]) / denom[valid] * 100)

    k = pd.Series(50.0, index=df.index, dtype=float)
    d = pd.Series(50.0, index=df.index, dtype=float)
    for i in range(1, len(df)):
        k.iloc[i] = 2 / 3 * k.iloc[i - 1] + 1 / 3 * rsv.iloc[i]
        d.iloc[i] = 2 / 3 * d.iloc[i - 1] + 1 / 3 * k.iloc[i]

    df['k'] = k.round(2)
    df['d'] = d.round(2)
    df['j'] = (3 * k - 2 * d).round(2)

    # Cross detection
    df['kd_golden_cross'] = (k.shift(1) <= d.shift(1)) & (k > d)
    df['kd_death_cross'] = (k.shift(1) >= d.shift(1)) & (k < d)

    return df


def compute_weekly_kdj(daily_df: pd.DataFrame, n: int = 9) -> pd.DataFrame:
    """Resample daily to weekly (Fri) and compute KDJ."""
    w = daily_df.set_index('date')[['fwd_open', 'fwd_high', 'fwd_low', 'fwd_close', 'volume']].resample('W-FRI').agg({
        'fwd_open': 'first',
        'fwd_high': 'max',
        'fwd_low': 'min',
        'fwd_close': 'last',
        'volume': 'sum',
    }).dropna()
    w = compute_kdj(w, n=n)
    return w


def detect_kdj_divergence(df: pd.DataFrame, lookback: int = 15,
                          close_col: str = 'fwd_close') -> dict:
    """Detect bullish KDJ divergence (底背离) within lookback window.

    Precondition: J must have dropped below 0 at some point in the window.
    Only then do we check for divergence: price makes lower low,
    but J makes higher low (J diverging upward from oversold).

    Counts how many divergence events occurred in the window.

    Returns dict:
      bullish_divergence: bool
      divergence_count: int  (number of divergence events)
      detail: str
      j_trending_up: bool  (last J > J 3 bars ago)
      j_below_zero: bool   (J dropped below 0 in the window)
    """
    if len(df) < lookback or 'j' not in df.columns:
        return dict(bullish_divergence=False, divergence_count=0,
                    detail='insufficient data',
                    j_trending_up=False, j_below_zero=False)

    window = df.tail(lookback).reset_index(drop=True)
    closes = window[close_col].values
    js = window['j'].values

    j_below_zero = bool(np.any(js < 0))
    j_trending_up = js[-1] > js[-3] if len(js) >= 3 else False

    # J must have been below 0 in the window — otherwise no oversold condition
    if not j_below_zero:
        return dict(bullish_divergence=False, divergence_count=0,
                    detail='J never below 0',
                    j_trending_up=j_trending_up, j_below_zero=False)

    # Find all local price lows
    lows = []  # (window_index, close, j)
    for i in range(1, len(closes) - 1):
        if closes[i] <= closes[i - 1] and closes[i] <= closes[i + 1]:
            lows.append((i, closes[i], js[i]))
    if closes[0] <= closes[1]:
        lows.insert(0, (0, closes[0], js[0]))
    if closes[-1] <= closes[-2]:
        lows.append((len(closes) - 1, closes[-1], js[-1]))

    # Count divergence events between consecutive low pairs
    div_count = 0
    last_detail = ''
    for k in range(len(lows) - 1):
        _, p1, j1 = lows[k]
        _, p2, j2 = lows[k + 1]
        if p2 < p1 and j2 > j1:
            div_count += 1
            last_detail = (f'price {p1:.1f}->{p2:.1f} (lower low), '
                           f'J {j1:.1f}->{j2:.1f} (higher low)')

    # Also check: current bar at price low but J not at J low
    min_close_idx = int(np.argmin(closes))
    min_j_idx = int(np.argmin(js))
    if min_close_idx == len(closes) - 1 and min_j_idx != len(closes) - 1:
        div_count += 1
        min_j = js[min_j_idx]
        last_detail = (f'current price at window low {closes[-1]:.1f}, '
                       f'but J {js[-1]:.1f} > window J low {min_j:.1f}')

    if div_count > 0 and not last_detail and div_count > 0:
        last_detail = f'{div_count} divergence(s)'

    return dict(bullish_divergence=div_count > 0, divergence_count=div_count,
                detail=last_detail if last_detail else 'no divergence',
                j_trending_up=j_trending_up, j_below_zero=True)


# ── MACD ──────────────────────────────────────────────────────────

def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26,
                 signal: int = 9,
                 close_col: str = 'fwd_close') -> pd.DataFrame:
    """Compute MACD indicator.

    DIF  = EMA(fast) - EMA(slow)
    DEA  = EMA(signal) of DIF
    MACD = 2 * (DIF - DEA)

    Adds columns: dif, dea, macd, macd_golden_cross, macd_death_cross
    """
    df = df.copy()
    ema_fast = df[close_col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[close_col].ewm(span=slow, adjust=False).mean()
    df['dif'] = (ema_fast - ema_slow).round(4)
    df['dea'] = df['dif'].ewm(span=signal, adjust=False).mean().round(4)
    df['macd'] = (2 * (df['dif'] - df['dea'])).round(4)

    df['macd_golden_cross'] = (df['dif'].shift(1) <= df['dea'].shift(1)) & (df['dif'] > df['dea'])
    df['macd_death_cross'] = (df['dif'].shift(1) >= df['dea'].shift(1)) & (df['dif'] < df['dea'])

    return df


def compute_weekly_macd(daily_df: pd.DataFrame, fast: int = 12,
                        slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Resample daily to weekly (Fri) and compute MACD."""
    w = daily_df.set_index('date')[['fwd_open', 'fwd_high', 'fwd_low', 'fwd_close', 'volume']].resample('W-FRI').agg({
        'fwd_open': 'first',
        'fwd_high': 'max',
        'fwd_low': 'min',
        'fwd_close': 'last',
        'volume': 'sum',
    }).dropna()
    return compute_macd(w, fast=fast, slow=slow, signal=signal)


def detect_macd_golden_cross(df: pd.DataFrame, lookback: int = 10) -> dict:
    """Check if MACD golden cross occurred within lookback window.

    Also checks: DIF is below 0 (MACD below zero line = bearish context,
    golden cross below zero = early reversal signal).

    Returns dict:
      golden_cross: bool   (any golden cross in the window)
      recent_golden: bool   (golden cross in last 3 bars)
      dif_below_zero: bool  (DIF < 0 at golden cross = stronger signal)
      detail: str
    """
    if len(df) < lookback or 'dif' not in df.columns:
        return dict(golden_cross=False, recent_golden=False,
                    dif_below_zero=False, detail='insufficient data')

    window = df.tail(lookback)
    gc = window['macd_golden_cross']
    has_gc = bool(gc.any())

    if not has_gc:
        return dict(golden_cross=False, recent_golden=False,
                    dif_below_zero=False, detail='no golden cross')

    # Find the most recent golden cross
    gc_bars = window[gc]
    last_gc = gc_bars.iloc[-1]
    dif_at_gc = last_gc['dif']
    dif_below_zero = dif_at_gc < 0

    # Is it in the last 3 bars?
    recent = gc.iloc[-3:].any() if len(gc) >= 3 else gc.any()

    context = 'below zero' if dif_below_zero else 'above zero'
    detail = f'DIF={dif_at_gc:.4f} ({context})'

    return dict(golden_cross=True, recent_golden=bool(recent),
                dif_below_zero=dif_below_zero, detail=detail)


def no_weekly_cross_down(weekly_row: pd.Series, fast_ma: int, slow_ma: int) -> bool:
    """Check that fast weekly MA is above slow weekly MA (no cross-down).

    Returns True if no cross-down (bullish) or if values are unavailable.
    """
    fast_val = weekly_row.get(f'wma{fast_ma}')
    slow_val = weekly_row.get(f'wma{slow_ma}')
    if pd.isna(fast_val) or pd.isna(slow_val):
        return True
    return fast_val > slow_val

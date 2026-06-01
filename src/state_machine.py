"""Core state machine and backtest runner for Wyckoff v5.4.

State machine (6 main states):
  POOL ⇄ ENTANGLED ⇄ SETUP_OK → TRENDING ⇄ PULLBACK
                                    ↓          ↓
                                  EXIT ←───────┘
                                    ↓
                              (cooldown 5d) → POOL/ENTANGLED/SETUP_OK/TRENDING
"""
from __future__ import annotations

import os
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import load_config
from data_provider import WindFetcher, forward_adjust
from indicators import (compute_daily_mas, compute_ma60_slope, compute_weekly_mas,
                        is_bear_market, is_entangled)
from substate import trending_substate
from sos import classify_sos


def make_rec(row, state, substate, days_in_state, consec_below_ma10,
             days_in_pullback, pullback_peak, base_max, sos, new_high,
             is_new_stock, sub_raw, bear_gate, idle_days=0):
    """Build one output record from a day's state."""
    return dict(
        date=row['date'].strftime('%Y-%m-%d'),
        state=state,
        substate=substate,
        sub_raw=sub_raw,
        days_in_state=days_in_state,
        fwd_close=round(float(row['fwd_close']), 3),
        ma10=round(float(row['ma10']), 3) if pd.notna(row.get('ma10')) else None,
        ma20=round(float(row['ma20']), 3) if pd.notna(row.get('ma20')) else None,
        ma60=round(float(row['ma60']), 3) if pd.notna(row.get('ma60')) else None,
        consec_below_ma10=consec_below_ma10,
        days_in_pullback=days_in_pullback,
        pullback_peak=round(pullback_peak, 3) if pullback_peak else None,
        base_max=round(base_max, 3) if base_max else None,
        pullback_dd_pct=round((row['fwd_close'] / pullback_peak - 1) * 100, 2) if pullback_peak else None,
        new_high=new_high,
        sos=sos,
        is_new_stock=is_new_stock,
        bear_gate=bear_gate,
        idle_days=idle_days,
    )


def run_state_machine(df, weekly_df, weekly_dates, cfg, listing_first_day):
    """Core state machine loop with v5.4 event tracking.

    Args:
        df: Daily DataFrame with pre-computed sos_raw and is_new_high columns.
        weekly_df: Weekly indicator DataFrame.
        weekly_dates: List of weekly period end dates.
        cfg: Configuration dict.
        listing_first_day: First listing date of the stock.

    Returns (records, recent_events, recent_new_highs):
      records: list of daily state record dicts
      recent_events: list of (day_index, event_type, detail, state_at_time)
      recent_new_highs: list of day_index values
    """
    entangled_disp = cfg.get('entangled_disp', 0.08)
    setup_idle_days = cfg.get('setup_idle_days', 5)

    state = 'POOL'
    substate = ''
    days_in_state = 0
    consec_below_ma10 = 0
    pullback_peak = None
    days_in_pullback = 0
    base_max = None
    days_in_exit = 0
    setup_idle = 0
    recs = []

    # v5.4 event tracking
    recent_events = []
    recent_new_highs = []

    for i, row in df.iterrows():
        days_in_state += 1
        days_listed_real = (row['date'] - listing_first_day).days
        is_new_stock = days_listed_real < cfg['new_stock_days']

        date_period_end = pd.Timestamp(row['date']).to_period('W-FRI').end_time.normalize()
        try:
            week_idx = weekly_dates.index(date_period_end)
        except ValueError:
            week_idx = -1

        sub_today = trending_substate(df, i, weekly_df, week_idx, days_listed_real, cfg)

        sos_today = df.at[i, 'sos_raw']
        new_high_today = bool(df.at[i, 'is_new_high'])
        bear = is_bear_market(weekly_df, week_idx, cfg)
        entangled = is_entangled(weekly_df, week_idx,
                                 mas=tuple(cfg['weekly_short_mas']),
                                 max_disp=entangled_disp)

        # v5.4: track SOS from SETUP_OK/ENTANGLED and new highs for TRENDING
        if sos_today and state in ('SETUP_OK', 'ENTANGLED'):
            recent_events.append((i, 'sos_from_setup', sos_today, state))
        if state == 'TRENDING' and new_high_today:
            recent_new_highs.append(i)
        sos_caused_setup = (sos_today in ('SOS-B', 'SOS-C') and not bear
                            and state in ('POOL', 'ENTANGLED', 'EXIT'))

        broke_ma60 = (
            pd.notna(row.get('ma60')) and row['fwd_close'] < row['ma60']
            and pd.notna(row.get('ma60_slope5')) and row['ma60_slope5'] < 0
        )

        # ── EXIT trigger ──
        if state in ('TRENDING', 'PULLBACK', 'SETUP_OK', 'ENTANGLED') and broke_ma60:
            prev_state = state
            state = 'EXIT'; substate = ''
            days_in_state = 1; days_in_exit = 1
            consec_below_ma10 = 0; pullback_peak = None; base_max = None
            setup_idle = 0
            recent_events.append((i, 'to_EXIT', prev_state, prev_state))
            recs.append(make_rec(row, state, substate, days_in_state,
                                 consec_below_ma10, days_in_pullback,
                                 pullback_peak, base_max,
                                 sos_today, new_high_today, is_new_stock,
                                 sub_today, bear, setup_idle))
            continue

        # ── EXIT ──
        if state == 'EXIT':
            days_in_exit += 1
            if days_in_exit < cfg['exit_cooldown_days']:
                recs.append(make_rec(row, state, substate, days_in_state,
                                     consec_below_ma10, days_in_pullback,
                                     pullback_peak, base_max,
                                     sos_today, new_high_today, is_new_stock,
                                     sub_today, bear, setup_idle))
                continue
            can_trending = bool(sub_today) or (sos_today == 'SOS-A' and not bear)
            can_setup = sos_today in ('SOS-B', 'SOS-C') and not bear
            if can_trending:
                state = 'TRENDING'
                substate = 'NEW' if is_new_stock else sub_today
                days_in_state = 1; days_in_exit = 0
                base_max = row['fwd_close']; consec_below_ma10 = 0; setup_idle = 0
            elif can_setup:
                state = 'SETUP_OK'; substate = ''
                days_in_state = 1; days_in_exit = 0; setup_idle = 0
                if sos_caused_setup:
                    recent_events.append((i, 'sos_from_setup', sos_today, 'SETUP_OK'))
            elif entangled:
                state = 'ENTANGLED'; substate = ''
                days_in_state = 1; days_in_exit = 0; setup_idle = 0
            else:
                state = 'POOL'; substate = ''
                days_in_state = 1; days_in_exit = 0; setup_idle = 0
            recs.append(make_rec(row, state, substate, days_in_state,
                                 consec_below_ma10, days_in_pullback,
                                 pullback_peak, base_max,
                                 sos_today, new_high_today, is_new_stock,
                                 sub_today, bear, setup_idle))
            continue

        # ── POOL ──
        if state == 'POOL':
            can_trending = bool(sub_today) or (sos_today == 'SOS-A' and not bear)
            can_setup = sos_today in ('SOS-B', 'SOS-C') and not bear
            if can_trending:
                state = 'TRENDING'
                substate = 'NEW' if is_new_stock else sub_today
                days_in_state = 1
                base_max = row['fwd_close']; consec_below_ma10 = 0; setup_idle = 0
            elif can_setup:
                state = 'SETUP_OK'; substate = ''
                days_in_state = 1; setup_idle = 0
                if sos_caused_setup:
                    recent_events.append((i, 'sos_from_setup', sos_today, 'SETUP_OK'))
            elif entangled:
                state = 'ENTANGLED'; substate = ''
                days_in_state = 1; setup_idle = 0
            recs.append(make_rec(row, state, substate, days_in_state,
                                 consec_below_ma10, days_in_pullback,
                                 pullback_peak, base_max,
                                 sos_today, new_high_today, is_new_stock,
                                 sub_today, bear, setup_idle))
            continue

        # ── ENTANGLED ──
        if state == 'ENTANGLED':
            can_trending = bool(sub_today) or (sos_today == 'SOS-A' and not bear)
            can_setup = sos_today in ('SOS-B', 'SOS-C') and not bear
            if can_trending:
                state = 'TRENDING'
                substate = 'NEW' if is_new_stock else sub_today
                days_in_state = 1
                base_max = row['fwd_close']; consec_below_ma10 = 0; setup_idle = 0
            elif can_setup:
                state = 'SETUP_OK'; substate = ''
                days_in_state = 1; setup_idle = 0
                if sos_caused_setup:
                    recent_events.append((i, 'sos_from_setup', sos_today, 'SETUP_OK'))
            elif not entangled:
                state = 'POOL'; substate = ''
                days_in_state = 1; setup_idle = 0
            recs.append(make_rec(row, state, substate, days_in_state,
                                 consec_below_ma10, days_in_pullback,
                                 pullback_peak, base_max,
                                 sos_today, new_high_today, is_new_stock,
                                 sub_today, bear, setup_idle))
            continue

        # ── SETUP_OK ──
        if state == 'SETUP_OK':
            can_trending = bool(sub_today) or (sos_today == 'SOS-A' and not bear)
            if can_trending:
                state = 'TRENDING'
                substate = 'NEW' if is_new_stock else sub_today
                days_in_state = 1
                base_max = row['fwd_close']; consec_below_ma10 = 0; setup_idle = 0
                recs.append(make_rec(row, state, substate, days_in_state,
                                     consec_below_ma10, days_in_pullback,
                                     pullback_peak, base_max,
                                     sos_today, new_high_today, is_new_stock,
                                     sub_today, bear, setup_idle))
                continue

            if sos_today in ('SOS-A', 'SOS-B', 'SOS-C'):
                setup_idle = 0
            else:
                setup_idle += 1

            if setup_idle >= setup_idle_days:
                if entangled:
                    state = 'ENTANGLED'; substate = ''
                else:
                    state = 'POOL'; substate = ''
                days_in_state = 1; setup_idle = 0

            recs.append(make_rec(row, state, substate, days_in_state,
                                 consec_below_ma10, days_in_pullback,
                                 pullback_peak, base_max,
                                 sos_today, new_high_today, is_new_stock,
                                 sub_today, bear, setup_idle))
            continue

        # ── TRENDING ──
        if state == 'TRENDING':
            if is_new_stock:
                substate = 'NEW'
            else:
                substate = sub_today

            if pd.notna(row.get('ma10')) and row['fwd_close'] < row['ma10']:
                consec_below_ma10 += 1
            else:
                consec_below_ma10 = 0

            if consec_below_ma10 >= cfg['exit_consec_below_ma10']:
                state = 'PULLBACK'
                pullback_peak = max(df['fwd_close'].iloc[max(0, i - 60):i + 1])
                days_in_pullback = 0; days_in_state = 1
                recent_events.append((i, 'TRENDING_to_PULLBACK', None, 'TRENDING'))
                recs.append(make_rec(row, state, substate, days_in_state,
                                     consec_below_ma10, days_in_pullback,
                                     pullback_peak, base_max,
                                     sos_today, new_high_today, is_new_stock,
                                     sub_today, bear, setup_idle))
                continue

            if base_max is None or row['fwd_close'] > base_max:
                base_max = row['fwd_close']

            recs.append(make_rec(row, state, substate, days_in_state,
                                 consec_below_ma10, days_in_pullback,
                                 pullback_peak, base_max,
                                 sos_today, new_high_today, is_new_stock,
                                 sub_today, bear, setup_idle))
            continue

        # ── PULLBACK ──
        if state == 'PULLBACK':
            days_in_pullback += 1
            if days_in_pullback >= cfg['pullback_max_days']:
                state = 'EXIT'; substate = ''
                days_in_state = 1; days_in_exit = 1
                consec_below_ma10 = 0; pullback_peak = None; base_max = None; setup_idle = 0
                recent_events.append((i, 'to_EXIT', 'PULLBACK', 'PULLBACK'))
                recs.append(make_rec(row, state, substate, days_in_state,
                                     consec_below_ma10, days_in_pullback,
                                     pullback_peak, base_max,
                                     sos_today, new_high_today, is_new_stock,
                                     sub_today, bear, setup_idle))
                continue
            if pd.notna(row.get('ma10')) and row['fwd_close'] >= row['ma10']:
                state = 'TRENDING'
                if is_new_stock:
                    substate = 'NEW'
                else:
                    substate = sub_today
                days_in_state = 1; consec_below_ma10 = 0
                base_max = row['fwd_close']
                days_in_pullback = 0; pullback_peak = None; setup_idle = 0
            recs.append(make_rec(row, state, substate, days_in_state,
                                 consec_below_ma10, days_in_pullback,
                                 pullback_peak, base_max,
                                 sos_today, new_high_today, is_new_stock,
                                 sub_today, bear, setup_idle))

    return recs, recent_events, recent_new_highs


def run_one(code, asof='2026-05-29', months=12, label=None, cfg=None):
    """Run backtest for a single stock (v5.4 state machine)."""
    if cfg is None:
        cfg = load_config()
    label = label or code.replace('.HK', '').lstrip('0') or '0'

    f = WindFetcher(lookback_days=cfg['lookback_days'])
    df_full = f.fetch(code, asof=asof)
    df_full = forward_adjust(df_full).sort_values('date').reset_index(drop=True)
    f.close()

    if df_full.empty:
        return pd.DataFrame()

    listing_first_day = df_full['date'].min()

    # Daily indicators
    df_full = compute_daily_mas(df_full)
    df_full = compute_ma60_slope(df_full, cfg.get('exit_ma60_slope_lookback', 5))

    # Weekly indicators
    all_weekly_mas = tuple(set(cfg['weekly_long_mas'] + cfg['weekly_short_mas'] + [cfg.get('weekly_fast_ma', 5)]))
    weekly_df = compute_weekly_mas(df_full, mas=all_weekly_mas)
    weekly_df = weekly_df.sort_index()
    weekly_dates = list(weekly_df.index)

    # Backtest window
    cutoff = pd.to_datetime(asof) - pd.DateOffset(months=months)
    df = df_full[df_full['date'] >= cutoff].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame()

    # Pre-compute SOS
    df['high60'] = df['fwd_close'].rolling(60, min_periods=10).max().shift(1)
    df['is_new_high'] = df['fwd_close'] > df['high60']
    df['sos_raw'] = df.apply(
        lambda r: classify_sos(r, cfg, is_new_high=bool(r.get('is_new_high', False))), axis=1
    )

    recs, _, _ = run_state_machine(df, weekly_df, weekly_dates, cfg, listing_first_day)

    out = pd.DataFrame(recs)
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            '..', 'backtest', f'v5_3_{label}_history.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_csv(out_path, index=False)
    return out


def print_summary(label, df, months):
    """Print a human-readable summary of backtest results."""
    if df.empty:
        print(f"\n{'=' * 60}\n{label} — no data\n{'=' * 60}")
        return

    print(f"\n{'=' * 60}\n{label} ({months} months)\n{'=' * 60}")
    print(f"Data: {df['date'].min()} ~ {df['date'].max()}, {len(df)} rows")
    print(f"\n=== State distribution ===")
    print(df['state'].value_counts())
    print(f"\n=== TRENDING substate ===")
    trd = df[df['state'] == 'TRENDING']
    if len(trd):
        print(trd['substate'].value_counts())

    sos_list = df[df['sos'] != '']
    print(f"\n=== SOS events ({len(sos_list)}) ===")
    if len(sos_list):
        print(sos_list[['date', 'state', 'substate', 'fwd_close', 'sos']].to_string(index=False))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Wyckoff v5.4 backtest')
    parser.add_argument('--asof', default='2026-05-29')
    parser.add_argument('--months', type=int, default=12)
    parser.add_argument('--stock', default=None, help='Single stock, e.g. 0992.HK')
    args = parser.parse_args()

    if args.stock:
        lbl = args.stock.replace('.HK', '').lstrip('0')
        result = run_one(args.stock, asof=args.asof, months=args.months, label=lbl)
        print_summary(lbl, result, args.months)

"""Full scan: fetch from Wind DB, run v5.3 state machine, output JSON for frontend (v5.4).

v5.4 changes (output/display layer only, zero state machine modification):
  - Event tracking: SOS from SETUP_OK/ENTANGLED, TRENDING→PULLBACK, →EXIT in last 5d
  - ma10_slope_pct: unified sorting metric
  - recent_new_high_flag: for display_tier upgrade
  - Three-tab structured JSON output + Excel export

Usage:
    python src/run_scan.py [--asof 2026-05-29] [--workers 4]
"""
from __future__ import annotations

import sys
import os
import csv
import json
import logging
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np

from config import load_config
from data_provider import WindFetcher, forward_adjust
from indicators import (compute_daily_mas, compute_ma60_slope, compute_weekly_mas,
                        daily_ma10_slope, is_bear_market, is_entangled)
from substate import trending_substate
from sos import classify_sos


SECTOR_MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', 'configs', 'hk_sector_map.csv')


def load_sector_map() -> dict[str, dict]:
    result = {}
    if not os.path.isfile(SECTOR_MAP_PATH):
        return result
    with open(SECTOR_MAP_PATH, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            code = row.get('code', '').strip()
            if not code:
                continue
            if code not in result:
                result[code] = {
                    'name_cn': row.get('name_cn', ''),
                    'gics_sector': row.get('gics_sector', ''),
                    'sub_industry': row.get('sub_industry', ''),
                }
    return result


def compute_ma200_slope(df: pd.DataFrame, lookback: int = 20) -> float | None:
    if 'fwd_close' not in df.columns or len(df) < 220:
        return None
    closes = df['fwd_close'].values
    ma200_now = np.mean(closes[-200:])
    ma200_prev = np.mean(closes[-(200 + lookback):-lookback])
    if ma200_prev == 0:
        return None
    return round((ma200_now / ma200_prev - 1) * 100, 2)


def scan_one(code: str, asof: str, cfg: dict, sector_info: dict) -> dict | None:
    """Scan a single stock with v5.3 state machine."""
    try:
        f = WindFetcher(lookback_days=cfg['lookback_days'])
        df_full = f.fetch(code, asof=asof)
        df_full = forward_adjust(df_full).sort_values('date').reset_index(drop=True)
        f.close()

        if df_full.empty:
            return None
        listing_days = (df_full['date'].max() - df_full['date'].min()).days
        is_new_stock = listing_days < cfg['new_stock_days']
        min_days = cfg.get('min_history_days_new', 60) if is_new_stock else cfg.get('min_history_days', 230)
        if len(df_full) < min_days:
            return None

        listing_first_day = df_full['date'].min()

        df_full = compute_daily_mas(df_full)
        df_full = compute_ma60_slope(df_full, cfg.get('exit_ma60_slope_lookback', 5))
        df_full['ma200'] = df_full['fwd_close'].rolling(200).mean()

        all_weekly_mas = tuple(set(cfg['weekly_long_mas'] + cfg['weekly_short_mas'] + [cfg.get('weekly_fast_ma', 5)]))
        weekly_df = compute_weekly_mas(df_full, mas=all_weekly_mas)
        weekly_df = weekly_df.sort_index()
        weekly_dates = list(weekly_df.index)

        cutoff = pd.to_datetime(asof) - pd.DateOffset(months=3)
        df = df_full[df_full['date'] >= cutoff].reset_index(drop=True)

        if df.empty:
            return None

        # Pre-compute SOS (v5.3: needs is_new_high)
        df['high60'] = df['fwd_close'].rolling(60, min_periods=10).max().shift(1)
        df['is_new_high'] = df['fwd_close'] > df['high60']
        df['sos_raw'] = df.apply(
            lambda r: classify_sos(r, cfg, is_new_high=bool(r.get('is_new_high', False))), axis=1
        )

        entangled_disp = cfg.get('entangled_disp', 0.08)
        setup_idle_days = cfg.get('setup_idle_days', 5)

        # State machine
        state = 'POOL'
        substate = ''
        days_in_state = 0
        consec_below_ma10 = 0
        pullback_peak = None
        days_in_pullback = 0
        base_max = None
        days_in_exit = 0
        setup_idle = 0
        last_sos = ''
        bear = False

        # v5.4 event tracking (observation only, does NOT affect state transitions)
        recent_events = []       # [(day_index, event_type, detail, state_at_time)]
        recent_new_highs = []    # [day_index] where TRENDING stock hit 60d new high

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
            # Also track: SOS that *causes* entry into SETUP_OK (state will change below)
            sos_caused_setup = (sos_today in ('SOS-B', 'SOS-C') and not bear
                                and state in ('POOL', 'ENTANGLED', 'EXIT'))

            broke_ma60 = (
                pd.notna(row.get('ma60')) and row['fwd_close'] < row['ma60']
                and pd.notna(row.get('ma60_slope5')) and row['ma60_slope5'] < 0
            )

            # ── EXIT trigger ──
            if state in ('TRENDING', 'PULLBACK', 'SETUP_OK', 'ENTANGLED') and broke_ma60:
                prev_state_for_evt = state  # v5.4: capture before overwriting
                state = 'EXIT'; substate = ''
                days_in_state = 1; days_in_exit = 1
                consec_below_ma10 = 0; pullback_peak = None; base_max = None
                setup_idle = 0; last_sos = sos_today
                recent_events.append((i, 'to_EXIT', prev_state_for_evt, prev_state_for_evt))  # v5.4
                continue

            # ── EXIT ──
            if state == 'EXIT':
                days_in_exit += 1
                if days_in_exit < cfg['exit_cooldown_days']:
                    last_sos = sos_today
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
                    if sos_caused_setup: recent_events.append((i, 'sos_from_setup', sos_today, 'SETUP_OK'))
                elif entangled:
                    state = 'ENTANGLED'; substate = ''
                    days_in_state = 1; days_in_exit = 0; setup_idle = 0
                else:
                    state = 'POOL'; substate = ''
                    days_in_state = 1; days_in_exit = 0; setup_idle = 0
                last_sos = sos_today
                continue

            # ── POOL ──
            if state == 'POOL':
                can_trending = bool(sub_today) or (sos_today == 'SOS-A' and not bear)
                can_setup = sos_today in ('SOS-B', 'SOS-C') and not bear
                if can_trending:
                    state = 'TRENDING'
                    substate = 'NEW' if is_new_stock else sub_today
                    days_in_state = 1; base_max = row['fwd_close']; consec_below_ma10 = 0; setup_idle = 0
                elif can_setup:
                    state = 'SETUP_OK'; substate = ''
                    days_in_state = 1; setup_idle = 0
                    if sos_caused_setup: recent_events.append((i, 'sos_from_setup', sos_today, 'SETUP_OK'))
                elif entangled:
                    state = 'ENTANGLED'; substate = ''
                    days_in_state = 1; setup_idle = 0
                last_sos = sos_today
                continue

            # ── ENTANGLED ──
            if state == 'ENTANGLED':
                can_trending = bool(sub_today) or (sos_today == 'SOS-A' and not bear)
                can_setup = sos_today in ('SOS-B', 'SOS-C') and not bear
                if can_trending:
                    state = 'TRENDING'
                    substate = 'NEW' if is_new_stock else sub_today
                    days_in_state = 1; base_max = row['fwd_close']; consec_below_ma10 = 0; setup_idle = 0
                elif can_setup:
                    state = 'SETUP_OK'; substate = ''
                    days_in_state = 1; setup_idle = 0
                    if sos_caused_setup: recent_events.append((i, 'sos_from_setup', sos_today, 'SETUP_OK'))
                elif not entangled:
                    state = 'POOL'; substate = ''
                    days_in_state = 1; setup_idle = 0
                last_sos = sos_today
                continue

            # ── SETUP_OK ──
            if state == 'SETUP_OK':
                can_trending = bool(sub_today) or (sos_today == 'SOS-A' and not bear)
                if can_trending:
                    state = 'TRENDING'
                    substate = 'NEW' if is_new_stock else sub_today
                    days_in_state = 1; base_max = row['fwd_close']; consec_below_ma10 = 0; setup_idle = 0
                    last_sos = sos_today
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

                last_sos = sos_today
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
                    last_sos = sos_today
                    recent_events.append((i, 'TRENDING_to_PULLBACK', None, 'TRENDING'))  # v5.4
                    continue

                if base_max is None or row['fwd_close'] > base_max:
                    base_max = row['fwd_close']
                last_sos = sos_today
                continue

            # ── PULLBACK ──
            if state == 'PULLBACK':
                days_in_pullback += 1
                if days_in_pullback >= cfg['pullback_max_days']:
                    state = 'EXIT'; substate = ''
                    days_in_state = 1; days_in_exit = 1
                    consec_below_ma10 = 0; pullback_peak = None; base_max = None; setup_idle = 0
                    last_sos = sos_today
                    recent_events.append((i, 'to_EXIT', 'PULLBACK', 'PULLBACK'))  # v5.4
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
                last_sos = sos_today

        # Build result
        last = df.iloc[-1]
        ma200_slope = compute_ma200_slope(df_full)
        info = sector_info.get(code, {})

        # v5.4: compute ma10_slope_pct
        ma10_slope_pct = None
        if len(df) > 10:
            slope_val = daily_ma10_slope(df, len(df) - 1, lookback=10)
            if slope_val is not None:
                ma10_slope_pct = round(slope_val * 100, 2)

        # v5.4: compute 5-day event flags
        last_n = len(df)
        n_days = min(5, last_n)
        last_5_indices = set(range(last_n - n_days, last_n))

        sos_setup_recent = ''
        for (day_i, evt_type, detail, evt_state) in recent_events:
            if evt_type == 'sos_from_setup' and day_i in last_5_indices:
                if not sos_setup_recent or detail < sos_setup_recent:
                    sos_setup_recent = detail

        recent_new_high_flag = any(d in last_5_indices for d in recent_new_highs)
        trending_to_pullback_recent = any(
            day_i in last_5_indices for (day_i, evt_type, _, _) in recent_events
            if evt_type == 'TRENDING_to_PULLBACK'
        )
        to_exit_recent = any(
            day_i in last_5_indices for (day_i, evt_type, _, _) in recent_events
            if evt_type == 'to_EXIT'
        )

        return dict(
            ticker=code,
            name=info.get('name_cn', ''),
            name_cn=info.get('name_cn', ''),
            gics_sector=info.get('gics_sector', ''),
            sub_industry=info.get('sub_industry', ''),
            state=state,
            substate=substate,
            last_close=round(float(last['fwd_close']), 3),
            ma10=round(float(last['ma10']), 3) if pd.notna(last.get('ma10')) else None,
            ma20=round(float(last['ma20']), 3) if pd.notna(last.get('ma20')) else None,
            ma60=round(float(last['ma60']), 3) if pd.notna(last.get('ma60')) else None,
            ma200=round(float(last['ma200']), 3) if pd.notna(last.get('ma200')) else None,
            ma200_slope=ma200_slope,
            ma10_slope_pct=ma10_slope_pct,
            days_in_state=days_in_state,
            days_in_pullback=days_in_pullback,
            pullback_dd_pct=round((last['fwd_close'] / pullback_peak - 1) * 100, 2) if pullback_peak else None,
            sos=last_sos,
            bear_gate=bear,
            is_new_stock=is_new_stock,
            sos_setup_recent=sos_setup_recent,
            recent_new_high_flag=recent_new_high_flag,
            trending_to_pullback_recent=trending_to_pullback_recent,
            to_exit_recent=to_exit_recent,
            date=asof,
        )
    except Exception as e:
        logging.error(f"{code} scan failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Wyckoff v5.3 full scan from Wind DB')
    parser.add_argument('--asof', default=None, help='Cutoff date (default: today)')
    parser.add_argument('--config', default=None, help='Config file path')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--workers', type=int, default=4, help='Concurrent threads')
    args = parser.parse_args()

    if args.asof is None:
        args.asof = date.today().strftime('%Y-%m-%d')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(args.config or os.path.join(base_dir, '..', 'configs', 'v5_3.json'))

    sector_map = load_sector_map()
    stocks = list(sector_map.keys())
    print(f"Loaded {len(stocks)} stocks from sector map, asof={args.asof}")

    out_dir = args.output or os.path.join(base_dir, '..', 'public', 'data')
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, 'scan_errors.log')
    logging.basicConfig(filename=log_path, level=logging.ERROR,
                        format='%(asctime)s %(message)s')

    results = []
    errors = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(scan_one, code, args.asof, cfg, sector_map): code
            for code in stocks
        }
        done = 0
        for future in as_completed(futures):
            code = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    errors.append((code, 'no data'))
            except Exception as e:
                errors.append((code, str(e)))
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {done}/{len(stocks)} ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    print(f"\nScan complete: {len(results)} ok, {len(errors)} failed, {elapsed:.1f}s")

    if errors:
        print(f"Errors (first 10):")
        for code, err in errors[:10]:
            print(f"  {code}: {err}")

    if not results:
        print("No results, exiting")
        return

    rdf = pd.DataFrame(results)
    pools_path = os.path.join(out_dir, 'pools_hk.json')
    rdf.to_json(pools_path, orient='records', indent=2, force_ascii=False)
    print(f"\nPools JSON: {len(rdf)} stocks -> {pools_path}")

    # v5.4: Three-tab structured output + Excel
    from v5_4_three_tabs import build_three_tabs
    from v5_4_export_excel import export_excel

    three_tabs = build_three_tabs(results)

    # Clean NaN/inf values (not valid JSON)
    def _clean(obj):
        import math
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        return obj

    three_tabs = _clean(three_tabs)

    three_tabs_path = os.path.join(out_dir, 'three_tabs_hk.json')
    with open(three_tabs_path, 'w', encoding='utf-8') as f:
        json.dump(three_tabs, f, ensure_ascii=False, indent=2)
    print(f"Three-tabs JSON -> {three_tabs_path}")

    excel_path = os.path.join(out_dir, 'v5_4_hk.xlsx')
    export_excel(three_tabs, excel_path)
    print(f"Excel -> {excel_path}")

    from collections import Counter
    print(f"\n=== {args.asof} Scan Summary ===")
    states = Counter(rdf['state'])
    for s, c in states.most_common():
        print(f"  {s}: {c}")

    trending = rdf[rdf['state'] == 'TRENDING']
    if len(trending):
        substates = Counter(trending['substate'])
        print(f"\n  TRENDING substates:")
        for s, c in substates.most_common():
            print(f"    {s}: {c}")

    sectors = Counter(rdf['gics_sector'])
    print(f"\n  By GICS Sector:")
    for s, c in sectors.most_common():
        print(f"    {s}: {c}")


if __name__ == '__main__':
    main()

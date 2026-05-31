"""Wyckoff v5.2 batch scanner.

Scans a pool of HK stocks and outputs 4 CSV files:
  - scan_v5_2_strong.csv   — TRENDING(substate=STRONG) pool
  - scan_v5_2_trending.csv — All TRENDING (with substate)
  - scan_v5_2_pullback.csv — PULLBACK pool
  - scan_v5_2_sos.csv      — Stocks with SOS triggers
"""
from __future__ import annotations

import sys
import os
import logging
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

from config import load_config
from state_machine import run_one


def load_pool(path: str) -> list[tuple[str, str]]:
    """Load stock pool from CSV. Returns [(code, name), ...]."""
    df = pd.read_csv(path)
    cols = df.columns.tolist()
    code_col = cols[0]
    name_col = cols[1] if len(cols) > 1 else None
    result = []
    for _, r in df.iterrows():
        code = str(r[code_col]).strip()
        if not code.endswith('.HK'):
            code = f"{int(code):04d}.HK"
        name = str(r[name_col]).strip() if name_col else ''
        result.append((code, name))
    return result


def scan_one(code: str, name: str, asof: str, cfg: dict) -> dict | None:
    """Scan a single stock. Returns state summary or None on failure."""
    try:
        df = run_one(code, asof=asof, months=3, cfg=cfg)
        if df.empty:
            return None
        last = df.iloc[-1]
        today_rows = df[df['date'] == last['date']]
        today_sos = ''
        for _, r in today_rows.iterrows():
            if r.get('sos', ''):
                today_sos = r['sos']
                break

        return dict(
            code=code,
            name=name,
            date=last['date'],
            state=last['state'],
            substate=last.get('substate', ''),
            fwd_close=last['fwd_close'],
            ma10=last.get('ma10'),
            ma20=last.get('ma20'),
            ma60=last.get('ma60'),
            days_in_state=last.get('days_in_state', 0),
            days_in_pullback=last.get('days_in_pullback', 0),
            pullback_dd_pct=last.get('pullback_dd_pct'),
            is_new_stock=last.get('is_new_stock', False),
            bear_gate=last.get('bear_gate', False),
            sos=today_sos,
        )
    except Exception as e:
        logging.error(f"{code} scan failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Wyckoff v5.2 daily scanner')
    parser.add_argument('--asof', default=None, help='Cutoff date (default: today)')
    parser.add_argument('--config', default=None, help='Config file path')
    parser.add_argument('--pool', default=None, help='Stock pool CSV path')
    parser.add_argument('--output', default=None, help='Output directory')
    parser.add_argument('--workers', type=int, default=4, help='Concurrent threads')
    args = parser.parse_args()

    if args.asof is None:
        args.asof = date.today().strftime('%Y-%m-%d')

    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg = load_config(args.config or os.path.join(base_dir, '..', 'configs', 'v5_2.json'))

    pool_path = args.pool or os.path.join(base_dir, '..', 'configs', 'pool_hk.csv')
    stocks = load_pool(pool_path)
    print(f"Loaded pool: {len(stocks)} stocks, asof={args.asof}")

    out_dir = args.output or os.path.join(base_dir, '..', 'output')
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, 'scan_v5_2_errors.log')
    logging.basicConfig(filename=log_path, level=logging.ERROR,
                        format='%(asctime)s %(message)s')

    results = []
    errors = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(scan_one, code, name, args.asof, cfg): (code, name)
            for code, name in stocks
        }
        done = 0
        for future in as_completed(futures):
            code, name = futures[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                else:
                    errors.append((code, name, 'no data'))
            except Exception as e:
                errors.append((code, name, str(e)))
            done += 1
            if done % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {done}/{len(stocks)} ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    print(f"\nScan complete: {len(results)} ok, {len(errors)} failed, {elapsed:.1f}s")

    if errors:
        print("Errors (first 10):")
        for code, name, err in errors[:10]:
            print(f"  {code} {name}: {err}")

    if not results:
        print("No results, exiting")
        return

    rdf = pd.DataFrame(results)

    # 1. STRONG pool
    strong = rdf[(rdf['state'] == 'TRENDING') & (rdf['substate'] == 'STRONG')]
    strong.to_csv(os.path.join(out_dir, 'scan_v5_2_strong.csv'), index=False)
    print(f"\nSTRONG: {len(strong)}")

    # 2. All TRENDING
    trending = rdf[rdf['state'] == 'TRENDING']
    trending.to_csv(os.path.join(out_dir, 'scan_v5_2_trending.csv'), index=False)
    print(f"TRENDING: {len(trending)}")

    # 3. PULLBACK pool
    pullback = rdf[rdf['state'] == 'PULLBACK']
    pullback.to_csv(os.path.join(out_dir, 'scan_v5_2_pullback.csv'), index=False)
    print(f"PULLBACK: {len(pullback)}")

    # 4. SOS triggers
    sos = rdf[rdf['sos'] != '']
    sos.to_csv(os.path.join(out_dir, 'scan_v5_2_sos.csv'), index=False)
    print(f"SOS: {len(sos)}")

    # Summary
    print(f"\n=== {args.asof} Scan Summary ===")
    print(f"  TRENDING: {len(trending)}")
    if len(trending):
        print(f"    STRONG: {(trending['substate'] == 'STRONG').sum()}")
        print(f"    MID:    {(trending['substate'] == 'MID').sum()}")
        print(f"    EARLY:  {(trending['substate'] == 'EARLY').sum()}")
        print(f"    NEW:    {(trending['substate'] == 'NEW').sum()}")
    print(f"  PULLBACK: {len(pullback)}")
    print(f"  EXIT:     {len(rdf[rdf['state'] == 'EXIT'])}")
    print(f"  POOL:     {len(rdf[rdf['state'] == 'POOL'])}")
    print(f"  SOS:      {len(sos)} (SOS-A: {len(sos[sos['sos'] == 'SOS-A'])})")


if __name__ == '__main__':
    main()

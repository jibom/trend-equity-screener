"""Scan HK stocks for KDJ bottom + MACD golden cross combined signal.

Filter: weekly J<0 in past 3 months (oversold precondition)
Signal layers:
  1. KDJ divergence = 情绪底部 setup
  2. MACD golden cross = 改善确认
  3. Combined: 情绪底部+改善
"""
from __future__ import annotations

import sys
import os
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data_provider import WindFetcher, forward_adjust
from indicators import (
    compute_kdj, compute_weekly_kdj, detect_kdj_divergence,
    compute_macd, detect_macd_golden_cross,
)
import numpy as np

SECTOR_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'hk_sector_map.csv')
ASOF = os.getenv('KDJ_ASOF', '2026-06-01')


def check(code: str, name: str):
    try:
        f = WindFetcher(lookback_days=600)
        df = f.fetch(code, asof=ASOF)
        df = forward_adjust(df).sort_values('date').reset_index(drop=True)
        f.close()
        if df.empty or len(df) < 60:
            return None

        df = compute_kdj(df)
        df = compute_macd(df)
        wk = compute_weekly_kdj(df)
        wk = compute_macd(wk)

        # ── Filter: weekly J<0 in past 3 months ──
        if len(wk) < 13:
            return None
        if not np.any(wk.tail(13)['j'].values < 0):
            return None

        # ── Weekly signals ──
        w_kdj = detect_kdj_divergence(wk, lookback=15)
        w_macd = detect_macd_golden_cross(wk, lookback=10)

        # ── Daily signals ──
        d_kdj = detect_kdj_divergence(df, lookback=30)
        d_macd = detect_macd_golden_cross(df, lookback=15)

        # ── Combined signal ──
        # Daily: KDJ divergence + MACD golden cross
        d_combined = d_kdj['bullish_divergence'] and d_macd['golden_cross']
        # Weekly: KDJ divergence + MACD golden cross
        w_combined = w_kdj['bullish_divergence'] and w_macd['golden_cross']
        # Any layer: weekly or daily combined
        any_combined = d_combined or w_combined

        last = df.iloc[-1]
        last_wk = wk.iloc[-1]

        return dict(code=code, name=name,
                    close=last['fwd_close'],
                    k=last['k'], d=last['d'], j=last['j'],
                    wk_j=last_wk['j'],
                    # Daily KDJ
                    d_kdj_div=d_kdj['bullish_divergence'],
                    d_kdj_count=d_kdj['divergence_count'],
                    # Daily MACD
                    d_macd_gc=d_macd['golden_cross'],
                    d_macd_recent=d_macd['recent_golden'],
                    d_macd_below0=d_macd['dif_below_zero'],
                    # Daily combined
                    d_combined=d_combined,
                    # Weekly KDJ
                    w_kdj_div=w_kdj['bullish_divergence'],
                    w_kdj_count=w_kdj['divergence_count'],
                    # Weekly MACD
                    w_macd_gc=w_macd['golden_cross'],
                    w_macd_recent=w_macd['recent_golden'],
                    w_macd_below0=w_macd['dif_below_zero'],
                    # Weekly combined
                    w_combined=w_combined,
                    # Overall
                    any_combined=any_combined)
    except Exception:
        return None


def main():
    codes = []
    with open(SECTOR_MAP, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            c = row.get('code', '').strip()
            if c:
                codes.append((c, row.get('name_cn', '')))

    print(f"Scanning {len(codes)} stocks, asof={ASOF}")
    t0 = time.time()
    results = []

    with ThreadPoolExecutor(max_workers=4) as ex:
        futs = {ex.submit(check, c, n): c for c, n in codes}
        done = 0
        for fut in as_completed(futs):
            r = fut.result()
            if r is not None:
                results.append(r)
            done += 1
            if done % 50 == 0:
                print(f"  {done}/{len(codes)} ({time.time()-t0:.0f}s)", flush=True)

    elapsed = time.time() - t0

    # Sort: combined signal first, then weekly KDJ+MACD, then daily
    results.sort(key=lambda x: (
        not x['any_combined'],
        not x['w_combined'],
        not x['d_combined'],
        not (x['w_kdj_div'] and x['w_macd_gc']),
        not (x['d_kdj_div'] and x['d_macd_gc']),
        -x['wk_j'],
    ))

    # Summary
    n_wk_j0 = len(results)
    n_w_kdj = sum(1 for r in results if r['w_kdj_div'])
    n_d_kdj = sum(1 for r in results if r['d_kdj_div'])
    n_w_macd = sum(1 for r in results if r['w_macd_gc'])
    n_d_macd = sum(1 for r in results if r['d_macd_gc'])
    n_w_combo = sum(1 for r in results if r['w_combined'])
    n_d_combo = sum(1 for r in results if r['d_combined'])
    n_any = sum(1 for r in results if r['any_combined'])

    print(f"\n{'='*60}")
    print(f"Scan complete ({elapsed:.0f}s)")
    print(f"  Weekly J<0 in 3m:     {n_wk_j0}")
    print(f"  Weekly KDJ divergence: {n_w_kdj}")
    print(f"  Daily  KDJ divergence: {n_d_kdj}")
    print(f"  Weekly MACD golden:    {n_w_macd}")
    print(f"  Daily  MACD golden:    {n_d_macd}")
    print(f"  --- Combined ---")
    print(f"  Weekly 情绪底部+改善:  {n_w_combo}")
    print(f"  Daily  情绪底部+改善:  {n_d_combo}")
    print(f"  Any    情绪底部+改善:  {n_any}")
    print(f"{'='*60}\n")

    # ── Print table ──
    fmt = '{:<10} {:<6} {:>6} {:>6} {:>5}  {:>4} {:>4} {:>4}  {:>4} {:>4} {:>4}  {}'
    print(fmt.format('code', 'name', 'close', 'D-J', 'W-J',
                     'DKDJ', 'DMAC', 'Dsig',
                     'WKDJ', 'WMAC', 'Wsig',
                     'Signal'))
    print('-' * 115)

    for r in results:
        d_kdj_s = str(r['d_kdj_count']) if r['d_kdj_div'] else '-'
        d_macd_s = 'GC' if r['d_macd_gc'] else '-'
        d_combo = 'Y' if r['d_combined'] else ''

        w_kdj_s = str(r['w_kdj_count']) if r['w_kdj_div'] else '-'
        w_macd_s = 'GC' if r['w_macd_gc'] else '-'
        w_combo = 'Y' if r['w_combined'] else ''

        # Build signal label
        signals = []
        if r['d_combined']:
            signals.append('D:bottom+improve')
        elif r['d_kdj_div'] and not r['d_macd_gc']:
            signals.append('D:bottom_setup')
        if r['w_combined']:
            signals.append('W:bottom+improve')
        elif r['w_kdj_div'] and not r['w_macd_gc']:
            signals.append('W:bottom_setup')
        sig_str = ' | '.join(signals) if signals else ''

        print(fmt.format(
            r['code'], r['name'][:6],
            f"{r['close']:.1f}", f"{r['j']:.1f}", f"{r['wk_j']:.1f}",
            d_kdj_s, d_macd_s, d_combo,
            w_kdj_s, w_macd_s, w_combo,
            sig_str[:40]))


if __name__ == '__main__':
    main()

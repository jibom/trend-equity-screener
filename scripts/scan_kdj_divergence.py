"""Scan HK stocks for KDJ bottom + MACD golden cross combined signal.

Filter: weekly D<50 + weekly J<0 in past 3 months (oversold precondition)
Signal layers:
  1. KDJ divergence (with J-confirmed bounce) = 情绪底部 setup
  2. MACD golden cross = 改善确认
  3. Combined: 情绪底部+改善
"""
from __future__ import annotations

import sys
import os
import csv
import json
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data_provider import WindFetcher, forward_adjust
from indicators import (
    compute_kdj, compute_weekly_kdj, detect_kdj_divergence,
    compute_macd, detect_macd_golden_cross,
)
from kdj_divergence import detect_kdj_bullish_divergence
import numpy as np

SECTOR_MAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'configs', 'hk_sector_map.csv')
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'public', 'data')
ASOF = os.getenv('KDJ_ASOF') or datetime.now().strftime('%Y-%m-%d')

# Weekly KDJ divergence detection parameters
WEEKLY_LOOKBACK = 40    # ~10 months window for divergence detection
RECENT_BARS = 5         # divergence must occur within last 5 weekly bars (~1 month)
CONFIRM_BARS_FULL = 2   # ≥2 weekly bars after J-low = fully confirmed
CONFIRM_BARS_MIN = 1    # 1 weekly bar after J-low = needs daily confirmation


def _daily_confirms_reversal(df, lookback=10):
    """Check if daily data confirms a reversal in the last N trading days.

    Returns True if ANY of:
      1. Doji / spinning top (body < 20% of range)
      2. Daily KDJ golden cross (K crosses above D)
      3. Volume-surge big up day (close up >3% with volume >1.5x 20d MA)
    """
    if len(df) < lookback + 1:
        return False
    recent = df.tail(lookback + 1).copy()
    vol_ma = recent['volume'].rolling(20, min_periods=5).mean()

    for i in range(1, len(recent)):
        row = recent.iloc[i]
        o, h, l, c = row['fwd_open'], row['fwd_high'], row['fwd_low'], row['fwd_close']
        v = row['volume']
        rng = h - l

        # 1. Doji / spinning top: body < 20% of range
        if rng > 0 and abs(c - o) / rng < 0.2:
            return True

        # 2. Daily KDJ golden cross
        if row.get('kd_golden_cross', False):
            return True

        # 3. Volume-surge big up day: +3% and volume > 1.5x MA
        if o > 0 and (c - o) / o > 0.03:
            vma = vol_ma.iloc[i]
            if not np.isnan(vma) and vma > 0 and v > vma * 1.5:
                return True

    return False


def _weekly_kdj_divergence_confirmed(wk, daily_df=None):
    """Detect weekly KDJ divergence with two complementary methods:

    Method A: J-low vs J-low divergence (from kdj_divergence module)
      - Two J local minima: price lower-low, J higher-low

    Method B: Price-low vs prior J-low divergence
      - J already bounced (not at a local low), but price makes new low
      - This catches cases like 2313.HK: J turned up at 3/27, price kept
        falling to 5/22 new low while J stayed elevated = strong divergence

    Extra filters:
      - second low must have ≥ CONFIRM_BARS_FULL bars after it (confirmed)
      - if only CONFIRM_BARS_MIN bar after, require daily reversal confirmation
      - divergence must occur within last RECENT_BARS weekly bars (~1 month)
    Returns (has_divergence, total_count, recent_count, details_list).
    """
    window_len = min(len(wk), WEEKLY_LOOKBACK)
    wk_tail = wk.tail(window_len).copy().reset_index(drop=True)
    closes = wk_tail['fwd_close'].values
    js = wk_tail['j'].values
    ds = wk_tail['d'].values
    dates = wk_tail.index  # integer index in window

    all_details = []  # collect from both methods

    # ── Method A: J-low vs J-low (existing module) ──
    result = detect_kdj_bullish_divergence(
        close=wk['fwd_close'], kdj_j=wk['j'], kdj_d=wk['d'],
        high=wk.get('fwd_high'), low=wk.get('fwd_low'),
        volume=wk.get('volume'),
        lookback=WEEKLY_LOOKBACK,
        order=5, d_threshold=50.0,
        use_volume_filter=False,
    )
    all_details.extend(result['divergence_details'])

    # ── Method B: Price-low vs prior J-low ──
    # Find J local minima (reference points) and price local minima
    from scipy.signal import argrelextrema as _argrel

    j_low_idxs = _argrel(js, np.less_equal, order=5)[0]
    p_low_idxs = _argrel(closes, np.less_equal, order=5)[0]

    # Also include the very last bar as a candidate price low
    # (if it's near the lowest close in the tail)
    last_close_rank = np.sum(closes <= closes[-1])
    if last_close_rank <= 3 and (len(p_low_idxs) == 0 or p_low_idxs[-1] != window_len - 1):
        p_low_idxs = np.append(p_low_idxs, window_len - 1)

    for pi in p_low_idxs:
        if ds[pi] >= 50:
            continue
        # Find the most recent J-low before this price low
        prev_j_lows = j_low_idxs[j_low_idxs < pi]
        if len(prev_j_lows) == 0:
            continue
        ji = prev_j_lows[-1]  # nearest prior J low

        # Price at price-low must be lower than price at J-low
        if not (closes[pi] < closes[ji]):
            continue
        # J at price-low must be higher than J at J-low (divergence)
        if not (js[pi] > js[ji]):
            continue
        # J diff meaningful
        if abs(js[ji]) > 1e-6 and (js[pi] - js[ji]) / abs(js[ji]) < 0.03:
            continue
        # Min spacing
        if (pi - ji) < 8:
            continue
        # Avoid duplicates with Method A
        dup = False
        for d in all_details:
            if d['idx1'] == ji and d['idx2'] == pi:
                dup = True
                break
        if dup:
            continue

        date_ji = str(wk_tail.index[ji]) if not hasattr(wk_tail.index[ji], 'strftime') else str(wk_tail.index[ji])[:10]
        date_pi = str(wk_tail.index[pi]) if not hasattr(wk_tail.index[pi], 'strftime') else str(wk_tail.index[pi])[:10]
        # Get actual dates from wk
        actual_dates = wk.tail(window_len).index
        date_ji = str(actual_dates[ji])[:10]
        date_pi = str(actual_dates[pi])[:10]

        all_details.append(dict(
            idx1=int(ji), idx2=int(pi),
            date1=date_ji, date2=date_pi,
            price1=round(float(closes[ji]), 4),
            price2=round(float(closes[pi]), 4),
            j1=round(float(js[ji]), 2),
            j2=round(float(js[pi]), 2),
            d2=round(float(ds[pi]), 2),
        ))

    if not all_details:
        return False, 0, 0, []

    # ── Apply confirmation & recency filters ──
    valid_recent = []
    for d in all_details:
        bars_after = window_len - 1 - d['idx2']
        # Must be within recent window
        if bars_after >= RECENT_BARS:
            continue
        # J must have bounced: at least CONFIRM_BARS_MIN bars after, and J[idx2+1] > J[idx2]
        if bars_after < CONFIRM_BARS_MIN:
            continue
        if js[d['idx2'] + 1] <= d['j2']:
            continue
        # Confirmation tier
        if bars_after < CONFIRM_BARS_FULL:
            if daily_df is None or not _daily_confirms_reversal(daily_df):
                continue
        valid_recent.append(d)

    total = len(all_details)
    has = len(valid_recent) > 0
    return has, total, len(valid_recent), valid_recent


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

        # ── Filter: weekly J<5 in past 5 months (~20 weeks) ──
        if len(wk) < 20:
            return None
        if not np.any(wk.tail(20)['j'].values < 5):
            return None

        # ── Weekly KDJ divergence (confirmed, recent) ──
        w_kdj_conf, w_kdj_total, w_kdj_recent, w_kdj_details = _weekly_kdj_divergence_confirmed(wk, daily_df=df)
        w_macd = detect_macd_golden_cross(wk, lookback=10)

        # ── Daily signals ──
        d_kdj = detect_kdj_divergence(df, lookback=30)
        d_macd = detect_macd_golden_cross(df, lookback=15)

        # ── Combined signal ──
        d_combined = d_kdj['bullish_divergence'] and d_macd['golden_cross']
        w_combined = w_kdj_conf and w_macd['golden_cross']
        any_combined = d_combined or w_combined

        last = df.iloc[-1]
        last_wk = wk.iloc[-1]

        # Build divergence description
        div_desc = '; '.join(
            f"p={d['price1']:.1f}->{d['price2']:.1f}, J={d['j1']:.1f}->{d['j2']:.1f}"
            for d in w_kdj_details
        ) if w_kdj_details else ''

        return dict(code=code, name=name,
                    close=last['fwd_close'],
                    k=last['k'], d=last['d'], j=last['j'],
                    wk_j=last_wk['j'], wk_d=last_wk['d'],
                    # Daily KDJ
                    d_kdj_div=d_kdj['bullish_divergence'],
                    d_kdj_count=d_kdj['divergence_count'],
                    # Daily MACD
                    d_macd_gc=d_macd['golden_cross'],
                    d_macd_recent=d_macd['recent_golden'],
                    d_macd_below0=d_macd['dif_below_zero'],
                    # Daily combined
                    d_combined=d_combined,
                    # Weekly KDJ (confirmed divergence)
                    w_kdj_div=w_kdj_conf,
                    w_kdj_count=w_kdj_total,
                    w_kdj_recent=w_kdj_recent,
                    # Weekly MACD
                    w_macd_gc=w_macd['golden_cross'],
                    w_macd_recent=w_macd['recent_golden'],
                    w_macd_below0=w_macd['dif_below_zero'],
                    # Weekly combined
                    w_combined=w_combined,
                    # Overall
                    any_combined=any_combined,
                    # Divergence description
                    div_desc=div_desc)
    except Exception:
        return None


def main():
    # Load sector map with industry info
    codes = []
    industry_map = {}
    with open(SECTOR_MAP, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            c = row.get('code', '').strip()
            if c:
                codes.append((c, row.get('name_cn', '')))
                industry_map[c] = row.get('sub_industry', '')

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

    # Dedup by code (keep first = highest ranked)
    seen = set()
    unique = []
    for r in results:
        if r['code'] not in seen:
            seen.add(r['code'])
            r['industry'] = industry_map.get(r['code'], '')
            unique.append(r)
    results = unique

    # Summary
    n_wk_j0 = len(results)
    n_w_kdj = sum(1 for r in results if r['w_kdj_div'])
    n_d_kdj = sum(1 for r in results if r['d_kdj_div'])
    n_w_macd = sum(1 for r in results if r['w_macd_gc'])
    n_d_macd = sum(1 for r in results if r['d_macd_gc'])
    n_w_combo = sum(1 for r in results if r['w_combined'])
    n_d_combo = sum(1 for r in results if r['d_combined'])
    n_any = sum(1 for r in results if r['any_combined'])

    print(f"\n{'='*70}")
    print(f"Scan complete ({elapsed:.0f}s)")
    print(f"  Weekly J<0 in 3m:          {n_wk_j0}")
    print(f"  Weekly KDJ div (confirmed): {n_w_kdj}")
    print(f"  Daily  KDJ divergence:      {n_d_kdj}")
    print(f"  Weekly MACD golden:         {n_w_macd}")
    print(f"  Daily  MACD golden:         {n_d_macd}")
    print(f"  --- Combined ---")
    print(f"  Weekly 情绪底部+改善:       {n_w_combo}")
    print(f"  Daily  情绪底部+改善:       {n_d_combo}")
    print(f"  Any    情绪底部+改善:       {n_any}")
    print(f"{'='*70}\n")

    # ── Print table ──
    fmt = '{:<12} {:<8} {:>7} {:>6} {:>5}  {:>4} {:>4} {:>4}  {:>4} {:>4} {:>4}  {}'
    print(fmt.format('code', 'name', 'close', 'D-J', 'W-J',
                     'DKDJ', 'DMAC', 'Dsig',
                     'WKDJ', 'WMAC', 'Wsig',
                     'Divergence Detail'))
    print('-' * 125)

    for r in results:
        d_kdj_s = str(r['d_kdj_count']) if r['d_kdj_div'] else '-'
        d_macd_s = 'GC' if r['d_macd_gc'] else '-'
        d_combo = 'Y' if r['d_combined'] else ''

        w_kdj_s = f"{r['w_kdj_recent']}/{r['w_kdj_count']}" if r['w_kdj_div'] else '-'
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

        desc = r.get('div_desc', '')[:45]

        print(fmt.format(
            r['code'], r['name'][:8],
            f"{r['close']:.1f}", f"{r['j']:.1f}", f"{r['wk_j']:.1f}",
            d_kdj_s, d_macd_s, d_combo,
            w_kdj_s, w_macd_s, w_combo,
            desc or sig_str))

    # ── Write JSON output ──
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'kdj_divergence_hk.json')

    def _json_safe(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    json_output = {
        'date': ASOF,
        'summary': {
            'total_oversold': n_wk_j0,
            'weekly_kdj_divergence': n_w_kdj,
            'daily_kdj_divergence': n_d_kdj,
            'weekly_macd_golden': n_w_macd,
            'daily_macd_golden': n_d_macd,
            'weekly_combined': n_w_combo,
            'daily_combined': n_d_combo,
            'any_combined': n_any,
        },
        'stocks': results,
    }
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False, default=_json_safe)
    print(f"\nJSON output: {out_path} ({len(results)} stocks)")


if __name__ == '__main__':
    main()

"""KDJ+MACD v2 scan: 周线J曾<0 + 当前J<20 + 底背离.

筛选条件:
  1. 周线: 过去3个月(13周) J曾经跌破0
  2. 周线: 当前J < 20 (还在低位)
  3. 周线/日线: 出现KDJ底背离

信号分层:
  - W层: 周线底背离 (更强)
  - D层: 日线底背离
  - MACD辅助: 金叉确认 / 零轴下方金叉
"""
from __future__ import annotations

import sys
import os
import csv
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from data_provider import WindFetcher, forward_adjust
from indicators import (
    compute_kdj, compute_weekly_kdj, detect_kdj_divergence,
    compute_macd, compute_weekly_macd, detect_macd_golden_cross,
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

        # ── 条件1: 周线J曾<0 (过去3个月=13周) ──
        if len(wk) < 13:
            return None
        recent_wk = wk.tail(13)
        if not np.any(recent_wk['j'].values < 0):
            return None

        # ── 条件2: 当前周线J < 20 ──
        last_wk = wk.iloc[-1]
        if last_wk['j'] >= 20:
            return None

        # ── 条件3: 周线KDJ底背离 ──
        w_kdj = detect_kdj_divergence(wk, lookback=15)
        d_kdj = detect_kdj_divergence(df, lookback=30)
        if not w_kdj['bullish_divergence']:
            return None

        # ── MACD辅助信号 ──
        w_macd = detect_macd_golden_cross(wk, lookback=10)
        d_macd = detect_macd_golden_cross(df, lookback=15)

        last = df.iloc[-1]

        # 信号强度评分
        score = 0
        signals = []
        if w_kdj['bullish_divergence']:
            score += 3
            signals.append('W:底背离')
        if d_kdj['bullish_divergence']:
            score += 2
            signals.append('D:底背离')
        if w_macd['golden_cross']:
            score += 2
            if w_macd['dif_below_zero']:
                score += 1
                signals.append('W:零下金叉')
            else:
                signals.append('W:金叉')
        if d_macd['golden_cross']:
            score += 1
            signals.append('D:金叉')
        if w_kdj['j_trending_up']:
            score += 1
            signals.append('W:J向上')

        return dict(code=code, name=name,
                    close=last['fwd_close'],
                    d_j=last['j'], d_k=last['k'], d_d=last['d'],
                    w_j=last_wk['j'], w_k=last_wk['k'], w_d=last_wk['d'],
                    w_kdj_div=w_kdj['bullish_divergence'],
                    w_kdj_count=w_kdj['divergence_count'],
                    d_kdj_div=d_kdj['bullish_divergence'],
                    d_kdj_count=d_kdj['divergence_count'],
                    w_macd_gc=w_macd['golden_cross'],
                    w_macd_below0=w_macd['dif_below_zero'],
                    d_macd_gc=d_macd['golden_cross'],
                    d_macd_below0=d_macd['dif_below_zero'],
                    j_trending_up=w_kdj['j_trending_up'],
                    score=score,
                    signal=' | '.join(signals))
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
    print(f"条件: 周线J曾<0 + 当前J<20 + KDJ底背离\n")
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

    # Dedup by code & sort by score desc
    seen = set()
    unique = []
    for r in results:
        if r['code'] not in seen:
            seen.add(r['code'])
            unique.append(r)
    results = unique
    results.sort(key=lambda x: (-x['score'], x['w_j']))

    # Summary
    n_total = len(results)
    n_w_div = sum(1 for r in results if r['w_kdj_div'])
    n_d_div = sum(1 for r in results if r['d_kdj_div'])
    n_w_gc = sum(1 for r in results if r['w_macd_gc'])
    n_d_gc = sum(1 for r in results if r['d_macd_gc'])
    n_high_score = sum(1 for r in results if r['score'] >= 4)

    print(f"\n{'='*70}")
    print(f"Scan complete ({elapsed:.0f}s)")
    print(f"  J曾<0 & 当前J<20:      {n_total}")
    print(f"  周线底背离:            {n_w_div}")
    print(f"  日线底背离:            {n_d_div}")
    print(f"  周线MACD金叉:          {n_w_gc}")
    print(f"  日线MACD金叉:          {n_d_gc}")
    print(f"  高分信号(score>=4):    {n_high_score}")
    print(f"{'='*70}\n")

    # Print table: ticker | 周线J | 日线J | 周线金叉 | 日线金叉
    fmt = '{:<12} {:>8} {:>8} {:>10} {:>10}'
    print(fmt.format('Ticker', 'W-J', 'D-J', 'W-MACD', 'D-MACD'))
    print('-' * 50)

    for r in results:
        w_gc = ('G0' if r['w_macd_below0'] else 'Y') if r['w_macd_gc'] else '-'
        d_gc = ('G0' if r['d_macd_below0'] else 'Y') if r['d_macd_gc'] else '-'
        print(fmt.format(
            r['code'],
            f"{r['w_j']:.1f}",
            f"{r['d_j']:.1f}",
            w_gc,
            d_gc))


if __name__ == '__main__':
    main()

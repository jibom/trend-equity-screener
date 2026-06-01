"""Quick KDJ check: fetch a HK stock, compute daily & weekly KDJ, print results."""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from datetime import date
from data_provider import WindFetcher, forward_adjust
from indicators import compute_daily_mas, compute_kdj, compute_weekly_kdj

STOCK = os.getenv('KDJ_STOCK', '0700.HK')  # Tencent
ASOF = os.getenv('KDJ_ASOF', date.today().strftime('%Y-%m-%d'))


def main():
    print(f"=== KDJ for {STOCK} asof {ASOF} ===\n")

    f = WindFetcher(lookback_days=120)
    df = f.fetch(STOCK, asof=ASOF)
    df = forward_adjust(df).sort_values('date').reset_index(drop=True)
    f.close()

    if df.empty:
        print("No data")
        return

    # ── Daily KDJ (last ~1 month = ~22 trading days) ──
    df = compute_daily_mas(df)
    df = compute_kdj(df)

    print("── Daily KDJ (last 22 trading days) ──")
    recent = df.tail(22)
    for _, r in recent.iterrows():
        cross = ''
        if r.get('kd_golden_cross'):
            cross = ' <<< GOLDEN CROSS'
        elif r.get('kd_death_cross'):
            cross = ' <<< DEATH CROSS'
        j_str = f"{r['j']:7.2f}"
        print(f"  {r['date'].strftime('%Y-%m-%d')}  K={r['k']:6.2f}  D={r['d']:6.2f}  J={j_str}{cross}")

    # ── Weekly KDJ (last ~3 months = ~13 weeks) ──
    weekly = compute_weekly_kdj(df)

    print(f"\n── Weekly KDJ (last 13 weeks) ──")
    recent_w = weekly.tail(13)
    for idx, r in recent_w.iterrows():
        cross = ''
        if r.get('kd_golden_cross'):
            cross = ' <<< GOLDEN CROSS'
        elif r.get('kd_death_cross'):
            cross = ' <<< DEATH CROSS'
        j_str = f"{r['j']:7.2f}"
        print(f"  {idx.strftime('%Y-%m-%d')}  K={r['k']:6.2f}  D={r['d']:6.2f}  J={j_str}{cross}")

    # ── Summary ──
    last = df.iloc[-1]
    print(f"\n── Current state ──")
    print(f"  Date:   {last['date'].strftime('%Y-%m-%d')}")
    print(f"  Close:  {last['fwd_close']:.3f}")
    print(f"  K:      {last['k']:.2f}")
    print(f"  D:      {last['d']:.2f}")
    print(f"  J:      {last['j']:.2f}")
    print(f"  J > D:  {'YES' if last['j'] > last['d'] else 'NO'}")
    print(f"  J > 100: {'YES' if last['j'] > 100 else 'NO'}")
    print(f"  J < 0:   {'YES' if last['j'] < 0 else 'NO'}")

    last_w = weekly.iloc[-1]
    print(f"\n  Weekly K: {last_w['k']:.2f}")
    print(f"  Weekly D: {last_w['d']:.2f}")
    print(f"  Weekly J: {last_w['j']:.2f}")
    print(f"  Weekly J > D: {'YES' if last_w['j'] > last_w['d'] else 'NO'}")


if __name__ == '__main__':
    main()

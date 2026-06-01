"""US stock trend scanner using yfinance.

Fetches US stock data, applies weekly/daily MA alignment logic,
and outputs JSON files to public/data/ for the frontend.
"""
from __future__ import annotations

import json
import os
import sys
import time
import logging
from datetime import date, datetime, timedelta

import yfinance as yf
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, '..')
PUBLIC_DATA = os.path.join(PROJECT_DIR, 'public', 'data')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

sys.path.insert(0, os.path.join(PROJECT_DIR, 'src'))


# US universe: major indices + watchlist
US_WATCHLIST = [
    # Semiconductors
    'NVDA', 'AMD', 'AVGO', 'INTC', 'MU', 'QCOM', 'TXN', 'AMAT', 'LRCX', 'KLAC',
    'MRVL', 'ON', 'MCHP', 'NXPI', 'MPWR', 'SWKS', 'ADI', 'TER', 'ASML', 'ARM',
    'TSM', 'SMCI', 'MRAM', 'SIMO', 'SITM', 'ONTO', 'LSCC', 'COHR', 'VRT', 'FSLR',
    'STM', 'VSH', 'GLW', 'TTMI', 'AEHR', 'CPSH', 'POET', 'NVTS', 'LITE', 'SIDU',
    # Software & Cloud
    'GOOGL', 'GOOG', 'ORCL', 'FTNT', 'TWLO', 'OKTA', 'PL', 'SNPS', 'ROKU',
    # AI & Data Center
    'SMH', 'SOXX', 'SOXL', 'TECL', 'NVDL', 'QQQ', 'QQQM', 'TQQQ', 'QLD',
    # Infrastructure
    'GE', 'GEV', 'CAT', 'DELL', 'HWM', 'EMR', 'ETN', 'CMI', 'ET', 'MOD',
    # Financials
    'GS', 'MS', 'C', 'BK', 'PNC', 'STT', 'TFC', 'BNY', 'RF', 'USB',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'FANG', 'OKE', 'KMI', 'MPC', 'VLO',
    'ENB', 'CNQ', 'CVE', 'TECK', 'RIO', 'BHP', 'VALE',
    # Healthcare
    'LLY', 'MRK', 'JNJ', 'UNH', 'ABBV', 'NVS', 'AMGN', 'ILMN',
    # Consumer
    'AMZN', 'SBUX', 'TGT', 'TJX', 'COST', 'NKE', 'RL', 'CROX', 'BURL',
    # Industrial
    'UNP', 'NSC', 'CSX', 'FDX', 'UPS', 'URI', 'JCI', 'EME', 'FIX', 'FLNC',
    # ETFs & Indices
    'SPY', 'VOO', 'IVV', 'SCHB', 'VTI', 'ITOT', 'DIA', 'IWM', 'QQQ',
    'XLK', 'XLE', 'XLI', 'XLF', 'XLP', 'XLV', 'XLU', 'XLC', 'XLY', 'XLB',
    'VNQ', 'SCHD', 'KRE', 'HYG', 'LQD',
    # Crypto miners
    'RIOT', 'HUT', 'CLSK', 'IREN', 'MARA', 'BTDR', 'CORZ', 'WULF', 'CIFR',
    # Space & Defense
    'RKLB', 'LUNR', 'RDW', 'ASTS', 'SAT', 'BA', 'NOC', 'LMT', 'GD',
    # Other growth
    'APP', 'MSTR', 'CRWD', 'PANW', 'ABNB', 'COIN', 'AFRM', 'SOFI', 'RIVN',
]

MA_WINDOWS_DAILY = [10, 20, 30, 50, 100, 200]
MA_WINDOWS_WEEKLY = [5, 10, 20, 30, 40, 50]


def fetch_us_stock(ticker: str, period: str = '2y') -> pd.DataFrame | None:
    """Fetch US stock daily OHLCV from yfinance."""
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df.empty:
            return None
        df = df.rename(columns={
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume',
        })
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = 'date'
        return df
    except Exception as e:
        logging.warning(f"yfinance fetch failed for {ticker}: {e}")
        return None


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily and weekly MAs."""
    df = df.copy()
    for w in MA_WINDOWS_DAILY:
        df[f'sma{w}'] = df['close'].rolling(w).mean()
    df['vol_ma30'] = df['volume'].rolling(30).mean()
    return df


def compute_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Resample to weekly and compute weekly MAs."""
    w = df[['close']].resample('W-FRI').last().dropna()
    for m in MA_WINDOWS_WEEKLY:
        w[f'ma_{m}w'] = w['close'].rolling(m).mean()
    return w


def classify_trend(row, weekly_row, cfg: dict) -> str:
    """Classify US stock into pool status."""
    close = row['close']
    sma10 = row.get('sma10')
    sma20 = row.get('sma20')
    sma50 = row.get('sma50')
    sma200 = row.get('sma200')

    if pd.isna(sma200):
        return 'NEW'

    above_200 = close > sma200
    daily_aligned = (pd.notna(sma10) and pd.notna(sma20) and pd.notna(sma50)
                     and sma10 > sma20 > sma50)

    # Weekly alignment
    ma_5w = weekly_row.get('ma_5w')
    ma_10w = weekly_row.get('ma_10w')
    ma_20w = weekly_row.get('ma_20w')
    ma_40w = weekly_row.get('ma_40w')
    ma_50w = weekly_row.get('ma_50w')

    weekly_aligned = (pd.notna(ma_5w) and pd.notna(ma_10w) and pd.notna(ma_20w)
                      and pd.notna(ma_40w) and pd.notna(ma_50w)
                      and ma_5w > ma_10w > ma_20w > ma_40w > ma_50w)

    if above_200 and daily_aligned and weekly_aligned:
        return 'STRONGEST'
    elif above_200 and (daily_aligned or weekly_aligned):
        return 'STRONG'
    elif above_200:
        return 'WATCHING'
    elif close > sma50:
        return 'RECOVERING'
    else:
        return 'BROKEN'


def generate_chartdata(ticker: str, df: pd.DataFrame) -> dict:
    """Generate chartdata JSON for frontend."""
    daily = []
    for _, row in df.iterrows():
        ts = int(row.name.timestamp())
        daily.append([
            ts, round(row['open'], 2), round(row['high'], 2),
            round(row['low'], 2), round(row['close'], 2), int(row['volume']),
        ])
    return {'ticker': ticker, 'updated': date.today().isoformat(), 'daily': daily}


def main():
    asof = date.today().strftime('%Y-%m-%d')
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s %(message)s')

    os.makedirs(PUBLIC_DATA, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Load previous results for delta detection
    prev_pools_path = os.path.join(DATA_DIR, 'pools_us.json')
    prev_pools = {}
    if os.path.isfile(prev_pools_path):
        with open(prev_pools_path, encoding='utf-8') as f:
            for item in json.load(f):
                prev_pools[item['ticker']] = item

    results = []
    chartdata_dir = os.path.join(PUBLIC_DATA, 'chartdata', 'us')
    os.makedirs(chartdata_dir, exist_ok=True)

    tickers = list(dict.fromkeys(US_WATCHLIST))  # dedupe
    print(f"Scanning {len(tickers)} US stocks, asof={asof}")

    for i, ticker in enumerate(tickers):
        df = fetch_us_stock(ticker)
        if df is None or len(df) < 60:
            continue

        df = compute_indicators(df)
        weekly_df = compute_weekly(df)

        last = df.iloc[-1]
        last_weekly = weekly_df.iloc[-1] if len(weekly_df) > 0 else pd.Series()

        pool_status = classify_trend(last, last_weekly, {})

        # Delta detection vs previous scan
        prev = prev_pools.get(f'{ticker}.US', {})
        prev_status = prev.get('pool_status', '')
        status_change = 'none'
        if prev_status and prev_status != pool_status:
            status_change = 'upgrade' if pool_status > prev_status else 'downgrade'

        result = {
            'ticker': f'{ticker}.US',
            'market': 'US',
            'pool': 'trend',
            'pool_status': pool_status,
            'prev_pool_status': prev_status,
            'status_change': status_change,
            'first_breakout_date': '',
            'last_breakout_date': asof if pool_status in ('STRONGEST', 'STRONG') else '',
            'breakout_count_60d': 0,
            'from_bottom': prev_status == 'BROKEN' and pool_status in ('STRONGEST', 'STRONG', 'WATCHING'),
            'days_in_trend': 0,
            'last_close': round(float(last['close']), 2),
            'weekly_aligned': bool(last_weekly.get('ma_5w', 0) > last_weekly.get('ma_20w', 0)),
            'daily_aligned': bool(last.get('sma10', 0) > last.get('sma20', 0) > last.get('sma50', 0)),
            'ma_5w': round(float(last_weekly.get('ma_5w', 0)), 2),
            'ma_20w': round(float(last_weekly.get('ma_20w', 0)), 2),
            'ma_40w': round(float(last_weekly.get('ma_40w', 0)), 2),
            'ma_50w': round(float(last_weekly.get('ma_50w', 0)), 2),
            'sma10': round(float(last.get('sma10', 0)), 2) if pd.notna(last.get('sma10')) else None,
            'sma50': round(float(last.get('sma50', 0)), 2) if pd.notna(last.get('sma50')) else None,
            'sma200': round(float(last.get('sma200', 0)), 2) if pd.notna(last.get('sma200')) else None,
        }
        results.append(result)

        # Generate chartdata
        chart = generate_chartdata(ticker, df)
        chart_path = os.path.join(chartdata_dir, f'{ticker}.json')
        with open(chart_path, 'w') as f:
            json.dump(chart, f)

        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(tickers)}")

    # Save pools_us.json
    pools_path = os.path.join(DATA_DIR, 'pools_us.json')
    with open(pools_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Also save to public/data for frontend
    public_pools_path = os.path.join(PUBLIC_DATA, 'pools_us.json')
    with open(public_pools_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Generate alerts
    alerts = []
    for item in results:
        if item['status_change'] == 'upgrade':
            alerts.append({
                'ticker': item['ticker'],
                'event': 'upgrade',
                'from': item['prev_pool_status'],
                'to': item['pool_status'],
                'date': asof,
            })
        elif item['status_change'] == 'downgrade':
            alerts.append({
                'ticker': item['ticker'],
                'event': 'downgrade',
                'from': item['prev_pool_status'],
                'to': item['pool_status'],
                'date': asof,
            })
        if item['from_bottom']:
            alerts.append({
                'ticker': item['ticker'],
                'event': 'NEW',
                'from_bottom': True,
                'date': asof,
            })

    alerts_path = os.path.join(DATA_DIR, 'alerts_us.json')
    with open(alerts_path, 'w', encoding='utf-8') as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2)
    public_alerts_path = os.path.join(PUBLIC_DATA, 'alerts_us.json')
    with open(public_alerts_path, 'w', encoding='utf-8') as f:
        json.dump(alerts, f, ensure_ascii=False, indent=2)

    # Update chartdata index
    us_tickers = [r['ticker'].replace('.US', '') for r in results]
    hk_chartdata = os.path.join(PUBLIC_DATA, 'chartdata', 'hk')
    hk_tickers = [f.replace('.json', '') for f in os.listdir(hk_chartdata) if f.endswith('.json') and f != 'index.json'] if os.path.isdir(hk_chartdata) else []
    index_data = {
        'us': us_tickers,
        'hk': hk_tickers,
        'updated': datetime.utcnow().isoformat() + '+00:00',
    }
    with open(os.path.join(PUBLIC_DATA, 'chartdata', 'index.json'), 'w') as f:
        json.dump(index_data, f)

    # Summary
    from collections import Counter
    statuses = Counter(r['pool_status'] for r in results)
    print(f"\n=== US Scan Summary ({asof}) ===")
    for s, c in statuses.most_common():
        print(f"  {s}: {c}")
    print(f"  Alerts: {len(alerts)}")


if __name__ == '__main__':
    main()

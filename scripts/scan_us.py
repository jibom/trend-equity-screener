#!/usr/bin/env python3
"""
US Stock Screener using EODHD API + yfinance for market cap.

Strategy:
  Phase 1 — EODHD bulk: symbol list + last-day OHLCV
             Filter by exchange, type, daily turnover ≥ $200M
  Phase 2 — EODHD EOD: 2y daily history (parallel), compute SMAs + volume
  Phase 3 — yfinance: market cap for survivors (亿 USD)
"""

import json
import sys
import os
import time
import datetime as dt
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import requests
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

try:
    import pandas as pd
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "-q"])
    import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_FILE = os.path.join(ROOT_DIR, "public", "data", "us.json")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ── EODHD API ──
EODHD_API_KEY = "6a10e8411d06d1.41490389"
EODHD_BASE_URL = "https://eodhd.com/api"
EODHD_MAX_RETRIES = 3
KEEP_EXCHANGES = {"NASDAQ", "NYSE", "NYSE ARCA", "BATS", "AMEX", "NYSE MKT"}
KEEP_TYPES = {"Common Stock", "ADR", "ETF", "ETN"}

# ── Screener params ──
MIN_DAILY_DOLLAR_VOL = 200_000_000  # Phase 1: $200M USD daily turnover
LOOKBACK_DAYS = 730                 # 2y for SMA200
US_WORKERS = 20                     # parallel EODHD downloads


def calc_sma(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


# ── EODHD helpers ──

def _eodhd_get(url, timeout=30):
    for attempt in range(EODHD_MAX_RETRIES):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return None


def fetch_us_symbol_list():
    print("[Phase 1] Fetching US symbol list ...", file=sys.stderr)
    url = f"{EODHD_BASE_URL}/exchange-symbol-list/US?api_token={EODHD_API_KEY}&fmt=json"
    data = _eodhd_get(url, timeout=60)
    if not data:
        raise RuntimeError("Failed to fetch US symbol list from EODHD")
    df = pd.DataFrame(data)
    df = df[df["Type"].isin(KEEP_TYPES) & df["Exchange"].isin(KEEP_EXCHANGES)]
    df = df[~df["Code"].str.contains(r"-WS|-WT|\.WS", regex=True, na=False)]
    print(f"  Common Stock + ADR + ETF/ETN (major exchanges): {len(df)}", file=sys.stderr)
    return df


def fetch_us_bulk_last_day():
    print("[Phase 1] Fetching bulk last-day data ...", file=sys.stderr)
    url = f"{EODHD_BASE_URL}/eod-bulk-last-day/US?api_token={EODHD_API_KEY}&fmt=json"
    data = _eodhd_get(url, timeout=60)
    if not data:
        raise RuntimeError("Failed to fetch bulk last-day data from EODHD")
    df = pd.DataFrame(data)
    df = df[(df["close"] > 0) & (df["volume"] > 0)].copy()
    df["dollar_vol"] = df["close"] * df["volume"]
    print(f"  Bulk last-day: {len(df)} tickers", file=sys.stderr)
    return df


def build_initial_pool(meta_df, bulk_df):
    pool = bulk_df.merge(
        meta_df[["Code", "Name", "Exchange", "Type"]],
        left_on="code", right_on="Code", how="inner",
    )
    pool = pool[pool["dollar_vol"] >= MIN_DAILY_DOLLAR_VOL]
    pool = pool.sort_values("dollar_vol", ascending=False).reset_index(drop=True)
    print(f"  After turnover filter (>=${MIN_DAILY_DOLLAR_VOL/1e6:.0f}M): {len(pool)}", file=sys.stderr)
    if not pool.empty:
        print(f"  Latest trade date: {pool['date'].iloc[0]}", file=sys.stderr)
    return pool


def _fetch_eod_history(code, start_date, end_date):
    url = (
        f"{EODHD_BASE_URL}/eod/{code}.US?api_token={EODHD_API_KEY}"
        f"&from={start_date}&to={end_date}&period=d&fmt=json"
    )
    data = _eodhd_get(url, timeout=30)
    if not data or len(data) < 200:
        return None
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def fetch_histories_parallel(codes, start_date, end_date):
    print(f"[Phase 2] Fetching 2y history for {len(codes)} tickers ...", file=sys.stderr)
    t0 = time.time()
    histories = {}
    with ThreadPoolExecutor(max_workers=US_WORKERS) as ex:
        futs = {ex.submit(_fetch_eod_history, c, start_date, end_date): c for c in codes}
        for n, f in enumerate(as_completed(futs), 1):
            code = futs[f]
            df = f.result()
            if df is not None:
                histories[code] = df
            if n % 100 == 0:
                print(f"  {n}/{len(codes)} done ({time.time()-t0:.0f}s)", file=sys.stderr)
    print(f"  History fetched: {len(histories)} tickers ({time.time()-t0:.0f}s)", file=sys.stderr)
    return histories


def analyze_from_history(code, df):
    if len(df) < 200:
        return None

    closes = df["adjusted_close"].tolist()
    volumes = df["volume"].tolist()

    last_price = closes[-1]
    prev_close = closes[-2] if len(closes) >= 2 else last_price

    if last_price < 10:
        return None

    sma10 = calc_sma(closes, 10)
    sma20 = calc_sma(closes, 20)
    sma30 = calc_sma(closes, 30)
    sma50 = calc_sma(closes, 50)
    sma100 = calc_sma(closes, 100)
    sma200 = calc_sma(closes, 200)

    if not all([sma10, sma20, sma30, sma50, sma100, sma200]):
        return None

    if sma10 <= sma20 or sma20 <= sma50 or sma50 <= sma100 or sma100 <= sma200:
        return None
    if last_price <= sma30:
        return None

    avg_vol_10 = calc_sma(volumes, 10)
    avg_vol_60 = calc_sma(volumes, 60)
    avg_vol_90 = calc_sma(volumes, 90)
    daily_volume = volumes[-1]

    if not all([avg_vol_10, avg_vol_60, avg_vol_90]):
        return None
    if daily_volume < 500_000:
        return None
    if avg_vol_10 < 500_000:
        return None
    if avg_vol_60 < 500_000:
        return None
    if avg_vol_90 < 500_000:
        return None

    recent_closes = closes[-20:]
    recent_volumes = volumes[-20:]
    avg_trading_value = sum(c * v for c, v in zip(recent_closes, recent_volumes)) / len(recent_closes)

    if avg_trading_value < 100_000_000:
        return None

    change = last_price - prev_close
    change_pct = (change / prev_close * 100) if prev_close != 0 else 0

    if len(closes) >= 6:
        price_5d_ago = closes[-6]
        change_5d_pct = ((last_price - price_5d_ago) / price_5d_ago * 100) if price_5d_ago != 0 else 0
    else:
        change_5d_pct = 0

    return {
        "ticker": code,
        "name": "",
        "price": round(last_price, 2),
        "change": round(change, 2),
        "changePercent": round(change_pct, 2),
        "change5dPercent": round(change_5d_pct, 2),
        "marketCap": 0,
        "volume": int(daily_volume),
        "avgVolume10d": int(avg_vol_10),
        "avgVolume60d": int(avg_vol_60),
        "avgVolume90d": int(avg_vol_90),
        "avgTradingValue": int(avg_trading_value),
        "sma10": round(sma10, 2),
        "sma20": round(sma20, 2),
        "sma30": round(sma30, 2),
        "sma50": round(sma50, 2),
        "sma100": round(sma100, 2),
        "sma200": round(sma200, 2),
        "sector": "",
        "indices": [],
    }


def fetch_market_cap_yfinance(ticker, retries=3):
    """Fetch market cap + name + sector via yfinance"""
    for attempt in range(retries):
        try:
            tk = yf.Ticker(ticker)
            market_cap = 0
            name = ticker
            sector = ''

            try:
                fi = tk.fast_info
                market_cap = getattr(fi, 'market_cap', 0) or 0
            except Exception:
                pass

            try:
                info = tk.info
                name = info.get('shortName', '') or info.get('longName', ticker)
                sector = info.get('sector', '') or ''
            except Exception:
                pass

            if market_cap > 0 or name != ticker:
                return {"market_cap": market_cap, "name": name, "sector": sector}

            if attempt < retries - 1:
                time.sleep(2)
                continue
            return {"market_cap": market_cap, "name": name, "sector": sector}
        except Exception:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return {"market_cap": 0, "name": ticker, "sector": ""}

    return {"market_cap": 0, "name": ticker, "sector": ""}


def main():
    print("[US Screener] Starting EODHD + yfinance scan ...", file=sys.stderr)
    start_time = time.time()

    # ─── Phase 1: Bulk data + turnover filter ───
    try:
        meta_df = fetch_us_symbol_list()
        bulk_df = fetch_us_bulk_last_day()
        pool_df = build_initial_pool(meta_df, bulk_df)
    except Exception as e:
        print(f"[US Screener] ERROR in Phase 1: {e}", file=sys.stderr)
        return

    if pool_df.empty:
        print("[US Screener] No stocks pass initial filter", file=sys.stderr)
        return

    total_universe = len(meta_df)

    name_map = {}
    for _, row in pool_df.iterrows():
        name_map[row["code"]] = row.get("Name", "")

    # ─── Phase 2: 2y history + technical filters ───
    latest_date = pd.to_datetime(pool_df["date"].iloc[0]).date()
    start_date = latest_date - dt.timedelta(days=LOOKBACK_DAYS)
    codes = pool_df["code"].tolist()

    histories = fetch_histories_parallel(codes, start_date, latest_date)

    technical_passers = []
    for code, df in histories.items():
        result = analyze_from_history(code, df)
        if result:
            if not result["name"]:
                result["name"] = name_map.get(code, code)
            technical_passers.append(result)

    elapsed = time.time() - start_time
    print(f"[Phase 2] Complete: {len(technical_passers)} pass technical filters ({elapsed:.0f}s)", file=sys.stderr)

    # ─── Phase 3: Market cap via yfinance ───
    print(f"[Phase 3] Fetching market cap (yfinance) for {len(technical_passers)} stocks ...", file=sys.stderr)

    passing = []

    for idx, item in enumerate(technical_passers):
        code = item["ticker"]
        meta = fetch_market_cap_yfinance(code)
        market_cap = meta["market_cap"]

        if market_cap == 0:
            continue

        item["marketCap"] = int(market_cap)
        if meta.get("name"):
            item["name"] = meta["name"]
        if meta.get("sector"):
            item["sector"] = meta["sector"]
        passing.append(item)

        print(f"  [Pass] {item['ticker']} ({item['name']}) ${item['price']:.2f} MCap={market_cap/1e8:.1f}亿USD Sector={item['sector']}", file=sys.stderr)

        if (idx + 1) % 20 == 0:
            elapsed = time.time() - start_time
            print(f"  [Phase 3] {idx+1}/{len(technical_passers)} ({elapsed:.0f}s)", file=sys.stderr)
        time.sleep(0.15)

    passing.sort(key=lambda x: x["marketCap"], reverse=True)

    elapsed = time.time() - start_time
    print(f"\n[US Screener] ═══════════════════════════════════════", file=sys.stderr)
    print(f"[US Screener] Complete. {len(passing)} stocks pass all filters.", file=sys.stderr)
    print(f"[US Screener] Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)", file=sys.stderr)

    output = {
        "stocks": passing,
        "totalUniverse": total_universe,
        "totalPassing": len(passing),
        "lastUpdated": datetime.now().isoformat(),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f)
    print(f"[Done] Output: {OUTPUT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()

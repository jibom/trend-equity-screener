#!/usr/bin/env python3
"""
Hong Kong Stock Screener using Wind Database + yfinance for market cap.

Strategy (optimized — filter early, fetch EOD last):
  Phase 1 — Wind DB: fetch 60d EOD for all ORD, liquidity filter (avg value >= HK$5000万)
  Phase 2 — Wind DB: fetch 2y EOD only for liquid survivors, compute SMA alignment
  Phase 3 — yfinance: market cap for SMA survivors

Pool System (new):
  After Phase 2, also run pool state machine on all stocks with history data.
  Output: public/data/hk.json (original), data/pools_hk.json + data/alerts_hk.json (new)
"""

import json
import sys
import os
import time
import datetime as dt
from datetime import datetime

try:
    import pymysql
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pymysql", "-q"])
    import pymysql

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

# Import pool manager
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pool_manager import (
    calc_sma, resample_to_weekly, check_6m_high, check_from_bottom,
    check_weekly_alignment, check_daily_alignment, rate_trend_stock,
    determine_status_change, count_trading_days_since,
    run_pool_state_machine, load_pools, save_pools, load_themes,
    generate_alerts, save_alerts,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_FILE = os.path.join(ROOT_DIR, "public", "data", "hk.json")
POOL_FILE = os.path.join(ROOT_DIR, "data", "pools_hk.json")
ALERT_FILE = os.path.join(ROOT_DIR, "data", "alerts_hk.json")
THEME_FILE = os.path.join(ROOT_DIR, "data", "themes_hk.json")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
os.makedirs(os.path.dirname(POOL_FILE), exist_ok=True)

# ── Wind DB ──
DB_CONFIG = {
    "host": "rm-uf62imd2xjxj647jho.mysql.rds.aliyuncs.com",
    "user": "yangdong_gf",
    "password": "4S7Q4pNUzh",
    "database": "jianxin",
    "port": 3306,
    "charset": "utf8mb4",
}

# ── Screener params ──
MIN_AVG_VALUE = 50_000_000       # HK$5000万 avg trading value
LOOKBACK_DAYS = 500               # ~2y for SMA200
VOL_LOOKBACK_DAYS = 90            # ~60 trading days for volume calc


# ── Ticker format conversion ──

def wind_to_pool_ticker(wind_code):
    """Convert Wind code to pool ticker format: 00700.HK -> 0700.HK"""
    parts = wind_code.split(".")
    if len(parts) != 2:
        return wind_code
    code = parts[0]
    # Wind 5-digit -> Yahoo 4-digit: 00700 -> 0700
    if len(code) == 5 and code.startswith("0"):
        return f"{code[1:]}.{parts[1]}"
    return wind_code


def pool_to_wind_ticker(pool_ticker):
    """Convert pool ticker format back to Wind code: 0700.HK -> 00700.HK"""
    parts = pool_ticker.split(".")
    if len(parts) != 2:
        return pool_ticker
    code = parts[0]
    # Pad to 5 digits: 0700 -> 00700
    if code.isdigit():
        return f"{code.zfill(5)}.{parts[1]}"
    return pool_ticker


# ── Wind DB helpers ──

def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


def fetch_hk_stock_list(conn):
    print("[Phase 1] Fetching HK ORD stock list ...", file=sys.stderr)
    cur = conn.cursor()
    cur.execute("SELECT S_INFO_WINDCODE, S_INFO_NAME FROM hksharedescription WHERE SECURITYTYPE='ORD'")
    rows = cur.fetchall()
    cur.close()
    df = pd.DataFrame(rows, columns=["S_INFO_WINDCODE", "S_INFO_NAME"])
    print(f"  ORD stocks: {len(df)}", file=sys.stderr)
    return df


def filter_by_liquidity(conn, ord_codes):
    """Phase 1: Fetch 60d EOD, filter by avg trading value >= HK$5000万."""
    print(f"[Phase 1] Filtering by liquidity (avg value >= HK${MIN_AVG_VALUE/1e6:.0f}M) ...", file=sys.stderr)
    t0 = time.time()

    start_dt = (dt.date.today() - dt.timedelta(days=VOL_LOOKBACK_DAYS)).strftime("%Y%m%d")
    chunk_size = 500
    all_rows = []

    for i in range(0, len(ord_codes), chunk_size):
        chunk = ord_codes[i:i + chunk_size]
        placeholders = ",".join(["%s"] * len(chunk))
        sql = (
            "SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_CLOSE, S_DQ_VOLUME "
            "FROM hkshareeodprices "
            f"WHERE TRADE_DT >= %s AND S_INFO_WINDCODE IN ({placeholders})"
        )
        cur = conn.cursor()
        cur.execute(sql, [start_dt] + chunk)
        rows = cur.fetchall()
        cur.close()
        all_rows.extend(rows)

    if not all_rows:
        print("  No EOD data returned", file=sys.stderr)
        return set()

    df = pd.DataFrame(all_rows, columns=["code", "trade_dt", "close", "volume"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["close", "volume"])
    df["value"] = df["close"] * df["volume"]

    avg_value = df.groupby("code")["value"].mean()
    liquid_codes = set(avg_value[avg_value >= MIN_AVG_VALUE].index)

    print(f"  After liquidity filter: {len(liquid_codes)} / {df['code'].nunique()} stocks ({time.time()-t0:.0f}s)", file=sys.stderr)
    return liquid_codes


def fetch_eod_for_sma(conn, codes):
    """Phase 2: Fetch 2y EOD for survivors."""
    print(f"[Phase 2] Fetching 2y EOD for {len(codes)} survivors ...", file=sys.stderr)
    t0 = time.time()

    start_dt = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    codes_list = list(codes)
    chunk_size = 300
    all_dfs = []

    for i in range(0, len(codes_list), chunk_size):
        chunk = codes_list[i:i + chunk_size]
        placeholders = ",".join(["%s"] * len(chunk))
        sql = (
            "SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, "
            "S_DQ_VOLUME, S_DQ_AMOUNT, S_DQ_ADJCLOSE_BACKWARD "
            "FROM hkshareeodprices "
            f"WHERE TRADE_DT >= %s AND S_INFO_WINDCODE IN ({placeholders})"
        )
        cur = conn.cursor()
        cur.execute(sql, [start_dt] + chunk)
        rows = cur.fetchall()
        cur.close()
        if rows:
            all_dfs.append(pd.DataFrame(rows, columns=[
                "code", "trade_dt", "open", "high", "low", "close",
                "volume", "amount", "adj_close",
            ]))

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df["trade_dt"] = pd.to_datetime(df["trade_dt"], format="%Y%m%d")
    for c in ["open", "high", "low", "close", "volume", "amount", "adj_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close", "adj_close"]).copy()

    print(f"  EOD rows: {len(df):,} ({time.time()-t0:.0f}s)", file=sys.stderr)
    return df


def analyze_from_wind(code, group_df):
    """Original SMA alignment analysis for public/data/hk.json output."""
    g = group_df.sort_values("trade_dt").reset_index(drop=True)
    if len(g) < 200:
        return None

    closes = g["adj_close"].tolist()
    volumes = g["volume"].tolist()

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

    avg_vol_60 = calc_sma(volumes, 60)
    avg_vol_90 = calc_sma(volumes, 90)

    if not all([avg_vol_60, avg_vol_90]):
        return None
    if avg_vol_60 < 500_000:
        return None
    if avg_vol_90 < 500_000:
        return None

    recent_closes = closes[-60:]
    recent_volumes = volumes[-60:]
    avg_trading_value = sum(c * v for c, v in zip(recent_closes, recent_volumes)) / len(recent_closes)

    if avg_trading_value < MIN_AVG_VALUE:
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


def fetch_market_caps_yfinance(windcodes):
    """Phase 3: Fetch market cap + sector via yfinance. Wind: 00700.HK -> Yahoo: 0700.HK."""
    print(f"[Phase 3] Fetching market cap + sector (yfinance) for {len(windcodes)} stocks ...", file=sys.stderr)
    t0 = time.time()

    mcap_map = {}
    sector_map = {}
    for wc in windcodes:
        parts = wc.split(".")
        code = parts[0]
        # Wind 5-digit -> Yahoo 4-digit: 00700.HK -> 0700.HK
        if len(code) == 5 and code.startswith("0"):
            yf_ticker = f"{code[1:]}.{parts[1]}"
        else:
            yf_ticker = wc
        try:
            tk = yf.Ticker(yf_ticker)
            fi = tk.fast_info
            mcap = getattr(fi, "market_cap", 0) or 0
            mcap_map[wc] = mcap
            try:
                sector_map[wc] = tk.info.get("sector", "") or ""
            except Exception:
                sector_map[wc] = ""
        except Exception:
            mcap_map[wc] = 0
            sector_map[wc] = ""
        time.sleep(0.2)

    found = sum(1 for v in mcap_map.values() if v > 0)
    print(f"  Market cap fetched: {found}/{len(windcodes)} ({time.time()-t0:.0f}s)", file=sys.stderr)
    return mcap_map, sector_map


def run_pool_system(eod_df, today_str, prev_pools, themes, name_map, bootstrap=False):
    """Run pool state machine on all stocks with history data.

    Args:
        eod_df: DataFrame with EOD data for all stocks
        today_str: today's date string YYYY-MM-DD
        prev_pools: dict of previous pool entries keyed by pool ticker
        themes: dict of theme info keyed by pool ticker
        name_map: dict of Wind code -> stock name
        bootstrap: if True, fill pools from scratch (first run)

    Returns:
        (pools_data, alerts) tuple
    """
    print(f"[Pool] Running pool state machine for {eod_df['code'].nunique()} stocks (bootstrap={bootstrap}) ...", file=sys.stderr)
    t0 = time.time()
    pools_data = {}

    for code, group in eod_df.groupby("code"):
        g = group.sort_values("trade_dt").reset_index(drop=True)
        closes = g["adj_close"].tolist()
        dates = g["trade_dt"].tolist()
        date_strs = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
                     for d in dates]

        weekly = resample_to_weekly(date_strs, closes)

        ticker = wind_to_pool_ticker(code)
        prev_entry = prev_pools.get(ticker)
        themes_info = themes.get(ticker, {})

        result = run_pool_state_machine(
            ticker=ticker, market="HK",
            closes=closes, dates=date_strs, weekly_closes=weekly,
            prev_entry=prev_entry, today_str=today_str,
            themes_info=themes_info, bootstrap=bootstrap,
        )
        if result:
            pools_data[ticker] = result

    # Keep pool stocks that weren't in this scan's EOD data (still in pool but no data today)
    for ticker, entry in prev_pools.items():
        if ticker not in pools_data and entry.get("pool") in ("breakout", "trend"):
            entry["last_update_date"] = today_str

    alerts = generate_alerts(pools_data, today_str)

    elapsed = time.time() - t0
    breakout_count = sum(1 for e in pools_data.values() if e.get("pool") == "breakout")
    trend_count = sum(1 for e in pools_data.values() if e.get("pool") == "trend")
    print(f"[Pool] Breakout: {breakout_count}, Trend: {trend_count}, Alerts: {len(alerts)} ({elapsed:.0f}s)", file=sys.stderr)

    return pools_data, alerts


def main():
    print("[HK Screener] Starting Wind DB + yfinance scan ...", file=sys.stderr)
    start_time = time.time()

    # ── Load pool state and themes ──
    prev_pools = load_pools(POOL_FILE)
    themes = load_themes(THEME_FILE)
    print(f"[Pool] Loaded {len(prev_pools)} pool entries, {len(themes)} theme entries", file=sys.stderr)

    try:
        conn = get_db_connection()
    except Exception as e:
        print(f"[HK Screener] ERROR: Cannot connect to Wind DB: {e}", file=sys.stderr)
        return

    try:
        # ─── Phase 1: Liquidity filter (Wind DB 60d) ───
        desc_df = fetch_hk_stock_list(conn)
        name_map = dict(zip(desc_df["S_INFO_WINDCODE"], desc_df["S_INFO_NAME"]))
        total_universe = len(desc_df)

        liquid_codes = filter_by_liquidity(conn, desc_df["S_INFO_WINDCODE"].tolist())
        if not liquid_codes:
            print("[HK Screener] No stocks pass liquidity filter", file=sys.stderr)
            return

        # Include pool stocks in EOD fetch
        pool_wind_codes = set()
        for ticker in prev_pools:
            wind_code = pool_to_wind_ticker(ticker)
            pool_wind_codes.add(wind_code)
        all_codes = liquid_codes | pool_wind_codes

        if len(all_codes) > len(liquid_codes):
            print(f"  Including {len(pool_wind_codes)} pool stocks in EOD fetch (total: {len(all_codes)})", file=sys.stderr)

        # ─── Phase 2: SMA computation (Wind DB 2y EOD) ───
        eod_df = fetch_eod_for_sma(conn, all_codes)
        if eod_df.empty:
            print("[HK Screener] No EOD data returned", file=sys.stderr)
            return

        # Original SMA analysis for public/data/hk.json
        sma_passers = []
        for code, group in eod_df.groupby("code"):
            if code not in liquid_codes:
                continue  # Skip pool-only stocks for original output
            result = analyze_from_wind(code, group)
            if result:
                result["name"] = name_map.get(code, code)
                sma_passers.append(result)

        print(f"[Phase 2] SMA passers: {len(sma_passers)}", file=sys.stderr)

        # ─── Phase 3: Market cap (yfinance) ───
        windcodes = [item["ticker"] for item in sma_passers]
        mcap_map, sector_map = fetch_market_caps_yfinance(windcodes)

        passing = []
        for item in sma_passers:
            mcap = mcap_map.get(item["ticker"], 0)
            if mcap > 0:
                item["marketCap"] = int(mcap)
                if sector_map.get(item["ticker"]):
                    item["sector"] = sector_map[item["ticker"]]
                passing.append(item)
                print(f"  [Pass] {item['ticker']} ({item['name']}) HK${item['price']:.2f} MCap={float(mcap)/1e8:.1f}亿HKD", file=sys.stderr)

    finally:
        conn.close()

    passing.sort(key=lambda x: x["marketCap"], reverse=True)

    elapsed = time.time() - start_time
    print(f"\n[HK Screener] ========================================", file=sys.stderr)
    print(f"[HK Screener] Complete. {len(passing)} stocks pass all filters.", file=sys.stderr)
    print(f"[HK Screener] Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)", file=sys.stderr)

    # ── Save original output ──
    output = {
        "stocks": passing,
        "totalUniverse": total_universe,
        "totalPassing": len(passing),
        "lastUpdated": datetime.now().isoformat(),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f)
    print(f"[Done] Output: {OUTPUT_FILE}", file=sys.stderr)

    # ── Run pool system ──
    today_str = dt.date.today().isoformat()
    # Bootstrap mode: first run with empty pool → fill from scratch
    bootstrap = len(prev_pools) == 0
    pools_data, alerts = run_pool_system(eod_df, today_str, prev_pools, themes, name_map, bootstrap=bootstrap)

    save_pools(POOL_FILE, pools_data)
    print(f"[Done] Pool: {POOL_FILE} ({len(pools_data)} entries)", file=sys.stderr)

    save_alerts(ALERT_FILE, alerts)
    print(f"[Done] Alerts: {ALERT_FILE} ({len(alerts)} alerts)", file=sys.stderr)


if __name__ == "__main__":
    main()

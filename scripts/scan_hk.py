#!/usr/bin/env python3
"""
Hong Kong Stock Screener using Wind Database (Aliyun MySQL).
Replaces yfinance batch downloads with Wind DB for EOD data.
Market cap from Wind DB derivative indicator table, yfinance fallback.

Strategy:
  Phase 1 — Wind DB: stock descriptions + 2y EOD prices (one SQL query)
             Filter by security type (ORD), compute SMAs + volume
  Phase 2 — Market cap: Wind DB derivative indicator / yfinance fallback
             Apply market cap filter
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
    import pandas as pd
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "pandas", "-q"])
    import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
OUTPUT_FILE = os.path.join(ROOT_DIR, "public", "data", "hk.json")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

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
MCAP_THRESHOLD = 1_000_000_000   # HK$1B
LOOKBACK_DAYS = 500               # ~2y for SMA200


def calc_sma(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


def fetch_hk_stock_list(conn):
    print("[Phase 1] Fetching HK stock list ...", file=sys.stderr)
    cur = conn.cursor()
    cur.execute("SELECT S_INFO_WINDCODE, S_INFO_NAME FROM hksharedescription WHERE SECURITYTYPE='ORD'")
    rows = cur.fetchall()
    cur.close()
    df = pd.DataFrame(rows, columns=["S_INFO_WINDCODE", "S_INFO_NAME"])
    print(f"  ORD stocks: {len(df)}", file=sys.stderr)
    return df


def fetch_hk_eod_prices(conn, ord_codes):
    """Fetch 2y EOD prices for all ORD stocks (single query)"""
    start_dt = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    end_dt = dt.date.today().strftime("%Y%m%d")
    print(f"[Phase 1] Fetching EOD prices {start_dt} ~ {end_dt} ...", file=sys.stderr)
    t0 = time.time()

    placeholders = ",".join(["%s"] * len(ord_codes))
    sql = (
        "SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, "
        "S_DQ_VOLUME, S_DQ_AMOUNT, S_DQ_ADJCLOSE_BACKWARD "
        "FROM hkshareeodprices "
        f"WHERE TRADE_DT >= %s AND S_INFO_WINDCODE IN ({placeholders})"
    )
    cur = conn.cursor()
    cur.execute(sql, [start_dt] + ord_codes)
    rows = cur.fetchall()
    cur.close()

    df = pd.DataFrame(rows, columns=[
        "code", "trade_dt", "open", "high", "low", "close",
        "volume", "amount", "adj_close",
    ])
    if df.empty:
        return df

    df["trade_dt"] = pd.to_datetime(df["trade_dt"], format="%Y%m%d")
    for c in ["open", "high", "low", "close", "volume", "amount", "adj_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close", "adj_close"]).copy()

    print(f"  EOD rows: {len(df):,} ({time.time()-t0:.0f}s)", file=sys.stderr)
    return df


def analyze_from_wind(code, group_df):
    """Compute SMA and volume filters from Wind DB DataFrame"""
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

    if avg_trading_value < 50_000_000:
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


def fetch_market_caps(conn, windcodes):
    """Get latest market cap. Try Wind DB first, fall back to yfinance."""
    print(f"[Phase 2] Fetching market cap for {len(windcodes)} stocks ...", file=sys.stderr)

    # Try Wind DB derivative indicator table
    try:
        placeholders = ",".join(["%s"] * len(windcodes))
        cur = conn.cursor()
        cur.execute(f"""
            SELECT S_INFO_WINDCODE, S_VAL_MV
            FROM hkshareeodderivativeindicator
            WHERE TRADE_DT = (SELECT MAX(TRADE_DT) FROM hkshareeodderivativeindicator)
              AND S_INFO_WINDCODE IN ({placeholders})
        """, windcodes)
        rows = cur.fetchall()
        cur.close()
        if rows:
            mcap_map = {r[0]: r[1] for r in rows if r[1]}
            if mcap_map:
                # Check unit: if max mcap < 1T (1e12), likely in HKD
                max_mcap = max(mcap_map.values())
                print(f"  Wind DB market cap: {len(mcap_map)} stocks, max={max_mcap/1e9:.1f}B", file=sys.stderr)
                return mcap_map
    except Exception as e:
        print(f"  Wind DB market cap failed: {e}", file=sys.stderr)

    # Fallback to yfinance
    print("  Falling back to yfinance ...", file=sys.stderr)
    try:
        import yfinance as yf
    except ImportError:
        print("  yfinance not available, skipping market cap", file=sys.stderr)
        return {}

    mcap_map = {}
    for wc in windcodes:
        parts = wc.split(".")
        # Wind: 00700.HK → Yahoo: 0700.HK
        code = parts[0]
        if len(code) == 5 and code.startswith("0"):
            yf_ticker = f"{code[1:]}.{parts[1]}"
        else:
            yf_ticker = wc
        try:
            tk = yf.Ticker(yf_ticker)
            fi = tk.fast_info
            mcap = getattr(fi, "market_cap", 0) or 0
            mcap_map[wc] = mcap
        except Exception:
            mcap_map[wc] = 0
        time.sleep(0.3)

    return mcap_map


def main():
    print("[HK Screener] Starting Wind DB-based scan ...", file=sys.stderr)
    start_time = time.time()

    try:
        conn = get_db_connection()
    except Exception as e:
        print(f"[HK Screener] ERROR: Cannot connect to Wind DB: {e}", file=sys.stderr)
        return

    try:
        # ─── Phase 1: Stock list + EOD data ───
        desc_df = fetch_hk_stock_list(conn)
        name_map = dict(zip(desc_df["S_INFO_WINDCODE"], desc_df["S_INFO_NAME"]))
        ord_codes = desc_df["S_INFO_WINDCODE"].tolist()
        total_universe = len(ord_codes)

        eod_df = fetch_hk_eod_prices(conn, ord_codes)
        if eod_df.empty:
            print("[HK Screener] No EOD data returned", file=sys.stderr)
            return

        # Compute SMAs
        technical_passers = []
        for code, group in eod_df.groupby("code"):
            result = analyze_from_wind(code, group)
            if result:
                result["name"] = name_map.get(code, code)
                technical_passers.append(result)

        elapsed = time.time() - start_time
        print(f"[Phase 1] Complete: {len(technical_passers)} pass technical filters ({elapsed:.0f}s)", file=sys.stderr)

        # ─── Phase 2: Market cap ───
        windcodes = [item["ticker"] for item in technical_passers]
        mcap_map = fetch_market_caps(conn, windcodes)

        passing = []
        for item in technical_passers:
            wc = item["ticker"]
            mcap = mcap_map.get(wc, 0)
            if mcap == 0:
                continue
            if mcap < MCAP_THRESHOLD:
                continue
            item["marketCap"] = int(mcap)
            passing.append(item)
            print(f"  [Pass] {item['ticker']} ({item['name']}) HK${item['price']:.2f} MCap={mcap/1e9:.1f}B", file=sys.stderr)

    finally:
        conn.close()

    passing.sort(key=lambda x: x["marketCap"], reverse=True)

    elapsed = time.time() - start_time
    print(f"\n[HK Screener] ═══════════════════════════════════════", file=sys.stderr)
    print(f"[HK Screener] Complete. {len(passing)} stocks pass all filters.", file=sys.stderr)
    print(f"[HK Screener] Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)", file=sys.stderr)

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

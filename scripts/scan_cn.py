#!/usr/bin/env python3
"""
A-Share Stock Screener using Wind Database (Aliyun MySQL).

Criteria:
  - Common Stock only (filter from asharedescription)
  - Market cap > 100亿 RMB (from ashareeodderivativeindicator)
  - Volume in top 20% of A-share market
  - SMA trend alignment: 10 > 20 > 50 > 100 > 200
  - Price > SMA30

Strategy:
  Phase 1 — Wind DB: stock descriptions + 2y EOD prices (one SQL query)
             Compute avg volume, find top 20% threshold, filter
  Phase 2 — Compute SMAs for top 20% volume stocks
  Phase 3 — Market cap filter from Wind DB derivative indicator table
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
OUTPUT_FILE = os.path.join(ROOT_DIR, "public", "data", "cn.json")

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
MCAP_THRESHOLD = 10_000_000_000   # 100亿 RMB
VOL_PCT_THRESHOLD = 0.20          # Volume top 20%
LOOKBACK_DAYS = 500               # ~2y for SMA200


def calc_sma(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def get_db_connection():
    return pymysql.connect(**DB_CONFIG)


def fetch_cn_stock_list(conn):
    """Get A-share common stock descriptions"""
    print("[Phase 1] Fetching A-share stock list ...", file=sys.stderr)
    cur = conn.cursor()
    cur.execute("""
        SELECT S_INFO_WINDCODE, S_INFO_NAME, S_INFO_EXCHMARKET
        FROM asharedescription
        WHERE S_INFO_EXCHMARKET IN ('SSE', 'SZSE')
    """)
    rows = cur.fetchall()
    cur.close()
    df = pd.DataFrame(rows, columns=["S_INFO_WINDCODE", "S_INFO_NAME", "exchange"])
    print(f"  Common stocks (SSE+SZSE): {len(df)}", file=sys.stderr)
    return df


def fetch_cn_eod_prices(conn, stock_codes):
    """Fetch 2y EOD prices for all A-share stocks (single query)"""
    start_dt = (dt.date.today() - dt.timedelta(days=LOOKBACK_DAYS)).strftime("%Y%m%d")
    print(f"[Phase 1] Fetching EOD prices from {start_dt} ...", file=sys.stderr)
    t0 = time.time()

    # Process in chunks to avoid MySQL packet size issues
    chunk_size = 2000
    all_dfs = []

    for i in range(0, len(stock_codes), chunk_size):
        chunk = stock_codes[i:i + chunk_size]
        placeholders = ",".join(["%s"] * len(chunk))
        sql = (
            "SELECT S_INFO_WINDCODE, TRADE_DT, S_DQ_OPEN, S_DQ_HIGH, S_DQ_LOW, S_DQ_CLOSE, "
            "S_DQ_VOLUME, S_DQ_AMOUNT, S_DQ_PCTCHANGE, S_DQ_ADJCLOSE_BACKWARD "
            "FROM ashareeodprices "
            f"WHERE TRADE_DT >= %s AND S_INFO_WINDCODE IN ({placeholders})"
        )
        cur = conn.cursor()
        cur.execute(sql, [start_dt] + chunk)
        rows = cur.fetchall()
        cur.close()

        if rows:
            chunk_df = pd.DataFrame(rows, columns=[
                "code", "trade_dt", "open", "high", "low", "close",
                "volume", "amount", "pct_chg", "adj_close",
            ])
            all_dfs.append(chunk_df)

        print(f"  Chunk {i//chunk_size+1}: {len(rows):,} rows ({time.time()-t0:.0f}s)", file=sys.stderr)

    if not all_dfs:
        return pd.DataFrame()

    df = pd.concat(all_dfs, ignore_index=True)
    df["trade_dt"] = pd.to_datetime(df["trade_dt"], format="%Y%m%d")
    for c in ["open", "high", "low", "close", "volume", "amount", "pct_chg", "adj_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["close", "adj_close"]).copy()

    print(f"  EOD total: {len(df):,} rows, {df['code'].nunique()} stocks ({time.time()-t0:.0f}s)", file=sys.stderr)
    return df


def compute_volume_threshold(eod_df, pct=VOL_PCT_THRESHOLD):
    """Compute average daily volume per stock, find top 20% threshold"""
    print("[Phase 1] Computing volume percentiles ...", file=sys.stderr)
    # Use last 60 trading days for average volume
    latest_dt = eod_df["trade_dt"].max()
    recent = eod_df[eod_df["trade_dt"] >= (latest_dt - pd.Timedelta(days=90))]

    avg_vol = recent.groupby("code")["volume"].mean()
    cutoff = avg_vol.quantile(1 - pct)
    top_codes = set(avg_vol[avg_vol >= cutoff].index)
    print(f"  Volume top {int(pct*100)}%: {len(top_codes)} stocks (cutoff: {cutoff:,.0f} shares)", file=sys.stderr)
    return top_codes


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
    """Get latest market cap from ashareeodderivativeindicator.
    S_VAL_MV: total market value. Unit is likely 万元 — multiply by 10,000 to get 元."""
    print(f"[Phase 3] Fetching market cap for {len(windcodes)} stocks ...", file=sys.stderr)
    t0 = time.time()

    # Query in chunks to avoid MySQL packet limits
    chunk_size = 500
    mcap_map = {}

    for i in range(0, len(windcodes), chunk_size):
        chunk = windcodes[i:i + chunk_size]
        placeholders = ",".join(["%s"] * len(chunk))
        cur = conn.cursor()
        cur.execute(f"""
            SELECT S_INFO_WINDCODE, S_VAL_MV
            FROM ashareeodderivativeindicator
            WHERE TRADE_DT = (SELECT MAX(TRADE_DT) FROM ashareeodderivativeindicator)
              AND S_INFO_WINDCODE IN ({placeholders})
        """, chunk)
        rows = cur.fetchall()
        cur.close()
        for r in rows:
            if r[1]:
                mcap_map[r[0]] = r[1]

    print(f"  Market cap fetched: {len(mcap_map)} stocks ({time.time()-t0:.0f}s)", file=sys.stderr)

    # Auto-detect unit: if max mcap < 10T (1e13), likely in 万元 → convert to 元
    if mcap_map:
        max_mv = max(mcap_map.values())
        # 贵州茅台 total mcap ~2万亿 = 2e12 元
        # If in 万元: ~2e8 万元, if in 元: ~2e12 元
        if max_mv < 1e10:
            # Values are in 万元, convert to 元
            mcap_map = {k: v * 10_000 for k, v in mcap_map.items()}
            print(f"  Unit auto-detected: 万元 → converted to 元 (max={max_mv/1e4:.0f}亿)", file=sys.stderr)
        else:
            print(f"  Unit: 元 (max={max_mv/1e8:.0f}亿)", file=sys.stderr)

    return mcap_map


def main():
    print("[CN Screener] Starting Wind DB-based scan ...", file=sys.stderr)
    start_time = time.time()

    try:
        conn = get_db_connection()
    except Exception as e:
        print(f"[CN Screener] ERROR: Cannot connect to Wind DB: {e}", file=sys.stderr)
        return

    try:
        # ─── Phase 1: Stock list + EOD data ───
        desc_df = fetch_cn_stock_list(conn)
        name_map = dict(zip(desc_df["S_INFO_WINDCODE"], desc_df["S_INFO_NAME"]))
        stock_codes = desc_df["S_INFO_WINDCODE"].tolist()
        total_universe = len(stock_codes)

        eod_df = fetch_cn_eod_prices(conn, stock_codes)
        if eod_df.empty:
            print("[CN Screener] No EOD data returned", file=sys.stderr)
            return

        # Volume top 20% filter
        top_codes = compute_volume_threshold(eod_df)
        eod_top = eod_df[eod_df["code"].isin(top_codes)]

        # ─── Phase 2: Compute SMAs for top 20% volume stocks ───
        print(f"[Phase 2] Computing SMAs for {len(top_codes)} top-volume stocks ...", file=sys.stderr)
        technical_passers = []
        for code, group in eod_top.groupby("code"):
            result = analyze_from_wind(code, group)
            if result:
                result["name"] = name_map.get(code, code)
                technical_passers.append(result)

        elapsed = time.time() - start_time
        print(f"[Phase 2] Complete: {len(technical_passers)} pass technical filters ({elapsed:.0f}s)", file=sys.stderr)

        # ─── Phase 3: Market cap filter ───
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
            print(f"  [Pass] {item['ticker']} ({item['name']}) ¥{item['price']:.2f} MCap={mcap/1e8:.1f}亿", file=sys.stderr)

    finally:
        conn.close()

    passing.sort(key=lambda x: x["marketCap"], reverse=True)

    elapsed = time.time() - start_time
    print(f"\n[CN Screener] ═══════════════════════════════════════", file=sys.stderr)
    print(f"[CN Screener] Complete. {len(passing)} stocks pass all filters.", file=sys.stderr)
    print(f"[CN Screener] Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)", file=sys.stderr)

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

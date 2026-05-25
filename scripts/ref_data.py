#!/usr/bin/env python3
"""
Reference data cache — company name, sector, market cap from Yahoo Finance.

Cache files: data/ref_us.json, data/ref_hk.json
Format: { "TICKER": { "name": "", "sector": "", "marketCap": N, "mcapDate": "YYYY-MM-DD" } }

- Name & sector: fetched once, cached permanently (rarely change)
- Market cap: re-fetched only when mcapDate is > 7 days old

Usage (standalone):
  python ref_data.py [--market us|hk|all] [--force-mcap]
"""

import json
import os
import sys
import time
import datetime as dt

try:
    import yfinance as yf
except ImportError:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "yfinance", "-q"])
    import yfinance as yf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MCAP_STALE_DAYS = 7


def ref_path(market):
    return os.path.join(DATA_DIR, f"ref_{market}.json")


def load_ref(market):
    path = ref_path(market)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_ref(market, data):
    path = ref_path(market)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Ref] Saved {len(data)} entries to {path}", file=sys.stderr)


def yf_symbol(ticker):
    """Convert pool ticker to yfinance symbol."""
    if ticker.endswith(".HK"):
        return ticker  # 0700.HK
    if ticker.endswith(".US"):
        return ticker.replace(".US", "")  # AAPL
    return ticker


def fetch_one(ticker):
    """Fetch name, sector, marketCap for one ticker from yfinance."""
    symbol = yf_symbol(ticker)
    result = {"name": "", "sector": "", "marketCap": 0, "mcapDate": ""}
    try:
        tk = yf.Ticker(symbol)
        try:
            fi = tk.fast_info
            mcap = getattr(fi, "market_cap", 0) or 0
            result["marketCap"] = mcap
            if mcap > 0:
                result["mcapDate"] = dt.date.today().isoformat()
        except Exception:
            pass
        try:
            info = tk.info
            result["name"] = info.get("shortName", "") or info.get("longName", "")
            result["sector"] = info.get("sector", "") or ""
        except Exception:
            pass
    except Exception:
        pass
    return result


def batch_fetch(tickers, ref=None, force_mcap=False):
    """Fetch ref data for tickers, using cache where possible.

    Args:
        tickers: list of pool-format tickers (e.g. "AAPL.US", "0700.HK")
        ref: existing ref data dict (loaded from cache)
        force_mcap: if True, re-fetch market cap even if fresh

    Returns:
        updated ref data dict
    """
    if ref is None:
        ref = {}

    stale_date = (dt.date.today() - dt.timedelta(days=MCAP_STALE_DAYS)).isoformat()

    need_fetch = []
    for ticker in tickers:
        entry = ref.get(ticker)
        if not entry:
            need_fetch.append(ticker)
            continue
        if not entry.get("name") or not entry.get("sector"):
            need_fetch.append(ticker)
            continue
        if force_mcap or not entry.get("mcapDate") or entry["mcapDate"] < stale_date:
            need_fetch.append(ticker)

    if not need_fetch:
        print(f"[Ref] All {len(tickers)} tickers cached, nothing to fetch", file=sys.stderr)
        return ref

    print(f"[Ref] Fetching yfinance for {len(need_fetch)}/{len(tickers)} tickers (rest cached) ...", file=sys.stderr)
    t0 = time.time()

    for i, ticker in enumerate(need_fetch):
        existing = ref.get(ticker, {})
        fresh = fetch_one(ticker)

        entry = {
            "name": fresh.get("name") or existing.get("name", ""),
            "sector": fresh.get("sector") or existing.get("sector", ""),
            "marketCap": fresh.get("marketCap", 0) or existing.get("marketCap", 0),
            "mcapDate": fresh.get("mcapDate") or existing.get("mcapDate", ""),
        }
        if fresh.get("marketCap", 0) > 0:
            entry["mcapDate"] = fresh["mcapDate"]

        ref[ticker] = entry

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(need_fetch)} ({time.time()-t0:.0f}s)", file=sys.stderr)
        time.sleep(0.15)

    print(f"[Ref] Fetched {len(need_fetch)} tickers in {time.time()-t0:.0f}s", file=sys.stderr)
    return ref


# ── Standalone: build / refresh cache from pool + screener data ──

def _collect_tickers(market):
    tickers = set()

    pool_path = os.path.join(DATA_DIR, f"pools_{market}.json")
    if os.path.exists(pool_path):
        with open(pool_path, "r", encoding="utf-8") as f:
            pdata = json.load(f)
        if isinstance(pdata, dict):
            tickers.update(pdata.keys())
        elif isinstance(pdata, list):
            tickers.update(e.get("ticker", "") for e in pdata)

    screener_path = os.path.join(ROOT_DIR, "public", "data", f"{market}.json")
    if os.path.exists(screener_path):
        with open(screener_path, "r", encoding="utf-8") as f:
            sdata = json.load(f)
        tickers.update(s.get("ticker", "") for s in sdata.get("stocks", []))

    tickers.discard("")
    return list(tickers)


if __name__ == "__main__":
    market = "all"
    force_mcap = False
    if "--market" in sys.argv:
        market = sys.argv[sys.argv.index("--market") + 1]
    if "--force-mcap" in sys.argv:
        force_mcap = True

    for m in (["us", "hk"] if market == "all" else [market]):
        tickers = _collect_tickers(m)
        if tickers:
            ref = load_ref(m)
            ref = batch_fetch(tickers, ref, force_mcap=force_mcap)
            save_ref(m, ref)
        else:
            print(f"[Ref] No tickers found for {m}", file=sys.stderr)

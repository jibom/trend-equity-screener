#!/usr/bin/env python3
"""
Build per-stock chart data JSON for Lightweight Charts.
Output: public/data/chartdata/<market>/<code>.json + index.json

Usage:
  python build_chartdata.py [--market us|hk|all] [--max-tickers N]
"""

import json, os, sys, time, datetime as dt, argparse

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
OUTPUT_DIR = os.path.join(ROOT_DIR, "public", "data", "chartdata")
BATCH = 50


def _read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, "r") as f:
        return [l.strip().split("\t")[1] for l in f if "\t" in l.strip()]


def _read_json_keys(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    if isinstance(d, dict):
        return list(d.keys())
    return [e.get("ticker", "") for e in d if e.get("ticker")]


def _read_screener_tickers(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return [s.get("ticker", "") for s in d.get("stocks", []) if s.get("ticker")]


def collect(market):
    uni = _read_lines(os.path.join(SCRIPT_DIR, f"{market}_universe.txt"))
    pool = _read_json_keys(os.path.join(ROOT_DIR, "data", f"pools_{market}.json"))
    scr = _read_screener_tickers(os.path.join(ROOT_DIR, "public", "data", f"{market}.json"))
    return list(dict.fromkeys(t for t in uni + pool + scr if t))


def yf_tk(t):
    if t.endswith(".US"): return t[:-3]
    return t  # 0700.HK stays as-is for yfinance


def fcode(t):
    return t.replace(".HK", "").replace(".US", "")


def to_rows(df):
    if df is None or df.empty:
        return []
    out = []
    for idx, r in df.iterrows():
        ts = int(idx.timestamp()) if hasattr(idx, "timestamp") else int(idx)
        o, h, l = round(float(r.get("Open", 0) or 0), 2), round(float(r.get("High", 0) or 0), 2), round(float(r.get("Low", 0) or 0), 2)
        c, v = round(float(r.get("Close", 0) or 0), 2), int(r.get("Volume", 0) or 0)
        if c > 0:
            out.append([ts, o, h, l, c, v])
    return out


def extract(df, yf_sym):
    """Extract one ticker's data from yfinance batch download result."""
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        if yf_sym in df.columns.get_level_values(0):
            sub = df[yf_sym]
            return sub.dropna(subset=["Close"]) if isinstance(sub, pd.DataFrame) else pd.DataFrame()
        return pd.DataFrame()
    return df.dropna(subset=["Close"])


def build(market, max_tickers=None):
    print(f"[ChartData] {market.upper()} starting ...", file=sys.stderr)
    t0 = time.time()

    tickers = collect(market)
    if max_tickers:
        tickers = tickers[:max_tickers]
    print(f"  {len(tickers)} tickers", file=sys.stderr)

    mdir = os.path.join(OUTPUT_DIR, market)
    os.makedirs(mdir, exist_ok=True)

    ym = {yf_tk(t): t for t in tickers}
    yl = list(ym.keys())
    results = {}

    for i in range(0, len(yl), BATCH):
        batch = yl[i:i + BATCH]
        bn = i // BATCH + 1
        tb = (len(yl) + BATCH - 1) // BATCH
        print(f"  Batch {bn}/{tb} ({len(batch)} tickers) ...", file=sys.stderr)

        try:
            daily = yf.download(tickers=batch, period="2y", interval="1d",
                                group_by="ticker", threads=True, progress=False, auto_adjust=True)
            weekly = yf.download(tickers=batch, period="5y", interval="1wk",
                                 group_by="ticker", threads=True, progress=False, auto_adjust=True)
        except Exception as e:
            print(f"  [WARN] download failed: {e}", file=sys.stderr)
            continue

        for yf_s in batch:
            pt = ym[yf_s]
            cd = fcode(pt)
            dr = to_rows(extract(daily, yf_s))
            wr = to_rows(extract(weekly, yf_s))
            if not dr and not wr:
                continue
            obj = {"ticker": pt, "updated": dt.date.today().isoformat(), "daily": dr, "weekly": wr}
            with open(os.path.join(mdir, f"{cd}.json"), "w") as f:
                json.dump(obj, f, separators=(",", ":"))
            results[pt] = cd

    # index.json
    ip = os.path.join(OUTPUT_DIR, "index.json")
    idx = {}
    if os.path.exists(ip):
        with open(ip, "r") as f:
            idx = json.load(f)
    idx[market] = sorted(results.values())
    idx["updated"] = dt.datetime.now().astimezone().isoformat()
    with open(ip, "w") as f:
        json.dump(idx, f, separators=(",", ":"))

    print(f"[ChartData] {market.upper()}: {len(results)} stocks, {time.time()-t0:.0f}s", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", default="all", choices=["us", "hk", "all"])
    ap.add_argument("--max-tickers", type=int, default=None)
    a = ap.parse_args()
    for m in (["us", "hk"] if a.market == "all" else [a.market]):
        build(m, a.max_tickers)


if __name__ == "__main__":
    main()

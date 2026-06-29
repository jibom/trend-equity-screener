"""生成美国ETF chartdata → public/data/chartdata/us/<code>.json (与美股股票同目录, showChart 自动路由).
复用 export_chartdata_us.build_ohlc (日线260 + 周线3年) + us_trend.fetch_all (etf 缓存前缀).
用法: python scripts/export_chartdata_etf.py [--asof 2026-06-26]
"""
from __future__ import annotations
import os, sys, json, argparse, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import us_trend as ust
import export_chartdata_us as ecu
import pandas as pd

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
ETF_UNIVERSE = os.path.join(PROJECT_DIR, "configs", "etf_universe.csv")
OUT_DIR = os.path.join(PROJECT_DIR, "public", "data", "chartdata", "us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()
    asof = args.asof or datetime.date.today().isoformat()
    pool = pd.read_csv(ETF_UNIVERSE)
    pool["Industry"] = pool["Industry"].fillna("未分类")
    pool["Sector"] = pool["Sector"].fillna("未分类")
    print(f"=== export_chartdata_etf (asof={asof}) ===\n[Pool] {len(pool)}")
    raw = ust.fetch_all(pool[["Ticker", "Code", "Name", "Sector", "Industry"]], asof, prefix="etf_eod_raw")
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    for code, g in raw.groupby("code"):
        res = ecu.build_ohlc(g)
        if res is None:
            continue
        daily, weekly = res
        payload = {"ticker": code, "updated": asof, "daily": daily, "weekly": weekly}
        with open(os.path.join(OUT_DIR, code.replace(".US", "") + ".json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        n += 1
    print(f"[Done] {n} 只 → {OUT_DIR}")


if __name__ == "__main__":
    main()

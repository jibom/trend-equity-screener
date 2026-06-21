"""抓取美股基本面 (Sales Growth / Earning Growth / ROE) → configs/us_fundamentals.csv。

EODHD fundamentals 403, 改用 yfinance get_info:
  revenueGrowth   季度营收同比 (fraction)
  earningsGrowth  季度盈利同比 (fraction)
  returnOnEquity  TTM ROE (fraction)

供「美股-趋势」选个股参考。断点续跑 (configs/us_fundamentals_partial.csv)。
扫描时 export_trend_us.py 只读本 CSV, 不再调 yfinance。

用法: python scripts/build_us_fundamentals.py
"""
from __future__ import annotations
import os, sys, csv, time
import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
UNIVERSE_CSV = os.path.join(PROJECT_DIR, "configs", "us_universe.csv")
OUT_CSV = os.path.join(PROJECT_DIR, "configs", "us_universe_fundamentals.csv")
CACHE_CSV = os.path.join(PROJECT_DIR, "configs", "us_universe_fundamentals_partial.csv")

FIELDS = ["revenueGrowth", "earningsGrowth", "returnOnEquity"]


def _load_cache():
    if not os.path.exists(CACHE_CSV):
        return {}
    done = {}
    with open(CACHE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done[row["code"]] = row
    return done


def _save_cache(done):
    os.makedirs(os.path.dirname(CACHE_CSV), exist_ok=True)
    with open(CACHE_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["code", "revenueGrowth", "earningsGrowth", "returnOnEquity"])
        for c, r in done.items():
            w.writerow([c, r.get("revenueGrowth", ""), r.get("earningsGrowth", ""), r.get("returnOnEquity", "")])


def main():
    pool = pd.read_csv(UNIVERSE_CSV)
    codes = pool["Code"].tolist()
    done = _load_cache()
    todo = [c for c in codes if c not in done]
    print(f"=== build_us_fundamentals ===\n[Pool] {len(codes)}; cached {len(done)}; to fetch {len(todo)}")

    ok = 0
    for i, code in enumerate(todo):
        row = {"code": code, "revenueGrowth": "", "earningsGrowth": "", "returnOnEquity": ""}
        for attempt in range(2):
            try:
                info = yf.Ticker(code).get_info()
                if info:
                    for k in FIELDS:
                        v = info.get(k)
                        row[k] = round(float(v), 4) if isinstance(v, (int, float)) else ""
                break
            except Exception:
                time.sleep(0.8)
        if row["revenueGrowth"] or row["returnOnEquity"]:
            ok += 1
        done[code] = row
        time.sleep(0.25)  # 防 yfinance 限流
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(todo)} (有值 {ok})"); _save_cache(done)
    _save_cache(done)

    # 写最终 CSV (全池, 按 universe 顺序)
    out = []
    for c in codes:
        r = done.get(c, {})
        out.append({"Ticker": c + ".US", "Code": c,
                    "SalesGrowth": r.get("revenueGrowth", ""),
                    "EarnGrowth": r.get("earningsGrowth", ""),
                    "ROE": r.get("returnOnEquity", "")})
    df = pd.DataFrame(out)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    n_with = int(((df["SalesGrowth"] != "") | (df["ROE"] != "")).sum())
    print(f"[Done] {len(df)} 只 → {OUT_CSV}  (有基本面数据: ~{n_with})")


if __name__ == "__main__":
    main()

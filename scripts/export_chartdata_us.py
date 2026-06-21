"""生成 public/data/chartdata/us/<code>.json (1年日K + 周K, 前复权) 供美股图表模态使用。

格式与港股 chartdata/hk 一致: {ticker, updated, daily:[[ts,o,h,l,c,v],...], weekly:[[...]]}
ts = UTC 0:00 epoch(秒)。前复权 (最新日 fwd_close == 原始 close)。复用 us_trend 的 EOD 缓存 + forward_adjust_group。

用法: python scripts/export_chartdata_us.py [--asof 2026-06-18]
"""
from __future__ import annotations
import os, sys, json
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import us_trend as ust   # 模块级已设 utf-8 stdout
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "data", "chartdata", "us")
DAYS = 260


def _ts(dt_str):
    return int(datetime.strptime(str(dt_str), "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def build_ohlc(g):
    """g: 原始EOD组 → 前复权 → daily/weekly (fwd OHLC + vol)."""
    gg = ust.forward_adjust_group(g)
    if gg is None or len(gg) == 0:
        return None
    d = gg.tail(DAYS).copy()
    daily = []
    for _, r in d.iterrows():
        daily.append([_ts(r["date"]),
                      round(float(r["fwd_open"]), 4), round(float(r["fwd_high"]), 4),
                      round(float(r["fwd_low"]), 4), round(float(r["fwd_close"]), 4),
                      int(r["vol"]) if pd.notna(r["vol"]) else 0])
    d["date"] = pd.to_datetime(d["date"])
    w = d.set_index("date").resample("W-FRI").agg({
        "fwd_open": "first", "fwd_high": "max", "fwd_low": "min",
        "fwd_close": "last", "vol": "sum"}).dropna()
    weekly = []
    for wd, r in w.iterrows():
        weekly.append([_ts(wd.strftime("%Y-%m-%d")),
                       round(float(r["fwd_open"]), 4), round(float(r["fwd_high"]), 4),
                       round(float(r["fwd_low"]), 4), round(float(r["fwd_close"]), 4),
                       int(r["vol"])])
    return daily, weekly


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()
    asof = args.asof or str(pd.read_csv(ust.UNIVERSE_CSV)["LatestDate"].iloc[0])

    pool = ust.load_pool()
    print(f"=== export_chartdata_us (asof={asof}) ===\n[Pool] {len(pool)}")
    raw = ust.fetch_all(pool, asof)
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    for code, g in raw.groupby("code"):
        res = build_ohlc(g)
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

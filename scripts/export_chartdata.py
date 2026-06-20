"""生成 public/data/chartdata/hk/<code>.json (1年日K + 周K, 前复权) 供图表模态使用。

格式与现有 chartdata 一致: {ticker, updated, daily:[[ts,o,h,l,c,v],...], weekly:[[...]]}
ts = UTC 0:00 epoch(秒)。前复权 (最新日 = 原始收盘价)。

用法: python scripts/export_chartdata.py
"""
from __future__ import annotations
import os, sys, json
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import sector_cluster as sc  # 模块级已设 utf-8 stdout
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "data", "chartdata", "hk")
DAYS = 260  # ~1年交易日


def _ts(dt_str):
    return int(datetime.strptime(str(dt_str), "%Y%m%d").replace(tzinfo=timezone.utc).timestamp())


def build_ohlc(g):
    g = g.sort_values("TRADE_DT").reset_index(drop=True)
    lr = g["S_DQ_CLOSE"].iloc[-1]
    la = g["S_DQ_ADJCLOSE"].iloc[-1]
    if pd.isna(lr) or pd.isna(la) or lr == 0:
        return None
    f = la / lr
    d = g.tail(DAYS).copy()
    daily = []
    for _, r in d.iterrows():
        daily.append([_ts(r["TRADE_DT"]),
                      round(float(r["S_DQ_ADJOPEN"]) / f, 4),
                      round(float(r["S_DQ_ADJHIGH"]) / f, 4),
                      round(float(r["S_DQ_ADJLOW"]) / f, 4),
                      round(float(r["S_DQ_ADJCLOSE"]) / f, 4),
                      int(r["S_DQ_VOLUME"]) if pd.notna(r["S_DQ_VOLUME"]) else 0])
    d["date"] = pd.to_datetime(d["TRADE_DT"], format="%Y%m%d")
    w = d.set_index("date").resample("W-FRI").agg({
        "S_DQ_ADJOPEN": "first", "S_DQ_ADJHIGH": "max", "S_DQ_ADJLOW": "min",
        "S_DQ_ADJCLOSE": "last", "S_DQ_VOLUME": "sum", "TRADE_DT": "last"}).dropna()
    weekly = []
    for _, r in w.iterrows():
        weekly.append([_ts(r["TRADE_DT"]),
                       round(float(r["S_DQ_ADJOPEN"]) / f, 4), round(float(r["S_DQ_ADJHIGH"]) / f, 4),
                       round(float(r["S_DQ_ADJLOW"]) / f, 4), round(float(r["S_DQ_ADJCLOSE"]) / f, 4),
                       int(r["S_DQ_VOLUME"])])
    return daily, weekly


def main():
    pool = sc.load_pool()
    asof = sc.latest_asof()
    print(f"=== export_chartdata (asof={asof}) ===\n[Pool] {len(pool)}")
    raw = sc.fetch_all(pool["Ticker"].tolist(), asof)
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    for code, g in raw.groupby("code"):
        res = build_ohlc(g)
        if res is None:
            continue
        daily, weekly = res
        payload = {"ticker": code, "updated": asof, "daily": daily, "weekly": weekly}
        with open(os.path.join(OUT_DIR, code.replace(".HK", "") + ".json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        n += 1
    print(f"[Done] {n} 只 → {OUT_DIR}")


if __name__ == "__main__":
    main()

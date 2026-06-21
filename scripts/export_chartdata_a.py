"""生成 public/data/chartdata/a/<code>.json (1年日K + 周K, 前复权) 供A股图表模态使用。

镜像港股 export_chartdata.py, 取数改 jianxin ashareeodprices (Wind schema). 复用 a_trend 的 EOD 缓存.
格式: {ticker, updated, daily:[[ts,o,h,l,c,v],...], weekly:[[...]]}, ts=UTC0:00 epoch, 前复权.
用法: python scripts/export_chartdata_a.py [--asof 2026-06-18]
"""
from __future__ import annotations
import os, sys, json
from datetime import datetime, timezone
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import a_trend as at   # 透传 utf-8 stdout
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "data", "chartdata", "a")
DAYS = 260


def _ts(dt_str):
    return int(datetime.strptime(str(dt_str), "%Y%m%d").replace(tzinfo=timezone.utc).timestamp())


def build_ohlc(g):
    g = g.sort_values("TRADE_DT").reset_index(drop=True)
    lr = g["S_DQ_CLOSE"].iloc[-1]; la = g["S_DQ_ADJCLOSE"].iloc[-1]
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
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()
    asof = args.asof or at.latest_asof()

    pool = at.load_pool()
    print(f"=== export_chartdata_a (asof={asof}) ===\n[Pool] {len(pool)}")
    raw = at.fetch_all(pool["Ticker"].tolist(), asof)
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    for code, g in raw.groupby("code"):
        res = build_ohlc(g)
        if res is None:
            continue
        daily, weekly = res
        payload = {"ticker": code, "updated": asof, "daily": daily, "weekly": weekly}
        # code 形如 '600519.SH' → 文件名取数字部分
        fname = code.split(".")[0] + ".json"
        with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        n += 1
    print(f"[Done] {n} 只 → {OUT_DIR}")


if __name__ == "__main__":
    main()

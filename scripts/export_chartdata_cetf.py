"""生成中国ETF chartdata → public/data/chartdata/a/<code>.json (与A股股票同目录, showChart .SH/.SZ 自动路由).
复用 export_chartdata_a.build_ohlc (Wind schema, 日线260 + 周线3年) + cetf_trend.fetch_all (cetf 缓存).
用法: python scripts/export_chartdata_cetf.py [--asof 2026-06-26]
"""
from __future__ import annotations
import os, sys, json, argparse, datetime, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cetf_trend as ct
import export_chartdata_a as eca

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
OUT_DIR = os.path.join(PROJECT_DIR, "public", "data", "chartdata", "a")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()
    asof = args.asof or datetime.date.today().isoformat()
    pool = ct.load_pool()
    print(f"=== export_chartdata_cetf (asof={asof}) ===\n[Pool] {len(pool)}")
    raw = ct.fetch_all(pool["Ticker"].tolist(), asof)
    os.makedirs(OUT_DIR, exist_ok=True)
    n = 0
    for code, g in raw.groupby("code"):
        res = eca.build_ohlc(g)
        if res is None:
            continue
        daily, weekly = res
        payload = {"ticker": code, "updated": asof, "daily": daily, "weekly": weekly}
        name = re.sub(r"\.(SH|SZ|BJ)$", "", code)
        with open(os.path.join(OUT_DIR, name + ".json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh, separators=(",", ":"))
        n += 1
    print(f"[Done] {n} 只 → {OUT_DIR}")


if __name__ == "__main__":
    main()

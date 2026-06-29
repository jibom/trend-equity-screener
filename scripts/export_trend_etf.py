"""导出美国ETF趋势 (首次新高/持续新高/趋势向上) 到 public/data/trend_hotspot_etf.json.
复用 src/us_trend.py (fetch_all/compute_features/hotspot_scores), universe=configs/etf_universe.csv.
标准同美股: 6月(126日)新高 + ma_stack 多头 + hotness 排序; 首次/持续按 nh_ratio_126 4% 拆分.
用法: python scripts/export_trend_etf.py [--asof 2026-06-26]
"""
from __future__ import annotations
import os, sys, json, argparse, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import us_trend as ust
import pandas as pd

PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
ETF_UNIVERSE = os.path.join(PROJECT_DIR, "configs", "etf_universe.csv")
OUT_PATH = os.path.join(PROJECT_DIR, "public", "data", "trend_hotspot_etf.json")


def load_pool():
    df = pd.read_csv(ETF_UNIVERSE)
    df["Industry"] = df["Industry"].fillna("未分类")
    df["Sector"] = df["Sector"].fillna("未分类")
    return df[["Ticker", "Code", "Name", "Sector", "Industry"]]


def clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int,)):
        return v
    if isinstance(v, float):
        return round(v, 3)
    return v


def rec(d):
    return {k: clean(v) for k, v in d.items()}


def record(r):
    return rec({
        "ticker": r["Ticker"], "name": r["Name"], "sub_industry": r["Industry"],
        "close": r.get("close"), "pct_high_250": r.get("pct_high_250"),
        "pct_high_126": r.get("pct_high_126"), "nh_ratio_126": r.get("nh_ratio_126"),
        "amt_surge": r.get("amt_surge"), "amt_rank": r.get("amt_rank_pct"),
        "ret_60": r.get("ret_60"),
        "ma_aligned": int(r["ma_aligned"]) if r.get("ma_aligned") is not None else 0,
        "ma_stack": int(r["ma_stack"]) if r.get("ma_stack") is not None else 0,
        "hotness": r.get("hotness"),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", default=None)
    args = ap.parse_args()
    import datetime as _dt
    asof = args.asof or _dt.date.today().isoformat()

    pool = load_pool()
    print(f"=== export_trend_etf (asof={asof}) ===\n[Pool] {len(pool)} 只 ETF")
    raw = ust.fetch_all(pool, asof, prefix="etf_eod_raw")
    print(f"[Fetch] {raw['code'].nunique()} 只有数据")
    asof = str(raw["date"].max())

    df_full, _, _ = ust.hotspot_scores(pool, raw)
    print(f"[Hotspot] {len(df_full)} 只有效")

    # Part1c1 首次新高: 6月新高 + 近6月新高天数<4%, 低位优先
    nh = df_full[df_full["pct_high_126"] >= 0.98]
    first_nh = nh[nh["nh_ratio_126"] < 0.04].sort_values("hotness", ascending=False).head(20)
    # Part1c2 持续新高: 占比≥4% + 多头排列(ma_stack), hotness 排序
    sust_nh = nh[(nh["nh_ratio_126"] >= 0.04) & (nh["ma_stack"] == 1)].sort_values("hotness", ascending=False).head(20)
    # Part2 趋势向上: 60日>10%, hotness 排序
    top = df_full[df_full["ret_60"] > 0.10].sort_values("hotness", ascending=False).head(20)

    part1c1 = [record(r) for _, r in first_nh.iterrows()]
    part1c2 = [record(r) for _, r in sust_nh.iterrows()]
    part2 = [record(r) for _, r in top.iterrows()]
    all_stocks = [rec({"ticker": r["Ticker"], "name": r["Name"], "sub_industry": r["Industry"],
                       "close": r.get("close"), "pct_high_250": r.get("pct_high_250"),
                       "ret_60": r.get("ret_60"),
                       "ma_aligned": int(r["ma_aligned"]) if r.get("ma_aligned") is not None else 0,
                       "hotness": r.get("hotness")})
                  for _, r in df_full.sort_values("hotness", ascending=False).iterrows()]

    hottest = first_nh.iloc[0]["Industry"] if len(first_nh) else (sust_nh.iloc[0]["Industry"] if len(sust_nh) else "—")
    payload = {"date": asof, "market": "ETF", "hottest_sub": hottest,
               "part1c1": part1c1, "part1c2": part1c2, "part2": part2, "all_stocks": all_stocks}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n[Export] -> {OUT_PATH}")
    print(f"  part1c1: {len(part1c1)} 首次新高 | part1c2: {len(part1c2)} 持续新高 | part2: {len(part2)} 趋势向上")


if __name__ == "__main__":
    main()
